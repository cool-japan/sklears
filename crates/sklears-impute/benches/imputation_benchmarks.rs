//! Comprehensive benchmarks for imputation methods
//!
//! This benchmark suite measures performance across different:
//! - Imputation methods (Simple, KNN, Iterative, etc.)
//! - Data sizes (small, medium, large)
//! - Missing data patterns (MCAR, MAR, MNAR)
//! - Missing data percentages (5%, 15%, 30%, 50%)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Transform};
use sklears_impute::{KNNImputer, SimpleImputer};

/// Generate synthetic data with missing values
fn generate_data_with_missing(
    n_samples: usize,
    n_features: usize,
    missing_rate: f64,
) -> Array2<f64> {
    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    // Generate random data
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.random_range(-10.0..10.0);
        }
    }

    // Introduce missing values (MCAR pattern)
    for i in 0..n_samples {
        for j in 0..n_features {
            if rng.random::<f64>() < missing_rate {
                data[[i, j]] = f64::NAN;
            }
        }
    }

    data
}

/// Benchmark SimpleImputer with mean strategy
fn bench_simple_imputer_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_imputer_mean");

    for size in [100, 500, 1000, 2000].iter() {
        for missing_rate in [0.05, 0.15, 0.30].iter() {
            let n_features = 20;
            let data = generate_data_with_missing(*size, n_features, *missing_rate);

            group.throughput(Throughput::Elements((*size * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}x{}_miss{:.0}%",
                    size,
                    n_features,
                    missing_rate * 100.0
                )),
                &data,
                |b, data| {
                    b.iter(|| {
                        let imputer = SimpleImputer::new().strategy("mean".to_string());
                        let fitted = imputer.fit(&data.view(), &()).unwrap();
                        let _result = fitted.transform(&data.view()).unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SimpleImputer with median strategy
fn bench_simple_imputer_median(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_imputer_median");

    for size in [100, 500, 1000].iter() {
        let n_features = 20;
        let missing_rate = 0.15;
        let data = generate_data_with_missing(*size, n_features, missing_rate);

        group.throughput(Throughput::Elements((*size * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = SimpleImputer::new().strategy("median".to_string());
                    let fitted = imputer.fit(&data.view(), &()).unwrap();
                    let _result = fitted.transform(&data.view()).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark KNNImputer with different k values
fn bench_knn_imputer(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_imputer");
    group.sample_size(20); // Fewer samples for KNN as it's slower

    for size in [100, 300, 500].iter() {
        for k in [3, 5, 10].iter() {
            let n_features = 10;
            let missing_rate = 0.15;
            let data = generate_data_with_missing(*size, n_features, missing_rate);

            group.throughput(Throughput::Elements((*size * n_features) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_k{}", size, n_features, k)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let imputer = KNNImputer::new().n_neighbors(*k);
                        let fitted = imputer.fit(&data.view(), &()).unwrap();
                        let _result = fitted.transform(&data.view()).unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark KNNImputer with distance vs uniform weights
fn bench_knn_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_weights");
    group.sample_size(20);

    let size = 300;
    let n_features = 10;
    let missing_rate = 0.15;
    let data = generate_data_with_missing(size, n_features, missing_rate);

    for weights in ["uniform", "distance"].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(weights), &data, |b, data| {
            b.iter(|| {
                let imputer = KNNImputer::new()
                    .n_neighbors(5)
                    .weights(weights.to_string());
                let fitted = imputer.fit(&data.view(), &()).unwrap();
                let _result = fitted.transform(&data.view()).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark imputation with varying missing rates
fn bench_varying_missing_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("varying_missing_rates");

    let size = 500;
    let n_features = 20;

    for missing_rate in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50].iter() {
        let data = generate_data_with_missing(size, n_features, *missing_rate);

        group.throughput(Throughput::Elements((size * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.0}%", missing_rate * 100.0)),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = SimpleImputer::new().strategy("mean".to_string());
                    let fitted = imputer.fit(&data.view(), &()).unwrap();
                    let _result = fitted.transform(&data.view()).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark imputation with varying feature counts
fn bench_varying_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("varying_features");

    let size = 500;
    let missing_rate = 0.15;

    for n_features in [5, 10, 20, 50, 100].iter() {
        let data = generate_data_with_missing(size, *n_features, missing_rate);

        group.throughput(Throughput::Elements((size * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}features", n_features)),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = SimpleImputer::new().strategy("mean".to_string());
                    let fitted = imputer.fit(&data.view(), &()).unwrap();
                    let _result = fitted.transform(&data.view()).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark fit vs transform performance
fn bench_fit_vs_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_vs_transform");

    let size = 1000;
    let n_features = 20;
    let missing_rate = 0.15;
    let data = generate_data_with_missing(size, n_features, missing_rate);

    // Benchmark fit only
    group.bench_function("fit_only", |b| {
        b.iter(|| {
            let imputer = SimpleImputer::new().strategy("mean".to_string());
            let _fitted = imputer.fit(&data.view(), &()).unwrap();
        });
    });

    // Benchmark transform only (with pre-fitted model)
    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer.fit(&data.view(), &()).unwrap();

    group.bench_function("transform_only", |b| {
        b.iter(|| {
            let _result = fitted.transform(&data.view()).unwrap();
        });
    });

    // Benchmark fit + transform
    group.bench_function("fit_and_transform", |b| {
        b.iter(|| {
            let imputer = SimpleImputer::new().strategy("mean".to_string());
            let fitted = imputer.fit(&data.view(), &()).unwrap();
            let _result = fitted.transform(&data.view()).unwrap();
        });
    });

    group.finish();
}

/// Benchmark memory efficiency with large datasets
fn bench_large_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_dataset");
    group.sample_size(10); // Very few samples for large datasets

    for size in [5000, 10000].iter() {
        let n_features = 30;
        let missing_rate = 0.15;
        let data = generate_data_with_missing(*size, n_features, missing_rate);

        group.throughput(Throughput::Elements((*size * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = SimpleImputer::new().strategy("mean".to_string());
                    let fitted = imputer.fit(&data.view(), &()).unwrap();
                    let _result = fitted.transform(&data.view()).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_imputer_mean,
    bench_simple_imputer_median,
    bench_knn_imputer,
    bench_knn_weights,
    bench_varying_missing_rates,
    bench_varying_features,
    bench_fit_vs_transform,
    bench_large_dataset,
);

criterion_main!(benches);
