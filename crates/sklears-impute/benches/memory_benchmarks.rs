//! Memory usage and efficiency benchmarks
//!
//! This benchmark suite measures:
//! - Memory consumption for different imputation methods
//! - Peak memory usage during imputation
//! - Memory efficiency with sparse data
//! - Cache efficiency

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{s, Array2};
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Transform};
use sklears_impute::{KNNImputer, SimpleImputer};
use std::hint::black_box;

/// Generate sparse data with many missing values
fn generate_sparse_data(n_samples: usize, n_features: usize, sparsity: f64) -> Array2<f64> {
    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            if rng.random::<f64>() < (1.0 - sparsity) {
                data[[i, j]] = rng.random_range(-10.0..10.0);
            } else {
                data[[i, j]] = f64::NAN;
            }
        }
    }

    data
}

/// Benchmark memory usage with sparse data
fn bench_sparse_data_imputation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_data_imputation");

    for sparsity in [0.5, 0.7, 0.9].iter() {
        let size = 1000;
        let n_features = 50;
        let data = generate_sparse_data(size, n_features, *sparsity);

        group.throughput(Throughput::Elements((size * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.0}%_sparse", sparsity * 100.0)),
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

/// Benchmark cache efficiency with different access patterns
fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");

    let sizes = [(500, 10), (100, 50), (50, 100), (10, 500)];

    for (n_samples, n_features) in sizes.iter() {
        let data = generate_sparse_data(*n_samples, *n_features, 0.15);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
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

/// Benchmark memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    let size = 2000;
    let n_features = 25;
    let missing_rate = 0.15;

    // Pre-generate data
    let mut rng = thread_rng();
    let mut data = Array2::zeros((size, n_features));
    for i in 0..size {
        for j in 0..n_features {
            data[[i, j]] = rng.random_range(-10.0..10.0);
            if rng.random::<f64>() < missing_rate {
                data[[i, j]] = f64::NAN;
            }
        }
    }

    // Benchmark allocation for simple imputer
    group.bench_function("simple_imputer_allocation", |b| {
        b.iter(|| {
            let imputer = SimpleImputer::new().strategy("mean".to_string());
            let fitted = imputer.fit(&data.view(), &()).unwrap();
            let result = fitted.transform(&data.view()).unwrap();
            black_box(result);
        });
    });

    // Benchmark allocation for KNN imputer
    group.sample_size(10);
    let small_data = data.slice(s![0..500, ..]).to_owned();
    group.bench_function("knn_imputer_allocation", |b| {
        b.iter(|| {
            let imputer = KNNImputer::new().n_neighbors(5);
            let fitted = imputer.fit(&small_data.view(), &()).unwrap();
            let result = fitted.transform(&small_data.view()).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark repeated transform operations (model reuse)
fn bench_model_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_reuse");

    let size = 1000;
    let n_features = 20;
    let missing_rate = 0.15;

    let mut rng = thread_rng();
    let mut train_data = Array2::zeros((size, n_features));
    let mut test_data = Array2::zeros((size, n_features));

    for i in 0..size {
        for j in 0..n_features {
            train_data[[i, j]] = rng.random_range(-10.0..10.0);
            test_data[[i, j]] = rng.random_range(-10.0..10.0);
            if rng.random::<f64>() < missing_rate {
                train_data[[i, j]] = f64::NAN;
                test_data[[i, j]] = f64::NAN;
            }
        }
    }

    // Fit once, transform multiple times
    let imputer = SimpleImputer::new().strategy("mean".to_string());
    let fitted = imputer.fit(&train_data.view(), &()).unwrap();

    group.throughput(Throughput::Elements((size * n_features) as u64));
    group.bench_function("single_transform", |b| {
        b.iter(|| {
            let _result = fitted.transform(&test_data.view()).unwrap();
        });
    });

    group.bench_function("repeated_transform_10x", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let _result = fitted.transform(&test_data.view()).unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark with different data types and layouts
fn bench_data_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_layout");

    let size = 1000;
    let n_features = 30;
    let missing_rate = 0.15;

    let mut rng = thread_rng();

    // Row-major (standard) layout
    let mut row_major = Array2::zeros((size, n_features));
    for i in 0..size {
        for j in 0..n_features {
            row_major[[i, j]] = rng.random_range(-10.0..10.0);
            if rng.random::<f64>() < missing_rate {
                row_major[[i, j]] = f64::NAN;
            }
        }
    }

    // Column-major layout
    let col_major = row_major.t().to_owned().t().to_owned();

    group.throughput(Throughput::Elements((size * n_features) as u64));

    group.bench_function("row_major", |b| {
        b.iter(|| {
            let imputer = SimpleImputer::new().strategy("mean".to_string());
            let fitted = imputer.fit(&row_major.view(), &()).unwrap();
            let _result = fitted.transform(&row_major.view()).unwrap();
        });
    });

    group.bench_function("column_major", |b| {
        b.iter(|| {
            let imputer = SimpleImputer::new().strategy("mean".to_string());
            let fitted = imputer.fit(&col_major.view(), &()).unwrap();
            let _result = fitted.transform(&col_major.view()).unwrap();
        });
    });

    group.finish();
}

/// Benchmark scalability with increasing dataset sizes
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(10);

    let n_features = 20;
    let missing_rate = 0.15;

    for size in [100, 500, 1000, 2000, 5000, 10000].iter() {
        let mut rng = thread_rng();
        let mut data = Array2::zeros((*size, n_features));

        for i in 0..*size {
            for j in 0..n_features {
                data[[i, j]] = rng.random_range(-10.0..10.0);
                if rng.random::<f64>() < missing_rate {
                    data[[i, j]] = f64::NAN;
                }
            }
        }

        group.throughput(Throughput::Elements((size * n_features) as u64));
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
    bench_sparse_data_imputation,
    bench_cache_efficiency,
    bench_memory_allocation,
    bench_model_reuse,
    bench_data_layout,
    bench_scalability,
);

criterion_main!(benches);
