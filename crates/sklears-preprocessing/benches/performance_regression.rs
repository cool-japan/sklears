//! Performance regression benchmarks for sklears-preprocessing
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, CoreRandom};
use sklears_core::{Fit, Transform};
use sklears_preprocessing::*;

/// Generate test data for benchmarks
fn generate_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let data: Vec<f64> = (0..nrows * ncols)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

/// Benchmark StandardScaler fit operation
fn benchmark_standard_scaler_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_scaler_fit");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 42);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let scaler = StandardScaler::new(true, true);
                black_box(scaler.fit(&x, &y))
            });
        });
    }

    group.finish();
}

/// Benchmark StandardScaler transform operation
fn benchmark_standard_scaler_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_scaler_transform");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 42);
        let y = Array1::zeros(x.nrows());
        let scaler = StandardScaler::new(true, true);
        let fitted = scaler.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(fitted.transform(&x)));
        });
    }

    group.finish();
}

/// Benchmark StandardScaler fit + transform (most common use case)
fn benchmark_standard_scaler_fit_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_scaler_fit_transform");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 42);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let scaler = StandardScaler::new(true, true);
                let fitted = scaler.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark MinMaxScaler operations
fn benchmark_minmax_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_scaler");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 123);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let scaler = MinMaxScaler::new((0.0, 1.0));
                let fitted = scaler.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark RobustScaler operations
fn benchmark_robust_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("robust_scaler");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 456);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let scaler = RobustScaler::new(true, (25.0, 75.0));
                let fitted = scaler.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark Normalizer operations
fn benchmark_normalizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalizer");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 789);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let normalizer = Normalizer::new(NormType::L2);
                let fitted = normalizer.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark PolynomialFeatures (computational intensive)
fn benchmark_polynomial_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_features");
    group.sample_size(10); // Reduce sample size for slow operations

    for degree in [2, 3].iter() {
        for size in [100, 1000, 5000].iter() {
            let x = generate_data(*size, 5, 321);
            let y = Array1::zeros(x.nrows());

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_deg{}", size, 5, degree)),
                &(size, degree),
                |b, _| {
                    b.iter(|| {
                        let poly = PolynomialFeatures::new(*degree, false, false);
                        let fitted = poly.fit(&x, &y).unwrap();
                        black_box(fitted.transform(&x))
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SimpleImputer
fn benchmark_simple_imputer(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_imputer");

    for size in [100, 1000, 10000, 100000].iter() {
        let mut x = generate_data(*size, 10, 654);
        let y = Array1::zeros(x.nrows());

        // Add 10% missing values
        for i in (0..*size).step_by(10) {
            if i < x.nrows() {
                x[[i, 0]] = f64::NAN;
            }
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let imputer = SimpleImputer::new(ImputationStrategy::Mean);
                let fitted = imputer.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark LabelEncoder
fn benchmark_label_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_encoder");

    for size in [100, 1000, 10000, 100000].iter() {
        let y: Vec<String> = (0..*size).map(|i| format!("label_{}", i % 100)).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut encoder = LabelEncoder::new();
                black_box(encoder.fit(&y))
            });
        });
    }

    group.finish();
}

/// Benchmark OneHotEncoder
fn benchmark_one_hot_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("one_hot_encoder");
    group.sample_size(20);

    for size in [100, 1000, 10000].iter() {
        let data: Vec<Vec<String>> = (0..*size)
            .map(|i| vec![format!("cat_{}", i % 10), format!("type_{}", i % 5)])
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut encoder = OneHotEncoder::new(false, false, None);
                let fitted = encoder.fit(&data).unwrap();
                black_box(fitted.transform(&data))
            });
        });
    }

    group.finish();
}

/// Benchmark QuantileTransformer
fn benchmark_quantile_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile_transformer");
    group.sample_size(10);

    for size in [100, 1000, 10000].iter() {
        let x = generate_data(*size, 10, 987);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let transformer =
                    QuantileTransformer::new(QuantileOutput::Uniform, 1000, None, None);
                let fitted = transformer.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark PowerTransformer
fn benchmark_power_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_transformer");
    group.sample_size(10);

    for size in [100, 1000, 10000].iter() {
        let x = generate_data(*size, 10, 111);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let transformer = PowerTransformer::new(PowerMethod::YeoJohnson, true);
                let fitted = transformer.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark varying feature dimensions
fn benchmark_feature_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_dimensions");

    for ncols in [5, 10, 50, 100, 500].iter() {
        let x = generate_data(10000, *ncols, 222);
        let y = Array1::zeros(x.nrows());

        group.throughput(Throughput::Elements((10000 * ncols) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(ncols), ncols, |b, _| {
            b.iter(|| {
                let scaler = StandardScaler::new(true, true);
                let fitted = scaler.fit(&x, &y).unwrap();
                black_box(fitted.transform(&x))
            });
        });
    }

    group.finish();
}

/// Benchmark inverse transform operations
fn benchmark_inverse_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("inverse_transform");

    for size in [100, 1000, 10000, 100000].iter() {
        let x = generate_data(*size, 10, 333);
        let y = Array1::zeros(x.nrows());
        let scaler = StandardScaler::new(true, true);
        let fitted = scaler.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(fitted.inverse_transform(&transformed)));
        });
    }

    group.finish();
}

criterion_group!(
    scalers,
    benchmark_standard_scaler_fit,
    benchmark_standard_scaler_transform,
    benchmark_standard_scaler_fit_transform,
    benchmark_minmax_scaler,
    benchmark_robust_scaler,
    benchmark_normalizer,
);

criterion_group!(feature_engineering, benchmark_polynomial_features,);

criterion_group!(imputation, benchmark_simple_imputer,);

criterion_group!(encoding, benchmark_label_encoder, benchmark_one_hot_encoder,);

criterion_group!(
    transformers,
    benchmark_quantile_transformer,
    benchmark_power_transformer,
);

criterion_group!(
    dimensions,
    benchmark_feature_dimensions,
    benchmark_inverse_transform,
);

criterion_main!(
    scalers,
    feature_engineering,
    imputation,
    encoding,
    transformers,
    dimensions,
);
