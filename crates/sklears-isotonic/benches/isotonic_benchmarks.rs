//! Performance benchmarks for sklears-isotonic
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict};
use sklears_isotonic::{isotonic_regression, IsotonicRegression};
use std::hint::black_box;

/// Generate a deterministic, roughly-increasing-but-noisy 1D series (seeded).
fn generate_noisy_series(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");

    let x: Array1<f64> = (0..n).map(|i| i as f64).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| i as f64 * 0.01 + normal.sample(&mut rng))
        .collect();

    (x, y)
}

/// Benchmark IsotonicRegression::fit (Pool Adjacent Violators Algorithm).
fn benchmark_isotonic_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("isotonic_fit");

    for size in [100, 1000, 5000, 10000].iter() {
        let (x, y) = generate_noisy_series(*size, 321);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let model = IsotonicRegression::new();
                black_box(model.fit(&x, &y))
            });
        });
    }

    group.finish();
}

/// Benchmark IsotonicRegression::predict (threshold lookup + linear
/// interpolation) on a model that is fitted once outside the timed loop.
fn benchmark_isotonic_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("isotonic_predict");

    let (x_train, y_train) = generate_noisy_series(5000, 111);
    let fitted = IsotonicRegression::new()
        .fit(&x_train, &y_train)
        .expect("operation should succeed");

    for size in [100, 1000, 5000, 10000].iter() {
        let (x_query, _) = generate_noisy_series(*size, 654);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(preds) = fitted.predict(&x_query) {
                    black_box(preds)
                } else {
                    black_box(Array1::<f64>::zeros(x_query.len()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark the standalone `isotonic_regression` free function (direct PAVA
/// call, bypassing the builder/estimator API) for comparison against `fit`.
fn benchmark_isotonic_regression_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("isotonic_regression_function");

    for size in [100, 1000, 5000, 10000].iter() {
        let (_, y) = generate_noisy_series(*size, 987);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(isotonic_regression(&y, true)));
        });
    }

    group.finish();
}

criterion_group!(fitting, benchmark_isotonic_fit,);
criterion_group!(
    prediction,
    benchmark_isotonic_predict,
    benchmark_isotonic_regression_function,
);

criterion_main!(fitting, prediction,);
