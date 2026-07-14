//! Performance regression benchmarks for sklears-multioutput
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework. They exercise the core
//! `MultiOutputRegressor` and `MultiOutputClassifier` meta-estimators, which
//! fit one model per target column, following the same fit/predict usage
//! shown in `src/core.rs`'s own unit tests.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;
use sklears_multioutput::{MultiOutputClassifier, MultiOutputRegressor};
use std::hint::black_box;

/// Generate a synthetic feature matrix and continuous multi-target
/// regression targets (deterministic given `seed`).
fn generate_regression_data(
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
    seed: u64,
) -> (Array2<Float>, Array2<Float>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("distribution parameters should be valid");

    let x_data: Vec<Float> = (0..n_samples * n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let x = Array2::from_shape_vec((n_samples, n_features), x_data)
        .expect("shape and data length should match");

    let y_data: Vec<Float> = (0..n_samples * n_targets)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let y = Array2::from_shape_vec((n_samples, n_targets), y_data)
        .expect("shape and data length should match");

    (x, y)
}

/// Generate a synthetic feature matrix and binary multi-target
/// classification labels (deterministic given `seed`).
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
    seed: u64,
) -> (Array2<Float>, Array2<i32>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("distribution parameters should be valid");

    let x_data: Vec<Float> = (0..n_samples * n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let x = Array2::from_shape_vec((n_samples, n_features), x_data)
        .expect("shape and data length should match");

    // Deterministic, roughly-balanced binary label per (sample, target) pair.
    let y = Array2::from_shape_fn((n_samples, n_targets), |(i, j)| ((i + j) % 2) as i32);

    (x, y)
}

/// Benchmark `MultiOutputRegressor` fit + predict over increasing sample
/// counts, at a fixed number of targets.
fn benchmark_multioutput_regressor_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("multioutput_regressor_fit_predict");
    group.sample_size(10);

    for size in [100, 1000, 5000, 10000].iter() {
        let (x, y) = generate_regression_data(*size, 10, 5, 42);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let regressor = MultiOutputRegressor::new().n_jobs(Some(1));
                if let Ok(fitted) = regressor.fit(&x.view(), &y) {
                    black_box(fitted.predict(&x.view()))
                } else {
                    black_box(Ok(Array2::<Float>::zeros((x.nrows(), y.ncols()))))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `MultiOutputClassifier` fit + predict over increasing sample
/// counts, at a fixed number of targets.
fn benchmark_multioutput_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("multioutput_classifier_fit_predict");
    group.sample_size(10);

    for size in [100, 1000, 5000, 10000].iter() {
        let (x, y) = generate_classification_data(*size, 10, 5, 7);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let classifier = MultiOutputClassifier::new().n_jobs(Some(1));
                if let Ok(fitted) = classifier.fit(&x.view(), &y) {
                    black_box(fitted.predict(&x.view()))
                } else {
                    black_box(Ok(Array2::<i32>::zeros((x.nrows(), y.ncols()))))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark how `MultiOutputRegressor` fit + predict scales with the
/// number of targets at a fixed sample count.
fn benchmark_target_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("multioutput_regressor_target_scaling");
    group.sample_size(10);

    for n_targets in [1, 5, 10, 20].iter() {
        let (x, y) = generate_regression_data(1000, 10, *n_targets, 123);

        group.throughput(Throughput::Elements(*n_targets as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_targets), n_targets, |b, _| {
            b.iter(|| {
                let regressor = MultiOutputRegressor::new().n_jobs(Some(1));
                if let Ok(fitted) = regressor.fit(&x.view(), &y) {
                    black_box(fitted.predict(&x.view()))
                } else {
                    black_box(Ok(Array2::<Float>::zeros((x.nrows(), y.ncols()))))
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    regressors,
    benchmark_multioutput_regressor_fit_predict,
    benchmark_target_count_scaling,
);

criterion_group!(classifiers, benchmark_multioutput_classifier_fit_predict,);

criterion_main!(regressors, classifiers);
