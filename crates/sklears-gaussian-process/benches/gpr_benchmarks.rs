//! Performance benchmarks for sklears-gaussian-process
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict};
use sklears_gaussian_process::{GaussianProcessRegressor, RBF};
use std::hint::black_box;

/// Generate deterministic regression data: features ~ N(0,1), target = row sum + noise.
fn generate_regression_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");

    let x_data: Vec<f64> = (0..n_samples * n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let x = Array2::from_shape_vec((n_samples, n_features), x_data)
        .expect("shape and data length should match");

    let y: Array1<f64> = x
        .axis_iter(Axis(0))
        .map(|row| row.sum() + 0.1 * normal.sample(&mut rng))
        .collect();

    (x, y)
}

/// Benchmark GaussianProcessRegressor::fit (RBF kernel, Cholesky decomposition).
///
/// GP regression is cubic in `n_samples` (Cholesky factorization of the n x n
/// kernel matrix), so sizes stay far smaller than the thousands used for
/// linear/near-linear preprocessing benchmarks elsewhere in the workspace.
fn benchmark_gpr_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpr_fit");
    group.sample_size(10);

    for size in [50, 100, 200, 400].iter() {
        let (x, y) = generate_regression_data(*size, 3, 42);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let gpr = GaussianProcessRegressor::new().kernel(Box::new(RBF::new(1.0)));
                black_box(gpr.fit(&x, &y))
            });
        });
    }

    group.finish();
}

/// Benchmark GaussianProcessRegressor::predict (posterior mean) on a model
/// that is fitted once outside the timed loop.
fn benchmark_gpr_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpr_predict");
    group.sample_size(10);

    let (x_train, y_train) = generate_regression_data(200, 3, 7);
    let gpr = GaussianProcessRegressor::new().kernel(Box::new(RBF::new(1.0)));
    let fitted = gpr
        .fit(&x_train, &y_train)
        .expect("fit should succeed with valid training data");

    for size in [50, 100, 200, 400].iter() {
        let (x_test, _) = generate_regression_data(*size, 3, 99);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(mean) = fitted.predict(&x_test) {
                    black_box(mean)
                } else {
                    black_box(Array1::<f64>::zeros(x_test.nrows()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark predict_with_std (posterior mean + std-dev) -- more expensive than
/// plain `predict` due to a per-test-point triangular solve against the
/// Cholesky factor.
fn benchmark_gpr_predict_with_std(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpr_predict_with_std");
    group.sample_size(10);

    let (x_train, y_train) = generate_regression_data(200, 3, 7);
    let gpr = GaussianProcessRegressor::new().kernel(Box::new(RBF::new(1.0)));
    let fitted = gpr
        .fit(&x_train, &y_train)
        .expect("fit should succeed with valid training data");

    for size in [50, 100, 200, 400].iter() {
        let (x_test, _) = generate_regression_data(*size, 3, 123);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok((mean, std)) = fitted.predict_with_std(&x_test) {
                    black_box((mean, std))
                } else {
                    black_box((
                        Array1::<f64>::zeros(x_test.nrows()),
                        Array1::<f64>::zeros(x_test.nrows()),
                    ))
                }
            });
        });
    }

    group.finish();
}

criterion_group!(fitting, benchmark_gpr_fit,);
criterion_group!(
    prediction,
    benchmark_gpr_predict,
    benchmark_gpr_predict_with_std,
);

criterion_main!(fitting, prediction,);
