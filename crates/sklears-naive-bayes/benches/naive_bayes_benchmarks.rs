//! Performance benchmarks for sklears-naive-bayes
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_naive_bayes::GaussianNB;
use std::hint::black_box;

/// Number of classes used in every classification benchmark below.
const N_CLASSES: usize = 2;

/// Generate deterministic, class-separated Gaussian features (seeded).
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");

    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<i32>::zeros(n_samples);
    for i in 0..n_samples {
        let class = (i % N_CLASSES) as i32;
        y[i] = class;
        for j in 0..n_features {
            let offset = if j == 0 { class as f64 * 6.0 } else { 0.0 };
            x[[i, j]] = normal.sample(&mut rng) + offset;
        }
    }
    (x, y)
}

/// Benchmark GaussianNB::fit (single-pass per-class mean/variance estimation).
fn benchmark_gaussian_nb_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_nb_fit");

    for size in [100, 1000, 5000, 10000].iter() {
        let (x, y) = generate_classification_data(*size, 10, 321);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let nb = GaussianNB::new();
                black_box(nb.fit(&x, &y))
            });
        });
    }

    group.finish();
}

/// Benchmark GaussianNB::predict on a model that is fitted once outside the timed loop.
fn benchmark_gaussian_nb_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_nb_predict");

    let (x_train, y_train) = generate_classification_data(2000, 10, 111);
    let fitted = GaussianNB::new()
        .fit(&x_train, &y_train)
        .expect("operation should succeed");

    for size in [100, 1000, 5000, 10000].iter() {
        let (x_test, _) = generate_classification_data(*size, 10, 654);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(preds) = fitted.predict(&x_test) {
                    black_box(preds)
                } else {
                    black_box(Array1::<i32>::zeros(x_test.nrows()))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark GaussianNB::predict_proba on a model that is fitted once outside the timed loop.
fn benchmark_gaussian_nb_predict_proba(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_nb_predict_proba");

    let (x_train, y_train) = generate_classification_data(2000, 10, 222);
    let fitted = GaussianNB::new()
        .fit(&x_train, &y_train)
        .expect("operation should succeed");

    for size in [100, 1000, 5000, 10000].iter() {
        let (x_test, _) = generate_classification_data(*size, 10, 987);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(proba) = fitted.predict_proba(&x_test) {
                    black_box(proba)
                } else {
                    black_box(Array2::<f64>::zeros((x_test.nrows(), N_CLASSES)))
                }
            });
        });
    }

    group.finish();
}

criterion_group!(fitting, benchmark_gaussian_nb_fit,);
criterion_group!(
    prediction,
    benchmark_gaussian_nb_predict,
    benchmark_gaussian_nb_predict_proba,
);

criterion_main!(fitting, prediction,);
