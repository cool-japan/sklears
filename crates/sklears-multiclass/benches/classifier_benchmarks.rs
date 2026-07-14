//! Performance regression benchmarks for sklears-multiclass
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework. They exercise the
//! One-vs-Rest, One-vs-One, and AdaBoost multiclass meta-estimators using
//! `LinearRegression` (from `sklears-linear`) as the wrapped base estimator,
//! mirroring the crate's own integration tests
//! (`tests/integration_basic.rs`).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::{
    traits::{Fit, Predict},
    types::{Array1, Float},
};
use sklears_linear::LinearRegression;
use sklears_multiclass::{AdaBoostClassifier, OneVsOneClassifier, OneVsRestClassifier};
use std::hint::black_box;

/// Generate a synthetic multiclass dataset made of `n_classes` separated
/// Gaussian blobs (deterministic given `seed`).
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<Float>, Array1<i32>) {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 0.5).expect("distribution parameters should be valid");

    let noise: Vec<Float> = (0..n_samples * n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();

    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let class = (i % n_classes) as Float;
        class * 5.0 + noise[i * n_features + j]
    });
    let y = Array1::from_shape_fn(n_samples, |i| (i % n_classes) as i32);

    (x, y)
}

/// Benchmark One-vs-Rest fit + predict over increasing sample counts.
fn benchmark_one_vs_rest_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("one_vs_rest_fit_predict");
    group.sample_size(10);

    for size in [100, 1000, 5000].iter() {
        let (x, y) = generate_classification_data(*size, 10, 3, 42);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let ovr = OneVsRestClassifier::new(LinearRegression::new());
                if let Ok(fitted) = ovr.fit(&x, &y) {
                    black_box(fitted.predict(&x))
                } else {
                    black_box(Ok(Array1::<i32>::zeros(x.nrows())))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark One-vs-One fit + predict over increasing sample counts.
///
/// One-vs-One trains `n_classes * (n_classes - 1) / 2` pairwise classifiers,
/// so it is expected to be costlier than One-vs-Rest for the same data.
fn benchmark_one_vs_one_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("one_vs_one_fit_predict");
    group.sample_size(10);

    for size in [100, 1000, 5000].iter() {
        let (x, y) = generate_classification_data(*size, 10, 3, 7);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let ovo = OneVsOneClassifier::new(LinearRegression::new());
                if let Ok(fitted) = ovo.fit(&x, &y) {
                    black_box(fitted.predict(&x))
                } else {
                    black_box(Ok(Array1::<i32>::zeros(x.nrows())))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark AdaBoost (SAMME-style boosting) fit + predict on a binary
/// problem over increasing sample counts.
fn benchmark_adaboost_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaboost_fit_predict");
    group.sample_size(10);

    for size in [100, 1000, 5000].iter() {
        let (x, y) = generate_classification_data(*size, 10, 2, 99);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let adaboost = AdaBoostClassifier::new(LinearRegression::new())
                    .n_estimators(5)
                    .random_state(Some(42));
                if let Ok(fitted) = adaboost.fit(&x, &y) {
                    black_box(fitted.predict(&x))
                } else {
                    black_box(Ok(Array1::<i32>::zeros(x.nrows())))
                }
            });
        });
    }

    group.finish();
}

/// Benchmark how One-vs-Rest fit + predict scales with the number of classes
/// at a fixed sample count.
fn benchmark_class_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("one_vs_rest_class_scaling");
    group.sample_size(10);

    for n_classes in [2, 5, 10, 20].iter() {
        let (x, y) = generate_classification_data(1000, 10, *n_classes, 123);

        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(BenchmarkId::from_parameter(n_classes), n_classes, |b, _| {
            b.iter(|| {
                let ovr = OneVsRestClassifier::new(LinearRegression::new());
                if let Ok(fitted) = ovr.fit(&x, &y) {
                    black_box(fitted.predict(&x))
                } else {
                    black_box(Ok(Array1::<i32>::zeros(x.nrows())))
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    meta_estimators,
    benchmark_one_vs_rest_fit_predict,
    benchmark_one_vs_one_fit_predict,
    benchmark_adaboost_fit_predict,
);

criterion_group!(scaling, benchmark_class_count_scaling,);

criterion_main!(meta_estimators, scaling);
