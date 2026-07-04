//! Performance regression benchmarks for sklears-ensemble
//!
//! These benchmarks establish performance baselines and track regression
//! across releases using the Criterion framework.
//!
//! Only the ensemble estimators with complete, functional implementations
//! are benchmarked here: `BaggingClassifier` and `AdaBoostClassifier`.
//! `GradientBoostingRegressor`/`GradientBoostingClassifier` and
//! `BaggingRegressor` are intentionally excluded: their `fit`/`predict`
//! logic is currently stubbed out (or entirely absent), so benchmarking
//! them would not produce meaningful results.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict};
use sklears_ensemble::{AdaBoostClassifier, BaggingClassifier};
use std::hint::black_box;

/// Generate test data for benchmarks
fn generate_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");

    let data: Vec<f64> = (0..nrows * ncols)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((nrows, ncols), data).expect("shape and data length should match")
}

/// Derive deterministic binary integer class labels from `x` (sign of each
/// row's sum), for use with `BaggingClassifier` which expects `Array1<Int>`.
fn labels_int(x: &Array2<f64>) -> Array1<i32> {
    x.rows()
        .into_iter()
        .map(|row| if row.sum() > 0.0 { 1 } else { 0 })
        .collect()
}

/// Derive deterministic binary float-encoded class labels from `x` (sign of
/// each row's sum), for use with `AdaBoostClassifier` which expects
/// `Array1<Float>`.
fn labels_float(x: &Array2<f64>) -> Array1<f64> {
    x.rows()
        .into_iter()
        .map(|row| if row.sum() > 0.0 { 1.0 } else { 0.0 })
        .collect()
}

/// Benchmark BaggingClassifier fit+predict across increasing sample counts.
fn benchmark_bagging_n_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("bagging_classifier_n_samples");
    group.sample_size(10); // Tree ensembles are relatively expensive per iteration

    for size in [100, 500, 1000, 2000].iter() {
        let x = generate_data(*size, 10, 4242);
        let y = labels_int(&x);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let estimator = BaggingClassifier::new()
                    .n_estimators(10)
                    .max_depth(Some(5))
                    .random_state(0);
                if let Ok(fitted) = estimator.fit(&x, &y) {
                    black_box(fitted.predict(&x))
                } else {
                    Ok(y.clone())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark BaggingClassifier fit+predict scaling with ensemble size
/// (fixed dataset, varying `n_estimators`).
fn benchmark_bagging_n_estimators(c: &mut Criterion) {
    let mut group = c.benchmark_group("bagging_classifier_n_estimators");
    group.sample_size(10);

    let n_samples = 500;
    let x = generate_data(n_samples, 10, 555);
    let y = labels_int(&x);

    for n_estimators in [5, 10, 25, 50].iter() {
        let n_estimators = *n_estimators;
        group.throughput(Throughput::Elements((n_samples * n_estimators) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_estimators),
            &n_estimators,
            |b, _| {
                b.iter(|| {
                    let estimator = BaggingClassifier::new()
                        .n_estimators(n_estimators)
                        .max_depth(Some(5))
                        .random_state(0);
                    if let Ok(fitted) = estimator.fit(&x, &y) {
                        black_box(fitted.predict(&x))
                    } else {
                        Ok(y.clone())
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark AdaBoostClassifier fit+predict across increasing sample counts.
fn benchmark_adaboost_n_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaboost_classifier_n_samples");
    group.sample_size(10);

    for size in [100, 500, 1000, 2000].iter() {
        let x = generate_data(*size, 10, 1729);
        let y = labels_float(&x);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let estimator = AdaBoostClassifier::new()
                    .n_estimators(10)
                    .learning_rate(1.0)
                    .random_state(0);
                if let Ok(fitted) = estimator.fit(&x, &y) {
                    black_box(fitted.predict(&x))
                } else {
                    Ok(y.clone())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark AdaBoostClassifier fit+predict scaling with ensemble size
/// (fixed dataset, varying `n_estimators`).
fn benchmark_adaboost_n_estimators(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaboost_classifier_n_estimators");
    group.sample_size(10);

    let n_samples = 500;
    let x = generate_data(n_samples, 10, 909);
    let y = labels_float(&x);

    for n_estimators in [5, 10, 25, 50].iter() {
        let n_estimators = *n_estimators;
        group.throughput(Throughput::Elements((n_samples * n_estimators) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_estimators),
            &n_estimators,
            |b, _| {
                b.iter(|| {
                    let estimator = AdaBoostClassifier::new()
                        .n_estimators(n_estimators)
                        .learning_rate(1.0)
                        .random_state(0);
                    if let Ok(fitted) = estimator.fit(&x, &y) {
                        black_box(fitted.predict(&x))
                    } else {
                        Ok(y.clone())
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    bagging_benches,
    benchmark_bagging_n_samples,
    benchmark_bagging_n_estimators,
);

criterion_group!(
    adaboost_benches,
    benchmark_adaboost_n_samples,
    benchmark_adaboost_n_estimators,
);

criterion_main!(bagging_benches, adaboost_benches);
