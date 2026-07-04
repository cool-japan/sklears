//! Performance benchmarks for MLP classifier and regressor fit/predict
//!
//! These benchmarks establish performance baselines for the training (`fit`)
//! and inference (`predict`, i.e. a forward pass) costs of `MLPClassifier`
//! and `MLPRegressor`, tracking regressions across releases using the
//! Criterion framework.
//!
//! Networks, epoch counts, and dataset sizes are intentionally kept tiny:
//! unlike classical ML estimators, a neural network's training cost scales
//! with epochs x batches x layers, so even a "small" benchmark input can
//! dominate wall-clock time if left unconstrained.

#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::traits::{Fit, Predict};
use sklears_neural::{Activation, MLPClassifier, MLPRegressor};
use std::hint::black_box;

/// Number of input features used for the synthetic benchmark datasets
const N_FEATURES: usize = 8;
/// Number of classes used for the synthetic classification benchmark dataset
const N_CLASSES: usize = 3;
/// Hidden layer shape for the tiny network used throughout these benchmarks
const HIDDEN_LAYERS: [usize; 2] = [16, 8];

/// Generate deterministic standard-normal feature data for benchmarks
fn generate_features(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");

    let data: Vec<f64> = (0..n_samples * n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((n_samples, n_features), data)
        .expect("shape and data length should match")
}

/// Generate deterministic class labels cycling through `n_classes`
fn generate_class_labels(n_samples: usize, n_classes: usize) -> Vec<usize> {
    (0..n_samples).map(|i| i % n_classes).collect()
}

/// Generate deterministic single-output regression targets derived from the features
fn generate_regression_targets(x: &Array2<f64>, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let normal = Normal::new(0.0, 1.0).expect("operation should succeed");
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mut targets = vec![0.0f64; n_samples];
    for (i, target) in targets.iter_mut().enumerate() {
        let mut value = 0.0;
        for j in 0..n_features {
            value += x[[i, j]];
        }
        value += normal.sample(&mut rng);
        *target = value;
    }

    Array2::from_shape_vec((n_samples, 1), targets).expect("shape and data length should match")
}

/// Benchmark `MLPClassifier::fit` over a handful of epochs on small datasets
fn benchmark_mlp_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_classifier_fit");
    group.sample_size(10); // Even a few epochs of training is comparatively expensive

    for size in [32, 128, 512].iter() {
        let x = generate_features(*size, N_FEATURES, 42);
        let y = generate_class_labels(*size, N_CLASSES);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mlp = MLPClassifier::new()
                    .hidden_layer_sizes(&HIDDEN_LAYERS)
                    .activation(Activation::Relu)
                    .max_iter(5)
                    .learning_rate_init(0.01)
                    .random_state(42);

                if let Ok(fitted) = mlp.fit(&x, &y) {
                    black_box(fitted.loss())
                } else {
                    black_box(f64::NAN)
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `MLPClassifier::predict` (forward pass only) on a pre-trained tiny network
fn benchmark_mlp_classifier_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_classifier_predict");

    for size in [32, 128, 512].iter() {
        let x = generate_features(*size, N_FEATURES, 7);
        let y = generate_class_labels(*size, N_CLASSES);

        let mlp = MLPClassifier::new()
            .hidden_layer_sizes(&HIDDEN_LAYERS)
            .activation(Activation::Relu)
            .max_iter(20)
            .learning_rate_init(0.01)
            .random_state(42);

        let trained_mlp = mlp
            .fit(&x, &y)
            .expect("small MLP classifier should fit on generated benchmark data");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(predictions) = trained_mlp.predict(&x) {
                    black_box(predictions)
                } else {
                    black_box(Vec::new())
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `MLPRegressor::fit` over a handful of epochs on small datasets
fn benchmark_mlp_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_regressor_fit");
    group.sample_size(10);

    for size in [32, 128, 512].iter() {
        let x = generate_features(*size, N_FEATURES, 99);
        let y = generate_regression_targets(&x, 100);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mlp = MLPRegressor::new()
                    .hidden_layer_sizes(&HIDDEN_LAYERS)
                    .activation(Activation::Relu)
                    .max_iter(5)
                    .learning_rate_init(0.01)
                    .random_state(42);

                if let Ok(fitted) = mlp.fit(&x, &y) {
                    black_box(fitted.loss())
                } else {
                    black_box(f64::NAN)
                }
            });
        });
    }

    group.finish();
}

/// Benchmark `MLPRegressor::predict` (forward pass only) on a pre-trained tiny network
fn benchmark_mlp_regressor_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_regressor_predict");

    for size in [32, 128, 512].iter() {
        let x = generate_features(*size, N_FEATURES, 13);
        let y = generate_regression_targets(&x, 21);

        let mlp = MLPRegressor::new()
            .hidden_layer_sizes(&HIDDEN_LAYERS)
            .activation(Activation::Relu)
            .max_iter(20)
            .learning_rate_init(0.01)
            .random_state(42);

        let trained_mlp = mlp
            .fit(&x, &y)
            .expect("small MLP regressor should fit on generated benchmark data");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                if let Ok(predictions) = trained_mlp.predict(&x) {
                    black_box(predictions)
                } else {
                    black_box(Array2::zeros((0, 0)))
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    classifier,
    benchmark_mlp_classifier_fit,
    benchmark_mlp_classifier_predict,
);

criterion_group!(
    regressor,
    benchmark_mlp_regressor_fit,
    benchmark_mlp_regressor_predict,
);

criterion_main!(classifier, regressor);
