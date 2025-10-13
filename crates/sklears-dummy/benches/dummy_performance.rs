use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use sklears_core::traits::{Fit, Predict, Untrained};
use sklears_dummy::{
    ClassifierStrategy, DummyClassifier, DummyRegressor, OnlineClassificationStrategy,
    OnlineDummyClassifier, OnlineDummyRegressor, OnlineStrategy, RegressorStrategy,
};

/// Generate synthetic regression data for benchmarks
fn generate_regression_data(
    n_samples: usize,
    n_features: usize,
    random_state: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(random_state);

    // Generate random features (uniform distribution scaled to [-1, 1])
    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>() * 2.0 - 1.0);

    // Generate targets as sum of features with some noise
    let y = Array1::from_shape_fn(n_samples, |i| {
        x.row(i).sum() + (rng.gen::<f64>() * 2.0 - 1.0) * 0.1
    });

    (x, y)
}

/// Generate synthetic classification data for benchmarks
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    random_state: u64,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(random_state);

    // Generate random features (uniform distribution scaled to [-1, 1])
    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>() * 2.0 - 1.0);

    // Generate random class labels
    let y = Array1::from_shape_fn(n_samples, |_| {
        (rng.gen::<f64>() * n_classes as f64).floor() as i32
    });

    (x, y)
}

/// Benchmark dummy regressor fitting performance
fn bench_dummy_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("dummy_regressor_fit");

    for size in [100, 1000, 10000, 100000].iter() {
        let (x, y) = generate_regression_data(*size, 10, 42);

        group.throughput(Throughput::Elements(*size as u64));

        // Benchmark different strategies
        group.bench_with_input(BenchmarkId::new("mean", size), size, |b, &_| {
            b.iter(|| {
                let regressor = DummyRegressor::new(RegressorStrategy::Mean);
                black_box(regressor.fit(&x, &y).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("median", size), size, |b, &_| {
            b.iter(|| {
                let regressor = DummyRegressor::new(RegressorStrategy::Median);
                black_box(regressor.fit(&x, &y).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("quantile", size), size, |b, &_| {
            b.iter(|| {
                let regressor = DummyRegressor::new(RegressorStrategy::Quantile(0.75));
                black_box(regressor.fit(&x, &y).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("normal", size), size, |b, &_| {
            b.iter(|| {
                let regressor = DummyRegressor::new(RegressorStrategy::Normal {
                    mean: Some(0.0),
                    std: Some(1.0),
                });
                black_box(regressor.fit(&x, &y).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark dummy regressor prediction performance
fn bench_dummy_regressor_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("dummy_regressor_predict");

    for size in [100, 1000, 10000, 100000].iter() {
        let (x_train, y_train) = generate_regression_data(1000, 10, 42);
        let (x_test, _) = generate_regression_data(*size, 10, 123);

        group.throughput(Throughput::Elements(*size as u64));

        // Pre-fit models
        let mean_fitted = DummyRegressor::new(RegressorStrategy::Mean)
            .fit(&x_train, &y_train)
            .unwrap();
        let median_fitted = DummyRegressor::new(RegressorStrategy::Median)
            .fit(&x_train, &y_train)
            .unwrap();
        let normal_fitted = DummyRegressor::new(RegressorStrategy::Normal {
            mean: Some(0.0),
            std: Some(1.0),
        })
        .fit(&x_train, &y_train)
        .unwrap();

        group.bench_with_input(BenchmarkId::new("mean", size), size, |b, &_| {
            b.iter(|| {
                black_box(mean_fitted.predict(&x_test).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("median", size), size, |b, &_| {
            b.iter(|| {
                black_box(median_fitted.predict(&x_test).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("normal", size), size, |b, &_| {
            b.iter(|| {
                black_box(normal_fitted.predict(&x_test).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark dummy classifier fitting performance
fn bench_dummy_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("dummy_classifier_fit");

    for size in [100, 1000, 10000, 100000].iter() {
        let (x, y) = generate_classification_data(*size, 10, 5, 42);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("most_frequent", size), size, |b, &_| {
            b.iter(|| {
                let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);
                black_box(classifier.fit(&x, &y).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("stratified", size), size, |b, &_| {
            b.iter(|| {
                let classifier = DummyClassifier::new(ClassifierStrategy::Stratified);
                black_box(classifier.fit(&x, &y).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("uniform", size), size, |b, &_| {
            b.iter(|| {
                let classifier = DummyClassifier::new(ClassifierStrategy::Uniform);
                black_box(classifier.fit(&x, &y).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark dummy classifier prediction performance
fn bench_dummy_classifier_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("dummy_classifier_predict");

    for size in [100, 1000, 10000, 100000].iter() {
        let (x_train, y_train) = generate_classification_data(1000, 10, 5, 42);
        let (x_test, _) = generate_classification_data(*size, 10, 5, 123);

        group.throughput(Throughput::Elements(*size as u64));

        // Pre-fit models
        let most_frequent_fitted = DummyClassifier::new(ClassifierStrategy::MostFrequent)
            .fit(&x_train, &y_train)
            .unwrap();
        let stratified_fitted = DummyClassifier::new(ClassifierStrategy::Stratified)
            .fit(&x_train, &y_train)
            .unwrap();
        let uniform_fitted = DummyClassifier::new(ClassifierStrategy::Uniform)
            .fit(&x_train, &y_train)
            .unwrap();

        group.bench_with_input(BenchmarkId::new("most_frequent", size), size, |b, &_| {
            b.iter(|| {
                black_box(most_frequent_fitted.predict(&x_test).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("stratified", size), size, |b, &_| {
            b.iter(|| {
                black_box(stratified_fitted.predict(&x_test).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("uniform", size), size, |b, &_| {
            b.iter(|| {
                black_box(uniform_fitted.predict(&x_test).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark online learning performance
fn bench_online_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_learning");

    for size in [100, 1000, 10000, 100000].iter() {
        let (_, y) = generate_regression_data(*size, 1, 42);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("online_mean", size), size, |b, &_| {
            b.iter(|| {
                let mut regressor: OnlineDummyRegressor<Untrained> =
                    OnlineDummyRegressor::new(OnlineStrategy::OnlineMean {
                        drift_detection: None,
                    });
                for &target in y.iter() {
                    regressor.partial_fit(target).unwrap();
                    black_box(());
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("ewma", size), size, |b, &_| {
            b.iter(|| {
                let mut regressor: OnlineDummyRegressor<Untrained> =
                    OnlineDummyRegressor::new(OnlineStrategy::EWMA { alpha: 0.1 });
                for &target in y.iter() {
                    regressor.partial_fit(target).unwrap();
                    black_box(());
                }
            });
        });

        group.bench_with_input(
            BenchmarkId::new("forgetting_factor", size),
            size,
            |b, &_| {
                b.iter(|| {
                    let mut regressor: OnlineDummyRegressor<Untrained> =
                        OnlineDummyRegressor::new(OnlineStrategy::ForgettingFactor { lambda: 0.9 });
                    for &target in y.iter() {
                        regressor.partial_fit(target).unwrap();
                        black_box(());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark online classification performance
fn bench_online_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_classification");

    for size in [100, 1000, 10000, 100000].iter() {
        let (_, y) = generate_classification_data(*size, 1, 5, 42);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("most_frequent", size), size, |b, &_| {
            b.iter(|| {
                let mut classifier: OnlineDummyClassifier<Untrained> =
                    OnlineDummyClassifier::new(OnlineClassificationStrategy::OnlineMostFrequent);
                for &target in y.iter() {
                    classifier.partial_fit(target);
                    black_box(());
                }
            });
        });

        group.bench_with_input(
            BenchmarkId::new("exponentially_weighted", size),
            size,
            |b, &_| {
                b.iter(|| {
                    let mut classifier: OnlineDummyClassifier<Untrained> =
                        OnlineDummyClassifier::new(
                            OnlineClassificationStrategy::ExponentiallyWeighted { alpha: 0.1 },
                        );
                    for &target in y.iter() {
                        classifier.partial_fit(target);
                        black_box(());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test memory efficiency of different window strategies
    for window_size in [100, 1000, 10000].iter() {
        let (_, y) = generate_regression_data(*window_size * 2, 1, 42);

        group.bench_with_input(
            BenchmarkId::new("fixed_window", window_size),
            window_size,
            |b, &size| {
                b.iter(|| {
                    let mut regressor: OnlineDummyRegressor<Untrained> =
                        OnlineDummyRegressor::new(OnlineStrategy::OnlineMean {
                            drift_detection: None,
                        })
                        .with_window_strategy(sklears_dummy::WindowStrategy::FixedWindow(size));

                    for &target in y.iter() {
                        regressor.partial_fit(target).unwrap();
                        black_box(());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("adaptive_window", window_size),
            window_size,
            |b, &size| {
                b.iter(|| {
                    let mut regressor: OnlineDummyRegressor<Untrained> =
                        OnlineDummyRegressor::new(OnlineStrategy::AdaptiveWindow {
                            max_window_size: size,
                            drift_threshold: 1.0,
                        });

                    for &target in y.iter() {
                        regressor.partial_fit(target).unwrap();
                        black_box(());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scaling with feature dimensions
fn bench_feature_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_scaling");

    for n_features in [1, 10, 100, 1000].iter() {
        let (x, y) = generate_regression_data(1000, *n_features, 42);

        group.throughput(Throughput::Elements(*n_features as u64));

        group.bench_with_input(
            BenchmarkId::new("dummy_regressor", n_features),
            n_features,
            |b, &_| {
                b.iter(|| {
                    let regressor = DummyRegressor::new(RegressorStrategy::Mean);
                    let fitted = regressor.fit(&x, &y).unwrap();
                    black_box(fitted.predict(&x).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Compare batch vs online learning performance
fn bench_batch_vs_online(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_online");

    for size in [1000, 10000, 100000].iter() {
        let (x, y) = generate_regression_data(*size, 5, 42);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("batch_mean", size), size, |b, &_| {
            b.iter(|| {
                let regressor = DummyRegressor::new(RegressorStrategy::Mean);
                black_box(regressor.fit(&x, &y).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("online_mean", size), size, |b, &_| {
            b.iter(|| {
                let mut regressor: OnlineDummyRegressor<Untrained> =
                    OnlineDummyRegressor::new(OnlineStrategy::OnlineMean {
                        drift_detection: None,
                    });
                for &target in y.iter() {
                    regressor.partial_fit(target).unwrap();
                    black_box(());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark drift detection overhead
fn bench_drift_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("drift_detection");

    for size in [1000, 10000].iter() {
        let (_, y) = generate_regression_data(*size, 1, 42);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("no_drift_detection", size),
            size,
            |b, &_| {
                b.iter(|| {
                    let mut regressor: OnlineDummyRegressor<Untrained> =
                        OnlineDummyRegressor::new(OnlineStrategy::OnlineMean {
                            drift_detection: None,
                        });
                    for &target in y.iter() {
                        regressor.partial_fit(target).unwrap();
                        black_box(());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("adwin_drift_detection", size),
            size,
            |b, &_| {
                b.iter(|| {
                    let mut regressor: OnlineDummyRegressor<Untrained> =
                        OnlineDummyRegressor::new(OnlineStrategy::OnlineMean {
                            drift_detection: Some(sklears_dummy::DriftDetectionMethod::ADWIN),
                        });
                    for &target in y.iter() {
                        regressor.partial_fit(target).unwrap();
                        black_box(());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("page_hinkley_drift_detection", size),
            size,
            |b, &_| {
                b.iter(|| {
                    let mut regressor: OnlineDummyRegressor<Untrained> =
                        OnlineDummyRegressor::new(OnlineStrategy::OnlineMean {
                            drift_detection: Some(sklears_dummy::DriftDetectionMethod::PageHinkley),
                        });
                    for &target in y.iter() {
                        regressor.partial_fit(target).unwrap();
                        black_box(());
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dummy_regressor_fit,
    bench_dummy_regressor_predict,
    bench_dummy_classifier_fit,
    bench_dummy_classifier_predict,
    bench_online_learning,
    bench_online_classification,
    bench_memory_usage,
    bench_feature_scaling,
    bench_batch_vs_online,
    bench_drift_detection
);

criterion_main!(benches);
