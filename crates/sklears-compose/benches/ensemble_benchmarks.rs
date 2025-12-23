//! Ensemble Methods Performance Benchmarks
//!
//! This benchmark suite measures the performance of ensemble learning methods
//! including voting classifiers/regressors, stacking, and dynamic selection.
//!
//! Run with: `cargo bench --bench ensemble_benchmarks`

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_compose::ensemble::voting::{VotingClassifier, VotingRegressor, VotingType};
use sklears_core::error::Result as SklResult;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;
use std::time::Duration;

/// Mock classifier for benchmarking
#[derive(Debug, Clone)]
struct MockClassifier {
    bias: Float,
}

impl MockClassifier {
    fn new(bias: Float) -> Self {
        Self { bias }
    }
}

impl Fit<ArrayView2<'static, Float>, ArrayView1<'static, i32>> for MockClassifier {
    type Fitted = FittedMockClassifier;

    fn fit(
        self,
        _x: &ArrayView2<'static, Float>,
        _y: &ArrayView1<'static, i32>,
    ) -> SklResult<Self::Fitted> {
        Ok(FittedMockClassifier { bias: self.bias })
    }
}

#[derive(Debug, Clone)]
struct FittedMockClassifier {
    bias: Float,
}

impl Predict for FittedMockClassifier {
    type Input = ArrayView2<'static, Float>;
    type Output = Array1<i32>;

    fn predict(&self, x: &Self::Input) -> SklResult<Self::Output> {
        Ok(x.rows()
            .into_iter()
            .map(|row| {
                let sum: Float = row.sum();
                if sum + self.bias > 0.0 {
                    1
                } else {
                    0
                }
            })
            .collect())
    }
}

/// Mock regressor for benchmarking
#[derive(Debug, Clone)]
struct MockRegressor {
    weight: Float,
}

impl MockRegressor {
    fn new(weight: Float) -> Self {
        Self { weight }
    }
}

impl Fit<ArrayView2<'static, Float>, ArrayView1<'static, Float>> for MockRegressor {
    type Fitted = FittedMockRegressor;

    fn fit(
        self,
        _x: &ArrayView2<'static, Float>,
        _y: &ArrayView1<'static, Float>,
    ) -> SklResult<Self::Fitted> {
        Ok(FittedMockRegressor {
            weight: self.weight,
        })
    }
}

#[derive(Debug, Clone)]
struct FittedMockRegressor {
    weight: Float,
}

impl Predict for FittedMockRegressor {
    type Input = ArrayView2<'static, Float>;
    type Output = Array1<Float>;

    fn predict(&self, x: &Self::Input) -> SklResult<Self::Output> {
        Ok(x.rows()
            .into_iter()
            .map(|row| row.sum() * self.weight)
            .collect())
    }
}

/// Generate random classification data
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
) -> (Array2<Float>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(42);
    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0));
    let y = Array1::from_shape_fn(n_samples, |_| if rng.gen::<bool>() { 1 } else { 0 });
    (x, y)
}

/// Generate random regression data
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<Float>, Array1<Float>) {
    let mut rng = StdRng::seed_from_u64(42);
    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0));
    let y = Array1::from_shape_fn(n_samples, |_| rng.gen_range(-10.0..10.0));
    (x, y)
}

/// Benchmark voting classifier with different ensemble sizes
fn bench_voting_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("voting_classifier");
    group.measurement_time(Duration::from_secs(10));

    let (x, y) = generate_classification_data(10000, 20);
    let x_view = x.view();
    let y_view = y.view();

    let ensemble_sizes = [3, 5, 10, 20, 50];

    for n_estimators in ensemble_sizes.iter() {
        group.throughput(Throughput::Elements(*n_estimators as u64));

        // Benchmark hard voting
        group.bench_with_input(
            BenchmarkId::new("hard_voting", n_estimators),
            n_estimators,
            |bench, &n| {
                let mut estimators = Vec::new();
                for i in 0..n {
                    estimators.push((format!("clf_{}", i), MockClassifier::new(i as Float * 0.1)));
                }

                bench.iter(|| {
                    let voter =
                        VotingClassifier::new(estimators.clone(), VotingType::Hard).unwrap();
                    let fitted = voter.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );

        // Benchmark soft voting (if applicable)
        group.bench_with_input(
            BenchmarkId::new("soft_voting", n_estimators),
            n_estimators,
            |bench, &n| {
                let mut estimators = Vec::new();
                for i in 0..n {
                    estimators.push((format!("clf_{}", i), MockClassifier::new(i as Float * 0.1)));
                }

                bench.iter(|| {
                    let voter =
                        VotingClassifier::new(estimators.clone(), VotingType::Soft).unwrap();
                    let fitted = voter.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark voting regressor with different ensemble sizes
fn bench_voting_regressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("voting_regressor");
    group.measurement_time(Duration::from_secs(10));

    let (x, y) = generate_regression_data(10000, 20);
    let x_view = x.view();
    let y_view = y.view();

    let ensemble_sizes = [3, 5, 10, 20, 50];

    for n_estimators in ensemble_sizes.iter() {
        group.throughput(Throughput::Elements(*n_estimators as u64));

        // Benchmark uniform weights
        group.bench_with_input(
            BenchmarkId::new("uniform_weights", n_estimators),
            n_estimators,
            |bench, &n| {
                let mut estimators = Vec::new();
                for i in 0..n {
                    estimators.push((
                        format!("reg_{}", i),
                        MockRegressor::new(1.0 + i as Float * 0.1),
                    ));
                }

                bench.iter(|| {
                    let voter = VotingRegressor::new(estimators.clone(), None).unwrap();
                    let fitted = voter.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );

        // Benchmark custom weights
        group.bench_with_input(
            BenchmarkId::new("custom_weights", n_estimators),
            n_estimators,
            |bench, &n| {
                let mut estimators = Vec::new();
                let mut weights = Vec::new();
                for i in 0..n {
                    estimators.push((
                        format!("reg_{}", i),
                        MockRegressor::new(1.0 + i as Float * 0.1),
                    ));
                    weights.push(1.0 / (i as Float + 1.0));
                }

                bench.iter(|| {
                    let voter =
                        VotingRegressor::new(estimators.clone(), Some(weights.clone())).unwrap();
                    let fitted = voter.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ensemble performance vs data size
fn bench_ensemble_data_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_data_scaling");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(100, 10), (1000, 20), (10000, 50), (50000, 100)];

    for (n_samples, n_features) in sizes.iter() {
        let (x, y) = generate_classification_data(*n_samples, *n_features);
        let x_view = x.view();
        let y_view = y.view();

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        group.bench_with_input(
            BenchmarkId::new("5_estimators", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                let estimators = vec![
                    ("clf_0".to_string(), MockClassifier::new(0.0)),
                    ("clf_1".to_string(), MockClassifier::new(0.1)),
                    ("clf_2".to_string(), MockClassifier::new(0.2)),
                    ("clf_3".to_string(), MockClassifier::new(0.3)),
                    ("clf_4".to_string(), MockClassifier::new(0.4)),
                ];

                bench.iter(|| {
                    let voter =
                        VotingClassifier::new(estimators.clone(), VotingType::Hard).unwrap();
                    let fitted = voter.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ensemble fit vs predict performance
fn bench_ensemble_fit_vs_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_fit_vs_predict");
    group.measurement_time(Duration::from_secs(10));

    let (x, y) = generate_classification_data(10000, 20);
    let x_view = x.view();
    let y_view = y.view();

    let estimators = vec![
        ("clf_0".to_string(), MockClassifier::new(0.0)),
        ("clf_1".to_string(), MockClassifier::new(0.1)),
        ("clf_2".to_string(), MockClassifier::new(0.2)),
        ("clf_3".to_string(), MockClassifier::new(0.3)),
        ("clf_4".to_string(), MockClassifier::new(0.4)),
    ];

    // Benchmark fit operation
    group.bench_function("fit_5_estimators", |bench| {
        bench.iter(|| {
            let voter = VotingClassifier::new(estimators.clone(), VotingType::Hard).unwrap();
            black_box(voter.fit(&x_view, &y_view).unwrap())
        });
    });

    // Benchmark predict operation
    let voter = VotingClassifier::new(estimators, VotingType::Hard).unwrap();
    let fitted = voter.fit(&x_view, &y_view).unwrap();

    group.bench_function("predict_5_estimators", |bench| {
        bench.iter(|| black_box(fitted.predict(&x_view).unwrap()));
    });

    group.finish();
}

/// Benchmark ensemble construction overhead
fn bench_ensemble_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_construction");
    group.measurement_time(Duration::from_secs(5));

    let ensemble_sizes = [3, 5, 10, 20, 50, 100];

    for n_estimators in ensemble_sizes.iter() {
        group.bench_with_input(
            BenchmarkId::new("construction", n_estimators),
            n_estimators,
            |bench, &n| {
                bench.iter(|| {
                    let mut estimators = Vec::new();
                    for i in 0..n {
                        estimators
                            .push((format!("clf_{}", i), MockClassifier::new(i as Float * 0.01)));
                    }
                    black_box(VotingClassifier::new(estimators, VotingType::Hard).unwrap())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_voting_classifier,
    bench_voting_regressor,
    bench_ensemble_data_scaling,
    bench_ensemble_fit_vs_predict,
    bench_ensemble_construction,
);

criterion_main!(benches);
