//! Continuous benchmarking suite for sklears
//!
//! This benchmark suite is designed to track performance over time
//! and detect regressions in CI/CD pipelines.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;

// Import specific types we'll use
use sklears::{
    clustering::{KMeans, KMeansConfig},
    ensemble::{AdaBoostClassifier, GradientBoostingRegressor},
    linear::LinearRegression,
    neural::MLPClassifier,
    tree::{DecisionTreeClassifier, RandomForestClassifier},
};
use sklears_core::traits::{Fit, Predict};
// Note: Lasso, Ridge, StandardScaler, MinMaxScaler, PCA not yet fully exported or implemented

// Benchmark configuration
struct BenchmarkConfig {
    pub small_size: usize,
    pub medium_size: usize,
    pub large_size: usize,
    pub n_features: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            small_size: 100,
            medium_size: 1000,
            large_size: 10000,
            n_features: 10,
        }
    }
}

/// Generate synthetic data for benchmarking
fn generate_benchmark_data(n_samples: usize, n_features: usize) -> (Array2<Float>, Array1<Float>) {
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::Rng;
    use scirs2_core::random::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);

    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.random::<Float>() * 10.0);
    let y = Array1::from_shape_fn(n_samples, |i| (i % 2) as Float);

    (x, y)
}

/// Generate classification data with multiple classes
fn generate_multiclass_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<Float>, Array1<i32>) {
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::Rng;
    use scirs2_core::random::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);

    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.random::<Float>() * 10.0);
    let y = Array1::from_shape_fn(n_samples, |i| (i % n_classes) as i32);

    (x, y)
}

/// Configure criterion for consistent benchmarking
fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(3))
}

/// Benchmark linear models
fn benchmark_linear_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear Models");
    let config = BenchmarkConfig::default();

    for size in &[config.small_size, config.medium_size, config.large_size] {
        let (x, y) = generate_benchmark_data(*size, config.n_features);

        // LinearRegression
        group.bench_with_input(
            BenchmarkId::new("LinearRegression/fit", size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let model = LinearRegression::new()
                        .fit_intercept(true)
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );

        // Note: Ridge and Lasso benchmarks disabled - not yet exported from sklears facade
        /*
        // Ridge
        group.bench_with_input(
            BenchmarkId::new("Ridge/fit", size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let model = Ridge::new()
                        .alpha(1.0)
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );

        // Lasso
        group.bench_with_input(
            BenchmarkId::new("Lasso/fit", size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let model = Lasso::new()
                        .alpha(0.1)
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );
        */
    }

    group.finish();
}

/// Benchmark tree-based models
fn benchmark_tree_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tree Models");
    let config = BenchmarkConfig::default();

    for size in &[config.small_size, config.medium_size] {
        let (x, y_i32) = generate_multiclass_data(*size, config.n_features, 3);
        let y = y_i32.mapv(|v| v as f64);

        // DecisionTree
        group.bench_with_input(
            BenchmarkId::new("DecisionTreeClassifier/fit", size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let model = DecisionTreeClassifier::new()
                        .max_depth(10)
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );

        // RandomForest
        group.bench_with_input(
            BenchmarkId::new("RandomForestClassifier/fit", size),
            &(&x, &y_i32),
            |b, (x, y)| {
                b.iter(|| {
                    let model = RandomForestClassifier::new()
                        .n_estimators(10)
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ensemble methods
fn benchmark_ensemble_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ensemble Methods");
    let config = BenchmarkConfig::default();

    for size in &[config.small_size, config.medium_size] {
        let (x, y) = generate_benchmark_data(*size, config.n_features);

        // GradientBoosting
        group.bench_with_input(
            BenchmarkId::new("GradientBoostingRegressor/fit", size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let model = GradientBoostingRegressor::builder()
                        .n_estimators(10)
                        .learning_rate(0.1)
                        .build()
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );

        // AdaBoost
        let y_binary = y.mapv(|v| v.round());
        group.bench_with_input(
            BenchmarkId::new("AdaBoostClassifier/fit", size),
            &(&x, &y_binary),
            |b, (x, y)| {
                b.iter(|| {
                    let model = AdaBoostClassifier::new()
                        .n_estimators(10)
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark preprocessing operations
fn benchmark_preprocessing(c: &mut Criterion) {
    let group = c.benchmark_group("Preprocessing");
    // Note: StandardScaler, MinMaxScaler, and PCA benchmarks disabled -
    // these types are not yet exported from sklears facade or fully implemented

    /*
    let config = BenchmarkConfig::default();

    for size in &[config.small_size, config.medium_size, config.large_size] {
        let (x, _) = generate_benchmark_data(*size, config.n_features);

        // StandardScaler
        group.bench_with_input(
            BenchmarkId::new("StandardScaler/fit_transform", size),
            &x,
            |b, x| {
                b.iter(|| {
                    let scaler = StandardScaler::new().fit(black_box(x)).unwrap();
                    let transformed = scaler.transform(black_box(x)).unwrap();
                    black_box(transformed)
                });
            },
        );

        // MinMaxScaler
        group.bench_with_input(
            BenchmarkId::new("MinMaxScaler/fit_transform", size),
            &x,
            |b, x| {
                b.iter(|| {
                    let scaler = MinMaxScaler::new().fit(black_box(x)).unwrap();
                    let transformed = scaler.transform(black_box(x)).unwrap();
                    black_box(transformed)
                });
            },
        );

        // PCA
        group.bench_with_input(BenchmarkId::new("PCA/fit_transform", size), &x, |b, x| {
            b.iter(|| {
                let pca = PCA::new().n_components(5).fit(black_box(x)).unwrap();
                let transformed = pca.transform(black_box(x)).unwrap();
                black_box(transformed)
            });
        });
    }
    */

    group.finish();
}

/// Benchmark clustering algorithms
fn benchmark_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("Clustering");
    let config = BenchmarkConfig::default();

    for size in &[config.small_size, config.medium_size] {
        let (x, _) = generate_benchmark_data(*size, config.n_features);

        // KMeans
        group.bench_with_input(BenchmarkId::new("KMeans/fit_predict", size), &x, |b, x| {
            b.iter(|| {
                let config = KMeansConfig {
                    n_clusters: 3,
                    ..Default::default()
                };
                let kmeans_model = KMeans::new(config);
                let y_dummy = Array1::zeros(x.nrows());
                let fitted = kmeans_model.fit(black_box(x), &y_dummy).unwrap();
                let labels = fitted.predict(black_box(x)).unwrap();
                black_box(labels)
            });
        });
    }

    group.finish();
}

/// Benchmark neural networks
fn benchmark_neural_networks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Neural Networks");
    let config = BenchmarkConfig::default();

    {
        let size = &config.small_size;
        let (x, y_i32) = generate_multiclass_data(*size, config.n_features, 3);
        let y: Vec<usize> = y_i32.iter().map(|&v| v as usize).collect();

        // MLPClassifier
        group.bench_with_input(
            BenchmarkId::new("MLPClassifier/fit", size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let model = MLPClassifier::new()
                        .hidden_layer_sizes(&[10, 5])
                        .max_iter(100)
                        .fit(black_box(x), black_box(y))
                        .unwrap();
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark distance calculations
fn benchmark_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("Distance Calculations");

    for size in &[10, 100, 1000] {
        let a = Array1::<Float>::from_shape_fn(*size, |i| i as Float);
        let b = Array1::<Float>::from_shape_fn(*size, |i| (i * 2) as Float);

        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    use sklears::neighbors::distance::euclidean_distance;
                    let dist = euclidean_distance(&(*a).view(), &(*b).view());
                    black_box(dist)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("manhattan_distance", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    use sklears::neighbors::distance::manhattan_distance;
                    let dist = manhattan_distance(&(*a).view(), &(*b).view());
                    black_box(dist)
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        benchmark_linear_models,
        benchmark_tree_models,
        benchmark_ensemble_methods,
        benchmark_preprocessing,
        benchmark_clustering,
        benchmark_neural_networks,
        benchmark_distances
}

criterion_main!(benches);
