//! Comprehensive benchmarks for sklears-tree
//!
//! Run with: cargo bench --bench tree_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{essentials::Normal, thread_rng};
use sklears_core::traits::{Fit, Predict};
use sklears_tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, IsolationForest, LeafModelType, ModelTree,
    RandomForestClassifier, SplitCriterion,
};

/// Generate synthetic classification dataset for DecisionTree (f64 labels)
fn generate_classification_data_f64(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample(normal);
        }
        y[i] = (i % n_classes) as f64;
    }

    (x, y)
}

/// Generate synthetic classification dataset for RandomForest (i32 labels)
fn generate_classification_data_i32(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample(normal);
        }
        y[i] = (i % n_classes) as i32;
    }

    (x, y)
}

/// Generate synthetic regression dataset
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    // Generate linear relationship with noise
    let coefficients: Vec<f64> = (0..n_features).map(|i| (i as f64) * 0.5).collect();

    for i in 0..n_samples {
        let mut target = 0.0;
        for j in 0..n_features {
            let val = rng.sample(normal);
            x[[i, j]] = val;
            target += val * coefficients[j];
        }
        y[i] = target + rng.sample(normal) * 0.1; // Add noise
    }

    (x, y)
}

fn bench_decision_tree_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_tree_classifier");

    for size in [100, 500, 1000].iter() {
        let (x, y) = generate_classification_data_f64(*size, 10, 2);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("fit", size), &size, |b, _| {
            b.iter(|| {
                let model = DecisionTreeClassifier::new()
                    .max_depth(10)
                    .criterion(SplitCriterion::Gini);

                black_box(model.fit(&x, &y).unwrap())
            });
        });

        // Benchmark prediction
        let model = DecisionTreeClassifier::new()
            .max_depth(10)
            .criterion(SplitCriterion::Gini)
            .fit(&x, &y)
            .unwrap();

        group.bench_with_input(BenchmarkId::new("predict", size), &size, |b, _| {
            b.iter(|| black_box(model.predict(&x).unwrap()));
        });
    }

    group.finish();
}

fn bench_decision_tree_regressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_tree_regressor");

    for size in [100, 500, 1000].iter() {
        let (x, y) = generate_regression_data(*size, 10);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("fit", size), &size, |b, _| {
            b.iter(|| {
                let model = DecisionTreeRegressor::new()
                    .max_depth(10)
                    .criterion(SplitCriterion::MSE);

                black_box(model.fit(&x, &y).unwrap())
            });
        });

        // Benchmark prediction
        let model = DecisionTreeRegressor::new()
            .max_depth(10)
            .criterion(SplitCriterion::MSE)
            .fit(&x, &y)
            .unwrap();

        group.bench_with_input(BenchmarkId::new("predict", size), &size, |b, _| {
            b.iter(|| black_box(model.predict(&x).unwrap()));
        });
    }

    group.finish();
}

fn bench_random_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_forest");
    group.sample_size(20); // Reduce sample size for slower benchmarks

    for size in [100, 500].iter() {
        let (x, y) = generate_classification_data_i32(*size, 10, 2);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("fit_10_trees", size), &size, |b, _| {
            b.iter(|| {
                let model = RandomForestClassifier::new()
                    .n_estimators(10)
                    .max_depth(5)
                    .criterion(SplitCriterion::Gini);

                black_box(model.fit(&x, &y).unwrap())
            });
        });

        // Benchmark prediction
        let model = RandomForestClassifier::new()
            .n_estimators(10)
            .max_depth(5)
            .criterion(SplitCriterion::Gini)
            .fit(&x, &y)
            .unwrap();

        group.bench_with_input(BenchmarkId::new("predict_10_trees", size), &size, |b, _| {
            b.iter(|| black_box(model.predict(&x).unwrap()));
        });
    }

    group.finish();
}

fn bench_isolation_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("isolation_forest");
    group.sample_size(20);

    for size in [100, 500, 1000].iter() {
        let (x, _) = generate_regression_data(*size, 10);
        let y = Array1::zeros(*size); // Dummy labels for unsupervised learning

        group.throughput(Throughput::Elements(*size as u64));

        // Standard Isolation Forest
        group.bench_with_input(BenchmarkId::new("fit_standard", size), &size, |b, _| {
            b.iter(|| {
                let model = IsolationForest::new().n_estimators(50).contamination(0.1);

                black_box(model.fit(&x, &y).unwrap())
            });
        });

        // Extended Isolation Forest
        group.bench_with_input(BenchmarkId::new("fit_extended", size), &size, |b, _| {
            b.iter(|| {
                let model = IsolationForest::new()
                    .n_estimators(50)
                    .extended(true)
                    .contamination(0.1);

                black_box(model.fit(&x, &y).unwrap())
            });
        });

        // Benchmark prediction
        let model = IsolationForest::new()
            .n_estimators(50)
            .contamination(0.1)
            .fit(&x, &y)
            .unwrap();

        group.bench_with_input(BenchmarkId::new("predict", size), &size, |b, _| {
            b.iter(|| black_box(model.predict(&x).unwrap()));
        });
    }

    group.finish();
}

fn bench_model_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_tree");

    for size in [100, 500, 1000].iter() {
        let (x, y) = generate_regression_data(*size, 5);

        group.throughput(Throughput::Elements(*size as u64));

        // Linear leaf models
        group.bench_with_input(BenchmarkId::new("fit_linear", size), &size, |b, _| {
            b.iter(|| {
                let model = ModelTree::new()
                    .max_depth(5)
                    .leaf_model(LeafModelType::Linear);

                black_box(model.fit(&x, &y).unwrap())
            });
        });

        // Constant leaf models
        group.bench_with_input(BenchmarkId::new("fit_constant", size), &size, |b, _| {
            b.iter(|| {
                let model = ModelTree::new()
                    .max_depth(5)
                    .leaf_model(LeafModelType::Constant);

                black_box(model.fit(&x, &y).unwrap())
            });
        });

        // Benchmark prediction
        let model = ModelTree::new()
            .max_depth(5)
            .leaf_model(LeafModelType::Linear)
            .fit(&x, &y)
            .unwrap();

        group.bench_with_input(BenchmarkId::new("predict", size), &size, |b, _| {
            b.iter(|| black_box(model.predict(&x).unwrap()));
        });
    }

    group.finish();
}

fn bench_tree_depth_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_depth_scaling");
    let (x_data, y_data) = generate_classification_data_f64(1000, 10, 2);

    for depth in [5, 10, 15, 20].iter() {
        group.bench_with_input(BenchmarkId::new("depth", depth), depth, |b, depth| {
            b.iter(|| {
                let model = DecisionTreeClassifier::new()
                    .max_depth(*depth)
                    .criterion(SplitCriterion::Gini);

                black_box(model.fit(&x_data, &y_data).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_tree_feature_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_feature_scaling");

    for n_features in [5, 10, 20, 50].iter() {
        let (x_data, y_data) = generate_classification_data_f64(500, *n_features, 2);

        group.throughput(Throughput::Elements(500));
        group.bench_with_input(
            BenchmarkId::new("features", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let model = DecisionTreeClassifier::new()
                        .max_depth(10)
                        .criterion(SplitCriterion::Gini);

                    black_box(model.fit(&x_data, &y_data).unwrap())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_decision_tree_classifier,
    bench_decision_tree_regressor,
    bench_random_forest,
    bench_isolation_forest,
    bench_model_tree,
    bench_tree_depth_scaling,
    bench_tree_feature_scaling,
);

criterion_main!(benches);
