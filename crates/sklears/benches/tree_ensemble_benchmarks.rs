//! Benchmarks for tree-based models and ensemble methods
//!
//! This benchmark suite focuses on the algorithms we've been implementing and improving.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict};
use sklears_ensemble::{AdaBoostClassifier, StackingClassifier};
use sklears_tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, SplitCriterion,
};
use sklears_utils::data_generation::{make_classification, make_regression};
use std::hint::black_box;

fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i32>) {
    make_classification(n_samples, n_features, 3, None, None, 0.0, 1.0, Some(42)).unwrap()
}

fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    make_regression(
        n_samples,
        n_features,
        Some(n_features / 2),
        0.1,
        0.0,
        Some(42),
    )
    .unwrap()
}

// Decision Tree Benchmarks
fn bench_decision_tree_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_tree_classification");

    for &n_samples in &[100, 500, 1000, 2000] {
        for &n_features in &[5, 10, 20] {
            let (x, y_i32) = generate_classification_data(n_samples, n_features);
            let y = y_i32.mapv(|x| x as f64); // Convert to f64 for tree API

            // Benchmark different split criteria
            for criterion in [SplitCriterion::Gini, SplitCriterion::Entropy] {
                group.throughput(Throughput::Elements(n_samples as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("fit_{:?}", criterion),
                        format!("{}x{}", n_samples, n_features),
                    ),
                    &(&x, &y),
                    |b, (x, y)| {
                        b.iter(|| {
                            let tree = DecisionTreeClassifier::new()
                                .criterion(criterion)
                                .max_depth(10);
                            black_box(tree.fit(*x, *y).unwrap())
                        })
                    },
                );
            }

            // Benchmark prediction
            let tree = DecisionTreeClassifier::new()
                .criterion(SplitCriterion::Gini)
                .max_depth(10);
            let fitted_tree = tree.fit(&x, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&x, &fitted_tree),
                |b, (x, fitted_tree)| b.iter(|| black_box(fitted_tree.predict(x).unwrap())),
            );
        }
    }
    group.finish();
}

fn bench_decision_tree_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_tree_regression");

    for &n_samples in &[100, 500, 1000, 2000] {
        for &n_features in &[5, 10, 20] {
            let (x, y) = generate_regression_data(n_samples, n_features);

            // Benchmark different split criteria
            for criterion in [SplitCriterion::MSE, SplitCriterion::MAE] {
                group.throughput(Throughput::Elements(n_samples as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("fit_{:?}", criterion),
                        format!("{}x{}", n_samples, n_features),
                    ),
                    &(&x, &y),
                    |b, (x, y)| {
                        b.iter(|| {
                            let tree = DecisionTreeRegressor::new()
                                .criterion(criterion)
                                .max_depth(10);
                            black_box(tree.fit(x, y).unwrap())
                        })
                    },
                );
            }

            // Benchmark prediction
            let tree = DecisionTreeRegressor::new()
                .criterion(SplitCriterion::MSE)
                .max_depth(10);
            let fitted_tree = tree.fit(&x, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&x, &fitted_tree),
                |b, (x, fitted_tree)| b.iter(|| black_box(fitted_tree.predict(x).unwrap())),
            );
        }
    }
    group.finish();
}

// Random Forest Benchmarks
fn bench_random_forest_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_forest_classification");

    for &n_samples in &[500, 1000] {
        // Fewer samples due to computational cost
        for &n_features in &[10, 20] {
            let (x, y) = generate_classification_data(n_samples, n_features);

            // Benchmark different numbers of estimators
            for n_estimators in [10, 50, 100] {
                group.throughput(Throughput::Elements(n_samples as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("fit_{}trees", n_estimators),
                        format!("{}x{}", n_samples, n_features),
                    ),
                    &(&x, &y),
                    |b, (x, y)| {
                        b.iter(|| {
                            let rf = RandomForestClassifier::new()
                                .n_estimators(n_estimators)
                                .criterion(SplitCriterion::Gini)
                                .random_state(42);
                            black_box(rf.fit(x, y).unwrap())
                        })
                    },
                );
            }

            // Benchmark prediction
            let rf = RandomForestClassifier::new()
                .n_estimators(50)
                .random_state(42);
            let fitted_rf = rf.fit(&x, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&x, &fitted_rf),
                |b, (x, fitted_rf)| b.iter(|| black_box(fitted_rf.predict(x).unwrap())),
            );

            // Benchmark feature importance calculation
            group.bench_with_input(
                BenchmarkId::new(
                    "feature_importances",
                    format!("{}x{}", n_samples, n_features),
                ),
                &fitted_rf,
                |b, fitted_rf| b.iter(|| black_box(fitted_rf.feature_importances().unwrap())),
            );
        }
    }
    group.finish();
}

// Commented out: RandomForestRegressor not yet implemented
#[cfg(any())] // Never compile - RandomForestRegressor not implemented
fn _bench_random_forest_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_forest_regression");

    for &n_samples in &[500, 1000] {
        for &n_features in &[10, 20] {
            let (x, y) = generate_regression_data(n_samples, n_features);

            // Benchmark different numbers of estimators
            for n_estimators in [10, 50] {
                group.throughput(Throughput::Elements(n_samples as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("fit_{}trees", n_estimators),
                        format!("{}x{}", n_samples, n_features),
                    ),
                    &(&x, &y),
                    |b, (x, y)| {
                        b.iter(|| {
                            let rf = RandomForestRegressor::new()
                                .n_estimators(n_estimators)
                                .criterion(SplitCriterion::MSE)
                                .random_state(42);
                            black_box(rf.fit(x, y).unwrap())
                        })
                    },
                );
            }

            // Benchmark prediction and feature importance
            let rf = RandomForestRegressor::new()
                .n_estimators(25)
                .criterion(SplitCriterion::MSE)
                .random_state(42);
            let fitted_rf = rf.fit(&x, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&x, &fitted_rf),
                |b, (x, fitted_rf)| b.iter(|| black_box(fitted_rf.predict(x).unwrap())),
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "feature_importances",
                    format!("{}x{}", n_samples, n_features),
                ),
                &fitted_rf,
                |b, fitted_rf| b.iter(|| black_box(fitted_rf.feature_importances().unwrap())),
            );
        }
    }
    group.finish();
}

// Extra Trees Benchmarks
// Commented out: ExtraTreesClassifier not yet implemented
#[cfg(any())] // Never compile - ExtraTreesClassifier not implemented
fn _bench_extra_trees_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("extra_trees_classification");

    for &n_samples in &[500, 1000] {
        for &n_features in &[10, 20] {
            let (x, y) = generate_classification_data(n_samples, n_features);

            group.throughput(Throughput::Elements(n_samples as u64));
            group.bench_with_input(
                BenchmarkId::new("fit", format!("{}x{}", n_samples, n_features)),
                &(&x, &y),
                |b, (x, y)| {
                    b.iter(|| {
                        let et = ExtraTreesClassifier::new()
                            .n_estimators(50)
                            .random_state(Some(42));
                        black_box(et.fit(x, y).unwrap())
                    })
                },
            );

            // Benchmark prediction
            let et = ExtraTreesClassifier::new()
                .n_estimators(50)
                .random_state(Some(42));
            let fitted_et = et.fit(&x, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&x, &fitted_et),
                |b, (x, fitted_et)| b.iter(|| black_box(fitted_et.predict(x).unwrap())),
            );
        }
    }
    group.finish();
}

// AdaBoost Benchmarks
fn bench_adaboost_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaboost_classification");

    for &n_samples in &[200, 500] {
        // Smaller due to boosting computational cost
        for &n_features in &[5, 10] {
            let (x, y) = generate_classification_data(n_samples, n_features);

            // Benchmark different numbers of estimators
            for n_estimators in [10, 25, 50] {
                group.throughput(Throughput::Elements(n_samples as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("fit_{}estimators", n_estimators),
                        format!("{}x{}", n_samples, n_features),
                    ),
                    &(&x, &y),
                    |b, (x, y)| {
                        b.iter(|| {
                            let ada = AdaBoostClassifier::new()
                                .n_estimators(n_estimators)
                                .learning_rate(1.0)
                                .random_state(42);
                            black_box(ada.fit(x, &y.mapv(|v| v as f64)).unwrap())
                        })
                    },
                );
            }

            // Benchmark prediction
            let ada = AdaBoostClassifier::new().n_estimators(25).random_state(42);
            let fitted_ada = ada.fit(&x, &y.mapv(|v| v as f64)).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&x, &fitted_ada),
                |b, (x, fitted_ada)| b.iter(|| black_box(fitted_ada.predict(x).unwrap())),
            );

            // Benchmark feature importance calculation
            group.bench_with_input(
                BenchmarkId::new(
                    "feature_importances",
                    format!("{}x{}", n_samples, n_features),
                ),
                &fitted_ada,
                |b, fitted_ada| b.iter(|| black_box(fitted_ada.feature_importances().unwrap())),
            );
        }
    }
    group.finish();
}

// Stacking Benchmarks
fn bench_stacking_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("stacking_classification");

    for &n_samples in &[200, 400] {
        // Smaller due to stacking computational cost
        for &n_features in &[5, 10] {
            let (x, y) = generate_classification_data(n_samples, n_features);

            group.throughput(Throughput::Elements(n_samples as u64));
            group.bench_with_input(
                BenchmarkId::new("fit", format!("{}x{}", n_samples, n_features)),
                &(&x, &y),
                |b, (x, y)| {
                    b.iter(|| {
                        let stacking = StackingClassifier::new(3).cv(3).random_state(42);
                        black_box(stacking.fit(x, y).unwrap())
                    })
                },
            );

            // Benchmark prediction
            let stacking = StackingClassifier::new(3).cv(3).random_state(42);
            let fitted_stacking = stacking.fit(&x, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&x, &fitted_stacking),
                |b, (x, fitted_stacking)| b.iter(|| black_box(fitted_stacking.predict(x).unwrap())),
            );
        }
    }
    group.finish();
}

// Tree depth comparison benchmark
fn bench_tree_depth_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_depth_comparison");

    let (x, y_i32) = generate_classification_data(1000, 10);
    let y = y_i32.mapv(|x| x as f64);

    for max_depth in [Some(3), Some(5), Some(10), Some(15), None] {
        let depth_str = match max_depth {
            Some(d) => d.to_string(),
            None => "unlimited".to_string(),
        };

        group.bench_with_input(
            BenchmarkId::new("decision_tree", &depth_str),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut tree = DecisionTreeClassifier::new().criterion(SplitCriterion::Gini);

                    if let Some(depth) = max_depth {
                        tree = tree.max_depth(depth);
                    }

                    black_box(tree.fit(*x, *y).unwrap())
                })
            },
        );
    }
    group.finish();
}

// Memory usage benchmark for tree models
fn bench_tree_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_memory_patterns");

    let (x, y_i32) = generate_classification_data(500, 10);
    let y = y_i32.mapv(|x| x as f64);

    // Benchmark repeated fits to check for memory leaks
    group.bench_function("repeated_decision_tree_fits", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let tree = DecisionTreeClassifier::new()
                    .max_depth(10)
                    .random_state(Some(42));
                let fitted = tree.fit(&x, &y).unwrap();
                black_box(fitted.predict(&x).unwrap());
            }
        })
    });

    group.bench_function("repeated_random_forest_fits", |b| {
        b.iter(|| {
            for _ in 0..3 {
                // Fewer iterations due to cost
                let rf = RandomForestClassifier::new()
                    .n_estimators(10)
                    .max_depth(5)
                    .random_state(42);
                let fitted = rf.fit(&x, &y_i32).unwrap();
                black_box(fitted.predict(&x).unwrap());
            }
        })
    });

    group.finish();
}

criterion_group!(
    tree_ensemble_benches,
    bench_decision_tree_classification,
    bench_decision_tree_regression,
    bench_random_forest_classification,
    // bench_random_forest_regression, // Commented out: RandomForestRegressor not implemented
    // bench_extra_trees_classification, // Commented out: ExtraTreesClassifier not implemented
    bench_adaboost_classification,
    bench_stacking_classification,
    bench_tree_depth_comparison,
    bench_tree_memory_patterns
);

criterion_main!(tree_ensemble_benches);
