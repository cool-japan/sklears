//! Comprehensive benchmarks for sklears
//!
//! These benchmarks measure performance across different algorithms and data sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict, Transform};
use sklears_metrics::classification::accuracy_score;
use sklears_metrics::regression::mean_squared_error;
use sklears_model_selection::KFold;
use sklears_neighbors::{KNeighborsClassifier, KNeighborsRegressor};
use sklears_preprocessing::feature_engineering::PolynomialFeatures;
use sklears_tree::{DecisionTreeClassifier, RandomForestClassifier};
use sklears_utils::data_generation::{make_blobs, make_classification, make_regression};

fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i32>) {
    make_classification(n_samples, n_features, 3, None, None, 0.0, 1.0, Some(42)).unwrap()
}

fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    make_regression(
        n_samples,
        n_features,
        Some(n_features / 2),
        0.1,
        0.1,
        Some(42),
    )
    .unwrap()
}

fn bench_knn_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_classification");

    for &n_samples in &[100, 500, 1000, 2000] {
        for &n_features in &[5, 10, 20] {
            let (X, y) = generate_classification_data(n_samples, n_features);

            group.throughput(Throughput::Elements(n_samples as u64));
            group.bench_with_input(
                BenchmarkId::new("fit", format!("{}x{}", n_samples, n_features)),
                &(&X, &y),
                |b, (X, y)| {
                    b.iter(|| {
                        let classifier = KNeighborsClassifier::new(5);
                        black_box(classifier.fit(X, y).unwrap())
                    })
                },
            );

            // Benchmark prediction
            let classifier = KNeighborsClassifier::new(5);
            let fitted_classifier = classifier.fit(&X, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&X, &fitted_classifier),
                |b, (X, classifier)| b.iter(|| black_box(classifier.predict(X).unwrap())),
            );
        }
    }
    group.finish();
}

fn bench_knn_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_regression");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[5, 10, 20] {
            let (X, y) = generate_regression_data(n_samples, n_features);

            group.throughput(Throughput::Elements(n_samples as u64));
            group.bench_with_input(
                BenchmarkId::new("fit", format!("{}x{}", n_samples, n_features)),
                &(&X, &y),
                |b, (X, y)| {
                    b.iter(|| {
                        let regressor = KNeighborsRegressor::new(5);
                        black_box(regressor.fit(X, y).unwrap())
                    })
                },
            );

            let regressor = KNeighborsRegressor::new(5);
            let fitted_regressor = regressor.fit(&X, &y).unwrap();

            group.bench_with_input(
                BenchmarkId::new("predict", format!("{}x{}", n_samples, n_features)),
                &(&X, &fitted_regressor),
                |b, (X, regressor)| b.iter(|| black_box(regressor.predict(X).unwrap())),
            );
        }
    }
    group.finish();
}

fn bench_decision_trees(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_trees");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[5, 10, 15] {
            let (X, y_i32) = generate_classification_data(n_samples, n_features);
            let y = y_i32.mapv(|x| x as f64);

            group.throughput(Throughput::Elements(n_samples as u64));

            // Decision Tree
            group.bench_with_input(
                BenchmarkId::new("decision_tree_fit", format!("{}x{}", n_samples, n_features)),
                &(&X, &y),
                |b, (X, y)| {
                    b.iter(|| {
                        let tree = DecisionTreeClassifier::new();
                        black_box(tree.fit(*X, *y).unwrap())
                    })
                },
            );

            // Random Forest
            group.bench_with_input(
                BenchmarkId::new("random_forest_fit", format!("{}x{}", n_samples, n_features)),
                &(&X, &y_i32),
                |b, (X, y)| {
                    b.iter(|| {
                        let forest = RandomForestClassifier::new();
                        black_box(forest.fit(*X, *y).unwrap())
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    for &n_samples in &[100, 500, 1000, 5000] {
        for &n_features in &[5, 10, 20, 50] {
            let (X, _) = generate_classification_data(n_samples, n_features);

            group.throughput(Throughput::Elements((n_samples * n_features) as u64));

            // Note: Scaler benchmarks disabled - StandardScaler, MinMaxScaler, RobustScaler
            // are placeholders without full implementations
            /*
            // Standard Scaler
            group.bench_with_input(
                BenchmarkId::new(
                    "standard_scaler_fit",
                    format!("{}x{}", n_samples, n_features),
                ),
                &X,
                |b, X| {
                    b.iter(|| {
                        let scaler = StandardScaler::new();
                        black_box(scaler.fit(X, &()).unwrap())
                    })
                },
            );

            let scaler = StandardScaler::new();
            let fitted_scaler = scaler.fit(&X, &()).unwrap();

            group.bench_with_input(
                BenchmarkId::new(
                    "standard_scaler_transform",
                    format!("{}x{}", n_samples, n_features),
                ),
                &(&X, &fitted_scaler),
                |b, (X, scaler)| b.iter(|| black_box(scaler.transform(X).unwrap())),
            );

            // MinMax Scaler
            group.bench_with_input(
                BenchmarkId::new("minmax_scaler_fit", format!("{}x{}", n_samples, n_features)),
                &X,
                |b, X| {
                    b.iter(|| {
                        let scaler = MinMaxScaler::new().feature_range(0.0, 1.0);
                        black_box(scaler.fit(X, &()).unwrap())
                    })
                },
            );

            // Robust Scaler
            group.bench_with_input(
                BenchmarkId::new("robust_scaler_fit", format!("{}x{}", n_samples, n_features)),
                &X,
                |b, X| {
                    b.iter(|| {
                        let scaler = RobustScaler::new();
                        black_box(scaler.fit(X, &()).unwrap())
                    })
                },
            );
            */
        }
    }
    group.finish();
}

fn bench_feature_engineering(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_engineering");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[3, 5, 8] {
            // Keep features low for polynomial expansion
            let (X, _) = generate_classification_data(n_samples, n_features);

            for &degree in &[2, 3] {
                group.throughput(Throughput::Elements((n_samples * n_features) as u64));

                group.bench_with_input(
                    BenchmarkId::new(
                        "polynomial_fit",
                        format!("{}x{}_deg{}", n_samples, n_features, degree),
                    ),
                    &(&X, degree),
                    |b, (X, degree)| {
                        b.iter(|| {
                            let poly = PolynomialFeatures::new();
                            black_box(poly.fit(X, &()).unwrap())
                        })
                    },
                );

                let poly = PolynomialFeatures::new();
                let fitted_poly = poly.fit(&X, &()).unwrap();

                group.bench_with_input(
                    BenchmarkId::new(
                        "polynomial_transform",
                        format!("{}x{}_deg{}", n_samples, n_features, degree),
                    ),
                    &(&X, &fitted_poly),
                    |b, (X, poly)| b.iter(|| black_box(poly.transform(X).unwrap())),
                );
            }
        }
    }
    group.finish();
}

fn bench_cross_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_validation");
    group.sample_size(10); // Reduce sample size as CV is expensive

    for &n_samples in &[100, 300, 500] {
        for &n_features in &[5, 10] {
            let (X, y) = generate_classification_data(n_samples, n_features);

            group.throughput(Throughput::Elements(n_samples as u64));

            let kfold = KFold::new(5);
            let regressor = KNeighborsRegressor::new(3);

            // Cross-validation benchmark temporarily disabled due to trait issues
            // group.bench_with_input(
            //     BenchmarkId::new("knn_5fold_cv", format!("{}x{}", n_samples, n_features)),
            //     &(&X, &y, &regressor, &kfold),
            //     |b, (X, y, regressor, kfold)| {
            //         b.iter(|| {
            //             let y_float = y.mapv(|v| v as f64);
            //             black_box(
            //                 cross_val_score(
            //                     (*regressor).clone(),
            //                     X,
            //                     &y_float,
            //                     kfold.clone(),
            //                     Some(Scoring::EstimatorScore),
            //                     None,
            //                 )
            //                 .unwrap(),
            //             )
            //         })
            //     },
            // );
        }
    }
    group.finish();
}

fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");

    for &n_samples in &[100, 500, 1000, 5000, 10000] {
        // Generate predictions for accuracy benchmarking
        let y_true = Array1::from_vec((0..n_samples).map(|i| (i % 3)).collect());
        let y_pred = Array1::from_vec((0..n_samples).map(|i| ((i + 1) % 3)).collect());

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("accuracy_score", n_samples),
            &(&y_true, &y_pred),
            |b, (y_true, y_pred)| b.iter(|| black_box(accuracy_score(y_true, y_pred).unwrap())),
        );

        // Generate regression predictions
        let y_true_reg = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());
        let y_pred_reg = Array1::from_vec((0..n_samples).map(|i| (i as f64) + 0.1).collect());

        group.bench_with_input(
            BenchmarkId::new("mean_squared_error", n_samples),
            &(&y_true_reg, &y_pred_reg),
            |b, (y_true, y_pred)| b.iter(|| black_box(mean_squared_error(y_true, y_pred).unwrap())),
        );
    }
    group.finish();
}

fn bench_data_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_generation");

    for &n_samples in &[100, 500, 1000, 5000] {
        for &n_features in &[5, 10, 20] {
            group.throughput(Throughput::Elements((n_samples * n_features) as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    "make_classification",
                    format!("{}x{}", n_samples, n_features),
                ),
                &(n_samples, n_features),
                |b, &(n_samples, n_features)| {
                    b.iter(|| {
                        black_box(
                            make_classification(
                                n_samples,
                                n_features,
                                3,
                                None,
                                None,
                                0.0,
                                1.0,
                                Some(42),
                            )
                            .unwrap(),
                        )
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("make_regression", format!("{}x{}", n_samples, n_features)),
                &(n_samples, n_features),
                |b, &(n_samples, n_features)| {
                    b.iter(|| {
                        black_box(
                            make_regression(
                                n_samples,
                                n_features,
                                Some(n_features / 2),
                                0.1,
                                0.0,
                                Some(42),
                            )
                            .unwrap(),
                        )
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("make_blobs", format!("{}x{}", n_samples, n_features)),
                &(n_samples, n_features),
                |b, &(n_samples, n_features)| {
                    b.iter(|| {
                        black_box(
                            make_blobs(n_samples, n_features, Some(3), 1.0, (-5.0, 5.0), Some(42))
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test with larger datasets to measure memory efficiency
    for &n_samples in &[1000, 5000, 10000] {
        let (X, y) = generate_classification_data(n_samples, 10);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Test multiple fits/predictions to see memory usage patterns
        group.bench_with_input(
            BenchmarkId::new("repeated_knn_fit", n_samples),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    for _ in 0..10 {
                        let classifier = KNeighborsClassifier::new(5);
                        let fitted = classifier.fit(X, y).unwrap();
                        black_box(fitted.predict(X).unwrap());
                    }
                })
            },
        );

        // Note: Scaling benchmark disabled - StandardScaler is a placeholder
        /*
        // Test scaling with large datasets
        group.bench_with_input(
            BenchmarkId::new("repeated_scaling", n_samples),
            &X,
            |b, X| {
                b.iter(|| {
                    for _ in 0..10 {
                        let scaler = StandardScaler::new();
                        let fitted = scaler.fit(X, &()).unwrap();
                        black_box(fitted.transform(X).unwrap());
                    }
                })
            },
        );
        */
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_knn_classification,
    bench_knn_regression,
    bench_decision_trees,
    bench_preprocessing,
    bench_feature_engineering,
    bench_cross_validation,
    bench_metrics,
    bench_data_generation,
    bench_memory_efficiency
);

criterion_main!(benches);
