//! Benchmarks for feature selection algorithms
//!
//! This module contains benchmarks comparing our feature selection implementations
//! against reference implementations and measuring performance characteristics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Transform};
use sklears_feature_selection::{
    ConvexFeatureSelector, LassoSelector, ParallelCorrelationComputer, ParallelFeatureRanker,
    ParallelVarianceComputer, RidgeSelector, SelectKBest, VarianceThreshold,
};

/// Generate synthetic data for benchmarking
fn generate_benchmark_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i32>) {
    let mut features = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    // Generate features with varying predictive power
    for i in 0..n_samples {
        for j in 0..n_features {
            if j < n_features / 4 {
                // Highly predictive features
                features[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.3).sin() + 0.2 * j as f64;
            } else if j < n_features / 2 {
                // Moderately predictive features
                features[[i, j]] = (i as f64 * 0.05 + j as f64 * 0.1).cos() + 0.1 * j as f64;
            } else if j < 3 * n_features / 4 {
                // Weakly predictive features
                features[[i, j]] = (i as f64 * 0.01 + j as f64 * 0.05).sin() + 0.05 * j as f64;
            } else {
                // Random noise features
                features[[i, j]] = 0.01 * (i as f64 * j as f64).sin();
            }
        }

        // Create binary target based on first few features
        let decision_value = features[[i, 0]] + 0.5 * features[[i, 1]] + 0.2 * features[[i, 2]];
        target[i] = if decision_value > 0.0 { 1 } else { 0 };
    }

    (features, target)
}

/// Generate regression data for benchmarking
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut features = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    // Generate features with varying predictive power
    for i in 0..n_samples {
        for j in 0..n_features {
            if j < n_features / 4 {
                // Highly predictive features
                features[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.3).sin() + 0.2 * j as f64;
            } else if j < n_features / 2 {
                // Moderately predictive features
                features[[i, j]] = (i as f64 * 0.05 + j as f64 * 0.1).cos() + 0.1 * j as f64;
            } else if j < 3 * n_features / 4 {
                // Weakly predictive features
                features[[i, j]] = (i as f64 * 0.01 + j as f64 * 0.05).sin() + 0.05 * j as f64;
            } else {
                // Random noise features
                features[[i, j]] = 0.01 * (i as f64 * j as f64).sin();
            }
        }

        // Create continuous target based on first few features
        target[i] = features[[i, 0]] + 0.5 * features[[i, 1]] + 0.2 * features[[i, 2]];
    }

    (features, target)
}

/// Benchmark SelectKBest vs SelectKBestParallel
fn bench_select_k_best(c: &mut Criterion) {
    let mut group = c.benchmark_group("SelectKBest");

    let sizes = vec![(100, 50), (500, 100), (1000, 200), (2000, 500)];

    for (n_samples, n_features) in sizes {
        let (features, target_i32) = generate_benchmark_data(n_samples, n_features);
        let target = target_i32.mapv(|x| x as f64); // Convert i32 to f64
        let k = n_features / 4;

        // Benchmark sequential SelectKBest
        group.bench_with_input(
            BenchmarkId::new("Sequential", format!("{}x{}", n_samples, n_features)),
            &(&features, &target, k),
            |b, (features, target, k)| {
                b.iter(|| {
                    let selector = SelectKBest::new(*k, "f_classif");
                    let trained = selector
                        .fit(&(*features).view(), &(*target).view())
                        .unwrap();
                    black_box(trained.transform(&(*features).view()).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark VarianceThreshold with different dataset sizes
fn bench_variance_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("VarianceThreshold");

    let sizes = vec![
        (100, 50),
        (500, 100),
        (1000, 200),
        (2000, 500),
        (5000, 1000),
    ];

    for (n_samples, n_features) in sizes {
        let (features, _) = generate_benchmark_data(n_samples, n_features);

        group.bench_with_input(
            BenchmarkId::new("VarianceThreshold", format!("{}x{}", n_samples, n_features)),
            &features,
            |b, features| {
                b.iter(|| {
                    let selector = VarianceThreshold::new(0.01);
                    let dummy_y = Array1::<f64>::zeros(features.nrows());
                    let trained = selector.fit(&features.view(), &dummy_y.view()).unwrap();
                    black_box(trained.transform(&features.view()).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark regularized feature selection methods
fn bench_regularized_selectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("RegularizedSelectors");

    let sizes = vec![(100, 50), (500, 100), (1000, 200)];

    for (n_samples, n_features) in sizes {
        let (features, target) = generate_regression_data(n_samples, n_features);

        // Benchmark LASSO selector
        group.bench_with_input(
            BenchmarkId::new("LASSO", format!("{}x{}", n_samples, n_features)),
            &(&features, &target),
            |b, (features, target)| {
                b.iter(|| {
                    let selector = LassoSelector::new().alpha(0.1).max_iter(100);
                    let trained = selector.fit(features, target).unwrap();
                    black_box(trained.transform(features).unwrap())
                })
            },
        );

        // Benchmark Ridge selector
        group.bench_with_input(
            BenchmarkId::new("Ridge", format!("{}x{}", n_samples, n_features)),
            &(&features, &target),
            |b, (features, target)| {
                b.iter(|| {
                    let selector = RidgeSelector::new().alpha(0.1);
                    let trained = selector.fit(features, target).unwrap();
                    black_box(trained.transform(features).unwrap())
                })
            },
        );

        // Benchmark Convex selector
        group.bench_with_input(
            BenchmarkId::new("Convex", format!("{}x{}", n_samples, n_features)),
            &(&features, &target),
            |b, (features, target)| {
                b.iter(|| {
                    let selector = ConvexFeatureSelector::new()
                        .k(n_features / 4)
                        .regularization(0.1)
                        .max_iter(50);
                    let trained = selector.fit(features, target).unwrap();
                    black_box(trained.transform(features).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel utilities
fn bench_parallel_utilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("ParallelUtilities");

    let sizes = vec![
        (100, 50),
        (500, 100),
        (1000, 200),
        (2000, 500),
        (5000, 1000),
    ];

    for (n_samples, n_features) in sizes {
        let (features, target) = generate_benchmark_data(n_samples, n_features);

        // Benchmark parallel correlation computation
        group.bench_with_input(
            BenchmarkId::new(
                "ParallelCorrelation",
                format!("{}x{}", n_samples, n_features),
            ),
            &(&features, &target),
            |b, (features, target)| {
                b.iter(|| {
                    black_box(
                        ParallelCorrelationComputer::compute_feature_target_correlation_parallel(
                            features, target,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        // Benchmark parallel variance computation
        group.bench_with_input(
            BenchmarkId::new("ParallelVariance", format!("{}x{}", n_samples, n_features)),
            &features,
            |b, features| {
                b.iter(|| {
                    black_box(
                        ParallelVarianceComputer::compute_feature_variances_parallel(features),
                    )
                })
            },
        );

        // Benchmark parallel feature ranking
        let scores = Array1::from_iter((0..n_features).map(|i| i as f64 * 0.1));
        group.bench_with_input(
            BenchmarkId::new("ParallelRanking", format!("{}features", n_features)),
            &scores,
            |b, scores| b.iter(|| black_box(ParallelFeatureRanker::rank_features_parallel(scores))),
        );
    }

    group.finish();
}

/// Benchmark memory usage and scalability
fn bench_memory_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("MemoryScalability");
    group.sample_size(10); // Reduce sample size for large datasets

    let sizes = vec![(1000, 100), (2000, 200), (5000, 500), (10000, 1000)];

    for (n_samples, n_features) in sizes {
        let (features, target_i32) = generate_benchmark_data(n_samples, n_features);
        let target = target_i32.mapv(|x| x as f64); // Convert i32 to f64

        // Test memory-efficient SelectKBest
        group.bench_with_input(
            BenchmarkId::new(
                "SelectKBest_Memory",
                format!("{}x{}", n_samples, n_features),
            ),
            &(&features, &target),
            |b, (features, target)| {
                b.iter(|| {
                    let selector = SelectKBest::new(n_features / 10, "f_classif");
                    let trained = selector
                        .fit(&(*features).view(), &(*target).view())
                        .unwrap();
                    black_box(trained.transform(&(*features).view()).unwrap())
                })
            },
        );

        // Test memory-efficient VarianceThreshold
        group.bench_with_input(
            BenchmarkId::new(
                "VarianceThreshold_Memory",
                format!("{}x{}", n_samples, n_features),
            ),
            &features,
            |b, features| {
                b.iter(|| {
                    let selector = VarianceThreshold::new(0.01);
                    let dummy_y = Array1::<f64>::zeros(features.nrows());
                    let trained = selector.fit(&features.view(), &dummy_y.view()).unwrap();
                    black_box(trained.transform(&features.view()).unwrap())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_select_k_best,
    bench_variance_threshold,
    bench_regularized_selectors,
    bench_parallel_utilities,
    bench_memory_scalability
);
criterion_main!(benches);
