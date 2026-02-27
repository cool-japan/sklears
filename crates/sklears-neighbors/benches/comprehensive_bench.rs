//! Comprehensive benchmarks comparing algorithms, distance metrics, and advanced features

#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use sklears_core::traits::{Fit, Predict, Transform};
use sklears_neighbors::{Distance, KNeighborsClassifier, OnlineMetricLearning};
use std::hint::black_box;

/// Generate synthetic classification data
fn generate_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<i32>) {
    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = (i % n_classes) as i32;
        labels.push(class);

        for j in 0..n_features {
            let base = class as f64 * 3.0 + j as f64 * 0.2;
            let noise = thread_rng().random_range(-0.5..0.5);
            data.push(base + noise);
        }
    }

    (
        Array2::from_shape_vec((n_samples, n_features), data).unwrap(),
        Array1::from(labels),
    )
}

/// Benchmark different tree algorithms
fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");

    let sizes = vec![(100, 5, "small"), (1000, 10, "medium"), (5000, 20, "large")];

    for (n_samples, n_features, label) in sizes {
        let (X, y) = generate_data(n_samples, n_features, 3);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Brute force
        group.bench_with_input(
            BenchmarkId::new("brute_fit", label),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    let knn = KNeighborsClassifier::new(10);
                    black_box(knn.fit(X, y).unwrap())
                })
            },
        );

        // KD-Tree
        group.bench_with_input(
            BenchmarkId::new("kdtree_fit", label),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    let knn = KNeighborsClassifier::new(10)
                        .with_algorithm(sklears_neighbors::knn::Algorithm::KdTree);
                    black_box(knn.fit(X, y).unwrap())
                })
            },
        );

        // Ball Tree
        group.bench_with_input(
            BenchmarkId::new("balltree_fit", label),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    let knn = KNeighborsClassifier::new(10)
                        .with_algorithm(sklears_neighbors::knn::Algorithm::BallTree);
                    black_box(knn.fit(X, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark prediction with different algorithms
fn bench_algorithm_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_prediction");

    let (X_train, y_train) = generate_data(2000, 10, 3);
    let (X_test, _) = generate_data(100, 10, 3);

    // Fit models once
    let brute_knn = KNeighborsClassifier::new(10);
    let brute_fitted = brute_knn.fit(&X_train, &y_train).unwrap();

    let kdtree_knn =
        KNeighborsClassifier::new(10).with_algorithm(sklears_neighbors::knn::Algorithm::KdTree);
    let kdtree_fitted = kdtree_knn.fit(&X_train, &y_train).unwrap();

    let balltree_knn =
        KNeighborsClassifier::new(10).with_algorithm(sklears_neighbors::knn::Algorithm::BallTree);
    let balltree_fitted = balltree_knn.fit(&X_train, &y_train).unwrap();

    group.throughput(Throughput::Elements(100));

    group.bench_function("brute_predict", |b| {
        b.iter(|| black_box(brute_fitted.predict(&X_test).unwrap()))
    });

    group.bench_function("kdtree_predict", |b| {
        b.iter(|| black_box(kdtree_fitted.predict(&X_test).unwrap()))
    });

    group.bench_function("balltree_predict", |b| {
        b.iter(|| black_box(balltree_fitted.predict(&X_test).unwrap()))
    });

    group.finish();
}

/// Benchmark different distance metrics
fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    let (X, y) = generate_data(1000, 10, 3);

    let metrics = vec![
        ("euclidean", Distance::Euclidean),
        ("manhattan", Distance::Manhattan),
        ("minkowski_3", Distance::Minkowski(3.0)),
        ("cosine", Distance::Cosine),
    ];

    for (name, metric) in metrics {
        group.bench_with_input(
            BenchmarkId::new("fit", name),
            &(&X, &y, metric),
            |b, (X, y, metric)| {
                b.iter(|| {
                    let knn = KNeighborsClassifier::new(10).with_metric(metric.clone());
                    black_box(knn.fit(X, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark online metric learning
fn bench_online_metric_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_metric_learning");

    let batch_sizes = vec![10, 50, 100];

    for batch_size in batch_sizes {
        let (X, y) = generate_data(batch_size, 5, 3);

        group.throughput(Throughput::Elements(batch_size as u64));

        // Initial fit
        group.bench_with_input(
            BenchmarkId::new("initial_fit", batch_size),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    let online = OnlineMetricLearning::new(3).with_learning_rate(0.01);
                    black_box(online.fit(X, y).unwrap())
                })
            },
        );

        // Partial fit (after initial fit)
        // Note: We need to create a fresh instance for each benchmark iteration
        // since partial_fit modifies state
        group.bench_with_input(
            BenchmarkId::new("partial_fit", batch_size),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    let online = OnlineMetricLearning::new(3).with_learning_rate(0.01);
                    let mut fitted = online.fit(X, y).unwrap();
                    fitted.partial_fit(&X.view(), &y.view()).unwrap();
                    black_box(())
                })
            },
        );

        // Transform
        let online = OnlineMetricLearning::new(3).with_learning_rate(0.01);
        let fitted = online.fit(&X, &y).unwrap();

        group.bench_with_input(
            BenchmarkId::new("transform", batch_size),
            &(&X, &fitted),
            |b, (X, fitted)| b.iter(|| black_box(fitted.transform(X).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark k-value scaling
fn bench_k_value_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_value_scaling");

    let (X, y) = generate_data(1000, 10, 3);

    for k in [1, 5, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("fit_predict", k),
            &(&X, &y, k),
            |b, (X, y, k)| {
                b.iter(|| {
                    let knn = KNeighborsClassifier::new(*k);
                    let fitted = knn.fit(X, y).unwrap();
                    black_box(fitted.predict(X).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark high-dimensional data
fn bench_high_dimensional(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional");

    let dimensions = vec![10, 50, 100];
    let n_samples = 500;

    for n_features in dimensions {
        let (X, y) = generate_data(n_samples, n_features, 3);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Brute force should work for all dimensions
        group.bench_with_input(
            BenchmarkId::new("brute", n_features),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    let knn = KNeighborsClassifier::new(10);
                    black_box(knn.fit(X, y).unwrap())
                })
            },
        );

        // Ball tree is better for high dimensions
        group.bench_with_input(
            BenchmarkId::new("balltree", n_features),
            &(&X, &y),
            |b, (X, y)| {
                b.iter(|| {
                    let knn = KNeighborsClassifier::new(10)
                        .with_algorithm(sklears_neighbors::knn::Algorithm::BallTree);
                    black_box(knn.fit(X, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency vs accuracy trade-offs
fn bench_memory_vs_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_vs_accuracy");

    let (X, y) = generate_data(1000, 10, 3);

    // Benchmark different approaches
    group.bench_function("full_knn", |b| {
        b.iter(|| {
            let knn = KNeighborsClassifier::new(10);
            let fitted = knn.fit(&X, &y).unwrap();
            black_box(fitted.predict(&X).unwrap())
        })
    });

    // Could add sparse neighbor benchmarks here if needed

    group.finish();
}

criterion_group!(
    benches,
    bench_algorithm_comparison,
    bench_algorithm_prediction,
    bench_distance_metrics,
    bench_online_metric_learning,
    bench_k_value_scaling,
    bench_high_dimensional,
    bench_memory_vs_accuracy,
);

criterion_main!(benches);
