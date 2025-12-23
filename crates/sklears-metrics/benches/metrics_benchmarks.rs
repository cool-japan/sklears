//! Comprehensive Benchmarks for sklears-metrics
//!
//! This benchmark suite measures performance across all major metric categories:
//! - Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
//! - Regression metrics (MAE, MSE, RMSE, R²)
//! - Clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
//! - Ranking metrics (NDCG, MAP, MRR)
//!
//! Run with: `cargo bench --bench metrics_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::seeded_rng;
use sklears_metrics::{classification, clustering, ranking, regression};

// Helper function to generate random f64 arrays
fn generate_random_f64(size: usize, seed: u64) -> Array1<f64> {
    let mut rng = seeded_rng(seed);
    let dist = Uniform::new(0.0, 1.0).unwrap();
    Array1::from_iter((0..size).map(|_| rng.sample(&dist)))
}

// Helper function to generate random i32 arrays for classification
fn generate_random_labels(size: usize, n_classes: i32, seed: u64) -> Array1<i32> {
    let mut rng = seeded_rng(seed);
    let dist = Uniform::new(0, n_classes).unwrap();
    Array1::from_iter((0..size).map(|_| rng.sample(&dist)))
}

// Helper function to generate random 2D array for clustering
fn generate_random_2d(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = seeded_rng(seed);
    let dist = Uniform::new(-1.0, 1.0).unwrap();
    Array2::from_shape_fn((n_samples, n_features), |_| rng.sample(&dist))
}

// ============================================================================
// Classification Metrics Benchmarks
// ============================================================================

fn benchmark_classification_metrics(c: &mut Criterion) {
    let sizes = vec![100, 1000, 10_000, 100_000];

    let mut group = c.benchmark_group("classification");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let y_true = generate_random_labels(size, 2, 42);
        let y_pred = generate_random_labels(size, 2, 43);

        group.bench_with_input(BenchmarkId::new("accuracy", size), &size, |b, _| {
            b.iter(|| classification::accuracy_score(black_box(&y_true), black_box(&y_pred)))
        });

        group.bench_with_input(BenchmarkId::new("precision", size), &size, |b, _| {
            b.iter(|| {
                classification::precision_score(black_box(&y_true), black_box(&y_pred), Some(1))
            })
        });

        group.bench_with_input(BenchmarkId::new("recall", size), &size, |b, _| {
            b.iter(|| classification::recall_score(black_box(&y_true), black_box(&y_pred), Some(1)))
        });

        group.bench_with_input(BenchmarkId::new("f1", size), &size, |b, _| {
            b.iter(|| classification::f1_score(black_box(&y_true), black_box(&y_pred), Some(1)))
        });

        group.bench_with_input(BenchmarkId::new("confusion_matrix", size), &size, |b, _| {
            b.iter(|| classification::confusion_matrix(black_box(&y_true), black_box(&y_pred)))
        });
    }

    group.finish();
}

// ============================================================================
// Regression Metrics Benchmarks
// ============================================================================

fn benchmark_regression_metrics(c: &mut Criterion) {
    let sizes = vec![100, 1000, 10_000, 100_000];

    let mut group = c.benchmark_group("regression");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let y_true = generate_random_f64(size, 42);
        let y_pred = generate_random_f64(size, 43);

        group.bench_with_input(BenchmarkId::new("mae", size), &size, |b, _| {
            b.iter(|| regression::mean_absolute_error(black_box(&y_true), black_box(&y_pred)))
        });

        group.bench_with_input(BenchmarkId::new("mse", size), &size, |b, _| {
            b.iter(|| regression::mean_squared_error(black_box(&y_true), black_box(&y_pred)))
        });

        group.bench_with_input(BenchmarkId::new("rmse", size), &size, |b, _| {
            b.iter(|| regression::root_mean_squared_error(black_box(&y_true), black_box(&y_pred)))
        });

        group.bench_with_input(BenchmarkId::new("r2", size), &size, |b, _| {
            b.iter(|| regression::r2_score(black_box(&y_true), black_box(&y_pred)))
        });

        group.bench_with_input(BenchmarkId::new("mape", size), &size, |b, _| {
            b.iter(|| {
                regression::mean_absolute_percentage_error(black_box(&y_true), black_box(&y_pred))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Clustering Metrics Benchmarks
// ============================================================================

fn benchmark_clustering_metrics(c: &mut Criterion) {
    let configs = vec![
        (100, 2, "100x2"),
        (1000, 10, "1000x10"),
        (5000, 20, "5000x20"),
    ];

    let mut group = c.benchmark_group("clustering");

    for (n_samples, n_features, name) in configs {
        group.throughput(Throughput::Elements(n_samples as u64));

        let x = generate_random_2d(n_samples, n_features, 42);
        let labels_true = generate_random_labels(n_samples, 3, 42);
        let labels_pred = generate_random_labels(n_samples, 3, 43);

        group.bench_with_input(BenchmarkId::new("silhouette", name), &name, |b, _| {
            b.iter(|| clustering::silhouette_score(black_box(&x), black_box(&labels_pred)))
        });

        group.bench_with_input(BenchmarkId::new("davies_bouldin", name), &name, |b, _| {
            b.iter(|| clustering::davies_bouldin_score(black_box(&x), black_box(&labels_pred)))
        });

        group.bench_with_input(
            BenchmarkId::new("calinski_harabasz", name),
            &name,
            |b, _| {
                b.iter(|| {
                    clustering::calinski_harabasz_score(black_box(&x), black_box(&labels_pred))
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("adjusted_rand", name), &name, |b, _| {
            b.iter(|| {
                clustering::adjusted_rand_score(black_box(&labels_true), black_box(&labels_pred))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Ranking Metrics Benchmarks
// ============================================================================

fn benchmark_ranking_metrics(c: &mut Criterion) {
    let sizes = vec![100, 1000, 10_000];

    let mut group = c.benchmark_group("ranking");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let y_true = generate_random_labels(size, 2, 42);
        let y_score = generate_random_f64(size, 43);

        group.bench_with_input(BenchmarkId::new("roc_auc", size), &size, |b, _| {
            b.iter(|| ranking::roc_auc_score(black_box(&y_true), black_box(&y_score)))
        });

        group.bench_with_input(BenchmarkId::new("roc_curve", size), &size, |b, _| {
            b.iter(|| ranking::roc_curve(black_box(&y_true), black_box(&y_score)))
        });

        group.bench_with_input(
            BenchmarkId::new("average_precision", size),
            &size,
            |b, _| {
                b.iter(|| ranking::average_precision_score(black_box(&y_true), black_box(&y_score)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Parallel vs Serial Comparison
// ============================================================================

#[cfg(feature = "parallel")]
fn benchmark_parallel_vs_serial(c: &mut Criterion) {
    use sklears_metrics::optimized::{optimized_mean_absolute_error, OptimizedConfig};

    let sizes = vec![1000, 10_000, 100_000];

    let mut group = c.benchmark_group("parallel_comparison");

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let y_true = generate_random_f64(size, 42);
        let y_pred = generate_random_f64(size, 43);

        // Serial
        let config_serial = OptimizedConfig {
            use_simd: false,
            parallel_threshold: usize::MAX, // Disable parallel
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("mae_serial", size), &size, |b, _| {
            b.iter(|| {
                optimized_mean_absolute_error(
                    black_box(&y_true),
                    black_box(&y_pred),
                    Some(&config_serial),
                )
            })
        });

        // Parallel
        let config_parallel = OptimizedConfig {
            use_simd: false,
            parallel_threshold: 100, // Enable parallel
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("mae_parallel", size), &size, |b, _| {
            b.iter(|| {
                optimized_mean_absolute_error(
                    black_box(&y_true),
                    black_box(&y_pred),
                    Some(&config_parallel),
                )
            })
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    benches,
    benchmark_classification_metrics,
    benchmark_regression_metrics,
    benchmark_clustering_metrics,
    benchmark_ranking_metrics,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, benchmark_parallel_vs_serial,);

// Main benchmark runner
#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);
