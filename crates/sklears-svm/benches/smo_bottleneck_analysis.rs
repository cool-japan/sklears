//! SMO Solver Performance Bottleneck Analysis
//!
//! This benchmark identifies specific bottlenecks in the SMO algorithm
//! for small datasets where performance should be nearly instantaneous.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::Fit;
use sklears_svm::svc::SVC;
use std::hint::black_box;
use std::time::Duration;

/// Generate simple linearly separable dataset
fn generate_simple_dataset(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();

    // Generate positive class samples
    for i in 0..(n_samples / 2) {
        for j in 0..n_features {
            x_vec.push((i as f64 + 1.0) + (j as f64 * 0.1));
        }
        y_vec.push(1.0);
    }

    // Generate negative class samples
    for i in 0..(n_samples / 2) {
        for j in 0..n_features {
            x_vec.push(-(i as f64 + 1.0) - (j as f64 * 0.1));
        }
        y_vec.push(0.0);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_vec).expect("Failed to create array");
    let y = Array1::from_vec(y_vec);

    (x, y)
}

/// Benchmark very small datasets (should be near-instantaneous)
fn bench_tiny_datasets(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiny_datasets");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    let sizes = vec![6, 10, 20];
    let n_features = 2;

    for size in sizes {
        let (x, y) = generate_simple_dataset(size, n_features);

        group.bench_with_input(
            BenchmarkId::new("linear_kernel_default_params", size),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    let svc = SVC::new().linear();
                    black_box(svc.fit(x, y))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("linear_kernel_relaxed_tol", size),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    let svc = SVC::new().linear().tol(0.1).max_iter(10);
                    black_box(svc.fit(x, y))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("linear_kernel_no_shrinking", size),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    let svc = SVC::new().linear().tol(0.1).max_iter(10).shrinking(false);
                    black_box(svc.fit(x, y))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark medium datasets to observe scaling
fn bench_medium_datasets(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_datasets");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let sizes = vec![50, 100, 200];
    let n_features = 5;

    for size in sizes {
        let (x, y) = generate_simple_dataset(size, n_features);

        group.bench_with_input(
            BenchmarkId::new("linear_svc", size),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    let svc = SVC::new().linear().tol(0.01).max_iter(100);
                    black_box(svc.fit(x, y))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark kernel computation overhead
fn bench_kernel_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_overhead");
    group.measurement_time(Duration::from_secs(10));

    let size = 20;
    let n_features = 2;
    let (x, y) = generate_simple_dataset(size, n_features);

    // Linear kernel (simplest case)
    group.bench_function("linear_kernel_20samples", |b| {
        b.iter(|| {
            let svc = SVC::new().linear().tol(0.1).max_iter(10);
            black_box(svc.fit(&x, &y))
        })
    });

    // RBF kernel (more complex)
    group.bench_function("rbf_kernel_20samples", |b| {
        b.iter(|| {
            let svc = SVC::new().rbf(Some(1.0)).tol(0.1).max_iter(10);
            black_box(svc.fit(&x, &y))
        })
    });

    group.finish();
}

/// Benchmark different tolerance settings
fn bench_tolerance_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tolerance_impact");
    group.measurement_time(Duration::from_secs(10));

    let size = 20;
    let n_features = 2;
    let (x, y) = generate_simple_dataset(size, n_features);

    let tolerances = vec![0.001, 0.01, 0.1, 0.5];

    for tol in tolerances {
        group.bench_with_input(
            BenchmarkId::new("tolerance", format!("{:.3}", tol)),
            &tol,
            |b, &tol| {
                b.iter(|| {
                    let svc = SVC::new().linear().tol(tol).max_iter(50);
                    black_box(svc.fit(&x, &y))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark max_iter impact
fn bench_max_iter_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_iter_impact");
    group.measurement_time(Duration::from_secs(10));

    let size = 20;
    let n_features = 2;
    let (x, y) = generate_simple_dataset(size, n_features);

    let max_iters = vec![5, 10, 20, 50, 100];

    for max_iter in max_iters {
        group.bench_with_input(
            BenchmarkId::new("max_iter", max_iter),
            &max_iter,
            |b, &max_iter| {
                b.iter(|| {
                    let svc = SVC::new().linear().tol(0.1).max_iter(max_iter);
                    black_box(svc.fit(&x, &y))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    smo_bottlenecks,
    bench_tiny_datasets,
    bench_medium_datasets,
    bench_kernel_overhead,
    bench_tolerance_impact,
    bench_max_iter_impact
);

criterion_main!(smo_bottlenecks);
