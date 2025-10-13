//! Comprehensive SVM Performance Benchmarks
//!
//! This benchmark suite compares the performance of different SVM implementations
//! and algorithms across various dataset sizes and complexities.

//!
//! NOTE: This benchmark is currently disabled due to incomplete API implementation.
//! Enable with `--features incomplete-benchmarks` once the required types are implemented.

#![cfg(feature = "incomplete-benchmarks")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use sklears_svm::{
    kernels::{Kernel, KernelType},
    linear_svc::LinearSVC,
    ls_svm::LSSVM,
    online_svm::OnlineSVM,
    sgd_svm::SGDClassifier,
    svc::SVC,
    svr::SVR,
};
use std::time::Duration;

/// Generate synthetic dataset for benchmarking
fn generate_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Generate random features
    let X_var = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-5.0..5.0));

    // Generate labels based on simple linear combination with noise
    let weights: Array1<f64> = Array1::from_shape_fn(n_features, |_| rng.gen_range(-1.0..1.0));
    let y = X.dot(&weights).mapv(|x| {
        let class = ((x + rng.gen_range(-0.5..0.5)) / 2.0).floor() as i32;
        (class % n_classes as i32) as f64
    });

    (X, y)
}

/// Generate regression dataset
fn generate_regression_dataset(
    n_samples: usize,
    n_features: usize,
    noise_level: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let X_var = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-5.0..5.0));
    let weights: Array1<f64> = Array1::from_shape_fn(n_features, |_| rng.gen_range(-1.0..1.0));
    let y = X
        .dot(&weights)
        .mapv(|x| x + rng.gen_range(-noise_level..noise_level));

    (X, y)
}

/// Benchmark SVC training across different algorithms
fn bench_svc_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("svc_training");

    let sizes = vec![100, 500, 1000, 2000];
    let n_features = 20;

    for size in sizes {
        let (X, y) = generate_dataset(size, n_features, 2, 42);

        group.throughput(Throughput::Elements(size as u64));

        // Linear SVM
        group.bench_with_input(
            BenchmarkId::new("Linear_SVM", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut svm = SVC::new().kernel(KernelType::Linear).c(1.0).build();
                    black_box(svm.fit(X, y).unwrap())
                })
            },
        );

        // RBF SVM
        group.bench_with_input(
            BenchmarkId::new("RBF_SVM", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut svm = SVC::new()
                        .kernel(KernelType::Rbf { gamma: 0.1 })
                        .c(1.0)
                        .build();
                    black_box(svm.fit(X, y).unwrap())
                })
            },
        );

        // Linear SVC (coordinate descent)
        group.bench_with_input(
            BenchmarkId::new("LinearSVC", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut svm = LinearSVC::new().with_c(1.0).with_max_iter(1000).build();
                    black_box(svm.fit(X, y).unwrap())
                })
            },
        );

        // SGD Classifier
        group.bench_with_input(
            BenchmarkId::new("SGD_Classifier", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut sgd = SGDClassifier::new()
                        .loss_function("hinge")
                        .learning_rate(0.01)
                        .with_max_iter(1000)
                        .build();
                    black_box(sgd.fit(X, y).unwrap())
                })
            },
        );

        // Least Squares SVM
        group.bench_with_input(
            BenchmarkId::new("LS_SVM", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut lssvm = LSSVM::new().gamma(1.0).sigma(0.1).build();
                    black_box(lssvm.fit(X, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark SVR training
fn bench_svr_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("svr_training");

    let sizes = vec![100, 500, 1000, 2000];
    let n_features = 20;

    for size in sizes {
        let (X, y) = generate_regression_dataset(size, n_features, 0.1, 42);

        group.throughput(Throughput::Elements(size as u64));

        // Linear SVR
        group.bench_with_input(
            BenchmarkId::new("Linear_SVR", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut svr = SVR::new()
                        .kernel(KernelType::Linear)
                        .c(1.0)
                        .epsilon(0.1)
                        .build();
                    black_box(svr.fit(X, y).unwrap())
                })
            },
        );

        // RBF SVR
        group.bench_with_input(
            BenchmarkId::new("RBF_SVR", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut svr = SVR::new()
                        .kernel(KernelType::Rbf { gamma: 0.1 })
                        .c(1.0)
                        .epsilon(0.1)
                        .build();
                    black_box(svr.fit(X, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark prediction performance
fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");

    let train_size = 1000;
    let n_features = 20;
    let test_sizes = vec![100, 500, 1000, 5000];

    let (X_train, y_train) = generate_dataset(train_size, n_features, 2, 42);

    // Pre-train models
    let mut linear_svm = SVC::new().kernel(KernelType::Linear).c(1.0).build();
    linear_svm.fit(&X_train, &y_train).unwrap();

    let mut rbf_svm = SVC::new()
        .kernel(KernelType::Rbf { gamma: 0.1 })
        .c(1.0)
        .build();
    rbf_svm.fit(&X_train, &y_train).unwrap();

    for test_size in test_sizes {
        let (X_test, _) = generate_dataset(test_size, n_features, 2, 123);

        group.throughput(Throughput::Elements(test_size as u64));

        group.bench_with_input(
            BenchmarkId::new("Linear_SVM_predict", test_size),
            &X_test,
            |b, X_test| b.iter(|| black_box(linear_svm.predict(X_test).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("RBF_SVM_predict", test_size),
            &X_test,
            |b, X_test| b.iter(|| black_box(rbf_svm.predict(X_test).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark online learning
fn bench_online_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_learning");

    let n_features = 20;
    let batch_sizes = vec![10, 50, 100, 500];

    for batch_size in batch_sizes {
        let (X, y) = generate_dataset(batch_size, n_features, 2, 42);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("OnlineSVM_update", batch_size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter_batched(
                    || OnlineSVM::new().c(1.0).budget(100).build(),
                    |mut online_svm| {
                        for i in 0..X.nrows() {
                            let x_i = X.row(i).to_owned();
                            let y_i = y[i];
                            black_box(
                                online_svm
                                    .partial_fit(
                                        &x_i.view().insert_axis(scirs2_core::ndarray::Axis(0)),
                                        &Array1::from_vec(vec![y_i]),
                                    )
                                    .unwrap(),
                            );
                        }
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark kernel computation
fn bench_kernel_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_computation");

    let sizes = vec![100, 500, 1000];
    let n_features = 50;

    for size in sizes {
        let (X, _) = generate_dataset(size, n_features, 2, 42);

        group.throughput(Throughput::Elements((size * size) as u64));

        // Linear kernel
        group.bench_with_input(BenchmarkId::new("Linear_kernel", size), &X, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Linear);
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // RBF kernel
        group.bench_with_input(BenchmarkId::new("RBF_kernel", size), &X, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // Polynomial kernel
        group.bench_with_input(BenchmarkId::new("Polynomial_kernel", size), &X, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Polynomial {
                    degree: 3,
                    coef0: 1.0,
                });
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark memory usage and scalability
fn bench_memory_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scalability");
    group.sample_size(10); // Fewer samples for large datasets
    group.measurement_time(Duration::from_secs(30));

    let sizes = vec![1000, 2000, 5000, 10000];
    let n_features = 20;

    for size in sizes {
        let (X, y) = generate_dataset(size, n_features, 2, 42);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Large_dataset_training", size),
            &(X.clone(), y.clone()),
            |b, (X, y)| {
                b.iter(|| {
                    let mut svm = LinearSVC::new()
                        .c(1.0)
                        .with_max_iter(100) // Reduced iterations for large datasets
                        .build();
                    black_box(svm.fit(X, y).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different optimization algorithms
fn bench_optimization_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_algorithms");

    let size = 1000;
    let n_features = 20;
    let (X, y) = generate_dataset(size, n_features, 2, 42);

    group.throughput(Throughput::Elements(size as u64));

    // SMO algorithm
    group.bench_function("SMO_algorithm", |b| {
        b.iter(|| {
            let mut svm = SVC::new()
                .kernel(KernelType::Rbf { gamma: 0.1 })
                .c(1.0)
                .build();
            black_box(svm.fit(&X, &y).unwrap())
        })
    });

    // Coordinate descent
    group.bench_function("Coordinate_descent", |b| {
        b.iter(|| {
            let mut svm = LinearSVC::new().with_c(1.0).with_max_iter(1000).build();
            black_box(svm.fit(&X, &y).unwrap())
        })
    });

    // SGD
    group.bench_function("SGD_optimization", |b| {
        b.iter(|| {
            let mut sgd = SGDClassifier::new()
                .loss_function("hinge")
                .learning_rate(0.01)
                .with_max_iter(1000)
                .build();
            black_box(sgd.fit(&X, &y).unwrap())
        })
    });

    group.finish();
}

criterion_group!(
    svm_benches,
    bench_svc_training,
    bench_svr_training,
    bench_prediction,
    bench_online_learning,
    bench_kernel_computation,
    bench_memory_scalability,
    bench_optimization_algorithms
);

criterion_main!(svm_benches);
