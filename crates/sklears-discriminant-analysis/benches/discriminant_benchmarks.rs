//! Comprehensive benchmarking framework for discriminant analysis algorithms
//!
//! This module provides benchmarking utilities for comparing performance of
//! different discriminant analysis methods across various datasets and scenarios.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_discriminant_analysis::{
    DiagonalLinearDiscriminantAnalysis, KernelDiscriminantAnalysis, KernelType,
    LinearDiscriminantAnalysis, MixtureDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
};
use std::hint::black_box;
use std::time::Instant;

/// Generate synthetic dataset for benchmarking
fn generate_synthetic_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = fastrand::Rng::new();
    rng.seed(42);

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    let samples_per_class = n_samples / n_classes;

    for class in 0..n_classes {
        let start_idx = class * samples_per_class;
        let end_idx = if class == n_classes - 1 {
            n_samples
        } else {
            (class + 1) * samples_per_class
        };

        // Generate class-specific mean
        let class_mean = (class as f64 + 1.0) * 3.0;

        for i in start_idx..end_idx {
            y[i] = class as i32;
            for j in 0..n_features {
                x[[i, j]] = rng.f64() + class_mean + (j as f64 * 0.1);
            }
        }
    }

    (x, y)
}

/// Generate challenging dataset with overlapping classes
fn generate_challenging_dataset(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = fastrand::Rng::new();
    rng.seed(123);

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let class = if i < n_samples / 2 { 0 } else { 1 };
        y[i] = class;

        let base_value = if class == 0 { 1.0 } else { 1.5 };

        for j in 0..n_features {
            x[[i, j]] = rng.f64() * 2.0 + base_value;
        }
    }

    (x, y)
}

/// Benchmark LDA performance across different dataset sizes
fn benchmark_lda_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lda_scaling");

    for n_samples in [100, 500, 1000, 2000].iter() {
        let (x, y) = generate_synthetic_dataset(*n_samples, 10, 2);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), n_samples, |b, _| {
            b.iter(|| {
                let lda = LinearDiscriminantAnalysis::new();
                black_box(lda.fit(&x, &y).unwrap());
            });
        });

        let lda = LinearDiscriminantAnalysis::new();
        let trained_lda = lda.fit(&x, &y).unwrap();

        group.bench_with_input(BenchmarkId::new("predict", n_samples), n_samples, |b, _| {
            b.iter(|| {
                black_box(trained_lda.predict(&x).unwrap());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("predict_proba", n_samples),
            n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(trained_lda.predict_proba(&x).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark QDA performance across different dataset sizes
fn benchmark_qda_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qda_scaling");

    for n_samples in [100, 500, 1000, 2000].iter() {
        let (x, y) = generate_synthetic_dataset(*n_samples, 10, 2);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), n_samples, |b, _| {
            b.iter(|| {
                let qda = QuadraticDiscriminantAnalysis::new();
                black_box(qda.fit(&x, &y).unwrap());
            });
        });

        let qda = QuadraticDiscriminantAnalysis::new();
        let trained_qda = qda.fit(&x, &y).unwrap();

        group.bench_with_input(BenchmarkId::new("predict", n_samples), n_samples, |b, _| {
            b.iter(|| {
                black_box(trained_qda.predict(&x).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark different discriminant analysis methods
fn benchmark_method_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_comparison");
    let (x, y) = generate_synthetic_dataset(1000, 10, 3);

    group.bench_function("lda_fit", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new();
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("qda_fit", |b| {
        b.iter(|| {
            let qda = QuadraticDiscriminantAnalysis::new();
            black_box(qda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("kernel_lda_fit", |b| {
        b.iter(|| {
            let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
            black_box(kda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("mixture_da_fit", |b| {
        b.iter(|| {
            let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(1);
            black_box(mda.fit(&x, &y).unwrap());
        });
    });

    group.finish();
}

/// Benchmark dimensionality impact on performance
fn benchmark_dimensionality_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensionality_impact");

    for n_features in [5, 10, 20, 50].iter() {
        let (x, y) = generate_synthetic_dataset(1000, *n_features, 2);

        group.bench_with_input(
            BenchmarkId::new("lda_fit", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let lda = LinearDiscriminantAnalysis::new();
                    black_box(lda.fit(&x, &y).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("qda_fit", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let qda = QuadraticDiscriminantAnalysis::new();
                    black_box(qda.fit(&x, &y).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark regularization techniques
fn benchmark_regularization(c: &mut Criterion) {
    let mut group = c.benchmark_group("regularization");
    let (x, y) = generate_challenging_dataset(1000, 20);

    group.bench_function("lda_no_regularization", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new();
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("lda_shrinkage", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new().shrinkage(Some(0.1));
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("lda_l1_regularization", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new().l1_reg(0.1);
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("lda_elastic_net", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new()
                .l1_reg(0.05)
                .l2_reg(0.05)
                .elastic_net_ratio(0.5);
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.finish();
}

/// Benchmark transform operation for dimensionality reduction
fn benchmark_transform_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_performance");
    let (x, y) = generate_synthetic_dataset(1000, 20, 3);

    for n_components in [1, 2, 5, 10].iter() {
        let lda = LinearDiscriminantAnalysis::new().n_components(Some(*n_components));
        let trained_lda = lda.fit(&x, &y).unwrap();

        group.bench_with_input(
            BenchmarkId::new("transform", n_components),
            n_components,
            |b, _| {
                b.iter(|| {
                    black_box(trained_lda.transform(&x).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Memory usage benchmark
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for n_samples in [500, 1000, 2000, 4000].iter() {
        let (x, y) = generate_synthetic_dataset(*n_samples, 10, 2);

        group.bench_with_input(
            BenchmarkId::new("lda_memory", n_samples),
            n_samples,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();

                    for _ in 0..iters {
                        let lda = LinearDiscriminantAnalysis::new();
                        let trained = lda.fit(&x, &y).unwrap();
                        let _ = trained.predict(&x).unwrap();
                        let _ = trained.predict_proba(&x).unwrap();
                        black_box(trained);
                    }

                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark robust estimation methods
fn benchmark_robust_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("robust_methods");
    let (x, y) = generate_synthetic_dataset(1000, 10, 2);

    group.bench_function("lda_standard", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new();
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("lda_robust_mcd", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new()
                .robust(true)
                .robust_method("mcd");
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("qda_standard", |b| {
        b.iter(|| {
            let qda = QuadraticDiscriminantAnalysis::new();
            black_box(qda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("qda_robust_mcd", |b| {
        b.iter(|| {
            let qda = QuadraticDiscriminantAnalysis::new()
                .robust(true)
                .robust_method("mcd");
            black_box(qda.fit(&x, &y).unwrap());
        });
    });

    group.finish();
}

/// Benchmark kernel types for kernel discriminant analysis
fn benchmark_kernel_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_types");
    let (x, y) = generate_synthetic_dataset(500, 10, 2);

    group.bench_function("kernel_linear", |b| {
        b.iter(|| {
            let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Linear);
            black_box(kda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("kernel_rbf", |b| {
        b.iter(|| {
            let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
            black_box(kda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("kernel_polynomial", |b| {
        b.iter(|| {
            let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Polynomial {
                gamma: 1.0,
                coef0: 1.0,
                degree: 3,
            });
            black_box(kda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("kernel_sigmoid", |b| {
        b.iter(|| {
            let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Sigmoid {
                gamma: 1.0,
                coef0: 1.0,
            });
            black_box(kda.fit(&x, &y).unwrap());
        });
    });

    group.finish();
}

/// Comprehensive performance benchmark for large-scale datasets
/// This simulates the performance characteristics that would be compared against scikit-learn
fn benchmark_large_scale_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_performance");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Large dataset sizes similar to real-world scenarios
    for n_samples in [10000, 50000, 100000].iter() {
        for n_features in [50, 100, 500].iter() {
            let (x, y) = generate_synthetic_dataset(*n_samples, *n_features, 5);

            group.bench_with_input(
                BenchmarkId::new("lda_large", format!("{}x{}", n_samples, n_features)),
                &(n_samples, n_features),
                |b, _| {
                    b.iter(|| {
                        let lda = LinearDiscriminantAnalysis::new();
                        let trained = lda.fit(&x, &y).unwrap();
                        black_box(
                            trained
                                .predict(&x.slice(s![..1000, ..]).to_owned())
                                .unwrap(),
                        );
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("qda_large", format!("{}x{}", n_samples, n_features)),
                &(n_samples, n_features),
                |b, _| {
                    b.iter(|| {
                        let qda = QuadraticDiscriminantAnalysis::new();
                        let trained = qda.fit(&x, &y).unwrap();
                        black_box(
                            trained
                                .predict(&x.slice(s![..1000, ..]).to_owned())
                                .unwrap(),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark prediction throughput for batch processing
fn benchmark_prediction_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction_throughput");

    let (x_train, y_train) = generate_synthetic_dataset(10000, 20, 3);
    let lda = LinearDiscriminantAnalysis::new();
    let trained_lda = lda.fit(&x_train, &y_train).unwrap();

    let qda = QuadraticDiscriminantAnalysis::new();
    let trained_qda = qda.fit(&x_train, &y_train).unwrap();

    for batch_size in [100, 1000, 10000, 50000].iter() {
        let (x_test, _) = generate_synthetic_dataset(*batch_size, 20, 3);

        group.bench_with_input(
            BenchmarkId::new("lda_predict_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(trained_lda.predict(&x_test).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("qda_predict_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(trained_qda.predict(&x_test).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lda_predict_proba_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(trained_lda.predict_proba(&x_test).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark high-dimensional sparse data scenarios
fn benchmark_high_dimensional_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional_sparse");

    for n_features in [1000, 5000, 10000].iter() {
        let n_samples = 2000;
        let (x, y) = generate_synthetic_dataset(n_samples, *n_features, 2);

        group.bench_with_input(
            BenchmarkId::new("sparse_lda", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let lda = LinearDiscriminantAnalysis::new().l1_reg(0.01);
                    black_box(lda.fit(&x, &y).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("diagonal_lda", n_features),
            n_features,
            |b, _| {
                b.iter(|| {
                    let lda = DiagonalLinearDiscriminantAnalysis::new();
                    black_box(lda.fit(&x, &y).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark numerical stability scenarios
fn benchmark_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");

    // Create challenging datasets with high condition numbers
    let mut rng = fastrand::Rng::new();
    rng.seed(42);

    let n_samples = 1000;
    let n_features = 50;

    // Create ill-conditioned dataset
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        y[i] = (i % 2) as i32;
        for j in 0..n_features {
            // Create highly correlated features
            let base_value = rng.f64();
            x[[i, j]] = base_value + (j as f64 * 1e-8);
        }
    }

    group.bench_function("lda_ill_conditioned", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new();
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("lda_regularized_stability", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new().shrinkage(Some(0.1));
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.bench_function("robust_lda_stability", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new()
                .robust(true)
                .robust_method("mcd");
            black_box(lda.fit(&x, &y).unwrap());
        });
    });

    group.finish();
}

/// Comprehensive accuracy vs performance trade-off analysis
fn benchmark_accuracy_performance_tradeoffs(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_performance_tradeoffs");

    let (x, y) = generate_challenging_dataset(5000, 50);

    // Standard methods
    group.bench_function("lda_standard", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new();
            let trained = lda.fit(&x, &y).unwrap();
            black_box(trained.predict(&x).unwrap());
        });
    });

    // Regularized methods (may be slower but more stable)
    group.bench_function("lda_elastic_net", |b| {
        b.iter(|| {
            let lda = LinearDiscriminantAnalysis::new()
                .l1_reg(0.01)
                .l2_reg(0.01)
                .elastic_net_ratio(0.5);
            let trained = lda.fit(&x, &y).unwrap();
            black_box(trained.predict(&x).unwrap());
        });
    });

    // Kernel methods (more accurate but computationally intensive)
    group.bench_function("kernel_lda_rbf", |b| {
        b.iter(|| {
            let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
            let trained = kda
                .fit(
                    &x.slice(s![..1000, ..]).to_owned(),
                    &y.slice(s![..1000]).to_owned(),
                )
                .unwrap();
            black_box(trained.predict(&x.slice(s![..500, ..]).to_owned()).unwrap());
        });
    });

    // Mixture methods (handling complex distributions)
    group.bench_function("mixture_da", |b| {
        b.iter(|| {
            let mda = MixtureDiscriminantAnalysis::new().n_components_per_class(2);
            let trained = mda.fit(&x, &y).unwrap();
            black_box(trained.predict(&x).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_lda_scaling,
    benchmark_qda_scaling,
    benchmark_method_comparison,
    benchmark_dimensionality_impact,
    benchmark_regularization,
    benchmark_transform_performance,
    benchmark_memory_usage,
    benchmark_robust_methods,
    benchmark_kernel_types,
    benchmark_large_scale_performance,
    benchmark_prediction_throughput,
    benchmark_high_dimensional_sparse,
    benchmark_numerical_stability,
    benchmark_accuracy_performance_tradeoffs
);

criterion_main!(benches);
