//! Kernel Computation Benchmarks
//!
//! This benchmark suite specifically focuses on kernel computation performance,
//! including CPU vs GPU comparisons, SIMD optimizations, and different kernel types.

//!
//! NOTE: This benchmark is currently disabled due to incomplete API implementation.
//! Enable with `--features incomplete-benchmarks` once the required types are implemented.

#![cfg(feature = "incomplete-benchmarks")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use sklears_svm::{
    computer_vision_kernels::{CVKernelFunction, CVKernelType},
    kernels::{Kernel, KernelType},
    simd_kernels::SimdKernel,
    text_classification::TextKernel,
    time_series::TimeSeriesKernel,
};

#[cfg(feature = "gpu")]
use sklears_svm::gpu_kernels::{GpuKernel, GpuKernelBenchmark};

/// Generate random dataset for kernel benchmarks
fn generate_kernel_dataset(n_samples: usize, n_features: usize, seed: u64) -> Array2<f32> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-1.0..1.0))
}

/// Generate histogram data for computer vision kernels
fn generate_histogram_data(n_samples: usize, n_bins: usize, seed: u64) -> Array2<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    Array2::from_shape_fn((n_samples, n_bins), |_| rng.random_range(0.0..100.0))
}

/// Benchmark basic kernel types
fn bench_basic_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_kernels");

    let sizes = vec![100, 500, 1000];
    let n_features = 50;

    for size in sizes {
        let X_var = generate_kernel_dataset(size, n_features, 42);
        let X_f64_var = X.mapv(|x| x as f64);

        group.throughput(Throughput::Elements((size * size) as u64));

        // Linear kernel
        group.bench_with_input(BenchmarkId::new("Linear", size), &X_f64, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Linear);
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // RBF kernel
        group.bench_with_input(BenchmarkId::new("RBF", size), &X_f64, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // Polynomial kernel
        group.bench_with_input(BenchmarkId::new("Polynomial", size), &X_f64, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Polynomial {
                    degree: 3,
                    coef0: 1.0,
                });
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // Sigmoid kernel
        group.bench_with_input(BenchmarkId::new("Sigmoid", size), &X_f64, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Sigmoid {
                    gamma: 0.1,
                    coef0: 1.0,
                });
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark SIMD-optimized kernels
fn bench_simd_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_kernels");

    let sizes = vec![100, 500, 1000];
    let n_features = 64; // Aligned for SIMD

    for size in sizes {
        let X_var = generate_kernel_dataset(size, n_features, 42);

        group.throughput(Throughput::Elements((size * size) as u64));

        // SIMD Linear kernel
        group.bench_with_input(BenchmarkId::new("SIMD_Linear", size), &X, |b, X| {
            b.iter(|| {
                let kernel = SimdKernel::linear();
                black_box(kernel.compute_matrix(X, X))
            })
        });

        // SIMD RBF kernel
        group.bench_with_input(BenchmarkId::new("SIMD_RBF", size), &X, |b, X| {
            b.iter(|| {
                let kernel = SimdKernel::rbf(0.1);
                black_box(kernel.compute_matrix(X, X))
            })
        });

        // Compare with standard implementation
        let X_f64_var = X.mapv(|x| x as f64);
        group.bench_with_input(BenchmarkId::new("Standard_Linear", size), &X_f64, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Linear);
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark computer vision kernels
fn bench_cv_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("cv_kernels");

    let sizes = vec![50, 100, 200];
    let n_bins = 256; // Typical histogram size

    for size in sizes {
        let X_var = generate_histogram_data(size, n_bins, 42);

        group.throughput(Throughput::Elements((size * size) as u64));

        // Histogram intersection kernel
        group.bench_with_input(
            BenchmarkId::new("Histogram_Intersection", size),
            &X,
            |b, X| {
                b.iter(|| {
                    let kernel = CVKernelFunction::new(CVKernelType::HistogramIntersection);
                    black_box(kernel.compute_matrix(X, X).unwrap())
                })
            },
        );

        // Chi-Square kernel
        group.bench_with_input(BenchmarkId::new("Chi_Square", size), &X, |b, X| {
            b.iter(|| {
                let kernel = CVKernelFunction::new(CVKernelType::ChiSquare { gamma: 1.0 });
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // Bhattacharyya kernel
        group.bench_with_input(BenchmarkId::new("Bhattacharyya", size), &X, |b, X| {
            b.iter(|| {
                let kernel = CVKernelFunction::new(CVKernelType::Bhattacharyya);
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // Jensen-Shannon kernel
        group.bench_with_input(BenchmarkId::new("Jensen_Shannon", size), &X, |b, X| {
            b.iter(|| {
                let kernel = CVKernelFunction::new(CVKernelType::JensenShannon);
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark text classification kernels
fn bench_text_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_kernels");

    let sizes = vec![50, 100, 200];
    let vocab_size = 1000;

    for size in sizes {
        // Generate TF-IDF-like vectors
        let X_var = generate_histogram_data(size, vocab_size, 42);

        group.throughput(Throughput::Elements((size * size) as u64));

        // N-gram kernel simulation (using histogram intersection)
        group.bench_with_input(BenchmarkId::new("NGram_Kernel", size), &X, |b, X| {
            b.iter(|| {
                let kernel = CVKernelFunction::new(CVKernelType::HistogramIntersection);
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });

        // Document similarity kernel (using linear kernel on normalized vectors)
        group.bench_with_input(BenchmarkId::new("Document_Similarity", size), &X, |b, X| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Linear);
                black_box(kernel.compute_matrix(X, X).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark GPU vs CPU kernel computation
#[cfg(feature = "gpu")]
fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_vs_cpu");

    let sizes = vec![500, 1000, 2000];
    let n_features = 100;

    for size in sizes {
        let X_var = generate_kernel_dataset(size, n_features, 42);

        group.throughput(Throughput::Elements((size * size) as u64));

        // CPU Linear kernel
        group.bench_with_input(BenchmarkId::new("CPU_Linear", size), &X, |b, X| {
            b.iter(|| {
                let kernel = GpuKernel::new(KernelType::Linear, false);
                black_box(kernel.compute_cpu_kernel_matrix(X, X))
            })
        });

        // CPU RBF kernel
        group.bench_with_input(BenchmarkId::new("CPU_RBF", size), &X, |b, X| {
            b.iter(|| {
                let kernel = GpuKernel::new(KernelType::Rbf { gamma: 0.1 }, false);
                black_box(kernel.compute_cpu_kernel_matrix(X, X))
            })
        });

        // Note: GPU benchmarks would require async runtime setup
        // which is complex in criterion. We include the framework
        // but actual GPU benchmarks are better run separately.
    }

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_vs_cpu(_c: &mut Criterion) {
    // GPU feature not enabled, skip GPU benchmarks
}

/// Benchmark kernel matrix memory access patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    let size = 1000;
    let n_features = 50;
    let X_var = generate_kernel_dataset(size, n_features, 42);
    let X_f64_var = X.mapv(|x| x as f64);

    group.throughput(Throughput::Elements((size * size) as u64));

    // Row-major access pattern
    group.bench_function("Row_major_access", |b| {
        b.iter(|| {
            let kernel = Kernel::new(KernelType::Linear);
            black_box(kernel.compute_matrix(&X_f64, &X_f64).unwrap())
        })
    });

    // Cached kernel computation (simulate repeated access)
    group.bench_function("Cached_computation", |b| {
        b.iter(|| {
            let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });
            // Simulate cache-friendly access by computing smaller blocks
            let block_size = 100;
            for i in (0..size).step_by(block_size) {
                let end_i = (i + block_size).min(size);
                let x_block = X_f64.slice(scirs2_core::ndarray::s![i..end_i, ..]);
                for j in (0..size).step_by(block_size) {
                    let end_j = (j + block_size).min(size);
                    let y_block = X_f64.slice(scirs2_core::ndarray::s![j..end_j, ..]);
                    black_box(
                        kernel
                            .compute_matrix(&x_block.to_owned(), &y_block.to_owned())
                            .unwrap(),
                    );
                }
            }
        })
    });

    group.finish();
}

/// Benchmark different kernel parameterizations
fn bench_kernel_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_parameters");

    let size = 500;
    let n_features = 50;
    let X_var = generate_kernel_dataset(size, n_features, 42);
    let X_f64_var = X.mapv(|x| x as f64);

    group.throughput(Throughput::Elements((size * size) as u64));

    // Different RBF gamma values
    let gammas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
    for gamma in gammas {
        group.bench_with_input(BenchmarkId::new("RBF_gamma", gamma), &gamma, |b, &gamma| {
            b.iter(|| {
                let kernel = Kernel::new(KernelType::Rbf { gamma });
                black_box(kernel.compute_matrix(&X_f64, &X_f64).unwrap())
            })
        });
    }

    // Different polynomial degrees
    let degrees = vec![2.0, 3.0, 4.0, 5.0];
    for degree in degrees {
        group.bench_with_input(
            BenchmarkId::new("Poly_degree", degree as i32),
            &degree,
            |b, &degree| {
                b.iter(|| {
                    let kernel = Kernel::new(KernelType::Polynomial { degree, coef0: 1.0 });
                    black_box(kernel.compute_matrix(&X_f64, &X_f64).unwrap())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    kernel_benches,
    bench_basic_kernels,
    bench_simd_kernels,
    bench_cv_kernels,
    bench_text_kernels,
    bench_gpu_vs_cpu,
    bench_memory_patterns,
    bench_kernel_parameters
);

criterion_main!(kernel_benches);
