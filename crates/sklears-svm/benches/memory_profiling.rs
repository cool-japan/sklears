//! Memory Usage Profiling Benchmarks
//!
//! This benchmark suite focuses on memory usage patterns, allocation efficiency,
//! and memory scaling behavior of different SVM implementations.

//!
//! NOTE: This benchmark is currently disabled due to incomplete API implementation.
//! Enable with `--features incomplete-benchmarks` once the required types are implemented.

#![cfg(feature = "incomplete-benchmarks")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use sklears_svm::{
    chunked_processing::ChunkedSVM,
    compressed_kernels::{CompressedKernelMatrix, CompressionMethod},
    kernels::{Kernel, KernelType},
    linear_svc::LinearSVC,
    memory_mapped_kernels::MemoryMappedKernelMatrix,
    out_of_core_svm::OutOfCoreSVM,
    svc::SVC,
};
use std::time::Duration;

/// Memory usage tracker
struct MemoryTracker {
    initial_memory: usize,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            initial_memory: Self::get_memory_usage(),
        }
    }

    fn get_memory_usage() -> usize {
        // Simple approximation - in practice you'd use a proper memory profiler
        // This is a placeholder that would be replaced with actual memory measurement
        0
    }

    fn memory_delta(&self) -> usize {
        Self::get_memory_usage().saturating_sub(self.initial_memory)
    }
}

/// Generate synthetic dataset for memory profiling
fn generate_dataset(n_samples: usize, n_features: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let X_var = Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-5.0, 5.0));
    let y = Array2::from_shape_fn((n_samples, 1), |_| if rng.gen() > 0.5 { 1.0 } else { -1.0 });

    (X, y)
}

/// Benchmark memory usage for different dataset sizes
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let sizes = vec![500, 1000, 2000, 5000];
    let n_features = 20;

    for size in sizes {
        let (X, y) = generate_dataset(size, n_features, 42);
        let y_vec = y.column(0).to_owned();

        group.throughput(Throughput::Elements(size as u64));

        // Standard SVM memory usage
        group.bench_with_input(
            BenchmarkId::new("Standard_SVM", size),
            &(X.clone(), y_vec.clone()),
            |b, (X, y)| {
                b.iter_batched(
                    || MemoryTracker::new(),
                    |_tracker| {
                        let mut svm = SVC::new()
                            .kernel(KernelType::Rbf { gamma: 0.1 })
                            .c(1.0)
                            .build();
                        black_box(svm.fit(X, y).unwrap())
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Linear SVM memory usage (should be more efficient)
        group.bench_with_input(
            BenchmarkId::new("Linear_SVM", size),
            &(X.clone(), y_vec.clone()),
            |b, (X, y)| {
                b.iter_batched(
                    || MemoryTracker::new(),
                    |_tracker| {
                        let mut svm = LinearSVC::new().with_c(1.0).with_max_iter(1000).build();
                        black_box(svm.fit(X, y).unwrap())
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Chunked processing for large datasets
        if size >= 2000 {
            group.bench_with_input(
                BenchmarkId::new("Chunked_SVM", size),
                &(X.clone(), y_vec.clone()),
                |b, (X, y)| {
                    b.iter_batched(
                        || MemoryTracker::new(),
                        |_tracker| {
                            let mut chunked_svm = ChunkedSVM::new()
                                .chunk_size(500)
                                .kernel(KernelType::Rbf { gamma: 0.1 })
                                .c(1.0)
                                .build();
                            black_box(chunked_svm.fit(X, y).unwrap())
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    group.finish();
}

/// Benchmark kernel matrix memory usage
fn bench_kernel_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_memory");

    let sizes = vec![200, 500, 1000];
    let n_features = 50;

    for size in sizes {
        let (X, _) = generate_dataset(size, n_features, 42);

        group.throughput(Throughput::Elements((size * size) as u64));

        // Standard kernel matrix
        group.bench_with_input(
            BenchmarkId::new("Standard_kernel_matrix", size),
            &X,
            |b, X| {
                b.iter_batched(
                    || MemoryTracker::new(),
                    |_tracker| {
                        let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });
                        black_box(kernel.compute_matrix(X, X).unwrap())
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Memory-mapped kernel matrix (for large matrices)
        if size >= 500 {
            group.bench_with_input(
                BenchmarkId::new("Memory_mapped_kernel", size),
                &X,
                |b, X| {
                    b.iter_batched(
                        || {
                            let tracker = MemoryTracker::new();
                            let temp_file = tempfile::NamedTempFile::new().unwrap();
                            let path = temp_file.path().to_path_buf();
                            (tracker, path)
                        },
                        |(_tracker, path)| {
                            let mut mmap_matrix =
                                MemoryMappedKernelMatrix::new(&path, size, size).unwrap();
                            let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });

                            // Compute and store in memory-mapped file
                            for i in 0..size.min(100) {
                                // Limit for benchmark
                                for j in 0..size.min(100) {
                                    let x_i = X.row(i).to_owned();
                                    let x_j = X.row(j).to_owned();
                                    let value = kernel.compute(&x_i, &x_j).unwrap();
                                    black_box(mmap_matrix.set(i, j, value).unwrap());
                                }
                            }
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }

        // Compressed kernel matrix
        group.bench_with_input(BenchmarkId::new("Compressed_kernel", size), &X, |b, X| {
            b.iter_batched(
                || MemoryTracker::new(),
                |_tracker| {
                    let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });
                    let full_matrix = kernel.compute_matrix(X, X).unwrap();

                    let compressed = CompressedKernelMatrix::compress(
                        &full_matrix,
                        CompressionMethod::LowRank { rank: size.min(50) },
                    )
                    .unwrap();

                    black_box(compressed)
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Benchmark out-of-core training memory efficiency
fn bench_out_of_core_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("out_of_core_memory");
    group.sample_size(5);
    group.measurement_time(Duration::from_secs(30));

    let sizes = vec![2000, 5000, 10000];
    let n_features = 20;

    for size in sizes {
        let (X, y) = generate_dataset(size, n_features, 42);
        let y_vec = y.column(0).to_owned();

        group.throughput(Throughput::Elements(size as u64));

        // Out-of-core SVM training
        group.bench_with_input(
            BenchmarkId::new("Out_of_core_SVM", size),
            &(X.clone(), y_vec.clone()),
            |b, (X, y)| {
                b.iter_batched(
                    || {
                        let tracker = MemoryTracker::new();
                        let temp_dir = tempfile::TempDir::new().unwrap();
                        (tracker, temp_dir)
                    },
                    |(_tracker, temp_dir)| {
                        let mut out_of_core_svm = OutOfCoreSVM::new()
                            .cache_dir(temp_dir.path())
                            .chunk_size(1000)
                            .kernel(KernelType::Linear) // Use linear for speed
                            .c(1.0)
                            .build();

                        black_box(out_of_core_svm.fit(X, y).unwrap())
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Compare with in-memory version for smaller sizes
        if size <= 5000 {
            group.bench_with_input(
                BenchmarkId::new("In_memory_SVM", size),
                &(X.clone(), y_vec.clone()),
                |b, (X, y)| {
                    b.iter_batched(
                        || MemoryTracker::new(),
                        |_tracker| {
                            let mut svm = LinearSVC::new()
                                .c(1.0)
                                .with_max_iter(100) // Reduced for large datasets
                                .build();
                            black_box(svm.fit(X, y).unwrap())
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    let size = 1000;
    let n_features = 50;
    let (X, y) = generate_dataset(size, n_features, 42);
    let y_vec = y.column(0).to_owned();

    group.throughput(Throughput::Elements(size as u64));

    // Frequent small allocations (kernel computations)
    group.bench_function("Small_allocations", |b| {
        b.iter_batched(
            || MemoryTracker::new(),
            |_tracker| {
                let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });
                for i in 0..size.min(100) {
                    for j in 0..size.min(100) {
                        let x_i = X.row(i).to_owned();
                        let x_j = X.row(j).to_owned();
                        black_box(kernel.compute(&x_i, &x_j).unwrap());
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Large single allocation (full kernel matrix)
    group.bench_function("Large_allocation", |b| {
        b.iter_batched(
            || MemoryTracker::new(),
            |_tracker| {
                let kernel = Kernel::new(KernelType::Rbf { gamma: 0.1 });
                black_box(kernel.compute_matrix(&X, &X).unwrap())
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Incremental allocation (online learning)
    group.bench_function("Incremental_allocation", |b| {
        b.iter_batched(
            || MemoryTracker::new(),
            |_tracker| {
                let mut svm = SVC::new().kernel(KernelType::Linear).c(1.0).build();

                // Simulate incremental training
                for i in (10..size).step_by(100) {
                    let X_partial_var = X.slice(scirs2_core::ndarray::s![0..i, ..]).to_owned();
                    let y_partial = y_vec.slice(scirs2_core::ndarray::s![0..i]).to_owned();
                    black_box(svm.fit(&X_partial, &y_partial).unwrap());
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark cache-friendly memory access patterns
fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");

    let size = 1000;
    let n_features = 100;
    let (X, _) = generate_dataset(size, n_features, 42);

    group.throughput(Throughput::Elements((size * size) as u64));

    // Row-major kernel computation (cache-friendly)
    group.bench_function("Row_major_access", |b| {
        b.iter(|| {
            let kernel = Kernel::new(KernelType::Linear);
            let mut result = Array2::zeros((size, size));

            for i in 0..size {
                for j in 0..size {
                    let x_i = X.row(i).to_owned();
                    let x_j = X.row(j).to_owned();
                    result[(i, j)] = kernel.compute(&x_i, &x_j).unwrap();
                }
            }
            black_box(result)
        })
    });

    // Column-major kernel computation (cache-unfriendly)
    group.bench_function("Column_major_access", |b| {
        b.iter(|| {
            let kernel = Kernel::new(KernelType::Linear);
            let mut result = Array2::zeros((size, size));

            for j in 0..size {
                for i in 0..size {
                    let x_i = X.row(i).to_owned();
                    let x_j = X.row(j).to_owned();
                    result[(i, j)] = kernel.compute(&x_i, &x_j).unwrap();
                }
            }
            black_box(result)
        })
    });

    // Blocked kernel computation (cache-optimized)
    group.bench_function("Blocked_access", |b| {
        b.iter(|| {
            let kernel = Kernel::new(KernelType::Linear);
            let mut result = Array2::zeros((size, size));
            let block_size = 64;

            for ii in (0..size).step_by(block_size) {
                for jj in (0..size).step_by(block_size) {
                    for i in ii..(ii + block_size).min(size) {
                        for j in jj..(jj + block_size).min(size) {
                            let x_i = X.row(i).to_owned();
                            let x_j = X.row(j).to_owned();
                            result[(i, j)] = kernel.compute(&x_i, &x_j).unwrap();
                        }
                    }
                }
            }
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    memory_benches,
    bench_memory_scaling,
    bench_kernel_memory,
    bench_out_of_core_memory,
    bench_allocation_patterns,
    bench_cache_efficiency
);

criterion_main!(memory_benches);
