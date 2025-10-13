use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::prelude::Array2;
use sklears_simd::activation::{relu, sigmoid, softmax};
use sklears_simd::distance::{cosine_distance, euclidean_distance, manhattan_distance};
use sklears_simd::matrix::{elementwise_add_simd, matrix_multiply_f32_simd, transpose_simd};
use sklears_simd::memory::{bandwidth, cache_aware, AlignedAlloc};
use sklears_simd::sorting::{median_f32_simd, quicksort_f32_simd};
use sklears_simd::vector::{dot_product, mean, norm, scale};

fn generate_random_vector(size: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn bench_distance_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_functions");

    for size in [10, 100, 1000, 10000].iter() {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", size),
            size,
            |bencher, _| {
                bencher.iter(|| euclidean_distance(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("manhattan_distance", size),
            size,
            |bencher, _| {
                bencher.iter(|| manhattan_distance(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cosine_distance", size),
            size,
            |bencher, _| {
                bencher.iter(|| cosine_distance(black_box(&a), black_box(&b)));
            },
        );
    }

    group.finish();
}

fn bench_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    for size in [100, 1000, 10000, 100000].iter() {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(BenchmarkId::new("dot_product", size), size, |bench, _| {
            bench.iter(|| dot_product(black_box(&a), black_box(&b)));
        });

        group.bench_with_input(BenchmarkId::new("norm", size), size, |bench, _| {
            bench.iter(|| norm(black_box(&a)));
        });

        group.bench_with_input(BenchmarkId::new("mean", size), size, |bench, _| {
            bench.iter(|| mean(black_box(&a)));
        });
    }

    group.finish();
}

fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");

    for size in [10, 50, 100, 200].iter() {
        let a = generate_random_matrix(*size, *size);
        let b = generate_random_matrix(*size, *size);

        group.bench_with_input(
            BenchmarkId::new("matrix_multiply", size),
            size,
            |bench, _| {
                bench.iter(|| matrix_multiply_f32_simd(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(BenchmarkId::new("transpose", size), size, |bench, _| {
            bench.iter(|| transpose_simd(black_box(&a)));
        });

        group.bench_with_input(
            BenchmarkId::new("elementwise_add", size),
            size,
            |bench, _| {
                bench.iter(|| elementwise_add_simd(black_box(&a), black_box(&b)));
            },
        );
    }

    group.finish();
}

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    // Test different vector sizes to see SIMD benefits
    for size in [4, 8, 16, 32, 64, 128, 256, 512, 1024].iter() {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        // Scalar implementation for comparison
        let scalar_dot =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

        group.bench_with_input(
            BenchmarkId::new("dot_product_scalar", size),
            size,
            |bench, _| {
                bench.iter(|| scalar_dot(black_box(&a), black_box(&b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dot_product_simd", size),
            size,
            |bench, _| {
                bench.iter(|| dot_product(black_box(&a), black_box(&b)));
            },
        );
    }

    group.finish();
}

fn bench_memory_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_optimizations");

    for size in [1024, 4096, 16384].iter() {
        let src = generate_random_vector(*size);
        let mut dest = vec![0.0f32; *size];

        // Regular copy vs optimized copy
        group.bench_with_input(BenchmarkId::new("regular_copy", size), size, |bench, _| {
            bench.iter(|| dest.copy_from_slice(black_box(&src)));
        });

        group.bench_with_input(
            BenchmarkId::new("optimized_copy", size),
            size,
            |bench, _| {
                bench.iter(|| bandwidth::copy_with_prefetch(black_box(&mut dest), black_box(&src)));
            },
        );

        // Cache-aware matrix transpose
        let rows = (*size as f64).sqrt() as usize;
        let cols = rows;
        if rows * cols <= *size {
            let input = generate_random_vector(rows * cols);
            let mut output = vec![0.0f32; rows * cols];

            group.bench_with_input(
                BenchmarkId::new("cache_aware_transpose", rows),
                &rows,
                |bench, _| {
                    bench.iter(|| {
                        cache_aware::transpose_blocked(
                            black_box(&input),
                            black_box(&mut output),
                            rows,
                            cols,
                            16,
                        )
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");

    for size in [100, 1000, 10000].iter() {
        let input = generate_random_vector(*size);
        let mut output = vec![0.0f32; *size];

        group.bench_with_input(BenchmarkId::new("relu", size), size, |bench, _| {
            bench.iter(|| relu(black_box(&input), black_box(&mut output)));
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |bench, _| {
            bench.iter(|| sigmoid(black_box(&input), black_box(&mut output)));
        });

        // For softmax, use smaller sizes to avoid numerical issues
        if *size <= 1000 {
            group.bench_with_input(BenchmarkId::new("softmax", size), size, |bench, _| {
                bench.iter(|| softmax(black_box(&input), black_box(&mut output)));
            });
        }
    }

    group.finish();
}

fn bench_sorting_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_operations");

    for size in [100, 1000, 10000].iter() {
        let data = generate_random_vector(*size);

        group.bench_with_input(
            BenchmarkId::new("quicksort_simd", size),
            size,
            |bench, _| {
                bench.iter_batched(
                    || data.clone(),
                    |mut data| quicksort_f32_simd(black_box(&mut data)),
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(BenchmarkId::new("median", size), size, |bench, _| {
            bench.iter_batched(
                || data.clone(),
                |mut data| median_f32_simd(black_box(&mut data)),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_clustering_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering_operations");

    // Simple clustering-related benchmarks
    for size in [1000, 5000, 10000].iter() {
        let data = generate_random_vector(*size);

        group.bench_with_input(
            BenchmarkId::new("euclidean_distance_for_clustering", size),
            size,
            |bench, _| {
                let centroid = generate_random_vector(data.len().min(10));
                bench.iter(|| {
                    euclidean_distance(black_box(&data[..centroid.len()]), black_box(&centroid))
                });
            },
        );
    }

    group.finish();
}

fn bench_optimization_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_operations");

    // Simple vector scaling benchmark using the vector module
    for size in [1000, 10000, 100000].iter() {
        let mut x = generate_random_vector(*size);
        let alpha = 0.1f32;

        group.bench_with_input(BenchmarkId::new("vector_scale", size), size, |bench, _| {
            bench.iter(|| scale(black_box(&mut x), black_box(alpha)));
        });
    }

    group.finish();
}

fn bench_aligned_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("aligned_memory");

    for size in [1024, 4096, 16384].iter() {
        group.bench_with_input(
            BenchmarkId::new("aligned_allocation", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let _alloc = AlignedAlloc::<f32>::new(black_box(*size)).unwrap();
                });
            },
        );

        // Compare aligned vs unaligned access patterns
        let aligned_alloc = AlignedAlloc::<f32>::new(*size).unwrap();
        let regular_vec = vec![1.0f32; *size];

        group.bench_with_input(BenchmarkId::new("aligned_sum", size), size, |bench, _| {
            bench.iter(|| black_box(aligned_alloc.as_slice().iter().sum::<f32>()));
        });

        group.bench_with_input(BenchmarkId::new("regular_sum", size), size, |bench, _| {
            bench.iter(|| black_box(regular_vec.iter().sum::<f32>()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_functions,
    bench_vector_operations,
    bench_matrix_operations,
    bench_simd_vs_scalar,
    bench_memory_optimizations,
    bench_activation_functions,
    bench_sorting_operations,
    bench_clustering_operations,
    bench_optimization_operations,
    bench_aligned_memory
);
criterion_main!(benches);
