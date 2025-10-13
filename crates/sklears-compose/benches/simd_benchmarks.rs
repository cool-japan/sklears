//! Comprehensive SIMD Performance Benchmarks for sklears-compose
//!
//! This benchmark suite measures the performance improvements from SIMD optimizations
//! across various pipeline operations, comparing vectorized vs scalar implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_compose::simd_optimizations::{SimdConfig, SimdOps};
use sklears_core::types::Float;
use std::time::Duration;

/// Generate test data for benchmarking
fn generate_test_data(size: usize) -> (Array1<Float>, Array1<Float>) {
    let a = Array1::from_iter((0..size).map(|i| (i as Float) * 0.1));
    let b = Array1::from_iter((0..size).map(|i| (i as Float) * 0.2 + 1.0));
    (a, b)
}

/// Generate 2D test data for matrix operations
fn generate_matrix_data(rows: usize, cols: usize) -> Array2<Float> {
    Array2::from_shape_fn((rows, cols), |(i, j)| (i + j) as Float * 0.1)
}

/// Benchmark SIMD vector addition operations
fn bench_vector_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vector_addition");

    let sizes = [64, 256, 1024, 4096, 16384];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let (a, b) = generate_test_data(*size);
        let simd_ops = SimdOps::default();

        // Benchmark SIMD-optimized addition
        group.bench_with_input(
            BenchmarkId::new("simd_optimized", size),
            size,
            |bench, _| {
                bench.iter(|| black_box(simd_ops.add_arrays(&a.view(), &b.view()).unwrap()));
            },
        );

        // Benchmark scalar addition for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar_baseline", size),
            size,
            |bench, _| {
                bench.iter(|| black_box(&a + &b));
            },
        );
    }

    group.finish();
}

/// Benchmark matrix multiplication with SIMD optimizations
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_matrix_multiplication");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(64, 64), (128, 128), (256, 256), (512, 512)];

    for (rows, cols) in sizes.iter() {
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let matrix_a = generate_matrix_data(*rows, *cols);
        let matrix_b = generate_matrix_data(*cols, *rows);
        let simd_ops = SimdOps::default();

        // Benchmark SIMD-optimized matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("simd_matrix_mul", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| {
                    black_box(
                        simd_ops
                            .matrix_multiply(&matrix_a.view(), &matrix_b.view())
                            .unwrap(),
                    )
                });
            },
        );

        // Benchmark standard ndarray dot product for comparison
        group.bench_with_input(
            BenchmarkId::new("ndarray_baseline", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| black_box(matrix_a.dot(&matrix_b)));
            },
        );
    }

    group.finish();
}

/// Benchmark feature standardization with SIMD
fn bench_feature_standardization(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_standardization");

    let sizes = [100, 500, 1000, 5000, 10000];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data = generate_matrix_data(*size, 10);
        let simd_ops = SimdOps::default();

        // Benchmark SIMD-optimized standardization
        group.bench_with_input(
            BenchmarkId::new("simd_standardize", size),
            size,
            |bench, _| {
                bench.iter(|| black_box(simd_ops.standardize_features(&data.view()).unwrap()));
            },
        );

        // Benchmark manual standardization
        group.bench_with_input(
            BenchmarkId::new("manual_standardize", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let mean = data.mean_axis(Axis(0)).unwrap();
                    let std = data.std_axis(Axis(0), 1.0);
                    black_box((&data - &mean) / &std)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark min-max scaling with SIMD
fn bench_min_max_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_min_max_scaling");

    let sizes = [100, 500, 1000, 5000, 10000];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data = generate_matrix_data(*size, 8);
        let simd_ops = SimdOps::default();

        // Benchmark SIMD-optimized min-max scaling
        group.bench_with_input(BenchmarkId::new("simd_min_max", size), size, |bench, _| {
            bench.iter(|| black_box(simd_ops.min_max_scale(&data.view()).unwrap()));
        });

        // Benchmark manual min-max scaling
        group.bench_with_input(
            BenchmarkId::new("manual_min_max", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let min_vals = data.fold_axis(Axis(0), Float::INFINITY, |&a, &b| a.min(b));
                    let max_vals = data.fold_axis(Axis(0), Float::NEG_INFINITY, |&a, &b| a.max(b));
                    let range = &max_vals - &min_vals;
                    black_box((&data - &min_vals) / &range)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark polynomial feature generation with SIMD
fn bench_polynomial_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_polynomial_features");
    group.measurement_time(Duration::from_secs(8));

    let sizes = [(100, 5), (500, 5), (1000, 5), (2000, 3)];

    for (samples, features) in sizes.iter() {
        group.throughput(Throughput::Elements((samples * features) as u64));

        let data = generate_matrix_data(*samples, *features);
        let simd_ops = SimdOps::default();

        // Benchmark SIMD-optimized polynomial features (degree 2)
        group.bench_with_input(
            BenchmarkId::new("simd_poly_degree2", format!("{}x{}", samples, features)),
            &(samples, features),
            |bench, _| {
                bench.iter(|| black_box(simd_ops.polynomial_features(&data.view(), 2).unwrap()));
            },
        );

        // Benchmark manual polynomial feature generation
        group.bench_with_input(
            BenchmarkId::new("manual_poly_degree2", format!("{}x{}", samples, features)),
            &(samples, features),
            |bench, _| {
                bench.iter(|| {
                    let n_samples = data.nrows();
                    let n_features = data.ncols();
                    let n_output_features = n_features + (n_features * (n_features + 1)) / 2;
                    let mut result = Array2::zeros((n_samples, n_output_features));

                    for i in 0..n_samples {
                        let mut col_idx = 0;

                        // Linear terms
                        for j in 0..n_features {
                            result[[i, col_idx]] = data[[i, j]];
                            col_idx += 1;
                        }

                        // Quadratic terms
                        for j in 0..n_features {
                            for k in j..n_features {
                                result[[i, col_idx]] = data[[i, j]] * data[[i, k]];
                                col_idx += 1;
                            }
                        }
                    }

                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory-aligned operations
fn bench_memory_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_memory_alignment");

    let sizes = [512, 1024, 2048, 4096];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data = Array1::from_iter((0..*size).map(|i| i as Float));
        let simd_ops = SimdOps::default();

        // Benchmark aligned operations
        group.bench_with_input(BenchmarkId::new("aligned_sum", size), size, |bench, _| {
            bench.iter(|| black_box(simd_ops.vectorized_sum(&data.view()).unwrap()));
        });

        // Benchmark standard sum
        group.bench_with_input(BenchmarkId::new("standard_sum", size), size, |bench, _| {
            bench.iter(|| black_box(data.sum()));
        });
    }

    group.finish();
}

/// Benchmark different SIMD configurations
fn bench_simd_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_configurations");

    let size = 4096;
    let (a, b) = generate_test_data(size);

    // Test different vector widths
    let configs = vec![
        (
            "avx512",
            SimdConfig {
                use_avx512: true,
                vector_width: 16,
                ..Default::default()
            },
        ),
        (
            "avx2",
            SimdConfig {
                use_avx2: true,
                vector_width: 8,
                ..Default::default()
            },
        ),
        (
            "sse",
            SimdConfig {
                vector_width: 4,
                ..Default::default()
            },
        ),
        (
            "scalar",
            SimdConfig {
                vector_width: 1,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in configs {
        let simd_ops = SimdOps::new(config);

        group.bench_function(&format!("config_{}", name), |bench| {
            bench.iter(|| black_box(simd_ops.add_arrays(&a.view(), &b.view()).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark cache-friendly operations
fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_cache_performance");

    let sizes = [
        ("L1_cache", 16 * 1024 / std::mem::size_of::<Float>()), // ~L1 cache size
        ("L2_cache", 256 * 1024 / std::mem::size_of::<Float>()), // ~L2 cache size
        ("L3_cache", 8 * 1024 * 1024 / std::mem::size_of::<Float>()), // ~L3 cache size
        (
            "beyond_cache",
            64 * 1024 * 1024 / std::mem::size_of::<Float>(),
        ), // Beyond cache
    ];

    for (name, size) in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let data = Array1::from_iter((0..size).map(|i| i as Float));
        let simd_ops = SimdOps::default();

        group.bench_function(&format!("cache_test_{}", name), |bench| {
            bench.iter(|| black_box(simd_ops.vectorized_sum(&data.view()).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    simd_benches,
    bench_vector_addition,
    bench_matrix_multiplication,
    bench_feature_standardization,
    bench_min_max_scaling,
    bench_polynomial_features,
    bench_memory_alignment,
    bench_simd_configurations,
    bench_cache_performance
);

criterion_main!(simd_benches);
