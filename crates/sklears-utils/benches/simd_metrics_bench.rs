#![cfg(feature = "incomplete-benchmarks")]

//! Benchmarks for SIMD-optimized distance metrics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::essentials::Uniform;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{seeded_rng, thread_rng};
use sklears_utils::metrics::{
    cosine_distance, cosine_distance_f32, euclidean_distance, euclidean_distance_f32,
    manhattan_distance, manhattan_distance_f32,
};

fn generate_test_data_f32(size: usize) -> (Array1<f32>, Array1<f32>) {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0f32, 1.0f32);
    let a: Array1<f32> = Array1::from_vec((0..size).map(|_| uniform.sample(&mut rng)).collect());
    let b: Array1<f32> = Array1::from_vec((0..size).map(|_| uniform.sample(&mut rng)).collect());
    (a, b)
}

fn generate_test_data_f64(size: usize) -> (Array1<f64>, Array1<f64>) {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0f64, 1.0f64);
    let a: Array1<f64> = Array1::from_vec((0..size).map(|_| uniform.sample(&mut rng)).collect());
    let b: Array1<f64> = Array1::from_vec((0..size).map(|_| uniform.sample(&mut rng)).collect());
    (a, b)
}

fn benchmark_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");

    for size in [16, 64, 256, 1024, 4096].iter() {
        // Test f32 (SIMD-optimized) case
        let (a_f32, b_f32) = generate_test_data_f32(*size);
        group.bench_with_input(BenchmarkId::new("f32_simd", size), size, |bench, _| {
            bench.iter(|| euclidean_distance_f32(black_box(&a_f32), black_box(&b_f32)));
        });

        // Test f64 (scalar fallback) case for comparison
        let (a_f64, b_f64) = generate_test_data_f64(*size);
        group.bench_with_input(BenchmarkId::new("f64_scalar", size), size, |bench, _| {
            bench.iter(|| euclidean_distance(black_box(&a_f64), black_box(&b_f64)));
        });
    }

    group.finish();
}

fn benchmark_manhattan(c: &mut Criterion) {
    let mut group = c.benchmark_group("manhattan_distance");

    for size in [16, 64, 256, 1024, 4096].iter() {
        // Test f32 (SIMD-optimized) case
        let (a_f32, b_f32) = generate_test_data_f32(*size);
        group.bench_with_input(BenchmarkId::new("f32_simd", size), size, |bench, _| {
            bench.iter(|| manhattan_distance_f32(black_box(&a_f32), black_box(&b_f32)));
        });

        // Test f64 (scalar fallback) case for comparison
        let (a_f64, b_f64) = generate_test_data_f64(*size);
        group.bench_with_input(BenchmarkId::new("f64_scalar", size), size, |bench, _| {
            bench.iter(|| manhattan_distance(black_box(&a_f64), black_box(&b_f64)));
        });
    }

    group.finish();
}

fn benchmark_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for size in [16, 64, 256, 1024, 4096].iter() {
        // Test f32 (SIMD-optimized) case
        let (a_f32, b_f32) = generate_test_data_f32(*size);
        group.bench_with_input(BenchmarkId::new("f32_simd", size), size, |bench, _| {
            bench.iter(|| cosine_distance_f32(black_box(&a_f32), black_box(&b_f32)));
        });

        // Test f64 (scalar fallback) case for comparison
        let (a_f64, b_f64) = generate_test_data_f64(*size);
        group.bench_with_input(BenchmarkId::new("f64_scalar", size), size, |bench, _| {
            bench.iter(|| cosine_distance(black_box(&a_f64), black_box(&b_f64)));
        });
    }

    group.finish();
}

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distance_operations");

    // Test with multiple query points
    let query_sizes = [100, 500, 1000];
    let feature_dims = [128, 512];

    for &query_size in &query_sizes {
        for &dim in &feature_dims {
            // Generate f32 dataset (SIMD-optimized)
            let dataset_f32: Vec<Array1<f32>> = (0..query_size)
                .map(|_| Array::random(dim, Uniform::new(0.0f32, 1.0f32)))
                .collect();
            let query_f32: Array1<f32> = Array::random(dim, Uniform::new(0.0f32, 1.0f32));

            group.bench_with_input(
                BenchmarkId::new("euclidean_batch_f32", format!("{}x{}", query_size, dim)),
                &(query_size, dim),
                |bench, _| {
                    bench.iter(|| {
                        for point in &dataset_f32 {
                            black_box(euclidean_distance_f32(point, &query_f32));
                        }
                    });
                },
            );

            // Generate f64 dataset (scalar fallback) for comparison
            let dataset_f64: Vec<Array1<f64>> = (0..query_size)
                .map(|_| Array::random(dim, Uniform::new(0.0f64, 1.0f64)))
                .collect();
            let query_f64: Array1<f64> = Array::random(dim, Uniform::new(0.0f64, 1.0f64));

            group.bench_with_input(
                BenchmarkId::new("euclidean_batch_f64", format!("{}x{}", query_size, dim)),
                &(query_size, dim),
                |bench, _| {
                    bench.iter(|| {
                        for point in &dataset_f64 {
                            black_box(euclidean_distance(point, &query_f64));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_euclidean,
    benchmark_manhattan,
    benchmark_cosine,
    benchmark_batch_operations
);
criterion_main!(benches);
