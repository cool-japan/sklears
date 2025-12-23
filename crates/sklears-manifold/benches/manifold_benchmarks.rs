//! Comprehensive benchmarks for sklears-manifold algorithms
//!
//! This benchmark suite measures the performance of various manifold learning
//! algorithms implemented in sklears-manifold, providing detailed timing
//! and scalability analysis.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;

// Import the algorithms we want to benchmark
// Note: These would need to be properly exposed from the lib.rs
// For now, we'll create a simplified benchmark structure

/// Generate synthetic high-dimensional data for benchmarking
fn generate_synthetic_data(n_samples: usize, n_features: usize, random_state: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(random_state);
    Array2::from_shape_fn((n_samples, n_features), |_| rng.gen() * 2.0 - 1.0)
}

/// Generate Swiss roll dataset (common manifold learning benchmark)
fn generate_swiss_roll(n_samples: usize, noise: f64, random_state: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(random_state);
    let mut data = Array2::zeros((n_samples, 3));

    for i in 0..n_samples {
        let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * rng.gen());
        let height = 21.0 * rng.gen();

        data[[i, 0]] = t * t.cos() + noise * (rng.gen() * 2.0 - 1.0);
        data[[i, 1]] = height + noise * (rng.gen() * 2.0 - 1.0);
        data[[i, 2]] = t * t.sin() + noise * (rng.gen() * 2.0 - 1.0);
    }

    data
}

/// Generate S-curve dataset
fn generate_s_curve(n_samples: usize, noise: f64, random_state: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(random_state);
    let mut data = Array2::zeros((n_samples, 3));

    for i in 0..n_samples {
        let t = 3.0 * std::f64::consts::PI * rng.gen();
        let height = 2.0 * rng.gen();

        data[[i, 0]] = t.sin() + noise * (rng.gen() * 2.0 - 1.0);
        data[[i, 1]] = height + noise * (rng.gen() * 2.0 - 1.0);
        data[[i, 2]] = (t / 2.0).sin() + noise * (rng.gen() * 2.0 - 1.0);
    }

    data
}

/// Benchmark data generation utilities
fn benchmark_data_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Data Generation");

    for &n_samples in &[100, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("swiss_roll", n_samples),
            &n_samples,
            |b, &size| b.iter(|| black_box(generate_swiss_roll(size, 0.1, 42))),
        );

        group.bench_with_input(
            BenchmarkId::new("s_curve", n_samples),
            &n_samples,
            |b, &size| b.iter(|| black_box(generate_s_curve(size, 0.1, 42))),
        );

        group.bench_with_input(
            BenchmarkId::new("synthetic_high_dim", n_samples),
            &n_samples,
            |b, &size| b.iter(|| black_box(generate_synthetic_data(size, 50, 42))),
        );
    }

    group.finish();
}

/// Benchmark distance computation utilities
fn benchmark_distance_computations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Distance Computations");

    let data_100 = generate_synthetic_data(100, 10, 42);
    let data_500 = generate_synthetic_data(500, 10, 42);
    let data_1000 = generate_synthetic_data(1000, 10, 42);

    // Benchmark pairwise euclidean distance computation
    group.bench_function("euclidean_100x10", |b| {
        b.iter(|| {
            let data = black_box(&data_100);
            // Compute pairwise distances
            let n = data.nrows();
            let mut distances = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let diff = &data.row(i) - &data.row(j);
                    distances[[i, j]] = diff.dot(&diff).sqrt();
                }
            }
            black_box(distances)
        })
    });

    group.bench_function("euclidean_500x10", |b| {
        b.iter(|| {
            let data = black_box(&data_500);
            let n = data.nrows();
            let mut distances = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let diff = &data.row(i) - &data.row(j);
                    distances[[i, j]] = diff.dot(&diff).sqrt();
                }
            }
            black_box(distances)
        })
    });

    group.bench_function("euclidean_1000x10", |b| {
        b.iter(|| {
            let data = black_box(&data_1000);
            let n = data.nrows();
            let mut distances = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let diff = &data.row(i) - &data.row(j);
                    distances[[i, j]] = diff.dot(&diff).sqrt();
                }
            }
            black_box(distances)
        })
    });

    group.finish();
}

/// Benchmark nearest neighbor computations
fn benchmark_nearest_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("Nearest Neighbors");

    for &n_samples in &[100, 500, 1000] {
        for &k in &[5, 10, 20] {
            let data = generate_synthetic_data(n_samples, 10, 42);

            group.bench_with_input(
                BenchmarkId::new("brute_force", format!("{}x10_k{}", n_samples, k)),
                &(&data, k),
                |b, (data, k)| {
                    b.iter(|| {
                        let data = black_box(data);
                        let k = black_box(*k);

                        // Brute force k-nearest neighbors
                        let n = data.nrows();
                        let mut neighbors = Vec::new();

                        for i in 0..n {
                            let mut distances: Vec<(f64, usize)> = Vec::new();
                            for j in 0..n {
                                if i != j {
                                    let diff = &data.row(i) - &data.row(j);
                                    let dist = diff.dot(&diff).sqrt();
                                    distances.push((dist, j));
                                }
                            }
                            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            let knn: Vec<usize> =
                                distances.iter().take(k).map(|(_, idx)| *idx).collect();
                            neighbors.push(knn);
                        }

                        black_box(neighbors)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark embedding quality metrics
fn benchmark_quality_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quality Metrics");

    let original_data = generate_swiss_roll(500, 0.1, 42);
    let embedded_data = generate_synthetic_data(500, 2, 42); // Mock embedded data

    // Benchmark trustworthiness computation
    group.bench_function("trustworthiness_500x3_to_2", |b| {
        b.iter(|| {
            let orig = black_box(&original_data);
            let embed = black_box(&embedded_data);

            // Simplified trustworthiness computation
            let n = orig.nrows();
            let k = 10; // neighborhood size
            let mut trustworthiness = 0.0;

            for i in 0..n {
                // Compute neighbors in original space
                let mut orig_distances: Vec<(f64, usize)> = Vec::new();
                for j in 0..n {
                    if i != j {
                        let diff = &orig.row(i) - &orig.row(j);
                        let dist = diff.dot(&diff).sqrt();
                        orig_distances.push((dist, j));
                    }
                }
                orig_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let orig_neighbors: Vec<usize> =
                    orig_distances.iter().take(k).map(|(_, idx)| *idx).collect();

                // Compute neighbors in embedded space
                let mut embed_distances: Vec<(f64, usize)> = Vec::new();
                for j in 0..n {
                    if i != j {
                        let diff = &embed.row(i) - &embed.row(j);
                        let dist = diff.dot(&diff).sqrt();
                        embed_distances.push((dist, j));
                    }
                }
                embed_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let embed_neighbors: Vec<usize> = embed_distances
                    .iter()
                    .take(k)
                    .map(|(_, idx)| *idx)
                    .collect();

                // Count preserved neighbors
                let preserved = orig_neighbors
                    .iter()
                    .filter(|&&x| embed_neighbors.contains(&x))
                    .count();
                trustworthiness += preserved as f64 / k as f64;
            }

            trustworthiness /= n as f64;
            black_box(trustworthiness)
        })
    });

    group.finish();
}

/// Benchmark memory usage patterns
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Patterns");

    // Benchmark different array creation patterns
    group.bench_function("array_creation_row_major", |b| {
        b.iter(|| {
            let n = black_box(1000);
            let mut data = Array2::zeros((n, 10));
            for i in 0..n {
                for j in 0..10 {
                    data[[i, j]] = (i * 10 + j) as f64;
                }
            }
            black_box(data)
        })
    });

    group.bench_function("array_creation_column_major", |b| {
        b.iter(|| {
            let n = black_box(1000);
            let mut data = Array2::zeros((n, 10));
            for j in 0..10 {
                for i in 0..n {
                    data[[i, j]] = (i * 10 + j) as f64;
                }
            }
            black_box(data)
        })
    });

    group.finish();
}

/// Benchmark random number generation performance
fn benchmark_random_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Generation");

    group.bench_function("standard_normal_1000", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            let samples: Vec<f64> = (0..1000).map(|_| rng.gen() * 2.0 - 1.0).collect();
            black_box(samples)
        })
    });

    group.bench_function("uniform_1000", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            let samples: Vec<f64> = (0..1000).map(|_| rng.gen()).collect();
            black_box(samples)
        })
    });

    group.finish();
}

// Configure benchmark groups
criterion_group!(
    benches,
    benchmark_data_generation,
    benchmark_distance_computations,
    benchmark_nearest_neighbors,
    benchmark_quality_metrics,
    benchmark_memory_patterns,
    benchmark_random_generation
);

criterion_main!(benches);
