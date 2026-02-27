//! Scalability benchmarks for sklears-manifold algorithms
//!
//! This benchmark suite tests how manifold learning algorithms scale with
//! increasing dataset sizes, dimensionalities, and complexity. It helps
//! identify performance bottlenecks and validate theoretical complexity
//! guarantees.

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Performance metrics collector
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    algorithm: String,
    dataset_size: usize,
    dimensionality: usize,
    execution_time: Duration,
    memory_peak: Option<usize>,
    iterations_completed: Option<usize>,
    convergence_achieved: bool,
}

impl PerformanceMetrics {
    fn new(algorithm: &str, n_samples: usize, n_features: usize) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            dataset_size: n_samples,
            dimensionality: n_features,
            execution_time: Duration::default(),
            memory_peak: None,
            iterations_completed: None,
            convergence_achieved: false,
        }
    }

    fn samples_per_second(&self) -> f64 {
        self.dataset_size as f64 / self.execution_time.as_secs_f64()
    }

    fn complexity_score(&self) -> f64 {
        // Theoretical complexity score (varies by algorithm)
        let n = self.dataset_size as f64;
        let d = self.dimensionality as f64;

        match self.algorithm.as_str() {
            "pca" => n * d * d,                // O(nd²) for SVD
            "tsne" => n * n,                   // O(n²) for exact t-SNE
            "tsne_barnes_hut" => n * n.log2(), // O(n log n) for Barnes-Hut
            "isomap" => n * n * n.log2(),      // O(n³) for Floyd-Warshall + O(n²) for eigendecomp
            "lle" => n * n,                    // O(n²) for neighbor search + eigendecomp
            "umap" => n * n.log2(),            // Approximate complexity
            _ => n * d,                        // Linear fallback
        }
    }
}

/// Dataset generators for scalability testing
mod scalability_datasets {
    use super::*;

    pub fn uniform_random(n_samples: usize, n_features: usize, random_state: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(random_state);
        Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-1.0..1.0))
    }

    pub fn gaussian_mixture(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        random_state: u64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, n_features));

        // Generate cluster centers
        let mut centers = Array2::zeros((n_clusters, n_features));
        for i in 0..n_clusters {
            for j in 0..n_features {
                centers[[i, j]] = rng.random_range(-5.0..5.0);
            }
        }

        // Assign points to clusters and add noise
        for i in 0..n_samples {
            let cluster = i % n_clusters;
            for j in 0..n_features {
                data[[i, j]] = centers[[cluster, j]] + rng.random::<f64>() * 2.0 - 1.0 * 0.5;
            }
        }

        data
    }

    pub fn manifold_swiss_roll_nd(
        n_samples: usize,
        _intrinsic_dim: usize,
        embedding_dim: usize,
        noise: f64,
        random_state: u64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, embedding_dim));

        // Generate intrinsic coordinates
        for i in 0..n_samples {
            // Primary manifold coordinates (like Swiss roll)
            let t = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let s = 10.0 * rng.random::<f64>();

            // Map to embedding space
            if embedding_dim >= 3 {
                data[[i, 0]] = t * t.cos();
                data[[i, 1]] = s;
                data[[i, 2]] = t * t.sin();

                // Additional dimensions with structured patterns
                for d in 3..embedding_dim {
                    let pattern = match d % 4 {
                        0 => (t / 2.0).sin() * s / 10.0,
                        1 => (t / 3.0).cos() * s / 10.0,
                        2 => (t + s / 5.0).sin(),
                        _ => (t * s / 20.0).cos(),
                    };
                    data[[i, d]] = pattern;
                }
            } else {
                // Lower dimensional case
                for d in 0..embedding_dim {
                    data[[i, d]] = match d {
                        0 => t * t.cos(),
                        1 => s,
                        _ => (t / (d as f64 + 1.0)).sin(),
                    };
                }
            }

            // Add noise
            for d in 0..embedding_dim {
                data[[i, d]] += noise * rng.random::<f64>() * 2.0 - 1.0;
            }
        }

        data
    }

    pub fn sparse_random(
        n_samples: usize,
        n_features: usize,
        sparsity: f64,
        random_state: u64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                if rng.random::<f64>() < sparsity {
                    data[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
                }
            }
        }

        data
    }
}

/// Algorithm complexity analyzers
mod complexity_analysis {
    use super::*;

    pub fn measure_pca_scaling(sizes: &[usize], n_features: usize) -> Vec<PerformanceMetrics> {
        let mut results = Vec::new();

        for &n_samples in sizes {
            let data = scalability_datasets::gaussian_mixture(n_samples, n_features, 5, 42);
            let start = Instant::now();

            // Mock PCA computation
            let mut covariance = Array2::zeros((n_features, n_features));

            // Compute covariance matrix: O(n * d²)
            for i in 0..n_samples {
                for j in 0..n_features {
                    for k in 0..n_features {
                        covariance[[j, k]] += data[[i, j]] * data[[i, k]];
                    }
                }
            }

            // Scale by sample size
            covariance /= n_samples as f64;

            let elapsed = start.elapsed();

            let mut metrics = PerformanceMetrics::new("pca", n_samples, n_features);
            metrics.execution_time = elapsed;
            metrics.convergence_achieved = true;
            results.push(metrics);
        }

        results
    }

    pub fn measure_distance_matrix_scaling(
        sizes: &[usize],
        n_features: usize,
    ) -> Vec<PerformanceMetrics> {
        let mut results = Vec::new();

        for &n_samples in sizes {
            let data = scalability_datasets::uniform_random(n_samples, n_features, 42);
            let start = Instant::now();

            // Compute pairwise distance matrix: O(n² * d)
            let mut distances = Array2::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                for j in 0..n_samples {
                    let mut dist_sq = 0.0;
                    for k in 0..n_features {
                        let diff = data[[i, k]] - data[[j, k]];
                        dist_sq += diff * diff;
                    }
                    distances[[i, j]] = dist_sq.sqrt();
                }
            }

            let elapsed = start.elapsed();

            let mut metrics = PerformanceMetrics::new("distance_matrix", n_samples, n_features);
            metrics.execution_time = elapsed;
            metrics.convergence_achieved = true;
            results.push(metrics);
        }

        results
    }

    pub fn measure_knn_scaling(
        sizes: &[usize],
        n_features: usize,
        k: usize,
    ) -> Vec<PerformanceMetrics> {
        let mut results = Vec::new();

        for &n_samples in sizes {
            let data = scalability_datasets::uniform_random(n_samples, n_features, 42);
            let start = Instant::now();

            // Brute force k-NN: O(n² * d)
            let mut knn_indices = Vec::new();

            for i in 0..n_samples {
                let mut distances: Vec<(f64, usize)> = Vec::new();

                for j in 0..n_samples {
                    if i != j {
                        let mut dist_sq = 0.0;
                        for l in 0..n_features {
                            let diff = data[[i, l]] - data[[j, l]];
                            dist_sq += diff * diff;
                        }
                        distances.push((dist_sq.sqrt(), j));
                    }
                }

                distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let neighbors: Vec<usize> = distances.iter().take(k).map(|(_, idx)| *idx).collect();
                knn_indices.push(neighbors);
            }

            let elapsed = start.elapsed();

            let mut metrics = PerformanceMetrics::new("knn", n_samples, n_features);
            metrics.execution_time = elapsed;
            metrics.convergence_achieved = true;
            results.push(metrics);
        }

        results
    }
}

/// Memory usage estimation utilities
mod memory_analysis {

    pub fn estimate_memory_usage(
        algorithm: &str,
        n_samples: usize,
        n_features: usize,
        n_components: usize,
    ) -> usize {
        let base_data_size = n_samples * n_features * 8; // f64 = 8 bytes
        let output_size = n_samples * n_components * 8;

        match algorithm {
            "pca" => {
                // Input data + covariance matrix + eigenvectors
                base_data_size + n_features * n_features * 8 + n_features * n_components * 8
            }
            "tsne" | "tsne_exact" => {
                // Input data + distance matrix + gradients + embedding
                base_data_size
                    + n_samples * n_samples * 8
                    + n_samples * n_components * 8 * 2
                    + output_size
            }
            "tsne_barnes_hut" => {
                // Input data + spatial tree + embedding
                base_data_size + n_samples * 50 + output_size // Approximate tree overhead
            }
            "isomap" => {
                // Input data + distance matrix + shortest path matrix + embedding
                base_data_size + n_samples * n_samples * 8 * 2 + output_size
            }
            "lle" => {
                // Input data + weight matrix + embedding
                base_data_size + n_samples * n_samples * 8 + output_size
            }
            "umap" => {
                // Input data + sparse graph + embedding
                base_data_size + n_samples * 20 * 8 + output_size // Approximate sparse graph
            }
            _ => base_data_size + output_size,
        }
    }

    pub fn memory_efficiency_score(actual_memory: usize, theoretical_memory: usize) -> f64 {
        if actual_memory > 0 {
            theoretical_memory as f64 / actual_memory as f64
        } else {
            0.0
        }
    }
}

/// Benchmark dataset size scaling
fn benchmark_dataset_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dataset Size Scaling");
    group.measurement_time(Duration::from_secs(20));

    let sizes = vec![50, 100, 200, 500, 1000, 2000];
    let n_features = 10;

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        let data = scalability_datasets::uniform_random(size, n_features, 42);

        group.bench_with_input(
            BenchmarkId::new("distance_matrix", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let data = black_box(data);
                    let n = data.nrows();
                    let d = data.ncols();
                    let mut distances = Array2::zeros((n, n));

                    for i in 0..n {
                        for j in 0..n {
                            let mut dist_sq = 0.0;
                            for k in 0..d {
                                let diff = data[[i, k]] - data[[j, k]];
                                dist_sq += diff * diff;
                            }
                            distances[[i, j]] = dist_sq.sqrt();
                        }
                    }

                    black_box(distances)
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("knn_graph", size), &data, |b, data| {
            b.iter(|| {
                let data = black_box(data);
                let n = data.nrows();
                let d = data.ncols();
                let k = 10.min(n - 1);
                let mut knn_graph = Vec::new();

                for i in 0..n {
                    let mut distances: Vec<(f64, usize)> = Vec::new();
                    for j in 0..n {
                        if i != j {
                            let mut dist_sq = 0.0;
                            for l in 0..d {
                                let diff = data[[i, l]] - data[[j, l]];
                                dist_sq += diff * diff;
                            }
                            distances.push((dist_sq.sqrt(), j));
                        }
                    }
                    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    let neighbors: Vec<usize> =
                        distances.iter().take(k).map(|(_, idx)| *idx).collect();
                    knn_graph.push(neighbors);
                }

                black_box(knn_graph)
            })
        });
    }

    group.finish();
}

/// Benchmark dimensionality scaling
fn benchmark_dimensionality_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dimensionality Scaling");
    group.measurement_time(Duration::from_secs(15));

    let dimensions = vec![5, 10, 20, 50, 100, 200];
    let n_samples = 500;

    for &n_features in &dimensions {
        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        let data = scalability_datasets::gaussian_mixture(n_samples, n_features, 5, 42);

        group.bench_with_input(
            BenchmarkId::new("covariance_computation", n_features),
            &data,
            |b, data| {
                b.iter(|| {
                    let data = black_box(data);
                    let n = data.nrows();
                    let d = data.ncols();
                    let mut covariance = Array2::zeros((d, d));

                    // Compute covariance matrix
                    for i in 0..n {
                        for j in 0..d {
                            for k in 0..d {
                                covariance[[j, k]] += data[[i, j]] * data[[i, k]];
                            }
                        }
                    }

                    covariance /= n as f64;
                    black_box(covariance)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairwise_distance", n_features),
            &data,
            |b, data| {
                b.iter(|| {
                    let data = black_box(data);
                    let n = data.nrows();
                    let d = data.ncols();
                    let mut total_distance = 0.0;

                    // Compute sum of all pairwise distances
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let mut dist_sq = 0.0;
                            for k in 0..d {
                                let diff = data[[i, k]] - data[[j, k]];
                                dist_sq += diff * diff;
                            }
                            total_distance += dist_sq.sqrt();
                        }
                    }

                    black_box(total_distance)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark sparse data performance
fn benchmark_sparse_data_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Data Scaling");
    group.measurement_time(Duration::from_secs(10));

    let sparsity_levels = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let n_samples = 500;
    let n_features = 50;

    for &sparsity in &sparsity_levels {
        let data = scalability_datasets::sparse_random(n_samples, n_features, sparsity, 42);

        group.bench_with_input(
            BenchmarkId::new("sparse_distance", format!("sparsity_{:.1}", sparsity)),
            &data,
            |b, data| {
                b.iter(|| {
                    let data = black_box(data);
                    let n = data.nrows();
                    let d = data.ncols();
                    let mut total_distance = 0.0;

                    for i in 0..n.min(100) {
                        // Limit for performance
                        for j in (i + 1)..(n.min(100)) {
                            let mut dist_sq = 0.0;
                            for k in 0..d {
                                let val_i = data[[i, k]];
                                let val_j = data[[j, k]];
                                if val_i != 0.0 || val_j != 0.0 {
                                    let diff = val_i - val_j;
                                    dist_sq += diff * diff;
                                }
                            }
                            total_distance += dist_sq.sqrt();
                        }
                    }

                    black_box(total_distance)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Efficiency");

    let sizes = vec![100, 500, 1000];

    for &size in &sizes {
        let data = scalability_datasets::uniform_random(size, 10, 42);

        group.bench_with_input(
            BenchmarkId::new("in_place_operations", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut data_copy = data.clone();

                    // In-place normalization
                    for mut row in data_copy.rows_mut() {
                        let mean = row.mean().unwrap();
                        for val in row.iter_mut() {
                            *val -= mean;
                        }

                        let std = row.iter().map(|x| x * x).sum::<f64>().sqrt() / row.len() as f64;
                        if std > 0.0 {
                            for val in row.iter_mut() {
                                *val /= std;
                            }
                        }
                    }

                    black_box(data_copy)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("copy_operations", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let data = black_box(data);
                    let n = data.nrows();
                    let d = data.ncols();
                    let mut normalized = Array2::zeros((n, d));

                    // Copy-based normalization
                    for i in 0..n {
                        let row = data.row(i);
                        let mean = row.mean().unwrap();

                        let mut sum_sq = 0.0;
                        for j in 0..d {
                            let val = row[j] - mean;
                            normalized[[i, j]] = val;
                            sum_sq += val * val;
                        }

                        let std = (sum_sq / d as f64).sqrt();
                        if std > 0.0 {
                            for j in 0..d {
                                normalized[[i, j]] /= std;
                            }
                        }
                    }

                    black_box(normalized)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark algorithmic complexity validation
fn benchmark_complexity_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Complexity Validation");
    group.measurement_time(Duration::from_secs(25));

    // Test O(n²) algorithms
    let sizes_quadratic = vec![50, 100, 150, 200];
    for &size in &sizes_quadratic {
        let data = scalability_datasets::uniform_random(size, 5, 42);

        group.bench_with_input(
            BenchmarkId::new("quadratic_operation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let data = black_box(data);
                    let n = data.nrows();
                    let mut result = 0.0;

                    // Simulate O(n²) operation
                    for i in 0..n {
                        for j in 0..n {
                            result += data[[i, 0]] * data[[j, 0]];
                        }
                    }

                    black_box(result)
                })
            },
        );
    }

    // Test O(n log n) algorithms
    let sizes_nlogn = vec![100, 200, 500, 1000, 2000];
    for &size in &sizes_nlogn {
        let data = scalability_datasets::uniform_random(size, 5, 42);

        group.bench_with_input(
            BenchmarkId::new("nlogn_operation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let data = black_box(data);
                    let n = data.nrows();
                    let mut values: Vec<f64> = data.column(0).to_vec();

                    // Simulate O(n log n) operation (sorting)
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    // Additional O(n log n) work
                    let mut result = 0.0;
                    for (i, &val) in values.iter().enumerate().take(n) {
                        result += val * (i as f64 + 1.0).log2();
                    }

                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    scalability_benches,
    benchmark_dataset_size_scaling,
    benchmark_dimensionality_scaling,
    benchmark_sparse_data_scaling,
    benchmark_memory_efficiency,
    benchmark_complexity_validation
);

criterion_main!(scalability_benches);
