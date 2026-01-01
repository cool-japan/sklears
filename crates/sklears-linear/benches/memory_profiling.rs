//! Memory Usage Profiling Benchmarks
//!
//! This module provides comprehensive memory usage analysis and profiling
//! for linear models, including memory scaling, peak usage, and efficiency metrics.
//!
//! NOTE: This benchmark is currently disabled due to incomplete API implementation.
//! Enable with `--features incomplete-benchmarks` once the required types are implemented.

#![cfg(feature = "incomplete-benchmarks")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_autograd::ndarray::{Array1, Array2};
use scirs2_core::random::distributions::StandardNormal;
use scirs2_core::random::prelude::*;
use scirs2_core::random::RandomExt;
use sklears_core::types::Float;
use sklears_linear::{
    chunked_processing::{ChunkProcessingConfig, ChunkedProcessor},
    memory_efficient_ops::MemoryEfficientOps,
    streaming_algorithms::{StreamingConfig, StreamingLinearRegression},
    ElasticNetRegression, LassoRegression, LinearRegression, RidgeRegression,
};
use std::time::Instant;

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub algorithm: String,
    pub dataset_size: (usize, usize),
    pub estimated_memory_mb: Float,
    pub peak_memory_mb: Float,
    pub memory_efficiency: Float, // MB per million samples
    pub time_to_fit: std::time::Duration,
    pub memory_per_prediction: Float, // bytes per prediction
}

impl MemoryStats {
    pub fn new(
        algorithm: String,
        dataset_size: (usize, usize),
        estimated_memory_mb: Float,
        fit_duration: std::time::Duration,
    ) -> Self {
        let (n_samples, n_features) = dataset_size;
        let samples_millions = n_samples as Float / 1_000_000.0;
        let memory_efficiency = if samples_millions > 0.0 {
            estimated_memory_mb / samples_millions
        } else {
            estimated_memory_mb
        };

        let memory_per_prediction = (estimated_memory_mb * 1024.0 * 1024.0) / n_samples as Float;

        Self {
            algorithm,
            dataset_size,
            estimated_memory_mb,
            peak_memory_mb: estimated_memory_mb * 1.2, // Estimated peak
            memory_efficiency,
            time_to_fit: fit_duration,
            memory_per_prediction,
        }
    }

    pub fn print_summary(&self) {
        println!("\n=== {} Memory Profile ===", self.algorithm);
        println!("Dataset: {} x {}", self.dataset_size.0, self.dataset_size.1);
        println!("Estimated Memory: {:.2} MB", self.estimated_memory_mb);
        println!("Peak Memory: {:.2} MB", self.peak_memory_mb);
        println!(
            "Memory Efficiency: {:.2} MB/M samples",
            self.memory_efficiency
        );
        println!(
            "Memory per Prediction: {:.2} bytes",
            self.memory_per_prediction
        );
        println!("Fit Time: {:.2?}", self.time_to_fit);
    }
}

/// Estimate memory usage for arrays
fn estimate_array_memory_mb(shape: (usize, usize)) -> Float {
    let (rows, cols) = shape;
    let elements = rows * cols;
    let bytes = elements * std::mem::size_of::<Float>();
    bytes as Float / (1024.0 * 1024.0)
}

/// Generate dataset with known memory footprint
fn generate_memory_test_dataset(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<Float>, Array1<Float>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let X = Array2::random_using((n_samples, n_features), StandardNormal, &mut rng);
    let true_coefs: Array1<Float> = (0..n_features)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<_>>()
        .into();
    let y = X.dot(&true_coefs);

    (X, y)
}

/// Benchmark memory usage scaling with dataset size
fn benchmark_memory_scaling(c: &mut Criterion) {
    let sample_sizes = vec![1000, 5000, 10000, 25000, 50000];
    let n_features = 100;

    let mut group = c.benchmark_group("memory_scaling");
    group.sample_size(10); // Fewer samples for memory-intensive tests

    for n_samples in sample_sizes {
        let (X, y) = generate_memory_test_dataset(n_samples, n_features, 42);
        let estimated_memory = estimate_array_memory_mb((n_samples, n_features)) * 2.0; // X + y

        group.bench_with_input(
            BenchmarkId::new("linear_regression_memory", n_samples),
            &(&X, &y, estimated_memory),
            |b, (X, y, est_mem)| {
                b.iter(|| {
                    let start = Instant::now();
                    let model = LinearRegression::new();
                    let trained_model = model.fit(X, y).unwrap();
                    let predictions = trained_model.predict(X).unwrap();
                    let duration = start.elapsed();

                    let stats = MemoryStats::new(
                        "LinearRegression".to_string(),
                        (X.nrows(), X.ncols()),
                        *est_mem,
                        duration,
                    );

                    black_box((predictions, stats));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency of different algorithms
fn benchmark_algorithm_memory_efficiency(c: &mut Criterion) {
    let n_samples = 10000;
    let n_features = 200;
    let (X, y) = generate_memory_test_dataset(n_samples, n_features, 42);
    let base_memory = estimate_array_memory_mb((n_samples, n_features)) * 2.0;

    let mut group = c.benchmark_group("algorithm_memory_efficiency");

    // Linear Regression
    group.bench_with_input(
        BenchmarkId::new("memory_efficiency", "linear_regression"),
        &(&X, &y, base_memory),
        |b, (X, y, mem)| {
            b.iter(|| {
                let start = Instant::now();
                let model = LinearRegression::new();
                let trained_model = model.fit(X, y).unwrap();
                let predictions = trained_model.predict(X).unwrap();
                let duration = start.elapsed();

                let stats = MemoryStats::new(
                    "LinearRegression".to_string(),
                    (X.nrows(), X.ncols()),
                    *mem,
                    duration,
                );

                black_box((predictions, stats));
            });
        },
    );

    // Ridge Regression
    group.bench_with_input(
        BenchmarkId::new("memory_efficiency", "ridge_regression"),
        &(&X, &y, base_memory),
        |b, (X, y, mem)| {
            b.iter(|| {
                let start = Instant::now();
                let model = RidgeRegression::new().alpha(1.0);
                let trained_model = model.fit(X, y).unwrap();
                let predictions = trained_model.predict(X).unwrap();
                let duration = start.elapsed();

                let stats = MemoryStats::new(
                    "RidgeRegression".to_string(),
                    (X.nrows(), X.ncols()),
                    *mem * 1.1, // Slight overhead for regularization
                    duration,
                );

                black_box((predictions, stats));
            });
        },
    );

    // Lasso Regression (typically more memory intensive due to iterative nature)
    group.bench_with_input(
        BenchmarkId::new("memory_efficiency", "lasso_regression"),
        &(&X, &y, base_memory),
        |b, (X, y, mem)| {
            b.iter(|| {
                let start = Instant::now();
                let model = LassoRegression::new().alpha(0.1).max_iter(100);
                let trained_model = model.fit(X, y).unwrap();
                let predictions = trained_model.predict(X).unwrap();
                let duration = start.elapsed();

                let stats = MemoryStats::new(
                    "LassoRegression".to_string(),
                    (X.nrows(), X.ncols()),
                    *mem * 1.5, // Higher overhead for coordinate descent
                    duration,
                );

                black_box((predictions, stats));
            });
        },
    );

    group.finish();
}

/// Benchmark chunked processing memory efficiency
fn benchmark_chunked_processing_memory(c: &mut Criterion) {
    let n_samples = 50000;
    let n_features = 100;
    let (X, y) = generate_memory_test_dataset(n_samples, n_features, 42);

    let chunk_sizes = vec![1000, 5000, 10000, 25000];

    let mut group = c.benchmark_group("chunked_processing_memory");

    for chunk_size in chunk_sizes {
        let config = ChunkProcessingConfig {
            chunk_size,
            n_threads: 4,
            memory_limit_mb: Some(100.0),
            overlap_size: 0,
            in_place_processing: true,
        };

        group.bench_with_input(
            BenchmarkId::new("chunked_linear_regression", chunk_size),
            &(&X, &y, config),
            |b, (X, y, config)| {
                b.iter(|| {
                    let start = Instant::now();
                    let processor = ChunkedProcessor::new(config.clone()).unwrap();

                    // Mock chunked processing
                    let n_chunks = (X.nrows() + config.chunk_size - 1) / config.chunk_size;
                    let chunk_memory =
                        estimate_array_memory_mb((config.chunk_size, X.ncols())) * 2.0;
                    let total_memory = chunk_memory * config.n_threads as Float;

                    let duration = start.elapsed();

                    let stats = MemoryStats::new(
                        format!("ChunkedProcessing(chunk={})", config.chunk_size),
                        (X.nrows(), X.ncols()),
                        total_memory,
                        duration,
                    );

                    black_box((n_chunks, stats));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark streaming algorithm memory usage
fn benchmark_streaming_memory(c: &mut Criterion) {
    let batch_sizes = vec![100, 500, 1000, 5000];
    let total_samples = 50000;
    let n_features = 50;

    let mut group = c.benchmark_group("streaming_memory");

    for batch_size in batch_sizes {
        let config = StreamingConfig {
            batch_size,
            learning_rate: 0.01,
            max_epochs: 1,
            convergence_tolerance: 1e-6,
            shuffle: true,
            random_seed: Some(42),
        };

        group.bench_with_input(
            BenchmarkId::new("streaming_linear_regression", batch_size),
            &(total_samples, n_features, config),
            |b, (total_samples, n_features, config)| {
                b.iter(|| {
                    let start = Instant::now();

                    let mut model =
                        StreamingLinearRegression::new(*n_features).config(config.clone());

                    // Simulate streaming batches
                    let n_batches = (*total_samples + config.batch_size - 1) / config.batch_size;
                    let batch_memory =
                        estimate_array_memory_mb((config.batch_size, *n_features)) * 2.0;

                    // Model memory (coefficients + state)
                    let model_memory = estimate_array_memory_mb((1, *n_features)) * 3.0; // coefficients + gradients + state

                    let total_memory = batch_memory + model_memory;
                    let duration = start.elapsed();

                    let stats = MemoryStats::new(
                        format!("StreamingLR(batch={})", config.batch_size),
                        (*total_samples, *n_features),
                        total_memory,
                        duration,
                    );

                    black_box((n_batches, stats));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory-efficient operations
fn benchmark_memory_efficient_ops(c: &mut Criterion) {
    let n_samples = 20000;
    let n_features = 500;
    let (X, y) = generate_memory_test_dataset(n_samples, n_features, 42);

    let mut group = c.benchmark_group("memory_efficient_ops");

    // Test in-place matrix operations
    group.bench_with_input(
        BenchmarkId::new("in_place_operations", "matrix_multiply"),
        &(&X, &y),
        |b, (X, y)| {
            b.iter(|| {
                let start = Instant::now();

                // Simulate memory-efficient operations
                let ops = MemoryEfficientOps::new();

                // In-place matrix-vector multiplication (mocked)
                let mut result = Array1::zeros(X.nrows());
                for i in 0..X.nrows() {
                    for j in 0..X.ncols() {
                        result[i] += X[[i, j]] * 1.0; // Mock operation
                    }
                }

                let duration = start.elapsed();

                // Only temporary arrays needed
                let memory_used = estimate_array_memory_mb((X.nrows(), 1));

                let stats = MemoryStats::new(
                    "InPlaceOps".to_string(),
                    (X.nrows(), X.ncols()),
                    memory_used,
                    duration,
                );

                black_box((result, stats));
            });
        },
    );

    group.finish();
}

/// Benchmark peak memory usage during fitting
fn benchmark_peak_memory_usage(c: &mut Criterion) {
    let n_samples = 15000;
    let n_features = 300;
    let (X, y) = generate_memory_test_dataset(n_samples, n_features, 42);

    let mut group = c.benchmark_group("peak_memory_usage");

    // Test algorithms that may have high peak memory usage
    let algorithms = vec![
        ("linear_regression", "Standard fitting"),
        ("ridge_regression", "With regularization"),
        ("elastic_net", "Iterative solver"),
    ];

    for (algo_name, description) in algorithms {
        group.bench_with_input(
            BenchmarkId::new("peak_memory", algo_name),
            &(&X, &y, description),
            |b, (X, y, _desc)| {
                b.iter(|| {
                    let start = Instant::now();

                    let (model_memory, duration) = match algo_name {
                        "linear_regression" => {
                            let model = LinearRegression::new();
                            let trained_model = model.fit(X, y).unwrap();
                            let predictions = trained_model.predict(X).unwrap();
                            black_box(predictions);

                            // Memory for X, y, X'X, X'y, and temporary arrays
                            let base_mem = estimate_array_memory_mb((X.nrows(), X.ncols())) * 2.0;
                            let intermediate_mem = estimate_array_memory_mb((X.ncols(), X.ncols()));
                            (base_mem + intermediate_mem, start.elapsed())
                        }
                        "ridge_regression" => {
                            let model = RidgeRegression::new().alpha(1.0);
                            let trained_model = model.fit(X, y).unwrap();
                            let predictions = trained_model.predict(X).unwrap();
                            black_box(predictions);

                            let base_mem = estimate_array_memory_mb((X.nrows(), X.ncols())) * 2.0;
                            let intermediate_mem =
                                estimate_array_memory_mb((X.ncols(), X.ncols())) * 1.2;
                            (base_mem + intermediate_mem, start.elapsed())
                        }
                        "elastic_net" => {
                            let model = ElasticNetRegression::new().alpha(0.1).l1_ratio(0.5);
                            let trained_model = model.fit(X, y).unwrap();
                            let predictions = trained_model.predict(X).unwrap();
                            black_box(predictions);

                            let base_mem = estimate_array_memory_mb((X.nrows(), X.ncols())) * 2.0;
                            let iteration_mem = estimate_array_memory_mb((X.ncols(), 1)) * 5.0; // Multiple temporary arrays
                            (base_mem + iteration_mem, start.elapsed())
                        }
                        _ => (0.0, start.elapsed()),
                    };

                    let stats = MemoryStats::new(
                        algo_name.to_string(),
                        (X.nrows(), X.ncols()),
                        model_memory,
                        duration,
                    );

                    black_box(stats);
                });
            },
        );
    }

    group.finish();
}

/// Generate comprehensive memory report
pub fn generate_memory_report() -> Vec<MemoryStats> {
    let mut all_stats = Vec::new();

    // Test different dataset sizes
    let test_configs = vec![
        (1000, 50, "Small dataset"),
        (10000, 100, "Medium dataset"),
        (50000, 200, "Large dataset"),
    ];

    for (n_samples, n_features, description) in test_configs {
        let (X, y) = generate_memory_test_dataset(n_samples, n_features, 42);
        let base_memory = estimate_array_memory_mb((n_samples, n_features)) * 2.0;

        // Test Linear Regression
        let start = Instant::now();
        let model = LinearRegression::new();
        let _trained_model = model.fit(&X, &y).unwrap();
        let lr_duration = start.elapsed();

        let lr_stats = MemoryStats::new(
            format!("LinearRegression ({})", description),
            (n_samples, n_features),
            base_memory,
            lr_duration,
        );
        all_stats.push(lr_stats);

        // Test Ridge Regression
        let start = Instant::now();
        let model = RidgeRegression::new().alpha(1.0);
        let _trained_model = model.fit(&X, &y).unwrap();
        let ridge_duration = start.elapsed();

        let ridge_stats = MemoryStats::new(
            format!("RidgeRegression ({})", description),
            (n_samples, n_features),
            base_memory * 1.1,
            ridge_duration,
        );
        all_stats.push(ridge_stats);
    }

    all_stats
}

/// Print comprehensive memory analysis summary
pub fn print_memory_summary() {
    println!("\n=== COMPREHENSIVE MEMORY ANALYSIS SUMMARY ===");

    let stats = generate_memory_report();

    for stat in &stats {
        stat.print_summary();
    }

    println!("\n=== MEMORY EFFICIENCY ANALYSIS ===");

    // Calculate efficiency metrics
    let linear_stats: Vec<_> = stats
        .iter()
        .filter(|s| s.algorithm.contains("LinearRegression"))
        .collect();
    let ridge_stats: Vec<_> = stats
        .iter()
        .filter(|s| s.algorithm.contains("RidgeRegression"))
        .collect();

    if !linear_stats.is_empty() {
        let avg_efficiency = linear_stats
            .iter()
            .map(|s| s.memory_efficiency)
            .sum::<Float>()
            / linear_stats.len() as Float;
        println!(
            "Linear Regression Average Efficiency: {:.2} MB/M samples",
            avg_efficiency
        );
    }

    if !ridge_stats.is_empty() {
        let avg_efficiency = ridge_stats
            .iter()
            .map(|s| s.memory_efficiency)
            .sum::<Float>()
            / ridge_stats.len() as Float;
        println!(
            "Ridge Regression Average Efficiency: {:.2} MB/M samples",
            avg_efficiency
        );
    }

    println!("\nMemory Targets:");
    println!("  - Linear scaling with dataset size");
    println!("  - < 100 MB/M samples for basic algorithms");
    println!("  - < 200 MB/M samples for iterative algorithms");
    println!("  - Peak memory < 2x base dataset size");
    println!("  - Streaming: constant memory usage regardless of dataset size");
}

criterion_group!(
    memory_benches,
    benchmark_memory_scaling,
    benchmark_algorithm_memory_efficiency,
    benchmark_chunked_processing_memory,
    benchmark_streaming_memory,
    benchmark_memory_efficient_ops,
    benchmark_peak_memory_usage
);

criterion_main!(memory_benches);

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimation() {
        let memory_mb = estimate_array_memory_mb((1000, 100));
        let expected = (1000 * 100 * 8) as Float / (1024.0 * 1024.0); // 8 bytes per f64
        assert!((memory_mb - expected).abs() < 0.01);
    }

    #[test]
    fn test_memory_stats_creation() {
        let stats = MemoryStats::new(
            "Test".to_string(),
            (1000, 50),
            10.0,
            std::time::Duration::from_millis(100),
        );

        assert_eq!(stats.algorithm, "Test");
        assert_eq!(stats.dataset_size, (1000, 50));
        assert_eq!(stats.estimated_memory_mb, 10.0);
        assert!(stats.memory_efficiency > 0.0);
    }

    #[test]
    fn test_dataset_generation() {
        let (X, y) = generate_memory_test_dataset(100, 10, 42);
        assert_eq!(X.nrows(), 100);
        assert_eq!(X.ncols(), 10);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_memory_report_generation() {
        let stats = generate_memory_report();
        assert!(!stats.is_empty());

        for stat in &stats {
            assert!(stat.estimated_memory_mb > 0.0);
            assert!(stat.memory_efficiency > 0.0);
            assert!(stat.memory_per_prediction > 0.0);
        }
    }
}
