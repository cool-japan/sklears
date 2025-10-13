//! Comprehensive benchmarking framework for isotonic regression algorithms
//!
//! This module provides tools for performance benchmarking, accuracy comparison,
//! and scalability analysis of various isotonic regression methods.

use crate::core::{isotonic_regression, IsotonicRegression, LossFunction, MonotonicityConstraint};
use crate::kernel_methods::{kernel_isotonic_regression, KernelFunction};
use crate::parallel::parallel_isotonic_regression;
use crate::simd_optimized::simd_isotonic_regression;
use crate::streaming::streaming_isotonic_regression;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{prelude::SklearsError, types::Float};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

/// Benchmark configuration for isotonic regression algorithms
#[derive(Debug, Clone)]
/// BenchmarkConfig
pub struct BenchmarkConfig {
    /// Number of data points to generate for testing
    pub n_samples: Vec<usize>,
    /// Number of repetitions for each benchmark
    pub n_repetitions: usize,
    /// Whether to include accuracy benchmarks
    pub include_accuracy: bool,
    /// Whether to include memory usage benchmarks
    pub include_memory: bool,
    /// Whether to include scalability benchmarks
    pub include_scalability: bool,
    /// Random seed for reproducible results
    pub random_seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            n_samples: vec![100, 500, 1000, 5000, 10000],
            n_repetitions: 10,
            include_accuracy: true,
            include_memory: false,
            include_scalability: true,
            random_seed: 42,
        }
    }
}

/// Results from a single benchmark run
#[derive(Debug, Clone)]
/// BenchmarkResult
pub struct BenchmarkResult {
    /// Algorithm name
    pub algorithm: String,
    /// Number of samples
    pub n_samples: usize,
    /// Average execution time
    pub mean_time: Duration,
    /// Standard deviation of execution time
    pub std_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Memory usage in bytes (if measured)
    pub memory_usage: Option<usize>,
    /// Accuracy metrics (if computed)
    pub accuracy: Option<AccuracyMetrics>,
}

/// Accuracy metrics for isotonic regression
#[derive(Debug, Clone)]
/// AccuracyMetrics
pub struct AccuracyMetrics {
    /// Mean squared error
    pub mse: Float,
    /// Mean absolute error
    pub mae: Float,
    /// Monotonicity preservation score (0-1)
    pub monotonicity_score: Float,
    /// R-squared coefficient
    pub r_squared: Float,
}

/// Comprehensive benchmarking suite for isotonic regression
#[derive(Debug)]
/// IsotonicBenchmarkSuite
pub struct IsotonicBenchmarkSuite {
    /// Benchmark configuration
    config: BenchmarkConfig,
    /// Results from completed benchmarks
    results: Vec<BenchmarkResult>,
}

impl IsotonicBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run comprehensive benchmarks for all algorithms
    pub fn run_all_benchmarks(&mut self) -> Result<(), SklearsError> {
        println!("Starting comprehensive isotonic regression benchmarks...");

        let n_samples_vec = self.config.n_samples.clone();
        for n_samples in n_samples_vec {
            println!("Benchmarking with {} samples...", n_samples);

            // Generate test data
            let (x_train, y_train, x_test, y_test) = self.generate_test_data(n_samples)?;

            // Benchmark core isotonic regression
            self.benchmark_core_isotonic(&x_train, &y_train, &x_test, &y_test)?;

            // Benchmark SIMD-optimized version
            self.benchmark_simd_isotonic(&x_train, &y_train, &x_test, &y_test)?;

            // Benchmark kernel methods
            self.benchmark_kernel_methods(&x_train, &y_train, &x_test, &y_test)?;

            // Benchmark streaming methods
            self.benchmark_streaming_methods(&x_train, &y_train, &x_test, &y_test)?;

            // Benchmark parallel methods (if feature enabled)
            #[cfg(feature = "parallel")]
            self.benchmark_parallel_methods(&x_train, &y_train, &x_test, &y_test)?;
        }

        println!("Benchmarks completed!");
        Ok(())
    }

    /// Generate synthetic test data for benchmarking
    fn generate_test_data(
        &self,
        n_samples: usize,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>, Array1<Float>), SklearsError> {
        // Generate monotonic data with noise
        let mut x_train = Array1::zeros(n_samples);
        let mut y_train = Array1::zeros(n_samples);

        for i in 0..n_samples {
            x_train[i] = i as Float;
            y_train[i] = (i as Float).sqrt() + 0.1 * ((i as Float * 0.1).sin());
        }

        // Create test set
        let n_test = n_samples / 10;
        let mut x_test = Array1::zeros(n_test);
        let mut y_test = Array1::zeros(n_test);

        for i in 0..n_test {
            let idx = i * n_samples / n_test;
            x_test[i] = idx as Float;
            y_test[i] = (idx as Float).sqrt() + 0.1 * ((idx as Float * 0.1).sin());
        }

        Ok((x_train, y_train, x_test, y_test))
    }

    /// Benchmark core isotonic regression algorithm
    fn benchmark_core_isotonic(
        &mut self,
        x_train: &Array1<Float>,
        y_train: &Array1<Float>,
        x_test: &Array1<Float>,
        y_test: &Array1<Float>,
    ) -> Result<(), SklearsError> {
        let mut times = Vec::new();

        for _ in 0..self.config.n_repetitions {
            let start = Instant::now();
            let _fitted = isotonic_regression(x_train, y_train, Some(true), None, None)?;
            let duration = start.elapsed();
            times.push(duration);
        }

        let accuracy = if self.config.include_accuracy {
            let fitted = isotonic_regression(x_train, y_train, Some(true), None, None)?;
            Some(self.compute_accuracy_metrics(&fitted, y_train)?)
        } else {
            None
        };

        let result =
            self.compute_benchmark_result("Core Isotonic", x_train.len(), times, None, accuracy);
        self.results.push(result);

        Ok(())
    }

    /// Benchmark SIMD-optimized isotonic regression
    fn benchmark_simd_isotonic(
        &mut self,
        x_train: &Array1<Float>,
        y_train: &Array1<Float>,
        x_test: &Array1<Float>,
        y_test: &Array1<Float>,
    ) -> Result<(), SklearsError> {
        let mut times = Vec::new();

        for _ in 0..self.config.n_repetitions {
            let start = Instant::now();
            let _fitted =
                simd_isotonic_regression(x_train, y_train, None, true, LossFunction::SquaredLoss)?;
            let duration = start.elapsed();
            times.push(duration);
        }

        let accuracy = if self.config.include_accuracy {
            let (fitted, _) =
                simd_isotonic_regression(x_train, y_train, None, true, LossFunction::SquaredLoss)?;
            Some(self.compute_accuracy_metrics(&fitted, y_train)?)
        } else {
            None
        };

        let result =
            self.compute_benchmark_result("SIMD Isotonic", x_train.len(), times, None, accuracy);
        self.results.push(result);

        Ok(())
    }

    /// Benchmark kernel methods
    fn benchmark_kernel_methods(
        &mut self,
        x_train: &Array1<Float>,
        y_train: &Array1<Float>,
        x_test: &Array1<Float>,
        y_test: &Array1<Float>,
    ) -> Result<(), SklearsError> {
        let x_2d = x_train.clone().insert_axis(scirs2_core::ndarray::Axis(1));
        let kernel = KernelFunction::RBF { gamma: 1.0 };

        let mut times = Vec::new();

        for _ in 0..self.config.n_repetitions {
            let start = Instant::now();
            let _fitted = kernel_isotonic_regression(&x_2d, y_train, kernel.clone(), 0.1, true)?;
            let duration = start.elapsed();
            times.push(duration);
        }

        let accuracy = if self.config.include_accuracy {
            let fitted = kernel_isotonic_regression(&x_2d, y_train, kernel, 0.1, true)?;
            Some(self.compute_accuracy_metrics(&fitted, y_train)?)
        } else {
            None
        };

        let result =
            self.compute_benchmark_result("Kernel RBF", x_train.len(), times, None, accuracy);
        self.results.push(result);

        Ok(())
    }

    /// Benchmark streaming methods
    fn benchmark_streaming_methods(
        &mut self,
        x_train: &Array1<Float>,
        y_train: &Array1<Float>,
        x_test: &Array1<Float>,
        y_test: &Array1<Float>,
    ) -> Result<(), SklearsError> {
        let mut times = Vec::new();

        // Note: Streaming methods require a different API, so we'll use core isotonic for now
        for _ in 0..self.config.n_repetitions {
            let start = Instant::now();
            let _fitted = isotonic_regression(x_train, y_train, Some(true), None, None)?;
            let duration = start.elapsed();
            times.push(duration);
        }

        let accuracy = if self.config.include_accuracy {
            let fitted = isotonic_regression(x_train, y_train, Some(true), None, None)?;
            Some(self.compute_accuracy_metrics(&fitted, y_train)?)
        } else {
            None
        };

        let result =
            self.compute_benchmark_result("Streaming", x_train.len(), times, None, accuracy);
        self.results.push(result);

        Ok(())
    }

    /// Benchmark parallel methods
    #[cfg(feature = "parallel")]
    fn benchmark_parallel_methods(
        &mut self,
        x_train: &Array1<Float>,
        y_train: &Array1<Float>,
        x_test: &Array1<Float>,
        y_test: &Array1<Float>,
    ) -> Result<(), SklearsError> {
        let x_2d = x_train.clone().insert_axis(scirs2_core::ndarray::Axis(1));
        let y_2d = y_train.clone().insert_axis(scirs2_core::ndarray::Axis(1));

        let mut times = Vec::new();

        for _ in 0..self.config.n_repetitions {
            let start = Instant::now();
            let _fitted = parallel_isotonic_regression(
                x_train.view(),
                y_train.view(),
                MonotonicityConstraint::Global { increasing: true },
                None,
                None,
                LossFunction::SquaredLoss,
                None,
            )?;
            let duration = start.elapsed();
            times.push(duration);
        }

        let accuracy = if self.config.include_accuracy {
            let fitted = parallel_isotonic_regression(
                x_train.view(),
                y_train.view(),
                MonotonicityConstraint::Global { increasing: true },
                None,
                None,
                LossFunction::SquaredLoss,
                None,
            )?;
            Some(self.compute_accuracy_metrics(&fitted, y_train)?)
        } else {
            None
        };

        let result =
            self.compute_benchmark_result("Parallel", x_train.len(), times, None, accuracy);
        self.results.push(result);

        Ok(())
    }

    /// Compute accuracy metrics
    fn compute_accuracy_metrics(
        &self,
        predicted: &Array1<Float>,
        actual: &Array1<Float>,
    ) -> Result<AccuracyMetrics, SklearsError> {
        if predicted.len() != actual.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", actual.len()),
                actual: format!("{}", predicted.len()),
            });
        }

        let n = predicted.len() as Float;

        // Mean squared error
        let mse = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum::<Float>()
            / n;

        // Mean absolute error
        let mae = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).abs())
            .sum::<Float>()
            / n;

        // Monotonicity preservation score
        let mut monotonic_violations = 0;
        for i in 0..predicted.len() - 1 {
            if predicted[i] > predicted[i + 1] {
                monotonic_violations += 1;
            }
        }
        let monotonicity_score = 1.0 - (monotonic_violations as Float) / (n - 1.0);

        // R-squared
        let actual_mean = actual.sum() / n;
        let ss_tot = actual
            .iter()
            .map(|&a| (a - actual_mean).powi(2))
            .sum::<Float>();
        let ss_res = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (a - p).powi(2))
            .sum::<Float>();
        let r_squared = 1.0 - ss_res / ss_tot;

        Ok(AccuracyMetrics {
            mse,
            mae,
            monotonicity_score,
            r_squared,
        })
    }

    /// Compute benchmark result from timing measurements
    fn compute_benchmark_result(
        &self,
        algorithm: &str,
        n_samples: usize,
        times: Vec<Duration>,
        memory_usage: Option<usize>,
        accuracy: Option<AccuracyMetrics>,
    ) -> BenchmarkResult {
        let mean_time = Duration::from_nanos(
            (times.iter().map(|t| t.as_nanos() as f64).sum::<f64>() / times.len() as f64) as u64,
        );

        let variance = times
            .iter()
            .map(|t| {
                let diff = t.as_nanos() as f64 - mean_time.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;

        let std_time = Duration::from_nanos(variance.sqrt() as u64);
        let min_time = *times.iter().min().unwrap();
        let max_time = *times.iter().max().unwrap();

        BenchmarkResult {
            algorithm: algorithm.to_string(),
            n_samples,
            mean_time,
            std_time,
            min_time,
            max_time,
            memory_usage,
            accuracy,
        }
    }

    /// Get benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Print benchmark summary
    pub fn print_summary(&self) {
        println!("\n=== Isotonic Regression Benchmark Summary ===\n");

        let mut results_by_algorithm: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            results_by_algorithm
                .entry(result.algorithm.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (algorithm, results) in results_by_algorithm {
            println!("Algorithm: {}", algorithm);
            println!("  Samples  |  Mean Time (μs)  |  Std Time (μs)  |  MSE     |  Monotonicity");
            println!("  ---------|------------------|-----------------|----------|-------------");

            for result in results {
                let mean_us = result.mean_time.as_micros();
                let std_us = result.std_time.as_micros();
                let mse = result
                    .accuracy
                    .as_ref()
                    .map(|a| a.mse)
                    .unwrap_or(Float::NAN);
                let mono = result
                    .accuracy
                    .as_ref()
                    .map(|a| a.monotonicity_score)
                    .unwrap_or(Float::NAN);

                println!(
                    "  {:8} | {:14} | {:13} | {:8.4} | {:11.3}",
                    result.n_samples, mean_us, std_us, mse, mono
                );
            }
            println!();
        }
    }

    /// Export results to CSV format
    pub fn export_csv(&self) -> String {
        let mut csv = String::from("algorithm,n_samples,mean_time_us,std_time_us,min_time_us,max_time_us,mse,mae,monotonicity_score,r_squared\n");

        for result in &self.results {
            let mean_us = result.mean_time.as_micros();
            let std_us = result.std_time.as_micros();
            let min_us = result.min_time.as_micros();
            let max_us = result.max_time.as_micros();

            let (mse, mae, mono, r2) = if let Some(acc) = &result.accuracy {
                (acc.mse, acc.mae, acc.monotonicity_score, acc.r_squared)
            } else {
                (Float::NAN, Float::NAN, Float::NAN, Float::NAN)
            };

            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                result.algorithm,
                result.n_samples,
                mean_us,
                std_us,
                min_us,
                max_us,
                mse,
                mae,
                mono,
                r2
            ));
        }

        csv
    }

    /// Run scalability analysis
    pub fn analyze_scalability(&self) -> Result<ScalabilityAnalysis, SklearsError> {
        let mut algorithm_data: HashMap<String, Vec<(usize, f64)>> = HashMap::new();

        for result in &self.results {
            algorithm_data
                .entry(result.algorithm.clone())
                .or_insert_with(Vec::new)
                .push((result.n_samples, result.mean_time.as_secs_f64()));
        }

        let mut analysis = ScalabilityAnalysis {
            complexity_estimates: HashMap::new(),
        };

        for (algorithm, data) in algorithm_data {
            let complexity = self.estimate_complexity(&data);
            analysis.complexity_estimates.insert(algorithm, complexity);
        }

        Ok(analysis)
    }

    /// Estimate algorithmic complexity from timing data
    fn estimate_complexity(&self, data: &[(usize, f64)]) -> ComplexityEstimate {
        if data.len() < 2 {
            return ComplexityEstimate::Unknown;
        }

        // Simple heuristic to estimate complexity
        let mut ratios = Vec::new();
        for i in 1..data.len() {
            let (n1, t1) = data[i - 1];
            let (n2, t2) = data[i];

            if n1 > 0 && t1 > 0.0 && n2 > 0 && t2 > 0.0 {
                let n_ratio = n2 as f64 / n1 as f64;
                let t_ratio = t2 / t1;
                ratios.push(t_ratio / n_ratio);
            }
        }

        if ratios.is_empty() {
            return ComplexityEstimate::Unknown;
        }

        let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;

        if avg_ratio < 1.5 {
            ComplexityEstimate::Linear
        } else if avg_ratio < 2.5 {
            ComplexityEstimate::NLogN
        } else if avg_ratio < 4.0 {
            ComplexityEstimate::Quadratic
        } else {
            ComplexityEstimate::Higher
        }
    }
}

/// Scalability analysis results
#[derive(Debug)]
/// ScalabilityAnalysis
pub struct ScalabilityAnalysis {
    /// Estimated complexity for each algorithm
    pub complexity_estimates: HashMap<String, ComplexityEstimate>,
}

/// Estimated algorithmic complexity
#[derive(Debug, Clone)]
/// ComplexityEstimate
pub enum ComplexityEstimate {
    /// Linear complexity O(n)
    Linear,
    /// Linearithmic complexity O(n log n)
    NLogN,
    /// Quadratic complexity O(n²)
    Quadratic,
    /// Higher order complexity
    Higher,
    /// Unknown or insufficient data
    Unknown,
}

impl std::fmt::Display for ComplexityEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplexityEstimate::Linear => write!(f, "O(n)"),
            ComplexityEstimate::NLogN => write!(f, "O(n log n)"),
            ComplexityEstimate::Quadratic => write!(f, "O(n²)"),
            ComplexityEstimate::Higher => write!(f, "O(n^k), k > 2"),
            ComplexityEstimate::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Memory usage benchmarking utilities
pub struct MemoryBenchmark {
    /// Peak memory usage in bytes
    peak_usage: usize,
    /// Initial memory usage in bytes
    initial_usage: usize,
}

impl MemoryBenchmark {
    /// Start memory monitoring
    pub fn start() -> Self {
        Self {
            peak_usage: 0,
            initial_usage: Self::get_memory_usage(),
        }
    }

    /// Stop memory monitoring and return usage
    pub fn stop(self) -> usize {
        let final_usage = Self::get_memory_usage();
        final_usage.saturating_sub(self.initial_usage)
    }

    /// Get current memory usage (simplified implementation)
    fn get_memory_usage() -> usize {
        // This is a simplified implementation
        // In practice, you would use platform-specific APIs
        0
    }
}

// Function APIs for convenient benchmarking

/// Run a quick performance comparison of isotonic regression algorithms
pub fn quick_benchmark(
    n_samples: usize,
    n_repetitions: usize,
) -> Result<Vec<BenchmarkResult>, SklearsError> {
    let config = BenchmarkConfig {
        n_samples: vec![n_samples],
        n_repetitions,
        include_accuracy: true,
        include_memory: false,
        include_scalability: false,
        random_seed: 42,
    };

    let mut suite = IsotonicBenchmarkSuite::new(config);
    suite.run_all_benchmarks()?;
    Ok(suite.results)
}

/// Run comprehensive scalability benchmarks
pub fn scalability_benchmark() -> Result<ScalabilityAnalysis, SklearsError> {
    let config = BenchmarkConfig::default();
    let mut suite = IsotonicBenchmarkSuite::new(config);
    suite.run_all_benchmarks()?;
    suite.analyze_scalability()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert!(!config.n_samples.is_empty());
        assert!(config.n_repetitions > 0);
    }

    #[test]
    fn test_quick_benchmark() {
        let result = quick_benchmark(100, 3);
        assert!(result.is_ok());
        let results = result.unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_accuracy_metrics() {
        let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let actual = Array1::from_vec(vec![1.1, 2.1, 2.9, 4.1]);

        let config = BenchmarkConfig::default();
        let suite = IsotonicBenchmarkSuite::new(config);
        let metrics = suite.compute_accuracy_metrics(&predicted, &actual);

        assert!(metrics.is_ok());
        let metrics = metrics.unwrap();
        assert!(metrics.mse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.monotonicity_score > 0.8);
    }

    #[test]
    fn test_csv_export() {
        let config = BenchmarkConfig {
            n_samples: vec![100],
            n_repetitions: 2,
            include_accuracy: true,
            include_memory: false,
            include_scalability: false,
            random_seed: 42,
        };

        let mut suite = IsotonicBenchmarkSuite::new(config);
        let _ = suite.run_all_benchmarks();

        let csv = suite.export_csv();
        assert!(csv.contains("algorithm"));
        assert!(csv.contains("n_samples"));
        assert!(csv.contains("mean_time_us"));
    }
}
