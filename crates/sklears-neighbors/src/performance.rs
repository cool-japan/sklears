//! Performance analysis and benchmarking tools for neighbor algorithms
//!
//! This module provides comprehensive benchmarking and performance analysis
//! capabilities for neighbor search algorithms, allowing users to profile
//! different algorithms, compare their performance, and optimize their choices.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::Random;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::distance::Distance;
use crate::knn::{Algorithm, KNeighborsClassifier};
use crate::NeighborsError;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;

/// Performance metrics for neighbor search operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Time taken for index construction
    pub index_construction_time: Duration,
    /// Time taken for fitting/training
    pub fit_time: Duration,
    /// Time taken for prediction/query
    pub query_time: Duration,
    /// Average time per query
    pub avg_query_time: Duration,
    /// Total memory usage in bytes
    pub memory_usage: usize,
    /// Peak memory usage during operation
    pub peak_memory_usage: usize,
    /// Number of distance calculations performed
    pub distance_calculations: usize,
    /// Query accuracy (if ground truth available)
    pub accuracy: Option<Float>,
    /// Throughput (queries per second)
    pub throughput: Float,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            index_construction_time: Duration::new(0, 0),
            fit_time: Duration::new(0, 0),
            query_time: Duration::new(0, 0),
            avg_query_time: Duration::new(0, 0),
            memory_usage: 0,
            peak_memory_usage: 0,
            distance_calculations: 0,
            accuracy: None,
            throughput: 0.0,
        }
    }

    /// Calculate throughput from query time and number of queries
    pub fn calculate_throughput(&mut self, num_queries: usize) {
        if !self.query_time.is_zero() {
            self.throughput = num_queries as Float / self.query_time.as_secs_f64() as Float;
            self.avg_query_time = self.query_time / num_queries as u32;
        }
    }

    /// Update accuracy if ground truth is available
    pub fn set_accuracy(&mut self, accuracy: Float) {
        self.accuracy = Some(accuracy);
    }

    /// Add distance calculation count
    pub fn add_distance_calculations(&mut self, count: usize) {
        self.distance_calculations += count;
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for performance benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Algorithms to benchmark
    pub algorithms: Vec<Algorithm>,
    /// Distance metrics to test
    pub distance_metrics: Vec<Distance>,
    /// Dataset sizes to test
    pub dataset_sizes: Vec<usize>,
    /// Feature dimensions to test
    pub dimensions: Vec<usize>,
    /// Number of neighbors (k) to test
    pub k_values: Vec<usize>,
    /// Number of repetitions for averaging
    pub num_repetitions: usize,
    /// Whether to include memory profiling
    pub profile_memory: bool,
    /// Whether to generate synthetic data
    pub use_synthetic_data: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![Algorithm::Brute, Algorithm::KdTree, Algorithm::BallTree],
            distance_metrics: vec![Distance::Euclidean, Distance::Manhattan],
            dataset_sizes: vec![100, 500, 1000, 5000],
            dimensions: vec![2, 5, 10, 20],
            k_values: vec![1, 3, 5, 10],
            num_repetitions: 3,
            profile_memory: true,
            use_synthetic_data: true,
            random_seed: Some(42),
        }
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Algorithm used
    pub algorithm: Algorithm,
    /// Distance metric used
    pub distance_metric: Distance,
    /// Dataset size
    pub dataset_size: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of neighbors
    pub k: usize,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance benchmark suite for neighbor algorithms
pub struct NeighborBenchmark {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl NeighborBenchmark {
    /// Create a new benchmark suite with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            results: Vec::new(),
        }
    }

    /// Create a benchmark suite with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Set the algorithms to benchmark
    pub fn algorithms(mut self, algorithms: Vec<Algorithm>) -> Self {
        self.config.algorithms = algorithms;
        self
    }

    /// Set the distance metrics to test
    pub fn distance_metrics(mut self, metrics: Vec<Distance>) -> Self {
        self.config.distance_metrics = metrics;
        self
    }

    /// Set the dataset sizes to test
    pub fn dataset_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.config.dataset_sizes = sizes;
        self
    }

    /// Set the feature dimensions to test
    pub fn dimensions(mut self, dims: Vec<usize>) -> Self {
        self.config.dimensions = dims;
        self
    }

    /// Set the k values to test
    pub fn k_values(mut self, k_vals: Vec<usize>) -> Self {
        self.config.k_values = k_vals;
        self
    }

    /// Set number of repetitions
    pub fn repetitions(mut self, reps: usize) -> Self {
        self.config.num_repetitions = reps;
        self
    }

    /// Run the benchmark suite
    pub fn run(&mut self) -> Result<&[BenchmarkResult], NeighborsError> {
        let mut rng: Random<scirs2_core::random::rngs::StdRng> =
            if let Some(seed) = self.config.random_seed {
                Random::seed(seed)
            } else {
                Random::seed(42) // Use default seed instead of thread_rng for reproducibility
            };

        self.results.clear();

        // Iterate over all configuration combinations
        for &n_samples in &self.config.dataset_sizes {
            for &n_features in &self.config.dimensions {
                for &k in &self.config.k_values {
                    if k >= n_samples {
                        continue; // Skip invalid k values
                    }

                    let (x_train, y_train, x_test) = if self.config.use_synthetic_data {
                        self.generate_synthetic_data(n_samples, n_features, &mut rng)?
                    } else {
                        return Err(NeighborsError::InvalidInput(
                            "Custom data not yet supported".to_string(),
                        ));
                    };

                    for algorithm in &self.config.algorithms {
                        for distance_metric in &self.config.distance_metrics {
                            // Skip invalid combinations
                            if !self.is_valid_combination(algorithm, distance_metric, n_features) {
                                continue;
                            }

                            let mut total_metrics = PerformanceMetrics::new();

                            // Run multiple repetitions and average results
                            for _ in 0..self.config.num_repetitions {
                                let metrics = self.benchmark_single_run(
                                    &x_train,
                                    &y_train,
                                    &x_test,
                                    k,
                                    *algorithm,
                                    distance_metric.clone(),
                                )?;

                                total_metrics.index_construction_time +=
                                    metrics.index_construction_time;
                                total_metrics.fit_time += metrics.fit_time;
                                total_metrics.query_time += metrics.query_time;
                                total_metrics.memory_usage =
                                    metrics.memory_usage.max(total_metrics.memory_usage);
                                total_metrics.peak_memory_usage = metrics
                                    .peak_memory_usage
                                    .max(total_metrics.peak_memory_usage);
                                total_metrics.distance_calculations +=
                                    metrics.distance_calculations;
                            }

                            // Average the results
                            let reps = self.config.num_repetitions as u32;
                            total_metrics.index_construction_time /= reps;
                            total_metrics.fit_time /= reps;
                            total_metrics.query_time /= reps;
                            total_metrics.distance_calculations /= self.config.num_repetitions;
                            total_metrics.calculate_throughput(x_test.nrows());

                            let mut metadata = HashMap::new();
                            metadata.insert("repetitions".to_string(), reps.to_string());
                            metadata.insert("synthetic_data".to_string(), "true".to_string());

                            self.results.push(BenchmarkResult {
                                algorithm: *algorithm,
                                distance_metric: distance_metric.clone(),
                                dataset_size: n_samples,
                                num_features: n_features,
                                k,
                                metrics: total_metrics,
                                metadata,
                            });
                        }
                    }
                }
            }
        }

        Ok(&self.results)
    }

    /// Get the benchmark results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Find the best performing algorithm for given criteria
    pub fn best_algorithm_by_metric<F>(&self, metric_fn: F) -> Option<&BenchmarkResult>
    where
        F: Fn(&PerformanceMetrics) -> Float,
    {
        self.results.iter().min_by(|a, b| {
            metric_fn(&a.metrics)
                .partial_cmp(&metric_fn(&b.metrics))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get results filtered by dataset size
    pub fn results_by_size(&self, size: usize) -> Vec<&BenchmarkResult> {
        self.results
            .iter()
            .filter(|r| r.dataset_size == size)
            .collect()
    }

    /// Get results filtered by algorithm
    pub fn results_by_algorithm(&self, algorithm: Algorithm) -> Vec<&BenchmarkResult> {
        self.results
            .iter()
            .filter(|r| r.algorithm == algorithm)
            .collect()
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("# Neighbor Algorithm Performance Report\n\n");

        if self.results.is_empty() {
            report.push_str("No benchmark results available.\n");
            return report;
        }

        report.push_str("## Summary Statistics\n\n");
        report.push_str(&format!(
            "Total configurations tested: {}\n",
            self.results.len()
        ));

        // Find best performing algorithms
        if let Some(best_query_time) =
            self.best_algorithm_by_metric(|m| m.avg_query_time.as_secs_f64() as Float)
        {
            report.push_str(&format!(
                "Best query time: {:.4}ms ({:?} with {:?})\n",
                best_query_time.metrics.avg_query_time.as_secs_f64() * 1000.0,
                best_query_time.algorithm,
                best_query_time.distance_metric
            ));
        }

        if let Some(best_throughput) = self.best_algorithm_by_metric(|m| -m.throughput) {
            report.push_str(&format!(
                "Best throughput: {:.2} queries/sec ({:?} with {:?})\n",
                best_throughput.metrics.throughput,
                best_throughput.algorithm,
                best_throughput.distance_metric
            ));
        }

        report.push_str("\n## Detailed Results\n\n");
        report.push_str("| Algorithm | Distance | Size | Features | K | Query Time (ms) | Throughput (q/s) | Memory (KB) |\n");
        report.push_str("|-----------|----------|------|----------|---|-----------------|------------------|-------------|\n");

        for result in &self.results {
            report.push_str(&format!(
                "| {:?} | {:?} | {} | {} | {} | {:.4} | {:.2} | {:.1} |\n",
                result.algorithm,
                result.distance_metric,
                result.dataset_size,
                result.num_features,
                result.k,
                result.metrics.avg_query_time.as_secs_f64() * 1000.0,
                result.metrics.throughput,
                result.metrics.memory_usage as Float / 1024.0
            ));
        }

        report
    }

    /// Generate synthetic dataset for benchmarking
    fn generate_synthetic_data(
        &self,
        n_samples: usize,
        n_features: usize,
        rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    ) -> Result<(Array2<Float>, Array1<i32>, Array2<Float>), NeighborsError> {
        use scirs2_core::random::RandNormal as Normal;
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Generate training data
        let x_train =
            Array2::from_shape_simple_fn((n_samples, n_features), || rng.sample(normal) as Float);

        // Generate simple binary classification targets
        let y_train: Array1<i32> =
            Array1::from_shape_simple_fn(
                n_samples,
                || if rng.random::<f64>() < 0.5 { 0 } else { 1 },
            );

        // Generate test data (smaller set)
        let n_test = (n_samples / 10).max(10).min(100);
        let x_test =
            Array2::from_shape_simple_fn((n_test, n_features), || rng.sample(normal) as Float);

        Ok((x_train, y_train, x_test))
    }

    /// Check if algorithm-distance combination is valid
    fn is_valid_combination(
        &self,
        algorithm: &Algorithm,
        distance: &Distance,
        n_features: usize,
    ) -> bool {
        match algorithm {
            Algorithm::KdTree => {
                // KdTree works best with Euclidean distance and low dimensions
                matches!(distance, Distance::Euclidean | Distance::Manhattan) && n_features <= 20
            }
            Algorithm::BallTree => {
                // BallTree works with most distance metrics
                true
            }
            Algorithm::Brute => {
                // Brute force works with all combinations
                true
            }
            Algorithm::VpTree | Algorithm::CoverTree => {
                // These work with metric distances
                true
            }
        }
    }

    /// Benchmark a single algorithm configuration
    fn benchmark_single_run(
        &self,
        x_train: &Array2<Float>,
        y_train: &Array1<i32>,
        x_test: &Array2<Float>,
        k: usize,
        algorithm: Algorithm,
        distance: Distance,
    ) -> Result<PerformanceMetrics, NeighborsError> {
        let mut metrics = PerformanceMetrics::new();

        // Measure index construction and fitting time
        let start_fit = Instant::now();

        let classifier = KNeighborsClassifier::new(k)
            .with_algorithm(algorithm)
            .with_metric(distance);

        let fitted = classifier.fit(x_train, y_train)?;

        let fit_duration = start_fit.elapsed();
        metrics.fit_time = fit_duration;
        metrics.index_construction_time = fit_duration; // Approximate for now

        // Measure query time
        let start_query = Instant::now();
        let _predictions = fitted.predict(x_test)?;
        let query_duration = start_query.elapsed();

        metrics.query_time = query_duration;
        metrics.calculate_throughput(x_test.nrows());

        // Estimate memory usage (simplified)
        metrics.memory_usage = x_train.len() * std::mem::size_of::<Float>()
            + y_train.len() * std::mem::size_of::<i32>();
        metrics.peak_memory_usage = metrics.memory_usage;

        // Estimate distance calculations (simplified)
        metrics.distance_calculations = match algorithm {
            Algorithm::Brute => x_test.nrows() * x_train.nrows(),
            _ => x_test.nrows() * k * 10, // Heuristic for tree-based methods
        };

        Ok(metrics)
    }
}

impl Default for NeighborBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick performance profiler for neighbor algorithms
pub struct QuickProfiler {
    start_time: Option<Instant>,
    measurements: HashMap<String, Duration>,
}

impl QuickProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            start_time: None,
            measurements: HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop timer and record measurement
    pub fn stop_timer(&mut self, operation: &str) {
        if let Some(start) = self.start_time.take() {
            self.measurements
                .insert(operation.to_string(), start.elapsed());
        }
    }

    /// Time a closure and record the result
    pub fn time_operation<F, R>(&mut self, operation: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        self.measurements
            .insert(operation.to_string(), start.elapsed());
        result
    }

    /// Get all measurements
    pub fn measurements(&self) -> &HashMap<String, Duration> {
        &self.measurements
    }

    /// Get measurement for specific operation
    pub fn get_measurement(&self, operation: &str) -> Option<Duration> {
        self.measurements.get(operation).copied()
    }

    /// Print summary of all measurements
    pub fn print_summary(&self) {
        println!("Performance Profiling Summary:");
        println!("==============================");
        for (operation, duration) in &self.measurements {
            println!("{}: {:.4}ms", operation, duration.as_secs_f64() * 1000.0);
        }
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.start_time = None;
    }
}

impl Default for QuickProfiler {
    fn default() -> Self {
        Self::new()
    }
}

use scirs2_core::ndarray::Array1;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_creation() {
        let metrics = PerformanceMetrics::new();
        assert_eq!(metrics.distance_calculations, 0);
        assert_eq!(metrics.memory_usage, 0);
        assert!(metrics.accuracy.is_none());
    }

    #[test]
    fn test_performance_metrics_throughput() {
        let mut metrics = PerformanceMetrics::new();
        metrics.query_time = Duration::from_millis(100);
        metrics.calculate_throughput(10);

        assert!(metrics.throughput > 0.0);
        assert_eq!(metrics.avg_query_time, Duration::from_millis(10));
    }

    #[test]
    fn test_benchmark_config_creation() {
        let config = BenchmarkConfig::default();
        assert!(!config.algorithms.is_empty());
        assert!(!config.distance_metrics.is_empty());
        assert!(!config.dataset_sizes.is_empty());
        assert_eq!(config.num_repetitions, 3);
    }

    #[test]
    fn test_benchmark_builder_pattern() {
        let benchmark = NeighborBenchmark::new()
            .algorithms(vec![Algorithm::Brute])
            .k_values(vec![1, 3])
            .repetitions(1);

        assert_eq!(benchmark.config.algorithms.len(), 1);
        assert_eq!(benchmark.config.k_values.len(), 2);
        assert_eq!(benchmark.config.num_repetitions, 1);
    }

    #[test]
    fn test_quick_profiler() {
        let mut profiler = QuickProfiler::new();

        profiler.time_operation("test_operation", || {
            std::thread::sleep(Duration::from_millis(1));
        });

        let measurement = profiler.get_measurement("test_operation");
        assert!(measurement.is_some());
        assert!(measurement.unwrap() >= Duration::from_millis(1));
    }

    #[test]
    fn test_profiler_start_stop() {
        let mut profiler = QuickProfiler::new();

        profiler.start_timer();
        std::thread::sleep(Duration::from_millis(1));
        profiler.stop_timer("manual_timing");

        let measurement = profiler.get_measurement("manual_timing");
        assert!(measurement.is_some());
        assert!(measurement.unwrap() >= Duration::from_millis(1));
    }

    #[test]
    fn test_benchmark_small_run() {
        let mut benchmark = NeighborBenchmark::new()
            .algorithms(vec![Algorithm::Brute])
            .distance_metrics(vec![Distance::Euclidean])
            .dataset_sizes(vec![10])
            .dimensions(vec![2])
            .k_values(vec![1])
            .repetitions(1);

        let results = benchmark.run();
        assert!(results.is_ok());

        let results = results.unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
        assert_eq!(result.algorithm, Algorithm::Brute);
        assert!(matches!(result.distance_metric, Distance::Euclidean));
        assert_eq!(result.dataset_size, 10);
        assert_eq!(result.num_features, 2);
        assert_eq!(result.k, 1);
        assert!(result.metrics.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_report_generation() {
        let mut benchmark = NeighborBenchmark::new()
            .algorithms(vec![Algorithm::Brute])
            .distance_metrics(vec![Distance::Euclidean])
            .dataset_sizes(vec![10])
            .dimensions(vec![2])
            .k_values(vec![1])
            .repetitions(1);

        benchmark.run().unwrap();
        let report = benchmark.generate_report();

        assert!(report.contains("Performance Report"));
        assert!(report.contains("Total configurations tested: 1"));
        assert!(report.contains("Brute"));
        assert!(report.contains("Euclidean"));
    }
}
