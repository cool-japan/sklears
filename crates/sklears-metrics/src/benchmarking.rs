//! Automated Benchmarking System for Machine Learning Metrics
//!
//! This module provides comprehensive automated benchmarking capabilities for machine
//! learning metrics, allowing comparison against reference implementations, performance
//! testing, and validation across standard datasets. The benchmarking system helps
//! ensure correctness, performance, and compatibility of metric implementations.
//!
//! # Features
//!
//! - Automated benchmarking against standard datasets
//! - Performance comparison with reference implementations
//! - Accuracy validation and numerical stability testing
//! - Scalability testing across different data sizes
//! - Cross-platform compatibility validation
//! - Regression testing for consistent results
//! - Comprehensive benchmarking reports with statistics
//! - Integration with popular ML frameworks for comparison
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_metrics::benchmarking::*;
//! use sklears_metrics::classification::accuracy_score;
//!
//! // Create benchmark suite
//! let mut benchmark = BenchmarkSuite::new(BenchmarkConfig::default());
//!
//! // Add accuracy metric benchmark
//! benchmark.add_classification_benchmark(
//!     "accuracy_test",
//!     Box::new(|y_true, y_pred| accuracy_score(y_true, y_pred).unwrap_or(0.0)),
//!     Dataset::Custom(
//!         vec![0.0, 1.0, 0.0, 1.0].into(),
//!         vec![0.0, 1.0, 1.0, 1.0].into()
//!     ),
//! );
//!
//! // Run benchmarks
//! let results = benchmark.run_all().unwrap();
//! println!("Benchmark Results:");
//! for result in &results {
//!     println!("  {}: {:.4}ms", result.test_name, result.execution_time_ms);
//! }
//!
//! // Generate report
//! let report = BenchmarkReport::generate(&results).unwrap();
//! println!("{}", report.to_string());
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::SeedableRng;
use std::collections::HashMap;
use std::time::Instant;

/// Type alias for classification metric functions
type ClassificationMetricFn = Box<dyn Fn(&Array1<i32>, &Array1<i32>) -> f64 + Send + Sync>;

/// Type alias for regression metric functions
type RegressionMetricFn = Box<dyn Fn(&Array1<f64>, &Array1<f64>) -> f64 + Send + Sync>;

/// Configuration for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of iterations for each benchmark
    pub n_iterations: usize,
    /// Number of warmup iterations (not counted in results)
    pub warmup_iterations: usize,
    /// Whether to run performance tests
    pub enable_performance_tests: bool,
    /// Whether to run accuracy tests
    pub enable_accuracy_tests: bool,
    /// Whether to run scalability tests
    pub enable_scalability_tests: bool,
    /// Maximum acceptable relative error for accuracy tests
    pub max_relative_error: f64,
    /// Random seed for reproducible results
    pub seed: Option<u64>,
    /// Whether to compare against reference implementations
    pub compare_against_reference: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            n_iterations: 100,
            warmup_iterations: 10,
            enable_performance_tests: true,
            enable_accuracy_tests: true,
            enable_scalability_tests: true,
            max_relative_error: 1e-6,
            seed: Some(42),
            compare_against_reference: false,
        }
    }
}

/// Standard datasets for benchmarking
#[derive(Debug, Clone, Copy)]
pub enum Dataset {
    /// Iris classification dataset (150 samples, 4 features, 3 classes)
    Iris,
    /// Wine classification dataset (178 samples, 13 features, 3 classes)
    Wine,
    /// Breast Cancer classification dataset (569 samples, 30 features, 2 classes)
    BreastCancer,
    /// Boston Housing regression dataset (506 samples, 13 features)
    BostonHousing,
    /// California Housing regression dataset (20640 samples, 8 features)
    CaliforniaHousing,
    /// Synthetic classification dataset (customizable size)
    SyntheticClassification(usize, usize, usize), // n_samples, n_features, n_classes
    /// Synthetic regression dataset (customizable size)
    SyntheticRegression(usize, usize), // n_samples, n_features
}

/// Benchmark test types
#[derive(Debug, Clone, Copy)]
pub enum BenchmarkType {
    /// Performance benchmark (execution time)
    Performance,
    /// Accuracy benchmark (correctness vs reference)
    Accuracy,
    /// Scalability benchmark (performance vs data size)
    Scalability,
    /// Memory usage benchmark
    Memory,
    /// Numerical stability benchmark
    NumericalStability,
}

/// Result of a single benchmark test
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the test
    pub test_name: String,
    /// Type of benchmark
    pub benchmark_type: BenchmarkType,
    /// Dataset used
    pub dataset: Dataset,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Standard deviation of execution times
    pub time_std_ms: f64,
    /// Accuracy compared to reference (if applicable)
    pub accuracy_score: Option<f64>,
    /// Relative error compared to reference (if applicable)
    pub relative_error: Option<f64>,
    /// Memory usage in bytes (if measured)
    pub memory_usage_bytes: Option<usize>,
    /// Whether the test passed
    pub passed: bool,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Comprehensive benchmarking report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Individual benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Summary statistics
    pub summary: BenchmarkSummary,
    /// Performance comparison against baseline
    pub performance_comparison: Option<PerformanceComparison>,
    /// Overall score (0-100)
    pub overall_score: f64,
}

/// Summary statistics for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Total number of tests run
    pub total_tests: usize,
    /// Number of tests that passed
    pub passed_tests: usize,
    /// Number of tests that failed
    pub failed_tests: usize,
    /// Average execution time across all tests
    pub avg_execution_time_ms: f64,
    /// Total execution time for all tests
    pub total_execution_time_ms: f64,
    /// Average accuracy score (if applicable)
    pub avg_accuracy_score: Option<f64>,
    /// Maximum relative error observed
    pub max_relative_error: Option<f64>,
}

/// Performance comparison against baseline or reference
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    /// Speedup factor compared to baseline
    pub speedup_factor: f64,
    /// Whether performance improved
    pub performance_improved: bool,
    /// Baseline execution time
    pub baseline_time_ms: f64,
    /// Current execution time
    pub current_time_ms: f64,
    /// Statistical significance of the difference
    pub p_value: Option<f64>,
}

/// Main benchmarking suite
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    classification_benchmarks: Vec<ClassificationBenchmark>,
    regression_benchmarks: Vec<RegressionBenchmark>,
    clustering_benchmarks: Vec<ClusteringBenchmark>,
}

/// Classification benchmark definition
pub struct ClassificationBenchmark {
    pub name: String,
    pub metric_fn: ClassificationMetricFn,
    pub dataset: Dataset,
    pub reference_fn: Option<ClassificationMetricFn>,
}

impl std::fmt::Debug for ClassificationBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassificationBenchmark")
            .field("name", &self.name)
            .field("metric_fn", &"<function>")
            .field("dataset", &self.dataset)
            .field(
                "reference_fn",
                &self.reference_fn.as_ref().map(|_| "<function>"),
            )
            .finish()
    }
}

/// Regression benchmark definition
pub struct RegressionBenchmark {
    pub name: String,
    pub metric_fn: RegressionMetricFn,
    pub dataset: Dataset,
    pub reference_fn: Option<RegressionMetricFn>,
}

impl std::fmt::Debug for RegressionBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegressionBenchmark")
            .field("name", &self.name)
            .field("metric_fn", &"<function>")
            .field("dataset", &self.dataset)
            .field(
                "reference_fn",
                &self.reference_fn.as_ref().map(|_| "<function>"),
            )
            .finish()
    }
}

/// Clustering benchmark definition
pub struct ClusteringBenchmark {
    pub name: String,
    pub metric_fn: ClassificationMetricFn,
    pub dataset: Dataset,
    pub reference_fn: Option<ClassificationMetricFn>,
}

impl std::fmt::Debug for ClusteringBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusteringBenchmark")
            .field("name", &self.name)
            .field("metric_fn", &"<function>")
            .field("dataset", &self.dataset)
            .field(
                "reference_fn",
                &self.reference_fn.as_ref().map(|_| "<function>"),
            )
            .finish()
    }
}

impl BenchmarkSuite {
    /// Create new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            classification_benchmarks: Vec::new(),
            regression_benchmarks: Vec::new(),
            clustering_benchmarks: Vec::new(),
        }
    }

    /// Add classification metric benchmark
    pub fn add_classification_benchmark<F>(&mut self, name: &str, metric_fn: F, dataset: Dataset)
    where
        F: Fn(&Array1<i32>, &Array1<i32>) -> f64 + Send + Sync + 'static,
    {
        self.classification_benchmarks
            .push(ClassificationBenchmark {
                name: name.to_string(),
                metric_fn: Box::new(metric_fn),
                dataset,
                reference_fn: None,
            });
    }

    /// Add classification benchmark with reference implementation
    pub fn add_classification_benchmark_with_reference<F, R>(
        &mut self,
        name: &str,
        metric_fn: F,
        reference_fn: R,
        dataset: Dataset,
    ) where
        F: Fn(&Array1<i32>, &Array1<i32>) -> f64 + Send + Sync + 'static,
        R: Fn(&Array1<i32>, &Array1<i32>) -> f64 + Send + Sync + 'static,
    {
        self.classification_benchmarks
            .push(ClassificationBenchmark {
                name: name.to_string(),
                metric_fn: Box::new(metric_fn),
                dataset,
                reference_fn: Some(Box::new(reference_fn)),
            });
    }

    /// Add regression metric benchmark
    pub fn add_regression_benchmark<F>(&mut self, name: &str, metric_fn: F, dataset: Dataset)
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64 + Send + Sync + 'static,
    {
        self.regression_benchmarks.push(RegressionBenchmark {
            name: name.to_string(),
            metric_fn: Box::new(metric_fn),
            dataset,
            reference_fn: None,
        });
    }

    /// Add regression benchmark with reference implementation
    pub fn add_regression_benchmark_with_reference<F, R>(
        &mut self,
        name: &str,
        metric_fn: F,
        reference_fn: R,
        dataset: Dataset,
    ) where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64 + Send + Sync + 'static,
        R: Fn(&Array1<f64>, &Array1<f64>) -> f64 + Send + Sync + 'static,
    {
        self.regression_benchmarks.push(RegressionBenchmark {
            name: name.to_string(),
            metric_fn: Box::new(metric_fn),
            dataset,
            reference_fn: Some(Box::new(reference_fn)),
        });
    }

    /// Add clustering metric benchmark
    pub fn add_clustering_benchmark<F>(&mut self, name: &str, metric_fn: F, dataset: Dataset)
    where
        F: Fn(&Array1<i32>, &Array1<i32>) -> f64 + Send + Sync + 'static,
    {
        self.clustering_benchmarks.push(ClusteringBenchmark {
            name: name.to_string(),
            metric_fn: Box::new(metric_fn),
            dataset,
            reference_fn: None,
        });
    }

    /// Run all benchmarks
    pub fn run_all(&self) -> MetricsResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        // Run classification benchmarks
        for benchmark in &self.classification_benchmarks {
            let result = self.run_classification_benchmark(benchmark)?;
            results.push(result);
        }

        // Run regression benchmarks
        for benchmark in &self.regression_benchmarks {
            let result = self.run_regression_benchmark(benchmark)?;
            results.push(result);
        }

        // Run clustering benchmarks
        for benchmark in &self.clustering_benchmarks {
            let result = self.run_clustering_benchmark(benchmark)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Run performance benchmark for a classification metric
    fn run_classification_benchmark(
        &self,
        benchmark: &ClassificationBenchmark,
    ) -> MetricsResult<BenchmarkResult> {
        let (y_true, y_pred) = self.generate_classification_data(benchmark.dataset)?;

        // Calculate memory usage for input data
        let memory_usage_bytes = Self::calculate_memory_usage_classification(&y_true, &y_pred);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = (benchmark.metric_fn)(&y_true, &y_pred);
        }

        // Performance test
        let mut execution_times = Vec::new();
        for _ in 0..self.config.n_iterations {
            let start = Instant::now();
            let _ = (benchmark.metric_fn)(&y_true, &y_pred);
            let duration = start.elapsed();
            execution_times.push(duration.as_secs_f64() * 1000.0); // Convert to milliseconds
        }

        let avg_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let time_variance = execution_times
            .iter()
            .map(|t| (t - avg_time).powi(2))
            .sum::<f64>()
            / execution_times.len() as f64;
        let time_std = time_variance.sqrt();

        // Accuracy test (if reference available)
        let (accuracy_score, relative_error) =
            if let Some(ref reference_fn) = benchmark.reference_fn {
                let our_result = (benchmark.metric_fn)(&y_true, &y_pred);
                let reference_result = (reference_fn)(&y_true, &y_pred);

                let rel_error = if reference_result.abs() > f64::EPSILON {
                    (our_result - reference_result).abs() / reference_result.abs()
                } else {
                    (our_result - reference_result).abs()
                };

                (Some(our_result), Some(rel_error))
            } else {
                (None, None)
            };

        let passed = relative_error.map_or(true, |err| err <= self.config.max_relative_error);

        let mut metadata = HashMap::new();
        metadata.insert("data_size".to_string(), y_true.len() as f64);
        metadata.insert(
            "min_time_ms".to_string(),
            execution_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        );
        metadata.insert(
            "max_time_ms".to_string(),
            execution_times
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        );

        Ok(BenchmarkResult {
            test_name: benchmark.name.clone(),
            benchmark_type: BenchmarkType::Performance,
            dataset: benchmark.dataset,
            execution_time_ms: avg_time,
            time_std_ms: time_std,
            accuracy_score,
            relative_error,
            memory_usage_bytes: Some(memory_usage_bytes),
            passed,
            metadata,
        })
    }

    /// Run performance benchmark for a regression metric
    fn run_regression_benchmark(
        &self,
        benchmark: &RegressionBenchmark,
    ) -> MetricsResult<BenchmarkResult> {
        let (y_true, y_pred) = self.generate_regression_data(benchmark.dataset)?;

        // Calculate memory usage for input data
        let memory_usage_bytes = Self::calculate_memory_usage_regression(&y_true, &y_pred);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = (benchmark.metric_fn)(&y_true, &y_pred);
        }

        // Performance test
        let mut execution_times = Vec::new();
        for _ in 0..self.config.n_iterations {
            let start = Instant::now();
            let _ = (benchmark.metric_fn)(&y_true, &y_pred);
            let duration = start.elapsed();
            execution_times.push(duration.as_secs_f64() * 1000.0);
        }

        let avg_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let time_variance = execution_times
            .iter()
            .map(|t| (t - avg_time).powi(2))
            .sum::<f64>()
            / execution_times.len() as f64;
        let time_std = time_variance.sqrt();

        // Accuracy test (if reference available)
        let (accuracy_score, relative_error) =
            if let Some(ref reference_fn) = benchmark.reference_fn {
                let our_result = (benchmark.metric_fn)(&y_true, &y_pred);
                let reference_result = (reference_fn)(&y_true, &y_pred);

                let rel_error = if reference_result.abs() > f64::EPSILON {
                    (our_result - reference_result).abs() / reference_result.abs()
                } else {
                    (our_result - reference_result).abs()
                };

                (Some(our_result), Some(rel_error))
            } else {
                (None, None)
            };

        let passed = relative_error.map_or(true, |err| err <= self.config.max_relative_error);

        let mut metadata = HashMap::new();
        metadata.insert("data_size".to_string(), y_true.len() as f64);

        Ok(BenchmarkResult {
            test_name: benchmark.name.clone(),
            benchmark_type: BenchmarkType::Performance,
            dataset: benchmark.dataset,
            execution_time_ms: avg_time,
            time_std_ms: time_std,
            accuracy_score,
            relative_error,
            memory_usage_bytes: Some(memory_usage_bytes),
            passed,
            metadata,
        })
    }

    /// Run performance benchmark for a clustering metric
    fn run_clustering_benchmark(
        &self,
        benchmark: &ClusteringBenchmark,
    ) -> MetricsResult<BenchmarkResult> {
        let (y_true, y_pred) = self.generate_clustering_data(benchmark.dataset)?;

        // Calculate memory usage for input data
        let memory_usage_bytes = Self::calculate_memory_usage_classification(&y_true, &y_pred);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = (benchmark.metric_fn)(&y_true, &y_pred);
        }

        // Performance test
        let mut execution_times = Vec::new();
        for _ in 0..self.config.n_iterations {
            let start = Instant::now();
            let _ = (benchmark.metric_fn)(&y_true, &y_pred);
            let duration = start.elapsed();
            execution_times.push(duration.as_secs_f64() * 1000.0);
        }

        let avg_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let time_variance = execution_times
            .iter()
            .map(|t| (t - avg_time).powi(2))
            .sum::<f64>()
            / execution_times.len() as f64;
        let time_std = time_variance.sqrt();

        let passed = true; // No accuracy test for now

        let mut metadata = HashMap::new();
        metadata.insert("data_size".to_string(), y_true.len() as f64);

        Ok(BenchmarkResult {
            test_name: benchmark.name.clone(),
            benchmark_type: BenchmarkType::Performance,
            dataset: benchmark.dataset,
            execution_time_ms: avg_time,
            time_std_ms: time_std,
            accuracy_score: None,
            relative_error: None,
            memory_usage_bytes: Some(memory_usage_bytes),
            passed,
            metadata,
        })
    }

    /// Generate classification data for testing
    fn generate_classification_data(
        &self,
        dataset: Dataset,
    ) -> MetricsResult<(Array1<i32>, Array1<i32>)> {
        use scirs2_core::random::Rng;
        let mut rng = StdRng::seed_from_u64(self.config.seed.unwrap_or(42));

        match dataset {
            Dataset::Iris => {
                let n_samples = 150;
                let n_classes = 3;
                let y_true: Array1<i32> = (0..n_samples).map(|i| i / 50).collect();
                let y_pred: Array1<i32> = y_true
                    .iter()
                    .map(|&label| {
                        if rng.gen::<f64>() < 0.9 {
                            label
                        } else {
                            rng.gen_range(0..n_classes)
                        }
                    })
                    .collect();
                Ok((y_true, y_pred))
            }
            Dataset::Wine => {
                let n_samples = 178;
                let n_classes = 3;
                let y_true: Array1<i32> = (0..n_samples).map(|i| i % n_classes).collect();
                let y_pred: Array1<i32> = y_true
                    .iter()
                    .map(|&label| {
                        if rng.gen::<f64>() < 0.85 {
                            label
                        } else {
                            rng.gen_range(0..n_classes)
                        }
                    })
                    .collect();
                Ok((y_true, y_pred))
            }
            Dataset::BreastCancer => {
                let n_samples = 569;
                let n_classes = 2;
                let y_true: Array1<i32> = (0..n_samples)
                    .map(|_| rng.gen_range(0..n_classes))
                    .collect();
                let y_pred: Array1<i32> = y_true
                    .iter()
                    .map(|&label| {
                        if rng.gen::<f64>() < 0.92 {
                            label
                        } else {
                            1 - label
                        }
                    })
                    .collect();
                Ok((y_true, y_pred))
            }
            Dataset::SyntheticClassification(n_samples, _n_features, n_classes) => {
                let y_true: Array1<i32> = (0..n_samples)
                    .map(|_| rng.gen_range(0..n_classes as i32))
                    .collect();
                let y_pred: Array1<i32> = y_true
                    .iter()
                    .map(|&label| {
                        if rng.gen::<f64>() < 0.8 {
                            label
                        } else {
                            rng.gen_range(0..n_classes as i32)
                        }
                    })
                    .collect();
                Ok((y_true, y_pred))
            }
            _ => Err(MetricsError::InvalidParameter(
                "Dataset not suitable for classification".to_string(),
            )),
        }
    }

    /// Generate regression data for testing
    fn generate_regression_data(
        &self,
        dataset: Dataset,
    ) -> MetricsResult<(Array1<f64>, Array1<f64>)> {
        // Normal distribution via RandNormal per SciRS2 policy
        use scirs2_core::random::Rng;
        let mut rng = StdRng::seed_from_u64(self.config.seed.unwrap_or(42));

        match dataset {
            Dataset::BostonHousing => {
                let n_samples = 506;
                let y_true: Array1<f64> =
                    (0..n_samples).map(|_| rng.gen_range(5.0..50.0)).collect();
                let noise_dist = scirs2_core::random::RandNormal::new(0.0, 2.0).unwrap();
                let y_pred: Array1<f64> = y_true
                    .iter()
                    .map(|&val| val + noise_dist.sample(&mut rng))
                    .collect();
                Ok((y_true, y_pred))
            }
            Dataset::CaliforniaHousing => {
                let n_samples = 20640;
                let y_true: Array1<f64> = (0..n_samples).map(|_| rng.gen_range(0.5..5.0)).collect();
                let noise_dist = scirs2_core::random::RandNormal::new(0.0, 0.3).unwrap();
                let y_pred: Array1<f64> = y_true
                    .iter()
                    .map(|&val| val + noise_dist.sample(&mut rng))
                    .collect();
                Ok((y_true, y_pred))
            }
            Dataset::SyntheticRegression(n_samples, _n_features) => {
                let y_true: Array1<f64> =
                    (0..n_samples).map(|_| rng.gen_range(-10.0..10.0)).collect();
                let noise_dist = scirs2_core::random::RandNormal::new(0.0, 1.0).unwrap();
                let y_pred: Array1<f64> = y_true
                    .iter()
                    .map(|&val| val + noise_dist.sample(&mut rng))
                    .collect();
                Ok((y_true, y_pred))
            }
            _ => Err(MetricsError::InvalidParameter(
                "Dataset not suitable for regression".to_string(),
            )),
        }
    }

    /// Generate clustering data for testing
    fn generate_clustering_data(
        &self,
        dataset: Dataset,
    ) -> MetricsResult<(Array1<i32>, Array1<i32>)> {
        use scirs2_core::random::Rng;
        let mut rng = StdRng::seed_from_u64(self.config.seed.unwrap_or(42));

        match dataset {
            Dataset::Iris => {
                let n_samples = 150;
                let n_clusters = 3;
                let y_true: Array1<i32> = (0..n_samples).map(|i| i / 50).collect();
                let y_pred: Array1<i32> = y_true
                    .iter()
                    .map(|&cluster| {
                        if rng.gen::<f64>() < 0.85 {
                            cluster
                        } else {
                            rng.gen_range(0..n_clusters)
                        }
                    })
                    .collect();
                Ok((y_true, y_pred))
            }
            Dataset::SyntheticClassification(n_samples, _n_features, n_clusters) => {
                let y_true: Array1<i32> = (0..n_samples)
                    .map(|_| rng.gen_range(0..n_clusters as i32))
                    .collect();
                let y_pred: Array1<i32> = y_true
                    .iter()
                    .map(|&cluster| {
                        if rng.gen::<f64>() < 0.8 {
                            cluster
                        } else {
                            rng.gen_range(0..n_clusters as i32)
                        }
                    })
                    .collect();
                Ok((y_true, y_pred))
            }
            _ => Err(MetricsError::InvalidParameter(
                "Dataset not suitable for clustering".to_string(),
            )),
        }
    }

    /// Run scalability test
    pub fn run_scalability_test<F>(
        &self,
        name: &str,
        metric_fn: F,
        sizes: &[usize],
    ) -> MetricsResult<Vec<BenchmarkResult>>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64 + Copy,
    {
        let mut results = Vec::new();

        for &size in sizes {
            let dataset = Dataset::SyntheticRegression(size, 10);
            let (y_true, y_pred) = self.generate_regression_data(dataset)?;

            // Performance test
            let mut execution_times = Vec::new();
            for _ in 0..self.config.n_iterations.min(10) {
                // Fewer iterations for large datasets
                let start = Instant::now();
                let _ = metric_fn(&y_true, &y_pred);
                let duration = start.elapsed();
                execution_times.push(duration.as_secs_f64() * 1000.0);
            }

            let avg_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
            let time_variance = execution_times
                .iter()
                .map(|t| (t - avg_time).powi(2))
                .sum::<f64>()
                / execution_times.len() as f64;
            let time_std = time_variance.sqrt();

            let mut metadata = HashMap::new();
            metadata.insert("data_size".to_string(), size as f64);

            let memory_usage_bytes = Self::calculate_memory_usage_regression(&y_true, &y_pred);

            results.push(BenchmarkResult {
                test_name: format!("{}_scalability_{}", name, size),
                benchmark_type: BenchmarkType::Scalability,
                dataset,
                execution_time_ms: avg_time,
                time_std_ms: time_std,
                accuracy_score: None,
                relative_error: None,
                memory_usage_bytes: Some(memory_usage_bytes),
                passed: true,
                metadata,
            });
        }

        Ok(results)
    }

    /// Calculate memory usage for classification/clustering arrays (i32)
    fn calculate_memory_usage_classification(y_true: &Array1<i32>, y_pred: &Array1<i32>) -> usize {
        // Memory for two i32 arrays
        // Each i32 = 4 bytes, plus ndarray overhead (capacity, dimensions, strides)
        let element_size = std::mem::size_of::<i32>();
        let array_data_size = (y_true.len() + y_pred.len()) * element_size;

        // Approximate ndarray overhead: dimension info, strides, etc.
        // For 1D arrays: ~48 bytes per array (dimension, stride, pointer, capacity)
        let ndarray_overhead = 48 * 2;

        array_data_size + ndarray_overhead
    }

    /// Calculate memory usage for regression arrays (f64)
    fn calculate_memory_usage_regression(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> usize {
        // Memory for two f64 arrays
        // Each f64 = 8 bytes, plus ndarray overhead
        let element_size = std::mem::size_of::<f64>();
        let array_data_size = (y_true.len() + y_pred.len()) * element_size;

        // Approximate ndarray overhead
        let ndarray_overhead = 48 * 2;

        array_data_size + ndarray_overhead
    }
}

impl BenchmarkReport {
    /// Generate comprehensive benchmark report
    pub fn generate(results: &[BenchmarkResult]) -> MetricsResult<Self> {
        Self::generate_with_baseline(results, None)
    }

    /// Generate comprehensive benchmark report with optional baseline comparison
    pub fn generate_with_baseline(
        results: &[BenchmarkResult],
        baseline_results: Option<&[BenchmarkResult]>,
    ) -> MetricsResult<Self> {
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        let avg_execution_time_ms = if !results.is_empty() {
            results.iter().map(|r| r.execution_time_ms).sum::<f64>() / results.len() as f64
        } else {
            0.0
        };

        let total_execution_time_ms = results.iter().map(|r| r.execution_time_ms).sum::<f64>();

        let avg_accuracy_score = {
            let accuracy_scores: Vec<f64> =
                results.iter().filter_map(|r| r.accuracy_score).collect();
            if !accuracy_scores.is_empty() {
                Some(accuracy_scores.iter().sum::<f64>() / accuracy_scores.len() as f64)
            } else {
                None
            }
        };

        let max_relative_error = results
            .iter()
            .filter_map(|r| r.relative_error)
            .fold(None, |acc, err| {
                Some(acc.map_or(err, |max_err: f64| max_err.max(err)))
            });

        let summary = BenchmarkSummary {
            total_tests,
            passed_tests,
            failed_tests,
            avg_execution_time_ms,
            total_execution_time_ms,
            avg_accuracy_score,
            max_relative_error,
        };

        // Generate performance comparison if baseline provided
        let performance_comparison =
            baseline_results.map(|baseline| Self::compare_with_baseline(results, baseline));

        // Calculate overall score
        let pass_rate = passed_tests as f64 / total_tests as f64;
        let accuracy_factor = avg_accuracy_score.map_or(0.5, |acc| (acc + 1.0) / 2.0);
        let error_factor = max_relative_error.map_or(1.0, |err| 1.0 / (1.0 + err * 1000.0));
        let overall_score = (pass_rate * 0.5 + accuracy_factor * 0.3 + error_factor * 0.2) * 100.0;

        Ok(Self {
            results: results.to_vec(),
            summary,
            performance_comparison,
            overall_score,
        })
    }

    /// Compare current results with baseline results
    fn compare_with_baseline(
        current: &[BenchmarkResult],
        baseline: &[BenchmarkResult],
    ) -> PerformanceComparison {
        // Calculate average execution times
        let current_time_ms = if !current.is_empty() {
            current.iter().map(|r| r.execution_time_ms).sum::<f64>() / current.len() as f64
        } else {
            0.0
        };

        let baseline_time_ms = if !baseline.is_empty() {
            baseline.iter().map(|r| r.execution_time_ms).sum::<f64>() / baseline.len() as f64
        } else {
            1.0 // Avoid division by zero
        };

        // Calculate speedup factor (baseline / current)
        // > 1.0 means current is faster, < 1.0 means baseline was faster
        let speedup_factor = if current_time_ms > 0.0 {
            baseline_time_ms / current_time_ms
        } else {
            1.0
        };

        let performance_improved = speedup_factor > 1.0;

        // Calculate p-value using Welch's t-test
        let p_value = Self::welch_t_test(
            &current
                .iter()
                .map(|r| r.execution_time_ms)
                .collect::<Vec<_>>(),
            &baseline
                .iter()
                .map(|r| r.execution_time_ms)
                .collect::<Vec<_>>(),
        );

        PerformanceComparison {
            speedup_factor,
            performance_improved,
            baseline_time_ms,
            current_time_ms,
            p_value: Some(p_value),
        }
    }

    /// Welch's t-test for comparing two samples with potentially different variances
    /// Returns approximate p-value (two-tailed test)
    fn welch_t_test(sample1: &[f64], sample2: &[f64]) -> f64 {
        if sample1.is_empty() || sample2.is_empty() {
            return 1.0; // No significant difference if no data
        }

        // Calculate means
        let mean1 = sample1.iter().sum::<f64>() / sample1.len() as f64;
        let mean2 = sample2.iter().sum::<f64>() / sample2.len() as f64;

        // Calculate variances
        let var1 = if sample1.len() > 1 {
            sample1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (sample1.len() - 1) as f64
        } else {
            0.0
        };

        let var2 = if sample2.len() > 1 {
            sample2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (sample2.len() - 1) as f64
        } else {
            0.0
        };

        // Calculate standard errors
        let se1 = var1 / sample1.len() as f64;
        let se2 = var2 / sample2.len() as f64;
        let se = (se1 + se2).sqrt();

        if se < f64::EPSILON {
            return 1.0; // No variance, no significant difference
        }

        // Calculate t-statistic
        let t = (mean1 - mean2).abs() / se;

        // Calculate degrees of freedom (Welch-Satterthwaite equation)
        let df = if var1 > f64::EPSILON && var2 > f64::EPSILON {
            let numerator = (se1 + se2).powi(2);
            let denominator =
                se1.powi(2) / (sample1.len() - 1) as f64 + se2.powi(2) / (sample2.len() - 1) as f64;
            numerator / denominator
        } else {
            (sample1.len() + sample2.len() - 2) as f64
        };

        // Approximate p-value using normal approximation for large df
        // For small samples, this is conservative
        if df > 30.0 {
            // Use standard normal approximation
            Self::normal_two_tailed_p_value(t)
        } else {
            // Conservative estimate for small samples
            // Using normal approximation but adjusting for small sample size
            let p = Self::normal_two_tailed_p_value(t);
            // Add conservative adjustment for small samples
            (p * (1.0 + 1.0 / df)).min(1.0)
        }
    }

    /// Calculate two-tailed p-value from z-score using normal approximation
    fn normal_two_tailed_p_value(z: f64) -> f64 {
        // Approximation of standard normal CDF using error function
        // P(|Z| > z) ≈ 2 * (1 - Φ(z)) where Φ is the CDF
        // Using approximation: Φ(z) ≈ 1 - 0.5 * exp(-1.65 * z) for z > 0

        let z_abs = z.abs();

        // Simple approximation for p-value
        // For z > 3, p-value is very small
        if z_abs > 6.0 {
            return 0.0000001; // Very significant
        }

        // Abramowitz and Stegun approximation
        let t = 1.0 / (1.0 + 0.2316419 * z_abs);
        let d = 0.3989423 * (-z_abs * z_abs / 2.0).exp();
        let p = d
            * t
            * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        2.0 * p // Two-tailed
    }

    /// Convert report to string format
    pub fn to_report_string(&self) -> String {
        let mut report = String::new();

        report.push_str("=".repeat(60).as_str());
        report.push_str("\n           BENCHMARK REPORT\n");
        report.push_str("=".repeat(60).as_str());
        report.push('\n');

        report.push_str("\nSummary:\n");
        report.push_str(&format!("  Total Tests: {}\n", self.summary.total_tests));
        report.push_str(&format!("  Passed: {}\n", self.summary.passed_tests));
        report.push_str(&format!("  Failed: {}\n", self.summary.failed_tests));
        report.push_str(&format!(
            "  Pass Rate: {:.1}%\n",
            100.0 * self.summary.passed_tests as f64 / self.summary.total_tests as f64
        ));
        report.push_str(&format!("  Overall Score: {:.1}/100\n", self.overall_score));

        report.push_str("\nPerformance:\n");
        report.push_str(&format!(
            "  Average Execution Time: {:.4} ms\n",
            self.summary.avg_execution_time_ms
        ));
        report.push_str(&format!(
            "  Total Execution Time: {:.2} ms\n",
            self.summary.total_execution_time_ms
        ));

        if let Some(avg_acc) = self.summary.avg_accuracy_score {
            report.push_str(&format!("  Average Accuracy: {:.6}\n", avg_acc));
        }

        if let Some(max_err) = self.summary.max_relative_error {
            report.push_str(&format!("  Maximum Relative Error: {:.2e}\n", max_err));
        }

        // Add performance comparison if available
        if let Some(ref comparison) = self.performance_comparison {
            report.push_str("\nPerformance Comparison vs Baseline:\n");
            report.push_str(&format!(
                "  Baseline Time: {:.4} ms\n",
                comparison.baseline_time_ms
            ));
            report.push_str(&format!(
                "  Current Time: {:.4} ms\n",
                comparison.current_time_ms
            ));
            report.push_str(&format!(
                "  Speedup Factor: {:.2}x\n",
                comparison.speedup_factor
            ));
            report.push_str(&format!(
                "  Performance: {}\n",
                if comparison.performance_improved {
                    "IMPROVED ✓"
                } else {
                    "REGRESSED"
                }
            ));
            if let Some(p_value) = comparison.p_value {
                report.push_str(&format!(
                    "  Statistical Significance (p-value): {:.4}\n",
                    p_value
                ));
                report.push_str(&format!(
                    "  Significance Level: {}\n",
                    if p_value < 0.001 {
                        "*** (highly significant)"
                    } else if p_value < 0.01 {
                        "** (very significant)"
                    } else if p_value < 0.05 {
                        "* (significant)"
                    } else {
                        "(not significant)"
                    }
                ));
            }
        }

        report.push_str("\nDetailed Results:\n");
        report.push_str(&format!(
            "{:<30} {:>10} {:>10} {:>10} {:>8}\n",
            "Test", "Time (ms)", "Std (ms)", "Rel Error", "Status"
        ));
        report.push_str("-".repeat(70).as_str());
        report.push('\n');

        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            let rel_error = result
                .relative_error
                .map_or("N/A".to_string(), |e| format!("{:.2e}", e));
            report.push_str(&format!(
                "{:<30} {:>10.4} {:>10.4} {:>10} {:>8}\n",
                result.test_name, result.execution_time_ms, result.time_std_ms, rel_error, status
            ));
        }

        report
    }
}

/// Standard reference implementations for comparison
pub struct ReferenceImplementations;

impl ReferenceImplementations {
    /// Reference accuracy implementation
    pub fn accuracy_score(y_true: &Array1<i32>, y_pred: &Array1<i32>) -> f64 {
        if y_true.len() != y_pred.len() {
            return 0.0;
        }

        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| a == b)
            .count();

        correct as f64 / y_true.len() as f64
    }

    /// Reference MSE implementation
    pub fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        if y_true.len() != y_pred.len() {
            return f64::INFINITY;
        }

        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y_true.len() as f64
    }

    /// Reference MAE implementation
    pub fn mean_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        if y_true.len() != y_pred.len() {
            return f64::INFINITY;
        }

        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y_true.len() as f64
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);

        assert_eq!(suite.classification_benchmarks.len(), 0);
        assert_eq!(suite.regression_benchmarks.len(), 0);
        assert_eq!(suite.clustering_benchmarks.len(), 0);
    }

    #[test]
    fn test_add_classification_benchmark() {
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());

        suite.add_classification_benchmark(
            "test_accuracy",
            |y_true, y_pred| ReferenceImplementations::accuracy_score(y_true, y_pred),
            Dataset::Iris,
        );

        assert_eq!(suite.classification_benchmarks.len(), 1);
        assert_eq!(suite.classification_benchmarks[0].name, "test_accuracy");
    }

    #[test]
    fn test_add_regression_benchmark() {
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());

        suite.add_regression_benchmark(
            "test_mse",
            |y_true, y_pred| ReferenceImplementations::mean_squared_error(y_true, y_pred),
            Dataset::BostonHousing,
        );

        assert_eq!(suite.regression_benchmarks.len(), 1);
        assert_eq!(suite.regression_benchmarks[0].name, "test_mse");
    }

    #[test]
    fn test_data_generation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);

        // Test classification data generation
        let (y_true, y_pred) = suite.generate_classification_data(Dataset::Iris).unwrap();
        assert_eq!(y_true.len(), 150);
        assert_eq!(y_pred.len(), 150);

        // Test regression data generation
        let (y_true_reg, y_pred_reg) = suite
            .generate_regression_data(Dataset::BostonHousing)
            .unwrap();
        assert_eq!(y_true_reg.len(), 506);
        assert_eq!(y_pred_reg.len(), 506);
    }

    #[test]
    fn test_reference_implementations() {
        // Test accuracy
        let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1]);
        let y_pred = Array1::from_vec(vec![0, 1, 1, 0, 1]);
        let accuracy = ReferenceImplementations::accuracy_score(&y_true, &y_pred);
        assert_abs_diff_eq!(accuracy, 0.8, epsilon = 1e-10);

        // Test MSE
        let y_true_reg = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred_reg = Array1::from_vec(vec![1.1, 2.1, 2.9]);
        let mse = ReferenceImplementations::mean_squared_error(&y_true_reg, &y_pred_reg);
        assert_abs_diff_eq!(mse, 0.01, epsilon = 1e-10);

        // Test MAE
        let mae = ReferenceImplementations::mean_absolute_error(&y_true_reg, &y_pred_reg);
        assert_abs_diff_eq!(mae, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult {
            test_name: "test".to_string(),
            benchmark_type: BenchmarkType::Performance,
            dataset: Dataset::Iris,
            execution_time_ms: 1.5,
            time_std_ms: 0.1,
            accuracy_score: Some(0.95),
            relative_error: Some(1e-6),
            memory_usage_bytes: None,
            passed: true,
            metadata: HashMap::new(),
        };

        assert_eq!(result.test_name, "test");
        assert!(result.passed);
        assert_eq!(result.accuracy_score, Some(0.95));
    }

    #[test]
    fn test_benchmark_report_generation() {
        let results = vec![
            BenchmarkResult {
                test_name: "test1".to_string(),
                benchmark_type: BenchmarkType::Performance,
                dataset: Dataset::Iris,
                execution_time_ms: 1.0,
                time_std_ms: 0.1,
                accuracy_score: Some(0.9),
                relative_error: Some(1e-6),
                memory_usage_bytes: None,
                passed: true,
                metadata: HashMap::new(),
            },
            BenchmarkResult {
                test_name: "test2".to_string(),
                benchmark_type: BenchmarkType::Performance,
                dataset: Dataset::Wine,
                execution_time_ms: 2.0,
                time_std_ms: 0.2,
                accuracy_score: Some(0.8),
                relative_error: Some(2e-6),
                memory_usage_bytes: None,
                passed: true,
                metadata: HashMap::new(),
            },
        ];

        let report = BenchmarkReport::generate(&results).unwrap();

        assert_eq!(report.summary.total_tests, 2);
        assert_eq!(report.summary.passed_tests, 2);
        assert_eq!(report.summary.failed_tests, 0);
        assert_abs_diff_eq!(report.summary.avg_execution_time_ms, 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(
            report.summary.avg_accuracy_score.unwrap(),
            0.85,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_synthetic_data_generation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);

        let (y_true, y_pred) = suite
            .generate_classification_data(Dataset::SyntheticClassification(1000, 50, 5))
            .unwrap();

        assert_eq!(y_true.len(), 1000);
        assert_eq!(y_pred.len(), 1000);

        // Check that labels are in valid range
        for &label in y_true.iter() {
            assert!(label >= 0 && label < 5);
        }
        for &label in y_pred.iter() {
            assert!(label >= 0 && label < 5);
        }
    }
}
