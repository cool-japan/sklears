//! Advanced Benchmarking Suite with Performance Regression Detection
//!
//! This module provides sophisticated benchmarking capabilities including
//! statistical analysis, regression detection, and performance tracking over time.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Advanced benchmark runner with regression detection
///
/// Tracks performance metrics over time and detects statistical anomalies
/// and performance regressions automatically.
#[derive(Debug)]
pub struct AdvancedBenchmarkRunner {
    /// Configuration for benchmarking
    pub config: BenchmarkConfig,
    /// Historical benchmark results
    pub history: BenchmarkHistory,
    /// Statistical analyzer for detecting regressions
    pub analyzer: RegressionAnalyzer,
    /// Performance baselines
    pub baselines: HashMap<String, PerformanceBaseline>,
}

/// Configuration for benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Confidence level for statistical tests (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Maximum acceptable performance degradation (as fraction, e.g., 0.10 for 10%)
    pub max_degradation_threshold: f64,
    /// Enable outlier detection and removal
    pub enable_outlier_detection: bool,
    /// Sample size for statistical analysis
    pub sample_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            confidence_level: 0.95,
            max_degradation_threshold: 0.10, // 10% degradation threshold
            enable_outlier_detection: true,
            sample_size: 50,
        }
    }
}

/// Historical benchmark results with time series data
#[derive(Debug, Clone)]
pub struct BenchmarkHistory {
    /// Benchmark results indexed by benchmark name
    pub results: HashMap<String, VecDeque<BenchmarkResult>>,
    /// Maximum history length to keep
    pub max_history_length: usize,
}

impl BenchmarkHistory {
    /// Create a new benchmark history with specified capacity
    pub fn new(max_history_length: usize) -> Self {
        Self {
            results: HashMap::new(),
            max_history_length,
        }
    }

    /// Add a benchmark result to history
    pub fn add_result(&mut self, name: String, result: BenchmarkResult) {
        let entry = self.results.entry(name).or_default();

        entry.push_back(result);

        // Maintain maximum history length
        while entry.len() > self.max_history_length {
            entry.pop_front();
        }
    }

    /// Get historical results for a benchmark
    pub fn get_history(&self, name: &str) -> Option<&VecDeque<BenchmarkResult>> {
        self.results.get(name)
    }

    /// Get statistical summary of historical performance
    pub fn get_summary(&self, name: &str) -> Option<HistoricalSummary> {
        let history = self.get_history(name)?;

        if history.is_empty() {
            return None;
        }

        let durations: Vec<f64> = history
            .iter()
            .map(|r| r.median_duration.as_secs_f64())
            .collect();

        let mean = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance =
            durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / durations.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted = durations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Some(HistoricalSummary {
            mean_duration: Duration::from_secs_f64(mean),
            std_deviation: std_dev,
            min_duration: Duration::from_secs_f64(sorted[0]),
            max_duration: Duration::from_secs_f64(*sorted.last().expect("last should succeed")),
            median_duration: Duration::from_secs_f64(sorted[sorted.len() / 2]),
            sample_count: durations.len(),
        })
    }
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Timestamp of execution
    pub timestamp: SystemTime,
    /// All measured durations
    pub durations: Vec<Duration>,
    /// Median duration
    pub median_duration: Duration,
    /// Mean duration
    pub mean_duration: Duration,
    /// Standard deviation
    pub std_deviation: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory usage statistics
    pub memory_stats: Option<MemoryStats>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Average memory usage in bytes
    pub average_bytes: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
}

/// Historical summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalSummary {
    /// Mean duration across all historical runs
    pub mean_duration: Duration,
    /// Standard deviation
    pub std_deviation: f64,
    /// Minimum observed duration
    pub min_duration: Duration,
    /// Maximum observed duration
    pub max_duration: Duration,
    /// Median duration
    pub median_duration: Duration,
    /// Number of samples
    pub sample_count: usize,
}

/// Performance baseline for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline name
    pub name: String,
    /// Baseline duration
    pub baseline_duration: Duration,
    /// Acceptable variance (as fraction)
    pub acceptable_variance: f64,
    /// When the baseline was established
    pub established_at: SystemTime,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
}

/// Regression analyzer for detecting performance issues
#[derive(Debug, Clone)]
pub struct RegressionAnalyzer {
    /// Configuration
    pub config: AnalyzerConfig,
    /// Detected regressions
    pub detected_regressions: Vec<RegressionReport>,
}

/// Configuration for regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Minimum sample size for analysis
    pub min_sample_size: usize,
    /// Sensitivity for detecting changes (lower = more sensitive)
    pub sensitivity: f64,
    /// Use statistical hypothesis testing
    pub use_hypothesis_testing: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 10,
            sensitivity: 0.05, // 5% significance level
            use_hypothesis_testing: true,
        }
    }
}

/// Report of a detected regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Benchmark name
    pub benchmark_name: String,
    /// Severity of the regression
    pub severity: RegressionSeverity,
    /// Performance degradation percentage
    pub degradation_percent: f64,
    /// Current performance
    pub current_performance: Duration,
    /// Expected performance based on baseline
    pub expected_performance: Duration,
    /// Statistical confidence of detection
    pub confidence: f64,
    /// Additional details
    pub details: String,
    /// Detected at
    pub detected_at: SystemTime,
}

/// Severity level of performance regression
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Minor regression (<5% degradation)
    Minor,
    /// Moderate regression (5-15% degradation)
    Moderate,
    /// Major regression (15-30% degradation)
    Major,
    /// Critical regression (>30% degradation)
    Critical,
}

impl AdvancedBenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            history: BenchmarkHistory::new(100),
            analyzer: RegressionAnalyzer {
                config: AnalyzerConfig::default(),
                detected_regressions: Vec::new(),
            },
            baselines: HashMap::new(),
        }
    }

    /// Create a runner with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            config,
            history: BenchmarkHistory::new(100),
            analyzer: RegressionAnalyzer {
                config: AnalyzerConfig::default(),
                detected_regressions: Vec::new(),
            },
            baselines: HashMap::new(),
        }
    }

    /// Run a benchmark and analyze results
    pub fn run_benchmark<F>(&mut self, name: &str, mut benchmark_fn: F) -> Result<BenchmarkResult>
    where
        F: FnMut(),
    {
        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            benchmark_fn();
        }

        // Measurement phase
        let mut durations = Vec::new();
        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            benchmark_fn();
            durations.push(start.elapsed());
        }

        // Remove outliers if enabled
        if self.config.enable_outlier_detection {
            durations = self.remove_outliers(durations);
        }

        // Calculate statistics
        let mut sorted_durations = durations.clone();
        sorted_durations.sort();

        let median = sorted_durations[sorted_durations.len() / 2];
        let mean = Duration::from_secs_f64(
            durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / durations.len() as f64,
        );

        let variance = durations
            .iter()
            .map(|d| (d.as_secs_f64() - mean.as_secs_f64()).powi(2))
            .sum::<f64>()
            / durations.len() as f64;
        let std_dev = variance.sqrt();

        let throughput = 1.0 / mean.as_secs_f64(); // ops/sec

        let result = BenchmarkResult {
            name: name.to_string(),
            timestamp: SystemTime::now(),
            durations,
            median_duration: median,
            mean_duration: mean,
            std_deviation: std_dev,
            throughput,
            memory_stats: None, // Could be populated with actual memory tracking
        };

        // Add to history
        self.history.add_result(name.to_string(), result.clone());

        // Check for regressions
        self.check_for_regression(name, &result)?;

        Ok(result)
    }

    /// Remove statistical outliers from measurements
    fn remove_outliers(&self, mut durations: Vec<Duration>) -> Vec<Duration> {
        if durations.len() < 10 {
            return durations; // Not enough data for outlier detection
        }

        // Convert to f64 for calculations
        let values: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();

        // Calculate IQR (Interquartile Range) method
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = sorted_values.len() / 4;
        let q3_idx = (sorted_values.len() * 3) / 4;

        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        // Filter outliers
        durations.retain(|d| {
            let val = d.as_secs_f64();
            val >= lower_bound && val <= upper_bound
        });

        durations
    }

    /// Check for performance regression
    fn check_for_regression(&mut self, name: &str, current: &BenchmarkResult) -> Result<()> {
        // Get baseline if it exists
        if let Some(baseline) = self.baselines.get(name) {
            let degradation = (current.median_duration.as_secs_f64()
                - baseline.baseline_duration.as_secs_f64())
                / baseline.baseline_duration.as_secs_f64();

            if degradation > self.config.max_degradation_threshold {
                let severity = match degradation {
                    d if d < 0.05 => RegressionSeverity::Minor,
                    d if d < 0.15 => RegressionSeverity::Moderate,
                    d if d < 0.30 => RegressionSeverity::Major,
                    _ => RegressionSeverity::Critical,
                };

                let report = RegressionReport {
                    benchmark_name: name.to_string(),
                    severity,
                    degradation_percent: degradation * 100.0,
                    current_performance: current.median_duration,
                    expected_performance: baseline.baseline_duration,
                    confidence: self.config.confidence_level,
                    details: format!(
                        "Performance degraded by {:.2}% compared to baseline",
                        degradation * 100.0
                    ),
                    detected_at: SystemTime::now(),
                };

                self.analyzer.detected_regressions.push(report);
            }
        }

        Ok(())
    }

    /// Set a performance baseline
    pub fn set_baseline(&mut self, name: String, duration: Duration) {
        let baseline = PerformanceBaseline {
            name: name.clone(),
            baseline_duration: duration,
            acceptable_variance: self.config.max_degradation_threshold,
            established_at: SystemTime::now(),
            git_commit: None,
        };

        self.baselines.insert(name, baseline);
    }

    /// Get all detected regressions
    pub fn get_regressions(&self) -> &[RegressionReport] {
        &self.analyzer.detected_regressions
    }

    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> BenchmarkReport {
        let mut benchmark_summaries = HashMap::new();

        for name in self.history.results.keys() {
            if let Some(summary) = self.history.get_summary(name) {
                benchmark_summaries.insert(name.clone(), summary);
            }
        }

        BenchmarkReport {
            total_benchmarks: self.history.results.len(),
            regressions_detected: self.analyzer.detected_regressions.len(),
            benchmark_summaries,
            regressions: self.analyzer.detected_regressions.clone(),
            generated_at: SystemTime::now(),
        }
    }
}

impl Default for AdvancedBenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Total number of benchmarks
    pub total_benchmarks: usize,
    /// Number of regressions detected
    pub regressions_detected: usize,
    /// Summary statistics for each benchmark
    pub benchmark_summaries: HashMap<String, HistoricalSummary>,
    /// All detected regressions
    pub regressions: Vec<RegressionReport>,
    /// When the report was generated
    pub generated_at: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner_creation() {
        let runner = AdvancedBenchmarkRunner::new();
        assert_eq!(runner.config.warmup_iterations, 10);
        assert_eq!(runner.config.measurement_iterations, 100);
    }

    #[test]
    fn test_custom_config() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 50,
            confidence_level: 0.99,
            max_degradation_threshold: 0.05,
            enable_outlier_detection: false,
            sample_size: 30,
        };

        let runner = AdvancedBenchmarkRunner::with_config(config);
        assert_eq!(runner.config.warmup_iterations, 5);
        assert_eq!(runner.config.measurement_iterations, 50);
    }

    #[test]
    fn test_simple_benchmark() {
        let mut runner = AdvancedBenchmarkRunner::new();

        let result = runner
            .run_benchmark("test_benchmark", || {
                // Simulate work
                let _sum: u64 = (0..1000).sum();
            })
            .expect("expected valid value");

        assert_eq!(result.name, "test_benchmark");
        assert!(result.median_duration > Duration::from_nanos(0));
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_baseline_setting() {
        let mut runner = AdvancedBenchmarkRunner::new();

        runner.set_baseline("test".to_string(), Duration::from_millis(10));

        assert!(runner.baselines.contains_key("test"));
        assert_eq!(
            runner
                .baselines
                .get("test")
                .expect("key should exist")
                .baseline_duration,
            Duration::from_millis(10)
        );
    }

    #[test]
    fn test_regression_detection() {
        let mut runner = AdvancedBenchmarkRunner::new();

        // Set a fast baseline
        runner.set_baseline("test".to_string(), Duration::from_micros(100));

        // Run a much slower benchmark
        let _result = runner
            .run_benchmark("test", || {
                std::thread::sleep(Duration::from_micros(200));
            })
            .expect("expected valid value");

        // Should detect a regression
        let regressions = runner.get_regressions();
        assert!(!regressions.is_empty());
    }

    #[test]
    fn test_history_tracking() {
        let mut runner = AdvancedBenchmarkRunner::new();

        runner
            .run_benchmark("test", || {
                let _x = 1 + 1;
            })
            .expect("expected valid value");

        runner
            .run_benchmark("test", || {
                let _x = 1 + 1;
            })
            .expect("expected valid value");

        let history = runner
            .history
            .get_history("test")
            .expect("get_history should succeed");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_historical_summary() {
        let mut runner = AdvancedBenchmarkRunner::new();

        for _ in 0..5 {
            runner
                .run_benchmark("test", || {
                    // Use black_box to prevent compiler optimization
                    // and make the computation take measurable time
                    let mut sum = 0u64;
                    for i in 0..100 {
                        sum = std::hint::black_box(sum.wrapping_add(i));
                    }
                    std::hint::black_box(sum);
                })
                .expect("expected valid value");
        }

        let summary = runner
            .history
            .get_summary("test")
            .expect("get_summary should succeed");
        assert_eq!(summary.sample_count, 5);
        assert!(summary.mean_duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_report_generation() {
        let mut runner = AdvancedBenchmarkRunner::new();

        runner
            .run_benchmark("bench1", || {
                let x = std::hint::black_box(1 + 1);
                std::hint::black_box(x);
            })
            .expect("expected valid value");

        runner
            .run_benchmark("bench2", || {
                let y = std::hint::black_box(2 + 2);
                std::hint::black_box(y);
            })
            .expect("expected valid value");

        let report = runner.generate_report();
        assert_eq!(report.total_benchmarks, 2);
    }

    #[test]
    fn test_outlier_removal() {
        let runner = AdvancedBenchmarkRunner::new();

        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(11),
            Duration::from_millis(10),
            Duration::from_millis(100), // Outlier
            Duration::from_millis(10),
            Duration::from_millis(11),
            Duration::from_millis(10),
            Duration::from_millis(11),
            Duration::from_millis(10),
            Duration::from_millis(11),
        ];

        let filtered = runner.remove_outliers(durations);
        assert!(filtered.len() < 10); // Outlier should be removed
    }

    #[test]
    fn test_regression_severity() {
        use RegressionSeverity::*;

        assert!(Minor < Moderate);
        assert!(Moderate < Major);
        assert!(Major < Critical);
    }
}
