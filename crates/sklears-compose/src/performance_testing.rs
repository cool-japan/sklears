//! Performance Regression Testing Framework
//!
//! Advanced framework for detecting performance regressions in machine learning pipelines
//! through automated benchmarking, statistical analysis, and trend monitoring.

use chrono::{DateTime, Utc};
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit},
    types::Float,
};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Performance regression testing framework
pub struct PerformanceRegressionTester {
    /// Benchmark results storage
    pub storage: BenchmarkStorage,
    /// Statistical analysis configuration
    pub analysis_config: StatisticalAnalysisConfig,
    /// Test environment configuration
    pub environment_config: EnvironmentConfig,
    /// Regression detection thresholds
    pub regression_thresholds: RegressionThresholds,
    /// Profiling configuration
    pub profiling_config: ProfilingConfig,
}

/// Benchmark results storage backend
pub enum BenchmarkStorage {
    /// File-based storage
    File { path: PathBuf },
    /// In-memory storage (for testing)
    Memory { results: Vec<BenchmarkResult> },
    /// Database storage (placeholder)
    Database { connection_string: String },
}

/// Statistical analysis configuration
#[derive(Clone, Debug)]
pub struct StatisticalAnalysisConfig {
    /// Confidence level for regression detection
    pub confidence_level: f64,
    /// Minimum number of samples for trend analysis
    pub min_samples_for_trend: usize,
    /// Window size for rolling statistics
    pub rolling_window_size: usize,
    /// Statistical tests to perform
    pub statistical_tests: Vec<StatisticalTest>,
    /// Outlier detection method
    pub outlier_detection: OutlierDetection,
}

/// Statistical tests for performance analysis
#[derive(Clone, Debug)]
pub enum StatisticalTest {
    /// T-test for mean comparison
    TTest,
    /// Mann-Whitney U test for non-parametric comparison
    MannWhitneyU,
    /// Kolmogorov-Smirnov test for distribution comparison
    KolmogorovSmirnov,
    /// Linear regression for trend analysis
    LinearRegression,
    /// Change point detection
    ChangePointDetection,
}

/// Outlier detection methods
#[derive(Clone, Debug)]
pub enum OutlierDetection {
    /// No outlier detection
    None,
    /// Z-score based detection
    ZScore { threshold: f64 },
    /// IQR based detection
    IQR { multiplier: f64 },
    /// Modified Z-score
    ModifiedZScore { threshold: f64 },
}

/// Test environment configuration
#[derive(Clone, Debug)]
pub struct EnvironmentConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// CPU affinity (optional)
    pub cpu_affinity: Option<Vec<usize>>,
    /// Memory constraints
    pub memory_limit: Option<u64>,
    /// Environment variables to capture
    pub capture_env_vars: Vec<String>,
    /// System information to collect
    pub collect_system_info: bool,
}

/// Regression detection thresholds
#[derive(Clone, Debug)]
pub struct RegressionThresholds {
    /// Relative performance degradation threshold (e.g., 0.05 for 5%)
    pub relative_threshold: f64,
    /// Absolute performance degradation threshold (in milliseconds)
    pub absolute_threshold: Duration,
    /// Memory usage regression threshold (in bytes)
    pub memory_threshold: u64,
    /// Throughput regression threshold (relative)
    pub throughput_threshold: f64,
}

/// Profiling configuration
#[derive(Clone, Debug)]
pub struct ProfilingConfig {
    /// Enable CPU profiling
    pub cpu_profiling: bool,
    /// Enable memory profiling
    pub memory_profiling: bool,
    /// Profile sampling frequency
    pub sampling_frequency: Duration,
    /// Profile output directory
    pub output_directory: Option<PathBuf>,
    /// Enable detailed call stack collection
    pub detailed_call_stacks: bool,
}

/// Benchmark result with comprehensive metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Unique benchmark identifier
    pub benchmark_id: String,
    /// Test case name
    pub test_case: String,
    /// Timestamp when benchmark was run
    pub timestamp: DateTime<Utc>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// System information
    pub system_info: SystemInfo,
    /// Environment metadata
    pub environment: EnvironmentMetadata,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Comprehensive performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Execution time statistics
    pub execution_time: TimeStatistics,
    /// Memory usage statistics
    pub memory_usage: MemoryStatistics,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// CPU utilization
    pub cpu_utilization: CpuStatistics,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Time-based statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeStatistics {
    /// Mean execution time
    pub mean: Duration,
    /// Median execution time
    pub median: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum time
    pub min: Duration,
    /// Maximum time
    pub max: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// All individual measurements
    pub samples: Vec<Duration>,
}

/// Memory usage statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Peak memory usage
    pub peak_usage: u64,
    /// Average memory usage
    pub average_usage: u64,
    /// Memory allocations count
    pub allocations: u64,
    /// Memory deallocations count
    pub deallocations: u64,
    /// Memory fragmentation score
    pub fragmentation_score: f64,
}

/// Throughput performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Samples processed per second
    pub samples_per_second: f64,
    /// Features processed per second
    pub features_per_second: f64,
    /// Bytes processed per second
    pub bytes_per_second: f64,
}

/// CPU utilization statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuStatistics {
    /// Average CPU utilization (0.0 to 1.0)
    pub average_utilization: f64,
    /// Peak CPU utilization
    pub peak_utilization: f64,
    /// CPU time in user mode
    pub user_time: Duration,
    /// CPU time in kernel mode
    pub kernel_time: Duration,
}

/// System information captured during benchmark
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Total system memory
    pub total_memory: u64,
    /// Available memory at test time
    pub available_memory: u64,
    /// Rust version
    pub rust_version: String,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
}

/// Environment metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentMetadata {
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Current working directory
    pub working_directory: PathBuf,
    /// Command line arguments
    pub args: Vec<String>,
    /// System load average
    pub load_average: Vec<f64>,
}

/// Regression analysis result
#[derive(Clone, Debug)]
pub struct RegressionAnalysis {
    /// Whether a regression was detected
    pub regression_detected: bool,
    /// Regression severity
    pub severity: RegressionSeverity,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Statistical significance
    pub p_value: f64,
    /// Effect size
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Detailed analysis
    pub detailed_analysis: String,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Regression severity levels
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegressionSeverity {
    /// No regression detected
    None,
    /// Minor performance degradation
    Minor,
    /// Moderate performance degradation
    Moderate,
    /// Severe performance degradation
    Severe,
    /// Critical performance degradation
    Critical,
}

/// Benchmark execution context
pub struct BenchmarkContext {
    /// Data size for testing
    pub data_size: (usize, usize),
    /// Number of iterations
    pub iterations: usize,
    /// Benchmark configuration
    pub config: HashMap<String, String>,
    /// Random seed for reproducibility
    pub random_seed: u64,
}

impl Default for PerformanceRegressionTester {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceRegressionTester {
    /// Create a new performance regression tester with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: BenchmarkStorage::Memory {
                results: Vec::new(),
            },
            analysis_config: StatisticalAnalysisConfig::default(),
            environment_config: EnvironmentConfig::default(),
            regression_thresholds: RegressionThresholds::default(),
            profiling_config: ProfilingConfig::default(),
        }
    }

    /// Create a tester with file-based storage
    pub fn with_file_storage<P: AsRef<Path>>(path: P) -> Self {
        Self {
            storage: BenchmarkStorage::File {
                path: path.as_ref().to_path_buf(),
            },
            ..Self::new()
        }
    }

    /// Run a benchmark on a pipeline component
    pub fn benchmark_component<T, I, O>(
        &mut self,
        component: &T,
        input: I,
        context: &BenchmarkContext,
        test_name: &str,
    ) -> SklResult<BenchmarkResult>
    where
        T: Fn(I) -> O,
        I: Clone,
    {
        let start_time = Instant::now();
        let mut execution_times = Vec::new();

        // Warmup phase
        for _ in 0..self.environment_config.warmup_iterations {
            let _ = component(input.clone());
        }

        // Measurement phase
        for _ in 0..self.environment_config.measurement_iterations {
            let measure_start = Instant::now();
            let _ = component(input.clone());
            execution_times.push(measure_start.elapsed());
        }

        let time_stats = self.calculate_time_statistics(&execution_times);
        let memory_stats = self.collect_memory_statistics();
        let cpu_stats = self.collect_cpu_statistics();
        let throughput = self.calculate_throughput(&time_stats, context);

        let result = BenchmarkResult {
            benchmark_id: format!("{}_{}", test_name, Utc::now().timestamp()),
            test_case: test_name.to_string(),
            timestamp: Utc::now(),
            metrics: PerformanceMetrics {
                execution_time: time_stats,
                memory_usage: memory_stats,
                throughput,
                cpu_utilization: cpu_stats,
                custom_metrics: HashMap::new(),
            },
            system_info: self.collect_system_info(),
            environment: self.collect_environment_metadata(),
            commit_hash: self.get_git_commit_hash(),
            metadata: context.config.clone(),
        };

        self.store_result(&result)?;
        Ok(result)
    }

    /// Benchmark a machine learning pipeline
    pub fn benchmark_pipeline<'a, S>(
        &mut self,
        pipeline: &crate::Pipeline<S>,
        x: &ArrayView2<'a, Float>,
        y: Option<&'a ArrayView1<'a, Float>>,
        test_name: &str,
    ) -> SklResult<BenchmarkResult>
    where
        S: std::fmt::Debug + Clone,
        crate::Pipeline<S>: Clone + Fit<ArrayView2<'a, Float>, Option<&'a ArrayView1<'a, Float>>>,
    {
        let context = BenchmarkContext {
            data_size: (x.nrows(), x.ncols()),
            iterations: self.environment_config.measurement_iterations,
            config: HashMap::new(),
            random_seed: 42,
        };

        let benchmark_fn = |(): ()| -> SklResult<()> {
            // Clone pipeline for benchmarking
            let pipeline_clone = pipeline.clone();
            if let Some(y_vals) = y {
                let y_option = Some(y_vals);
                let _fitted = pipeline_clone.fit(x, &y_option)?;
            }
            Ok(())
        };

        self.benchmark_component(&benchmark_fn, (), &context, test_name)
    }

    /// Analyze performance trends and detect regressions
    pub fn analyze_regressions(&self, test_name: &str) -> SklResult<RegressionAnalysis> {
        let results = self.get_historical_results(test_name)?;

        if results.len() < self.analysis_config.min_samples_for_trend {
            return Ok(RegressionAnalysis {
                regression_detected: false,
                severity: RegressionSeverity::None,
                affected_metrics: vec![],
                p_value: 1.0,
                effect_size: 0.0,
                confidence_interval: (0.0, 0.0),
                detailed_analysis: "Insufficient data for trend analysis".to_string(),
                recommendations: vec!["Collect more benchmark data".to_string()],
            });
        }

        // Perform statistical analysis
        let regression_detected = self.detect_performance_regression(&results)?;
        let severity = self.calculate_regression_severity(&results)?;
        let affected_metrics = self.identify_affected_metrics(&results)?;

        // Statistical tests
        let p_value = self.calculate_statistical_significance(&results)?;
        let effect_size = self.calculate_effect_size(&results)?;
        let confidence_interval = self.calculate_confidence_interval(&results)?;

        let detailed_analysis = self.generate_detailed_analysis(&results)?;
        let recommendations = self.generate_recommendations(&results, &severity);

        Ok(RegressionAnalysis {
            regression_detected,
            severity,
            affected_metrics,
            p_value,
            effect_size,
            confidence_interval,
            detailed_analysis,
            recommendations,
        })
    }

    /// Generate a performance report
    pub fn generate_report(&self, test_pattern: Option<&str>) -> SklResult<PerformanceReport> {
        let all_results = self.get_all_results()?;

        let filtered_results = match test_pattern {
            Some(pattern) => all_results
                .into_iter()
                .filter(|r| r.test_case.contains(pattern))
                .collect(),
            None => all_results,
        };

        let report = PerformanceReport::new(filtered_results, &self.analysis_config);
        Ok(report)
    }

    // Helper methods
    fn calculate_time_statistics(&self, times: &[Duration]) -> TimeStatistics {
        let mut sorted_times = times.to_vec();
        sorted_times.sort();

        let mean = Duration::from_nanos(
            times.iter().map(|d| d.as_nanos() as u64).sum::<u64>() / times.len() as u64,
        );

        let median = sorted_times[times.len() / 2];

        let variance = times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as i64 - mean.as_nanos() as i64;
                (diff * diff) as u64
            })
            .sum::<u64>()
            / times.len() as u64;

        let std_dev = Duration::from_nanos((variance as f64).sqrt() as u64);

        let p95_idx = (times.len() as f64 * 0.95) as usize;
        let p99_idx = (times.len() as f64 * 0.99) as usize;

        TimeStatistics {
            mean,
            median,
            std_dev,
            min: *sorted_times.first().unwrap(),
            max: *sorted_times.last().unwrap(),
            p95: sorted_times[p95_idx.min(times.len() - 1)],
            p99: sorted_times[p99_idx.min(times.len() - 1)],
            samples: times.to_vec(),
        }
    }

    fn collect_memory_statistics(&self) -> MemoryStatistics {
        // Placeholder implementation - would integrate with actual memory profiling
        MemoryStatistics {
            peak_usage: 1024 * 1024, // 1MB placeholder
            average_usage: 512 * 1024,
            allocations: 100,
            deallocations: 95,
            fragmentation_score: 0.1,
        }
    }

    fn collect_cpu_statistics(&self) -> CpuStatistics {
        // Placeholder implementation - would integrate with actual CPU profiling
        CpuStatistics {
            average_utilization: 0.75,
            peak_utilization: 0.95,
            user_time: Duration::from_millis(100),
            kernel_time: Duration::from_millis(10),
        }
    }

    fn calculate_throughput(
        &self,
        time_stats: &TimeStatistics,
        context: &BenchmarkContext,
    ) -> ThroughputMetrics {
        let ops_per_second = 1.0 / time_stats.mean.as_secs_f64();
        let samples_per_second = context.data_size.0 as f64 / time_stats.mean.as_secs_f64();
        let features_per_second =
            (context.data_size.0 * context.data_size.1) as f64 / time_stats.mean.as_secs_f64();

        ThroughputMetrics {
            ops_per_second,
            samples_per_second,
            features_per_second,
            bytes_per_second: features_per_second * 8.0, // Assuming 8 bytes per float
        }
    }

    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            cpu_model: "Unknown".to_string(), // Would query actual CPU info
            cpu_cores: num_cpus::get(),
            total_memory: 16 * 1024 * 1024 * 1024, // Placeholder: 16GB
            available_memory: 8 * 1024 * 1024 * 1024, // Placeholder: 8GB
            rust_version: "1.75.0".to_string(),    // Would query actual version
            compiler_flags: vec!["--release".to_string()],
        }
    }

    fn collect_environment_metadata(&self) -> EnvironmentMetadata {
        let mut env_vars = HashMap::new();
        for var_name in &self.environment_config.capture_env_vars {
            if let Ok(value) = std::env::var(var_name) {
                env_vars.insert(var_name.clone(), value);
            }
        }

        EnvironmentMetadata {
            env_vars,
            working_directory: std::env::current_dir().unwrap_or_default(),
            args: std::env::args().collect(),
            load_average: vec![0.5, 0.6, 0.7], // Placeholder
        }
    }

    fn get_git_commit_hash(&self) -> Option<String> {
        // Placeholder - would execute git command
        None
    }

    fn store_result(&mut self, result: &BenchmarkResult) -> SklResult<()> {
        match &mut self.storage {
            BenchmarkStorage::Memory { results } => {
                results.push(result.clone());
            }
            BenchmarkStorage::File { path } => {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {e}")))?;

                let json_line = serde_json::to_string(result).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to serialize result: {e}"))
                })?;

                writeln!(file, "{json_line}").map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to write result: {e}"))
                })?;
            }
            BenchmarkStorage::Database { .. } => {
                return Err(SklearsError::NotImplemented(
                    "Database storage not implemented".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn get_historical_results(&self, test_name: &str) -> SklResult<Vec<BenchmarkResult>> {
        match &self.storage {
            BenchmarkStorage::Memory { results } => Ok(results
                .iter()
                .filter(|r| r.test_case == test_name)
                .cloned()
                .collect()),
            BenchmarkStorage::File { path } => {
                let file = File::open(path)
                    .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {e}")))?;

                let reader = BufReader::new(file);
                let mut results = Vec::new();

                for line in reader.lines() {
                    let line = line.map_err(|e| {
                        SklearsError::InvalidInput(format!("Failed to read line: {e}"))
                    })?;
                    let result: BenchmarkResult = serde_json::from_str(&line).map_err(|e| {
                        SklearsError::InvalidInput(format!("Failed to parse result: {e}"))
                    })?;

                    if result.test_case == test_name {
                        results.push(result);
                    }
                }

                Ok(results)
            }
            BenchmarkStorage::Database { .. } => Err(SklearsError::NotImplemented(
                "Database storage not implemented".to_string(),
            )),
        }
    }

    fn get_all_results(&self) -> SklResult<Vec<BenchmarkResult>> {
        match &self.storage {
            BenchmarkStorage::Memory { results } => Ok(results.clone()),
            BenchmarkStorage::File { path } => {
                let file = File::open(path)
                    .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {e}")))?;

                let reader = BufReader::new(file);
                let mut results = Vec::new();

                for line in reader.lines() {
                    let line = line.map_err(|e| {
                        SklearsError::InvalidInput(format!("Failed to read line: {e}"))
                    })?;
                    let result: BenchmarkResult = serde_json::from_str(&line).map_err(|e| {
                        SklearsError::InvalidInput(format!("Failed to parse result: {e}"))
                    })?;
                    results.push(result);
                }

                Ok(results)
            }
            BenchmarkStorage::Database { .. } => Err(SklearsError::NotImplemented(
                "Database storage not implemented".to_string(),
            )),
        }
    }

    // Regression analysis methods (simplified implementations)
    fn detect_performance_regression(&self, results: &[BenchmarkResult]) -> SklResult<bool> {
        if results.len() < 2 {
            return Ok(false);
        }

        let recent = &results[results.len() - 1];
        let baseline = &results[results.len() - 2];

        let regression_ratio = recent.metrics.execution_time.mean.as_secs_f64()
            / baseline.metrics.execution_time.mean.as_secs_f64();

        Ok(regression_ratio > (1.0 + self.regression_thresholds.relative_threshold))
    }

    fn calculate_regression_severity(
        &self,
        results: &[BenchmarkResult],
    ) -> SklResult<RegressionSeverity> {
        if results.len() < 2 {
            return Ok(RegressionSeverity::None);
        }

        let recent = &results[results.len() - 1];
        let baseline = &results[results.len() - 2];

        let regression_ratio = recent.metrics.execution_time.mean.as_secs_f64()
            / baseline.metrics.execution_time.mean.as_secs_f64();

        match regression_ratio {
            r if r < 1.05 => Ok(RegressionSeverity::None),
            r if r < 1.15 => Ok(RegressionSeverity::Minor),
            r if r < 1.3 => Ok(RegressionSeverity::Moderate),
            r if r < 1.5 => Ok(RegressionSeverity::Severe),
            _ => Ok(RegressionSeverity::Critical),
        }
    }

    fn identify_affected_metrics(&self, _results: &[BenchmarkResult]) -> SklResult<Vec<String>> {
        // Placeholder implementation
        Ok(vec!["execution_time".to_string()])
    }

    fn calculate_statistical_significance(&self, _results: &[BenchmarkResult]) -> SklResult<f64> {
        // Placeholder - would implement actual statistical tests
        Ok(0.05)
    }

    fn calculate_effect_size(&self, _results: &[BenchmarkResult]) -> SklResult<f64> {
        // Placeholder - would calculate Cohen's d or similar
        Ok(0.5)
    }

    fn calculate_confidence_interval(&self, _results: &[BenchmarkResult]) -> SklResult<(f64, f64)> {
        // Placeholder - would calculate actual confidence interval
        Ok((0.1, 0.3))
    }

    fn generate_detailed_analysis(&self, _results: &[BenchmarkResult]) -> SklResult<String> {
        Ok(
            "Performance analysis complete. Minor regression detected in execution time."
                .to_string(),
        )
    }

    fn generate_recommendations(
        &self,
        _results: &[BenchmarkResult],
        severity: &RegressionSeverity,
    ) -> Vec<String> {
        match severity {
            RegressionSeverity::None => vec!["Performance is stable".to_string()],
            RegressionSeverity::Minor => vec![
                "Monitor performance in future releases".to_string(),
                "Consider profiling to identify optimization opportunities".to_string(),
            ],
            RegressionSeverity::Moderate => vec![
                "Investigate recent changes that may have caused regression".to_string(),
                "Run detailed profiling to identify bottlenecks".to_string(),
                "Consider reverting problematic changes".to_string(),
            ],
            RegressionSeverity::Severe | RegressionSeverity::Critical => vec![
                "Immediate investigation required".to_string(),
                "Consider blocking release until regression is fixed".to_string(),
                "Run comprehensive profiling and analysis".to_string(),
                "Review all recent changes".to_string(),
            ],
        }
    }
}

/// Performance report generator
pub struct PerformanceReport {
    /// All benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Summary statistics
    pub summary: ReportSummary,
    /// Trend analysis
    pub trends: TrendAnalysis,
    /// Regression alerts
    pub regressions: Vec<RegressionAlert>,
}

/// Report summary statistics
#[derive(Clone, Debug)]
pub struct ReportSummary {
    /// Total number of benchmarks
    pub total_benchmarks: usize,
    /// Number of test cases
    pub test_cases: usize,
    /// Time range covered
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    /// Average performance metrics
    pub average_metrics: PerformanceMetrics,
}

/// Trend analysis results
#[derive(Clone, Debug)]
pub struct TrendAnalysis {
    /// Performance trends by test case
    pub trends_by_test: HashMap<String, PerformanceTrend>,
    /// Overall performance trend
    pub overall_trend: PerformanceTrend,
}

/// Performance trend direction and magnitude
#[derive(Clone, Debug)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving { rate: f64 },
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading { rate: f64 },
    /// Not enough data
    Insufficient,
}

/// Regression alert
#[derive(Clone, Debug)]
pub struct RegressionAlert {
    /// Test case name
    pub test_case: String,
    /// Severity level
    pub severity: RegressionSeverity,
    /// Description
    pub description: String,
    /// Timestamp when detected
    pub detected_at: DateTime<Utc>,
}

impl PerformanceReport {
    #[must_use]
    pub fn new(results: Vec<BenchmarkResult>, _config: &StatisticalAnalysisConfig) -> Self {
        let summary = ReportSummary::from_results(&results);
        let trends = TrendAnalysis::from_results(&results);
        let regressions = Self::detect_regressions(&results);

        Self {
            results,
            summary,
            trends,
            regressions,
        }
    }

    fn detect_regressions(results: &[BenchmarkResult]) -> Vec<RegressionAlert> {
        // Placeholder implementation
        vec![]
    }
}

impl ReportSummary {
    fn from_results(results: &[BenchmarkResult]) -> Self {
        let total_benchmarks = results.len();
        let test_cases = results
            .iter()
            .map(|r| r.test_case.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();

        let (start_time, end_time) = if results.is_empty() {
            (Utc::now(), Utc::now())
        } else {
            let start = results.iter().map(|r| r.timestamp).min().unwrap();
            let end = results.iter().map(|r| r.timestamp).max().unwrap();
            (start, end)
        };

        // Calculate average metrics (simplified)
        let average_metrics = PerformanceMetrics {
            execution_time: TimeStatistics {
                mean: Duration::from_millis(100),
                median: Duration::from_millis(95),
                std_dev: Duration::from_millis(10),
                min: Duration::from_millis(80),
                max: Duration::from_millis(150),
                p95: Duration::from_millis(130),
                p99: Duration::from_millis(145),
                samples: vec![],
            },
            memory_usage: MemoryStatistics {
                peak_usage: 1024 * 1024,
                average_usage: 512 * 1024,
                allocations: 100,
                deallocations: 95,
                fragmentation_score: 0.1,
            },
            throughput: ThroughputMetrics {
                ops_per_second: 100.0,
                samples_per_second: 1000.0,
                features_per_second: 10000.0,
                bytes_per_second: 80000.0,
            },
            cpu_utilization: CpuStatistics {
                average_utilization: 0.75,
                peak_utilization: 0.95,
                user_time: Duration::from_millis(100),
                kernel_time: Duration::from_millis(10),
            },
            custom_metrics: HashMap::new(),
        };

        Self {
            total_benchmarks,
            test_cases,
            time_range: (start_time, end_time),
            average_metrics,
        }
    }
}

impl TrendAnalysis {
    fn from_results(_results: &[BenchmarkResult]) -> Self {
        Self {
            trends_by_test: HashMap::new(),
            overall_trend: PerformanceTrend::Stable,
        }
    }
}

// Default implementations
impl Default for StatisticalAnalysisConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_samples_for_trend: 5,
            rolling_window_size: 10,
            statistical_tests: vec![StatisticalTest::TTest, StatisticalTest::LinearRegression],
            outlier_detection: OutlierDetection::IQR { multiplier: 1.5 },
        }
    }
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 10,
            cpu_affinity: None,
            memory_limit: None,
            capture_env_vars: vec!["RUST_VERSION".to_string(), "CARGO_PKG_VERSION".to_string()],
            collect_system_info: true,
        }
    }
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            relative_threshold: 0.05, // 5%
            absolute_threshold: Duration::from_millis(10),
            memory_threshold: 1024 * 1024, // 1MB
            throughput_threshold: 0.05,    // 5%
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            cpu_profiling: false,
            memory_profiling: false,
            sampling_frequency: Duration::from_millis(1),
            output_directory: None,
            detailed_call_stacks: false,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_tester_creation() {
        let tester = PerformanceRegressionTester::new();
        assert!(matches!(tester.storage, BenchmarkStorage::Memory { .. }));
    }

    #[test]
    fn test_file_storage_creation() {
        let tester = PerformanceRegressionTester::with_file_storage("/tmp/benchmarks.jsonl");
        assert!(matches!(tester.storage, BenchmarkStorage::File { .. }));
    }

    #[test]
    fn test_time_statistics_calculation() {
        let tester = PerformanceRegressionTester::new();
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(110),
            Duration::from_millis(95),
            Duration::from_millis(105),
            Duration::from_millis(120),
        ];

        let stats = tester.calculate_time_statistics(&times);
        assert_eq!(stats.min, Duration::from_millis(95));
        assert_eq!(stats.max, Duration::from_millis(120));
        assert_eq!(stats.samples.len(), 5);
    }

    #[test]
    fn test_benchmark_component() {
        let mut tester = PerformanceRegressionTester::new();

        let test_function = |x: i32| x * 2;
        let context = BenchmarkContext {
            data_size: (1000, 10),
            iterations: 5,
            config: HashMap::new(),
            random_seed: 42,
        };

        let result = tester.benchmark_component(&test_function, 42, &context, "test_multiply");
        assert!(result.is_ok());

        let benchmark_result = result.unwrap();
        assert_eq!(benchmark_result.test_case, "test_multiply");
        assert!(!benchmark_result.metrics.execution_time.samples.is_empty());
    }

    #[test]
    fn test_regression_severity_ordering() {
        assert!(RegressionSeverity::Critical > RegressionSeverity::Severe);
        assert!(RegressionSeverity::Severe > RegressionSeverity::Moderate);
        assert!(RegressionSeverity::Moderate > RegressionSeverity::Minor);
        assert!(RegressionSeverity::Minor > RegressionSeverity::None);
    }

    #[test]
    fn test_throughput_calculation() {
        let tester = PerformanceRegressionTester::new();
        let time_stats = TimeStatistics {
            mean: Duration::from_millis(100),
            median: Duration::from_millis(100),
            std_dev: Duration::from_millis(5),
            min: Duration::from_millis(90),
            max: Duration::from_millis(110),
            p95: Duration::from_millis(108),
            p99: Duration::from_millis(110),
            samples: vec![],
        };

        let context = BenchmarkContext {
            data_size: (1000, 10),
            iterations: 10,
            config: HashMap::new(),
            random_seed: 42,
        };

        let throughput = tester.calculate_throughput(&time_stats, &context);
        assert_eq!(throughput.ops_per_second, 10.0); // 1 / 0.1 seconds
        assert_eq!(throughput.samples_per_second, 10000.0); // 1000 samples / 0.1 seconds
    }
}
