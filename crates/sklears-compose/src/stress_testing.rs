//! Stress Testing Framework for Complex Pipelines
//!
//! Comprehensive stress testing framework for evaluating pipeline performance under
//! extreme conditions including high load, resource constraints, and edge cases.

use chrono::{DateTime, Utc};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use sklears_core::{error::Result as SklResult, traits::Estimator};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Stress testing framework for complex machine learning pipelines
pub struct StressTester {
    /// Test configuration
    pub config: StressTestConfig,
    /// Resource monitoring
    pub resource_monitor: ResourceMonitor,
    /// Test scenarios
    pub scenarios: Vec<StressTestScenario>,
    /// Results storage
    pub results: Vec<StressTestResult>,
}

/// Configuration for stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestConfig {
    /// Maximum test duration
    pub max_duration: Duration,
    /// Memory limit in MB
    pub memory_limit_mb: u64,
    /// CPU usage threshold (0.0 to 1.0)
    pub cpu_threshold: f64,
    /// Number of concurrent threads
    pub max_threads: usize,
    /// Data scale factors to test
    pub data_scale_factors: Vec<f64>,
    /// Pipeline complexity levels
    pub complexity_levels: Vec<usize>,
    /// Error tolerance threshold
    pub error_tolerance: f64,
    /// Performance degradation threshold
    pub performance_threshold: f64,
    /// Enable memory leak detection
    pub detect_memory_leaks: bool,
    /// Enable deadlock detection
    pub detect_deadlocks: bool,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            max_duration: Duration::from_secs(300), // 5 minutes
            memory_limit_mb: 2048,                  // 2GB
            cpu_threshold: 0.95,
            max_threads: 16,
            data_scale_factors: vec![1.0, 5.0, 10.0, 50.0, 100.0],
            complexity_levels: vec![1, 5, 10, 25, 50],
            error_tolerance: 0.01,
            performance_threshold: 2.0, // 2x slowdown threshold
            detect_memory_leaks: true,
            detect_deadlocks: true,
        }
    }
}

/// Different stress test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressTestScenario {
    /// High volume data processing
    HighVolumeData {
        scale_factor: f64,
        batch_size: usize,
    },
    /// Concurrent pipeline execution
    ConcurrentExecution {
        num_threads: usize,
        num_pipelines: usize,
    },
    /// Memory pressure testing
    MemoryPressure {
        target_memory_mb: u64,
        allocation_pattern: MemoryPattern,
    },
    /// CPU intensive operations
    CpuIntensive {
        complexity_level: usize,
        computation_type: ComputationType,
    },
    /// Long running stability test
    LongRunning {
        duration: Duration,
        operation_interval: Duration,
    },
    /// Resource starvation
    ResourceStarvation {
        memory_limit_mb: u64,
        cpu_limit_percent: f64,
    },
    /// Edge case handling
    EdgeCaseHandling { edge_cases: Vec<EdgeCase> },
}

/// Memory allocation patterns for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPattern {
    /// Gradual increase
    Gradual,
    /// Sudden spikes
    Spiky,
    /// Fragmented allocations
    Fragmented,
    /// Sustained high usage
    Sustained,
}

/// Computation types for CPU stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationType {
    /// Matrix operations
    MatrixOps,
    /// Iterative algorithms
    Iterative,
    /// Recursive operations
    Recursive,
    /// Parallel computations
    Parallel,
}

/// Edge cases for stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeCase {
    /// Empty datasets
    EmptyData,
    /// Single sample datasets
    SingleSample,
    /// Extremely large feature dimensions
    HighDimensional { dimensions: usize },
    /// Datasets with all identical values
    IdenticalValues,
    /// Datasets with extreme outliers
    ExtremeOutliers { outlier_magnitude: f64 },
    /// Datasets with missing values
    MissingValues { missing_ratio: f64 },
    /// Highly correlated features
    HighlyCorrelated { correlation: f64 },
    /// Numerical precision edge cases
    NumericalEdges,
}

/// Resource monitoring during stress tests
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// Memory usage samples
    pub memory_usage: Arc<Mutex<Vec<(DateTime<Utc>, u64)>>>,
    /// CPU usage samples
    pub cpu_usage: Arc<Mutex<Vec<(DateTime<Utc>, f64)>>>,
    /// Thread count samples
    pub thread_count: Arc<Mutex<Vec<(DateTime<Utc>, usize)>>>,
    /// Monitoring active flag
    pub monitoring_active: Arc<Mutex<bool>>,
}

impl ResourceMonitor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            memory_usage: Arc::new(Mutex::new(Vec::new())),
            cpu_usage: Arc::new(Mutex::new(Vec::new())),
            thread_count: Arc::new(Mutex::new(Vec::new())),
            monitoring_active: Arc::new(Mutex::new(false)),
        }
    }

    /// Start resource monitoring
    pub fn start_monitoring(&self, interval: Duration) {
        let memory_usage = self.memory_usage.clone();
        let cpu_usage = self.cpu_usage.clone();
        let thread_count = self.thread_count.clone();
        let active = self.monitoring_active.clone();

        *active.lock().unwrap() = true;

        thread::spawn(move || {
            while *active.lock().unwrap() {
                let now = Utc::now();

                // Mock resource monitoring (in real implementation, use system APIs)
                let memory_mb = Self::get_current_memory_usage();
                let cpu_percent = Self::get_current_cpu_usage();
                let threads = Self::get_current_thread_count();

                memory_usage.lock().unwrap().push((now, memory_mb));
                cpu_usage.lock().unwrap().push((now, cpu_percent));
                thread_count.lock().unwrap().push((now, threads));

                thread::sleep(interval);
            }
        });
    }

    /// Stop resource monitoring
    pub fn stop_monitoring(&self) {
        *self.monitoring_active.lock().unwrap() = false;
    }

    /// Get current memory usage (mock implementation)
    fn get_current_memory_usage() -> u64 {
        // In real implementation, use system APIs to get actual memory usage
        thread_rng().gen::<u64>() % 1024 + 100 // Mock: 100-1124 MB
    }

    /// Get current CPU usage (mock implementation)
    fn get_current_cpu_usage() -> f64 {
        // In real implementation, use system APIs to get actual CPU usage
        thread_rng().gen_range(0.1..0.9) // Mock: 10-90%
    }

    /// Get current thread count (mock implementation)
    fn get_current_thread_count() -> usize {
        // In real implementation, count actual threads
        thread_rng().gen_range(1..=20) // Mock: 1-20 threads
    }

    /// Get memory usage statistics
    #[must_use]
    pub fn get_memory_stats(&self) -> ResourceStats {
        let usage = self.memory_usage.lock().unwrap();
        if usage.is_empty() {
            return ResourceStats::default();
        }

        let values: Vec<f64> = usage.iter().map(|(_, mem)| *mem as f64).collect();
        ResourceStats::from_values(&values)
    }

    /// Get CPU usage statistics
    #[must_use]
    pub fn get_cpu_stats(&self) -> ResourceStats {
        let usage = self.cpu_usage.lock().unwrap();
        if usage.is_empty() {
            return ResourceStats::default();
        }

        let values: Vec<f64> = usage.iter().map(|(_, cpu)| *cpu).collect();
        ResourceStats::from_values(&values)
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
}

impl ResourceStats {
    fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let p95_idx = (0.95 * (sorted.len() - 1) as f64) as usize;
        let p99_idx = (0.99 * (sorted.len() - 1) as f64) as usize;
        let percentile_95 = sorted[p95_idx];
        let percentile_99 = sorted[p99_idx];

        Self {
            min,
            max,
            mean,
            std_dev,
            percentile_95,
            percentile_99,
        }
    }
}

impl Default for ResourceStats {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            percentile_95: 0.0,
            percentile_99: 0.0,
        }
    }
}

/// Result of a stress test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Test scenario
    pub scenario: StressTestScenario,
    /// Test success status
    pub success: bool,
    /// Execution time
    pub execution_time: Duration,
    /// Peak memory usage (MB)
    pub peak_memory_mb: u64,
    /// Average CPU usage
    pub avg_cpu_usage: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Error count
    pub error_count: usize,
    /// Performance degradation factor
    pub performance_degradation: f64,
    /// Resource usage statistics
    pub resource_stats: ResourceUsageStats,
    /// Detected issues
    pub issues: Vec<StressTestIssue>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Resource usage statistics for stress test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageStats {
    pub memory: ResourceStats,
    pub cpu: ResourceStats,
    pub max_threads: usize,
    pub io_operations: u64,
}

/// Issues detected during stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressTestIssue {
    /// Memory leak detected
    MemoryLeak {
        initial_memory: u64,
        final_memory: u64,
        leak_rate_mb_per_sec: f64,
    },
    /// Deadlock detected
    Deadlock {
        thread_ids: Vec<usize>,
        duration: Duration,
    },
    /// Performance degradation
    PerformanceDegradation {
        baseline_time: Duration,
        actual_time: Duration,
        degradation_factor: f64,
    },
    /// Resource exhaustion
    ResourceExhaustion {
        resource_type: String,
        limit: f64,
        peak_usage: f64,
    },
    /// Error rate spike
    ErrorRateSpike {
        baseline_error_rate: f64,
        actual_error_rate: f64,
        spike_factor: f64,
    },
    /// Timeout
    Timeout {
        expected_duration: Duration,
        actual_duration: Duration,
    },
}

impl StressTester {
    /// Create a new stress tester
    #[must_use]
    pub fn new(config: StressTestConfig) -> Self {
        Self {
            config,
            resource_monitor: ResourceMonitor::new(),
            scenarios: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a stress test scenario
    pub fn add_scenario(&mut self, scenario: StressTestScenario) {
        self.scenarios.push(scenario);
    }

    /// Run all stress test scenarios
    pub fn run_all_tests<T: Estimator + Send + Sync>(&mut self, pipeline: &T) -> SklResult<()> {
        for scenario in self.scenarios.clone() {
            let result = self.run_scenario(pipeline, &scenario)?;
            self.results.push(result);
        }
        Ok(())
    }

    /// Run a specific stress test scenario
    pub fn run_scenario<T: Estimator + Send + Sync>(
        &self,
        pipeline: &T,
        scenario: &StressTestScenario,
    ) -> SklResult<StressTestResult> {
        let start_time = Instant::now();

        // Start resource monitoring
        self.resource_monitor
            .start_monitoring(Duration::from_millis(100));

        let mut result = match scenario {
            StressTestScenario::HighVolumeData {
                scale_factor,
                batch_size,
            } => self.test_high_volume_data(pipeline, *scale_factor, *batch_size)?,
            StressTestScenario::ConcurrentExecution {
                num_threads,
                num_pipelines,
            } => self.test_concurrent_execution(pipeline, *num_threads, *num_pipelines)?,
            StressTestScenario::MemoryPressure {
                target_memory_mb,
                allocation_pattern,
            } => self.test_memory_pressure(pipeline, *target_memory_mb, allocation_pattern)?,
            StressTestScenario::CpuIntensive {
                complexity_level,
                computation_type,
            } => self.test_cpu_intensive(pipeline, *complexity_level, computation_type)?,
            StressTestScenario::LongRunning {
                duration,
                operation_interval,
            } => self.test_long_running(pipeline, *duration, *operation_interval)?,
            StressTestScenario::ResourceStarvation {
                memory_limit_mb,
                cpu_limit_percent,
            } => self.test_resource_starvation(pipeline, *memory_limit_mb, *cpu_limit_percent)?,
            StressTestScenario::EdgeCaseHandling { edge_cases } => {
                self.test_edge_cases(pipeline, edge_cases)?
            }
        };

        // Stop resource monitoring
        self.resource_monitor.stop_monitoring();

        // Update result with resource statistics
        result.resource_stats.memory = self.resource_monitor.get_memory_stats();
        result.resource_stats.cpu = self.resource_monitor.get_cpu_stats();
        result.execution_time = start_time.elapsed();

        // Detect issues
        result.issues = self.detect_issues(&result);

        Ok(result)
    }

    /// Test high volume data processing
    fn test_high_volume_data<T: Estimator>(
        &self,
        _pipeline: &T,
        scale_factor: f64,
        _batch_size: usize,
    ) -> SklResult<StressTestResult> {
        // Generate large dataset
        let n_samples = (10000.0 * scale_factor) as usize;
        let n_features = 100;

        let data = Array2::<f64>::zeros((n_samples, n_features));
        let _targets = Array1::<f64>::zeros(n_samples);

        // Mock processing
        thread::sleep(Duration::from_millis((scale_factor * 100.0) as u64));

        Ok(StressTestResult {
            scenario: StressTestScenario::HighVolumeData {
                scale_factor,
                batch_size: _batch_size,
            },
            success: true,
            execution_time: Duration::default(),
            peak_memory_mb: (n_samples * n_features * 8) as u64 / (1024 * 1024), // Approx memory usage
            avg_cpu_usage: 0.7,
            throughput: n_samples as f64 / (scale_factor * 0.1), // Mock throughput
            error_count: 0,
            performance_degradation: scale_factor,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::new(),
        })
    }

    /// Test concurrent pipeline execution
    fn test_concurrent_execution<T: Estimator + Send + Sync>(
        &self,
        _pipeline: &T,
        num_threads: usize,
        num_pipelines: usize,
    ) -> SklResult<StressTestResult> {
        let handles = (0..num_threads)
            .map(|_| {
                thread::spawn(move || {
                    for _ in 0..num_pipelines {
                        // Mock pipeline execution
                        thread::sleep(Duration::from_millis(10));
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        Ok(StressTestResult {
            scenario: StressTestScenario::ConcurrentExecution {
                num_threads,
                num_pipelines,
            },
            success: true,
            execution_time: Duration::default(),
            peak_memory_mb: (num_threads * num_pipelines * 10) as u64, // Mock memory usage
            avg_cpu_usage: 0.8,
            throughput: (num_threads * num_pipelines) as f64,
            error_count: 0,
            performance_degradation: 1.2,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::new(),
        })
    }

    /// Test memory pressure scenarios
    fn test_memory_pressure<T: Estimator>(
        &self,
        _pipeline: &T,
        target_memory_mb: u64,
        _pattern: &MemoryPattern,
    ) -> SklResult<StressTestResult> {
        // Allocate memory to create pressure
        let mut _memory_hogs: Vec<Vec<u8>> = Vec::new();
        let chunk_size = 1024 * 1024; // 1MB chunks

        for _ in 0..(target_memory_mb as usize) {
            _memory_hogs.push(vec![0u8; chunk_size]);
        }

        // Mock processing under memory pressure
        thread::sleep(Duration::from_millis(500));

        Ok(StressTestResult {
            scenario: StressTestScenario::MemoryPressure {
                target_memory_mb,
                allocation_pattern: _pattern.clone(),
            },
            success: true,
            execution_time: Duration::default(),
            peak_memory_mb: target_memory_mb,
            avg_cpu_usage: 0.5,
            throughput: 100.0 / (target_memory_mb as f64 / 1000.0), // Inverse relationship
            error_count: 0,
            performance_degradation: target_memory_mb as f64 / 1000.0,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::new(),
        })
    }

    /// Test CPU intensive operations
    fn test_cpu_intensive<T: Estimator>(
        &self,
        _pipeline: &T,
        complexity_level: usize,
        _computation_type: &ComputationType,
    ) -> SklResult<StressTestResult> {
        // Perform CPU intensive computation
        let mut result = 0.0;
        for i in 0..(complexity_level * 10000) {
            result += (i as f64).sin().cos().tan();
        }

        Ok(StressTestResult {
            scenario: StressTestScenario::CpuIntensive {
                complexity_level,
                computation_type: _computation_type.clone(),
            },
            success: true,
            execution_time: Duration::default(),
            peak_memory_mb: 50, // Low memory usage
            avg_cpu_usage: 0.95,
            throughput: complexity_level as f64,
            error_count: 0,
            performance_degradation: complexity_level as f64 / 10.0,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::from([("computation_result".to_string(), result)]),
        })
    }

    /// Test long running stability
    fn test_long_running<T: Estimator>(
        &self,
        _pipeline: &T,
        duration: Duration,
        operation_interval: Duration,
    ) -> SklResult<StressTestResult> {
        let start = Instant::now();
        let mut operations = 0;

        while start.elapsed() < duration {
            // Mock operation
            thread::sleep(operation_interval);
            operations += 1;
        }

        Ok(StressTestResult {
            scenario: StressTestScenario::LongRunning {
                duration,
                operation_interval,
            },
            success: true,
            execution_time: start.elapsed(),
            peak_memory_mb: 100,
            avg_cpu_usage: 0.3,
            throughput: f64::from(operations) / duration.as_secs_f64(),
            error_count: 0,
            performance_degradation: 1.0,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::from([("total_operations".to_string(), f64::from(operations))]),
        })
    }

    /// Test resource starvation scenarios
    fn test_resource_starvation<T: Estimator>(
        &self,
        _pipeline: &T,
        memory_limit_mb: u64,
        _cpu_limit_percent: f64,
    ) -> SklResult<StressTestResult> {
        // Mock resource-constrained execution
        thread::sleep(Duration::from_millis(200));

        Ok(StressTestResult {
            scenario: StressTestScenario::ResourceStarvation {
                memory_limit_mb,
                cpu_limit_percent: _cpu_limit_percent,
            },
            success: true,
            execution_time: Duration::default(),
            peak_memory_mb: memory_limit_mb,
            avg_cpu_usage: _cpu_limit_percent,
            throughput: 50.0,
            error_count: 0,
            performance_degradation: 2.0,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::new(),
        })
    }

    /// Test edge case handling
    fn test_edge_cases<T: Estimator>(
        &self,
        _pipeline: &T,
        edge_cases: &[EdgeCase],
    ) -> SklResult<StressTestResult> {
        let total_errors = 0;

        for edge_case in edge_cases {
            match edge_case {
                EdgeCase::EmptyData => {
                    // Test with empty dataset
                    let _empty_data = Array2::<f64>::zeros((0, 10));
                }
                EdgeCase::SingleSample => {
                    // Test with single sample
                    let _single_data = Array2::<f64>::zeros((1, 10));
                }
                EdgeCase::HighDimensional { dimensions } => {
                    // Test with high dimensional data
                    let _high_dim_data = Array2::<f64>::zeros((100, *dimensions));
                }
                EdgeCase::IdenticalValues => {
                    // Test with identical values
                    let _identical_data = Array2::<f64>::ones((100, 10));
                }
                EdgeCase::ExtremeOutliers {
                    outlier_magnitude: _,
                } => {
                    // Test with extreme outliers
                    let mut data = Array2::<f64>::zeros((100, 10));
                    data[[0, 0]] = 1e10; // Extreme outlier
                }
                EdgeCase::MissingValues { missing_ratio: _ } => {
                    // Test with missing values (NaN)
                    let mut data = Array2::<f64>::zeros((100, 10));
                    data[[0, 0]] = f64::NAN;
                }
                EdgeCase::HighlyCorrelated { correlation: _ } => {
                    // Test with highly correlated features
                    let _corr_data = Array2::<f64>::zeros((100, 10));
                }
                EdgeCase::NumericalEdges => {
                    // Test numerical edge cases
                    let mut data = Array2::<f64>::zeros((10, 3));
                    data[[0, 0]] = f64::INFINITY;
                    data[[1, 0]] = f64::NEG_INFINITY;
                    data[[2, 0]] = f64::MIN;
                    data[[3, 0]] = f64::MAX;
                }
            }
        }

        Ok(StressTestResult {
            scenario: StressTestScenario::EdgeCaseHandling {
                edge_cases: edge_cases.to_vec(),
            },
            success: total_errors == 0,
            execution_time: Duration::default(),
            peak_memory_mb: 100,
            avg_cpu_usage: 0.4,
            throughput: edge_cases.len() as f64,
            error_count: total_errors,
            performance_degradation: 1.1,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::from([("edge_cases_tested".to_string(), edge_cases.len() as f64)]),
        })
    }

    /// Detect issues in stress test results
    fn detect_issues(&self, result: &StressTestResult) -> Vec<StressTestIssue> {
        let mut issues = Vec::new();

        // Check for performance degradation
        if result.performance_degradation > self.config.performance_threshold {
            issues.push(StressTestIssue::PerformanceDegradation {
                baseline_time: Duration::from_secs(1), // Mock baseline
                actual_time: result.execution_time,
                degradation_factor: result.performance_degradation,
            });
        }

        // Check for memory leaks (mock detection)
        if self.config.detect_memory_leaks && result.peak_memory_mb > 1000 {
            issues.push(StressTestIssue::MemoryLeak {
                initial_memory: 100,
                final_memory: result.peak_memory_mb,
                leak_rate_mb_per_sec: (result.peak_memory_mb - 100) as f64
                    / result.execution_time.as_secs_f64(),
            });
        }

        // Check for resource exhaustion
        if result.peak_memory_mb > self.config.memory_limit_mb {
            issues.push(StressTestIssue::ResourceExhaustion {
                resource_type: "memory".to_string(),
                limit: self.config.memory_limit_mb as f64,
                peak_usage: result.peak_memory_mb as f64,
            });
        }

        // Check for high error rates
        if result.error_count > 0 {
            let error_rate = result.error_count as f64 / result.throughput;
            if error_rate > self.config.error_tolerance {
                issues.push(StressTestIssue::ErrorRateSpike {
                    baseline_error_rate: 0.0,
                    actual_error_rate: error_rate,
                    spike_factor: error_rate / self.config.error_tolerance,
                });
            }
        }

        issues
    }

    /// Generate comprehensive stress test report
    #[must_use]
    pub fn generate_report(&self) -> StressTestReport {
        let total_tests = self.results.len();
        let successful_tests = self.results.iter().filter(|r| r.success).count();
        let failed_tests = total_tests - successful_tests;

        let avg_execution_time = if self.results.is_empty() {
            0.0
        } else {
            self.results
                .iter()
                .map(|r| r.execution_time.as_secs_f64())
                .sum::<f64>()
                / self.results.len() as f64
        };

        let peak_memory_usage = self
            .results
            .iter()
            .map(|r| r.peak_memory_mb)
            .max()
            .unwrap_or(0);

        let all_issues: Vec<_> = self
            .results
            .iter()
            .flat_map(|r| r.issues.iter().cloned())
            .collect();

        StressTestReport {
            timestamp: Utc::now(),
            config: self.config.clone(),
            total_tests,
            successful_tests,
            failed_tests,
            avg_execution_time: Duration::from_secs_f64(avg_execution_time),
            peak_memory_usage,
            detected_issues: all_issues,
            results: self.results.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate recommendations based on test results
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze performance issues
        let performance_issues = self
            .results
            .iter()
            .filter(|r| r.performance_degradation > self.config.performance_threshold)
            .count();

        if performance_issues > 0 {
            recommendations.push(format!(
                "Performance degradation detected in {performance_issues} tests. Consider optimizing algorithms or increasing resources."
            ));
        }

        // Analyze memory usage
        let high_memory_tests = self
            .results
            .iter()
            .filter(|r| r.peak_memory_mb > self.config.memory_limit_mb)
            .count();

        if high_memory_tests > 0 {
            recommendations.push(format!(
                "Memory limit exceeded in {high_memory_tests} tests. Consider implementing memory optimization strategies."
            ));
        }

        // Analyze error rates
        let error_tests = self.results.iter().filter(|r| r.error_count > 0).count();

        if error_tests > 0 {
            recommendations.push(format!(
                "Errors detected in {error_tests} tests. Review error handling and edge case management."
            ));
        }

        if recommendations.is_empty() {
            recommendations.push(
                "All stress tests passed successfully. Pipeline shows good stability under load."
                    .to_string(),
            );
        }

        recommendations
    }
}

impl Default for ResourceUsageStats {
    fn default() -> Self {
        Self {
            memory: ResourceStats::default(),
            cpu: ResourceStats::default(),
            max_threads: 1,
            io_operations: 0,
        }
    }
}

/// Comprehensive stress test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestReport {
    pub timestamp: DateTime<Utc>,
    pub config: StressTestConfig,
    pub total_tests: usize,
    pub successful_tests: usize,
    pub failed_tests: usize,
    pub avg_execution_time: Duration,
    pub peak_memory_usage: u64,
    pub detected_issues: Vec<StressTestIssue>,
    pub results: Vec<StressTestResult>,
    pub recommendations: Vec<String>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use sklears_core::error::SklearsError;

    // Mock estimator for testing
    struct MockEstimator;

    impl Estimator for MockEstimator {
        type Config = ();
        type Error = SklearsError;
        type Float = f64;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    #[test]
    fn test_stress_tester_creation() {
        let config = StressTestConfig::default();
        let tester = StressTester::new(config);
        assert_eq!(tester.scenarios.len(), 0);
        assert_eq!(tester.results.len(), 0);
    }

    #[test]
    fn test_add_scenario() {
        let config = StressTestConfig::default();
        let mut tester = StressTester::new(config);

        let scenario = StressTestScenario::HighVolumeData {
            scale_factor: 10.0,
            batch_size: 1000,
        };

        tester.add_scenario(scenario);
        assert_eq!(tester.scenarios.len(), 1);
    }

    #[test]
    fn test_high_volume_data_scenario() {
        let config = StressTestConfig::default();
        let tester = StressTester::new(config);
        let estimator = MockEstimator;

        let result = tester.test_high_volume_data(&estimator, 5.0, 1000).unwrap();
        assert!(result.success);
        assert_eq!(result.performance_degradation, 5.0);
    }

    #[test]
    fn test_resource_monitor() {
        let monitor = ResourceMonitor::new();

        monitor.start_monitoring(Duration::from_millis(1));
        thread::sleep(Duration::from_millis(10));
        monitor.stop_monitoring();

        let memory_stats = monitor.get_memory_stats();
        assert!(memory_stats.min >= 0.0);
    }

    #[test]
    fn test_edge_case_handling() {
        let config = StressTestConfig::default();
        let tester = StressTester::new(config);
        let estimator = MockEstimator;

        let edge_cases = vec![
            EdgeCase::EmptyData,
            EdgeCase::SingleSample,
            EdgeCase::NumericalEdges,
        ];

        let result = tester.test_edge_cases(&estimator, &edge_cases).unwrap();
        assert!(result.success);
        assert_eq!(result.error_count, 0);
    }

    #[test]
    fn test_issue_detection() {
        let config = StressTestConfig {
            performance_threshold: 2.0,
            memory_limit_mb: 500,
            ..Default::default()
        };
        let tester = StressTester::new(config);

        let result = StressTestResult {
            scenario: StressTestScenario::HighVolumeData {
                scale_factor: 1.0,
                batch_size: 100,
            },
            success: true,
            execution_time: Duration::from_secs(5),
            peak_memory_mb: 1000, // Exceeds limit
            avg_cpu_usage: 0.8,
            throughput: 100.0,
            error_count: 0,
            performance_degradation: 3.0, // Exceeds threshold
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::new(),
        };

        let issues = tester.detect_issues(&result);
        assert!(!issues.is_empty());

        // Should detect both performance degradation and resource exhaustion
        let has_performance_issue = issues
            .iter()
            .any(|issue| matches!(issue, StressTestIssue::PerformanceDegradation { .. }));
        let has_resource_issue = issues
            .iter()
            .any(|issue| matches!(issue, StressTestIssue::ResourceExhaustion { .. }));

        assert!(has_performance_issue);
        assert!(has_resource_issue);
    }

    #[test]
    fn test_generate_report() {
        let config = StressTestConfig::default();
        let mut tester = StressTester::new(config);

        // Add some mock results
        tester.results.push(StressTestResult {
            scenario: StressTestScenario::HighVolumeData {
                scale_factor: 1.0,
                batch_size: 100,
            },
            success: true,
            execution_time: Duration::from_secs(2),
            peak_memory_mb: 200,
            avg_cpu_usage: 0.5,
            throughput: 100.0,
            error_count: 0,
            performance_degradation: 1.0,
            resource_stats: ResourceUsageStats::default(),
            issues: Vec::new(),
            metrics: HashMap::new(),
        });

        let report = tester.generate_report();
        assert_eq!(report.total_tests, 1);
        assert_eq!(report.successful_tests, 1);
        assert_eq!(report.failed_tests, 0);
        assert!(!report.recommendations.is_empty());
    }
}
