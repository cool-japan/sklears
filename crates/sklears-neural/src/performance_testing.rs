//! Performance testing and benchmarking utilities for neural networks.
//!
//! This module provides comprehensive performance testing capabilities including
//! memory usage monitoring, execution time benchmarking, regression detection,
//! and comparison with other frameworks. It enables systematic performance
//! analysis and optimization of neural network implementations.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::SklearsError;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result type for performance testing operations
pub type PerformanceResult<T> = Result<T, SklearsError>;

/// Memory usage statistics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_usage_bytes: u64,
    /// Current memory usage in bytes
    pub current_usage_bytes: u64,
    /// Memory allocated during operation
    pub allocated_bytes: u64,
    /// Memory deallocated during operation
    pub deallocated_bytes: u64,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
}

impl MemoryStats {
    pub fn new() -> Self {
        Self {
            peak_usage_bytes: 0,
            current_usage_bytes: 0,
            allocated_bytes: 0,
            deallocated_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }

    /// Calculate memory efficiency (deallocated / allocated)
    pub fn efficiency_ratio(&self) -> f64 {
        if self.allocated_bytes > 0 {
            self.deallocated_bytes as f64 / self.allocated_bytes as f64
        } else {
            1.0
        }
    }

    /// Calculate average allocation size
    pub fn avg_allocation_size(&self) -> f64 {
        if self.allocation_count > 0 {
            self.allocated_bytes as f64 / self.allocation_count as f64
        } else {
            0.0
        }
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance measurement results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceMetrics {
    /// Execution time statistics
    pub execution_time: Duration,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Operations per second
    pub ops_per_second: f64,
    /// Throughput in samples per second
    pub samples_per_second: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// GPU usage percentage (if available)
    pub gpu_usage_percent: Option<f64>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl PerformanceMetrics {
    pub fn new(execution_time: Duration, memory_stats: MemoryStats) -> Self {
        Self {
            execution_time,
            memory_stats,
            ops_per_second: 0.0,
            samples_per_second: 0.0,
            cpu_usage_percent: 0.0,
            gpu_usage_percent: None,
            custom_metrics: HashMap::new(),
        }
    }

    pub fn with_ops_per_second(mut self, ops_per_second: f64) -> Self {
        self.ops_per_second = ops_per_second;
        self
    }

    pub fn with_samples_per_second(mut self, samples_per_second: f64) -> Self {
        self.samples_per_second = samples_per_second;
        self
    }

    pub fn with_cpu_usage(mut self, cpu_usage_percent: f64) -> Self {
        self.cpu_usage_percent = cpu_usage_percent;
        self
    }

    pub fn with_gpu_usage(mut self, gpu_usage_percent: f64) -> Self {
        self.gpu_usage_percent = Some(gpu_usage_percent);
        self
    }

    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }
}

/// Benchmark comparison results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BenchmarkComparison {
    /// Name of the benchmark
    pub benchmark_name: String,
    /// Baseline performance metrics
    pub baseline: PerformanceMetrics,
    /// Current performance metrics
    pub current: PerformanceMetrics,
    /// Performance ratio (current / baseline)
    pub performance_ratio: f64,
    /// Memory ratio (current / baseline)
    pub memory_ratio: f64,
    /// Regression detected flag
    pub regression_detected: bool,
    /// Improvement percentage
    pub improvement_percent: f64,
}

impl BenchmarkComparison {
    pub fn new(
        benchmark_name: String,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics,
    ) -> Self {
        let baseline_time = baseline.execution_time.as_secs_f64();
        let current_time = current.execution_time.as_secs_f64();

        let performance_ratio = if baseline_time > 0.0 {
            current_time / baseline_time
        } else {
            1.0
        };

        let memory_ratio = if baseline.memory_stats.peak_usage_bytes > 0 {
            current.memory_stats.peak_usage_bytes as f64
                / baseline.memory_stats.peak_usage_bytes as f64
        } else {
            1.0
        };

        let improvement_percent = (1.0 - performance_ratio) * 100.0;
        let regression_detected = performance_ratio > 1.1; // 10% threshold

        Self {
            benchmark_name,
            baseline,
            current,
            performance_ratio,
            memory_ratio,
            regression_detected,
            improvement_percent,
        }
    }

    /// Check if there's a significant improvement (>5%)
    pub fn has_improvement(&self) -> bool {
        self.improvement_percent > 5.0
    }

    /// Check if there's a significant memory increase (>20%)
    pub fn has_memory_regression(&self) -> bool {
        self.memory_ratio > 1.2
    }
}

/// Performance profiler for monitoring execution
pub struct PerformanceProfiler {
    start_time: Option<Instant>,
    memory_tracker: Arc<Mutex<MemoryStats>>,
    operation_count: u64,
    sample_count: u64,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            start_time: None,
            memory_tracker: Arc::new(Mutex::new(MemoryStats::new())),
            operation_count: 0,
            sample_count: 0,
        }
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop profiling and return metrics
    pub fn stop(&self) -> PerformanceResult<PerformanceMetrics> {
        let start_time = self
            .start_time
            .ok_or_else(|| SklearsError::InvalidParameter {
                name: "profiler_state".to_string(),
                reason: "Profiler was not started".to_string(),
            })?;

        let execution_time = start_time.elapsed();
        let memory_stats = self
            .memory_tracker
            .lock()
            .map_err(|_| SklearsError::InvalidParameter {
                name: "memory_tracker".to_string(),
                reason: "Failed to acquire memory tracker lock".to_string(),
            })?
            .clone();

        let ops_per_second = if execution_time.as_secs_f64() > 0.0 {
            self.operation_count as f64 / execution_time.as_secs_f64()
        } else {
            0.0
        };

        let samples_per_second = if execution_time.as_secs_f64() > 0.0 {
            self.sample_count as f64 / execution_time.as_secs_f64()
        } else {
            0.0
        };

        Ok(PerformanceMetrics::new(execution_time, memory_stats)
            .with_ops_per_second(ops_per_second)
            .with_samples_per_second(samples_per_second))
    }

    /// Record an operation
    pub fn record_operation(&mut self) {
        self.operation_count += 1;
    }

    /// Record multiple operations
    pub fn record_operations(&mut self, count: u64) {
        self.operation_count += count;
    }

    /// Record samples processed
    pub fn record_samples(&mut self, count: u64) {
        self.sample_count += count;
    }

    /// Simulate memory allocation (for testing purposes)
    pub fn record_allocation(&self, bytes: u64) {
        if let Ok(mut stats) = self.memory_tracker.lock() {
            stats.allocated_bytes += bytes;
            stats.allocation_count += 1;
            stats.current_usage_bytes += bytes;

            if stats.current_usage_bytes > stats.peak_usage_bytes {
                stats.peak_usage_bytes = stats.current_usage_bytes;
            }
        }
    }

    /// Simulate memory deallocation (for testing purposes)
    pub fn record_deallocation(&self, bytes: u64) {
        if let Ok(mut stats) = self.memory_tracker.lock() {
            stats.deallocated_bytes += bytes;
            stats.deallocation_count += 1;
            stats.current_usage_bytes = stats.current_usage_bytes.saturating_sub(bytes);
        }
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark suite for comprehensive performance testing
pub struct BenchmarkSuite {
    benchmarks: HashMap<String, Box<dyn Fn() -> PerformanceResult<PerformanceMetrics>>>,
    baselines: HashMap<String, PerformanceMetrics>,
    results: HashMap<String, BenchmarkComparison>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            benchmarks: HashMap::new(),
            baselines: HashMap::new(),
            results: HashMap::new(),
        }
    }

    /// Add a benchmark function
    pub fn add_benchmark<F>(&mut self, name: String, benchmark_fn: F)
    where
        F: Fn() -> PerformanceResult<PerformanceMetrics> + 'static,
    {
        self.benchmarks.insert(name, Box::new(benchmark_fn));
    }

    /// Set baseline for a benchmark
    pub fn set_baseline(&mut self, name: String, baseline: PerformanceMetrics) {
        self.baselines.insert(name, baseline);
    }

    /// Run all benchmarks
    pub fn run_all(&mut self) -> PerformanceResult<()> {
        for (name, benchmark_fn) in &self.benchmarks {
            let current_metrics = benchmark_fn()?;

            if let Some(baseline) = self.baselines.get(name) {
                let comparison =
                    BenchmarkComparison::new(name.clone(), baseline.clone(), current_metrics);
                self.results.insert(name.clone(), comparison);
            }
        }
        Ok(())
    }

    /// Run a specific benchmark
    pub fn run_benchmark(&mut self, name: &str) -> PerformanceResult<()> {
        if let Some(benchmark_fn) = self.benchmarks.get(name) {
            let current_metrics = benchmark_fn()?;

            if let Some(baseline) = self.baselines.get(name) {
                let comparison =
                    BenchmarkComparison::new(name.to_string(), baseline.clone(), current_metrics);
                self.results.insert(name.to_string(), comparison);
            }
        } else {
            return Err(SklearsError::InvalidParameter {
                name: "benchmark_name".to_string(),
                reason: format!("Benchmark '{}' not found", name),
            });
        }
        Ok(())
    }

    /// Get benchmark results
    pub fn get_results(&self) -> &HashMap<String, BenchmarkComparison> {
        &self.results
    }

    /// Check if any regressions were detected
    pub fn has_regressions(&self) -> bool {
        self.results.values().any(|r| r.regression_detected)
    }

    /// Get regressions
    pub fn get_regressions(&self) -> Vec<&BenchmarkComparison> {
        self.results
            .values()
            .filter(|r| r.regression_detected)
            .collect()
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let total_benchmarks = self.results.len();
        let regressions = self.get_regressions();
        let regression_count = regressions.len();

        let improvements: Vec<&BenchmarkComparison> = self
            .results
            .values()
            .filter(|r| r.has_improvement())
            .collect();
        let improvement_count = improvements.len();

        let memory_regressions: Vec<&BenchmarkComparison> = self
            .results
            .values()
            .filter(|r| r.has_memory_regression())
            .collect();
        let memory_regression_count = memory_regressions.len();

        PerformanceReport {
            total_benchmarks,
            regression_count,
            improvement_count,
            memory_regression_count,
            overall_status: if regression_count == 0 {
                PerformanceStatus::Pass
            } else {
                PerformanceStatus::Fail
            },
            regressions: regressions.into_iter().cloned().collect(),
            improvements: improvements.into_iter().cloned().collect(),
            memory_regressions: memory_regressions.into_iter().cloned().collect(),
        }
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance test status
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PerformanceStatus {
    Pass,
    Fail,
    Warning,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceReport {
    pub total_benchmarks: usize,
    pub regression_count: usize,
    pub improvement_count: usize,
    pub memory_regression_count: usize,
    pub overall_status: PerformanceStatus,
    pub regressions: Vec<BenchmarkComparison>,
    pub improvements: Vec<BenchmarkComparison>,
    pub memory_regressions: Vec<BenchmarkComparison>,
}

impl fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Performance Test Report ===")?;
        writeln!(f, "Total Benchmarks: {}", self.total_benchmarks)?;
        writeln!(f, "Overall Status: {:?}", self.overall_status)?;
        writeln!(f)?;

        writeln!(f, "Improvements: {} benchmarks", self.improvement_count)?;
        for improvement in &self.improvements {
            writeln!(
                f,
                "  {}: {:.1}% faster",
                improvement.benchmark_name, improvement.improvement_percent
            )?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "Performance Regressions: {} benchmarks",
            self.regression_count
        )?;
        for regression in &self.regressions {
            writeln!(
                f,
                "  {} {:.1}% slower (ratio: {:.2})",
                regression.benchmark_name,
                -regression.improvement_percent,
                regression.performance_ratio
            )?;
        }
        writeln!(f)?;

        writeln!(
            f,
            "Memory Regressions: {} benchmarks",
            self.memory_regression_count
        )?;
        for mem_regression in &self.memory_regressions {
            writeln!(
                f,
                "  {}: {:.1}% more memory (ratio: {:.2})",
                mem_regression.benchmark_name,
                (mem_regression.memory_ratio - 1.0) * 100.0,
                mem_regression.memory_ratio
            )?;
        }

        Ok(())
    }
}

/// Memory leak detector
pub struct MemoryLeakDetector {
    initial_memory: u64,
    allocations: HashMap<usize, u64>,
    allocation_tracker: Arc<Mutex<u64>>,
}

impl MemoryLeakDetector {
    pub fn new() -> Self {
        Self {
            initial_memory: Self::get_current_memory_usage(),
            allocations: HashMap::new(),
            allocation_tracker: Arc::new(Mutex::new(0)),
        }
    }

    /// Start monitoring for memory leaks
    pub fn start_monitoring(&mut self) {
        self.initial_memory = Self::get_current_memory_usage();
        self.allocations.clear();
    }

    /// Check for memory leaks
    pub fn check_for_leaks(&self) -> MemoryLeakReport {
        let current_memory = Self::get_current_memory_usage();
        let memory_increase = current_memory.saturating_sub(self.initial_memory);

        let leak_threshold = 1024 * 1024; // 1MB threshold
        let has_leak = memory_increase > leak_threshold;

        let active_allocations = self.allocations.len();
        let total_allocated = self.allocations.values().sum::<u64>();

        MemoryLeakReport {
            initial_memory: self.initial_memory,
            current_memory,
            memory_increase,
            has_leak,
            leak_threshold,
            active_allocations,
            total_allocated,
        }
    }

    /// Simulate allocation tracking (for testing purposes)
    pub fn track_allocation(&mut self, ptr: usize, size: u64) {
        self.allocations.insert(ptr, size);
    }

    /// Simulate deallocation tracking (for testing purposes)
    pub fn track_deallocation(&mut self, ptr: usize) {
        self.allocations.remove(&ptr);
    }

    /// Get current memory usage (simplified implementation)
    fn get_current_memory_usage() -> u64 {
        // This is a simplified implementation
        // In a real scenario, you'd use platform-specific APIs
        // or memory profiling tools to get accurate memory usage
        std::process::id() as u64 * 1024 // Placeholder
    }
}

impl Default for MemoryLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory leak detection report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryLeakReport {
    pub initial_memory: u64,
    pub current_memory: u64,
    pub memory_increase: u64,
    pub has_leak: bool,
    pub leak_threshold: u64,
    pub active_allocations: usize,
    pub total_allocated: u64,
}

impl fmt::Display for MemoryLeakReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Memory Leak Detection Report ===")?;
        writeln!(f, "Initial Memory: {} bytes", self.initial_memory)?;
        writeln!(f, "Current Memory: {} bytes", self.current_memory)?;
        writeln!(f, "Memory Increase: {} bytes", self.memory_increase)?;
        writeln!(f, "Leak Detected: {}", self.has_leak)?;
        writeln!(f, "Active Allocations: {}", self.active_allocations)?;
        writeln!(f, "Total Allocated: {} bytes", self.total_allocated)?;

        if self.has_leak {
            writeln!(
                f,
                "⚠️  Memory leak detected! Increase exceeds threshold of {} bytes",
                self.leak_threshold
            )?;
        } else {
            writeln!(f, "✅ No memory leaks detected")?;
        }

        Ok(())
    }
}

/// Utility functions for performance testing
pub mod utils {
    use super::*;

    /// Benchmark a function and return performance metrics
    pub fn benchmark_function<F, T>(f: F, iterations: u32) -> PerformanceResult<PerformanceMetrics>
    where
        F: Fn() -> T,
    {
        let mut profiler = PerformanceProfiler::new();
        profiler.start();

        for _ in 0..iterations {
            let _ = f();
            profiler.record_operation();
        }

        profiler.stop()
    }

    /// Benchmark matrix multiplication
    pub fn benchmark_matrix_multiply(
        a: &Array2<f64>,
        b: &Array2<f64>,
        iterations: u32,
    ) -> PerformanceResult<PerformanceMetrics> {
        benchmark_function(|| a.dot(b), iterations)
    }

    /// Benchmark neural network forward pass
    pub fn benchmark_forward_pass<F>(
        forward_fn: F,
        input: &Array2<f64>,
        iterations: u32,
    ) -> PerformanceResult<PerformanceMetrics>
    where
        F: Fn(&Array2<f64>) -> Array2<f64>,
    {
        let mut profiler = PerformanceProfiler::new();
        profiler.start();

        for _ in 0..iterations {
            let _ = forward_fn(input);
            profiler.record_operation();
            profiler.record_samples(input.nrows() as u64);
        }

        profiler.stop()
    }

    /// Create a standard benchmark suite for neural networks
    pub fn create_standard_benchmark_suite() -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new();

        // Matrix multiplication benchmark
        suite.add_benchmark("matrix_multiply_100x100".to_string(), || {
            let a = Array2::ones((100, 100));
            let b = Array2::ones((100, 100));
            benchmark_matrix_multiply(&a, &b, 1000)
        });

        // Large matrix multiplication benchmark
        suite.add_benchmark("matrix_multiply_1000x1000".to_string(), || {
            let a = Array2::ones((1000, 1000));
            let b = Array2::ones((1000, 1000));
            benchmark_matrix_multiply(&a, &b, 10)
        });

        suite
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx;

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();
        stats.allocated_bytes = 1000;
        stats.deallocated_bytes = 800;
        stats.allocation_count = 10;

        assert_eq!(stats.efficiency_ratio(), 0.8);
        assert_eq!(stats.avg_allocation_size(), 100.0);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();
        profiler.start();

        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));
        profiler.record_operation();
        profiler.record_samples(100);

        let metrics = profiler.stop().unwrap();
        assert!(metrics.execution_time >= Duration::from_millis(10));
        assert!(metrics.ops_per_second > 0.0);
        assert!(metrics.samples_per_second > 0.0);
    }

    #[test]
    fn test_benchmark_comparison() {
        let baseline = PerformanceMetrics::new(Duration::from_millis(100), MemoryStats::new());

        let mut improved = PerformanceMetrics::new(Duration::from_millis(80), MemoryStats::new());

        let comparison = BenchmarkComparison::new("test_benchmark".to_string(), baseline, improved);

        assert!(comparison.has_improvement());
        assert!(!comparison.regression_detected);
        approx::assert_abs_diff_eq!(comparison.improvement_percent, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_leak_detector() {
        let mut detector = MemoryLeakDetector::new();
        detector.start_monitoring();

        // Simulate allocations
        detector.track_allocation(0x1000, 1024);
        detector.track_allocation(0x2000, 2048);

        let report = detector.check_for_leaks();
        assert_eq!(report.active_allocations, 2);
        assert_eq!(report.total_allocated, 3072);

        // Simulate deallocation
        detector.track_deallocation(0x1000);
        let report = detector.check_for_leaks();
        assert_eq!(report.active_allocations, 1);
        assert_eq!(report.total_allocated, 2048);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();

        // Add a simple benchmark
        suite.add_benchmark("simple_add".to_string(), || {
            let mut profiler = PerformanceProfiler::new();
            profiler.start();

            let _ = 1 + 1;
            profiler.record_operation();

            profiler.stop()
        });

        // Set a baseline
        let baseline = PerformanceMetrics::new(Duration::from_millis(1), MemoryStats::new());
        suite.set_baseline("simple_add".to_string(), baseline);

        // Run the benchmark
        suite.run_benchmark("simple_add").unwrap();

        let results = suite.get_results();
        assert!(results.contains_key("simple_add"));
    }

    #[test]
    fn test_benchmark_utilities() {
        let result = utils::benchmark_function(
            || {
                let a = Array2::<f64>::ones((10, 10));
                let b = Array2::<f64>::ones((10, 10));
                a.dot(&b)
            },
            100,
        );

        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.ops_per_second > 0.0);
    }
}
