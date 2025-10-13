//! Performance measurement and profiling utilities
//!
//! This module provides utilities for measuring performance, memory usage,
//! and timing operations in machine learning workflows.

use crate::{UtilsError, UtilsResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Timer for measuring execution time
#[derive(Debug, Clone)]
pub struct Timer {
    start: Option<Instant>,
    measurements: HashMap<String, Duration>,
}

impl Timer {
    /// Create a new timer
    pub fn new() -> Self {
        Self {
            start: None,
            measurements: HashMap::new(),
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop timing and return elapsed duration
    pub fn stop(&mut self) -> UtilsResult<Duration> {
        let start = self
            .start
            .take()
            .ok_or_else(|| UtilsError::InvalidParameter("Timer not started".to_string()))?;
        Ok(start.elapsed())
    }

    /// Stop timing and store result with a label
    pub fn stop_and_store(&mut self, label: String) -> UtilsResult<Duration> {
        let duration = self.stop()?;
        self.measurements.insert(label, duration);
        Ok(duration)
    }

    /// Time a closure and return the result and duration
    pub fn time<F, R>(&mut self, f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Time a closure, store the result with a label, and return the result
    pub fn time_and_store<F, R>(&mut self, label: String, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let (result, duration) = self.time(f);
        self.measurements.insert(label, duration);
        result
    }

    /// Get all measurements
    pub fn measurements(&self) -> &HashMap<String, Duration> {
        &self.measurements
    }

    /// Get a specific measurement
    pub fn get_measurement(&self, label: &str) -> Option<Duration> {
        self.measurements.get(label).copied()
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }

    /// Get summary statistics
    pub fn summary(&self) -> TimerSummary {
        let mut total = Duration::from_secs(0);
        let mut min = Duration::from_secs(u64::MAX);
        let mut max = Duration::from_secs(0);

        for &duration in self.measurements.values() {
            total += duration;
            min = min.min(duration);
            max = max.max(duration);
        }

        let count = self.measurements.len();
        let average = if count > 0 {
            total / count as u32
        } else {
            Duration::from_secs(0)
        };

        TimerSummary {
            count,
            total,
            average,
            min: if min == Duration::from_secs(u64::MAX) {
                Duration::from_secs(0)
            } else {
                min
            },
            max,
        }
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for timer measurements
#[derive(Debug, Clone, PartialEq)]
pub struct TimerSummary {
    pub count: usize,
    pub total: Duration,
    pub average: Duration,
    pub min: Duration,
    pub max: Duration,
}

/// Memory usage tracker
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    baseline: Option<usize>,
    measurements: HashMap<String, usize>,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            baseline: None,
            measurements: HashMap::new(),
        }
    }

    /// Set baseline memory usage
    pub fn set_baseline(&mut self) {
        if let Ok(usage) = get_memory_usage() {
            self.baseline = Some(usage);
        }
    }

    /// Record current memory usage with a label
    pub fn record(&mut self, label: String) -> UtilsResult<usize> {
        let usage = get_memory_usage()?;
        self.measurements.insert(label, usage);
        Ok(usage)
    }

    /// Get memory usage relative to baseline
    pub fn relative_usage(&self) -> UtilsResult<Option<isize>> {
        if let Some(baseline) = self.baseline {
            let current = get_memory_usage()?;
            Ok(Some(current as isize - baseline as isize))
        } else {
            Ok(None)
        }
    }

    /// Get all measurements
    pub fn measurements(&self) -> &HashMap<String, usize> {
        &self.measurements
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> Option<usize> {
        self.measurements.values().max().copied()
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.baseline = None;
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance profiler combining timing and memory tracking
#[derive(Debug, Clone)]
pub struct Profiler {
    timer: Timer,
    memory: MemoryTracker,
    active_sessions: HashMap<String, (Instant, usize)>,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            timer: Timer::new(),
            memory: MemoryTracker::new(),
            active_sessions: HashMap::new(),
        }
    }

    /// Start a profiling session
    pub fn start_session(&mut self, name: String) -> UtilsResult<()> {
        let start_time = Instant::now();
        let start_memory = get_memory_usage()?;
        self.active_sessions
            .insert(name, (start_time, start_memory));
        Ok(())
    }

    /// End a profiling session and record results
    pub fn end_session(&mut self, name: &str) -> UtilsResult<ProfileResult> {
        let (start_time, start_memory) = self
            .active_sessions
            .remove(name)
            .ok_or_else(|| UtilsError::InvalidParameter(format!("Session '{name}' not found")))?;

        let duration = start_time.elapsed();
        let end_memory = get_memory_usage()?;
        let memory_delta = end_memory as isize - start_memory as isize;

        self.timer.measurements.insert(name.to_string(), duration);
        self.memory
            .measurements
            .insert(name.to_string(), end_memory);

        Ok(ProfileResult {
            name: name.to_string(),
            duration,
            memory_delta,
            start_memory,
            end_memory,
        })
    }

    /// Profile a closure
    pub fn profile<F, R>(&mut self, name: String, f: F) -> UtilsResult<(R, ProfileResult)>
    where
        F: FnOnce() -> R,
    {
        self.start_session(name.clone())?;
        let result = f();
        let profile_result = self.end_session(&name)?;
        Ok((result, profile_result))
    }

    /// Get timer reference
    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    /// Get memory tracker reference
    pub fn memory(&self) -> &MemoryTracker {
        &self.memory
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.timer.clear();
        self.memory.clear();
        self.active_sessions.clear();
    }

    /// Generate a performance report
    pub fn report(&self) -> ProfileReport {
        ProfileReport {
            timer_summary: self.timer.summary(),
            peak_memory: self.memory.peak_usage(),
            total_sessions: self.timer.measurements.len(),
            active_sessions: self.active_sessions.len(),
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from a profiling session
#[derive(Debug, Clone)]
pub struct ProfileResult {
    pub name: String,
    pub duration: Duration,
    pub memory_delta: isize,
    pub start_memory: usize,
    pub end_memory: usize,
}

/// Overall performance report
#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub timer_summary: TimerSummary,
    pub peak_memory: Option<usize>,
    pub total_sessions: usize,
    pub active_sessions: usize,
}

/// Benchmark runner for repeated measurements
#[derive(Debug)]
pub struct Benchmark {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
    measurements: Vec<Duration>,
}

impl Benchmark {
    /// Create a new benchmark
    pub fn new(name: String, iterations: usize) -> Self {
        Self {
            name,
            iterations,
            warmup_iterations: 10,
            measurements: Vec::with_capacity(iterations),
        }
    }

    /// Set number of warmup iterations
    pub fn with_warmup(mut self, warmup_iterations: usize) -> Self {
        self.warmup_iterations = warmup_iterations;
        self
    }

    /// Run the benchmark
    pub fn run<F>(&mut self, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup phase
        for _ in 0..self.warmup_iterations {
            f();
        }

        // Measurement phase
        self.measurements.clear();
        for _ in 0..self.iterations {
            let start = Instant::now();
            f();
            let duration = start.elapsed();
            self.measurements.push(duration);
        }

        self.analyze()
    }

    fn analyze(&self) -> BenchmarkResult {
        let mut measurements = self.measurements.clone();
        measurements.sort();

        let len = measurements.len();
        let total: Duration = measurements.iter().sum();
        let average = total / len as u32;

        let median = if len % 2 == 0 {
            (measurements[len / 2 - 1] + measurements[len / 2]) / 2
        } else {
            measurements[len / 2]
        };

        let min = measurements[0];
        let max = measurements[len - 1];

        // Calculate percentiles
        let p95_index = ((len as f64) * 0.95) as usize;
        let p99_index = ((len as f64) * 0.99) as usize;
        let p95 = measurements[p95_index.min(len - 1)];
        let p99 = measurements[p99_index.min(len - 1)];

        // Calculate standard deviation
        let variance_sum: f64 = measurements
            .iter()
            .map(|&d| {
                let diff = d.as_secs_f64() - average.as_secs_f64();
                diff * diff
            })
            .sum();
        let variance = variance_sum / len as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        BenchmarkResult {
            name: self.name.clone(),
            iterations: len,
            total,
            average,
            median,
            min,
            max,
            std_dev,
            p95,
            p99,
        }
    }
}

/// Benchmark analysis results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total: Duration,
    pub average: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

impl BenchmarkResult {
    /// Format results as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Benchmark: {}\n\
             Iterations: {}\n\
             Average: {:?}\n\
             Median: {:?}\n\
             Min: {:?}\n\
             Max: {:?}\n\
             Std Dev: {:?}\n\
             95th percentile: {:?}\n\
             99th percentile: {:?}",
            self.name,
            self.iterations,
            self.average,
            self.median,
            self.min,
            self.max,
            self.std_dev,
            self.p95,
            self.p99
        )
    }
}

/// Get current memory usage in bytes
fn get_memory_usage() -> UtilsResult<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status").map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to read memory info: {e}"))
        })?;

        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let kb: usize = parts[1].parse().map_err(|e| {
                        UtilsError::InvalidParameter(format!("Failed to parse memory: {e}"))
                    })?;
                    return Ok(kb * 1024); // Convert KB to bytes
                }
            }
        }
        Err(UtilsError::InvalidParameter(
            "Memory info not found".to_string(),
        ))
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, we'll use a simpler approach with task_info
        // For now, return a placeholder value
        use std::process::Command;
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output()
            .map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to get memory usage: {e}"))
            })?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        let kb: usize = output_str
            .trim()
            .parse()
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to parse memory: {e}")))?;
        Ok(kb * 1024) // Convert KB to bytes
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        // For other platforms, return a default value
        Ok(0)
    }
}

/// Performance regression detection system
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    baselines: HashMap<String, BaselineMetrics>,
    threshold_factor: f64,
    min_samples: usize,
}

/// Baseline performance metrics for regression detection
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    pub average_duration: Duration,
    pub std_dev: Duration,
    pub sample_count: usize,
    pub last_updated: std::time::SystemTime,
    pub historical_durations: Vec<Duration>,
}

/// Regression detection result
#[derive(Debug, Clone)]
pub struct RegressionResult {
    pub test_name: String,
    pub current_duration: Duration,
    pub baseline_average: Duration,
    pub deviation_factor: f64,
    pub is_regression: bool,
    pub confidence_level: f64,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            threshold_factor: 1.5, // 50% slower is considered a regression
            min_samples: 5,
        }
    }

    /// Set the regression threshold factor (e.g., 1.5 = 50% slower)
    pub fn with_threshold(mut self, threshold_factor: f64) -> Self {
        self.threshold_factor = threshold_factor;
        self
    }

    /// Set minimum samples required for regression detection
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Record a baseline measurement
    pub fn record_baseline(&mut self, test_name: String, duration: Duration) {
        let test_name_clone = test_name.clone();
        let entry = self
            .baselines
            .entry(test_name)
            .or_insert_with(|| BaselineMetrics {
                average_duration: Duration::from_secs(0),
                std_dev: Duration::from_secs(0),
                sample_count: 0,
                last_updated: std::time::SystemTime::now(),
                historical_durations: Vec::new(),
            });

        entry.historical_durations.push(duration);
        entry.sample_count += 1;
        entry.last_updated = std::time::SystemTime::now();

        // Keep only the last 100 measurements to avoid unbounded growth
        if entry.historical_durations.len() > 100 {
            entry.historical_durations.remove(0);
        }

        // Recalculate statistics - work with a cloned entry to avoid borrowing issues
        let mut entry_clone = entry.clone();
        let _ = entry; // Release the mutable reference to self.baselines
        self.update_statistics(&mut entry_clone, &test_name_clone);

        // Update the entry back in the map
        if let Some(stored_entry) = self.baselines.get_mut(&test_name_clone) {
            *stored_entry = entry_clone;
        }
    }

    /// Check for performance regression
    pub fn check_regression(
        &self,
        test_name: &str,
        current_duration: Duration,
    ) -> Option<RegressionResult> {
        let baseline = self.baselines.get(test_name)?;

        if baseline.sample_count < self.min_samples {
            return None;
        }

        let current_secs = current_duration.as_secs_f64();
        let baseline_secs = baseline.average_duration.as_secs_f64();
        let deviation_factor = current_secs / baseline_secs;

        let is_regression = deviation_factor > self.threshold_factor;

        // Calculate confidence level using t-test-like approach
        let std_dev_secs = baseline.std_dev.as_secs_f64();
        let z_score =
            (current_secs - baseline_secs) / (std_dev_secs / (baseline.sample_count as f64).sqrt());
        let confidence_level = if z_score > 0.0 {
            1.0 - (-z_score * z_score / 2.0).exp() // Approximation of normal CDF
        } else {
            0.0
        };

        Some(RegressionResult {
            test_name: test_name.to_string(),
            current_duration,
            baseline_average: baseline.average_duration,
            deviation_factor,
            is_regression,
            confidence_level,
        })
    }

    /// Get baseline metrics for a test
    pub fn get_baseline(&self, test_name: &str) -> Option<&BaselineMetrics> {
        self.baselines.get(test_name)
    }

    /// List all tracked tests
    pub fn tracked_tests(&self) -> Vec<&String> {
        self.baselines.keys().collect()
    }

    /// Clear all baselines
    pub fn clear_baselines(&mut self) {
        self.baselines.clear();
    }

    /// Update statistics for a baseline
    fn update_statistics(&mut self, baseline: &mut BaselineMetrics, test_name: &str) {
        if baseline.historical_durations.is_empty() {
            return;
        }

        // Calculate average
        let total: Duration = baseline.historical_durations.iter().sum();
        baseline.average_duration = total / baseline.historical_durations.len() as u32;

        // Calculate standard deviation
        let mean_secs = baseline.average_duration.as_secs_f64();
        let variance_sum: f64 = baseline
            .historical_durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum();

        let variance = variance_sum / baseline.historical_durations.len() as f64;
        baseline.std_dev = Duration::from_secs_f64(variance.sqrt());

        // Update the entry in the map
        self.baselines
            .insert(test_name.to_string(), baseline.clone());
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RegressionResult {
    /// Format the regression result as a human-readable string
    pub fn format(&self) -> String {
        let status = if self.is_regression {
            "REGRESSION DETECTED"
        } else {
            "OK"
        };
        format!(
            "Test: {} [{}]\n\
             Current: {:?}\n\
             Baseline: {:?}\n\
             Deviation: {:.2}x\n\
             Confidence: {:.1}%",
            self.test_name,
            status,
            self.current_duration,
            self.baseline_average,
            self.deviation_factor,
            self.confidence_level * 100.0
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timer_basic() {
        let mut timer = Timer::new();

        timer.start();
        thread::sleep(Duration::from_millis(10));
        let duration = timer.stop().unwrap();

        assert!(duration >= Duration::from_millis(10));
        assert!(duration < Duration::from_millis(50)); // Allow some tolerance
    }

    #[test]
    fn test_timer_closure() {
        let mut timer = Timer::new();

        let (result, duration) = timer.time(|| {
            thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_timer_store() {
        let mut timer = Timer::new();

        timer.start();
        thread::sleep(Duration::from_millis(10));
        timer.stop_and_store("test".to_string()).unwrap();

        assert!(timer.get_measurement("test").is_some());
        assert!(timer.get_measurement("test").unwrap() >= Duration::from_millis(10));
    }

    #[test]
    fn test_timer_summary() {
        let mut timer = Timer::new();

        timer
            .measurements
            .insert("test1".to_string(), Duration::from_millis(100));
        timer
            .measurements
            .insert("test2".to_string(), Duration::from_millis(200));
        timer
            .measurements
            .insert("test3".to_string(), Duration::from_millis(300));

        let summary = timer.summary();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.min, Duration::from_millis(100));
        assert_eq!(summary.max, Duration::from_millis(300));
        assert_eq!(summary.average, Duration::from_millis(200));
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();

        tracker.set_baseline();
        tracker.record("test".to_string()).unwrap();

        assert!(tracker.measurements().contains_key("test"));
        assert!(tracker.peak_usage().is_some());
    }

    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();

        let (result, profile_result) = profiler
            .profile("test".to_string(), || {
                thread::sleep(Duration::from_millis(10));
                42
            })
            .unwrap();

        assert_eq!(result, 42);
        assert_eq!(profile_result.name, "test");
        assert!(profile_result.duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_benchmark() {
        let mut benchmark = Benchmark::new("test_benchmark".to_string(), 10).with_warmup(2);

        let result = benchmark.run(|| {
            thread::sleep(Duration::from_millis(1));
        });

        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.iterations, 10);
        assert!(result.average >= Duration::from_millis(1));
        assert!(result.min <= result.median);
        assert!(result.median <= result.max);
    }

    #[test]
    fn test_benchmark_result_format() {
        let result = BenchmarkResult {
            name: "test".to_string(),
            iterations: 100,
            total: Duration::from_millis(1000),
            average: Duration::from_millis(10),
            median: Duration::from_millis(9),
            min: Duration::from_millis(5),
            max: Duration::from_millis(20),
            std_dev: Duration::from_millis(2),
            p95: Duration::from_millis(15),
            p99: Duration::from_millis(18),
        };

        let formatted = result.format();
        assert!(formatted.contains("test"));
        assert!(formatted.contains("100"));
        assert!(formatted.contains("10ms"));
    }

    #[test]
    fn test_profiler_sessions() {
        let mut profiler = Profiler::new();

        profiler.start_session("session1".to_string()).unwrap();
        thread::sleep(Duration::from_millis(10));
        let result = profiler.end_session("session1").unwrap();

        assert_eq!(result.name, "session1");
        assert!(result.duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_report() {
        let mut profiler = Profiler::new();

        profiler.profile("test1".to_string(), || {}).unwrap();
        profiler.profile("test2".to_string(), || {}).unwrap();

        let report = profiler.report();
        assert_eq!(report.total_sessions, 2);
        assert_eq!(report.active_sessions, 0);
    }

    #[test]
    fn test_regression_detector_baseline() {
        let mut detector = RegressionDetector::new();

        // Record several baseline measurements
        for i in 0..10 {
            detector.record_baseline("test_func".to_string(), Duration::from_millis(100 + i));
        }

        let baseline = detector.get_baseline("test_func").unwrap();
        assert_eq!(baseline.sample_count, 10);
        assert!(baseline.average_duration >= Duration::from_millis(100));
        assert!(baseline.average_duration <= Duration::from_millis(110));
    }

    #[test]
    fn test_regression_detector_no_regression() {
        let mut detector = RegressionDetector::new().with_min_samples(3);

        // Record baseline measurements around 100ms
        for _ in 0..5 {
            detector.record_baseline("test_func".to_string(), Duration::from_millis(100));
        }

        // Test with similar performance (no regression)
        let result = detector.check_regression("test_func", Duration::from_millis(105));
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(!result.is_regression);
        assert_eq!(result.test_name, "test_func");
    }

    #[test]
    fn test_regression_detector_with_regression() {
        let mut detector = RegressionDetector::new()
            .with_min_samples(3)
            .with_threshold(1.5);

        // Record baseline measurements around 100ms
        for _ in 0..5 {
            detector.record_baseline("test_func".to_string(), Duration::from_millis(100));
        }

        // Test with 2x slower performance (regression)
        let result = detector.check_regression("test_func", Duration::from_millis(200));
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.is_regression);
        assert!(result.deviation_factor > 1.5);
    }

    #[test]
    fn test_regression_detector_insufficient_samples() {
        let mut detector = RegressionDetector::new().with_min_samples(5);

        // Record only 2 baseline measurements (less than min_samples)
        for _ in 0..2 {
            detector.record_baseline("test_func".to_string(), Duration::from_millis(100));
        }

        // Should return None due to insufficient samples
        let result = detector.check_regression("test_func", Duration::from_millis(200));
        assert!(result.is_none());
    }

    #[test]
    fn test_regression_detector_tracked_tests() {
        let mut detector = RegressionDetector::new();

        detector.record_baseline("test1".to_string(), Duration::from_millis(100));
        detector.record_baseline("test2".to_string(), Duration::from_millis(200));

        let tracked = detector.tracked_tests();
        assert_eq!(tracked.len(), 2);
        assert!(tracked.contains(&&"test1".to_string()));
        assert!(tracked.contains(&&"test2".to_string()));
    }

    #[test]
    fn test_regression_result_format() {
        let result = RegressionResult {
            test_name: "test_function".to_string(),
            current_duration: Duration::from_millis(200),
            baseline_average: Duration::from_millis(100),
            deviation_factor: 2.0,
            is_regression: true,
            confidence_level: 0.95,
        };

        let formatted = result.format();
        assert!(formatted.contains("test_function"));
        assert!(formatted.contains("REGRESSION DETECTED"));
        assert!(formatted.contains("2.00x"));
        assert!(formatted.contains("95.0%"));
    }

    #[test]
    fn test_regression_detector_clear() {
        let mut detector = RegressionDetector::new();

        detector.record_baseline("test1".to_string(), Duration::from_millis(100));
        detector.record_baseline("test2".to_string(), Duration::from_millis(200));

        assert_eq!(detector.tracked_tests().len(), 2);

        detector.clear_baselines();
        assert_eq!(detector.tracked_tests().len(), 0);
    }
}
