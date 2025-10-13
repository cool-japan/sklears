//! Timing utilities for embedding speed benchmarks
//!
//! This module provides comprehensive timing and profiling utilities
//! for measuring the performance of manifold learning algorithms.
//! It includes high-precision timing, statistical analysis, and
//! comparative performance measurement capabilities.

use std::fmt;

/// High-precision timer for measuring algorithm performance
use sklears_core::traits::Estimator;
use std::collections::HashMap;
use std::time::{Duration, Instant};
#[derive(Debug, Clone)]
pub struct PerformanceTimer {
    start_time: Option<Instant>,
    end_time: Option<Instant>,
    checkpoints: HashMap<String, Instant>,
    durations: HashMap<String, Duration>,
    metadata: HashMap<String, String>,
}

impl PerformanceTimer {
    /// Create a new performance timer
    pub fn new() -> Self {
        Self {
            start_time: None,
            end_time: None,
            checkpoints: HashMap::new(),
            durations: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.end_time = None;
        self.checkpoints.clear();
        self.durations.clear();
    }

    /// Stop timing and return total duration
    pub fn stop(&mut self) -> Duration {
        self.end_time = Some(Instant::now());
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => end.duration_since(start),
            _ => Duration::default(),
        }
    }

    /// Add a checkpoint with a label
    pub fn checkpoint(&mut self, label: &str) {
        if self.start_time.is_some() {
            self.checkpoints.insert(label.to_string(), Instant::now());
        }
    }

    /// Get duration from start to a checkpoint
    pub fn duration_to_checkpoint(&self, label: &str) -> Option<Duration> {
        match (self.start_time, self.checkpoints.get(label)) {
            (Some(start), Some(checkpoint)) => Some(checkpoint.duration_since(start)),
            _ => None,
        }
    }

    /// Get duration between two checkpoints
    pub fn duration_between_checkpoints(&self, from: &str, to: &str) -> Option<Duration> {
        match (self.checkpoints.get(from), self.checkpoints.get(to)) {
            (Some(start), Some(end)) => Some(end.duration_since(*start)),
            _ => None,
        }
    }

    /// Add custom metadata
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> Option<Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            (Some(start), None) => Some(Instant::now().duration_since(start)),
            _ => None,
        }
    }

    /// Create a timing report
    pub fn create_report(&self) -> TimingReport {
        let total_duration = self.elapsed().unwrap_or_default();

        let mut checkpoint_durations = HashMap::new();
        for label in self.checkpoints.keys() {
            if let Some(duration) = self.duration_to_checkpoint(label) {
                checkpoint_durations.insert(label.clone(), duration);
            }
        }

        /// TimingReport
        TimingReport {
            total_duration,
            checkpoint_durations,
            metadata: self.metadata.clone(),
        }
    }
}

impl Default for PerformanceTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive timing report
#[derive(Debug, Clone)]
pub struct TimingReport {
    /// total_duration
    pub total_duration: Duration,
    /// checkpoint_durations
    pub checkpoint_durations: HashMap<String, Duration>,
    /// metadata
    pub metadata: HashMap<String, String>,
}

impl TimingReport {
    /// Get timing summary as string
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Total Duration: {:.3}s\n",
            self.total_duration.as_secs_f64()
        );

        if !self.checkpoint_durations.is_empty() {
            summary.push_str("\nCheckpoints:\n");
            let mut checkpoints: Vec<_> = self.checkpoint_durations.iter().collect();
            checkpoints.sort_by_key(|(_, duration)| *duration);

            for (label, duration) in checkpoints {
                summary.push_str(&format!("  {}: {:.3}s\n", label, duration.as_secs_f64()));
            }
        }

        if !self.metadata.is_empty() {
            summary.push_str("\nMetadata:\n");
            for (key, value) in &self.metadata {
                summary.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        summary
    }

    /// Get timing as milliseconds
    pub fn total_ms(&self) -> f64 {
        self.total_duration.as_secs_f64() * 1000.0
    }

    /// Get specific checkpoint duration in milliseconds
    pub fn checkpoint_ms(&self, label: &str) -> Option<f64> {
        self.checkpoint_durations
            .get(label)
            .map(|d| d.as_secs_f64() * 1000.0)
    }
}

impl fmt::Display for TimingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Statistical timing analyzer for multiple runs
#[derive(Debug)]
pub struct TimingStatistics {
    measurements: Vec<Duration>,
    algorithm_name: String,
    dataset_info: DatasetInfo,
}

impl TimingStatistics {
    /// Create new timing statistics collector
    pub fn new(algorithm_name: &str, dataset_info: DatasetInfo) -> Self {
        Self {
            measurements: Vec::new(),
            algorithm_name: algorithm_name.to_string(),
            dataset_info,
        }
    }

    /// Add a timing measurement
    pub fn add_measurement(&mut self, duration: Duration) {
        self.measurements.push(duration);
    }

    /// Calculate mean duration
    pub fn mean(&self) -> Duration {
        if self.measurements.is_empty() {
            return Duration::default();
        }

        let total_nanos: u64 = self.measurements.iter().map(|d| d.as_nanos() as u64).sum();
        Duration::from_nanos(total_nanos / self.measurements.len() as u64)
    }

    /// Calculate median duration
    pub fn median(&self) -> Duration {
        if self.measurements.is_empty() {
            return Duration::default();
        }

        let mut sorted_measurements = self.measurements.clone();
        sorted_measurements.sort();

        let mid = sorted_measurements.len() / 2;
        if sorted_measurements.len() % 2 == 0 {
            let sum_nanos =
                sorted_measurements[mid - 1].as_nanos() + sorted_measurements[mid].as_nanos();
            Duration::from_nanos((sum_nanos / 2) as u64)
        } else {
            sorted_measurements[mid]
        }
    }

    /// Calculate standard deviation
    pub fn std_dev(&self) -> Duration {
        if self.measurements.len() < 2 {
            return Duration::default();
        }

        let mean = self.mean();
        let mean_nanos = mean.as_nanos() as f64;

        let variance: f64 = self
            .measurements
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / (self.measurements.len() - 1) as f64;

        Duration::from_nanos(variance.sqrt() as u64)
    }

    /// Calculate minimum duration
    pub fn min(&self) -> Option<Duration> {
        self.measurements.iter().min().copied()
    }

    /// Calculate maximum duration
    pub fn max(&self) -> Option<Duration> {
        self.measurements.iter().max().copied()
    }

    /// Calculate 95th percentile
    pub fn percentile_95(&self) -> Duration {
        if self.measurements.is_empty() {
            return Duration::default();
        }

        let mut sorted_measurements = self.measurements.clone();
        sorted_measurements.sort();

        let index = ((sorted_measurements.len() as f64) * 0.95) as usize;
        sorted_measurements[index.min(sorted_measurements.len() - 1)]
    }

    /// Calculate throughput (samples per second)
    pub fn throughput(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }

        let mean_seconds = self.mean().as_secs_f64();
        if mean_seconds > 0.0 {
            self.dataset_info.n_samples as f64 / mean_seconds
        } else {
            0.0
        }
    }

    /// Calculate time per sample (microseconds)
    pub fn time_per_sample_us(&self) -> f64 {
        if self.measurements.is_empty() || self.dataset_info.n_samples == 0 {
            return 0.0;
        }

        let mean_us = self.mean().as_micros() as f64;
        mean_us / self.dataset_info.n_samples as f64
    }

    /// Generate statistical report
    pub fn generate_report(&self) -> StatisticalReport {
        /// StatisticalReport
        StatisticalReport {
            algorithm_name: self.algorithm_name.clone(),
            dataset_info: self.dataset_info.clone(),
            n_measurements: self.measurements.len(),
            mean: self.mean(),
            median: self.median(),
            std_dev: self.std_dev(),
            min: self.min().unwrap_or_default(),
            max: self.max().unwrap_or_default(),
            percentile_95: self.percentile_95(),
            throughput: self.throughput(),
            time_per_sample_us: self.time_per_sample_us(),
        }
    }
}

/// Dataset information for benchmarking
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// name
    pub name: String,
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
    /// n_components
    pub n_components: usize,
    /// complexity_class
    pub complexity_class: ComplexityClass,
}

impl DatasetInfo {
    pub fn new(
        name: &str,
        n_samples: usize,
        n_features: usize,
        n_components: usize,
        complexity_class: ComplexityClass,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_samples,
            n_features,
            n_components,
            complexity_class,
        }
    }
}

/// Algorithm complexity classification
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityClass {
    /// Linear
    Linear, // O(n)
    /// LogLinear
    LogLinear, // O(n log n)
    /// Quadratic
    Quadratic, // O(n²)
    /// Cubic
    Cubic, // O(n³)
    /// Polynomial
    Polynomial, // O(n^k) where k > 3
    /// Exponential
    Exponential, // O(2^n)
}

impl fmt::Display for ComplexityClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplexityClass::Linear => write!(f, "O(n)"),
            ComplexityClass::LogLinear => write!(f, "O(n log n)"),
            ComplexityClass::Quadratic => write!(f, "O(n²)"),
            ComplexityClass::Cubic => write!(f, "O(n³)"),
            ComplexityClass::Polynomial => write!(f, "O(n^k)"),
            ComplexityClass::Exponential => write!(f, "O(2^n)"),
        }
    }
}

/// Statistical report for algorithm performance
#[derive(Debug, Clone)]
pub struct StatisticalReport {
    /// algorithm_name
    pub algorithm_name: String,
    /// dataset_info
    pub dataset_info: DatasetInfo,
    /// n_measurements
    pub n_measurements: usize,
    /// mean
    pub mean: Duration,
    /// median
    pub median: Duration,
    /// std_dev
    pub std_dev: Duration,
    /// min
    pub min: Duration,
    /// max
    pub max: Duration,
    /// percentile_95
    pub percentile_95: Duration,
    /// throughput
    pub throughput: f64,
    /// time_per_sample_us
    pub time_per_sample_us: f64,
}

impl StatisticalReport {
    /// Generate a comprehensive summary
    pub fn summary(&self) -> String {
        format!(
            "Performance Report: {}\n\
             Dataset: {} ({}×{} → {})\n\
             Measurements: {}\n\
             Mean: {:.3}s (± {:.3}s)\n\
             Median: {:.3}s\n\
             Range: {:.3}s - {:.3}s\n\
             95th Percentile: {:.3}s\n\
             Throughput: {:.1} samples/sec\n\
             Time per sample: {:.2} μs\n\
             Expected complexity: {}",
            self.algorithm_name,
            self.dataset_info.name,
            self.dataset_info.n_samples,
            self.dataset_info.n_features,
            self.dataset_info.n_components,
            self.n_measurements,
            self.mean.as_secs_f64(),
            self.std_dev.as_secs_f64(),
            self.median.as_secs_f64(),
            self.min.as_secs_f64(),
            self.max.as_secs_f64(),
            self.percentile_95.as_secs_f64(),
            self.throughput,
            self.time_per_sample_us,
            self.dataset_info.complexity_class
        )
    }

    /// Calculate coefficient of variation (relative standard deviation)
    pub fn coefficient_of_variation(&self) -> f64 {
        let mean_ns = self.mean.as_nanos() as f64;
        let std_ns = self.std_dev.as_nanos() as f64;

        if mean_ns > 0.0 {
            std_ns / mean_ns
        } else {
            0.0
        }
    }

    /// Check if measurements are stable (low variance)
    pub fn is_stable(&self, threshold: f64) -> bool {
        self.coefficient_of_variation() < threshold
    }
}

impl fmt::Display for StatisticalReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Comparative performance analyzer
#[derive(Debug)]
pub struct ComparativeAnalyzer {
    reports: HashMap<String, StatisticalReport>,
}

impl ComparativeAnalyzer {
    /// Create new comparative analyzer
    pub fn new() -> Self {
        Self {
            reports: HashMap::new(),
        }
    }

    /// Add a statistical report
    pub fn add_report(&mut self, algorithm_name: &str, report: StatisticalReport) {
        self.reports.insert(algorithm_name.to_string(), report);
    }

    /// Find the fastest algorithm
    pub fn fastest_algorithm(&self) -> Option<(&String, &StatisticalReport)> {
        self.reports
            .iter()
            .min_by_key(|(_, report)| report.mean.as_nanos())
    }

    /// Find the most stable algorithm (lowest coefficient of variation)
    pub fn most_stable_algorithm(&self) -> Option<(&String, &StatisticalReport)> {
        self.reports.iter().min_by(|(_, a), (_, b)| {
            a.coefficient_of_variation()
                .partial_cmp(&b.coefficient_of_variation())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Calculate speedup relative to a baseline algorithm
    pub fn speedup_vs_baseline(&self, baseline: &str) -> HashMap<String, f64> {
        let mut speedups = HashMap::new();

        if let Some(baseline_report) = self.reports.get(baseline) {
            let baseline_time = baseline_report.mean.as_secs_f64();

            for (name, report) in &self.reports {
                if name != baseline {
                    let algorithm_time = report.mean.as_secs_f64();
                    if algorithm_time > 0.0 {
                        speedups.insert(name.clone(), baseline_time / algorithm_time);
                    }
                }
            }
        }

        speedups
    }

    /// Generate comprehensive comparison report
    pub fn comparison_report(&self, baseline: Option<&str>) -> String {
        let mut report = String::new();
        report.push_str("Comparative Performance Analysis\n");
        report.push_str("================================\n\n");

        if self.reports.is_empty() {
            report.push_str("No algorithms analyzed.\n");
            return report;
        }

        // Summary table
        report.push_str("Summary Table:\n");
        report.push_str(
            "Algorithm                 | Mean Time | Throughput   | CV     | Stability\n",
        );
        report.push_str(
            "-------------------------|-----------|--------------|--------|-----------\n",
        );

        let mut sorted_reports: Vec<_> = self.reports.iter().collect();
        sorted_reports.sort_by_key(|(_, report)| report.mean.as_nanos());

        for (name, stat_report) in &sorted_reports {
            let stability = if stat_report.is_stable(0.1) {
                "Stable"
            } else {
                "Variable"
            };
            report.push_str(&format!(
                "{:<24} | {:>8.3}s | {:>10.1}/s | {:>5.1}% | {}\n",
                name,
                stat_report.mean.as_secs_f64(),
                stat_report.throughput,
                stat_report.coefficient_of_variation() * 100.0,
                stability
            ));
        }

        report.push('\n');

        // Speedup analysis
        if let Some(baseline_algo) = baseline {
            if self.reports.contains_key(baseline_algo) {
                report.push_str(&format!("Speedup vs {} (baseline):\n", baseline_algo));
                let speedups = self.speedup_vs_baseline(baseline_algo);

                let mut speedup_vec: Vec<_> = speedups.iter().collect();
                speedup_vec
                    .sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (name, speedup) in speedup_vec {
                    report.push_str(&format!("  {}: {:.2}x\n", name, speedup));
                }
                report.push('\n');
            }
        }

        // Best performers
        if let Some((name, _)) = self.fastest_algorithm() {
            report.push_str(&format!("Fastest Algorithm: {}\n", name));
        }

        if let Some((name, _)) = self.most_stable_algorithm() {
            report.push_str(&format!("Most Stable Algorithm: {}\n", name));
        }

        report
    }
}

impl Default for ComparativeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for measuring specific algorithm phases
pub struct PhaseTimer {
    timers: HashMap<String, PerformanceTimer>,
    current_phase: Option<String>,
}

impl PhaseTimer {
    /// Create new phase timer
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            current_phase: None,
        }
    }

    /// Start timing a specific phase
    pub fn start_phase(&mut self, phase_name: &str) {
        let mut timer = PerformanceTimer::new();
        timer.start();
        self.timers.insert(phase_name.to_string(), timer);
        self.current_phase = Some(phase_name.to_string());
    }

    /// End the current phase
    pub fn end_phase(&mut self) -> Option<Duration> {
        if let Some(phase_name) = &self.current_phase {
            if let Some(timer) = self.timers.get_mut(phase_name) {
                let duration = timer.stop();
                self.current_phase = None;
                return Some(duration);
            }
        }
        None
    }

    /// Add checkpoint to current phase
    pub fn checkpoint(&mut self, label: &str) {
        if let Some(phase_name) = &self.current_phase {
            if let Some(timer) = self.timers.get_mut(phase_name) {
                timer.checkpoint(label);
            }
        }
    }

    /// Get timing report for all phases
    pub fn phase_report(&self) -> HashMap<String, TimingReport> {
        self.timers
            .iter()
            .map(|(name, timer)| (name.clone(), timer.create_report()))
            .collect()
    }

    /// Get total time across all phases
    pub fn total_time(&self) -> Duration {
        self.timers
            .values()
            .map(|timer| timer.elapsed().unwrap_or_default())
            .sum()
    }
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro for convenient timing of code blocks
#[macro_export]
macro_rules! time_block {
    ($timer:expr, $label:expr, $block:block) => {{
        $timer.checkpoint(&format!("start_{}", $label));
        let result = $block;
        $timer.checkpoint(&format!("end_{}", $label));
        result
    }};
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_timer() {
        let mut timer = PerformanceTimer::new();
        timer.start();

        thread::sleep(Duration::from_millis(10));
        timer.checkpoint("checkpoint1");

        thread::sleep(Duration::from_millis(10));
        let total = timer.stop();

        assert!(total >= Duration::from_millis(20));
        assert!(timer.duration_to_checkpoint("checkpoint1").unwrap() >= Duration::from_millis(10));
    }

    #[test]
    fn test_timing_statistics() {
        let dataset_info = DatasetInfo::new("test", 100, 10, 2, ComplexityClass::Linear);
        let mut stats = TimingStatistics::new("test_algorithm", dataset_info);

        stats.add_measurement(Duration::from_millis(100));
        stats.add_measurement(Duration::from_millis(120));
        stats.add_measurement(Duration::from_millis(80));

        let mean = stats.mean();
        assert_eq!(mean, Duration::from_millis(100));

        let median = stats.median();
        assert_eq!(median, Duration::from_millis(100));

        assert!(stats.throughput() > 0.0);
    }

    #[test]
    fn test_comparative_analyzer() {
        let mut analyzer = ComparativeAnalyzer::new();

        let dataset_info = DatasetInfo::new("test", 100, 10, 2, ComplexityClass::Linear);

        let mut stats1 = TimingStatistics::new("fast_algo", dataset_info.clone());
        stats1.add_measurement(Duration::from_millis(50));
        let report1 = stats1.generate_report();

        let mut stats2 = TimingStatistics::new("slow_algo", dataset_info);
        stats2.add_measurement(Duration::from_millis(100));
        let report2 = stats2.generate_report();

        analyzer.add_report("fast_algo", report1);
        analyzer.add_report("slow_algo", report2);

        let (fastest_name, _) = analyzer.fastest_algorithm().unwrap();
        assert_eq!(fastest_name, "fast_algo");

        let speedups = analyzer.speedup_vs_baseline("slow_algo");
        assert!(speedups.get("fast_algo").unwrap() > &1.5);
    }

    #[test]
    fn test_phase_timer() {
        let mut phase_timer = PhaseTimer::new();

        phase_timer.start_phase("phase1");
        thread::sleep(Duration::from_millis(10));
        phase_timer.checkpoint("middle");
        thread::sleep(Duration::from_millis(10));
        let duration = phase_timer.end_phase().unwrap();

        assert!(duration >= Duration::from_millis(20));

        let report = phase_timer.phase_report();
        assert!(report.contains_key("phase1"));
    }
}
