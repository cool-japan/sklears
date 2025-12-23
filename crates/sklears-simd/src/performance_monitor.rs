//! Performance monitoring and tracking utilities
//!
//! Provides automated performance tracking, historical analysis, and continuous integration support
//! for SIMD performance optimization.

use crate::benchmark_framework::{BenchmarkResult, BenchmarkSuite};

#[cfg(not(feature = "no-std"))]
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::Path,
    string::ToString,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "no-std")]
use alloc::collections::{BTreeMap as HashMap, BTreeSet as HashSet};
#[cfg(feature = "no-std")]
use alloc::format;
#[cfg(feature = "no-std")]
use alloc::string::{String, ToString};
#[cfg(feature = "no-std")]
use alloc::vec::Vec;

// Mock types for no-std compatibility
#[cfg(feature = "no-std")]
type SystemTime = u64;
#[cfg(feature = "no-std")]
type OpenOptions = ();
#[cfg(feature = "no-std")]
type File = ();
#[cfg(feature = "no-std")]
type BufReader = ();
#[cfg(feature = "no-std")]
type Path = str;

// Helper functions for no-std compatibility
#[cfg(feature = "no-std")]
fn current_timestamp() -> u64 {
    // Mock timestamp for no-std (could use a counter or external time source)
    0
}

// Error type for performance monitoring operations
#[derive(Debug)]
pub enum PerformanceError {
    #[cfg(not(feature = "no-std"))]
    IoError(std::io::Error),
    Message(String),
}

#[cfg(not(feature = "no-std"))]
impl From<std::io::Error> for PerformanceError {
    fn from(error: std::io::Error) -> Self {
        PerformanceError::IoError(error)
    }
}

/// Performance monitor for tracking results over time
pub struct PerformanceMonitor {
    results_file: String,
    historical_data: Vec<PerformanceRecord>,
    thresholds: PerformanceThresholds,
}

/// Historical performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub timestamp: u64,
    pub git_commit: Option<String>,
    pub operation: String,
    pub duration_ns: u64,
    pub throughput: Option<f64>,
    pub architecture: String,
    pub simd_width: usize,
    pub optimization_level: String,
}

/// Performance thresholds for alerting
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub regression_threshold: f64,  // percentage
    pub improvement_threshold: f64, // percentage
    pub critical_slowdown: f64,     // percentage
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            regression_threshold: 5.0,   // 5% regression threshold
            improvement_threshold: 10.0, // 10% improvement threshold
            critical_slowdown: 25.0,     // 25% critical slowdown
        }
    }
}

/// Performance alert
#[derive(Debug)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub operation: String,
    pub current_performance: f64,
    pub baseline_performance: f64,
    pub change_percent: f64,
    pub severity: AlertSeverity,
    pub recommendation: String,
}

#[derive(Debug)]
pub enum AlertType {
    Regression,
    Improvement,
    CriticalSlowdown,
    Anomaly,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    #[cfg(not(feature = "no-std"))]
    pub fn new(results_file: &str) -> std::io::Result<Self> {
        let mut monitor = Self {
            results_file: results_file.to_string(),
            historical_data: Vec::new(),
            thresholds: PerformanceThresholds::default(),
        };

        monitor.load_historical_data()?;
        Ok(monitor)
    }

    /// Create a new performance monitor (no-std version)
    #[cfg(feature = "no-std")]
    pub fn new(results_file: &str) -> Result<Self, &'static str> {
        let mut monitor = Self {
            results_file: results_file.to_string(),
            historical_data: Vec::new(),
            thresholds: PerformanceThresholds::default(),
        };

        monitor.load_historical_data()?;
        Ok(monitor)
    }

    /// Set custom performance thresholds
    pub fn set_thresholds(&mut self, thresholds: PerformanceThresholds) {
        self.thresholds = thresholds;
    }

    /// Record new performance results
    #[cfg(not(feature = "no-std"))]
    pub fn record_results(
        &mut self,
        results: &[BenchmarkResult],
        git_commit: Option<String>,
    ) -> Result<(), PerformanceError> {
        #[cfg(not(feature = "no-std"))]
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        #[cfg(feature = "no-std")]
        let timestamp = current_timestamp();

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.results_file)?;

        for result in results {
            let record = PerformanceRecord {
                timestamp,
                git_commit: git_commit.clone(),
                operation: result.name.clone(),
                duration_ns: result.duration.as_nanos() as u64,
                throughput: result.throughput,
                architecture: result.architecture.clone(),
                simd_width: result.simd_width,
                optimization_level: "release".to_string(), // Could be parameterized
            };

            // Write to file
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},",
                record.timestamp,
                record.git_commit.as_ref().unwrap_or(&"unknown".to_string()),
                record.operation,
                record.duration_ns,
                record.throughput.map(|t| t.to_string()).unwrap_or_default(),
                record.architecture,
                record.simd_width,
                record.optimization_level // Reserved for future use
            )?;

            self.historical_data.push(record);
        }

        Ok(())
    }

    /// Record new performance results (no-std version - in-memory only)
    #[cfg(feature = "no-std")]
    pub fn record_results(
        &mut self,
        results: &[BenchmarkResult],
        git_commit: Option<String>,
    ) -> Result<(), &'static str> {
        let timestamp = 0; // Mock timestamp for no-std

        for result in results {
            let record = PerformanceRecord {
                timestamp,
                git_commit: git_commit.clone(),
                operation: result.name.clone(),
                duration_ns: result.duration.as_nanos() as u64,
                throughput: result.throughput,
                architecture: result.architecture.clone(),
                simd_width: result.simd_width,
                optimization_level: "release".to_string(), // Default optimization level
            };
            self.historical_data.push(record);
        }

        Ok(())
    }

    /// Analyze performance trends
    #[cfg(not(feature = "no-std"))]
    pub fn analyze_trends(&self, operation: &str, days_back: u64) -> PerformanceTrend {
        #[cfg(not(feature = "no-std"))]
        let cutoff_timestamp = {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let window = days_back.saturating_mul(24 * 60 * 60);
            now.saturating_sub(window)
        };
        #[cfg(feature = "no-std")]
        let cutoff_timestamp = current_timestamp() - (days_back * 24 * 60 * 60);

        let relevant_data: Vec<&PerformanceRecord> = self
            .historical_data
            .iter()
            .filter(|record| record.operation == operation && record.timestamp >= cutoff_timestamp)
            .collect();

        if relevant_data.is_empty() {
            return PerformanceTrend::NoData;
        }

        // Calculate trend
        let durations: Vec<f64> = relevant_data.iter().map(|r| r.duration_ns as f64).collect();
        let trend_slope = self.calculate_trend_slope(&durations);

        let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_duration = durations.iter().fold(0.0f64, |a, &b| a.max(b));

        PerformanceTrend::Data {
            operation: operation.to_string(),
            data_points: relevant_data.len(),
            trend_slope,
            avg_duration_ns: avg_duration as u64,
            min_duration_ns: min_duration as u64,
            max_duration_ns: max_duration as u64,
            variance: self.calculate_variance(&durations, avg_duration),
        }
    }

    /// Analyze performance trends (no-std version - basic analysis only)
    #[cfg(feature = "no-std")]
    pub fn analyze_trends(&self, operation: &str, _days_back: u64) -> PerformanceTrend {
        let relevant_data: Vec<&PerformanceRecord> = self
            .historical_data
            .iter()
            .filter(|r| r.operation == operation)
            .collect();

        if relevant_data.is_empty() {
            return PerformanceTrend::NoData;
        }

        // Calculate trend
        let durations: Vec<f64> = relevant_data.iter().map(|r| r.duration_ns as f64).collect();
        let trend_slope = self.calculate_trend_slope(&durations);

        let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_duration = durations.iter().fold(0.0f64, |a, &b| a.max(b));

        PerformanceTrend::Data {
            operation: operation.to_string(),
            data_points: relevant_data.len(),
            trend_slope,
            avg_duration_ns: avg_duration as u64,
            min_duration_ns: min_duration as u64,
            max_duration_ns: max_duration as u64,
            variance: self.calculate_variance(&durations, avg_duration),
        }
    }

    /// Check for performance alerts
    pub fn check_alerts(&self, current_results: &[BenchmarkResult]) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();

        for result in current_results {
            if let Some(baseline) = self.get_baseline_performance(&result.name) {
                let current_ns = result.duration.as_nanos() as f64;
                let baseline_ns = baseline.duration_ns as f64;
                let change_percent = ((current_ns - baseline_ns) / baseline_ns) * 100.0;

                if change_percent > self.thresholds.critical_slowdown {
                    alerts.push(PerformanceAlert {
                        alert_type: AlertType::CriticalSlowdown,
                        operation: result.name.clone(),
                        current_performance: current_ns,
                        baseline_performance: baseline_ns,
                        change_percent,
                        severity: AlertSeverity::Critical,
                        recommendation:
                            "Critical performance regression detected. Investigate immediately."
                                .to_string(),
                    });
                } else if change_percent > self.thresholds.regression_threshold {
                    alerts.push(PerformanceAlert {
                        alert_type: AlertType::Regression,
                        operation: result.name.clone(),
                        current_performance: current_ns,
                        baseline_performance: baseline_ns,
                        change_percent,
                        severity: if change_percent > 15.0 {
                            AlertSeverity::Error
                        } else {
                            AlertSeverity::Warning
                        },
                        recommendation: format!(
                            "Performance regression of {:.1}%. Review recent changes.",
                            change_percent
                        ),
                    });
                } else if change_percent < -self.thresholds.improvement_threshold {
                    alerts.push(PerformanceAlert {
                        alert_type: AlertType::Improvement,
                        operation: result.name.clone(),
                        current_performance: current_ns,
                        baseline_performance: baseline_ns,
                        change_percent,
                        severity: AlertSeverity::Info,
                        recommendation: format!(
                            "Performance improvement of {:.1}%. Great work!",
                            -change_percent
                        ),
                    });
                }
            }
        }

        alerts
    }

    /// Generate performance report
    pub fn generate_performance_report(&self, days_back: u64) -> PerformanceReport {
        let operations: HashSet<String> = self
            .historical_data
            .iter()
            .map(|r| r.operation.clone())
            .collect();

        let mut trends = HashMap::new();
        for operation in operations {
            trends.insert(
                operation.clone(),
                self.analyze_trends(&operation, days_back),
            );
        }

        #[cfg(not(feature = "no-std"))]
        let cutoff_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        #[cfg(feature = "no-std")]
        let cutoff_timestamp = current_timestamp() - (days_back * 24 * 60 * 60);

        let recent_records: Vec<&PerformanceRecord> = self
            .historical_data
            .iter()
            .filter(|record| record.timestamp >= cutoff_timestamp)
            .collect();

        PerformanceReport {
            period_days: days_back,
            total_benchmarks: recent_records.len(),
            unique_operations: trends.len(),
            trends,
            summary: self.generate_summary(&recent_records),
        }
    }

    /// Load historical data from file
    #[cfg(not(feature = "no-std"))]
    fn load_historical_data(&mut self) -> std::io::Result<()> {
        if !Path::new(&self.results_file).exists() {
            return Ok(());
        }

        let file = File::open(&self.results_file)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if let Some(record) = self.parse_record_line(&line) {
                self.historical_data.push(record);
            }
        }

        // Sort by timestamp
        self.historical_data.sort_by_key(|r| r.timestamp);
        Ok(())
    }

    /// Load historical data from file (no-std version - no-op)
    #[cfg(feature = "no-std")]
    fn load_historical_data(&mut self) -> Result<(), &'static str> {
        // No file I/O in no-std mode
        Ok(())
    }

    fn parse_record_line(&self, line: &str) -> Option<PerformanceRecord> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 8 {
            return None;
        }

        Some(PerformanceRecord {
            timestamp: parts[0].parse().ok()?,
            git_commit: if parts[1] == "unknown" {
                None
            } else {
                Some(parts[1].to_string())
            },
            operation: parts[2].to_string(),
            duration_ns: parts[3].parse().ok()?,
            throughput: if parts[4].is_empty() {
                None
            } else {
                parts[4].parse().ok()
            },
            architecture: parts[5].to_string(),
            simd_width: parts[6].parse().ok()?,
            optimization_level: parts[7].to_string(),
        })
    }

    fn get_baseline_performance(&self, operation: &str) -> Option<&PerformanceRecord> {
        // Use the median of the last 10 results as baseline
        let mut relevant: Vec<&PerformanceRecord> = self
            .historical_data
            .iter()
            .filter(|r| r.operation == operation)
            .collect();

        if relevant.len() < 5 {
            return None;
        }

        relevant.sort_by_key(|r| r.timestamp);
        let recent = &relevant[relevant.len().saturating_sub(10)..];

        if recent.is_empty() {
            return None;
        }

        // Return median performance
        let mut durations: Vec<&PerformanceRecord> = recent.to_vec();
        durations.sort_by_key(|r| r.duration_ns);
        Some(durations[durations.len() / 2])
    }

    fn calculate_trend_slope(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = values
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = (0..values.len()).map(|i| (i as f64 - x_mean).powi(2)).sum();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn calculate_variance(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance
    }

    fn generate_summary(&self, records: &[&PerformanceRecord]) -> String {
        if records.is_empty() {
            return "No data available for the specified period.".to_string();
        }

        let total_duration: u64 = records.iter().map(|r| r.duration_ns).sum();
        let avg_duration = total_duration / records.len() as u64;

        let architectures: HashSet<&String> = records.iter().map(|r| &r.architecture).collect();

        format!(
            "Period summary: {} benchmarks across {} architectures. Average duration: {}ns",
            records.len(),
            architectures.len(),
            avg_duration
        )
    }
}

/// Performance trend analysis result
#[derive(Debug)]
pub enum PerformanceTrend {
    NoData,
    Data {
        operation: String,
        data_points: usize,
        trend_slope: f64,
        avg_duration_ns: u64,
        min_duration_ns: u64,
        max_duration_ns: u64,
        variance: f64,
    },
}

/// Comprehensive performance report
#[derive(Debug)]
pub struct PerformanceReport {
    pub period_days: u64,
    pub total_benchmarks: usize,
    pub unique_operations: usize,
    pub trends: HashMap<String, PerformanceTrend>,
    pub summary: String,
}

impl PerformanceReport {
    /// Format the report as a string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Performance Report ===\n");
        report.push_str(&format!("Period: {} days\n", self.period_days));
        report.push_str(&format!("Total benchmarks: {}\n", self.total_benchmarks));
        report.push_str(&format!("Unique operations: {}\n", self.unique_operations));
        report.push_str(&format!("Summary: {}\n\n", self.summary));

        report.push_str("Trends by operation:\n");
        for (operation, trend) in &self.trends {
            match trend {
                PerformanceTrend::NoData => {
                    report.push_str(&format!("  {}: No data\n", operation));
                }
                PerformanceTrend::Data {
                    data_points,
                    trend_slope,
                    avg_duration_ns,
                    ..
                } => {
                    let trend_direction = if *trend_slope > 1000.0 {
                        "SLOWER"
                    } else if *trend_slope < -1000.0 {
                        "FASTER"
                    } else {
                        "STABLE"
                    };

                    report.push_str(&format!(
                        "  {}: {} ({} data points, avg: {}ns, trend: {})\n",
                        operation, trend_direction, data_points, avg_duration_ns, trend_direction
                    ));
                }
            }
        }

        report.push_str("\n=== End Report ===\n");
        report
    }
}

/// Continuous integration integration
pub struct CIIntegration;

impl CIIntegration {
    /// Run performance tests suitable for CI
    #[cfg(not(feature = "no-std"))]
    pub fn run_ci_benchmarks() -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        let mut suite = BenchmarkSuite::new();
        let mut results = Vec::new();

        let data: Vec<f32> = (0..2048).map(|i| i as f32 + 1.0).collect();
        let mut scratch = vec![0.0f32; data.len()];

        results.push(suite.benchmark("ci_dot_product", 200, || {
            let _ = crate::vector::basic_operations::dot_product(&data, &data);
        }));

        results.push(suite.benchmark("ci_norm_l2", 200, || {
            let _ = crate::vector::statistics_ops::norm_l2(&data);
        }));

        results.push(suite.benchmark("ci_reciprocal", 100, || {
            crate::vector::math_functions::reciprocal_vec(&data, &mut scratch);
        }));

        Ok(results)
    }

    #[cfg(feature = "no-std")]
    pub fn run_ci_benchmarks() -> Result<Vec<BenchmarkResult>, crate::traits::SimdError> {
        let mut suite = BenchmarkSuite::new();
        let mut results = Vec::new();

        // Quick benchmarks suitable for CI
        let test_data = (0..1000).map(|i| i as f32).collect::<Vec<f32>>();

        results.push(suite.benchmark("ci_dot_product", 100, || {
            let _result = crate::vector::dot_product(&test_data, &test_data);
        }));

        results.push(suite.benchmark("ci_euclidean_distance", 100, || {
            let _result = crate::distance::euclidean_distance(&test_data, &test_data);
        }));

        Ok(results)
    }

    /// Check if performance is acceptable for CI
    pub fn check_ci_performance(monitor: &PerformanceMonitor, results: &[BenchmarkResult]) -> bool {
        let alerts = monitor.check_alerts(results);

        // Fail CI if there are critical alerts
        !alerts
            .iter()
            .any(|alert| alert.severity == AlertSeverity::Critical)
    }

    /// Generate CI performance summary
    pub fn generate_ci_summary(alerts: &[PerformanceAlert]) -> String {
        if alerts.is_empty() {
            "✅ No performance regressions detected".to_string()
        } else {
            let critical_count = alerts
                .iter()
                .filter(|a| a.severity == AlertSeverity::Critical)
                .count();
            let error_count = alerts
                .iter()
                .filter(|a| a.severity == AlertSeverity::Error)
                .count();
            let warning_count = alerts
                .iter()
                .filter(|a| a.severity == AlertSeverity::Warning)
                .count();

            format!(
                "⚠️ Performance alerts: {} critical, {} errors, {} warnings",
                critical_count, error_count, warning_count
            )
        }
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    #[cfg(not(feature = "no-std"))]
    use std::fs;
    #[cfg(not(feature = "no-std"))]
    use std::time::Duration;

    #[test]
    #[cfg(not(feature = "no-std"))]
    fn test_performance_monitor_creation() {
        let temp_file = "/tmp/test_perf_monitor.csv";
        let _ = fs::remove_file(temp_file); // Clean up if exists

        let monitor = PerformanceMonitor::new(temp_file);
        assert!(monitor.is_ok());

        let _ = fs::remove_file(temp_file); // Clean up
    }

    #[test]
    #[cfg(not(feature = "no-std"))]
    fn test_performance_record_parsing() {
        let temp_file = "/tmp/test_perf_parsing.csv";
        let _ = fs::remove_file(temp_file);

        let mut monitor = PerformanceMonitor::new(temp_file).unwrap();

        let test_results = vec![BenchmarkResult {
            name: "test_op".to_string(),
            duration: Duration::from_millis(10),
            throughput: Some(1000.0),
            simd_width: 8,
            architecture: "AVX2".to_string(),
            iterations: 1000,
        }];

        let result = monitor.record_results(&test_results, Some("abc123".to_string()));
        assert!(result.is_ok());

        let _ = fs::remove_file(temp_file);
    }

    #[test]
    #[cfg(not(feature = "no-std"))]
    fn test_trend_analysis() {
        let temp_file = "/tmp/test_trend_analysis.csv";
        let _ = fs::remove_file(temp_file);

        let monitor = PerformanceMonitor::new(temp_file).unwrap();
        let trend = monitor.analyze_trends("nonexistent_op", 7);

        match trend {
            PerformanceTrend::NoData => {
                // Expected for empty data
            }
            _ => panic!("Expected NoData for empty dataset"),
        }

        let _ = fs::remove_file(temp_file);
    }

    #[test]
    #[cfg(not(feature = "no-std"))]
    fn test_ci_integration() {
        let results = CIIntegration::run_ci_benchmarks();
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(!results.is_empty());

        for result in &results {
            assert!(result.duration > Duration::from_nanos(0));
        }
    }

    #[test]
    fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds::default();
        assert_eq!(thresholds.regression_threshold, 5.0);
        assert_eq!(thresholds.improvement_threshold, 10.0);
        assert_eq!(thresholds.critical_slowdown, 25.0);
    }
}
