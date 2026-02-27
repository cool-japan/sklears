//! Performance regression testing utilities
//!
//! This module provides utilities for tracking performance metrics over time
//! and detecting performance regressions automatically.

use crate::{UtilsError, UtilsResult};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Performance regression testing framework
pub struct PerformanceRegressionTester {
    baseline_file: PathBuf,
    current_results: HashMap<String, Vec<f64>>,
    thresholds: RegressionThresholds,
}

/// Thresholds for regression detection
#[derive(Clone, Debug)]
pub struct RegressionThresholds {
    /// Maximum allowed percentage increase in execution time
    pub time_increase_threshold: f64,
    /// Maximum allowed percentage increase in memory usage
    pub memory_increase_threshold: f64,
    /// Minimum number of samples required for statistical significance
    pub min_samples: usize,
    /// Confidence level for regression detection (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            time_increase_threshold: 10.0,   // 10% increase
            memory_increase_threshold: 15.0, // 15% increase
            min_samples: 10,
            confidence_level: 0.95,
        }
    }
}

/// Result of a regression test
#[derive(Debug, Clone)]
pub struct RegressionTestResult {
    pub test_name: String,
    pub baseline_mean: f64,
    pub current_mean: f64,
    pub percentage_change: f64,
    pub is_regression: bool,
    pub confidence_interval: (f64, f64),
    pub p_value: Option<f64>,
}

impl PerformanceRegressionTester {
    /// Create a new performance regression tester
    pub fn new<P: AsRef<Path>>(baseline_file: P) -> Self {
        Self {
            baseline_file: baseline_file.as_ref().to_path_buf(),
            current_results: HashMap::new(),
            thresholds: RegressionThresholds::default(),
        }
    }

    /// Set custom regression thresholds
    pub fn with_thresholds(mut self, thresholds: RegressionThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Record a performance measurement
    pub fn record_measurement(&mut self, test_name: &str, duration: Duration) {
        let duration_ms = duration.as_secs_f64() * 1000.0;
        self.current_results
            .entry(test_name.to_string())
            .or_default()
            .push(duration_ms);
    }

    /// Benchmark a function and record its performance
    pub fn benchmark_function<F, R>(
        &mut self,
        test_name: &str,
        iterations: usize,
        mut func: F,
    ) -> UtilsResult<R>
    where
        F: FnMut() -> R,
    {
        let mut result = None;
        let mut measurements = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            let r = func();
            let duration = start.elapsed();

            measurements.push(duration.as_secs_f64() * 1000.0);
            if result.is_none() {
                result = Some(r);
            }
        }

        self.current_results
            .insert(test_name.to_string(), measurements);

        result.ok_or_else(|| UtilsError::InvalidParameter("No measurements recorded".to_string()))
    }

    /// Load baseline measurements from file
    pub fn load_baseline(&self) -> UtilsResult<HashMap<String, Vec<f64>>> {
        if !self.baseline_file.exists() {
            return Ok(HashMap::new());
        }

        let file = File::open(&self.baseline_file).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to open baseline file: {e}"))
        })?;

        let reader = BufReader::new(file);
        let mut baseline = HashMap::new();

        for line in reader.lines() {
            let line = line
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read line: {e}")))?;

            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let test_name = parts[0].trim().to_string();
                let measurements: Result<Vec<f64>, _> =
                    parts[1..].iter().map(|s| s.trim().parse::<f64>()).collect();

                if let Ok(measurements) = measurements {
                    baseline.insert(test_name, measurements);
                }
            }
        }

        Ok(baseline)
    }

    /// Save current measurements as new baseline
    pub fn save_baseline(&self) -> UtilsResult<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.baseline_file)
            .map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to create baseline file: {e}"))
            })?;

        let mut writer = BufWriter::new(file);

        writeln!(writer, "# Performance baseline measurements")
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write header: {e}")))?;

        for (test_name, measurements) in &self.current_results {
            write!(writer, "{test_name}").map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to write test name: {e}"))
            })?;

            for measurement in measurements {
                write!(writer, ",{measurement}").map_err(|e| {
                    UtilsError::InvalidParameter(format!("Failed to write measurement: {e}"))
                })?;
            }

            writeln!(writer).map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to write newline: {e}"))
            })?;
        }

        writer
            .flush()
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to flush writer: {e}")))?;

        Ok(())
    }

    /// Run regression tests against baseline
    pub fn run_regression_tests(&self) -> UtilsResult<Vec<RegressionTestResult>> {
        let baseline = self.load_baseline()?;
        let mut results = Vec::new();

        for (test_name, current_measurements) in &self.current_results {
            if let Some(baseline_measurements) = baseline.get(test_name) {
                let result = self.analyze_regression(
                    test_name,
                    baseline_measurements,
                    current_measurements,
                )?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Analyze regression for a specific test
    fn analyze_regression(
        &self,
        test_name: &str,
        baseline: &[f64],
        current: &[f64],
    ) -> UtilsResult<RegressionTestResult> {
        if baseline.len() < self.thresholds.min_samples
            || current.len() < self.thresholds.min_samples
        {
            return Err(UtilsError::InsufficientData {
                min: self.thresholds.min_samples,
                actual: baseline.len().min(current.len()),
            });
        }

        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let current_mean = current.iter().sum::<f64>() / current.len() as f64;

        let percentage_change = ((current_mean - baseline_mean) / baseline_mean) * 100.0;

        // Calculate confidence interval for current measurements
        let current_std = self.calculate_std_dev(current, current_mean);
        let current_sem = current_std / (current.len() as f64).sqrt();
        let t_value = self.get_t_value(current.len() - 1, self.thresholds.confidence_level);

        let margin_of_error = t_value * current_sem;
        let confidence_interval = (
            current_mean - margin_of_error,
            current_mean + margin_of_error,
        );

        // Perform t-test for statistical significance
        let p_value = self.welch_t_test(baseline, current);

        // Determine if this is a regression
        let is_regression = percentage_change > self.thresholds.time_increase_threshold;

        Ok(RegressionTestResult {
            test_name: test_name.to_string(),
            baseline_mean,
            current_mean,
            percentage_change,
            is_regression,
            confidence_interval,
            p_value,
        })
    }

    /// Calculate standard deviation
    fn calculate_std_dev(&self, data: &[f64], mean: f64) -> f64 {
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Get t-value for given degrees of freedom and confidence level
    fn get_t_value(&self, df: usize, confidence_level: f64) -> f64 {
        // Simplified t-value lookup for common confidence levels
        // In a real implementation, you'd use a proper t-distribution
        match (df, (confidence_level * 1000.0) as usize) {
            (_, 950) => 1.96, // 95% confidence, approximation
            (_, 990) => 2.58, // 99% confidence, approximation
            (_, 995) => 2.81, // 99.5% confidence, approximation
            _ => 2.0,         // Default conservative estimate
        }
    }

    /// Perform Welch's t-test
    fn welch_t_test(&self, sample1: &[f64], sample2: &[f64]) -> Option<f64> {
        if sample1.len() < 2 || sample2.len() < 2 {
            return None;
        }

        let mean1 = sample1.iter().sum::<f64>() / sample1.len() as f64;
        let mean2 = sample2.iter().sum::<f64>() / sample2.len() as f64;

        let var1 =
            sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (sample1.len() - 1) as f64;

        let var2 =
            sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (sample2.len() - 1) as f64;

        let se1 = var1 / sample1.len() as f64;
        let se2 = var2 / sample2.len() as f64;

        let se_diff = (se1 + se2).sqrt();

        if se_diff == 0.0 {
            return None;
        }

        let t_stat = (mean1 - mean2) / se_diff;

        // Simplified p-value calculation (in practice, use proper statistical library)
        Some((2.0 * (1.0 - (t_stat.abs() / 3.0).min(1.0))).max(0.0))
    }

    /// Generate a performance report
    pub fn generate_report(&self, results: &[RegressionTestResult]) -> String {
        let mut report = String::new();

        report.push_str("# Performance Regression Test Report\n\n");

        let regression_count = results.iter().filter(|r| r.is_regression).count();
        let total_tests = results.len();
        report.push_str(&format!("Total tests: {total_tests}\n"));
        report.push_str(&format!("Regressions detected: {regression_count}\n\n"));

        if regression_count > 0 {
            report.push_str("## ⚠️ Performance Regressions\n\n");
            for result in results.iter().filter(|r| r.is_regression) {
                report.push_str(&format!(
                    "**{}**: {:.2}% slower ({:.2}ms → {:.2}ms)\n",
                    result.test_name,
                    result.percentage_change,
                    result.baseline_mean,
                    result.current_mean
                ));
            }
            report.push('\n');
        }

        report.push_str("## 📊 All Test Results\n\n");
        report.push_str("| Test Name | Baseline (ms) | Current (ms) | Change (%) | Status |\n");
        report.push_str("|-----------|---------------|--------------|------------|---------|\n");

        for result in results {
            let status = if result.is_regression {
                "🔴 REGRESSION"
            } else {
                "✅ OK"
            };
            report.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:+.2} | {} |\n",
                result.test_name,
                result.baseline_mean,
                result.current_mean,
                result.percentage_change,
                status
            ));
        }

        report
    }

    /// Clear current measurements
    pub fn clear_measurements(&mut self) {
        self.current_results.clear();
    }
}

/// Macro to easily benchmark functions
#[macro_export]
macro_rules! benchmark_regression {
    ($tester:expr, $name:expr, $iterations:expr, $func:expr) => {
        $tester.benchmark_function($name, $iterations, || $func)?
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_regression_tester_basic() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut tester = PerformanceRegressionTester::new(temp_file.path());

        // Record some measurements
        tester.record_measurement("fast_function", Duration::from_millis(10));
        tester.record_measurement("fast_function", Duration::from_millis(12));
        tester.record_measurement("slow_function", Duration::from_millis(100));

        // Save as baseline
        tester.save_baseline().unwrap();

        // Clear and add new measurements (simulating regression)
        tester.clear_measurements();
        tester.record_measurement("fast_function", Duration::from_millis(15)); // slower
        tester.record_measurement("fast_function", Duration::from_millis(16));

        // Load baseline and compare
        let baseline = tester.load_baseline().unwrap();
        assert!(baseline.contains_key("fast_function"));
        assert!(baseline.contains_key("slow_function"));
    }

    #[test]
    fn test_benchmark_function() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut tester = PerformanceRegressionTester::new(temp_file.path());

        let result = tester
            .benchmark_function("test_computation", 5, || {
                // Simulate some work
                (0..1000).sum::<i32>()
            })
            .unwrap();

        assert_eq!(result, 499500); // Expected sum
        assert!(tester.current_results.contains_key("test_computation"));
        assert_eq!(tester.current_results["test_computation"].len(), 5);
    }

    #[test]
    fn test_regression_detection() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut tester = PerformanceRegressionTester::new(temp_file.path()).with_thresholds(
            RegressionThresholds {
                time_increase_threshold: 5.0, // 5% threshold
                memory_increase_threshold: 10.0,
                min_samples: 3,
                confidence_level: 0.95,
            },
        );

        // Create baseline with consistent measurements
        for _ in 0..10 {
            tester.record_measurement("stable_function", Duration::from_millis(100));
        }
        tester.save_baseline().unwrap();

        // Clear and add regressed measurements
        tester.clear_measurements();
        for _ in 0..10 {
            tester.record_measurement("stable_function", Duration::from_millis(120));
            // 20% slower
        }

        let results = tester.run_regression_tests().unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_regression);
        assert!(results[0].percentage_change > 5.0);
    }

    #[test]
    fn test_report_generation() {
        let temp_file = NamedTempFile::new().unwrap();
        let tester = PerformanceRegressionTester::new(temp_file.path());

        let results = vec![
            RegressionTestResult {
                test_name: "fast_function".to_string(),
                baseline_mean: 10.0,
                current_mean: 12.0,
                percentage_change: 20.0,
                is_regression: true,
                confidence_interval: (11.0, 13.0),
                p_value: Some(0.05),
            },
            RegressionTestResult {
                test_name: "stable_function".to_string(),
                baseline_mean: 50.0,
                current_mean: 48.0,
                percentage_change: -4.0,
                is_regression: false,
                confidence_interval: (47.0, 49.0),
                p_value: Some(0.3),
            },
        ];

        let report = tester.generate_report(&results);
        assert!(report.contains("Performance Regression Test Report"));
        assert!(report.contains("Regressions detected: 1"));
        assert!(report.contains("fast_function"));
        assert!(report.contains("🔴 REGRESSION"));
        assert!(report.contains("✅ OK"));
    }
}
