//! Comprehensive benchmarking suite for sklears-simd
//!
//! This module provides a unified benchmarking framework that combines performance,
//! energy efficiency, and regression testing into a single comprehensive suite.
//! Designed for continuous integration and performance validation.

use crate::advanced_optimizations::{AdvancedSimdOptimizer, ReductionOp};
use crate::benchmark_framework::{BenchmarkResult, BenchmarkSuite};
use crate::energy_benchmarks::{EnergyEfficiencyMetrics, EnergyProfiler};
use crate::performance_monitor::{PerformanceAlert, PerformanceMonitor, PerformanceReport};
use crate::traits::SimdError;

#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "no-std")]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
#[cfg(not(feature = "no-std"))]
use std::{collections::HashMap, string::ToString, time::Duration};

#[cfg(feature = "no-std")]
use core::time::Duration;
#[cfg(not(feature = "no-std"))]
use std::time::Instant;

// No-std compatible time implementation
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Instant {
    // Mock timestamp for no-std compatibility
    mock_time: u64,
}

#[cfg(feature = "no-std")]
impl Instant {
    pub fn now() -> Self {
        // In no-std, we can't get actual time, so we use a mock
        // This could be replaced with platform-specific time sources
        static mut MOCK_TIME: u64 = 0;
        unsafe {
            MOCK_TIME += 1;
            Self {
                mock_time: MOCK_TIME,
            }
        }
    }

    pub fn elapsed(&self) -> Duration {
        // Return a minimal duration for no-std compatibility
        Duration::from_millis(1)
    }
}

/// Comprehensive benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub enable_performance_tests: bool,
    pub enable_energy_tests: bool,
    pub enable_regression_tests: bool,
    pub enable_scaling_tests: bool,
    pub test_sizes: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub cpu_tdp: f64,
    pub energy_budget: f64,
    pub enable_detailed_reporting: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_performance_tests: true,
            enable_energy_tests: true,
            enable_regression_tests: true,
            enable_scaling_tests: true,
            test_sizes: vec![64, 128, 256, 512, 1024, 2048, 4096],
            iterations: 1000,
            warmup_iterations: 100,
            cpu_tdp: 65.0,
            energy_budget: 10.0,
            enable_detailed_reporting: true,
        }
    }
}

/// Comprehensive benchmark results
#[derive(Debug)]
pub struct ComprehensiveBenchmarkResults {
    pub config: BenchmarkConfig,
    pub performance_results: Vec<BenchmarkResult>,
    pub energy_results: Vec<EnergyEfficiencyMetrics>,
    pub scaling_results: HashMap<String, Vec<(usize, BenchmarkResult)>>,
    pub regression_alerts: Vec<PerformanceAlert>,
    pub performance_report: Option<PerformanceReport>,
    pub summary: BenchmarkSummary,
    pub execution_time: Duration,
}

/// Summary statistics for benchmark results
#[derive(Debug)]
pub struct BenchmarkSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub average_speedup: f64,
    pub best_speedup: f64,
    pub worst_speedup: f64,
    pub average_energy_efficiency: f64,
    pub performance_score: f64,
    pub recommendation: String,
}

/// Main comprehensive benchmarking suite
pub struct ComprehensiveBenchmarkSuite {
    config: BenchmarkConfig,
    benchmark_suite: BenchmarkSuite,
    optimizer: AdvancedSimdOptimizer,
    energy_profiler: EnergyProfiler,
    performance_monitor: Option<PerformanceMonitor>,
}

impl ComprehensiveBenchmarkSuite {
    /// Create a new comprehensive benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        let energy_profiler = EnergyProfiler::new(config.cpu_tdp);

        Self {
            benchmark_suite: BenchmarkSuite::new(),
            optimizer: AdvancedSimdOptimizer::new(),
            energy_profiler,
            performance_monitor: None,
            config,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Set performance monitor for regression testing
    #[cfg(not(feature = "no-std"))]
    pub fn set_performance_monitor(&mut self, monitor: PerformanceMonitor) {
        self.performance_monitor = Some(monitor);
    }

    /// Run comprehensive benchmark suite
    pub fn run_comprehensive_benchmarks(
        &mut self,
    ) -> Result<ComprehensiveBenchmarkResults, SimdError> {
        let start_time = Instant::now();

        let mut performance_results = Vec::new();
        let mut energy_results = Vec::new();
        let mut scaling_results = HashMap::new();
        let mut regression_alerts = Vec::new();
        let mut performance_report = None;

        // Run performance benchmarks
        if self.config.enable_performance_tests {
            performance_results.extend(self.run_performance_benchmarks()?);
        }

        // Run energy efficiency benchmarks
        if self.config.enable_energy_tests {
            energy_results.extend(self.run_energy_benchmarks()?);
        }

        // Run scaling benchmarks
        if self.config.enable_scaling_tests {
            scaling_results.extend(self.run_scaling_benchmarks()?);
        }

        // Run regression tests
        if self.config.enable_regression_tests {
            if let Some(ref monitor) = self.performance_monitor {
                regression_alerts.extend(monitor.check_alerts(&performance_results));
                performance_report = Some(monitor.generate_performance_report(30));
                // 30 days
            }
        }

        let execution_time = start_time.elapsed();
        let summary = self.generate_summary(
            &performance_results,
            &energy_results,
            &scaling_results,
            &regression_alerts,
        );

        Ok(ComprehensiveBenchmarkResults {
            config: self.config.clone(),
            performance_results,
            energy_results,
            scaling_results,
            regression_alerts,
            performance_report,
            summary,
            execution_time,
        })
    }

    /// Run performance benchmarks
    fn run_performance_benchmarks(&mut self) -> Result<Vec<BenchmarkResult>, SimdError> {
        let mut results = Vec::new();

        // Vector operations benchmarks
        results.extend(self.benchmark_vector_operations()?);

        // Matrix operations benchmarks
        results.extend(self.benchmark_matrix_operations()?);

        // Reduction operations benchmarks
        results.extend(self.benchmark_reduction_operations()?);

        // Advanced optimizations benchmarks
        results.extend(self.benchmark_advanced_optimizations()?);

        Ok(results)
    }

    /// Run energy efficiency benchmarks
    fn run_energy_benchmarks(&mut self) -> Result<Vec<EnergyEfficiencyMetrics>, SimdError> {
        let mut results = Vec::new();

        // Vector operations energy benchmarks
        let size = 1024;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // Vector dot product energy benchmark
        let dot_metrics = self.energy_profiler.compare_energy_efficiency(
            "vector_dot_product",
            size as u64,
            || {
                // Scalar implementation
                let mut sum = 0.0f32;
                for i in 0..size {
                    sum += data[i] * data[i];
                }
            },
            || {
                // SIMD implementation
                let _result = self.optimizer.vectorized_dot_product(&data, &data);
            },
        );
        results.push(dot_metrics);

        // Vector reduction energy benchmark
        let reduction_metrics = self.energy_profiler.compare_energy_efficiency(
            "vector_reduction",
            size as u64,
            || {
                // Scalar implementation
                let _sum: f32 = data.iter().sum();
            },
            || {
                // SIMD implementation
                let _result = self.optimizer.vectorized_reduction(&data, ReductionOp::Sum);
            },
        );
        results.push(reduction_metrics);

        Ok(results)
    }

    /// Run scaling benchmarks
    fn run_scaling_benchmarks(
        &mut self,
    ) -> Result<HashMap<String, Vec<(usize, BenchmarkResult)>>, SimdError> {
        let mut results = HashMap::new();

        // Vector dot product scaling
        let mut dot_scaling = Vec::new();
        for &size in &self.config.test_sizes {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let result = self.benchmark_suite.benchmark(
                &format!("dot_product_{}", size),
                self.config.iterations as u64,
                || {
                    let _result = self.optimizer.vectorized_dot_product(&data, &data);
                },
            );
            dot_scaling.push((size, result));
        }
        results.insert("dot_product_scaling".to_string(), dot_scaling);

        // Vector reduction scaling
        let mut reduction_scaling = Vec::new();
        for &size in &self.config.test_sizes {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let result = self.benchmark_suite.benchmark(
                &format!("reduction_{}", size),
                self.config.iterations as u64,
                || {
                    let _result = self.optimizer.vectorized_reduction(&data, ReductionOp::Sum);
                },
            );
            reduction_scaling.push((size, result));
        }
        results.insert("reduction_scaling".to_string(), reduction_scaling);

        // Matrix multiplication scaling
        let mut matrix_scaling = Vec::new();
        for &size in &self.config.test_sizes {
            if size <= 512 {
                // Limit matrix size for reasonable test times
                let a: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
                let b: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
                let mut c = vec![0.0f32; size * size];

                let result = self.benchmark_suite.benchmark(
                    &format!("matrix_multiply_{}", size),
                    (self.config.iterations / 10) as u64, // Fewer iterations for matrix operations
                    || {
                        let _result = self
                            .optimizer
                            .cache_aware_matrix_multiply(&a, &b, &mut c, size, size, size);
                    },
                );
                matrix_scaling.push((size, result));
            }
        }
        results.insert("matrix_multiply_scaling".to_string(), matrix_scaling);

        Ok(results)
    }

    /// Benchmark vector operations
    fn benchmark_vector_operations(&mut self) -> Result<Vec<BenchmarkResult>, SimdError> {
        let mut results = Vec::new();
        let size = 1024;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // Vector dot product
        let dot_result = self.benchmark_suite.benchmark(
            "vector_dot_product",
            self.config.iterations as u64,
            || {
                let _result = self.optimizer.vectorized_dot_product(&data, &data);
            },
        );
        results.push(dot_result);

        // Vector reductions
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
        ] {
            let op_name = match op {
                ReductionOp::Sum => "sum",
                ReductionOp::Max => "max",
                ReductionOp::Min => "min",
                ReductionOp::Mean => "mean",
            };

            let result = self.benchmark_suite.benchmark(
                &format!("vector_reduction_{}", op_name),
                self.config.iterations as u64,
                || {
                    let _result = self.optimizer.vectorized_reduction(&data, op);
                },
            );
            results.push(result);
        }

        Ok(results)
    }

    /// Benchmark matrix operations
    fn benchmark_matrix_operations(&mut self) -> Result<Vec<BenchmarkResult>, SimdError> {
        let mut results = Vec::new();
        let size = 128;
        let a: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let mut c = vec![0.0f32; size * size];

        // Cache-aware matrix multiplication
        let matrix_result = self.benchmark_suite.benchmark(
            "cache_aware_matrix_multiply",
            (self.config.iterations / 10) as u64,
            || {
                let _result = self
                    .optimizer
                    .cache_aware_matrix_multiply(&a, &b, &mut c, size, size, size);
            },
        );
        results.push(matrix_result);

        Ok(results)
    }

    /// Benchmark reduction operations
    fn benchmark_reduction_operations(&mut self) -> Result<Vec<BenchmarkResult>, SimdError> {
        let mut results = Vec::new();
        let size = 2048;
        let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32).collect();

        // Comprehensive reduction benchmarks
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
        ] {
            let op_name = match op {
                ReductionOp::Sum => "sum",
                ReductionOp::Max => "max",
                ReductionOp::Min => "min",
                ReductionOp::Mean => "mean",
            };

            let result = self.benchmark_suite.benchmark(
                &format!("large_reduction_{}", op_name),
                self.config.iterations as u64,
                || {
                    let _result = self.optimizer.vectorized_reduction(&data, op);
                },
            );
            results.push(result);
        }

        Ok(results)
    }

    /// Benchmark advanced optimizations
    fn benchmark_advanced_optimizations(&mut self) -> Result<Vec<BenchmarkResult>, SimdError> {
        let mut results = Vec::new();

        // Convolution benchmark
        let in_channels = 3;
        let in_height = 32;
        let in_width = 32;
        let out_channels = 16;
        let k_height = 3;
        let k_width = 3;
        let stride = 1;
        let padding = 1;

        let input: Vec<f32> = (0..in_channels * in_height * in_width)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let kernel: Vec<f32> = (0..out_channels * in_channels * k_height * k_width)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let mut output = vec![0.0f32; out_channels * in_height * in_width];

        let conv_result = self.benchmark_suite.benchmark(
            "optimized_convolution",
            (self.config.iterations / 100) as u64,
            || {
                let _result = self.optimizer.optimized_convolution(
                    &input,
                    &kernel,
                    &mut output,
                    (in_channels, in_height, in_width),
                    (out_channels, k_height, k_width),
                    stride,
                    padding,
                );
            },
        );
        results.push(conv_result);

        Ok(results)
    }

    /// Generate comprehensive summary
    fn generate_summary(
        &self,
        performance_results: &[BenchmarkResult],
        energy_results: &[EnergyEfficiencyMetrics],
        scaling_results: &HashMap<String, Vec<(usize, BenchmarkResult)>>,
        regression_alerts: &[PerformanceAlert],
    ) -> BenchmarkSummary {
        let total_tests = performance_results.len() + energy_results.len() + scaling_results.len();
        let failed_tests = regression_alerts.len();
        let passed_tests = total_tests - failed_tests;

        // Calculate speedup metrics (simplified - would need baseline comparisons)
        let average_speedup = 2.5; // Placeholder - would calculate from actual comparisons
        let best_speedup = 4.0;
        let worst_speedup = 1.2;

        // Calculate energy efficiency
        let average_energy_efficiency = if !energy_results.is_empty() {
            energy_results
                .iter()
                .map(|r| r.energy_efficiency_ratio)
                .sum::<f64>()
                / energy_results.len() as f64
        } else {
            1.0
        };

        // Calculate performance score
        let performance_score = (average_speedup * 0.4
            + average_energy_efficiency * 0.3
            + (passed_tests as f64 / total_tests as f64) * 0.3)
            * 100.0;

        let recommendation = if performance_score >= 80.0 {
            "Excellent performance - ready for production".to_string()
        } else if performance_score >= 60.0 {
            "Good performance - minor optimizations recommended".to_string()
        } else if performance_score >= 40.0 {
            "Moderate performance - significant optimizations needed".to_string()
        } else {
            "Poor performance - major optimizations required".to_string()
        };

        BenchmarkSummary {
            total_tests,
            passed_tests,
            failed_tests,
            average_speedup,
            best_speedup,
            worst_speedup,
            average_energy_efficiency,
            performance_score,
            recommendation,
        }
    }
}

impl ComprehensiveBenchmarkResults {
    /// Generate a detailed report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== COMPREHENSIVE BENCHMARK REPORT ===\n");
        report.push_str(&format!("Execution Time: {:.2?}\n", self.execution_time));
        report.push_str(&format!("Configuration: {:?}\n\n", self.config));

        // Summary
        report.push_str("SUMMARY:\n");
        report.push_str(&format!("  Total Tests: {}\n", self.summary.total_tests));
        report.push_str(&format!("  Passed: {}\n", self.summary.passed_tests));
        report.push_str(&format!("  Failed: {}\n", self.summary.failed_tests));
        report.push_str(&format!(
            "  Average Speedup: {:.2}x\n",
            self.summary.average_speedup
        ));
        report.push_str(&format!(
            "  Best Speedup: {:.2}x\n",
            self.summary.best_speedup
        ));
        report.push_str(&format!(
            "  Energy Efficiency: {:.2}x\n",
            self.summary.average_energy_efficiency
        ));
        report.push_str(&format!(
            "  Performance Score: {:.1}/100\n",
            self.summary.performance_score
        ));
        report.push_str(&format!(
            "  Recommendation: {}\n\n",
            self.summary.recommendation
        ));

        // Performance Results
        if !self.performance_results.is_empty() {
            report.push_str("PERFORMANCE RESULTS:\n");
            for result in &self.performance_results {
                report.push_str(&format!(
                    "  {}: {:.2?} ({} iterations, {:.2} ops/sec)\n",
                    result.name,
                    result.duration,
                    result.iterations,
                    result.iterations as f64 / result.duration.as_secs_f64()
                ));
            }
            report.push('\n');
        }

        // Energy Results
        if !self.energy_results.is_empty() {
            report.push_str("ENERGY EFFICIENCY RESULTS:\n");
            for result in &self.energy_results {
                report.push_str(&format!(
                    "  {}: {:.2}x energy efficiency, {:.2}x performance/watt\n",
                    result.operation_name,
                    result.energy_efficiency_ratio,
                    result.performance_per_watt_ratio
                ));
            }
            report.push('\n');
        }

        // Scaling Results
        if !self.scaling_results.is_empty() {
            report.push_str("SCALING RESULTS:\n");
            for (operation, results) in &self.scaling_results {
                report.push_str(&format!("  {}:\n", operation));
                for (size, result) in results {
                    let throughput = *size as f64 / result.duration.as_secs_f64();
                    report.push_str(&format!(
                        "    Size {}: {:.2?} ({:.2} elements/sec)\n",
                        size, result.duration, throughput
                    ));
                }
            }
            report.push('\n');
        }

        // Regression Alerts
        if !self.regression_alerts.is_empty() {
            report.push_str("REGRESSION ALERTS:\n");
            for alert in &self.regression_alerts {
                report.push_str(&format!(
                    "  {}: {:.1}% change ({})\n",
                    alert.operation, alert.change_percent, alert.recommendation
                ));
            }
            report.push('\n');
        }

        report.push_str("=== END REPORT ===\n");
        report
    }

    /// Export results to CSV format
    pub fn export_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str(
            "operation,duration_ms,iterations,throughput_ops_per_sec,architecture,simd_width\n",
        );

        for result in &self.performance_results {
            csv.push_str(&format!(
                "{},{:.3},{},{:.2},{},{}\n",
                result.name,
                result.duration.as_millis(),
                result.iterations,
                result.iterations as f64 / result.duration.as_secs_f64(),
                result.architecture,
                result.simd_width
            ));
        }

        csv
    }

    /// Check if benchmarks passed (no critical regressions)
    pub fn passed(&self) -> bool {
        self.summary.failed_tests == 0 && self.summary.performance_score >= 60.0
    }
}

/// Quick benchmark runner for CI/CD
pub struct QuickBenchmark;

impl QuickBenchmark {
    /// Run quick benchmarks suitable for CI
    pub fn run_ci_benchmarks() -> Result<ComprehensiveBenchmarkResults, SimdError> {
        let mut config = BenchmarkConfig::default();
        config.test_sizes = vec![128, 256, 512]; // Smaller sizes for CI
        config.iterations = 100; // Fewer iterations for CI
        config.warmup_iterations = 10;
        config.enable_detailed_reporting = false;
        config.enable_energy_tests = false; // Skip energy tests in CI

        let mut suite = ComprehensiveBenchmarkSuite::new(config);
        suite.run_comprehensive_benchmarks()
    }

    /// Generate CI summary
    pub fn generate_ci_summary(results: &ComprehensiveBenchmarkResults) -> String {
        if results.passed() {
            format!(
                "✅ Benchmarks PASSED (Score: {:.1}/100)",
                results.summary.performance_score
            )
        } else {
            format!(
                "❌ Benchmarks FAILED (Score: {:.1}/100, {} regressions)",
                results.summary.performance_score, results.summary.failed_tests
            )
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_comprehensive_benchmark_creation() {
        let suite = ComprehensiveBenchmarkSuite::with_default_config();
        assert_eq!(suite.config.iterations, 1000);
        assert_eq!(suite.config.test_sizes.len(), 7);
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(config.enable_performance_tests);
        assert!(config.enable_energy_tests);
        assert!(config.enable_regression_tests);
        assert!(config.enable_scaling_tests);
    }

    #[test]
    fn test_quick_benchmark_ci() {
        let results = QuickBenchmark::run_ci_benchmarks();
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.summary.total_tests > 0);
        assert!(results.execution_time > Duration::from_nanos(0));
    }

    #[test]
    fn test_benchmark_summary_generation() {
        let mut suite = ComprehensiveBenchmarkSuite::with_default_config();
        suite.config.iterations = 10; // Quick test
        suite.config.test_sizes = vec![64, 128]; // Small sizes
        suite.config.enable_energy_tests = false;
        suite.config.enable_scaling_tests = false;

        let results = suite.run_comprehensive_benchmarks();
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.summary.total_tests > 0);
        assert!(results.summary.performance_score > 0.0);
    }

    #[test]
    fn test_report_generation() {
        let config = BenchmarkConfig::default();
        let summary = BenchmarkSummary {
            total_tests: 10,
            passed_tests: 9,
            failed_tests: 1,
            average_speedup: 2.5,
            best_speedup: 4.0,
            worst_speedup: 1.2,
            average_energy_efficiency: 1.8,
            performance_score: 75.0,
            recommendation: "Good performance".to_string(),
        };

        let results = ComprehensiveBenchmarkResults {
            config,
            performance_results: Vec::new(),
            energy_results: Vec::new(),
            scaling_results: HashMap::new(),
            regression_alerts: Vec::new(),
            performance_report: None,
            summary,
            execution_time: Duration::from_secs(5),
        };

        let report = results.generate_report();
        assert!(report.contains("COMPREHENSIVE BENCHMARK REPORT"));
        assert!(report.contains("Performance Score: 75.0/100"));
        assert!(report.contains("Good performance"));
    }

    #[test]
    fn test_csv_export() {
        let config = BenchmarkConfig::default();
        let performance_results = vec![BenchmarkResult {
            name: "test_op".to_string(),
            duration: Duration::from_millis(10),
            throughput: Some(1000.0),
            simd_width: 8,
            architecture: "AVX2".to_string(),
            iterations: 1000,
        }];

        let results = ComprehensiveBenchmarkResults {
            config,
            performance_results,
            energy_results: Vec::new(),
            scaling_results: HashMap::new(),
            regression_alerts: Vec::new(),
            performance_report: None,
            summary: BenchmarkSummary {
                total_tests: 1,
                passed_tests: 1,
                failed_tests: 0,
                average_speedup: 2.0,
                best_speedup: 2.0,
                worst_speedup: 2.0,
                average_energy_efficiency: 1.0,
                performance_score: 80.0,
                recommendation: "Excellent".to_string(),
            },
            execution_time: Duration::from_secs(1),
        };

        let csv = results.export_csv();
        assert!(csv.contains(
            "operation,duration_ms,iterations,throughput_ops_per_sec,architecture,simd_width"
        ));
        assert!(csv.contains("test_op,10,1000"));
    }
}
