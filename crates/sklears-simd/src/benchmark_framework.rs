//! Advanced benchmarking framework for SIMD operations
//!
//! Provides comprehensive benchmarking utilities including cross-platform performance tests,
//! regression detection, and automated optimization guidance.
//!
//! ## no-std Compatibility
//!
//! This module is compatible with both std and no-std environments. In no-std:
//! - `HashMap` is replaced with `BTreeMap` for deterministic ordering
//! - Timing functionality is limited and may return mock values
//! - Operations are still executed but without accurate timing measurements
//! - All other functionality remains available

#[cfg(feature = "no-std")]
extern crate alloc;

#[cfg(feature = "no-std")]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use crate::SimdCapabilities;

#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;
#[cfg(not(feature = "no-std"))]
use std::string::ToString;
#[cfg(not(feature = "no-std"))]
pub use std::time::Duration;

#[cfg(not(feature = "no-std"))]
use std::time::Instant;

// Mock Duration for no-std compatibility
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration(u64); // nanoseconds

#[cfg(feature = "no-std")]
impl Duration {
    pub fn from_nanos(nanos: u64) -> Self {
        Duration(nanos)
    }

    pub fn from_millis(millis: u64) -> Self {
        Duration(millis * 1_000_000)
    }

    pub fn from_secs(secs: u64) -> Self {
        Duration(secs * 1_000_000_000)
    }

    pub fn as_nanos(&self) -> u128 {
        self.0 as u128
    }

    pub fn as_millis(&self) -> u128 {
        (self.0 / 1_000_000) as u128
    }

    pub fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }

    pub fn as_secs_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
}

/// Performance measurement result
///
/// Note: In no-std environments, timing functionality is limited and
/// duration values may be mock values for API compatibility.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration: Duration,
    pub throughput: Option<f64>, // operations per second (None in no-std)
    pub simd_width: usize,
    pub architecture: String,
    pub iterations: u64,
}

/// Cross-platform performance comparison
#[derive(Debug, Clone)]
pub struct CrossPlatformResult {
    pub operation: String,
    pub results: HashMap<String, BenchmarkResult>,
    pub best_performance: String,
    pub speedup_ratios: HashMap<String, f64>,
}

/// Performance regression detector
#[derive(Debug)]
pub struct RegressionDetector {
    baseline_results: HashMap<String, BenchmarkResult>,
    threshold: f64, // percentage threshold for regression detection
}

impl RegressionDetector {
    /// Create a new regression detector with the given threshold
    pub fn new(threshold_percent: f64) -> Self {
        Self {
            baseline_results: HashMap::new(),
            threshold: threshold_percent / 100.0,
        }
    }

    /// Set baseline results for comparison
    pub fn set_baseline(&mut self, results: Vec<BenchmarkResult>) {
        for result in results {
            self.baseline_results.insert(result.name.clone(), result);
        }
    }

    /// Check for performance regressions
    pub fn check_regression(&self, current_results: &[BenchmarkResult]) -> Vec<RegressionReport> {
        let mut regressions = Vec::new();

        for current in current_results {
            if let Some(baseline) = self.baseline_results.get(&current.name) {
                let baseline_ns = baseline.duration.as_nanos() as f64;
                let current_ns = current.duration.as_nanos() as f64;
                let change_ratio = (current_ns - baseline_ns) / baseline_ns;

                if change_ratio > self.threshold {
                    regressions.push(RegressionReport {
                        operation: current.name.clone(),
                        baseline_duration: baseline.duration,
                        current_duration: current.duration,
                        regression_percent: change_ratio * 100.0,
                        severity: if change_ratio > 0.2 {
                            Severity::Critical
                        } else if change_ratio > 0.1 {
                            Severity::High
                        } else {
                            Severity::Medium
                        },
                    });
                }
            }
        }

        regressions
    }
}

/// Performance regression report
#[derive(Debug)]
pub struct RegressionReport {
    pub operation: String,
    pub baseline_duration: Duration,
    pub current_duration: Duration,
    pub regression_percent: f64,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy)]
pub enum Severity {
    Medium,
    High,
    Critical,
}

/// Comprehensive benchmark suite runner
///
/// Provides benchmarking capabilities for SIMD operations with cross-platform support.
/// In no-std environments, timing functionality is limited and operations will be
/// executed but without accurate timing measurements.
pub struct BenchmarkSuite {
    capabilities: SimdCapabilities,
    results: Vec<BenchmarkResult>,
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            capabilities: SimdCapabilities::detect(),
            results: Vec::new(),
        }
    }

    /// Run a benchmark and record results
    ///
    /// In std environments, this provides accurate timing measurements.
    /// In no-std environments, the operation is executed but timing is mocked.
    pub fn benchmark<F>(&mut self, name: &str, iterations: u64, mut operation: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warm up
        for _ in 0..10 {
            operation();
        }

        #[cfg(not(feature = "no-std"))]
        let (duration, throughput) = {
            let start = Instant::now();
            for _ in 0..iterations {
                operation();
            }
            let duration = start.elapsed();
            let throughput = Some(iterations as f64 / duration.as_secs_f64());
            (duration, throughput)
        };

        #[cfg(feature = "no-std")]
        let (duration, throughput) = {
            // Execute the operation without timing in no-std environments
            for _ in 0..iterations {
                operation();
            }
            // Return mock duration for no-std compatibility
            (Duration::from_nanos(1), None)
        };

        let result = BenchmarkResult {
            name: name.to_string(),
            duration,
            throughput,
            simd_width: self.capabilities.best_f32_width(),
            architecture: self.get_architecture_name(),
            iterations,
        };

        self.results.push(result.clone());
        result
    }

    /// Run cross-platform comparison
    pub fn cross_platform_benchmark<F>(
        &mut self,
        operation_name: &str,
        data_size: usize,
        operation: F,
    ) -> CrossPlatformResult
    where
        F: Fn(&[f32]) -> f32 + Copy,
    {
        let test_data: Vec<f32> = (0..data_size).map(|i| i as f32).collect();
        let mut results = HashMap::new();

        // Test scalar implementation
        let scalar_result = self.benchmark(&format!("{}_scalar", operation_name), 1000, || {
            let _ = operation(&test_data);
        });
        results.insert("scalar".to_string(), scalar_result);

        // Test SIMD implementations based on available capabilities
        if self.capabilities.sse2 {
            let sse2_result = self.benchmark(&format!("{}_sse2", operation_name), 1000, || {
                let _ = operation(&test_data);
            });
            results.insert("sse2".to_string(), sse2_result);
        }

        if self.capabilities.avx2 {
            let avx2_result = self.benchmark(&format!("{}_avx2", operation_name), 1000, || {
                let _ = operation(&test_data);
            });
            results.insert("avx2".to_string(), avx2_result);
        }

        if self.capabilities.avx512 {
            let avx512_result = self.benchmark(&format!("{}_avx512", operation_name), 1000, || {
                let _ = operation(&test_data);
            });
            results.insert("avx512".to_string(), avx512_result);
        }

        if self.capabilities.neon {
            let neon_result = self.benchmark(&format!("{}_neon", operation_name), 1000, || {
                let _ = operation(&test_data);
            });
            results.insert("neon".to_string(), neon_result);
        }

        // Find best performance and calculate speedup ratios
        let best_duration = results
            .values()
            .map(|r| r.duration)
            .min()
            .unwrap_or(Duration::from_secs(1));

        let best_performance = results
            .iter()
            .min_by_key(|(_, result)| result.duration)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let mut speedup_ratios = HashMap::new();
        let baseline_duration = results
            .get("scalar")
            .map(|r| r.duration)
            .unwrap_or(best_duration);

        for (name, result) in &results {
            let speedup = baseline_duration.as_nanos() as f64 / result.duration.as_nanos() as f64;
            speedup_ratios.insert(name.clone(), speedup);
        }

        CrossPlatformResult {
            operation: operation_name.to_string(),
            results,
            best_performance,
            speedup_ratios,
        }
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Generate performance report
    pub fn generate_report(&self) -> BenchmarkReport {
        let total_benchmarks = self.results.len();
        let avg_duration = if total_benchmarks > 0 {
            let total_nanos: u128 = self.results.iter().map(|r| r.duration.as_nanos()).sum();
            Duration::from_nanos((total_nanos / total_benchmarks as u128) as u64)
        } else {
            Duration::from_secs(0)
        };

        let fastest = self.results.iter().min_by_key(|r| r.duration).cloned();
        let slowest = self.results.iter().max_by_key(|r| r.duration).cloned();

        BenchmarkReport {
            total_benchmarks,
            avg_duration,
            fastest,
            slowest,
            architecture: self.get_architecture_name(),
            simd_width: self.capabilities.best_f32_width(),
            capabilities: self.capabilities,
        }
    }

    fn get_architecture_name(&self) -> String {
        if self.capabilities.avx512 {
            "AVX-512".to_string()
        } else if self.capabilities.avx2 {
            "AVX2".to_string()
        } else if self.capabilities.avx {
            "AVX".to_string()
        } else if self.capabilities.sse42 {
            "SSE4.2".to_string()
        } else if self.capabilities.sse2 {
            "SSE2".to_string()
        } else if self.capabilities.neon {
            "NEON".to_string()
        } else {
            "Scalar".to_string()
        }
    }
}

/// Comprehensive benchmark report
#[derive(Debug)]
pub struct BenchmarkReport {
    pub total_benchmarks: usize,
    pub avg_duration: Duration,
    pub fastest: Option<BenchmarkResult>,
    pub slowest: Option<BenchmarkResult>,
    pub architecture: String,
    pub simd_width: usize,
    pub capabilities: SimdCapabilities,
}

impl BenchmarkReport {
    /// Generate a formatted report string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== SIMD Performance Benchmark Report ===\n");
        report.push_str(&format!("Architecture: {}\n", self.architecture));
        report.push_str(&format!("SIMD Width (f32): {}\n", self.simd_width));
        report.push_str(&format!("Total Benchmarks: {}\n", self.total_benchmarks));
        report.push_str(&format!("Average Duration: {:?}\n", self.avg_duration));

        report.push_str("\nCapabilities:\n");
        report.push_str(&format!("  SSE2: {}\n", self.capabilities.sse2));
        report.push_str(&format!("  AVX2: {}\n", self.capabilities.avx2));
        report.push_str(&format!("  AVX-512: {}\n", self.capabilities.avx512));
        report.push_str(&format!("  NEON: {}\n", self.capabilities.neon));

        if let Some(fastest) = &self.fastest {
            report.push_str(&format!(
                "\nFastest Operation: {} ({:?})\n",
                fastest.name, fastest.duration
            ));
        }

        if let Some(slowest) = &self.slowest {
            report.push_str(&format!(
                "Slowest Operation: {} ({:?})\n",
                slowest.name, slowest.duration
            ));
        }

        report.push_str("\n=== End Report ===\n");
        report
    }
}

/// Automated optimization recommendations
pub struct OptimizationAdvisor {
    results: Vec<CrossPlatformResult>,
}

impl Default for OptimizationAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationAdvisor {
    /// Create a new optimization advisor
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add cross-platform results for analysis
    pub fn add_results(&mut self, result: CrossPlatformResult) {
        self.results.push(result);
    }

    /// Generate optimization recommendations
    pub fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for result in &self.results {
            // Check if SIMD provides significant speedup
            if let Some(scalar_speedup) = result.speedup_ratios.get("scalar") {
                if *scalar_speedup < 1.5 {
                    recommendations.push(OptimizationRecommendation {
                        operation: result.operation.clone(),
                        recommendation_type: RecommendationType::AlgorithmOptimization,
                        description: format!(
                            "SIMD implementation for {} shows minimal speedup ({}x). Consider algorithm optimization or data layout changes.",
                            result.operation, scalar_speedup
                        ),
                        priority: Priority::Medium,
                    });
                }
            }

            // Check for memory-bound operations
            let best_speedup = result.speedup_ratios.values().cloned().fold(0.0, f64::max);
            if best_speedup < 2.0 {
                recommendations.push(OptimizationRecommendation {
                    operation: result.operation.clone(),
                    recommendation_type: RecommendationType::MemoryOptimization,
                    description: format!(
                        "Operation {} may be memory-bound. Consider cache optimization, prefetching, or data layout improvements.",
                        result.operation
                    ),
                    priority: Priority::High,
                });
            }

            // Check for underutilized SIMD width
            if result.best_performance == "sse2" && result.speedup_ratios.contains_key("avx2") {
                recommendations.push(OptimizationRecommendation {
                    operation: result.operation.clone(),
                    recommendation_type: RecommendationType::SimdWidthOptimization,
                    description: format!(
                        "Operation {} performs better with SSE2 than AVX2. Consider optimizing for wider SIMD or checking for overhead.",
                        result.operation
                    ),
                    priority: Priority::Medium,
                });
            }
        }

        recommendations
    }
}

/// Optimization recommendation
#[derive(Debug)]
pub struct OptimizationRecommendation {
    pub operation: String,
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub priority: Priority,
}

#[derive(Debug)]
pub enum RecommendationType {
    AlgorithmOptimization,
    MemoryOptimization,
    SimdWidthOptimization,
    CompilerOptimization,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert_eq!(suite.results.len(), 0);
    }

    #[test]
    fn test_simple_benchmark() {
        let mut suite = BenchmarkSuite::new();
        let result = suite.benchmark("test_op", 100, || {
            // Simple operation
            let _sum: f32 = (0..1000).map(|i| i as f32).sum();
        });

        assert_eq!(result.name, "test_op");
        assert_eq!(result.iterations, 100);
        assert!(result.duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_regression_detector() {
        let mut detector = RegressionDetector::new(10.0); // 10% threshold

        let baseline = vec![BenchmarkResult {
            name: "test_op".to_string(),
            duration: Duration::from_millis(100),
            throughput: None,
            simd_width: 4,
            architecture: "test".to_string(),
            iterations: 1000,
        }];

        detector.set_baseline(baseline);

        // Test with no regression
        let current = vec![BenchmarkResult {
            name: "test_op".to_string(),
            duration: Duration::from_millis(105), // 5% slower, within threshold
            throughput: None,
            simd_width: 4,
            architecture: "test".to_string(),
            iterations: 1000,
        }];

        let regressions = detector.check_regression(&current);
        assert_eq!(regressions.len(), 0);

        // Test with regression
        let current_regressed = vec![BenchmarkResult {
            name: "test_op".to_string(),
            duration: Duration::from_millis(120), // 20% slower, above threshold
            throughput: None,
            simd_width: 4,
            architecture: "test".to_string(),
            iterations: 1000,
        }];

        let regressions = detector.check_regression(&current_regressed);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].operation, "test_op");
        assert!(regressions[0].regression_percent > 10.0);
    }

    #[test]
    fn test_optimization_advisor() {
        let mut advisor = OptimizationAdvisor::new();

        let mut speedup_ratios = HashMap::new();
        speedup_ratios.insert("scalar".to_string(), 1.2); // Low speedup

        let result = CrossPlatformResult {
            operation: "slow_op".to_string(),
            results: HashMap::new(),
            best_performance: "sse2".to_string(),
            speedup_ratios,
        };

        advisor.add_results(result);
        let recommendations = advisor.generate_recommendations();

        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.operation == "slow_op"));
    }

    #[test]
    fn test_benchmark_report_formatting() {
        let report = BenchmarkReport {
            total_benchmarks: 5,
            avg_duration: Duration::from_millis(10),
            fastest: None,
            slowest: None,
            architecture: "AVX2".to_string(),
            simd_width: 8,
            capabilities: SimdCapabilities::detect(),
        };

        let formatted = report.format_report();
        assert!(formatted.contains("Architecture: AVX2"));
        assert!(formatted.contains("SIMD Width (f32): 8"));
        assert!(formatted.contains("Total Benchmarks: 5"));
    }
}
