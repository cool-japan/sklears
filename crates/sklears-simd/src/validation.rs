//! Validation framework for SIMD operations
//!
//! This module provides comprehensive validation tools for ensuring numerical accuracy,
//! correctness, and performance of SIMD implementations against reference implementations.

#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;
#[cfg(not(feature = "no-std"))]
use std::string::{String, ToString};
#[cfg(not(feature = "no-std"))]
use std::time::Instant;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "no-std")]
use alloc::string::{String, ToString};
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(feature = "no-std")]
use alloc::{format, vec};

// Mock types for no-std compatibility
#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Instant(u64);

#[cfg(feature = "no-std")]
#[derive(Debug, Clone, Copy)]
pub struct Duration(u64);

#[cfg(feature = "no-std")]
impl Instant {
    pub fn now() -> Self {
        Instant(0) // Mock implementation for no-std
    }

    pub fn elapsed(&self) -> Duration {
        Duration(0) // Mock implementation
    }
}

#[cfg(feature = "no-std")]
impl Duration {
    pub fn as_nanos(&self) -> u128 {
        0 // Mock implementation
    }
}

/// Numerical precision validation with configurable tolerances
pub mod precision {
    use super::*;

    /// Tolerance levels for different types of operations
    #[derive(Debug, Clone, Copy)]
    pub struct Tolerance {
        pub absolute: f64,
        pub relative: f64,
    }

    impl Tolerance {
        pub const STRICT: Self = Self {
            absolute: 1e-15,
            relative: 1e-14,
        };

        pub const NORMAL: Self = Self {
            absolute: 1e-12,
            relative: 1e-11,
        };

        pub const RELAXED: Self = Self {
            absolute: 1e-9,
            relative: 1e-8,
        };

        pub const VERY_RELAXED: Self = Self {
            absolute: 1e-6,
            relative: 1e-5,
        };
    }

    /// Compare two floating-point values with given tolerance
    pub fn compare_f32(a: f32, b: f32, tolerance: Tolerance) -> bool {
        let abs_diff = (a - b).abs() as f64;
        let rel_diff = if b != 0.0 {
            abs_diff / (b.abs() as f64)
        } else {
            abs_diff
        };

        abs_diff <= tolerance.absolute || rel_diff <= tolerance.relative
    }

    /// Compare two f64 values with given tolerance
    pub fn compare_f64(a: f64, b: f64, tolerance: Tolerance) -> bool {
        let abs_diff = (a - b).abs();
        let rel_diff = if b != 0.0 {
            abs_diff / b.abs()
        } else {
            abs_diff
        };

        abs_diff <= tolerance.absolute || rel_diff <= tolerance.relative
    }

    /// Compare two slices of f32 values
    pub fn compare_f32_slice(a: &[f32], b: &[f32], tolerance: Tolerance) -> ValidationResult {
        if a.len() != b.len() {
            return ValidationResult::error("Length mismatch");
        }

        let mut mismatches = Vec::new();
        let mut max_abs_error = 0.0f64;
        let mut max_rel_error = 0.0f64;

        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            if !compare_f32(val_a, val_b, tolerance) {
                let abs_error = (val_a - val_b).abs() as f64;
                let rel_error = if val_b != 0.0 {
                    abs_error / (val_b.abs() as f64)
                } else {
                    abs_error
                };

                max_abs_error = max_abs_error.max(abs_error);
                max_rel_error = max_rel_error.max(rel_error);

                mismatches.push(ValidationError {
                    index: Some(i),
                    expected: val_b as f64,
                    actual: val_a as f64,
                    abs_error,
                    rel_error,
                    description: format!("Mismatch at index {}", i),
                });

                if mismatches.len() >= 10 {
                    break; // Limit reported errors
                }
            }
        }

        if mismatches.is_empty() {
            ValidationResult::success()
        } else {
            let failed_count = mismatches.len();
            ValidationResult {
                passed: false,
                errors: mismatches,
                statistics: Some(ValidationStatistics {
                    max_abs_error,
                    max_rel_error,
                    total_comparisons: a.len(),
                    failed_comparisons: failed_count,
                }),
            }
        }
    }

    /// Compare two slices of f64 values
    pub fn compare_f64_slice(a: &[f64], b: &[f64], tolerance: Tolerance) -> ValidationResult {
        if a.len() != b.len() {
            return ValidationResult::error("Length mismatch");
        }

        let mut mismatches = Vec::new();
        let mut max_abs_error = 0.0f64;
        let mut max_rel_error = 0.0f64;

        for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
            if !compare_f64(val_a, val_b, tolerance) {
                let abs_error = (val_a - val_b).abs();
                let rel_error = if val_b != 0.0 {
                    abs_error / val_b.abs()
                } else {
                    abs_error
                };

                max_abs_error = max_abs_error.max(abs_error);
                max_rel_error = max_rel_error.max(rel_error);

                mismatches.push(ValidationError {
                    index: Some(i),
                    expected: val_b,
                    actual: val_a,
                    abs_error,
                    rel_error,
                    description: format!("Mismatch at index {}", i),
                });

                if mismatches.len() >= 10 {
                    break;
                }
            }
        }

        if mismatches.is_empty() {
            ValidationResult::success()
        } else {
            let failed_count = mismatches.len();
            ValidationResult {
                passed: false,
                errors: mismatches,
                statistics: Some(ValidationStatistics {
                    max_abs_error,
                    max_rel_error,
                    total_comparisons: a.len(),
                    failed_comparisons: failed_count,
                }),
            }
        }
    }
}

/// Edge case testing for special values
pub mod edge_cases {
    use super::*;

    /// Special floating-point test values
    pub fn get_special_f32_values() -> Vec<f32> {
        vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            core::f32::consts::PI,
            core::f32::consts::E,
            1e-30,
            1e30,
            -1e-30,
            -1e30,
        ]
    }

    /// Special floating-point test values for f64
    pub fn get_special_f64_values() -> Vec<f64> {
        vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            core::f64::consts::PI,
            core::f64::consts::E,
            1e-100,
            1e100,
            -1e-100,
            -1e100,
        ]
    }

    /// Test a function with edge case values
    pub fn test_unary_f32<F>(
        func: F,
        reference_func: F,
        tolerance: precision::Tolerance,
    ) -> ValidationResult
    where
        F: Fn(f32) -> f32,
    {
        let test_values = get_special_f32_values();
        let mut errors = Vec::new();

        for &val in &test_values {
            let result = func(val);
            let expected = reference_func(val);

            if !are_equal_with_nan_handling_f32(result, expected, tolerance) {
                errors.push(ValidationError {
                    index: None,
                    expected: expected as f64,
                    actual: result as f64,
                    abs_error: (result - expected).abs() as f64,
                    rel_error: if expected != 0.0 {
                        ((result - expected) / expected).abs() as f64
                    } else {
                        (result - expected).abs() as f64
                    },
                    description: format!("Edge case failure for input: {}", val),
                });
            }
        }

        if errors.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult {
                passed: false,
                errors,
                statistics: None,
            }
        }
    }

    /// Test a binary function with edge case combinations
    pub fn test_binary_f32<F>(
        func: F,
        reference_func: F,
        tolerance: precision::Tolerance,
    ) -> ValidationResult
    where
        F: Fn(f32, f32) -> f32,
    {
        let test_values = get_special_f32_values();
        let mut errors = Vec::new();

        for &a in &test_values {
            for &b in &test_values {
                let result = func(a, b);
                let expected = reference_func(a, b);

                if !are_equal_with_nan_handling_f32(result, expected, tolerance) {
                    errors.push(ValidationError {
                        index: None,
                        expected: expected as f64,
                        actual: result as f64,
                        abs_error: (result - expected).abs() as f64,
                        rel_error: if expected != 0.0 {
                            ((result - expected) / expected).abs() as f64
                        } else {
                            (result - expected).abs() as f64
                        },
                        description: format!("Edge case failure for inputs: {}, {}", a, b),
                    });

                    if errors.len() >= 20 {
                        break;
                    }
                }
            }
            if errors.len() >= 20 {
                break;
            }
        }

        if errors.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult {
                passed: false,
                errors,
                statistics: None,
            }
        }
    }

    fn are_equal_with_nan_handling_f32(a: f32, b: f32, tolerance: precision::Tolerance) -> bool {
        if a.is_nan() && b.is_nan() {
            true
        } else if a.is_infinite() && b.is_infinite() {
            a.signum() == b.signum()
        } else {
            precision::compare_f32(a, b, tolerance)
        }
    }
}

/// Correctness verification against reference implementations
pub mod correctness {
    use super::*;

    /// Verify SIMD implementation against scalar reference
    pub fn verify_against_scalar<F1, F2, T, R>(
        simd_func: F1,
        scalar_func: F2,
        test_data: &[T],
        tolerance: precision::Tolerance,
        operation_name: &str,
    ) -> ValidationResult
    where
        F1: Fn(&[T]) -> R,
        F2: Fn(&[T]) -> R,
        R: PartialEq + core::fmt::Debug + Clone,
    {
        let simd_result = simd_func(test_data);
        let scalar_result = scalar_func(test_data);

        if simd_result == scalar_result {
            ValidationResult::success()
        } else {
            ValidationResult::error(&format!(
                "SIMD result {:?} does not match scalar result {:?} for operation: {}",
                simd_result, scalar_result, operation_name
            ))
        }
    }

    /// Verify SIMD f32 slice operations
    pub fn verify_f32_slice_operation<F1, F2>(
        simd_func: F1,
        scalar_func: F2,
        test_data: &[f32],
        tolerance: precision::Tolerance,
        operation_name: &str,
    ) -> ValidationResult
    where
        F1: Fn(&[f32]) -> Vec<f32>,
        F2: Fn(&[f32]) -> Vec<f32>,
    {
        let simd_result = simd_func(test_data);
        let scalar_result = scalar_func(test_data);

        let mut validation_result =
            precision::compare_f32_slice(&simd_result, &scalar_result, tolerance);

        if !validation_result.passed {
            for error in &mut validation_result.errors {
                error.description = format!("{}: {}", operation_name, error.description);
            }
        }

        validation_result
    }

    /// Verify SIMD f64 slice operations
    pub fn verify_f64_slice_operation<F1, F2>(
        simd_func: F1,
        scalar_func: F2,
        test_data: &[f64],
        tolerance: precision::Tolerance,
        operation_name: &str,
    ) -> ValidationResult
    where
        F1: Fn(&[f64]) -> Vec<f64>,
        F2: Fn(&[f64]) -> Vec<f64>,
    {
        let simd_result = simd_func(test_data);
        let scalar_result = scalar_func(test_data);

        let mut validation_result =
            precision::compare_f64_slice(&simd_result, &scalar_result, tolerance);

        if !validation_result.passed {
            for error in &mut validation_result.errors {
                error.description = format!("{}: {}", operation_name, error.description);
            }
        }

        validation_result
    }

    /// Generate comprehensive test datasets for validation
    pub fn generate_test_datasets_f32() -> Vec<Vec<f32>> {
        vec![
            // Empty
            vec![],
            // Single element
            vec![1.0],
            // Small arrays
            vec![1.0, 2.0, 3.0],
            vec![-1.0, 0.0, 1.0],
            // Power-of-2 sizes for SIMD alignment
            (0..4).map(|i| i as f32).collect(),
            (0..8).map(|i| i as f32).collect(),
            (0..16).map(|i| i as f32).collect(),
            (0..32).map(|i| i as f32).collect(),
            // Non-power-of-2 sizes
            (0..7).map(|i| i as f32).collect(),
            (0..15).map(|i| i as f32).collect(),
            (0..31).map(|i| i as f32).collect(),
            // Large arrays
            (0..1000).map(|i| (i as f32) * 0.1).collect(),
            // Random-like data
            vec![
                0.1, -2.3, 4.7, -0.9, 8.2, -3.1, 5.6, -7.4, 1.8, -6.5, 9.3, -4.7, 2.1, -8.9, 3.4,
                -1.2,
            ],
            // Large values
            vec![1e10, -1e10, 1e20, -1e20],
            // Small values
            vec![1e-10, -1e-10, 1e-20, -1e-20],
            // Mixed scales
            vec![1e-10, 1.0, 1e10, -1e-10, -1.0, -1e10],
        ]
    }

    /// Generate comprehensive test datasets for f64
    pub fn generate_test_datasets_f64() -> Vec<Vec<f64>> {
        vec![
            // Empty
            vec![],
            // Single element
            vec![1.0],
            // Small arrays
            vec![1.0, 2.0, 3.0],
            vec![-1.0, 0.0, 1.0],
            // Power-of-2 sizes
            (0..4).map(|i| i as f64).collect(),
            (0..8).map(|i| i as f64).collect(),
            (0..16).map(|i| i as f64).collect(),
            // Large arrays
            (0..1000).map(|i| (i as f64) * 0.1).collect(),
            // High precision values
            vec![
                core::f64::consts::PI,
                core::f64::consts::E,
                core::f64::consts::SQRT_2,
                core::f64::consts::LN_2,
            ],
            // Extreme values
            vec![f64::MIN, f64::MAX, f64::MIN_POSITIVE],
        ]
    }
}

/// Performance regression detection
pub mod performance {
    use super::*;

    /// Performance measurement result
    #[derive(Debug, Clone)]
    pub struct PerformanceResult {
        pub operation_name: String,
        pub duration_ns: u64,
        pub throughput_ops_per_sec: f64,
        pub data_size: usize,
    }

    /// Benchmark a function and return performance metrics
    pub fn benchmark_function<F, T, R>(
        func: F,
        data: &[T],
        operation_name: &str,
        iterations: usize,
    ) -> PerformanceResult
    where
        F: Fn(&[T]) -> R,
        T: Clone,
    {
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = func(data);
        }

        let duration = start.elapsed();
        let duration_ns = duration.as_nanos() as u64;
        let avg_duration_ns = duration_ns / iterations as u64;
        let throughput = if avg_duration_ns > 0 {
            1_000_000_000.0 / (avg_duration_ns as f64)
        } else {
            f64::INFINITY
        };

        PerformanceResult {
            operation_name: operation_name.to_string(),
            duration_ns: avg_duration_ns,
            throughput_ops_per_sec: throughput,
            data_size: data.len(),
        }
    }

    /// Compare SIMD vs scalar performance
    pub fn compare_simd_vs_scalar<F1, F2, T, R>(
        simd_func: F1,
        scalar_func: F2,
        data: &[T],
        operation_name: &str,
        iterations: usize,
    ) -> PerformanceComparison
    where
        F1: Fn(&[T]) -> R,
        F2: Fn(&[T]) -> R,
        T: Clone,
    {
        let simd_result = benchmark_function(
            simd_func,
            data,
            &format!("{operation_name}_simd"),
            iterations,
        );

        let scalar_result = benchmark_function(
            scalar_func,
            data,
            &format!("{operation_name}_scalar"),
            iterations,
        );

        let speedup = if scalar_result.duration_ns > 0 {
            scalar_result.duration_ns as f64 / simd_result.duration_ns as f64
        } else {
            1.0
        };

        PerformanceComparison {
            operation_name: operation_name.to_string(),
            simd_result,
            scalar_result,
            speedup,
        }
    }

    /// Performance regression threshold check
    pub fn check_performance_regression(
        current: &PerformanceResult,
        baseline: &PerformanceResult,
        max_regression_percent: f64,
    ) -> ValidationResult {
        if baseline.duration_ns == 0 {
            return ValidationResult::error("Baseline duration is zero");
        }

        let regression_ratio = current.duration_ns as f64 / baseline.duration_ns as f64;
        let regression_percent = (regression_ratio - 1.0) * 100.0;

        if regression_percent > max_regression_percent {
            ValidationResult::error(&format!(
                "Performance regression detected: {regression_percent:.2}% slower than baseline (max allowed: {max_regression_percent:.2}%)"
            ))
        } else {
            ValidationResult::success()
        }
    }

    #[derive(Debug, Clone)]
    pub struct PerformanceComparison {
        pub operation_name: String,
        pub simd_result: PerformanceResult,
        pub scalar_result: PerformanceResult,
        pub speedup: f64,
    }
}

/// Core validation types and utilities
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub index: Option<usize>,
    pub expected: f64,
    pub actual: f64,
    pub abs_error: f64,
    pub rel_error: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub total_comparisons: usize,
    pub failed_comparisons: usize,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub errors: Vec<ValidationError>,
    pub statistics: Option<ValidationStatistics>,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self {
            passed: true,
            errors: Vec::new(),
            statistics: None,
        }
    }

    pub fn error(message: &str) -> Self {
        Self {
            passed: false,
            errors: vec![ValidationError {
                index: None,
                expected: 0.0,
                actual: 0.0,
                abs_error: 0.0,
                rel_error: 0.0,
                description: message.to_string(),
            }],
            statistics: None,
        }
    }

    pub fn combine(mut self, other: ValidationResult) -> Self {
        self.passed = self.passed && other.passed;
        self.errors.extend(other.errors);
        self
    }
}

/// Comprehensive validation suite
pub struct ValidationSuite {
    pub results: HashMap<String, ValidationResult>,
    pub performance_results: HashMap<String, performance::PerformanceResult>,
}

impl Default for ValidationSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationSuite {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            performance_results: HashMap::new(),
        }
    }

    pub fn add_result(&mut self, name: String, result: ValidationResult) {
        self.results.insert(name, result);
    }

    pub fn add_performance_result(&mut self, name: String, result: performance::PerformanceResult) {
        self.performance_results.insert(name, result);
    }

    pub fn all_passed(&self) -> bool {
        self.results.values().all(|r| r.passed)
    }

    pub fn print_summary(&self) {
        let total_tests = self.results.len();
        let passed_tests = self.results.values().filter(|r| r.passed).count();

        #[cfg(not(feature = "no-std"))]
        {
            println!("Validation Summary:");
            println!("  Total tests: {total_tests}");
            println!("  Passed: {passed_tests}");
            println!("  Failed: {}", total_tests - passed_tests);

            for (name, result) in &self.results {
                if !result.passed {
                    println!("  FAILED: {name}");
                    for error in &result.errors {
                        println!("    {}", error.description);
                    }
                }
            }

            if !self.performance_results.is_empty() {
                println!("\nPerformance Results:");
                for (name, perf) in &self.performance_results {
                    println!(
                        "  {}: {:.2} ns/op ({:.2e} ops/sec)",
                        name, perf.duration_ns, perf.throughput_ops_per_sec
                    );
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_comparison() {
        assert!(precision::compare_f32(
            1.0,
            1.0,
            precision::Tolerance::STRICT
        ));
        assert!(precision::compare_f32(
            1.0,
            1.0 + 1e-12,
            precision::Tolerance::NORMAL
        ));
        assert!(!precision::compare_f32(
            1.0,
            1.1,
            precision::Tolerance::STRICT
        ));
    }

    #[test]
    fn test_edge_cases() {
        let special_values = edge_cases::get_special_f32_values();
        assert!(special_values.iter().any(|x| x.is_nan())); // NaN comparison needs special handling
        assert!(special_values.contains(&f32::INFINITY));
        assert!(special_values.contains(&0.0));
    }

    #[test]
    fn test_slice_comparison() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = precision::compare_f32_slice(&a, &b, precision::Tolerance::NORMAL);
        assert!(result.passed);

        let c = vec![1.0, 2.1, 3.0];
        let result2 = precision::compare_f32_slice(&a, &c, precision::Tolerance::STRICT);
        assert!(!result2.passed);
    }

    #[test]
    fn test_validation_suite() {
        let mut suite = ValidationSuite::new();
        suite.add_result("test1".to_string(), ValidationResult::success());
        suite.add_result("test2".to_string(), ValidationResult::error("Test error"));

        assert!(!suite.all_passed());
        assert_eq!(suite.results.len(), 2);
    }

    #[test]
    fn test_performance_measurement() {
        let data = vec![1.0f32; 1000];
        let result = performance::benchmark_function(
            |slice| slice.iter().sum::<f32>(),
            &data,
            "sum_test",
            100,
        );

        assert_eq!(result.operation_name, "sum_test");
        assert!(result.duration_ns > 0);
        assert!(result.throughput_ops_per_sec > 0.0);
    }

    #[test]
    fn test_test_data_generation() {
        let datasets = correctness::generate_test_datasets_f32();
        assert!(!datasets.is_empty());
        assert!(datasets.iter().any(|d| d.is_empty()));
        assert!(datasets.iter().any(|d| d.len() == 1));
        assert!(datasets.iter().any(|d| d.len() > 100));
    }
}
