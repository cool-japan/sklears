//! Validation Framework for Machine Learning Metrics
//!
//! This module provides comprehensive validation capabilities for ensuring
//! metric implementations are correct, reliable, and consistent. It includes
//! validation against known results, synthetic data generation for testing,
//! and metameric analysis for understanding metric behavior.
//!
//! # Features
//!
//! - Validation against reference implementations and known results
//! - Synthetic data generation for comprehensive testing
//! - Cross-validation for metric stability assessment
//! - Metameric analysis for understanding metric behavior patterns
//! - Automated benchmarking against standard datasets
//! - Statistical significance testing for metric comparisons
//! - Edge case testing and robustness analysis
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::validation::*;
//! use sklears_metrics::classification::accuracy_score;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create a validation framework
//! let mut validator = MetricValidator::new();
//!
//! // Add reference test cases
//! let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0]);
//! let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0]);
//!
//! validator.add_reference_case(
//!     "accuracy_basic".to_string(),
//!     y_true.clone(),
//!     y_pred.clone(),
//!     0.8, // Expected accuracy
//!     1e-10 // Tolerance
//! );
//!
//! // Validate accuracy implementation
//! let result = validator.validate_metric(
//!     "accuracy_basic",
//!     |y_true, y_pred| accuracy_score(y_true, y_pred).unwrap_or(0.0)
//! );
//!
//! println!("Validation result: {:?}", result);
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{s, Array1};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::Distribution;
// Normal distribution available via RandNormal per SciRS2 policy
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;

/// Configuration for the validation framework
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Tolerance for numerical comparisons
    pub numerical_tolerance: f64,
    /// Number of bootstrap samples for stability testing
    pub bootstrap_samples: usize,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to perform edge case testing
    pub test_edge_cases: bool,
    /// Whether to generate synthetic data for testing
    pub generate_synthetic_data: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            numerical_tolerance: 1e-10,
            bootstrap_samples: 1000,
            cv_folds: 5,
            seed: Some(42),
            test_edge_cases: true,
            generate_synthetic_data: true,
        }
    }
}

/// Result of metric validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the validation passed
    pub passed: bool,
    /// Expected value
    pub expected: f64,
    /// Actual computed value
    pub actual: f64,
    /// Absolute error
    pub absolute_error: f64,
    /// Relative error (if expected != 0)
    pub relative_error: Option<f64>,
    /// Additional validation metadata
    pub metadata: HashMap<String, f64>,
    /// Error message if validation failed
    pub error_message: Option<String>,
}

impl ValidationResult {
    pub fn new(expected: f64, actual: f64, tolerance: f64) -> Self {
        let absolute_error = (expected - actual).abs();
        let relative_error = if expected.abs() > f64::EPSILON {
            Some(absolute_error / expected.abs())
        } else {
            None
        };

        let passed = absolute_error <= tolerance;
        let error_message = if !passed {
            Some(format!(
                "Validation failed: expected {}, got {}, absolute error {} exceeds tolerance {}",
                expected, actual, absolute_error, tolerance
            ))
        } else {
            None
        };

        Self {
            passed,
            expected,
            actual,
            absolute_error,
            relative_error,
            metadata: HashMap::new(),
            error_message,
        }
    }

    pub fn with_metadata(mut self, key: String, value: f64) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Reference test case for metric validation
#[derive(Debug, Clone)]
pub struct ReferenceTestCase {
    pub name: String,
    pub y_true: Array1<f64>,
    pub y_pred: Array1<f64>,
    pub expected_value: f64,
    pub tolerance: f64,
    pub description: String,
}

/// Synthetic data generator for testing
pub struct SyntheticDataGenerator {
    rng: StdRng,
    _config: ValidationConfig,
}

impl SyntheticDataGenerator {
    pub fn new(config: ValidationConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed.unwrap_or(42));
        Self {
            rng,
            _config: config,
        }
    }

    /// Generate perfect predictions (no error)
    pub fn generate_perfect_predictions(&mut self, n_samples: usize) -> (Array1<f64>, Array1<f64>) {
        let y_true: Array1<f64> = (0..n_samples)
            .map(|_| self.rng.gen_range(0.0..10.0))
            .collect();
        let y_pred = y_true.clone();
        (y_true, y_pred)
    }

    /// Generate random predictions with specified noise level
    pub fn generate_noisy_predictions(
        &mut self,
        n_samples: usize,
        noise_std: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let y_true: Array1<f64> = (0..n_samples)
            .map(|_| self.rng.gen_range(0.0..10.0))
            .collect();

        let noise_dist = scirs2_core::random::RandNormal::new(0.0, noise_std).unwrap();
        let y_pred: Array1<f64> = y_true
            .iter()
            .map(|&true_val| true_val + noise_dist.sample(&mut self.rng))
            .collect();

        (y_true, y_pred)
    }

    /// Generate adversarial worst-case predictions
    pub fn generate_worst_case_predictions(
        &mut self,
        n_samples: usize,
    ) -> (Array1<f64>, Array1<f64>) {
        let y_true: Array1<f64> = (0..n_samples)
            .map(|_| if self.rng.gen::<bool>() { 1.0 } else { 0.0 })
            .collect();

        // Predictions are opposite of true values
        let y_pred: Array1<f64> = y_true.iter().map(|&true_val| 1.0 - true_val).collect();

        (y_true, y_pred)
    }

    /// Generate classification data with specified accuracy
    pub fn generate_classification_with_accuracy(
        &mut self,
        n_samples: usize,
        target_accuracy: f64,
    ) -> (Array1<i32>, Array1<i32>) {
        let y_true: Array1<i32> = (0..n_samples)
            .map(|_| if self.rng.gen::<bool>() { 1 } else { 0 })
            .collect();

        let n_correct = (target_accuracy * n_samples as f64).round() as usize;
        let mut y_pred = y_true.clone();

        // Randomly flip predictions to achieve target accuracy
        let indices: Vec<usize> = (0..n_samples).collect();
        let mut shuffled_indices = indices;
        shuffled_indices.shuffle(&mut self.rng);

        for &idx in shuffled_indices.iter().take(n_samples - n_correct) {
            y_pred[idx] = 1 - y_pred[idx];
        }

        (y_true, y_pred)
    }

    /// Generate regression data with specified R²
    pub fn generate_regression_with_r2(
        &mut self,
        n_samples: usize,
        target_r2: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let y_true: Array1<f64> = (0..n_samples)
            .map(|_| self.rng.gen_range(0.0..10.0))
            .collect();

        let _true_mean = y_true.mean().unwrap_or(0.0);
        let true_var = y_true.var(0.0);

        // Generate noise to achieve target R²
        let noise_var = true_var * (1.0 - target_r2) / target_r2;
        let noise_std = noise_var.sqrt();

        let noise_dist = scirs2_core::random::RandNormal::new(0.0, noise_std).unwrap();
        let y_pred: Array1<f64> = y_true
            .iter()
            .map(|&true_val| true_val + noise_dist.sample(&mut self.rng))
            .collect();

        (y_true, y_pred)
    }

    /// Generate edge case data
    pub fn generate_edge_cases(&mut self) -> Vec<(String, Array1<f64>, Array1<f64>)> {
        let mut cases = vec![
            (
                "single_element".to_string(),
                Array1::from_vec(vec![1.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                "all_zeros".to_string(),
                Array1::from_vec(vec![0.0; 10]),
                Array1::from_vec(vec![0.0; 10]),
            ),
            (
                "all_same".to_string(),
                Array1::from_vec(vec![5.0; 10]),
                Array1::from_vec(vec![5.0; 10]),
            ),
            (
                "large_values".to_string(),
                Array1::from_vec(vec![1e6; 10]),
                Array1::from_vec(vec![1e6; 10]),
            ),
            (
                "small_values".to_string(),
                Array1::from_vec(vec![1e-6; 10]),
                Array1::from_vec(vec![1e-6; 10]),
            ),
        ];

        // Mixed positive/negative
        let mixed_true = Array1::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0]);
        let mixed_pred = Array1::from_vec(vec![-4.8, -1.1, 0.1, 0.9, 5.2]);
        cases.push(("mixed_signs".to_string(), mixed_true, mixed_pred));

        cases
    }

    /// Generate outlier data
    pub fn generate_with_outliers(
        &mut self,
        n_samples: usize,
        outlier_fraction: f64,
        outlier_magnitude: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let y_true: Vec<f64> = (0..n_samples)
            .map(|_| self.rng.gen_range(0.0..10.0))
            .collect();

        let mut y_pred = y_true.clone();

        // Add outliers
        let n_outliers = (outlier_fraction * n_samples as f64).round() as usize;
        let indices: Vec<usize> = (0..n_samples).collect();
        let mut shuffled_indices = indices;
        shuffled_indices.shuffle(&mut self.rng);

        for &idx in shuffled_indices.iter().take(n_outliers) {
            y_pred[idx] += outlier_magnitude * if self.rng.gen::<bool>() { 1.0 } else { -1.0 };
        }

        (Array1::from_vec(y_true), Array1::from_vec(y_pred))
    }
}

/// Main metric validation framework
pub struct MetricValidator {
    config: ValidationConfig,
    reference_cases: HashMap<String, ReferenceTestCase>,
    data_generator: SyntheticDataGenerator,
}

impl MetricValidator {
    pub fn new() -> Self {
        let config = ValidationConfig::default();
        let data_generator = SyntheticDataGenerator::new(config.clone());

        Self {
            config,
            reference_cases: HashMap::new(),
            data_generator,
        }
    }

    pub fn with_config(config: ValidationConfig) -> Self {
        let data_generator = SyntheticDataGenerator::new(config.clone());

        Self {
            config,
            reference_cases: HashMap::new(),
            data_generator,
        }
    }

    /// Add a reference test case
    pub fn add_reference_case(
        &mut self,
        name: String,
        y_true: Array1<f64>,
        y_pred: Array1<f64>,
        expected_value: f64,
        tolerance: f64,
    ) {
        let test_case = ReferenceTestCase {
            name: name.clone(),
            y_true,
            y_pred,
            expected_value,
            tolerance,
            description: format!("Reference test case: {}", name),
        };

        self.reference_cases.insert(name, test_case);
    }

    /// Validate a metric against a specific reference case
    pub fn validate_metric<F>(
        &self,
        case_name: &str,
        metric_fn: F,
    ) -> MetricsResult<ValidationResult>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64,
    {
        let test_case = self.reference_cases.get(case_name).ok_or_else(|| {
            MetricsError::InvalidParameter(format!("Reference case '{}' not found", case_name))
        })?;

        let actual_value = metric_fn(&test_case.y_true, &test_case.y_pred);
        let result =
            ValidationResult::new(test_case.expected_value, actual_value, test_case.tolerance);

        Ok(result)
    }

    /// Validate a metric against all reference cases
    pub fn validate_all_cases<F>(&self, metric_fn: F) -> HashMap<String, ValidationResult>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64 + Copy,
    {
        let mut results = HashMap::new();

        for name in self.reference_cases.keys() {
            if let Ok(result) = self.validate_metric(name, metric_fn) {
                results.insert(name.clone(), result);
            }
        }

        results
    }

    /// Perform comprehensive validation including synthetic data
    pub fn comprehensive_validation<F>(
        &mut self,
        metric_fn: F,
        metric_name: &str,
    ) -> ComprehensiveValidationReport
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64 + Copy,
    {
        let mut report = ComprehensiveValidationReport::new(metric_name.to_string());

        // Validate against reference cases
        let reference_results = self.validate_all_cases(metric_fn);
        report.reference_validation = reference_results;

        // Test with perfect predictions
        if self.config.generate_synthetic_data {
            let (y_true, y_pred) = self.data_generator.generate_perfect_predictions(100);
            let perfect_result = metric_fn(&y_true, &y_pred);
            report.perfect_prediction_result = Some(perfect_result);
        }

        // Test edge cases
        if self.config.test_edge_cases {
            let edge_cases = self.data_generator.generate_edge_cases();
            for (case_name, y_true, y_pred) in edge_cases {
                let result = metric_fn(&y_true, &y_pred);
                report.edge_case_results.insert(case_name, result);
            }
        }

        // Test stability with bootstrap sampling
        if self.config.bootstrap_samples > 0 {
            let stability_result = self.test_metric_stability(metric_fn);
            report.stability_analysis = Some(stability_result);
        }

        // Test with various noise levels
        let noise_levels = vec![0.1, 0.5, 1.0, 2.0];
        for &noise in &noise_levels {
            let (y_true, y_pred) = self.data_generator.generate_noisy_predictions(100, noise);
            let result = metric_fn(&y_true, &y_pred);
            report.noise_sensitivity.insert(noise.to_string(), result);
        }

        report
    }

    /// Test metric stability using bootstrap sampling
    fn test_metric_stability<F>(&mut self, metric_fn: F) -> StabilityAnalysis
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64,
    {
        let (y_true, y_pred) = self.data_generator.generate_noisy_predictions(200, 0.5);
        let original_result = metric_fn(&y_true, &y_pred);

        let mut bootstrap_results = Vec::new();
        let mut rng = StdRng::seed_from_u64(self.config.seed.unwrap_or(42));

        for _ in 0..self.config.bootstrap_samples {
            // Bootstrap sampling
            let n = y_true.len();
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let y_true_boot: Array1<f64> = indices.iter().map(|&i| y_true[i]).collect();
            let y_pred_boot: Array1<f64> = indices.iter().map(|&i| y_pred[i]).collect();

            let boot_result = metric_fn(&y_true_boot, &y_pred_boot);
            bootstrap_results.push(boot_result);
        }

        // Calculate statistics
        let mean = bootstrap_results.iter().sum::<f64>() / bootstrap_results.len() as f64;
        let variance = bootstrap_results
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (bootstrap_results.len() - 1) as f64;
        let std_dev = variance.sqrt();

        // Calculate confidence interval
        bootstrap_results.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ci_lower_idx = (0.025 * bootstrap_results.len() as f64) as usize;
        let ci_upper_idx = (0.975 * bootstrap_results.len() as f64) as usize;
        let confidence_interval = (
            bootstrap_results[ci_lower_idx],
            bootstrap_results[ci_upper_idx],
        );

        StabilityAnalysis {
            original_value: original_result,
            bootstrap_mean: mean,
            bootstrap_std: std_dev,
            confidence_interval,
            n_bootstrap_samples: bootstrap_results.len(),
        }
    }

    /// Perform metameric analysis to understand metric behavior
    pub fn metameric_analysis<F>(
        &mut self,
        metric_fn: F,
        parameter_ranges: Vec<(&str, f64, f64, usize)>,
    ) -> MetamericAnalysis
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64,
    {
        let mut analysis = MetamericAnalysis::new();

        // Test metric behavior across different parameter values
        for (param_name, min_val, max_val, n_points) in parameter_ranges {
            let mut param_results = Vec::new();

            for i in 0..n_points {
                let param_value = min_val + (max_val - min_val) * i as f64 / (n_points - 1) as f64;

                // Generate data based on parameter
                let (y_true, y_pred) = match param_name {
                    "r2" => self
                        .data_generator
                        .generate_regression_with_r2(100, param_value.clamp(0.01, 0.99)),
                    "noise" => self
                        .data_generator
                        .generate_noisy_predictions(100, param_value),
                    "outlier_fraction" => self.data_generator.generate_with_outliers(
                        100,
                        param_value.clamp(0.0, 1.0),
                        5.0,
                    ),
                    _ => self.data_generator.generate_noisy_predictions(100, 0.5),
                };

                let metric_value = metric_fn(&y_true, &y_pred);
                param_results.push((param_value, metric_value));
            }

            analysis
                .parameter_sensitivity
                .insert(param_name.to_string(), param_results);
        }

        analysis
    }

    /// Cross-validation for metric stability
    pub fn cross_validation_stability<F>(
        &mut self,
        metric_fn: F,
        data_size: usize,
    ) -> CrossValidationAnalysis
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> f64,
    {
        let (y_true, y_pred) = self
            .data_generator
            .generate_noisy_predictions(data_size, 0.5);
        let fold_size = data_size / self.config.cv_folds;
        let mut fold_results = Vec::new();

        for fold in 0..self.config.cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.config.cv_folds - 1 {
                data_size
            } else {
                (fold + 1) * fold_size
            };

            // Create fold data
            let fold_y_true = y_true.slice(s![start_idx..end_idx]).to_owned();
            let fold_y_pred = y_pred.slice(s![start_idx..end_idx]).to_owned();

            let fold_result = metric_fn(&fold_y_true, &fold_y_pred);
            fold_results.push(fold_result);
        }

        let mean = fold_results.iter().sum::<f64>() / fold_results.len() as f64;
        let variance = fold_results.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (fold_results.len() - 1) as f64;

        CrossValidationAnalysis {
            fold_results,
            mean_score: mean,
            std_deviation: variance.sqrt(),
            coefficient_of_variation: variance.sqrt() / mean.abs(),
        }
    }
}

impl Default for MetricValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive validation report
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationReport {
    pub metric_name: String,
    pub reference_validation: HashMap<String, ValidationResult>,
    pub perfect_prediction_result: Option<f64>,
    pub edge_case_results: HashMap<String, f64>,
    pub stability_analysis: Option<StabilityAnalysis>,
    pub noise_sensitivity: HashMap<String, f64>,
    pub overall_passed: bool,
}

impl ComprehensiveValidationReport {
    fn new(metric_name: String) -> Self {
        Self {
            metric_name,
            reference_validation: HashMap::new(),
            perfect_prediction_result: None,
            edge_case_results: HashMap::new(),
            stability_analysis: None,
            noise_sensitivity: HashMap::new(),
            overall_passed: false,
        }
    }

    /// Check if all validations passed
    pub fn update_overall_status(&mut self) {
        self.overall_passed = self.reference_validation.values().all(|r| r.passed);
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut summary = format!("Validation Report for {}\n", self.metric_name);
        summary.push_str("=".repeat(50).as_str());
        summary.push('\n');

        // Reference validation summary
        let total_ref_cases = self.reference_validation.len();
        let passed_ref_cases = self
            .reference_validation
            .values()
            .filter(|r| r.passed)
            .count();
        summary.push_str(&format!(
            "Reference Cases: {}/{} passed\n",
            passed_ref_cases, total_ref_cases
        ));

        // Edge cases summary
        if !self.edge_case_results.is_empty() {
            summary.push_str(&format!(
                "Edge Cases Tested: {}\n",
                self.edge_case_results.len()
            ));
        }

        // Stability analysis summary
        if let Some(ref stability) = self.stability_analysis {
            summary.push_str(&format!(
                "Stability: μ={:.6}, σ={:.6}, CI=[{:.6}, {:.6}]\n",
                stability.bootstrap_mean,
                stability.bootstrap_std,
                stability.confidence_interval.0,
                stability.confidence_interval.1
            ));
        }

        summary.push_str(&format!(
            "Overall Status: {}\n",
            if self.overall_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        ));

        summary
    }
}

/// Stability analysis results
#[derive(Debug, Clone)]
pub struct StabilityAnalysis {
    pub original_value: f64,
    pub bootstrap_mean: f64,
    pub bootstrap_std: f64,
    pub confidence_interval: (f64, f64),
    pub n_bootstrap_samples: usize,
}

/// Metameric analysis results
#[derive(Debug, Clone)]
pub struct MetamericAnalysis {
    pub parameter_sensitivity: HashMap<String, Vec<(f64, f64)>>,
}

impl MetamericAnalysis {
    fn new() -> Self {
        Self {
            parameter_sensitivity: HashMap::new(),
        }
    }
}

/// Cross-validation analysis results
#[derive(Debug, Clone)]
pub struct CrossValidationAnalysis {
    pub fold_results: Vec<f64>,
    pub mean_score: f64,
    pub std_deviation: f64,
    pub coefficient_of_variation: f64,
}

/// Standard reference datasets and expected results
pub struct StandardReferenceDatasets;

impl StandardReferenceDatasets {
    /// Create standard accuracy test cases
    pub fn accuracy_test_cases() -> Vec<ReferenceTestCase> {
        vec![
            ReferenceTestCase {
                name: "perfect_accuracy".to_string(),
                y_true: Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]),
                y_pred: Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]),
                expected_value: 1.0,
                tolerance: 1e-10,
                description: "Perfect classification should give accuracy = 1.0".to_string(),
            },
            ReferenceTestCase {
                name: "half_accuracy".to_string(),
                y_true: Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]),
                y_pred: Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]),
                expected_value: 0.5,
                tolerance: 1e-10,
                description: "Half correct predictions should give accuracy = 0.5".to_string(),
            },
            ReferenceTestCase {
                name: "zero_accuracy".to_string(),
                y_true: Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]),
                y_pred: Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0]),
                expected_value: 0.0,
                tolerance: 1e-10,
                description: "All wrong predictions should give accuracy = 0.0".to_string(),
            },
        ]
    }

    /// Create standard MSE test cases
    pub fn mse_test_cases() -> Vec<ReferenceTestCase> {
        vec![
            ReferenceTestCase {
                name: "perfect_mse".to_string(),
                y_true: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                y_pred: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                expected_value: 0.0,
                tolerance: 1e-10,
                description: "Perfect predictions should give MSE = 0.0".to_string(),
            },
            ReferenceTestCase {
                name: "unit_error_mse".to_string(),
                y_true: Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
                y_pred: Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]),
                expected_value: 1.0,
                tolerance: 1e-10,
                description: "Unit errors should give MSE = 1.0".to_string(),
            },
            ReferenceTestCase {
                name: "mixed_errors_mse".to_string(),
                y_true: Array1::from_vec(vec![1.0, 2.0, 3.0]),
                y_pred: Array1::from_vec(vec![1.0, 4.0, 1.0]),
                expected_value: 8.0 / 3.0,
                tolerance: 1e-10,
                description: "Mixed errors: (0² + 2² + 2²) / 3 = 8/3".to_string(),
            },
        ]
    }

    /// Create standard R² test cases
    pub fn r2_test_cases() -> Vec<ReferenceTestCase> {
        vec![
            ReferenceTestCase {
                name: "perfect_r2".to_string(),
                y_true: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                y_pred: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                expected_value: 1.0,
                tolerance: 1e-10,
                description: "Perfect predictions should give R² = 1.0".to_string(),
            },
            ReferenceTestCase {
                name: "mean_prediction_r2".to_string(),
                y_true: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                y_pred: Array1::from_vec(vec![2.5, 2.5, 2.5, 2.5]),
                expected_value: 0.0,
                tolerance: 1e-10,
                description: "Predicting mean should give R² = 0.0".to_string(),
            },
        ]
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::classification::accuracy_score;
    use crate::regression::{mean_squared_error, r2_score};
    use approx::assert_relative_eq;

    #[test]
    fn test_validation_result() {
        let result = ValidationResult::new(1.0, 0.99, 0.1);
        assert!(result.passed);
        assert_eq!(result.expected, 1.0);
        assert_eq!(result.actual, 0.99);
        assert_relative_eq!(result.absolute_error, 0.01, epsilon = 1e-10);

        let result_fail = ValidationResult::new(1.0, 0.8, 0.1);
        assert!(!result_fail.passed);
    }

    #[test]
    fn test_synthetic_data_generator() {
        let config = ValidationConfig::default();
        let mut generator = SyntheticDataGenerator::new(config);

        // Test perfect predictions
        let (y_true, y_pred) = generator.generate_perfect_predictions(100);
        assert_eq!(y_true.len(), 100);
        assert_eq!(y_pred.len(), 100);
        for i in 0..100 {
            assert_eq!(y_true[i], y_pred[i]);
        }

        // Test classification with specific accuracy
        let (y_true_cls, y_pred_cls) = generator.generate_classification_with_accuracy(100, 0.8);
        let actual_accuracy = accuracy_score(&y_true_cls, &y_pred_cls).unwrap();
        assert!((actual_accuracy - 0.8).abs() < 0.05); // Allow some tolerance
    }

    #[test]
    fn test_metric_validator() {
        let mut validator = MetricValidator::new();

        // Add reference cases
        for case in StandardReferenceDatasets::accuracy_test_cases() {
            validator.reference_cases.insert(case.name.clone(), case);
        }

        // Validate accuracy metric
        let result = validator
            .validate_metric("perfect_accuracy", |y_true, y_pred| {
                accuracy_score(y_true, y_pred).unwrap_or(0.0)
            })
            .unwrap();

        assert!(result.passed);
        assert_relative_eq!(result.expected, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.actual, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_comprehensive_validation() {
        let mut validator = MetricValidator::new();

        let report = validator.comprehensive_validation(
            |y_true, y_pred| mean_squared_error(y_true, y_pred).unwrap_or(f64::INFINITY),
            "mse",
        );

        assert_eq!(report.metric_name, "mse");
        assert!(report.perfect_prediction_result.is_some());
        assert!(!report.edge_case_results.is_empty());
    }

    #[test]
    fn test_standard_reference_datasets() {
        let accuracy_cases = StandardReferenceDatasets::accuracy_test_cases();
        assert_eq!(accuracy_cases.len(), 3);

        let mse_cases = StandardReferenceDatasets::mse_test_cases();
        assert_eq!(mse_cases.len(), 3);

        // Validate a specific case
        let perfect_case = &mse_cases[0];
        let computed_mse = mean_squared_error(&perfect_case.y_true, &perfect_case.y_pred).unwrap();
        assert_relative_eq!(
            computed_mse,
            perfect_case.expected_value,
            epsilon = perfect_case.tolerance
        );
    }

    #[test]
    fn test_edge_case_generation() {
        let config = ValidationConfig::default();
        let mut generator = SyntheticDataGenerator::new(config);

        let edge_cases = generator.generate_edge_cases();
        assert!(!edge_cases.is_empty());

        // Check specific edge cases
        let case_names: Vec<&String> = edge_cases.iter().map(|(name, _, _)| name).collect();
        assert!(case_names.contains(&&"single_element".to_string()));
        assert!(case_names.contains(&&"all_zeros".to_string()));
    }

    #[test]
    fn test_metameric_analysis() {
        let mut validator = MetricValidator::new();

        let parameter_ranges = vec![("r2", 0.1, 0.9, 5), ("noise", 0.1, 2.0, 4)];

        let analysis = validator.metameric_analysis(
            |y_true, y_pred| r2_score(y_true, y_pred).unwrap_or(0.0),
            parameter_ranges,
        );

        assert!(analysis.parameter_sensitivity.contains_key("r2"));
        assert!(analysis.parameter_sensitivity.contains_key("noise"));
    }

    #[test]
    fn test_cross_validation_stability() {
        let mut validator = MetricValidator::new();

        let cv_analysis = validator.cross_validation_stability(
            |y_true, y_pred| mean_squared_error(y_true, y_pred).unwrap_or(f64::INFINITY),
            500,
        );

        assert_eq!(cv_analysis.fold_results.len(), validator.config.cv_folds);
        assert!(cv_analysis.mean_score >= 0.0);
        assert!(cv_analysis.std_deviation >= 0.0);
    }

    #[test]
    fn test_outlier_generation() {
        let config = ValidationConfig::default();
        let mut generator = SyntheticDataGenerator::new(config);

        let (y_true, y_pred) = generator.generate_with_outliers(100, 0.1, 10.0);
        assert_eq!(y_true.len(), 100);
        assert_eq!(y_pred.len(), 100);

        // Check that some predictions are significantly different (outliers)
        let large_errors = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| (**t - **p).abs() > 5.0)
            .count();
        assert!(large_errors > 0); // Should have some outliers
    }

    #[test]
    fn test_validation_report_summary() {
        let mut report = ComprehensiveValidationReport::new("test_metric".to_string());

        // Add some validation results
        report
            .reference_validation
            .insert("test1".to_string(), ValidationResult::new(1.0, 1.0, 1e-10));
        report
            .reference_validation
            .insert("test2".to_string(), ValidationResult::new(0.5, 0.6, 0.05));

        report.update_overall_status();

        let summary = report.summary();
        assert!(summary.contains("test_metric"));
        assert!(summary.contains("Reference Cases: 1/2 passed"));
    }
}
