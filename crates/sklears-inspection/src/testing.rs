//! Comprehensive Testing Framework
//!
//! This module provides comprehensive testing capabilities including property-based tests,
//! fidelity tests, consistency tests, and robustness validation for explanation methods.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for property-based testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyTestConfig {
    /// Number of test cases to generate
    pub num_test_cases: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Tolerance for floating point comparisons
    pub tolerance: Float,
    /// Maximum number of features to test
    pub max_features: usize,
    /// Maximum number of samples to test
    pub max_samples: usize,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        Self {
            num_test_cases: 100,
            seed: Some(42),
            tolerance: 1e-6,
            max_features: 100,
            max_samples: 1000,
            verbose: false,
        }
    }
}

/// Test result for property-based tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyTestResult {
    /// Test name
    pub test_name: String,
    /// Number of test cases run
    pub cases_run: usize,
    /// Number of test cases passed
    pub cases_passed: usize,
    /// Test passed overall
    pub passed: bool,
    /// Failure messages if any
    pub failure_messages: Vec<String>,
    /// Property violations found
    pub violations: Vec<PropertyViolation>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

/// Property violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyViolation {
    /// Property name that was violated
    pub property: String,
    /// Description of the violation
    pub description: String,
    /// Input data that caused the violation
    pub input_data: String,
    /// Expected behavior
    pub expected: String,
    /// Actual behavior
    pub actual: String,
}

/// Fidelity test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityTestConfig {
    /// Minimum required fidelity score
    pub min_fidelity: Float,
    /// Number of samples for fidelity testing
    pub num_samples: usize,
    /// Perturbation magnitude for fidelity testing
    pub perturbation_magnitude: Float,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for FidelityTestConfig {
    fn default() -> Self {
        Self {
            min_fidelity: 0.8,
            num_samples: 100,
            perturbation_magnitude: 0.1,
            seed: Some(42),
        }
    }
}

/// Consistency test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyTestConfig {
    /// Methods to compare for consistency
    pub methods: Vec<String>,
    /// Tolerance for consistency checks
    pub tolerance: Float,
    /// Number of test cases
    pub num_test_cases: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for ConsistencyTestConfig {
    fn default() -> Self {
        Self {
            methods: vec!["permutation".to_string(), "shap".to_string()],
            tolerance: 0.2,
            num_test_cases: 50,
            seed: Some(42),
        }
    }
}

/// Robustness test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessTestConfig {
    /// Noise levels to test
    pub noise_levels: Vec<Float>,
    /// Number of noise perturbations per level
    pub perturbations_per_level: usize,
    /// Maximum acceptable explanation change
    pub max_explanation_change: Float,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for RobustnessTestConfig {
    fn default() -> Self {
        Self {
            noise_levels: vec![0.01, 0.05, 0.1, 0.2],
            perturbations_per_level: 10,
            max_explanation_change: 0.3,
            seed: Some(42),
        }
    }
}

/// Comprehensive testing suite
pub struct TestingSuite {
    /// Property test configuration
    property_config: PropertyTestConfig,
    /// Fidelity test configuration
    fidelity_config: FidelityTestConfig,
    /// Consistency test configuration
    consistency_config: ConsistencyTestConfig,
    /// Robustness test configuration
    robustness_config: RobustnessTestConfig,
}

impl TestingSuite {
    /// Create a new testing suite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_inspection::testing::TestingSuite;
    ///
    /// let suite = TestingSuite::new();
    /// ```
    pub fn new() -> Self {
        Self {
            property_config: PropertyTestConfig::default(),
            fidelity_config: FidelityTestConfig::default(),
            consistency_config: ConsistencyTestConfig::default(),
            robustness_config: RobustnessTestConfig::default(),
        }
    }

    /// Create a new testing suite with custom configurations
    pub fn with_configs(
        property_config: PropertyTestConfig,
        fidelity_config: FidelityTestConfig,
        consistency_config: ConsistencyTestConfig,
        robustness_config: RobustnessTestConfig,
    ) -> Self {
        Self {
            property_config,
            fidelity_config,
            consistency_config,
            robustness_config,
        }
    }

    /// Run property-based tests for explanation properties
    pub fn test_explanation_properties<F>(
        &self,
        explanation_fn: F,
        test_name: &str,
    ) -> SklResult<PropertyTestResult>
    where
        F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
    {
        let start_time = std::time::Instant::now();
        let mut cases_passed = 0;
        let mut violations = Vec::new();
        let mut failure_messages = Vec::new();

        // Set up random number generator
        let mut rng = if let Some(seed) = self.property_config.seed {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng())
        };

        for case_idx in 0..self.property_config.num_test_cases {
            // Generate random test data
            let test_data = self.generate_test_data(&mut rng)?;

            // Run explanation function
            match explanation_fn(&test_data.view()) {
                Ok(explanation) => {
                    // Test explanation properties
                    let property_results = self.check_explanation_properties(
                        &test_data.view(),
                        &explanation.view(),
                        case_idx,
                    );

                    if property_results.is_empty() {
                        cases_passed += 1;
                    } else {
                        violations.extend(property_results);
                    }
                }
                Err(e) => {
                    failure_messages.push(format!("Case {}: {}", case_idx, e));
                }
            }
        }

        let execution_time = start_time.elapsed().as_millis() as f64;
        let passed = violations.is_empty() && failure_messages.is_empty();

        Ok(PropertyTestResult {
            test_name: test_name.to_string(),
            cases_run: self.property_config.num_test_cases,
            cases_passed,
            passed,
            failure_messages,
            violations,
            execution_time_ms: execution_time,
        })
    }

    /// Test fidelity of local explanations
    pub fn test_explanation_fidelity<F, M>(
        &self,
        model_fn: M,
        explanation_fn: F,
        test_data: &ArrayView2<Float>,
    ) -> SklResult<Float>
    where
        M: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
        F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
    {
        use scirs2_core::random::Rng;

        let mut rng = if let Some(seed) = self.fidelity_config.seed {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng())
        };

        let mut total_fidelity = 0.0;
        let n_features = test_data.ncols();

        for i in 0..self.fidelity_config.num_samples.min(test_data.nrows()) {
            let instance = test_data.row(i);
            let instance_2d = instance.insert_axis(Axis(0));
            let original_prediction = model_fn(&instance_2d.view())?;
            let explanation = explanation_fn(&instance_2d.view())?;

            // Create perturbed instances based on explanation
            let mut correct_predictions = 0;
            let mut total_predictions = 0;

            for _ in 0..10 {
                let mut perturbed_instance = instance.to_owned();

                // Perturb features based on explanation importance
                for j in 0..n_features {
                    if rng.gen::<Float>() < self.fidelity_config.perturbation_magnitude {
                        let importance = explanation[j].abs();
                        let perturbation = rng.gen_range(-importance..importance);
                        perturbed_instance[j] += perturbation;
                    }
                }

                let perturbed_2d = perturbed_instance.view().insert_axis(Axis(0));
                let perturbed_prediction = model_fn(&perturbed_2d)?;

                // Check if explanation correctly predicts direction of change
                let prediction_change = perturbed_prediction[0] - original_prediction[0];
                let expected_change = explanation
                    .iter()
                    .zip(perturbed_instance.iter().zip(instance.iter()))
                    .map(|(imp, (new_val, old_val))| imp * (new_val - old_val))
                    .sum::<Float>();

                if prediction_change.signum() == expected_change.signum() {
                    correct_predictions += 1;
                }
                total_predictions += 1;
            }

            total_fidelity += correct_predictions as Float / total_predictions as Float;
        }

        Ok(total_fidelity / self.fidelity_config.num_samples.min(test_data.nrows()) as Float)
    }

    /// Test consistency across different explanation methods
    pub fn test_method_consistency<F1, F2>(
        &self,
        method1: F1,
        method2: F2,
        test_data: &ArrayView2<Float>,
        method1_name: &str,
        method2_name: &str,
    ) -> SklResult<Float>
    where
        F1: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
        F2: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
    {
        let mut total_correlation = 0.0;
        let mut valid_comparisons = 0;

        for i in 0..self
            .consistency_config
            .num_test_cases
            .min(test_data.nrows())
        {
            let instance = test_data.row(i).insert_axis(Axis(0));

            let explanation1 = method1(&instance.view())?;
            let explanation2 = method2(&instance.view())?;

            // Calculate correlation between explanations
            let correlation =
                self.calculate_correlation(&explanation1.view(), &explanation2.view());

            if !correlation.is_nan() {
                total_correlation += correlation;
                valid_comparisons += 1;
            }
        }

        if valid_comparisons > 0 {
            Ok(total_correlation / valid_comparisons as Float)
        } else {
            Ok(0.0)
        }
    }

    /// Test robustness to input noise
    pub fn test_robustness<F>(
        &self,
        explanation_fn: F,
        test_data: &ArrayView2<Float>,
    ) -> SklResult<HashMap<String, Float>>
    where
        F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
    {
        use scirs2_core::random::Rng;

        let mut rng = if let Some(seed) = self.robustness_config.seed {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng())
        };

        let mut results = HashMap::new();

        for &noise_level in &self.robustness_config.noise_levels {
            let mut total_stability = 0.0;
            let mut valid_tests = 0;

            for i in 0..test_data.nrows() {
                let original_instance = test_data.row(i);
                let original_2d = original_instance.insert_axis(Axis(0));
                let original_explanation = explanation_fn(&original_2d.view())?;

                let mut perturbation_stabilities = Vec::new();

                for _ in 0..self.robustness_config.perturbations_per_level {
                    // Add noise to the instance
                    let mut noisy_instance = original_instance.to_owned();
                    for j in 0..noisy_instance.len() {
                        let noise = rng.gen_range(-noise_level..noise_level);
                        noisy_instance[j] += noise;
                    }

                    let noisy_2d = noisy_instance.view().insert_axis(Axis(0));
                    let noisy_explanation = explanation_fn(&noisy_2d)?;

                    // Calculate stability (1 - relative change)
                    let stability = self.calculate_explanation_stability(
                        &original_explanation.view(),
                        &noisy_explanation.view(),
                    );

                    if !stability.is_nan() {
                        perturbation_stabilities.push(stability);
                    }
                }

                if !perturbation_stabilities.is_empty() {
                    let avg_stability = perturbation_stabilities.iter().sum::<Float>()
                        / perturbation_stabilities.len() as Float;
                    total_stability += avg_stability;
                    valid_tests += 1;
                }
            }

            if valid_tests > 0 {
                results.insert(
                    format!("noise_{:.3}", noise_level),
                    total_stability / valid_tests as Float,
                );
            }
        }

        Ok(results)
    }

    /// Run comprehensive test suite
    pub fn run_comprehensive_tests<F, M>(
        &self,
        model_fn: M,
        explanation_fn: F,
        test_data: &ArrayView2<Float>,
        test_name: &str,
    ) -> SklResult<HashMap<String, serde_json::Value>>
    where
        F: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>> + Clone,
        M: Fn(&ArrayView2<Float>) -> SklResult<Array1<Float>>,
    {
        let mut results = HashMap::new();

        // Property-based tests
        let property_result =
            self.test_explanation_properties(explanation_fn.clone(), test_name)?;
        results.insert(
            "property_tests".to_string(),
            serde_json::to_value(property_result).map_err(|e| {
                crate::SklearsError::InvalidInput(format!(
                    "Failed to serialize property test results: {}",
                    e
                ))
            })?,
        );

        // Robustness tests
        let robustness_result = self.test_robustness(explanation_fn, test_data)?;
        results.insert(
            "robustness_tests".to_string(),
            serde_json::to_value(robustness_result).map_err(|e| {
                crate::SklearsError::InvalidInput(format!(
                    "Failed to serialize robustness test results: {}",
                    e
                ))
            })?,
        );

        Ok(results)
    }

    // Helper methods
    fn generate_test_data<R: scirs2_core::random::Rng>(
        &self,
        rng: &mut R,
    ) -> SklResult<Array2<Float>> {
        use scirs2_core::random::Rng;

        let n_samples = rng.gen_range(10..=self.property_config.max_samples.min(100));
        let n_features = rng.gen_range(5..=self.property_config.max_features.min(20));

        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(-2.0..2.0);
            }
        }

        Ok(data)
    }

    fn check_explanation_properties(
        &self,
        _data: &ArrayView2<Float>,
        explanation: &ArrayView1<Float>,
        case_idx: usize,
    ) -> Vec<PropertyViolation> {
        let mut violations = Vec::new();

        // Property 1: Explanation should not contain NaN or infinite values
        for (i, &value) in explanation.iter().enumerate() {
            if value.is_nan() {
                violations.push(PropertyViolation {
                    property: "no_nan_values".to_string(),
                    description: format!("Explanation contains NaN value at index {}", i),
                    input_data: format!("Case {}", case_idx),
                    expected: "Finite numeric value".to_string(),
                    actual: "NaN".to_string(),
                });
            }
            if value.is_infinite() {
                violations.push(PropertyViolation {
                    property: "no_infinite_values".to_string(),
                    description: format!("Explanation contains infinite value at index {}", i),
                    input_data: format!("Case {}", case_idx),
                    expected: "Finite numeric value".to_string(),
                    actual: "Infinite".to_string(),
                });
            }
        }

        // Property 2: Explanation should not be all zeros (unless intended)
        let sum_abs = explanation.iter().map(|x| x.abs()).sum::<Float>();
        if sum_abs < self.property_config.tolerance {
            violations.push(PropertyViolation {
                property: "non_trivial_explanation".to_string(),
                description: "Explanation is all zeros or nearly zeros".to_string(),
                input_data: format!("Case {}", case_idx),
                expected: "Non-zero explanation values".to_string(),
                actual: format!("Sum of absolute values: {}", sum_abs),
            });
        }

        violations
    }

    fn calculate_correlation(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        if a.len() != b.len() {
            return Float::NAN;
        }

        let n = a.len() as Float;
        let mean_a = a.iter().sum::<Float>() / n;
        let mean_b = b.iter().sum::<Float>() / n;

        let numerator: Float = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - mean_a) * (bi - mean_b))
            .sum();

        let sum_sq_a: Float = a.iter().map(|ai| (ai - mean_a).powi(2)).sum();
        let sum_sq_b: Float = b.iter().map(|bi| (bi - mean_b).powi(2)).sum();

        let denominator = (sum_sq_a * sum_sq_b).sqrt();

        if denominator == 0.0 {
            Float::NAN
        } else {
            numerator / denominator
        }
    }

    fn calculate_explanation_stability(
        &self,
        original: &ArrayView1<Float>,
        perturbed: &ArrayView1<Float>,
    ) -> Float {
        if original.len() != perturbed.len() {
            return Float::NAN;
        }

        let relative_changes: Vec<Float> = original
            .iter()
            .zip(perturbed.iter())
            .map(|(orig, pert)| {
                if orig.abs() < self.property_config.tolerance {
                    pert.abs()
                } else {
                    (pert - orig).abs() / orig.abs()
                }
            })
            .collect();

        let max_relative_change = relative_changes.iter().copied().fold(0.0f64, f64::max);

        // Stability is 1 - normalized change (clamped to [0, 1])
        (1.0 - max_relative_change).max(0.0).min(1.0)
    }
}

impl Default for TestingSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate explanation output against expected properties
pub fn validate_explanation_output(
    explanation: &ArrayView1<Float>,
    expected_properties: &ExplanationProperties,
) -> SklResult<ValidationResult> {
    let mut violations = Vec::new();
    let mut passed_checks = 0;
    let total_checks = 5; // Adjust based on number of checks

    // Check for finite values
    let has_finite_values = explanation.iter().all(|x| x.is_finite());
    if has_finite_values {
        passed_checks += 1;
    } else {
        violations.push("Explanation contains non-finite values".to_string());
    }

    // Check sum constraint if specified
    if let Some(expected_sum) = expected_properties.expected_sum {
        let actual_sum = explanation.sum();
        if (actual_sum - expected_sum).abs() < expected_properties.tolerance {
            passed_checks += 1;
        } else {
            violations.push(format!(
                "Sum constraint violated: expected {}, got {}",
                expected_sum, actual_sum
            ));
        }
    } else {
        passed_checks += 1; // Skip this check
    }

    // Check non-negativity if required
    if expected_properties.non_negative {
        let is_non_negative = explanation.iter().all(|x| *x >= 0.0);
        if is_non_negative {
            passed_checks += 1;
        } else {
            violations.push(
                "Explanation contains negative values when non-negativity is required".to_string(),
            );
        }
    } else {
        passed_checks += 1; // Skip this check
    }

    // Check magnitude bounds
    let max_magnitude = explanation.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if max_magnitude <= expected_properties.max_magnitude {
        passed_checks += 1;
    } else {
        violations.push(format!(
            "Magnitude bound violated: max magnitude {} exceeds limit {}",
            max_magnitude, expected_properties.max_magnitude
        ));
    }

    // Check sparsity if required
    if let Some(max_non_zero) = expected_properties.max_non_zero_features {
        let non_zero_count = explanation
            .iter()
            .filter(|x| x.abs() > expected_properties.tolerance)
            .count();
        if non_zero_count <= max_non_zero {
            passed_checks += 1;
        } else {
            violations.push(format!(
                "Sparsity constraint violated: {} non-zero features exceeds limit {}",
                non_zero_count, max_non_zero
            ));
        }
    } else {
        passed_checks += 1; // Skip this check
    }

    Ok(ValidationResult {
        passed: violations.is_empty(),
        passed_checks,
        total_checks,
        violations,
        score: passed_checks as Float / total_checks as Float,
    })
}

/// Expected properties for explanation validation
#[derive(Debug, Clone)]
pub struct ExplanationProperties {
    /// Expected sum of explanation values
    pub expected_sum: Option<Float>,
    /// Whether values should be non-negative
    pub non_negative: bool,
    /// Maximum allowed magnitude
    pub max_magnitude: Float,
    /// Maximum number of non-zero features
    pub max_non_zero_features: Option<usize>,
    /// Tolerance for floating point comparisons
    pub tolerance: Float,
}

impl Default for ExplanationProperties {
    fn default() -> Self {
        Self {
            expected_sum: None,
            non_negative: false,
            max_magnitude: 10.0,
            max_non_zero_features: None,
            tolerance: 1e-6,
        }
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all checks passed
    pub passed: bool,
    /// Number of checks that passed
    pub passed_checks: usize,
    /// Total number of checks
    pub total_checks: usize,
    /// Violation messages
    pub violations: Vec<String>,
    /// Overall validation score (0-1)
    pub score: Float,
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_testing_suite_creation() {
        let suite = TestingSuite::new();
        assert_eq!(suite.property_config.num_test_cases, 100);
        assert_eq!(suite.property_config.seed, Some(42));
    }

    #[test]
    fn test_property_test_config_default() {
        let config = PropertyTestConfig::default();
        assert_eq!(config.num_test_cases, 100);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.max_features, 100);
    }

    #[test]
    fn test_explanation_property_validation() {
        let explanation = array![0.3, 0.5, -0.2, 0.1];
        let properties = ExplanationProperties::default();

        let result = validate_explanation_output(&explanation.view(), &properties).unwrap();
        assert!(result.passed);
        assert!(result.score > 0.8);
    }

    #[test]
    fn test_explanation_with_nan_validation() {
        let explanation = array![0.3, Float::NAN, -0.2, 0.1];
        let properties = ExplanationProperties::default();

        let result = validate_explanation_output(&explanation.view(), &properties).unwrap();
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_non_negative_constraint() {
        let explanation = array![0.3, 0.5, -0.2, 0.1];
        let properties = ExplanationProperties {
            non_negative: true,
            ..Default::default()
        };

        let result = validate_explanation_output(&explanation.view(), &properties).unwrap();
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.contains("negative values")));
    }

    #[test]
    fn test_sum_constraint() {
        let explanation = array![0.3, 0.5, 0.2, 0.0];
        let properties = ExplanationProperties {
            expected_sum: Some(1.0),
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = validate_explanation_output(&explanation.view(), &properties).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_magnitude_constraint() {
        let explanation = array![0.3, 15.0, 0.2, 0.1]; // 15.0 exceeds default max_magnitude of 10.0
        let properties = ExplanationProperties::default();

        let result = validate_explanation_output(&explanation.view(), &properties).unwrap();
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.contains("Magnitude bound violated")));
    }

    #[test]
    fn test_sparsity_constraint() {
        let explanation = array![0.3, 0.5, 0.2, 0.1, 0.05];
        let properties = ExplanationProperties {
            max_non_zero_features: Some(3),
            tolerance: 1e-6,
            ..Default::default()
        };

        let result = validate_explanation_output(&explanation.view(), &properties).unwrap();
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.contains("Sparsity constraint violated")));
    }

    #[test]
    fn test_correlation_calculation() {
        let suite = TestingSuite::new();
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![2.0, 4.0, 6.0, 8.0]; // Perfect positive correlation

        let correlation = suite.calculate_correlation(&a.view(), &b.view());
        assert!((correlation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_explanation_stability() {
        let suite = TestingSuite::new();
        let original = array![0.3, 0.5, 0.2];
        let similar = array![0.31, 0.49, 0.21]; // Small changes

        let stability = suite.calculate_explanation_stability(&original.view(), &similar.view());
        assert!(stability > 0.8); // Should be quite stable
    }

    #[test]
    fn test_generate_test_data() {
        let config = PropertyTestConfig {
            max_samples: 50,
            max_features: 10,
            seed: Some(42),
            ..Default::default()
        };
        let suite = TestingSuite {
            property_config: config,
            ..Default::default()
        };

        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let data = suite.generate_test_data(&mut rng).unwrap();

        assert!(data.nrows() >= 10 && data.nrows() <= 50);
        assert!(data.ncols() >= 5 && data.ncols() <= 10);
    }

    #[test]
    fn test_property_violation_creation() {
        let violation = PropertyViolation {
            property: "test_property".to_string(),
            description: "Test violation".to_string(),
            input_data: "test_data".to_string(),
            expected: "expected_behavior".to_string(),
            actual: "actual_behavior".to_string(),
        };

        assert_eq!(violation.property, "test_property");
        assert_eq!(violation.description, "Test violation");
    }

    #[test]
    fn test_fidelity_config_default() {
        let config = FidelityTestConfig::default();
        assert_eq!(config.min_fidelity, 0.8);
        assert_eq!(config.num_samples, 100);
        assert_eq!(config.perturbation_magnitude, 0.1);
    }

    #[test]
    fn test_consistency_config_default() {
        let config = ConsistencyTestConfig::default();
        assert_eq!(config.methods.len(), 2);
        assert_eq!(config.tolerance, 0.2);
        assert_eq!(config.num_test_cases, 50);
    }

    #[test]
    fn test_robustness_config_default() {
        let config = RobustnessTestConfig::default();
        assert_eq!(config.noise_levels.len(), 4);
        assert_eq!(config.perturbations_per_level, 10);
        assert_eq!(config.max_explanation_change, 0.3);
    }
}
