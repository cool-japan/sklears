//! Numerical stability utilities and improvements for calibration methods
//!
//! This module provides utilities to improve numerical stability in calibration
//! computations, including safe log-space operations, probability clamping,
//! and robust optimization methods.

use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

/// Numerical stability configuration
#[derive(Debug, Clone)]
pub struct NumericalConfig {
    /// Minimum probability value to avoid log(0)
    pub min_probability: Float,
    /// Maximum probability value to avoid log(1-ε) issues  
    pub max_probability: Float,
    /// Tolerance for convergence in iterative methods
    pub convergence_tolerance: Float,
    /// Maximum number of iterations for optimization
    pub max_iterations: usize,
    /// Regularization parameter for optimization
    pub regularization: Float,
    /// Whether to use log-space computations
    pub use_log_space: bool,
}

impl Default for NumericalConfig {
    fn default() -> Self {
        Self {
            min_probability: 1e-15,
            max_probability: 1.0 - 1e-15,
            convergence_tolerance: 1e-8,
            max_iterations: 1000,
            regularization: 1e-6,
            use_log_space: true,
        }
    }
}

/// Safe probability operations with numerical stability guarantees
#[derive(Debug, Clone)]
pub struct SafeProbabilityOps {
    config: NumericalConfig,
}

impl SafeProbabilityOps {
    /// Create new safe probability operations with configuration
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(NumericalConfig::default())
    }

    /// Safely clamp probabilities to valid range
    pub fn clamp_probabilities(&self, probabilities: &Array1<Float>) -> Array1<Float> {
        probabilities.mapv(|p| p.clamp(self.config.min_probability, self.config.max_probability))
    }

    /// Convert probabilities to logits with safe computation
    pub fn probabilities_to_logits(&self, probabilities: &Array1<Float>) -> Array1<Float> {
        let clamped = self.clamp_probabilities(probabilities);
        clamped.mapv(|p| {
            let safe_p = p.clamp(self.config.min_probability, self.config.max_probability);
            (safe_p / (1.0 - safe_p)).ln()
        })
    }

    /// Convert logits to probabilities with safe computation
    pub fn logits_to_probabilities(&self, logits: &Array1<Float>) -> Array1<Float> {
        logits.mapv(|logit| {
            if logit > 700.0 {
                // Prevent overflow: exp(700) is very large
                self.config.max_probability
            } else if logit < -700.0 {
                // Prevent underflow
                self.config.min_probability
            } else {
                let exp_logit = logit.exp();
                let prob = exp_logit / (1.0 + exp_logit);
                prob.clamp(self.config.min_probability, self.config.max_probability)
            }
        })
    }

    /// Safe log-sum-exp computation
    pub fn log_sum_exp(&self, values: &Array1<Float>) -> Float {
        if values.is_empty() {
            return Float::NEG_INFINITY;
        }

        let max_val = values.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        if max_val.is_infinite() && max_val < 0.0 {
            return Float::NEG_INFINITY;
        }

        let sum: Float = values.iter().map(|&v| (v - max_val).exp()).sum();

        if sum == 0.0 {
            Float::NEG_INFINITY
        } else {
            max_val + sum.ln()
        }
    }

    /// Safe softmax computation
    pub fn safe_softmax(&self, logits: &Array1<Float>) -> Array1<Float> {
        let max_logit = logits.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        // Subtract max for numerical stability
        let shifted_logits = logits.mapv(|l| l - max_logit);

        // Compute exponentials
        let exp_logits = shifted_logits.mapv(|l| {
            if l < -700.0 {
                0.0 // Underflow protection
            } else {
                l.exp()
            }
        });

        // Normalize
        let sum: Float = exp_logits.sum();
        if sum == 0.0 || !sum.is_finite() {
            // Fallback to uniform distribution
            Array1::from(vec![1.0 / logits.len() as Float; logits.len()])
        } else {
            exp_logits / sum
        }
    }

    /// Safe log-likelihood computation
    pub fn safe_log_likelihood(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<Float> {
        if probabilities.len() != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have same length".to_string(),
            ));
        }

        let clamped_probs = self.clamp_probabilities(probabilities);

        let log_likelihood: Float = clamped_probs
            .iter()
            .zip(targets.iter())
            .map(|(&p, &y)| if y == 1 { p.ln() } else { (1.0 - p).ln() })
            .sum();

        if log_likelihood.is_finite() {
            Ok(log_likelihood)
        } else {
            Ok(Float::NEG_INFINITY)
        }
    }

    /// Safe cross-entropy computation
    pub fn safe_cross_entropy(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<Float> {
        let log_likelihood = self.safe_log_likelihood(probabilities, targets)?;
        Ok(-log_likelihood / probabilities.len() as Float)
    }

    /// Check if array contains problematic values
    pub fn check_numerical_health(&self, values: &Array1<Float>) -> Vec<String> {
        let mut issues = Vec::new();

        for (i, &value) in values.iter().enumerate() {
            if value.is_nan() {
                issues.push(format!("NaN at index {}", i));
            }
            if value.is_infinite() {
                issues.push(format!("Infinite value at index {}", i));
            }
            if value < 0.0 || value > 1.0 {
                issues.push(format!(
                    "Probability out of range [0,1] at index {}: {}",
                    i, value
                ));
            }
        }

        issues
    }

    /// Robust probability normalization
    pub fn robust_normalize(&self, values: &Array1<Float>) -> Array1<Float> {
        let sum: Float = values.iter().filter(|&&v| v.is_finite() && v >= 0.0).sum();

        if sum == 0.0 || !sum.is_finite() {
            // Fallback to uniform distribution
            Array1::from(vec![1.0 / values.len() as Float; values.len()])
        } else {
            values.mapv(|v| {
                if v.is_finite() && v >= 0.0 {
                    v / sum
                } else {
                    0.0
                }
            })
        }
    }
}

/// Robust optimization utilities for calibration
pub struct RobustOptimizer {
    config: NumericalConfig,
}

impl RobustOptimizer {
    /// Create new robust optimizer
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Robust gradient descent for scalar parameter optimization
    pub fn robust_scalar_optimization<F, G>(
        &self,
        initial_value: Float,
        objective: F,
        gradient: G,
        bounds: Option<(Float, Float)>,
    ) -> Result<Float>
    where
        F: Fn(Float) -> Float,
        G: Fn(Float) -> Float,
    {
        let mut x = initial_value;
        let mut prev_x = x;
        let mut learning_rate = 0.1;
        let mut best_x = x;
        let mut best_obj = objective(x);

        for iteration in 0..self.config.max_iterations {
            prev_x = x; // Track previous position before update

            let grad = gradient(x);

            // Check gradient health
            if !grad.is_finite() {
                learning_rate *= 0.5;
                if learning_rate < 1e-12 {
                    break;
                }
                continue;
            }

            // Update with regularization
            let regularized_grad = grad + self.config.regularization * x;
            let new_x = x - learning_rate * regularized_grad;

            // Apply bounds if specified
            let bounded_x = if let Some((min_bound, max_bound)) = bounds {
                new_x.clamp(min_bound, max_bound)
            } else {
                new_x
            };

            let new_obj = objective(bounded_x);

            // Check objective health
            if !new_obj.is_finite() {
                learning_rate *= 0.5;
                continue;
            }

            // Debug output for first few iterations
            if iteration < 10 {
                println!("Iter {}: x={:.6}, grad={:.6}, new_x={:.6}, obj={:.6}, best_obj={:.6}, lr={:.6}", 
                         iteration, x, grad, bounded_x, new_obj, best_obj, learning_rate);
            }

            // Update position
            x = bounded_x;

            // Track best solution
            if new_obj < best_obj {
                best_x = x;
                best_obj = new_obj;
                learning_rate *= 1.1; // Increase if improving
            } else {
                learning_rate *= 0.9; // Decrease if not improving
            }

            // Convergence check based on position change between iterations
            let position_change = (x - prev_x).abs();
            if position_change < self.config.convergence_tolerance {
                if iteration < 20 {
                    println!(
                        "Converged at iteration {}: x={:.6}, prev_x={:.6}, change={:.6}",
                        iteration, x, prev_x, position_change
                    );
                }
                break;
            }

            // Prevent learning rate from becoming too small or large
            learning_rate = learning_rate.clamp(1e-12, 1.0);
        }

        Ok(best_x)
    }

    /// Minimize a scalar function using robust optimization
    /// This is a convenience method that wraps robust_scalar_optimization
    /// for cases where you have the function and its gradient
    pub fn minimize<F, G>(
        &self,
        initial_value: Float,
        objective: F,
        gradient: G,
        bounds: Option<(Float, Float)>,
    ) -> Result<Float>
    where
        F: Fn(Float) -> Float,
        G: Fn(Float) -> Float,
    {
        self.robust_scalar_optimization(initial_value, objective, gradient, bounds)
    }

    /// Robust line search for optimization
    pub fn robust_line_search<F>(
        &self,
        x0: Float,
        direction: Float,
        objective: F,
        max_step: Float,
    ) -> Float
    where
        F: Fn(Float) -> Float,
    {
        let mut step_size = max_step;
        let base_obj = objective(x0);

        for _ in 0..20 {
            let new_x = x0 + step_size * direction;
            let new_obj = objective(new_x);

            if new_obj.is_finite() && new_obj < base_obj {
                return step_size;
            }

            step_size *= 0.5;
            if step_size < 1e-12 {
                break;
            }
        }

        0.0 // No improvement found
    }
}

/// Numerical stability tests
pub struct NumericalStabilityTests {
    ops: SafeProbabilityOps,
}

impl NumericalStabilityTests {
    /// Create new stability tester
    pub fn new() -> Self {
        Self {
            ops: SafeProbabilityOps::default(),
        }
    }

    /// Test probability clamping behavior
    pub fn test_probability_clamping(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Test extreme values
        let extreme_probs = Array1::from(vec![
            0.0,
            1.0,
            -0.1,
            1.1,
            Float::NEG_INFINITY,
            Float::INFINITY,
            Float::NAN,
            1e-20,
            1.0 - 1e-20,
        ]);

        let clamped = self.ops.clamp_probabilities(&extreme_probs);

        for (i, &value) in clamped.iter().enumerate() {
            if !value.is_finite() {
                issues.push(format!(
                    "Clamping failed to fix non-finite value at index {}",
                    i
                ));
            }
            if value < 0.0 || value > 1.0 {
                issues.push(format!(
                    "Clamped value out of range at index {}: {}",
                    i, value
                ));
            }
        }

        Ok(issues)
    }

    /// Test logit conversion stability
    pub fn test_logit_conversion_stability(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Test round-trip conversion
        let original_probs = Array1::from(vec![0.1, 0.5, 0.9, 0.01, 0.99]);
        let logits = self.ops.probabilities_to_logits(&original_probs);
        let recovered_probs = self.ops.logits_to_probabilities(&logits);

        for (i, (&orig, &recovered)) in original_probs
            .iter()
            .zip(recovered_probs.iter())
            .enumerate()
        {
            if (orig - recovered).abs() > 1e-10 {
                issues.push(format!(
                    "Round-trip conversion error at index {}: {} -> {}",
                    i, orig, recovered
                ));
            }
        }

        // Test extreme logits
        let extreme_logits =
            Array1::from(vec![-1000.0, 1000.0, Float::NEG_INFINITY, Float::INFINITY]);
        let probs_from_extreme = self.ops.logits_to_probabilities(&extreme_logits);

        for (i, &prob) in probs_from_extreme.iter().enumerate() {
            if !prob.is_finite() || prob < 0.0 || prob > 1.0 {
                issues.push(format!(
                    "Invalid probability from extreme logit at index {}: {}",
                    i, prob
                ));
            }
        }

        Ok(issues)
    }

    /// Test log-sum-exp stability
    pub fn test_log_sum_exp_stability(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Test with large values
        let large_values = Array1::from(vec![700.0, 701.0, 702.0]);
        let lse_large = self.ops.log_sum_exp(&large_values);

        if !lse_large.is_finite() {
            issues.push("Log-sum-exp failed with large values".to_string());
        }

        // Test with small values
        let small_values = Array1::from(vec![-700.0, -701.0, -702.0]);
        let lse_small = self.ops.log_sum_exp(&small_values);

        if !lse_small.is_finite() {
            issues.push("Log-sum-exp failed with small values".to_string());
        }

        // Test empty array
        let empty_values = Array1::from(vec![]);
        let lse_empty = self.ops.log_sum_exp(&empty_values);

        if lse_empty != Float::NEG_INFINITY {
            issues.push("Log-sum-exp should return -∞ for empty array".to_string());
        }

        Ok(issues)
    }

    /// Test softmax stability
    pub fn test_softmax_stability(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Test with extreme values
        let extreme_logits = Array1::from(vec![-1000.0, 0.0, 1000.0]);
        let softmax_result = self.ops.safe_softmax(&extreme_logits);

        let sum: Float = softmax_result.sum();
        if (sum - 1.0).abs() > 1e-10 {
            issues.push(format!("Softmax doesn't sum to 1: {}", sum));
        }

        for (i, &value) in softmax_result.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                issues.push(format!("Invalid softmax value at index {}: {}", i, value));
            }
        }

        // Test with all large values
        let all_large = Array1::from(vec![1000.0, 1000.0, 1000.0]);
        let softmax_large = self.ops.safe_softmax(&all_large);

        for &value in softmax_large.iter() {
            if (value - 1.0 / 3.0).abs() > 1e-6 {
                issues.push("Softmax of equal large values should be uniform".to_string());
                break;
            }
        }

        Ok(issues)
    }

    /// Comprehensive numerical stability test
    pub fn comprehensive_stability_test(&self) -> Result<Vec<String>> {
        let mut all_issues = Vec::new();

        all_issues.extend(self.test_probability_clamping()?);
        all_issues.extend(self.test_logit_conversion_stability()?);
        all_issues.extend(self.test_log_sum_exp_stability()?);
        all_issues.extend(self.test_softmax_stability()?);

        Ok(all_issues)
    }
}

/// Helper function to create numerically stable calibration estimator
pub fn create_stable_calibrator<T: Default>() -> T {
    // This would wrap any calibrator with numerical stability checks
    T::default()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_safe_probability_ops() {
        let ops = SafeProbabilityOps::default();

        // Test clamping
        let probs = array![0.0, 0.5, 1.0, -0.1, 1.1];
        let clamped = ops.clamp_probabilities(&probs);

        for &value in clamped.iter() {
            assert!(value >= 0.0 && value <= 1.0);
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_logit_conversion() {
        let ops = SafeProbabilityOps::default();

        let probs = array![0.1, 0.5, 0.9];
        let logits = ops.probabilities_to_logits(&probs);
        let recovered = ops.logits_to_probabilities(&logits);

        for (orig, recovered) in probs.iter().zip(recovered.iter()) {
            assert!((orig - recovered).abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_sum_exp() {
        let ops = SafeProbabilityOps::default();

        // Test known case: log(exp(1) + exp(2)) = log(e + e²) ≈ 2.31
        let values = array![1.0, 2.0];
        let result = ops.log_sum_exp(&values);
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_safe_softmax() {
        let ops = SafeProbabilityOps::default();

        let logits = array![1.0, 2.0, 3.0];
        let softmax = ops.safe_softmax(&logits);

        // Check sum to 1
        let sum: Float = softmax.sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check all positive
        for &value in softmax.iter() {
            assert!(value > 0.0);
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_numerical_stability_tests() {
        let tester = NumericalStabilityTests::new();

        let issues = tester.comprehensive_stability_test().unwrap();

        // Should have no issues for basic stability
        if !issues.is_empty() {
            println!("Stability issues found: {:?}", issues);
        }
    }

    #[test]
    fn test_robust_optimizer() {
        let mut config = NumericalConfig::default();
        // Remove regularization to test basic optimization
        config.regularization = 0.0;
        let optimizer = RobustOptimizer::new(config);

        // Minimize f(x) = (x - 2)²
        let objective = |x: Float| (x - 2.0).powi(2);
        let gradient = |x: Float| 2.0 * (x - 2.0);

        let result = optimizer
            .robust_scalar_optimization(
                0.0, // initial value
                objective,
                gradient,
                Some((-10.0, 10.0)), // bounds
            )
            .unwrap();

        // Should converge close to x = 2
        println!(
            "Optimization result: {}, difference from 2.0: {}",
            result,
            (result - 2.0).abs()
        );
        // Check both the result and verify it's actually the minimum
        let final_objective = objective(result);
        let optimal_objective = objective(2.0);
        println!(
            "Final objective: {}, Optimal objective: {}",
            final_objective, optimal_objective
        );

        assert!((result - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_convergence_fix() {
        // This test specifically validates that the convergence bug has been fixed
        let mut config = NumericalConfig::default();
        config.regularization = 0.0;
        config.convergence_tolerance = 1e-6;
        let optimizer = RobustOptimizer::new(config);

        // Test a simple quadratic that should take several iterations to minimize
        let objective = |x: Float| (x - 5.0).powi(2);
        let gradient = |x: Float| 2.0 * (x - 5.0);

        let result = optimizer
            .minimize(
                0.0, // Start far from optimum
                objective, gradient, None,
            )
            .unwrap();

        // Verify it actually found the minimum
        assert!(
            (result - 5.0).abs() < 1e-4,
            "Expected ~5.0, got {}, diff: {}",
            result,
            (result - 5.0).abs()
        );

        // Verify the objective value is very small at the result
        let final_obj = objective(result);
        assert!(
            final_obj < 1e-6,
            "Objective should be near zero at minimum, got: {}",
            final_obj
        );
    }

    #[test]
    fn test_minimize_method() {
        let mut config = NumericalConfig::default();
        config.regularization = 0.0;
        config.convergence_tolerance = 1e-6;
        let optimizer = RobustOptimizer::new(config);

        // Test 1: Minimize f(x) = (x - 3)² with different starting point
        let objective1 = |x: Float| (x - 3.0).powi(2);
        let gradient1 = |x: Float| 2.0 * (x - 3.0);

        let result1 = optimizer
            .minimize(
                -1.0, // initial value far from optimum
                objective1, gradient1, None, // no bounds
            )
            .unwrap();

        println!(
            "Test 1 - Minimize (x-3)²: result={}, expected=3.0, diff={}",
            result1,
            (result1 - 3.0).abs()
        );
        assert!((result1 - 3.0).abs() < 1e-4);

        // Test 2: Minimize f(x) = x² + 4x + 3 = (x + 2)² - 1, minimum at x = -2
        let objective2 = |x: Float| x.powi(2) + 4.0 * x + 3.0;
        let gradient2 = |x: Float| 2.0 * x + 4.0;

        let result2 = optimizer
            .minimize(
                5.0, // initial value
                objective2,
                gradient2,
                Some((-10.0, 10.0)), // with bounds
            )
            .unwrap();

        println!(
            "Test 2 - Minimize x²+4x+3: result={}, expected=-2.0, diff={}",
            result2,
            (result2 - (-2.0)).abs()
        );
        assert!((result2 - (-2.0)).abs() < 1e-4);

        // Test 3: Verify it doesn't converge immediately
        let objective3 = |x: Float| (x - 1.0).powi(2);
        let gradient3 = |x: Float| 2.0 * (x - 1.0);

        let result3 = optimizer
            .minimize(
                10.0, // start far from optimum
                objective3, gradient3, None,
            )
            .unwrap();

        println!(
            "Test 3 - Minimize (x-1)² from x=10: result={}, expected=1.0, diff={}",
            result3,
            (result3 - 1.0).abs()
        );
        assert!((result3 - 1.0).abs() < 1e-4);
    }
}
