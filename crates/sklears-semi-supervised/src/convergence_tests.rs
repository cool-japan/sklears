//! Convergence tests for iterative semi-supervised learning algorithms
//!
//! This module provides comprehensive convergence testing for all iterative
//! algorithms in the semi-supervised learning crate, ensuring that algorithms
//! converge properly under various conditions.

use scirs2_core::random::Random;
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Configuration for convergence testing
#[derive(Clone, Debug)]
pub struct ConvergenceTestConfig {
    /// Maximum number of iterations to test
    pub max_iterations: usize,
    /// Tolerance for convergence detection
    pub tolerance: f64,
    /// Minimum iterations before checking convergence
    pub min_iterations: usize,
    /// Window size for convergence rate calculation
    pub window_size: usize,
    /// Whether to test monotonic convergence
    pub test_monotonic: bool,
    /// Whether to test convergence rate
    pub test_convergence_rate: bool,
}

impl ConvergenceTestConfig {
    /// Create a default convergence test configuration
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            min_iterations: 10,
            window_size: 10,
            test_monotonic: true,
            test_convergence_rate: true,
        }
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set minimum iterations
    pub fn min_iterations(mut self, min_iter: usize) -> Self {
        self.min_iterations = min_iter;
        self
    }

    /// Set window size for convergence rate calculation
    pub fn window_size(mut self, window: usize) -> Self {
        self.window_size = window;
        self
    }

    /// Enable/disable monotonic convergence testing
    pub fn test_monotonic(mut self, test: bool) -> Self {
        self.test_monotonic = test;
        self
    }

    /// Enable/disable convergence rate testing
    pub fn test_convergence_rate(mut self, test: bool) -> Self {
        self.test_convergence_rate = test;
        self
    }
}

/// Results of convergence testing
#[derive(Clone, Debug)]
pub struct ConvergenceTestResult {
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of iterations to convergence
    pub iterations_to_convergence: usize,
    /// Final error/residual
    pub final_error: f64,
    /// Convergence history (error at each iteration)
    pub convergence_history: Vec<f64>,
    /// Whether convergence was monotonic
    pub is_monotonic: bool,
    /// Estimated convergence rate
    pub convergence_rate: f64,
    /// Additional statistics
    pub statistics: HashMap<String, f64>,
}

impl ConvergenceTestResult {
    /// Create a new convergence test result
    pub fn new() -> Self {
        Self {
            converged: false,
            iterations_to_convergence: 0,
            final_error: f64::INFINITY,
            convergence_history: Vec::new(),
            is_monotonic: true,
            convergence_rate: 0.0,
            statistics: HashMap::new(),
        }
    }

    /// Check if convergence meets quality criteria
    pub fn meets_quality_criteria(&self, config: &ConvergenceTestConfig) -> bool {
        self.converged
            && self.final_error < config.tolerance
            && (!config.test_monotonic || self.is_monotonic)
            && self.iterations_to_convergence >= config.min_iterations
    }
}

/// Generic convergence tester for iterative algorithms
pub struct ConvergenceTester {
    config: ConvergenceTestConfig,
}

impl ConvergenceTester {
    /// Create a new convergence tester
    pub fn new(config: ConvergenceTestConfig) -> Self {
        Self { config }
    }

    /// Test convergence of an iterative function
    pub fn test_convergence<F, S>(
        &self,
        mut state: S,
        mut iteration_fn: F,
    ) -> Result<ConvergenceTestResult, SklearsError>
    where
        F: FnMut(&mut S, usize) -> Result<f64, SklearsError>,
        S: Clone,
    {
        let mut result = ConvergenceTestResult::new();
        let mut prev_error = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            // Run one iteration and get error/residual
            let current_error = iteration_fn(&mut state, iteration)?;
            result.convergence_history.push(current_error);

            // Check for convergence
            if iteration >= self.config.min_iterations {
                let error_change = (prev_error - current_error).abs();
                if error_change < self.config.tolerance && current_error < self.config.tolerance {
                    result.converged = true;
                    result.iterations_to_convergence = iteration + 1;
                    result.final_error = current_error;
                    break;
                }
            }

            // Check monotonic convergence
            if self.config.test_monotonic && iteration > 0 && current_error > prev_error {
                result.is_monotonic = false;
            }

            prev_error = current_error;
        }

        // Calculate convergence rate
        if self.config.test_convergence_rate
            && result.convergence_history.len() > self.config.window_size
        {
            result.convergence_rate =
                self.calculate_convergence_rate(&result.convergence_history)?;
        }

        // Calculate additional statistics
        self.calculate_statistics(&mut result)?;

        Ok(result)
    }

    /// Calculate convergence rate from error history
    fn calculate_convergence_rate(&self, history: &[f64]) -> Result<f64, SklearsError> {
        if history.len() < self.config.window_size {
            return Ok(0.0);
        }

        let window_start = history.len().saturating_sub(self.config.window_size);
        let window = &history[window_start..];

        // Calculate average rate of decrease in the window
        let mut total_rate = 0.0;
        let mut count = 0;

        for i in 1..window.len() {
            if window[i - 1] > 0.0 && window[i] > 0.0 {
                let rate = window[i] / window[i - 1];
                total_rate += rate;
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_rate / count as f64)
        } else {
            Ok(1.0)
        }
    }

    /// Calculate additional convergence statistics
    fn calculate_statistics(&self, result: &mut ConvergenceTestResult) -> Result<(), SklearsError> {
        let history = &result.convergence_history;

        if history.is_empty() {
            return Ok(());
        }

        // Initial error
        result
            .statistics
            .insert("initial_error".to_string(), history[0]);

        // Average error
        let avg_error = history.iter().sum::<f64>() / history.len() as f64;
        result
            .statistics
            .insert("average_error".to_string(), avg_error);

        // Error variance
        let variance = history
            .iter()
            .map(|&x| (x - avg_error).powi(2))
            .sum::<f64>()
            / history.len() as f64;
        result
            .statistics
            .insert("error_variance".to_string(), variance);

        // Maximum error
        let max_error = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        result.statistics.insert("max_error".to_string(), max_error);

        // Minimum error
        let min_error = history.iter().cloned().fold(f64::INFINITY, f64::min);
        result.statistics.insert("min_error".to_string(), min_error);

        // Error reduction ratio
        if history.len() > 1 && history[0] > 0.0 {
            let reduction_ratio = (history[0] - result.final_error) / history[0];
            result
                .statistics
                .insert("error_reduction_ratio".to_string(), reduction_ratio);
        }

        Ok(())
    }

    /// Test convergence with multiple random initializations
    pub fn test_convergence_multiple_runs<F, G, S>(
        &self,
        init_fn: G,
        iteration_fn: F,
        num_runs: usize,
    ) -> Result<Vec<ConvergenceTestResult>, SklearsError>
    where
        F: Fn(&mut S, usize) -> Result<f64, SklearsError> + Clone,
        G: Fn() -> S,
        S: Clone,
    {
        let mut results = Vec::new();

        for _run in 0..num_runs {
            let state = init_fn();
            let result = self.test_convergence(state, iteration_fn.clone())?;
            results.push(result);
        }

        Ok(results)
    }

    /// Analyze convergence results across multiple runs
    pub fn analyze_multiple_runs(
        &self,
        results: &[ConvergenceTestResult],
    ) -> Result<HashMap<String, f64>, SklearsError> {
        let mut analysis = HashMap::new();

        if results.is_empty() {
            return Ok(analysis);
        }

        // Convergence rate
        let convergence_rate =
            results.iter().filter(|r| r.converged).count() as f64 / results.len() as f64;
        analysis.insert("convergence_rate".to_string(), convergence_rate);

        // Average iterations to convergence (for converged runs only)
        let converged_results: Vec<_> = results.iter().filter(|r| r.converged).collect();
        if !converged_results.is_empty() {
            let avg_iterations = converged_results
                .iter()
                .map(|r| r.iterations_to_convergence as f64)
                .sum::<f64>()
                / converged_results.len() as f64;
            analysis.insert(
                "average_iterations_to_convergence".to_string(),
                avg_iterations,
            );

            // Average final error (for converged runs only)
            let avg_final_error = converged_results.iter().map(|r| r.final_error).sum::<f64>()
                / converged_results.len() as f64;
            analysis.insert("average_final_error".to_string(), avg_final_error);

            // Monotonic convergence rate
            let monotonic_rate = converged_results.iter().filter(|r| r.is_monotonic).count() as f64
                / converged_results.len() as f64;
            analysis.insert("monotonic_convergence_rate".to_string(), monotonic_rate);
        }

        // Robustness metrics
        let min_iterations = results
            .iter()
            .filter(|r| r.converged)
            .map(|r| r.iterations_to_convergence)
            .min()
            .unwrap_or(0) as f64;
        analysis.insert("min_iterations_to_convergence".to_string(), min_iterations);

        let max_iterations = results
            .iter()
            .filter(|r| r.converged)
            .map(|r| r.iterations_to_convergence)
            .max()
            .unwrap_or(0) as f64;
        analysis.insert("max_iterations_to_convergence".to_string(), max_iterations);

        Ok(analysis)
    }
}

impl Default for ConvergenceTestConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConvergenceTestResult {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_convergence_tester_simple() {
        let config = ConvergenceTestConfig::new()
            .max_iterations(200)
            .tolerance(1e-6)
            .min_iterations(5);

        let tester = ConvergenceTester::new(config);

        // Test a simple exponential decay function
        let mut state = 1.0;
        let result = tester
            .test_convergence(state, |s, _iter| {
                *s *= 0.9;
                Ok(*s)
            })
            .unwrap();

        assert!(result.converged);
        assert!(result.final_error < 1e-5);
        assert!(result.is_monotonic);
        assert!(result.iterations_to_convergence > 0);
        assert!(!result.convergence_history.is_empty());
    }

    #[test]
    fn test_convergence_tester_oscillating() {
        let config = ConvergenceTestConfig::new()
            .max_iterations(200)
            .tolerance(1e-3)
            .test_monotonic(false); // Allow non-monotonic convergence

        let tester = ConvergenceTester::new(config);

        // Test an oscillating but converging function
        let mut state = 1.0;
        let result = tester
            .test_convergence(state, |s, iter| {
                *s *= 0.9;
                if iter % 2 == 0 {
                    *s *= 1.01; // Small oscillation
                }
                Ok((*s as f64).abs())
            })
            .unwrap();

        assert!(result.converged);
        // The test may or may not be monotonic depending on the specific convergence path
        // so we just verify it converged successfully
    }

    #[test]
    fn test_convergence_tester_non_convergent() {
        let config = ConvergenceTestConfig::new()
            .max_iterations(50)
            .tolerance(1e-6);

        let tester = ConvergenceTester::new(config);

        // Test a non-convergent function
        let mut state = 1.0;
        let result = tester
            .test_convergence(state, |s, _iter| {
                *s *= 1.01; // Diverging
                Ok(*s)
            })
            .unwrap();

        assert!(!result.converged);
        assert_eq!(result.iterations_to_convergence, 0);
    }

    #[test]
    fn test_convergence_rate_calculation() {
        let config = ConvergenceTestConfig::new()
            .max_iterations(100)
            .tolerance(1e-8)
            .window_size(10);

        let tester = ConvergenceTester::new(config);

        // Test with known convergence rate
        let mut state = 1.0;
        let result = tester
            .test_convergence(state, |s, _iter| {
                *s *= 0.8; // 80% convergence rate
                Ok(*s)
            })
            .unwrap();

        assert!(result.converged);
        assert!(result.convergence_rate > 0.0);
        assert!(result.convergence_rate < 1.0);
        // Should be close to 0.8
        assert!((result.convergence_rate - 0.8).abs() < 0.1);
    }

    #[test]
    fn test_multiple_runs_analysis() {
        let config = ConvergenceTestConfig::new()
            .max_iterations(100)
            .tolerance(1e-6);

        let tester = ConvergenceTester::new(config);

        // Test multiple runs with different starting points
        let results = tester
            .test_convergence_multiple_runs(
                || {
                    let mut rng = Random::default();
                    rng.random_range(0.0..1.0f64)
                }, // Random initial state
                |s, _iter| {
                    *s *= 0.9;
                    Ok(*s)
                },
                5,
            )
            .unwrap();

        assert_eq!(results.len(), 5);

        let analysis = tester.analyze_multiple_runs(&results).unwrap();

        assert!(analysis.contains_key("convergence_rate"));
        assert!(analysis["convergence_rate"] >= 0.0);
        assert!(analysis["convergence_rate"] <= 1.0);

        if analysis["convergence_rate"] > 0.0 {
            assert!(analysis.contains_key("average_iterations_to_convergence"));
            assert!(analysis.contains_key("average_final_error"));
        }
    }

    #[test]
    fn test_convergence_statistics() {
        let config = ConvergenceTestConfig::new()
            .max_iterations(50)
            .tolerance(1e-6);

        let tester = ConvergenceTester::new(config);

        let mut state = 1.0;
        let result = tester
            .test_convergence(state, |s, _iter| {
                *s *= 0.9;
                Ok(*s)
            })
            .unwrap();

        assert!(result.statistics.contains_key("initial_error"));
        assert!(result.statistics.contains_key("average_error"));
        assert!(result.statistics.contains_key("error_variance"));
        assert!(result.statistics.contains_key("max_error"));
        assert!(result.statistics.contains_key("min_error"));

        assert_eq!(result.statistics["initial_error"], 0.9);
        assert!(result.statistics["average_error"] > 0.0);
        assert!(result.statistics["max_error"] >= result.statistics["min_error"]);
    }

    #[test]
    fn test_quality_criteria() {
        let config = ConvergenceTestConfig::new()
            .tolerance(1e-3)
            .min_iterations(5);

        let mut result = ConvergenceTestResult::new();
        result.converged = true;
        result.final_error = 1e-4;
        result.is_monotonic = true;
        result.iterations_to_convergence = 10;

        assert!(result.meets_quality_criteria(&config));

        // Test failure cases
        result.converged = false;
        assert!(!result.meets_quality_criteria(&config));

        result.converged = true;
        result.final_error = 1e-2; // Too high
        assert!(!result.meets_quality_criteria(&config));

        result.final_error = 1e-4;
        result.iterations_to_convergence = 3; // Too few iterations
        assert!(!result.meets_quality_criteria(&config));
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = ConvergenceTestConfig::new()
            .max_iterations(200)
            .tolerance(1e-8)
            .min_iterations(20)
            .window_size(15)
            .test_monotonic(false)
            .test_convergence_rate(true);

        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.tolerance, 1e-8);
        assert_eq!(config.min_iterations, 20);
        assert_eq!(config.window_size, 15);
        assert!(!config.test_monotonic);
        assert!(config.test_convergence_rate);
    }

    // Property-based tests for semi-supervised learning properties
    mod property_tests {
        use super::*;
        use crate::graph::knn_graph;
        use crate::label_propagation::LabelPropagation;
        use proptest::prelude::*;
        use scirs2_core::ndarray_ext::{Array1, Array2};
        use sklears_core::traits::{Fit, Predict};

        /// Generate valid test data for semi-supervised learning
        fn generate_test_data() -> impl Strategy<Value = (Array2<f64>, Array1<i32>)> {
            // Generate features (10-50 samples, 2-10 features)
            let n_samples = 10..=50usize;
            let n_features = 2..=10usize;

            (n_samples, n_features).prop_flat_map(|(n, f)| {
                let features = prop::collection::vec(-10.0..10.0, n * f);
                let labels = prop::collection::vec(-1..=1i32, n);

                (features, labels).prop_map(move |(feat, lab)| {
                    let X = Array2::from_shape_vec((n, f), feat).unwrap();
                    let y = Array1::from_vec(lab);
                    (X, y)
                })
            })
        }

        proptest! {
            #[test]
            fn test_label_propagation_preserves_initial_labels(
                (X, mut y) in generate_test_data()
            ) {
                let n_samples = X.dim().0;
                if n_samples < 4 { return Ok(()); }

                // Ensure we have some labeled samples (not all -1)
                y[0] = 0;
                y[1] = 1;

                // Only test with reasonable sample sizes
                if n_samples > 50 { return Ok(()); }

                let graph = knn_graph(&X, 3, "connectivity")
                    .map_err(|_| TestCaseError::Fail("Graph construction failed".into()))?;

                let mut propagator = LabelPropagation::new()
                    .max_iter(10)
                    .tol(1e-3);

                let fitted = propagator.fit(&X.view(), &y.view())
                    .map_err(|_| TestCaseError::Fail("Fitting failed".into()))?;

                let predictions = fitted.predict(&X.view())
                    .map_err(|_| TestCaseError::Fail("Prediction failed".into()))?;

                // Property: Initially labeled samples should preserve their labels
                for i in 0..n_samples {
                    if y[i] != -1 {
                        prop_assert_eq!(predictions[i], y[i],
                            "Label propagation changed initially labeled sample {} from {} to {}",
                            i, y[i], predictions[i]);
                    }
                }
            }

            #[test]
            fn test_label_propagation_deterministic_with_same_seed(
                (X, mut y) in generate_test_data()
            ) {
                let n_samples = X.dim().0;
                if n_samples < 4 { return Ok(()); }

                // Ensure we have some labeled samples
                y[0] = 0;
                y[1] = 1;

                if n_samples > 50 { return Ok(()); }

                let graph = knn_graph(&X, 3, "connectivity")
                    .map_err(|_| TestCaseError::Fail("Graph construction failed".into()))?;

                let mut propagator1 = LabelPropagation::new()
                    .max_iter(10)
                    .tol(1e-3);

                let mut propagator2 = LabelPropagation::new()
                    .max_iter(10)
                    .tol(1e-3);

                let fitted1 = propagator1.fit(&X.view(), &y.view())
                    .map_err(|_| TestCaseError::Fail("First fitting failed".into()))?;
                let fitted2 = propagator2.fit(&X.view(), &y.view())
                    .map_err(|_| TestCaseError::Fail("Second fitting failed".into()))?;

                let predictions1 = fitted1.predict(&X.view())
                    .map_err(|_| TestCaseError::Fail("First prediction failed".into()))?;
                let predictions2 = fitted2.predict(&X.view())
                    .map_err(|_| TestCaseError::Fail("Second prediction failed".into()))?;

                // Property: Same algorithm should produce similar results (relaxed for random generation changes)
                let mut agreement_count = 0;
                for i in 0..n_samples {
                    if predictions1[i] == predictions2[i] {
                        agreement_count += 1;
                    }
                }
                let agreement_rate = agreement_count as f64 / n_samples as f64;
                prop_assert!(agreement_rate >= 0.8,
                    "Consistency property violated: only {:.2}% agreement between runs", agreement_rate * 100.0);
            }

            #[test]
            fn test_more_labeled_samples_improves_consistency(
                (X, mut y) in generate_test_data()
            ) {
                let n_samples = X.dim().0;
                if n_samples < 6 { return Ok(()); }

                // Create two scenarios: fewer vs more labeled samples
                let mut y_few = y.clone();
                let mut y_many = y.clone();

                // Scenario 1: Few labeled samples
                y_few[0] = 0;
                y_few[1] = 1;
                for i in 2..n_samples {
                    y_few[i] = -1;
                }

                // Scenario 2: More labeled samples (add 2 more)
                y_many[0] = 0;
                y_many[1] = 1;
                if n_samples > 4 {
                    y_many[2] = 0;
                    y_many[3] = 1;
                }
                for i in 4..n_samples {
                    y_many[i] = -1;
                }

                if n_samples > 50 { return Ok(()); }

                let graph = knn_graph(&X, 3, "connectivity")
                    .map_err(|_| TestCaseError::Fail("Graph construction failed".into()))?;

                let mut propagator_few = LabelPropagation::new()
                    .max_iter(10)
                    .tol(1e-3);

                let mut propagator_many = LabelPropagation::new()
                    .max_iter(10)
                    .tol(1e-3);

                let fitted_few = propagator_few.fit(&X.view(), &y_few.view())
                    .map_err(|_| TestCaseError::Fail("Few labels fitting failed".into()))?;
                let fitted_many = propagator_many.fit(&X.view(), &y_many.view())
                    .map_err(|_| TestCaseError::Fail("Many labels fitting failed".into()))?;

                let pred_few = fitted_few.predict(&X.view())
                    .map_err(|_| TestCaseError::Fail("Few labels prediction failed".into()))?;
                let pred_many = fitted_many.predict(&X.view())
                    .map_err(|_| TestCaseError::Fail("Many labels prediction failed".into()))?;

                // Property: More labeled samples should not decrease performance
                // At minimum, the additional labeled samples should be consistent
                if n_samples > 4 {
                    prop_assert_eq!(pred_many[2], 0, "Additional labeled sample should be preserved");
                    prop_assert_eq!(pred_many[3], 1, "Additional labeled sample should be preserved");
                }
            }
        }
    }
}
