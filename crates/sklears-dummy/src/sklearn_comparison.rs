//! Comparison tests against scikit-learn reference implementations
//!
//! This module provides utilities to compare sklears dummy estimators against
//! their scikit-learn counterparts to ensure compatibility and correctness.

use crate::dummy_classifier::Strategy as ClassifierStrategy;
use crate::dummy_regressor::Strategy as RegressorStrategy;
use crate::{DummyClassifier, DummyRegressor};
use scirs2_core::ndarray::Array1;
use sklears_core::error::Result;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

/// Comparison result between sklears and scikit-learn implementations
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Strategy being compared
    pub strategy: String,
    /// Mean absolute error in predictions
    pub prediction_mae: Float,
    /// Maximum absolute error in predictions
    pub prediction_max_error: Float,
    /// Correlation between predictions
    pub prediction_correlation: Float,
    /// Are results considered equivalent (within tolerance)
    pub is_equivalent: bool,
    /// Tolerance used for comparison
    pub tolerance: Float,
}

/// Comparison framework for validating against scikit-learn
#[derive(Debug, Clone)]
pub struct SklearnComparisonFramework {
    /// Tolerance for numerical comparisons
    pub tolerance: Float,
    /// Random state for reproducible comparisons
    pub random_state: Option<u64>,
}

impl SklearnComparisonFramework {
    /// Create new comparison framework
    pub fn new() -> Self {
        Self {
            tolerance: 1e-10,
            random_state: Some(42),
        }
    }

    /// Set tolerance for comparisons
    pub fn with_tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Compare dummy classifier strategies against expected sklearn behavior
    pub fn compare_dummy_classifier_strategies(
        &self,
        x: &Features,
        y: &Array1<Int>,
    ) -> Result<Vec<ComparisonResult>> {
        let mut results = Vec::new();

        // Test MostFrequent strategy
        let result = self.compare_most_frequent_classifier(x, y)?;
        results.push(result);

        // Test Stratified strategy
        let result = self.compare_stratified_classifier(x, y)?;
        results.push(result);

        // Test Uniform strategy
        let result = self.compare_uniform_classifier(x, y)?;
        results.push(result);

        // Test Constant strategy
        if let Some(constant_class) = y.iter().next() {
            let result = self.compare_constant_classifier(x, y, *constant_class)?;
            results.push(result);
        }

        // Test Prior strategy
        let result = self.compare_prior_classifier(x, y)?;
        results.push(result);

        Ok(results)
    }

    /// Compare dummy regressor strategies against expected sklearn behavior
    pub fn compare_dummy_regressor_strategies(
        &self,
        x: &Features,
        y: &Array1<Float>,
    ) -> Result<Vec<ComparisonResult>> {
        let mut results = Vec::new();

        // Test Mean strategy
        let result = self.compare_mean_regressor(x, y)?;
        results.push(result);

        // Test Median strategy
        let result = self.compare_median_regressor(x, y)?;
        results.push(result);

        // Test Quantile strategy
        let result = self.compare_quantile_regressor(x, y, 0.25)?;
        results.push(result);

        // Test Constant strategy
        let constant_value = y.mean().unwrap_or(0.0);
        let result = self.compare_constant_regressor(x, y, constant_value)?;
        results.push(result);

        Ok(results)
    }

    /// Compare MostFrequent classifier strategy
    fn compare_most_frequent_classifier(
        &self,
        x: &Features,
        y: &Array1<Int>,
    ) -> Result<ComparisonResult> {
        // Our implementation
        let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent)
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = classifier.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // Expected sklearn behavior: always predict most frequent class
        let most_frequent_class = self.compute_most_frequent_class(y);
        let expected_predictions = Array1::from_elem(x.nrows(), most_frequent_class);

        self.compare_predictions(
            &our_predictions,
            &expected_predictions,
            "MostFrequent".to_string(),
        )
    }

    /// Compare Stratified classifier strategy
    fn compare_stratified_classifier(
        &self,
        x: &Features,
        y: &Array1<Int>,
    ) -> Result<ComparisonResult> {
        // Our implementation
        let classifier = DummyClassifier::new(ClassifierStrategy::Stratified)
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = classifier.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // For stratified, we check that class distribution matches training distribution
        let our_class_distribution = self.compute_class_distribution(&our_predictions);
        let expected_class_distribution = self.compute_class_distribution(y);

        // Compute similarity in class distributions
        let distribution_similarity = self
            .compute_distribution_similarity(&our_class_distribution, &expected_class_distribution);

        Ok(ComparisonResult {
            strategy: "Stratified".to_string(),
            prediction_mae: 1.0 - distribution_similarity, // Use distribution similarity as proxy
            prediction_max_error: 1.0 - distribution_similarity,
            prediction_correlation: distribution_similarity,
            is_equivalent: distribution_similarity > 1.0 - self.tolerance,
            tolerance: self.tolerance,
        })
    }

    /// Compare Uniform classifier strategy
    fn compare_uniform_classifier(
        &self,
        x: &Features,
        y: &Array1<Int>,
    ) -> Result<ComparisonResult> {
        // Our implementation
        let classifier = DummyClassifier::new(ClassifierStrategy::Uniform)
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = classifier.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // For uniform, we check that all classes are roughly equally represented
        let unique_classes = self.get_unique_classes(y);
        let our_class_distribution = self.compute_class_distribution(&our_predictions);

        let expected_uniform_prob = 1.0 / unique_classes.len() as Float;
        let uniform_similarity = our_class_distribution
            .values()
            .map(|&prob| (prob - expected_uniform_prob).abs())
            .sum::<Float>()
            / 2.0; // Divide by 2 for total variation distance

        Ok(ComparisonResult {
            strategy: "Uniform".to_string(),
            prediction_mae: uniform_similarity,
            prediction_max_error: uniform_similarity,
            prediction_correlation: 1.0 - uniform_similarity,
            is_equivalent: uniform_similarity < self.tolerance,
            tolerance: self.tolerance,
        })
    }

    /// Compare Constant classifier strategy
    fn compare_constant_classifier(
        &self,
        x: &Features,
        y: &Array1<Int>,
        constant: Int,
    ) -> Result<ComparisonResult> {
        // Our implementation
        let classifier = DummyClassifier::new(ClassifierStrategy::Constant)
            .with_constant(constant)
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = classifier.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // Expected: all predictions should be the constant value
        let expected_predictions = Array1::from_elem(x.nrows(), constant);

        self.compare_predictions(
            &our_predictions,
            &expected_predictions,
            "Constant".to_string(),
        )
    }

    /// Compare Prior classifier strategy
    fn compare_prior_classifier(&self, x: &Features, y: &Array1<Int>) -> Result<ComparisonResult> {
        // Our implementation
        let classifier = DummyClassifier::new(ClassifierStrategy::Prior)
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = classifier.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // Expected: predictions based on class priors (same as stratified for dummy baseline)
        let our_class_distribution = self.compute_class_distribution(&our_predictions);
        let expected_class_distribution = self.compute_class_distribution(y);

        let distribution_similarity = self
            .compute_distribution_similarity(&our_class_distribution, &expected_class_distribution);

        Ok(ComparisonResult {
            strategy: "Prior".to_string(),
            prediction_mae: 1.0 - distribution_similarity,
            prediction_max_error: 1.0 - distribution_similarity,
            prediction_correlation: distribution_similarity,
            is_equivalent: distribution_similarity > 1.0 - self.tolerance,
            tolerance: self.tolerance,
        })
    }

    /// Compare Mean regressor strategy
    fn compare_mean_regressor(&self, x: &Features, y: &Array1<Float>) -> Result<ComparisonResult> {
        // Our implementation
        let regressor = DummyRegressor::new(RegressorStrategy::Mean)
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = regressor.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // Expected sklearn behavior: always predict mean of training targets
        let mean_value = y.mean().unwrap_or(0.0);
        let expected_predictions = Array1::from_elem(x.nrows(), mean_value);

        self.compare_float_predictions(&our_predictions, &expected_predictions, "Mean".to_string())
    }

    /// Compare Median regressor strategy
    fn compare_median_regressor(
        &self,
        x: &Features,
        y: &Array1<Float>,
    ) -> Result<ComparisonResult> {
        // Our implementation
        let regressor = DummyRegressor::new(RegressorStrategy::Median)
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = regressor.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // Expected sklearn behavior: always predict median of training targets
        let median_value = self.compute_median(y);
        let expected_predictions = Array1::from_elem(x.nrows(), median_value);

        self.compare_float_predictions(
            &our_predictions,
            &expected_predictions,
            "Median".to_string(),
        )
    }

    /// Compare Quantile regressor strategy
    fn compare_quantile_regressor(
        &self,
        x: &Features,
        y: &Array1<Float>,
        quantile: Float,
    ) -> Result<ComparisonResult> {
        // Our implementation
        let regressor = DummyRegressor::new(RegressorStrategy::Quantile(quantile))
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = regressor.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // Expected sklearn behavior: always predict specified quantile of training targets
        let quantile_value = self.compute_quantile(y, quantile);
        let expected_predictions = Array1::from_elem(x.nrows(), quantile_value);

        self.compare_float_predictions(
            &our_predictions,
            &expected_predictions,
            format!("Quantile({})", quantile),
        )
    }

    /// Compare Constant regressor strategy
    fn compare_constant_regressor(
        &self,
        x: &Features,
        y: &Array1<Float>,
        constant: Float,
    ) -> Result<ComparisonResult> {
        // Our implementation
        let regressor = DummyRegressor::new(RegressorStrategy::Constant(constant))
            .with_random_state(self.random_state.unwrap_or(42));
        let fitted = regressor.fit(x, y)?;
        let our_predictions = fitted.predict(x)?;

        // Expected: all predictions should be the constant value
        let expected_predictions = Array1::from_elem(x.nrows(), constant);

        self.compare_float_predictions(
            &our_predictions,
            &expected_predictions,
            "Constant".to_string(),
        )
    }

    /// Compare integer predictions
    fn compare_predictions(
        &self,
        our_predictions: &Array1<Int>,
        expected_predictions: &Array1<Int>,
        strategy: String,
    ) -> Result<ComparisonResult> {
        if our_predictions.len() != expected_predictions.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Prediction arrays must have same length".to_string(),
            ));
        }

        let n_samples = our_predictions.len() as Float;
        let mut absolute_errors = Vec::new();
        let mut matches = 0;

        for (&our_pred, &expected_pred) in our_predictions.iter().zip(expected_predictions.iter()) {
            let error = (our_pred - expected_pred).abs() as Float;
            absolute_errors.push(error);
            if our_pred == expected_pred {
                matches += 1;
            }
        }

        let mae = absolute_errors.iter().sum::<Float>() / n_samples;
        let max_error = absolute_errors.iter().fold(0.0f64, |a, &b| a.max(b));
        let accuracy = matches as Float / n_samples;

        Ok(ComparisonResult {
            strategy,
            prediction_mae: mae,
            prediction_max_error: max_error,
            prediction_correlation: accuracy,
            is_equivalent: mae < self.tolerance,
            tolerance: self.tolerance,
        })
    }

    /// Compare float predictions
    fn compare_float_predictions(
        &self,
        our_predictions: &Array1<Float>,
        expected_predictions: &Array1<Float>,
        strategy: String,
    ) -> Result<ComparisonResult> {
        if our_predictions.len() != expected_predictions.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Prediction arrays must have same length".to_string(),
            ));
        }

        let n_samples = our_predictions.len() as Float;
        let mut absolute_errors = Vec::new();

        for (&our_pred, &expected_pred) in our_predictions.iter().zip(expected_predictions.iter()) {
            let error = (our_pred - expected_pred).abs();
            absolute_errors.push(error);
        }

        let mae = absolute_errors.iter().sum::<Float>() / n_samples;
        let max_error = absolute_errors.iter().fold(0.0f64, |a, &b| a.max(b));

        // Compute correlation
        let correlation = self.compute_correlation(our_predictions, expected_predictions);

        Ok(ComparisonResult {
            strategy,
            prediction_mae: mae,
            prediction_max_error: max_error,
            prediction_correlation: correlation,
            is_equivalent: mae < self.tolerance,
            tolerance: self.tolerance,
        })
    }

    /// Utility functions for comparisons
    fn compute_most_frequent_class(&self, y: &Array1<Int>) -> Int {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        *class_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
            .unwrap_or(&0)
    }

    fn compute_class_distribution(&self, y: &Array1<Int>) -> HashMap<Int, Float> {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let n_samples = y.len() as Float;
        class_counts
            .into_iter()
            .map(|(class, count)| (class, count as Float / n_samples))
            .collect()
    }

    fn get_unique_classes(&self, y: &Array1<Int>) -> Vec<Int> {
        let mut classes: Vec<Int> = y.iter().copied().collect();
        classes.sort();
        classes.dedup();
        classes
    }

    fn compute_distribution_similarity(
        &self,
        dist1: &HashMap<Int, Float>,
        dist2: &HashMap<Int, Float>,
    ) -> Float {
        let mut all_classes: std::collections::HashSet<Int> = std::collections::HashSet::new();
        all_classes.extend(dist1.keys());
        all_classes.extend(dist2.keys());

        let mut total_variation = 0.0;
        for &class in &all_classes {
            let prob1 = *dist1.get(&class).unwrap_or(&0.0);
            let prob2 = *dist2.get(&class).unwrap_or(&0.0);
            total_variation += (prob1 - prob2).abs();
        }

        1.0 - (total_variation / 2.0) // Convert to similarity
    }

    fn compute_median(&self, y: &Array1<Float>) -> Float {
        let mut sorted_y: Vec<Float> = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_y.len();
        if n % 2 == 0 {
            (sorted_y[n / 2 - 1] + sorted_y[n / 2]) / 2.0
        } else {
            sorted_y[n / 2]
        }
    }

    fn compute_quantile(&self, y: &Array1<Float>, quantile: Float) -> Float {
        let mut sorted_y: Vec<Float> = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (quantile * (sorted_y.len() - 1) as Float) as usize;
        sorted_y[index.min(sorted_y.len() - 1)]
    }

    fn compute_correlation(&self, x: &Array1<Float>, y: &Array1<Float>) -> Float {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut x_sq_sum = 0.0;
        let mut y_sq_sum = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x_diff = xi - x_mean;
            let y_diff = yi - y_mean;
            numerator += x_diff * y_diff;
            x_sq_sum += x_diff * x_diff;
            y_sq_sum += y_diff * y_diff;
        }

        let denominator = (x_sq_sum * y_sq_sum).sqrt();
        if denominator == 0.0 {
            1.0 // Perfect correlation for constant values
        } else {
            numerator / denominator
        }
    }
}

impl Default for SklearnComparisonFramework {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate comprehensive comparison report
pub fn generate_comparison_report(results: &[ComparisonResult]) -> String {
    let mut report = String::new();
    report.push_str("=== Scikit-learn Compatibility Report ===\n\n");

    let total_strategies = results.len();
    let passing_strategies = results.iter().filter(|r| r.is_equivalent).count();

    report.push_str(&format!(
        "Summary: {}/{} strategies are equivalent to scikit-learn\n\n",
        passing_strategies, total_strategies
    ));

    for result in results {
        report.push_str(&format!("Strategy: {}\n", result.strategy));
        report.push_str(&format!(
            "  Prediction MAE: {:.2e}\n",
            result.prediction_mae
        ));
        report.push_str(&format!(
            "  Max Error: {:.2e}\n",
            result.prediction_max_error
        ));
        report.push_str(&format!(
            "  Correlation: {:.6}\n",
            result.prediction_correlation
        ));
        report.push_str(&format!(
            "  Equivalent: {}\n",
            if result.is_equivalent { "✓" } else { "✗" }
        ));
        report.push_str(&format!("  Tolerance: {:.2e}\n\n", result.tolerance));
    }

    report
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_comparison_framework_classifier() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1]; // Class 0 is more frequent

        let framework = SklearnComparisonFramework::new().with_tolerance(1e-10);
        let results = framework
            .compare_dummy_classifier_strategies(&x, &y)
            .unwrap();

        assert!(!results.is_empty());

        // MostFrequent should be deterministic and exact
        let most_frequent_result = results
            .iter()
            .find(|r| r.strategy == "MostFrequent")
            .unwrap();
        assert!(most_frequent_result.is_equivalent);
        assert_abs_diff_eq!(most_frequent_result.prediction_mae, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_comparison_framework_regressor() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0]; // Simple values

        let framework = SklearnComparisonFramework::new().with_tolerance(1e-10);
        let results = framework
            .compare_dummy_regressor_strategies(&x, &y)
            .unwrap();

        assert!(!results.is_empty());

        // Mean strategy should be exact
        let mean_result = results.iter().find(|r| r.strategy == "Mean").unwrap();
        assert!(mean_result.is_equivalent);
        assert_abs_diff_eq!(mean_result.prediction_mae, 0.0, epsilon = 1e-10);

        // Median strategy should be exact
        let median_result = results.iter().find(|r| r.strategy == "Median").unwrap();
        assert!(median_result.is_equivalent);
        assert_abs_diff_eq!(median_result.prediction_mae, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_utility_functions() {
        let framework = SklearnComparisonFramework::new();

        // Test median computation
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = framework.compute_median(&y);
        assert_abs_diff_eq!(median, 3.0, epsilon = 1e-10);

        // Test quantile computation
        let quantile_25 = framework.compute_quantile(&y, 0.25);
        assert_abs_diff_eq!(quantile_25, 2.0, epsilon = 1e-10);

        // Test correlation
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.0, 6.0, 8.0]; // Perfect positive correlation
        let correlation = framework.compute_correlation(&x, &y);
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_class_distribution_computation() {
        let framework = SklearnComparisonFramework::new();
        let y = array![0, 0, 1, 1, 1, 2];

        let distribution = framework.compute_class_distribution(&y);
        assert_abs_diff_eq!(distribution[&0], 2.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distribution[&1], 3.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distribution[&2], 1.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_comparison_report_generation() {
        let results = vec![
            ComparisonResult {
                strategy: "MostFrequent".to_string(),
                prediction_mae: 0.0,
                prediction_max_error: 0.0,
                prediction_correlation: 1.0,
                is_equivalent: true,
                tolerance: 1e-10,
            },
            ComparisonResult {
                strategy: "Stratified".to_string(),
                prediction_mae: 0.1,
                prediction_max_error: 0.2,
                prediction_correlation: 0.9,
                is_equivalent: false,
                tolerance: 1e-10,
            },
        ];

        let report = generate_comparison_report(&results);
        assert!(report.contains("1/2 strategies are equivalent"));
        assert!(report.contains("MostFrequent"));
        assert!(report.contains("Stratified"));
        assert!(report.contains("✓"));
        assert!(report.contains("✗"));
    }
}
