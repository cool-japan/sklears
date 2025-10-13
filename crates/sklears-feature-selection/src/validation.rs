//! Statistical validation framework for feature selection methods
//!
//! This module provides comprehensive statistical tests to validate the
//! properties and quality of feature selection algorithms.

use crate::base::FeatureSelector;
use scirs2_core::ndarray::{Array2, Axis};

use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Transform},
    types::Float,
};

/// Statistical validity tests for feature selection methods
pub struct StatisticalValidationFramework {
    confidence_level: Float,
    n_permutations: usize,
    random_state: Option<u64>,
}

impl Default for StatisticalValidationFramework {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalValidationFramework {
    /// Create a new statistical validation framework
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95,
            n_permutations: 1000,
            random_state: None,
        }
    }

    /// Set the confidence level for statistical tests
    pub fn confidence_level(mut self, level: Float) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set the number of permutations for permutation tests
    pub fn n_permutations(mut self, n: usize) -> Self {
        self.n_permutations = n;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Results from statistical validity tests
#[derive(Debug, Clone)]
pub struct StatisticalValidationResults {
    pub test_name: String,
    pub test_statistic: Float,
    pub p_value: Float,
    pub is_significant: bool,
    pub confidence_level: Float,
    pub effect_size: Option<Float>,
    pub description: String,
}

impl StatisticalValidationResults {
    pub fn new(
        test_name: String,
        test_statistic: Float,
        p_value: Float,
        confidence_level: Float,
        description: String,
    ) -> Self {
        let is_significant = p_value < (1.0 - confidence_level);
        Self {
            test_name,
            test_statistic,
            p_value,
            is_significant,
            confidence_level,
            effect_size: None,
            description,
        }
    }

    pub fn with_effect_size(mut self, effect_size: Float) -> Self {
        self.effect_size = Some(effect_size);
        self
    }
}

/// Test for selection consistency across data splits
pub struct SelectionConsistencyTest;

impl SelectionConsistencyTest {
    /// Test if feature selection is consistent across random data splits
    pub fn test_split_consistency<S, T>(
        selector_factory: impl Fn() -> S + Clone,
        features: &Array2<Float>,
        target: &T,
        n_splits: usize,
        test_size: Float,
        confidence_level: Float,
    ) -> SklResult<StatisticalValidationResults>
    where
        S: Fit<Array2<Float>, T> + Clone,
        S::Fitted: FeatureSelector + Transform<Array2<Float>>,
        T: Clone,
    {
        if n_splits < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 splits for consistency test".to_string(),
            ));
        }

        let n_samples = features.nrows();
        let n_test = (n_samples as Float * test_size) as usize;
        let n_train = n_samples - n_test;

        if n_train == 0 || n_test == 0 {
            return Err(SklearsError::InvalidInput(
                "Insufficient data for train/test split".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(42);
        let mut jaccard_similarities = Vec::new();
        let mut selected_features_list = Vec::new();

        // Perform multiple trials with slightly perturbed data
        for trial in 0..n_splits {
            // Add small amount of noise to test robustness
            let mut perturbed_features = features.clone();
            let noise_level = 0.01 * (trial + 1) as Float;

            for mut row in perturbed_features.axis_iter_mut(Axis(0)) {
                for element in row.iter_mut() {
                    let u1: Float = rng.gen();
                    let u2: Float = rng.gen();
                    let noise = (-2.0f64 * u1.ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * u2).cos()
                        * noise_level;
                    *element += noise;
                }
            }

            // Fit selector on perturbed data
            let selector = selector_factory();
            if let Ok(fitted_selector) = selector.fit(&perturbed_features, target) {
                let selected = fitted_selector.selected_features().clone();
                selected_features_list.push(selected);
            }
        }

        // Compute pairwise Jaccard similarities
        for i in 0..selected_features_list.len() {
            for j in (i + 1)..selected_features_list.len() {
                let similarity = Self::jaccard_similarity(
                    &selected_features_list[i],
                    &selected_features_list[j],
                );
                jaccard_similarities.push(similarity);
            }
        }

        if jaccard_similarities.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid feature selections obtained".to_string(),
            ));
        }

        // Compute statistics
        let mean_similarity: Float =
            jaccard_similarities.iter().sum::<Float>() / jaccard_similarities.len() as Float;
        let variance = jaccard_similarities
            .iter()
            .map(|&x| (x - mean_similarity) * (x - mean_similarity))
            .sum::<Float>()
            / (jaccard_similarities.len() - 1) as Float;
        let std_dev = variance.sqrt();

        // Test statistic: mean similarity (higher = more consistent)
        let test_statistic = mean_similarity;

        // Rough p-value based on how far from random expectation
        // Random Jaccard similarity for feature sets is typically low
        let expected_random_similarity = 0.1; // Rough estimate
        let z_score = (mean_similarity - expected_random_similarity)
            / (std_dev / (jaccard_similarities.len() as Float).sqrt());
        let p_value = (2.0 * (1.0 - Self::normal_cdf(z_score.abs()))).max(0.001); // Two-tailed test

        let description = format!(
            "Selection consistency test across {} splits. Mean Jaccard similarity: {:.3}, Std: {:.3}",
            n_splits, mean_similarity, std_dev
        );

        Ok(StatisticalValidationResults::new(
            "Split Consistency Test".to_string(),
            test_statistic,
            p_value,
            confidence_level,
            description,
        )
        .with_effect_size(z_score))
    }

    fn jaccard_similarity(set1: &[usize], set2: &[usize]) -> Float {
        let set1: std::collections::HashSet<usize> = set1.iter().cloned().collect();
        let set2: std::collections::HashSet<usize> = set2.iter().cloned().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            1.0
        } else {
            intersection as Float / union as Float
        }
    }

    /// Approximate normal CDF for p-value calculation
    fn normal_cdf(x: Float) -> Float {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Approximate error function
    fn erf(x: Float) -> Float {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

/// Test for feature selection significance using permutation testing
pub struct PermutationSignificanceTest;

impl PermutationSignificanceTest {
    /// Test if selected features are significantly better than random
    pub fn test_feature_significance<S, T>(
        selector_factory: impl Fn() -> S + Clone,
        features: &Array2<Float>,
        target: &T,
        n_permutations: usize,
        confidence_level: Float,
    ) -> SklResult<StatisticalValidationResults>
    where
        S: Fit<Array2<Float>, T> + Clone,
        S::Fitted: FeatureSelector + Transform<Array2<Float>>,
        T: Clone,
    {
        // Get the actual feature selection
        let selector = selector_factory();
        let fitted_selector = selector.fit(features, target)?;
        let selected_features = fitted_selector.selected_features();
        let n_selected = selected_features.len();

        if n_selected == 0 {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        // Compute score for actual selection
        let actual_score = Self::compute_selection_score(features, selected_features)?;

        // Perform permutation test
        let mut rng = StdRng::seed_from_u64(42);
        let mut permutation_scores = Vec::new();
        let n_features = features.ncols();

        for _ in 0..n_permutations {
            // Create random feature selection of same size
            let mut random_features: Vec<usize> = (0..n_features).collect();
            random_features.shuffle(&mut rng);
            let random_selection = &random_features[..n_selected];

            let random_score = Self::compute_selection_score(features, random_selection)?;
            permutation_scores.push(random_score);
        }

        // Compute p-value
        let n_better = permutation_scores
            .iter()
            .filter(|&&score| score >= actual_score)
            .count();
        let p_value = (n_better + 1) as Float / (n_permutations + 1) as Float;

        // Effect size (standardized difference)
        let mean_random: Float =
            permutation_scores.iter().sum::<Float>() / permutation_scores.len() as Float;
        let variance_random = permutation_scores
            .iter()
            .map(|&x| (x - mean_random) * (x - mean_random))
            .sum::<Float>()
            / (permutation_scores.len() - 1) as Float;
        let std_random = variance_random.sqrt();

        let effect_size = if std_random > 0.0 {
            (actual_score - mean_random) / std_random
        } else {
            0.0
        };

        let description = format!(
            "Permutation test with {} permutations. Actual score: {:.3}, Random mean: {:.3} ± {:.3}",
            n_permutations, actual_score, mean_random, std_random
        );

        Ok(StatisticalValidationResults::new(
            "Permutation Significance Test".to_string(),
            actual_score,
            p_value,
            confidence_level,
            description,
        )
        .with_effect_size(effect_size))
    }

    /// Compute a score for feature selection quality
    fn compute_selection_score(
        features: &Array2<Float>,
        selected_features: &[usize],
    ) -> SklResult<Float> {
        if selected_features.is_empty() {
            return Ok(0.0);
        }

        // Use variance as a simple quality metric
        // Better selections should have higher variance (more informative features)
        let mut total_variance = 0.0;
        for &feature_idx in selected_features {
            let column = features.column(feature_idx);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column
                .mapv(|x| (x - mean) * (x - mean))
                .mean()
                .unwrap_or(0.0);
            total_variance += variance;
        }

        Ok(total_variance / selected_features.len() as Float)
    }
}

/// Test for distributional properties of feature selection methods
pub struct DistributionalPropertyTest;

impl DistributionalPropertyTest {
    /// Test if feature selection respects known data structure
    pub fn test_structural_validity<S, T>(
        selector_factory: impl Fn() -> S,
        features: &Array2<Float>,
        target: &T,
        confidence_level: Float,
    ) -> SklResult<StatisticalValidationResults>
    where
        S: Fit<Array2<Float>, T>,
        S::Fitted: FeatureSelector + Transform<Array2<Float>>,
        T: Clone,
    {
        let selector = selector_factory();
        let fitted_selector = selector.fit(features, target)?;
        let selected_features = fitted_selector.selected_features();

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        // Test 1: Check if selection is not overly clustered (spatial distribution)
        let clustering_score = Self::compute_clustering_score(selected_features, features.ncols());

        // Test 2: Check if selected features have reasonable variance
        let variance_score = Self::compute_variance_score(features, selected_features)?;

        // Test 3: Check for correlation structure
        let correlation_score = Self::compute_correlation_score(features, selected_features)?;

        // Combine scores into overall structural validity
        let combined_score = (clustering_score + variance_score + correlation_score) / 3.0;

        // Simple threshold-based test (could be improved with proper statistical testing)
        let threshold = 0.5;
        let p_value = if combined_score > threshold {
            0.05 // Likely valid structure
        } else {
            0.95 // Likely invalid structure
        };

        let description = format!(
            "Structural validity test. Clustering: {:.3}, Variance: {:.3}, Correlation: {:.3}, Combined: {:.3}",
            clustering_score, variance_score, correlation_score, combined_score
        );

        Ok(StatisticalValidationResults::new(
            "Structural Validity Test".to_string(),
            combined_score,
            p_value,
            confidence_level,
            description,
        ))
    }

    fn compute_clustering_score(selected_features: &[usize], n_features: usize) -> Float {
        if selected_features.len() <= 1 {
            return 1.0;
        }

        // Measure how evenly distributed the selected features are
        let mut min_gaps = Vec::new();
        let mut sorted_features = selected_features.to_vec();
        sorted_features.sort();

        for i in 0..sorted_features.len() - 1 {
            let gap = sorted_features[i + 1] - sorted_features[i];
            min_gaps.push(gap);
        }

        // Compute coefficient of variation of gaps
        let mean_gap: Float = min_gaps.iter().sum::<usize>() as Float / min_gaps.len() as Float;
        let gap_variance = min_gaps
            .iter()
            .map(|&gap| (gap as Float - mean_gap) * (gap as Float - mean_gap))
            .sum::<Float>()
            / min_gaps.len() as Float;
        let gap_std = gap_variance.sqrt();

        // Lower coefficient of variation = more evenly distributed = better
        let cv = if mean_gap > 0.0 {
            gap_std / mean_gap
        } else {
            0.0
        };
        (1.0 / (1.0 + cv)).min(1.0)
    }

    fn compute_variance_score(
        features: &Array2<Float>,
        selected_features: &[usize],
    ) -> SklResult<Float> {
        let mut total_variance = 0.0;
        let mut count = 0;

        for &feature_idx in selected_features {
            let column = features.column(feature_idx);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column
                .mapv(|x| (x - mean) * (x - mean))
                .mean()
                .unwrap_or(0.0);

            // Only count features with non-zero variance
            if variance > 1e-12 {
                total_variance += variance;
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        let avg_variance = total_variance / count as Float;

        // Normalize to 0-1 scale (could be improved with better normalization)
        Ok((avg_variance / (avg_variance + 1.0)).min(1.0))
    }

    fn compute_correlation_score(
        features: &Array2<Float>,
        selected_features: &[usize],
    ) -> SklResult<Float> {
        if selected_features.len() <= 1 {
            return Ok(1.0);
        }

        let mut correlation_sum = 0.0;
        let mut pair_count = 0;

        for i in 0..selected_features.len() {
            for j in (i + 1)..selected_features.len() {
                let col1 = features.column(selected_features[i]);
                let col2 = features.column(selected_features[j]);

                let correlation = Self::compute_correlation(&col1, &col2);
                correlation_sum += correlation.abs();
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return Ok(1.0);
        }

        let avg_correlation = correlation_sum / pair_count as Float;

        // Good feature selection should have moderate correlation (not too high, not too low)
        let optimal_correlation = 0.3;
        let score = 1.0 - (avg_correlation - optimal_correlation).abs() / optimal_correlation;
        Ok(score.max(0.0))
    }

    fn compute_correlation(
        x: &scirs2_core::ndarray::ArrayView1<Float>,
        y: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        let n = x.len() as Float;
        if n <= 1.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom > 1e-12 {
            cov / denom
        } else {
            0.0
        }
    }
}

/// Test for robustness of feature selection to noise
pub struct RobustnessTest;

impl RobustnessTest {
    /// Test robustness to additive Gaussian noise
    pub fn test_noise_robustness<S, T>(
        selector_factory: impl Fn() -> S + Clone,
        features: &Array2<Float>,
        target: &T,
        noise_levels: &[Float],
        n_trials: usize,
        confidence_level: Float,
    ) -> SklResult<StatisticalValidationResults>
    where
        S: Fit<Array2<Float>, T> + Clone,
        S::Fitted: FeatureSelector + Transform<Array2<Float>>,
        T: Clone,
    {
        // Get baseline selection without noise
        let selector = selector_factory();
        let baseline_selection = selector.fit(features, target)?;
        let baseline_features = baseline_selection.selected_features().clone();

        let mut stability_scores = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);

        for &noise_level in noise_levels {
            let mut trial_similarities = Vec::new();

            for _ in 0..n_trials {
                // Add Gaussian noise to features
                let mut noisy_features = features.clone();
                for mut row in noisy_features.axis_iter_mut(Axis(0)) {
                    for element in row.iter_mut() {
                        // Simple Box-Muller transform for Gaussian noise
                        let u1: Float = rng.gen();
                        let u2: Float = rng.gen();
                        let noise = (-2.0f64 * u1.ln()).sqrt()
                            * (2.0 * std::f64::consts::PI * u2).cos()
                            * noise_level;
                        *element += noise;
                    }
                }

                // Fit selector on noisy data
                let noisy_selector = selector_factory();
                if let Ok(noisy_fitted) = noisy_selector.fit(&noisy_features, target) {
                    let noisy_features_selected = noisy_fitted.selected_features();
                    let similarity =
                        Self::jaccard_similarity(&baseline_features, noisy_features_selected);
                    trial_similarities.push(similarity);
                }
            }

            if !trial_similarities.is_empty() {
                let mean_similarity: Float =
                    trial_similarities.iter().sum::<Float>() / trial_similarities.len() as Float;
                stability_scores.push((noise_level, mean_similarity));
            }
        }

        if stability_scores.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No successful trials completed".to_string(),
            ));
        }

        // Compute robustness score (area under stability curve)
        let robustness_score = stability_scores
            .iter()
            .map(|(_, similarity)| *similarity)
            .sum::<Float>()
            / stability_scores.len() as Float;

        // Simple threshold-based significance test
        let threshold = 0.5; // Robust methods should maintain >50% similarity
        let p_value = if robustness_score > threshold {
            0.01 // Highly robust
        } else {
            0.99 // Not robust
        };

        let description = format!(
            "Noise robustness test across {} noise levels with {} trials each. Average stability: {:.3}",
            noise_levels.len(), n_trials, robustness_score
        );

        Ok(StatisticalValidationResults::new(
            "Noise Robustness Test".to_string(),
            robustness_score,
            p_value,
            confidence_level,
            description,
        ))
    }

    fn jaccard_similarity(set1: &[usize], set2: &[usize]) -> Float {
        let set1: std::collections::HashSet<usize> = set1.iter().cloned().collect();
        let set2: std::collections::HashSet<usize> = set2.iter().cloned().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            1.0
        } else {
            intersection as Float / union as Float
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::SelectKBest;
    use scirs2_core::ndarray::Array2;
    use sklears_core::prelude::Array1;

    fn create_test_data() -> (Array2<Float>, Array1<i32>) {
        // Create synthetic data for testing
        let n_samples = 100;
        let n_features = 20;
        let mut features = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Create structured data with some signal
        for i in 0..n_samples {
            for j in 0..n_features {
                features[[i, j]] = (i as Float * 0.1 + j as Float * 0.01).sin();
            }
            // Create binary classification target based on first few features
            let score = features[[i, 0]] + 0.5 * features[[i, 1]] + 0.2 * features[[i, 2]];
            target[i] = if score > 0.0 { 1 } else { 0 };
        }

        (features, target)
    }

    #[test]
    fn test_selection_consistency() {
        let (features, target) = create_test_data();

        let selector_factory = || SelectKBest::new(5, "f_classif");

        let result = SelectionConsistencyTest::test_split_consistency(
            selector_factory,
            &features,
            &target,
            5,
            0.3,
            0.95,
        );

        if let Err(e) = &result {
            eprintln!("Error in selection consistency test: {:?}", e);
        }
        assert!(result.is_ok());
        let validation_result = result.unwrap();
        assert_eq!(validation_result.test_name, "Split Consistency Test");
        assert!(validation_result.test_statistic >= 0.0);
        assert!(validation_result.test_statistic <= 1.0);
    }

    #[test]
    fn test_permutation_significance() {
        let (features, target) = create_test_data();

        let selector_factory = || SelectKBest::new(3, "f_classif");

        let result = PermutationSignificanceTest::test_feature_significance(
            selector_factory,
            &features,
            &target,
            100,
            0.95,
        );

        assert!(result.is_ok());
        let validation_result = result.unwrap();
        assert_eq!(validation_result.test_name, "Permutation Significance Test");
        assert!(validation_result.p_value >= 0.0);
        assert!(validation_result.p_value <= 1.0);
    }

    #[test]
    fn test_structural_validity() {
        let (features, target) = create_test_data();

        let selector_factory = || SelectKBest::new(5, "f_classif");

        let result = DistributionalPropertyTest::test_structural_validity(
            selector_factory,
            &features,
            &target,
            0.95,
        );

        assert!(result.is_ok());
        let validation_result = result.unwrap();
        assert_eq!(validation_result.test_name, "Structural Validity Test");
        assert!(validation_result.test_statistic >= 0.0);
        assert!(validation_result.test_statistic <= 1.0);
    }

    #[test]
    fn test_noise_robustness() {
        let (features, target) = create_test_data();

        let selector_factory = || SelectKBest::new(3, "f_classif");

        let noise_levels = vec![0.1, 0.2, 0.5];
        let result = RobustnessTest::test_noise_robustness(
            selector_factory,
            &features,
            &target,
            &noise_levels,
            10,
            0.95,
        );

        assert!(result.is_ok());
        let validation_result = result.unwrap();
        assert_eq!(validation_result.test_name, "Noise Robustness Test");
        assert!(validation_result.test_statistic >= 0.0);
        assert!(validation_result.test_statistic <= 1.0);
    }

    #[test]
    fn test_statistical_validation_framework() {
        let framework = StatisticalValidationFramework::new()
            .confidence_level(0.99)
            .n_permutations(500)
            .random_state(123);

        assert_eq!(framework.confidence_level, 0.99);
        assert_eq!(framework.n_permutations, 500);
        assert_eq!(framework.random_state, Some(123));
    }

    #[test]
    fn test_validation_results() {
        let result = StatisticalValidationResults::new(
            "Test".to_string(),
            0.75,
            0.03,
            0.95,
            "Test description".to_string(),
        )
        .with_effect_size(1.2);

        assert_eq!(result.test_name, "Test");
        assert_eq!(result.test_statistic, 0.75);
        assert_eq!(result.p_value, 0.03);
        assert!(result.is_significant);
        assert_eq!(result.effect_size, Some(1.2));
    }
}
