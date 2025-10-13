//! Feature importance computation for tree-based models
//!
//! This module provides various methods for computing feature importance scores
//! including permutation-based, gain-based, and other importance metrics.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use sklears_core::error::Result;

/// Different methods for computing feature importance
#[derive(Debug, Clone, Copy)]
pub enum FeatureImportanceMethod {
    /// Permutation-based importance (model-agnostic)
    Permutation,
    /// Gain-based importance (tree-specific)
    Gain,
    /// Split-based importance (tree-specific)
    Split,
    /// Cover-based importance (tree-specific)
    Cover,
}

/// Feature importance scores with metadata
#[derive(Debug, Clone)]
pub struct FeatureImportanceScores {
    /// Importance scores for each feature
    pub scores: Array1<f64>,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Method used to compute importance
    pub method: FeatureImportanceMethod,
    /// Standard deviation of importance scores (for permutation-based)
    pub std_dev: Option<Array1<f64>>,
}

impl FeatureImportanceScores {
    /// Create new feature importance scores
    pub fn new(
        scores: Array1<f64>,
        method: FeatureImportanceMethod,
        feature_names: Option<Vec<String>>,
        std_dev: Option<Array1<f64>>,
    ) -> Self {
        Self {
            scores,
            feature_names,
            method,
            std_dev,
        }
    }

    /// Get the most important features
    pub fn top_features(&self, n: usize) -> Vec<(usize, f64)> {
        let mut indexed_scores: Vec<(usize, f64)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_scores.truncate(n);
        indexed_scores
    }

    /// Get feature names for the top features
    pub fn top_feature_names(&self, n: usize) -> Vec<String> {
        let top_features = self.top_features(n);

        if let Some(ref names) = self.feature_names {
            top_features
                .iter()
                .map(|(idx, _)| names[*idx].clone())
                .collect()
        } else {
            top_features
                .iter()
                .map(|(idx, _)| format!("feature_{}", idx))
                .collect()
        }
    }

    /// Normalize importance scores to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.scores.sum();
        if sum > 0.0 {
            self.scores /= sum;
        }
    }

    /// Get feature ranking (0-based indices sorted by importance)
    pub fn feature_ranking(&self) -> Vec<usize> {
        let mut indexed_scores: Vec<(usize, f64)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_scores.iter().map(|(idx, _)| *idx).collect()
    }
}

/// Trait for computing feature importance
pub trait FeatureImportance {
    /// Compute feature importance using the specified method
    fn compute_feature_importance(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        method: FeatureImportanceMethod,
    ) -> Result<FeatureImportanceScores>;
}

/// Gain-based feature importance computation for tree models
pub struct GainBasedImportance {
    /// Feature importance scores based on impurity decrease
    pub gain_scores: Vec<f64>,
    /// Number of times each feature was used for splitting
    pub split_counts: Vec<usize>,
    /// Total number of samples reaching splits for each feature
    pub sample_counts: Vec<usize>,
}

impl GainBasedImportance {
    /// Create a new gain-based importance calculator
    pub fn new(n_features: usize) -> Self {
        Self {
            gain_scores: vec![0.0; n_features],
            split_counts: vec![0; n_features],
            sample_counts: vec![0; n_features],
        }
    }

    /// Update importance scores from a tree split
    pub fn update_from_split(
        &mut self,
        feature_idx: usize,
        impurity_decrease: f64,
        n_samples: usize,
    ) {
        if feature_idx < self.gain_scores.len() {
            self.gain_scores[feature_idx] += impurity_decrease * n_samples as f64;
            self.split_counts[feature_idx] += 1;
            self.sample_counts[feature_idx] += n_samples;
        }
    }

    /// Get normalized feature importance scores
    pub fn get_normalized_scores(&self) -> Array1<f64> {
        let scores = Array1::from_vec(self.gain_scores.clone());
        let sum = scores.sum();
        if sum > 0.0 {
            scores / sum
        } else {
            Array1::zeros(self.gain_scores.len())
        }
    }

    /// Get feature importance scores based on different criteria
    pub fn get_scores(&self, method: FeatureImportanceMethod) -> Array1<f64> {
        match method {
            FeatureImportanceMethod::Gain => self.get_normalized_scores(),
            FeatureImportanceMethod::Split => {
                let scores =
                    Array1::from_vec(self.split_counts.iter().map(|&x| x as f64).collect());
                let sum = scores.sum();
                if sum > 0.0 {
                    scores / sum
                } else {
                    Array1::zeros(self.split_counts.len())
                }
            }
            FeatureImportanceMethod::Cover => {
                let scores =
                    Array1::from_vec(self.sample_counts.iter().map(|&x| x as f64).collect());
                let sum = scores.sum();
                if sum > 0.0 {
                    scores / sum
                } else {
                    Array1::zeros(self.sample_counts.len())
                }
            }
            _ => self.get_normalized_scores(),
        }
    }

    /// Reset all importance scores
    pub fn reset(&mut self) {
        self.gain_scores.fill(0.0);
        self.split_counts.fill(0);
        self.sample_counts.fill(0);
    }
}

/// Permutation-based feature importance computation
pub struct PermutationImportance {
    /// Number of permutations to perform
    pub n_permutations: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl PermutationImportance {
    /// Create a new permutation-based importance calculator
    pub fn new(n_permutations: usize, random_state: Option<u64>) -> Self {
        Self {
            n_permutations,
            random_state,
        }
    }

    /// Compute permutation-based feature importance
    pub fn compute_importance<M, F>(
        &self,
        model: &M,
        x: &Array2<f64>,
        y: &Array1<f64>,
        score_fn: F,
    ) -> Result<FeatureImportanceScores>
    where
        F: Fn(&M, &Array2<f64>, &Array1<f64>) -> Result<f64>,
    {
        let n_features = x.ncols();
        let mut importance_scores = vec![Vec::new(); n_features];

        // Get baseline score
        let baseline_score = score_fn(model, x, y)?;

        // Create RNG
        let mut rng: scirs2_core::random::Random = scirs2_core::random::thread_rng();

        // Compute importance for each feature
        for feature_idx in 0..n_features {
            for _ in 0..self.n_permutations {
                // Create permuted data
                let mut x_permuted = x.clone();
                let mut feature_values: Vec<f64> = x_permuted.column(feature_idx).to_vec();

                // Shuffle the feature values
                feature_values.shuffle(&mut rng);

                // Replace the feature column with shuffled values
                for (i, &value) in feature_values.iter().enumerate() {
                    x_permuted[[i, feature_idx]] = value;
                }

                // Compute score with permuted feature
                let permuted_score = score_fn(model, &x_permuted, y)?;

                // Importance is the decrease in performance
                let importance = baseline_score - permuted_score;
                importance_scores[feature_idx].push(importance);
            }
        }

        // Compute mean and std dev for each feature
        let mut mean_scores = Vec::new();
        let mut std_scores = Vec::new();

        for scores in importance_scores {
            if scores.is_empty() {
                mean_scores.push(0.0);
                std_scores.push(0.0);
            } else {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                let variance =
                    scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
                let std_dev = variance.sqrt();

                mean_scores.push(mean);
                std_scores.push(std_dev);
            }
        }

        Ok(FeatureImportanceScores::new(
            Array1::from_vec(mean_scores),
            FeatureImportanceMethod::Permutation,
            None,
            Some(Array1::from_vec(std_scores)),
        ))
    }
}

/// Feature interaction detection
pub struct FeatureInteractionDetector {
    /// Threshold for considering an interaction significant
    pub interaction_threshold: f64,
    /// Maximum number of feature pairs to consider
    pub max_pairs: Option<usize>,
}

impl FeatureInteractionDetector {
    /// Create a new feature interaction detector
    pub fn new(interaction_threshold: f64, max_pairs: Option<usize>) -> Self {
        Self {
            interaction_threshold,
            max_pairs,
        }
    }

    /// Detect feature interactions using correlation-based methods
    pub fn detect_interactions(&self, x: &Array2<f64>) -> Result<Vec<(usize, usize, f64)>> {
        let n_features = x.ncols();
        let mut interactions = Vec::new();

        // Compute pairwise correlations
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let corr = self.compute_correlation(x, i, j)?;
                if corr.abs() > self.interaction_threshold {
                    interactions.push((i, j, corr));
                }
            }
        }

        // Sort by absolute correlation
        interactions.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

        // Limit number of pairs if specified
        if let Some(max_pairs) = self.max_pairs {
            interactions.truncate(max_pairs);
        }

        Ok(interactions)
    }

    /// Compute Pearson correlation between two features
    fn compute_correlation(&self, x: &Array2<f64>, idx1: usize, idx2: usize) -> Result<f64> {
        let col1 = x.column(idx1);
        let col2 = x.column(idx2);

        let mean1 = col1.mean().unwrap();
        let mean2 = col2.mean().unwrap();

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for (&x1, &x2) in col1.iter().zip(col2.iter()) {
            let diff1 = x1 - mean1;
            let diff2 = x2 - mean2;

            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();

        if denominator.abs() < f64::EPSILON {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

/// Recursive feature elimination
pub struct RecursiveFeatureElimination {
    /// Number of features to select
    pub n_features_to_select: usize,
    /// Step size for feature elimination
    pub step: usize,
    /// Cross-validation folds
    pub cv: Option<usize>,
}

impl RecursiveFeatureElimination {
    /// Create a new recursive feature elimination selector
    pub fn new(n_features_to_select: usize, step: usize, cv: Option<usize>) -> Self {
        Self {
            n_features_to_select,
            step,
            cv,
        }
    }

    /// Perform recursive feature elimination
    pub fn select_features<M, F>(
        &self,
        model: &M,
        x: &Array2<f64>,
        y: &Array1<f64>,
        importance_fn: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(&M, &Array2<f64>, &Array1<f64>) -> Result<Array1<f64>>,
    {
        let n_features = x.ncols();
        let mut selected_features: Vec<usize> = (0..n_features).collect();

        while selected_features.len() > self.n_features_to_select {
            // Create subset with current features
            let x_subset = self.create_feature_subset(x, &selected_features)?;

            // Compute importance scores
            let importance_scores = importance_fn(model, &x_subset, y)?;

            // Find features to eliminate
            let n_to_eliminate = std::cmp::min(
                self.step,
                selected_features.len() - self.n_features_to_select,
            );
            let mut indexed_scores: Vec<(usize, f64)> = importance_scores
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();

            // Sort by importance (ascending) to eliminate least important
            indexed_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Remove least important features
            for i in 0..n_to_eliminate {
                let feature_idx = indexed_scores[i].0;
                selected_features.remove(feature_idx);
            }
        }

        Ok(selected_features)
    }

    /// Create a subset of features from the original dataset
    fn create_feature_subset(&self, x: &Array2<f64>, features: &[usize]) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = features.len();

        let mut subset = Array2::zeros((n_samples, n_features));

        for (new_idx, &old_idx) in features.iter().enumerate() {
            if old_idx < x.ncols() {
                subset.column_mut(new_idx).assign(&x.column(old_idx));
            }
        }

        Ok(subset)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_feature_importance_scores() {
        let scores = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let mut importance =
            FeatureImportanceScores::new(scores, FeatureImportanceMethod::Gain, None, None);

        // Test top features
        let top_features = importance.top_features(2);
        assert_eq!(top_features.len(), 2);
        assert_eq!(top_features[0].0, 0); // First feature has highest importance
        assert_eq!(top_features[1].0, 1); // Second feature has second highest

        // Test normalization
        importance.normalize();
        assert_abs_diff_eq!(importance.scores.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gain_based_importance() {
        let mut importance = GainBasedImportance::new(3);

        // Update with some splits
        importance.update_from_split(0, 0.5, 100);
        importance.update_from_split(1, 0.3, 80);
        importance.update_from_split(0, 0.2, 50);

        let scores = importance.get_scores(FeatureImportanceMethod::Gain);
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]); // Feature 0 should have higher importance
        assert_eq!(scores[2], 0.0); // Feature 2 was never used
    }

    #[test]
    fn test_feature_interaction_detector() {
        let detector = FeatureInteractionDetector::new(0.5, None);

        // Create test data with perfect correlation
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0, 6.0, 2.0, 4.0, 8.0, 3.0, 5.0, 10.0, 1.0, 6.0,
                12.0, 2.0, 7.0, 14.0, 3.0, 8.0, 16.0, 1.0, 9.0, 18.0, 2.0, 10.0, 20.0, 3.0,
            ],
        )
        .unwrap();

        let interactions = detector.detect_interactions(&x).unwrap();
        assert!(!interactions.is_empty());

        // Features 0 and 1 should be highly correlated
        let (i, j, corr) = interactions[0];
        assert!((i == 0 && j == 1) || (i == 1 && j == 0));
        assert!(corr.abs() > 0.9); // Should be highly correlated
    }

    #[test]
    fn test_recursive_feature_elimination() {
        let rfe = RecursiveFeatureElimination::new(2, 1, None);

        // Mock importance function that returns random importance
        let importance_fn = |_model: &(), _x: &Array2<f64>, _y: &Array1<f64>| {
            Ok(Array1::from_vec(vec![0.1, 0.8, 0.1]))
        };

        let x = Array2::ones((10, 3));
        let y = Array1::ones(10);

        let selected = rfe.select_features(&(), &x, &y, importance_fn).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&1)); // Feature 1 should be selected (highest importance)
    }
}
