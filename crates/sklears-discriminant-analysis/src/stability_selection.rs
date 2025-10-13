//! # Stability-Based Feature Selection
//!
//! Stability-based feature selection uses resampling techniques to identify stable features
//! that are consistently selected across different random subsamples of the data. This approach
//! helps reduce overfitting and improves the reliability of feature selection results.
//!
//! ## Key Features
//! - Randomized Lasso and Elastic Net for stability selection
//! - Bootstrap-based feature stability estimation
//! - False Discovery Rate (FDR) control mechanisms
//! - Support for different base selectors (LDA, QDA, univariate tests)
//! - Stability path visualization support
//! - Threshold selection based on stability scores
//!
//! ## Algorithm
//! 1. Generate multiple random subsamples of the data
//! 2. Apply feature selection to each subsample
//! 3. Compute selection frequency for each feature
//! 4. Select features that exceed a stability threshold

use crate::feature_ranking::{DiscriminantFeatureRanking, DiscriminantFeatureRankingConfig};
use crate::lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig};
use crate::qda::QuadraticDiscriminantAnalysisConfig;
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Transform},
    types::Float,
};

/// Configuration for stability-based feature selection
#[derive(Debug, Clone)]
pub struct StabilitySelectionConfig {
    /// Base feature selector method ("lasso", "elastic_net", "univariate", "lda_ranking")
    pub base_selector: String,
    /// Number of bootstrap/subsampling iterations
    pub n_bootstrap: usize,
    /// Subsample size as fraction of total samples (0.0 to 1.0)
    pub subsample_size: Float,
    /// Stability threshold for feature selection (0.0 to 1.0)
    pub stability_threshold: Float,
    /// Regularization parameter for Lasso/Elastic Net
    pub alpha: Float,
    /// Elastic net mixing parameter (0.0 = Ridge, 1.0 = Lasso)
    pub l1_ratio: Float,
    /// False discovery rate control
    pub fdr: Option<Float>,
    /// Maximum number of features to select
    pub max_features: Option<usize>,
    /// Minimum number of features to select
    pub min_features: usize,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Whether to use replacement in bootstrap
    pub bootstrap_with_replacement: bool,
    /// LDA configuration for LDA-based selection
    pub lda_config: LinearDiscriminantAnalysisConfig,
    /// QDA configuration for QDA-based selection
    pub qda_config: QuadraticDiscriminantAnalysisConfig,
    /// Feature ranking configuration for univariate selection
    pub ranking_config: DiscriminantFeatureRankingConfig,
}

impl Default for StabilitySelectionConfig {
    fn default() -> Self {
        Self {
            base_selector: "lasso".to_string(),
            n_bootstrap: 100,
            subsample_size: 0.6,
            stability_threshold: 0.6,
            alpha: 1.0,
            l1_ratio: 1.0,
            fdr: None,
            max_features: None,
            min_features: 1,
            random_state: None,
            bootstrap_with_replacement: true,
            lda_config: LinearDiscriminantAnalysisConfig::default(),
            qda_config: QuadraticDiscriminantAnalysisConfig::default(),
            ranking_config: DiscriminantFeatureRankingConfig::default(),
        }
    }
}

/// Stability selection result for a single feature
#[derive(Debug, Clone)]
pub struct FeatureStability {
    /// Feature index
    pub feature_idx: usize,
    /// Selection frequency across bootstrap iterations
    pub stability_score: Float,
    /// Whether this feature is selected based on stability threshold
    pub selected: bool,
    /// Rank based on stability score (1-based, lower is better)
    pub rank: usize,
}

/// Stability-based feature selection estimator
#[derive(Debug, Clone)]
pub struct StabilitySelection {
    config: StabilitySelectionConfig,
}

impl StabilitySelection {
    /// Create a new stability selection instance
    pub fn new() -> Self {
        Self {
            config: StabilitySelectionConfig::default(),
        }
    }

    /// Set the base selector method
    pub fn base_selector(mut self, selector: &str) -> Self {
        self.config.base_selector = selector.to_string();
        self
    }

    /// Set the number of bootstrap iterations
    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.config.n_bootstrap = n_bootstrap;
        self
    }

    /// Set the subsample size
    pub fn subsample_size(mut self, size: Float) -> Self {
        self.config.subsample_size = size;
        self
    }

    /// Set the stability threshold
    pub fn stability_threshold(mut self, threshold: Float) -> Self {
        self.config.stability_threshold = threshold;
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set the elastic net mixing parameter
    pub fn l1_ratio(mut self, l1_ratio: Float) -> Self {
        self.config.l1_ratio = l1_ratio;
        self
    }

    /// Set false discovery rate control
    pub fn fdr(mut self, fdr: Option<Float>) -> Self {
        self.config.fdr = fdr;
        self
    }

    /// Set maximum number of features
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }

    /// Set minimum number of features
    pub fn min_features(mut self, min_features: usize) -> Self {
        self.config.min_features = min_features;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set bootstrap replacement option
    pub fn bootstrap_with_replacement(mut self, with_replacement: bool) -> Self {
        self.config.bootstrap_with_replacement = with_replacement;
        self
    }

    /// Generate bootstrap subsample indices
    fn generate_subsample_indices(&self, n_samples: usize, iteration: usize) -> Vec<usize> {
        let subsample_size = (n_samples as Float * self.config.subsample_size).round() as usize;
        let subsample_size = subsample_size.max(self.config.min_features).min(n_samples);

        // Simple deterministic pseudo-random generation
        let seed = self.config.random_state.unwrap_or(0) + iteration as u64;

        if self.config.bootstrap_with_replacement {
            // Bootstrap with replacement using simple linear congruential generator
            let mut state = seed;
            (0..subsample_size)
                .map(|_| {
                    state = (state.wrapping_mul(1103515245).wrapping_add(12345)) % (1u64 << 31);
                    (state as usize) % n_samples
                })
                .collect()
        } else {
            // Subsampling without replacement using Fisher-Yates-like approach
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut state = seed;

            // Simple shuffle using linear congruential generator
            for i in 0..subsample_size {
                state = (state.wrapping_mul(1103515245).wrapping_add(12345)) % (1u64 << 31);
                let j = i + ((state as usize) % (n_samples - i));
                indices.swap(i, j);
            }

            indices.into_iter().take(subsample_size).collect()
        }
    }

    /// Apply feature selection using Lasso regularization
    fn select_features_lasso(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Vec<usize>> {
        // Simplified Lasso implementation using coordinate descent
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Convert classes to continuous targets for regression-like approach
        let unique_classes: Vec<i32> = {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        if unique_classes.len() < 2 {
            return Ok(Vec::new());
        }

        // For binary classification, convert to 0/1
        let y_continuous = if unique_classes.len() == 2 {
            y.mapv(|label| if label == unique_classes[0] { 0.0 } else { 1.0 })
        } else {
            // For multiclass, use one-vs-rest approach for the first class
            y.mapv(|label| if label == unique_classes[0] { 1.0 } else { 0.0 })
        };

        // Normalize features
        let mut x_norm = x.clone();
        for j in 0..n_features {
            let mut col = x_norm.column_mut(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.var(0.0).sqrt().max(1e-8);
            col.map_inplace(|val| {
                *val = (*val - mean) / std;
            });
        }

        // Initialize coefficients
        let mut beta = Array1::zeros(n_features);
        let max_iter = 100;
        let tol = 1e-6;

        // Coordinate descent
        for _iter in 0..max_iter {
            let mut max_change: Float = 0.0;

            for j in 0..n_features {
                let old_beta_j = beta[j];

                // Compute residuals excluding current feature
                let mut residual = y_continuous.clone();
                for k in 0..n_features {
                    if k != j {
                        let x_k = x_norm.column(k);
                        residual = residual - beta[k] * &x_k;
                    }
                }

                // Compute correlation with residual
                let x_j = x_norm.column(j);
                let rho = x_j.dot(&residual) / n_samples as Float;

                // Soft thresholding
                let lambda = self.config.alpha;
                beta[j] = if rho > lambda {
                    rho - lambda
                } else if rho < -lambda {
                    rho + lambda
                } else {
                    0.0
                };

                let change = (beta[j] - old_beta_j).abs();
                max_change = max_change.max(change);
            }

            if max_change < tol {
                break;
            }
        }

        // Select non-zero coefficients
        let selected_features: Vec<usize> = beta
            .iter()
            .enumerate()
            .filter(|(_, &coef)| coef.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();

        Ok(selected_features)
    }

    /// Apply feature selection using Elastic Net regularization
    fn select_features_elastic_net(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Vec<usize>> {
        // Similar to Lasso but with L2 penalty
        let n_features = x.ncols();
        let n_samples = x.nrows();

        let unique_classes: Vec<i32> = {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        if unique_classes.len() < 2 {
            return Ok(Vec::new());
        }

        let y_continuous = if unique_classes.len() == 2 {
            y.mapv(|label| if label == unique_classes[0] { 0.0 } else { 1.0 })
        } else {
            y.mapv(|label| if label == unique_classes[0] { 1.0 } else { 0.0 })
        };

        // Normalize features
        let mut x_norm = x.clone();
        for j in 0..n_features {
            let mut col = x_norm.column_mut(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.var(0.0).sqrt().max(1e-8);
            col.map_inplace(|val| {
                *val = (*val - mean) / std;
            });
        }

        let mut beta = Array1::zeros(n_features);
        let max_iter = 100;
        let tol = 1e-6;

        // Coordinate descent with elastic net
        for _iter in 0..max_iter {
            let mut max_change: Float = 0.0;

            for j in 0..n_features {
                let old_beta_j = beta[j];

                let mut residual = y_continuous.clone();
                for k in 0..n_features {
                    if k != j {
                        let x_k = x_norm.column(k);
                        residual = residual - beta[k] * &x_k;
                    }
                }

                let x_j = x_norm.column(j);
                let rho = x_j.dot(&residual) / n_samples as Float;

                // Elastic net soft thresholding
                let l1_penalty = self.config.alpha * self.config.l1_ratio;
                let l2_penalty = self.config.alpha * (1.0 - self.config.l1_ratio);
                let denominator = 1.0 + l2_penalty;

                beta[j] = if rho > l1_penalty {
                    (rho - l1_penalty) / denominator
                } else if rho < -l1_penalty {
                    (rho + l1_penalty) / denominator
                } else {
                    0.0
                };

                let change = (beta[j] - old_beta_j).abs();
                max_change = max_change.max(change);
            }

            if max_change < tol {
                break;
            }
        }

        let selected_features: Vec<usize> = beta
            .iter()
            .enumerate()
            .filter(|(_, &coef)| coef.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();

        Ok(selected_features)
    }

    /// Apply feature selection using univariate tests
    fn select_features_univariate(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Vec<usize>> {
        let mut ranking_config = self.config.ranking_config.clone();
        ranking_config.k = Some(x.ncols() / 2); // Select top half features

        let ranker = DiscriminantFeatureRanking::new();
        let fitted_ranker = ranker.fit(x, y)?;

        Ok(fitted_ranker.selected_features().to_vec())
    }

    /// Apply feature selection using LDA ranking
    fn select_features_lda_ranking(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Vec<usize>> {
        // Use LDA transform to get feature importance
        let mut lda_config = self.config.lda_config.clone();
        lda_config.n_components = Some(1);

        let lda = LinearDiscriminantAnalysis::new();
        let fitted_lda = lda.fit(x, y)?;

        // Get LDA components (feature weights)
        let components = fitted_lda.components();
        if components.nrows() == 0 {
            return Ok(Vec::new());
        }

        // Use absolute values of the first discriminant component
        let feature_importance = components.row(0).mapv(|x| x.abs());

        // Select top features based on importance
        let n_select = x.ncols() / 2;
        let mut feature_scores: Vec<(usize, Float)> = feature_importance
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(feature_scores
            .into_iter()
            .take(n_select)
            .map(|(i, _)| i)
            .collect())
    }

    /// Apply base feature selector to a subsample
    fn apply_base_selector(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Vec<usize>> {
        match self.config.base_selector.as_str() {
            "lasso" => self.select_features_lasso(x, y),
            "elastic_net" => self.select_features_elastic_net(x, y),
            "univariate" => self.select_features_univariate(x, y),
            "lda_ranking" => self.select_features_lda_ranking(x, y),
            _ => Err(SklearsError::InvalidParameter {
                name: "base_selector".to_string(),
                reason: format!("Unknown base selector: {}", self.config.base_selector),
            }),
        }
    }

    /// Compute stability scores for features
    fn compute_stability_scores(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Vec<FeatureStability>> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Track selection frequency for each feature
        let mut selection_counts = vec![0; n_features];
        let mut valid_iterations = 0;

        for iteration in 0..self.config.n_bootstrap {
            // Generate subsample
            let subsample_indices = self.generate_subsample_indices(n_samples, iteration);

            // Extract subsample
            let x_subsample = x.select(Axis(0), &subsample_indices);
            let y_subsample = y.select(Axis(0), &subsample_indices);

            // Apply base selector
            match self.apply_base_selector(&x_subsample, &y_subsample) {
                Ok(selected_features) => {
                    valid_iterations += 1;
                    for &feature_idx in &selected_features {
                        if feature_idx < n_features {
                            selection_counts[feature_idx] += 1;
                        }
                    }
                }
                Err(_) => {
                    // Skip failed iterations
                    continue;
                }
            }
        }

        if valid_iterations == 0 {
            return Err(SklearsError::InvalidInput(
                "No valid iterations in stability selection".to_string(),
            ));
        }

        // Compute stability scores (selection frequency)
        let mut feature_stabilities: Vec<FeatureStability> = selection_counts
            .into_iter()
            .enumerate()
            .map(|(feature_idx, count)| {
                let stability_score = count as Float / valid_iterations as Float;
                let selected = stability_score >= self.config.stability_threshold;
                FeatureStability {
                    feature_idx,
                    stability_score,
                    selected,
                    rank: 0, // Will be filled later
                }
            })
            .collect();

        // Sort by stability score and assign ranks
        feature_stabilities.sort_by(|a, b| {
            b.stability_score
                .partial_cmp(&a.stability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, feature_stability) in feature_stabilities.iter_mut().enumerate() {
            feature_stability.rank = rank + 1;
        }

        // Apply additional constraints
        if let Some(max_features) = self.config.max_features {
            // Select only top max_features
            for (i, feature_stability) in feature_stabilities.iter_mut().enumerate() {
                if i >= max_features {
                    feature_stability.selected = false;
                }
            }
        }

        // Ensure minimum features are selected
        let n_selected = feature_stabilities.iter().filter(|f| f.selected).count();
        if n_selected < self.config.min_features {
            // Select top min_features regardless of threshold
            for feature_stability in feature_stabilities
                .iter_mut()
                .take(self.config.min_features)
            {
                feature_stability.selected = true;
            }
        }

        // Apply FDR control if requested
        if let Some(fdr) = self.config.fdr {
            self.apply_fdr_control(&mut feature_stabilities, fdr);
        }

        // Sort back by feature index for consistency
        feature_stabilities.sort_by_key(|f| f.feature_idx);

        Ok(feature_stabilities)
    }

    /// Apply False Discovery Rate control using Benjamini-Hochberg procedure
    fn apply_fdr_control(&self, feature_stabilities: &mut [FeatureStability], fdr: Float) {
        // Sort by stability score (descending)
        feature_stabilities.sort_by(|a, b| {
            b.stability_score
                .partial_cmp(&a.stability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_features = feature_stabilities.len();
        let mut n_selected = 0;

        // Benjamini-Hochberg procedure
        for (i, feature_stability) in feature_stabilities.iter_mut().enumerate() {
            let expected_fdr = (i + 1) as Float * fdr / n_features as Float;
            if feature_stability.stability_score >= expected_fdr {
                feature_stability.selected = true;
                n_selected = i + 1;
            } else {
                break;
            }
        }

        // Unselect remaining features
        for feature_stability in feature_stabilities.iter_mut().skip(n_selected) {
            feature_stability.selected = false;
        }
    }
}

impl Default for StabilitySelection {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StabilitySelection {
    type Config = StabilitySelectionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Trained stability selection model
#[derive(Debug)]
pub struct TrainedStabilitySelection {
    /// Feature stability information
    feature_stabilities: Vec<FeatureStability>,
    /// Selected feature indices
    selected_features: Vec<usize>,
    /// Original number of features
    n_features: usize,
    /// Configuration
    config: StabilitySelectionConfig,
}

impl TrainedStabilitySelection {
    /// Get feature stability information
    pub fn feature_stabilities(&self) -> &[FeatureStability] {
        &self.feature_stabilities
    }

    /// Get selected feature indices
    pub fn selected_features(&self) -> &[usize] {
        &self.selected_features
    }

    /// Get stability score for a specific feature
    pub fn stability_score(&self, feature_idx: usize) -> Option<Float> {
        self.feature_stabilities
            .iter()
            .find(|f| f.feature_idx == feature_idx)
            .map(|f| f.stability_score)
    }

    /// Get the number of selected features
    pub fn n_selected_features(&self) -> usize {
        self.selected_features.len()
    }

    /// Get boolean mask of selected features
    pub fn support(&self) -> Array1<bool> {
        let mut support = Array1::from_elem(self.n_features, false);
        for &feature_idx in &self.selected_features {
            if feature_idx < self.n_features {
                support[feature_idx] = true;
            }
        }
        support
    }

    /// Get stability path data for visualization
    pub fn stability_path(&self) -> Vec<(usize, Float)> {
        self.feature_stabilities
            .iter()
            .map(|f| (f.feature_idx, f.stability_score))
            .collect()
    }
}

impl Fit<Array2<Float>, Array1<i32>> for StabilitySelection {
    type Fitted = TrainedStabilitySelection;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let n_features = x.ncols();

        // Compute stability scores
        let feature_stabilities = self.compute_stability_scores(x, y)?;

        // Extract selected features
        let selected_features: Vec<usize> = feature_stabilities
            .iter()
            .filter(|f| f.selected)
            .map(|f| f.feature_idx)
            .collect();

        Ok(TrainedStabilitySelection {
            feature_stabilities,
            selected_features,
            n_features,
            config: self.config.clone(),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedStabilitySelection {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        if self.selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected".to_string(),
            ));
        }

        Ok(x.select(Axis(1), &self.selected_features))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_stability_selection_basic() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let stability_selector = StabilitySelection::new()
            .n_bootstrap(10)
            .stability_threshold(0.3);

        let fitted = stability_selector.fit(&x, &y).unwrap();

        assert!(fitted.n_selected_features() > 0);
        assert!(fitted.n_selected_features() <= 4);
        assert_eq!(fitted.feature_stabilities().len(), 4);
    }

    #[test]
    fn test_stability_selection_transform() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0]
        ];
        let y = array![0, 0, 1, 1];

        let stability_selector = StabilitySelection::new()
            .n_bootstrap(5)
            .stability_threshold(0.2);

        let fitted = stability_selector.fit(&x, &y).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 4);
        assert!(x_transformed.ncols() <= 5);
        assert!(x_transformed.ncols() > 0);
    }

    #[test]
    fn test_stability_selection_different_selectors() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let selectors = ["lasso", "elastic_net", "univariate"];
        for selector in &selectors {
            let stability_selector = StabilitySelection::new()
                .base_selector(selector)
                .n_bootstrap(5)
                .stability_threshold(0.2);

            let fitted = stability_selector.fit(&x, &y).unwrap();
            assert!(fitted.n_selected_features() > 0);
        }
    }

    #[test]
    fn test_stability_scores() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let stability_selector = StabilitySelection::new()
            .n_bootstrap(10)
            .stability_threshold(0.3);

        let fitted = stability_selector.fit(&x, &y).unwrap();

        // Check that stability scores are between 0 and 1
        for stability in fitted.feature_stabilities() {
            assert!(stability.stability_score >= 0.0);
            assert!(stability.stability_score <= 1.0);
        }
    }

    #[test]
    fn test_max_features_constraint() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0]
        ];
        let y = array![0, 0, 1, 1];

        let stability_selector = StabilitySelection::new()
            .n_bootstrap(5)
            .stability_threshold(0.1) // Low threshold to select many features
            .max_features(Some(2));

        let fitted = stability_selector.fit(&x, &y).unwrap();
        assert!(fitted.n_selected_features() <= 2);
    }

    #[test]
    fn test_min_features_constraint() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let stability_selector = StabilitySelection::new()
            .n_bootstrap(5)
            .stability_threshold(0.9) // High threshold to select few features
            .min_features(2);

        let fitted = stability_selector.fit(&x, &y).unwrap();
        assert!(fitted.n_selected_features() >= 2);
    }

    #[test]
    fn test_support_mask() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let stability_selector = StabilitySelection::new()
            .n_bootstrap(5)
            .stability_threshold(0.3);

        let fitted = stability_selector.fit(&x, &y).unwrap();
        let support = fitted.support();

        assert_eq!(support.len(), 4);

        // Check that support mask matches selected features
        let n_selected_from_support = support.iter().filter(|&&selected| selected).count();
        assert_eq!(n_selected_from_support, fitted.n_selected_features());
    }
}
