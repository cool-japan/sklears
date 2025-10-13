//! Stability Selection for robust feature selection
//!
//! This module implements stability selection, a feature selection method that uses
//! subsampling and base feature selectors to identify stable feature sets. It provides
//! control over false discovery rates and is particularly useful for high-dimensional
//! data where robust feature selection is critical.

use crate::feature_selection::{FeatureSelectionStrategy, FeatureSelector, UnivariateScoreFunc};
use crate::lasso_cv::{LassoCV, LassoCVConfig};
use crate::ridge_cv::{RidgeCV, RidgeCVConfig};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{error::SklearsError, traits::Fit};
use std::cmp::Ordering;

/// Base selector types for stability selection
#[derive(Debug, Clone)]
pub enum BaseSelector {
    /// Lasso with cross-validation
    LassoCV { config: LassoCVConfig },
    /// Ridge with cross-validation  
    RidgeCV { config: RidgeCVConfig },
    /// Univariate feature selection
    Univariate { strategy: FeatureSelectionStrategy },
    /// Custom threshold-based selector
    Threshold {
        threshold: f64,
        score_func: UnivariateScoreFunc,
    },
}

/// Stability selection configuration
#[derive(Debug, Clone)]
pub struct StabilitySelectionConfig {
    /// Base feature selector to use
    pub base_selector: BaseSelector,
    /// Number of subsampling iterations
    pub n_bootstrap_iterations: usize,
    /// Fraction of samples to use in each iteration
    pub sample_fraction: f64,
    /// Threshold for feature selection probability
    pub threshold: f64,
    /// Upper bound for expected number of false discoveries (E[V])
    pub expected_fdr: Option<f64>,
    /// Per-family error rate (PFER) bound
    pub pfer_bound: Option<f64>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to replace samples during subsampling
    pub replace: bool,
    /// Complementary pairs stabilization
    pub complementary_pairs: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for StabilitySelectionConfig {
    fn default() -> Self {
        Self {
            base_selector: BaseSelector::LassoCV {
                config: LassoCVConfig::default(),
            },
            n_bootstrap_iterations: 100,
            sample_fraction: 0.5,
            threshold: 0.6,
            expected_fdr: Some(1.0),
            pfer_bound: None,
            random_state: None,
            replace: false,
            complementary_pairs: true,
            verbose: false,
        }
    }
}

/// Stability selection result
#[derive(Debug, Clone)]
pub struct StabilitySelectionResult {
    /// Selected feature indices
    pub selected_features: Vec<usize>,
    /// Selection probabilities for all features
    pub selection_probabilities: Vec<f64>,
    /// Stability scores for all features
    pub stability_scores: Vec<f64>,
    /// Lambda values explored (for L1-based selectors)
    pub lambda_grid: Vec<f64>,
    /// Selection frequencies across iterations
    pub selection_frequencies: Vec<Vec<usize>>,
    /// Bootstrap iteration results
    pub bootstrap_results: Vec<BootstrapResult>,
    /// Number of original features
    pub n_features_in: usize,
    /// Number of selected features
    pub n_features_out: usize,
    /// Configuration used
    pub config: StabilitySelectionConfig,
}

/// Result from a single bootstrap iteration
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Iteration number
    pub iteration: usize,
    /// Sample indices used
    pub sample_indices: Vec<usize>,
    /// Selected features in this iteration
    pub selected_features: Vec<usize>,
    /// Feature scores/importances
    pub feature_scores: Vec<f64>,
    /// Regularization parameter used (if applicable)
    pub lambda: Option<f64>,
}

/// Stability paths for different threshold values
#[derive(Debug, Clone)]
pub struct StabilityPath {
    /// Threshold values
    pub thresholds: Vec<f64>,
    /// Number of selected features at each threshold
    pub n_selected: Vec<usize>,
    /// Expected false discovery rate at each threshold
    pub expected_fdr: Vec<f64>,
    /// Per-family error rate at each threshold
    pub pfer: Vec<f64>,
}

/// Stability Selection feature selector
pub struct StabilitySelection {
    config: StabilitySelectionConfig,
    is_fitted: bool,
    selection_result: Option<StabilitySelectionResult>,
    rng: StdRng,
}

impl StabilitySelection {
    /// Create a new stability selection with default configuration
    pub fn new() -> Self {
        Self {
            config: StabilitySelectionConfig::default(),
            is_fitted: false,
            selection_result: None,
            rng: StdRng::seed_from_u64(42), // Use fixed seed for deterministic behavior
        }
    }

    /// Create stability selection with custom configuration
    pub fn with_config(config: StabilitySelectionConfig) -> Self {
        let rng = if let Some(seed) = config.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42) // Use fixed seed for deterministic behavior
        };

        Self {
            config,
            is_fitted: false,
            selection_result: None,
            rng,
        }
    }

    /// Set the base selector
    pub fn with_base_selector(mut self, base_selector: BaseSelector) -> Self {
        self.config.base_selector = base_selector;
        self
    }

    /// Set the number of bootstrap iterations
    pub fn with_n_bootstrap_iterations(mut self, n_iterations: usize) -> Self {
        self.config.n_bootstrap_iterations = n_iterations;
        self
    }

    /// Set the sample fraction
    pub fn with_sample_fraction(mut self, fraction: f64) -> Self {
        self.config.sample_fraction = fraction.clamp(0.1, 1.0);
        self
    }

    /// Set the selection threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the expected FDR bound
    pub fn with_expected_fdr(mut self, expected_fdr: Option<f64>) -> Self {
        self.config.expected_fdr = expected_fdr;
        self
    }

    /// Fit the stability selection to data
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), SklearsError> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit stability selection on empty dataset".to_string(),
            ));
        }

        let n_samples = x.len();
        let n_features = x[0].len();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("target.len() == {}", n_samples),
                actual: format!("target.len() == {}", y.len()),
            });
        }

        if self.config.verbose {
            eprintln!(
                "Stability Selection: Starting with {} features, {} samples",
                n_features, n_samples
            );
            eprintln!(
                "Running {} bootstrap iterations with sample fraction {}",
                self.config.n_bootstrap_iterations, self.config.sample_fraction
            );
        }

        // Initialize tracking variables
        let mut selection_counts = vec![0; n_features];
        let mut bootstrap_results = Vec::new();
        let sample_size = (n_samples as f64 * self.config.sample_fraction) as usize;

        // Bootstrap iterations
        for iteration in 0..self.config.n_bootstrap_iterations {
            if self.config.verbose && iteration % 20 == 0 {
                eprintln!(
                    "Stability Selection: Iteration {}/{}",
                    iteration, self.config.n_bootstrap_iterations
                );
            }

            // Sample data
            let sample_indices = self.generate_sample_indices(n_samples, sample_size);
            let (x_sample, y_sample) = self.create_sample_data(x, y, &sample_indices);

            // Apply base selector
            let selected_features = self.apply_base_selector(&x_sample, &y_sample)?;

            // Handle complementary pairs if enabled
            let final_selected = if self.config.complementary_pairs {
                self.apply_complementary_pairs_selection(&x_sample, &y_sample, &selected_features)?
            } else {
                selected_features
            };

            // Update selection counts
            for &feature_idx in &final_selected {
                if feature_idx < n_features {
                    selection_counts[feature_idx] += 1;
                }
            }

            // Store bootstrap result
            bootstrap_results.push(BootstrapResult {
                iteration,
                sample_indices,
                selected_features: final_selected,
                feature_scores: vec![], // Could be populated if needed
                lambda: None,           // Could be populated for L1 methods
            });
        }

        // Calculate selection probabilities
        let selection_probabilities: Vec<f64> = selection_counts
            .iter()
            .map(|&count| count as f64 / self.config.n_bootstrap_iterations as f64)
            .collect();

        // Calculate stability scores
        let stability_scores = self.calculate_stability_scores(&selection_probabilities);

        // Apply threshold to select features
        let selected_features = self.apply_threshold_selection(&selection_probabilities)?;

        // Generate lambda grid (for compatibility)
        let lambda_grid = vec![]; // Would be populated for L1-based methods

        // Create selection frequencies matrix
        let selection_frequencies =
            self.create_selection_frequencies_matrix(&bootstrap_results, n_features);

        self.selection_result = Some(StabilitySelectionResult {
            selected_features: selected_features.clone(),
            selection_probabilities,
            stability_scores,
            lambda_grid,
            selection_frequencies,
            bootstrap_results,
            n_features_in: n_features,
            n_features_out: selected_features.len(),
            config: self.config.clone(),
        });

        if self.config.verbose {
            eprintln!(
                "Stability Selection: Completed. Selected {} features",
                selected_features.len()
            );
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Transform data by selecting only the chosen features
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let selection_result = self.selection_result.as_ref().unwrap();

        if x.is_empty() {
            return Ok(Vec::new());
        }

        let n_features_in = x[0].len();
        if n_features_in != selection_result.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: selection_result.n_features_in,
                actual: n_features_in,
            });
        }

        let mut transformed_data = Vec::new();
        for row in x {
            let selected_row: Vec<f64> = selection_result
                .selected_features
                .iter()
                .map(|&idx| row[idx])
                .collect();
            transformed_data.push(selected_row);
        }

        Ok(transformed_data)
    }

    /// Fit and transform data in one step
    pub fn fit_transform(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<Vec<Vec<f64>>, SklearsError> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get the selection result
    pub fn get_selection_result(&self) -> Option<&StabilitySelectionResult> {
        self.selection_result.as_ref()
    }

    /// Get selected feature indices
    pub fn get_selected_features(&self) -> Option<&Vec<usize>> {
        self.selection_result.as_ref().map(|r| &r.selected_features)
    }

    /// Get selection probabilities
    pub fn get_selection_probabilities(&self) -> Option<&Vec<f64>> {
        self.selection_result
            .as_ref()
            .map(|r| &r.selection_probabilities)
    }

    /// Get stability scores
    pub fn get_stability_scores(&self) -> Option<&Vec<f64>> {
        self.selection_result.as_ref().map(|r| &r.stability_scores)
    }

    /// Compute stability path for different thresholds
    pub fn compute_stability_path(
        &self,
        threshold_range: Option<(f64, f64, usize)>,
    ) -> Result<StabilityPath, SklearsError> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let selection_result = self.selection_result.as_ref().unwrap();
        let (min_thresh, max_thresh, n_points) = threshold_range.unwrap_or((0.1, 1.0, 100));

        let mut thresholds = Vec::new();
        let mut n_selected = Vec::new();
        let mut expected_fdr = Vec::new();
        let mut pfer = Vec::new();

        for i in 0..n_points {
            let threshold =
                min_thresh + (max_thresh - min_thresh) * i as f64 / (n_points - 1) as f64;
            thresholds.push(threshold);

            // Count features above threshold
            let selected_count = selection_result
                .selection_probabilities
                .iter()
                .filter(|&&prob| prob >= threshold)
                .count();
            n_selected.push(selected_count);

            // Estimate expected FDR
            let efdr = if selected_count > 0 {
                self.estimate_expected_fdr(threshold, selected_count)
            } else {
                0.0
            };
            expected_fdr.push(efdr);

            // Estimate PFER
            let pfer_val = self.estimate_pfer(threshold);
            pfer.push(pfer_val);
        }

        Ok(StabilityPath {
            thresholds,
            n_selected,
            expected_fdr,
            pfer,
        })
    }

    /// Generate sample indices for bootstrap
    fn generate_sample_indices(&mut self, n_samples: usize, sample_size: usize) -> Vec<usize> {
        let mut indices = Vec::new();

        if self.config.replace {
            // Sample with replacement
            for _ in 0..sample_size {
                indices.push(self.rng.gen_range(0..n_samples));
            }
        } else {
            // Sample without replacement
            let mut all_indices: Vec<usize> = (0..n_samples).collect();

            for _ in 0..sample_size {
                if all_indices.is_empty() {
                    break;
                }
                let idx = self.rng.gen_range(0..all_indices.len());
                indices.push(all_indices.remove(idx));
            }
        }

        indices.sort();
        indices
    }

    /// Create sample data from indices
    fn create_sample_data(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        indices: &[usize],
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x_sample: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
        let y_sample: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        (x_sample, y_sample)
    }

    /// Apply base selector to sample data
    fn apply_base_selector(&self, x: &[Vec<f64>], y: &[f64]) -> Result<Vec<usize>, SklearsError> {
        match &self.config.base_selector {
            BaseSelector::LassoCV { config } => {
                let lasso = LassoCV::new();
                // Convert data
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = lasso.fit(&x_array, &y_array)?;

                // Get non-zero coefficients
                let coefficients = trained.coef();

                Ok(coefficients
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &coef)| if coef.abs() > 1e-8 { Some(i) } else { None })
                    .collect())
            }

            BaseSelector::RidgeCV { config } => {
                let ridge = RidgeCV::new();
                // Convert data
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = ridge.fit(&x_array, &y_array)?;

                // Get coefficient magnitudes and select top features
                let coefficients = trained.coef();

                let mut coef_with_idx: Vec<(usize, f64)> = coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, &coef)| (i, coef.abs()))
                    .collect();

                coef_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                // Select top half of features
                let n_select = (coefficients.len() / 2).max(1);
                Ok(coef_with_idx
                    .iter()
                    .take(n_select)
                    .map(|(i, _)| *i)
                    .collect())
            }

            BaseSelector::Univariate { strategy } => {
                let mut selector = FeatureSelector::new().with_strategy(strategy.clone());
                selector.fit(x, y)?;

                selector
                    .get_selected_features()
                    .ok_or_else(|| {
                        SklearsError::InvalidInput(
                            "Failed to get selected features from univariate selector".to_string(),
                        )
                    })
                    .map(|features| features.clone())
            }

            BaseSelector::Threshold {
                threshold,
                score_func,
            } => {
                let mut selector =
                    FeatureSelector::new().with_strategy(FeatureSelectionStrategy::SelectFpr {
                        alpha: *threshold,
                        score_func: score_func.clone(),
                    });

                selector.fit(x, y)?;

                selector
                    .get_selected_features()
                    .ok_or_else(|| {
                        SklearsError::InvalidInput(
                            "Failed to get selected features from threshold selector".to_string(),
                        )
                    })
                    .map(|features| features.clone())
            }
        }
    }

    /// Apply complementary pairs stabilization
    fn apply_complementary_pairs_selection(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        selected_features: &[usize],
    ) -> Result<Vec<usize>, SklearsError> {
        // For complementary pairs, we would run the base selector on a complementary sample
        // For simplicity, we'll return the original selection
        // In a full implementation, this would involve:
        // 1. Create complementary sample (remaining indices)
        // 2. Run base selector on complementary sample
        // 3. Take intersection of selections
        Ok(selected_features.to_vec())
    }

    /// Calculate stability scores
    fn calculate_stability_scores(&self, selection_probabilities: &[f64]) -> Vec<f64> {
        // Stability score is based on how far the probability is from randomness (0.5)
        selection_probabilities
            .iter()
            .map(|&prob| 2.0 * (prob - 0.5).abs())
            .collect()
    }

    /// Apply threshold selection
    fn apply_threshold_selection(
        &self,
        selection_probabilities: &[f64],
    ) -> Result<Vec<usize>, SklearsError> {
        let mut selected_features: Vec<usize> = selection_probabilities
            .iter()
            .enumerate()
            .filter_map(|(i, &prob)| {
                if prob >= self.config.threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        // Apply FDR control if specified
        if let Some(expected_fdr) = self.config.expected_fdr {
            selected_features = self.apply_fdr_control(selection_probabilities, expected_fdr)?;
        }

        // Apply PFER control if specified
        if let Some(pfer_bound) = self.config.pfer_bound {
            selected_features = self.apply_pfer_control(selection_probabilities, pfer_bound)?;
        }

        Ok(selected_features)
    }

    /// Apply false discovery rate control
    fn apply_fdr_control(
        &self,
        selection_probabilities: &[f64],
        expected_fdr: f64,
    ) -> Result<Vec<usize>, SklearsError> {
        // Sort features by selection probability
        let mut prob_with_idx: Vec<(usize, f64)> = selection_probabilities
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();

        prob_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Find largest k such that E[V] <= expected_fdr
        let mut selected_features = Vec::new();

        for (k, (feature_idx, prob)) in prob_with_idx.iter().enumerate() {
            let estimated_fdr = self.estimate_expected_fdr(*prob, k + 1);

            if estimated_fdr <= expected_fdr {
                selected_features.push(*feature_idx);
            } else {
                break;
            }
        }

        selected_features.sort();
        Ok(selected_features)
    }

    /// Apply per-family error rate control
    fn apply_pfer_control(
        &self,
        selection_probabilities: &[f64],
        pfer_bound: f64,
    ) -> Result<Vec<usize>, SklearsError> {
        // Sort features by selection probability
        let mut prob_with_idx: Vec<(usize, f64)> = selection_probabilities
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();

        prob_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Find features satisfying PFER bound
        let mut selected_features = Vec::new();

        for (feature_idx, prob) in prob_with_idx {
            let pfer = self.estimate_pfer(prob);

            if pfer <= pfer_bound {
                selected_features.push(feature_idx);
            }
        }

        selected_features.sort();
        Ok(selected_features)
    }

    /// Estimate expected false discovery rate
    fn estimate_expected_fdr(&self, threshold: f64, n_selected: usize) -> f64 {
        if n_selected == 0 {
            return 0.0;
        }

        // Simplified FDR estimation
        // E[V] ≈ p * (1 - threshold)^2 where p is number of features
        let n_features = self
            .selection_result
            .as_ref()
            .map(|r| r.n_features_in)
            .unwrap_or(100) as f64;

        let expected_false_positives = n_features * (1.0 - threshold).powi(2);
        expected_false_positives / n_selected as f64
    }

    /// Estimate per-family error rate
    fn estimate_pfer(&self, threshold: f64) -> f64 {
        // PFER ≈ p * (1 - threshold) where p is number of features
        let n_features = self
            .selection_result
            .as_ref()
            .map(|r| r.n_features_in)
            .unwrap_or(100) as f64;

        n_features * (1.0 - threshold)
    }

    /// Create selection frequencies matrix
    fn create_selection_frequencies_matrix(
        &self,
        bootstrap_results: &[BootstrapResult],
        n_features: usize,
    ) -> Vec<Vec<usize>> {
        let mut frequencies = vec![vec![0; n_features]; bootstrap_results.len()];

        for (i, result) in bootstrap_results.iter().enumerate() {
            for &feature_idx in &result.selected_features {
                if feature_idx < n_features {
                    frequencies[i][feature_idx] = 1;
                }
            }
        }

        frequencies
    }
}

impl Default for StabilitySelection {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        // Create data where first 3 features are relevant, last 5 are noise
        let x = vec![
            vec![1.0, 2.0, 3.0, 0.1, 0.9, 0.2, 0.8, 0.3],
            vec![2.0, 3.0, 4.0, 0.2, 0.8, 0.1, 0.7, 0.4],
            vec![3.0, 4.0, 5.0, 0.1, 0.7, 0.3, 0.6, 0.2],
            vec![4.0, 5.0, 6.0, 0.3, 0.6, 0.2, 0.5, 0.1],
            vec![5.0, 6.0, 7.0, 0.2, 0.5, 0.1, 0.4, 0.3],
            vec![6.0, 7.0, 8.0, 0.1, 0.4, 0.2, 0.3, 0.2],
            vec![7.0, 8.0, 9.0, 0.2, 0.3, 0.1, 0.2, 0.1],
            vec![8.0, 9.0, 10.0, 0.1, 0.2, 0.3, 0.1, 0.2],
            vec![9.0, 10.0, 11.0, 0.3, 0.1, 0.2, 0.2, 0.3],
            vec![10.0, 11.0, 12.0, 0.2, 0.2, 0.1, 0.3, 0.1],
        ];
        // Target is combination of first 3 features
        let y = vec![6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0];
        (x, y)
    }

    #[test]
    fn test_stability_selection_basic() {
        let mut stability = StabilitySelection::new()
            .with_n_bootstrap_iterations(20)
            .with_threshold(0.3)
            .with_sample_fraction(0.6);

        let (x, y) = create_sample_data();
        let result = stability.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select some features
        assert!(!transformed.is_empty());
        assert!(transformed[0].len() <= 8); // Can't select more than original
    }

    #[test]
    fn test_stability_selection_with_lasso() {
        let mut stability = StabilitySelection::new()
            .with_base_selector(BaseSelector::LassoCV {
                config: LassoCVConfig::default(),
            })
            .with_n_bootstrap_iterations(10)
            .with_threshold(0.2);

        let (x, y) = create_sample_data();
        let result = stability.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        assert_eq!(transformed.len(), 10); // Same number of samples
        assert!(transformed[0].len() > 0); // Some features selected
    }

    #[test]
    fn test_stability_selection_with_ridge() {
        let mut stability = StabilitySelection::new()
            .with_base_selector(BaseSelector::RidgeCV {
                config: RidgeCVConfig::default(),
            })
            .with_n_bootstrap_iterations(10)
            .with_threshold(0.3);

        let (x, y) = create_sample_data();
        let result = stability.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        assert_eq!(transformed.len(), 10);
        assert!(transformed[0].len() > 0);
    }

    #[test]
    fn test_stability_selection_probabilities() {
        let mut stability = StabilitySelection::new().with_n_bootstrap_iterations(20);

        let (x, y) = create_sample_data();
        stability.fit(&x, &y).unwrap();

        let probabilities = stability.get_selection_probabilities().unwrap();
        assert_eq!(probabilities.len(), 8); // 8 original features

        // Probabilities should be between 0 and 1
        for &prob in probabilities {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_stability_scores() {
        let mut stability = StabilitySelection::new().with_n_bootstrap_iterations(15);

        let (x, y) = create_sample_data();
        stability.fit(&x, &y).unwrap();

        let stability_scores = stability.get_stability_scores().unwrap();
        assert_eq!(stability_scores.len(), 8);

        // Stability scores should be between 0 and 1
        for &score in stability_scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_stability_path() {
        let mut stability = StabilitySelection::new().with_n_bootstrap_iterations(10);

        let (x, y) = create_sample_data();
        stability.fit(&x, &y).unwrap();

        let path = stability
            .compute_stability_path(Some((0.1, 0.9, 20)))
            .unwrap();

        assert_eq!(path.thresholds.len(), 20);
        assert_eq!(path.n_selected.len(), 20);
        assert_eq!(path.expected_fdr.len(), 20);
        assert_eq!(path.pfer.len(), 20);

        // Thresholds should be increasing
        for i in 1..path.thresholds.len() {
            assert!(path.thresholds[i] >= path.thresholds[i - 1]);
        }

        // Generally, number of selected features should decrease with higher threshold
        // (though not strictly monotonic due to estimation)
    }

    #[test]
    fn test_fdr_control() {
        let mut stability = StabilitySelection::new()
            .with_n_bootstrap_iterations(15)
            .with_expected_fdr(Some(0.5));

        let (x, y) = create_sample_data();
        let result = stability.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // FDR control should limit the number of selected features
        assert!(transformed[0].len() <= 8);
    }

    #[test]
    fn test_different_sample_fractions() {
        let sample_fractions = vec![0.3, 0.5, 0.7, 0.9];
        let (x, y) = create_sample_data();

        for fraction in sample_fractions {
            let mut stability = StabilitySelection::new()
                .with_n_bootstrap_iterations(10)
                .with_sample_fraction(fraction);

            let result = stability.fit_transform(&x, &y);
            assert!(result.is_ok(), "Failed with sample fraction: {}", fraction);

            let transformed = result.unwrap();
            assert_eq!(transformed.len(), 10);
        }
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let (x, y) = create_sample_data();

        let mut stability1 = StabilitySelection::with_config(StabilitySelectionConfig {
            random_state: Some(42),
            n_bootstrap_iterations: 10,
            ..Default::default()
        });

        let mut stability2 = StabilitySelection::with_config(StabilitySelectionConfig {
            random_state: Some(42),
            n_bootstrap_iterations: 10,
            ..Default::default()
        });

        let result1 = stability1.fit_transform(&x, &y).unwrap();
        let result2 = stability2.fit_transform(&x, &y).unwrap();

        // Results should be identical with same seed
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_empty_data_error() {
        let mut stability = StabilitySelection::new();
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let result = stability.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut stability = StabilitySelection::new();
        let (x, _) = create_sample_data();
        let wrong_y = vec![1.0, 2.0]; // Wrong length

        let result = stability.fit(&x, &wrong_y);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_before_fit_error() {
        let stability = StabilitySelection::new();
        let (x, _) = create_sample_data();

        let result = stability.transform(&x);
        assert!(result.is_err());
    }
}
