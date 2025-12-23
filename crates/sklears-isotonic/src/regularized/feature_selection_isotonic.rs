//! Feature selection integration for isotonic regression
//!
//! This module combines isotonic regression with feature selection to automatically
//! identify the most relevant features for monotonic relationships.
//!
//! ## Features
//!
//! - **Multiple selection methods**: Univariate, forward/backward selection, recursive elimination
//! - **Cross-validation based**: Robust feature selection using CV scores
//! - **Monotonic correlation**: Features are selected based on monotonic relationships
//! - **Automatic thresholding**: Intelligent selection of important features
//! - **Additive modeling**: Combines selected features using additive isotonic models
//!
//! ## Selection Methods
//!
//! ### Univariate Methods
//! - **Monotonic correlation**: Based on Spearman rank correlation
//! - **Mutual information**: Based on information-theoretic measures
//!
//! ### Multivariate Methods
//! - **Forward selection**: Iteratively adds the best features
//! - **Backward elimination**: Iteratively removes the worst features
//! - **Recursive elimination**: Systematic feature ranking and elimination
//! - **L1 regularization**: Automatic selection via sparsity-inducing penalties
//!
//! ## Examples
//!
//! ```rust,ignore
//! use sklears_isotonic::regularized::feature_selection_isotonic::*;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Create feature matrix and target
//! let x = Array2::from_shape_vec((5, 3), vec![
//!     1.0, 2.0, 3.0,
//!     2.0, 3.0, 1.0,
//!     3.0, 1.0, 2.0,
//!     4.0, 4.0, 4.0,
//!     5.0, 5.0, 5.0,
//! ]).unwrap();
//! let y = Array1::from(vec![1.0, 2.0, 2.5, 4.0, 5.0]);
//!
//! // Create feature selection model
//! let model = FeatureSelectionIsotonicRegression::new()
//!     .selection_method(FeatureSelectionMethod::UnivariateMonotonic)
//!     .n_features_to_select(2)
//!     .selection_threshold(0.1);
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let predictions = fitted.predict(&x).unwrap();
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{
    argmax, extract_subset_labels, mutual_information, spearman_correlation,
    AdditiveIsotonicRegression, IsotonicRegression, LossFunction, MonotonicityConstraint,
};

use super::simd_operations::{simd_mean, simd_mse};

/// Feature selection methods for isotonic regression
#[derive(Debug, Clone, Copy, PartialEq)]
/// FeatureSelectionMethod
pub enum FeatureSelectionMethod {
    /// Univariate feature selection based on monotonic correlation
    UnivariateMonotonic,
    /// Forward selection based on cross-validation
    ForwardSelection,
    /// Backward elimination based on cross-validation
    BackwardElimination,
    /// Recursive feature elimination
    RecursiveElimination,
    /// L1 regularization for automatic feature selection
    L1Regularization { alpha: Float },
    /// Mutual information based selection for monotonic relationships
    MutualInformation,
}

/// Feature selection integration for isotonic regression
///
/// This struct combines isotonic regression with feature selection to automatically
/// identify the most relevant features for monotonic relationships.
#[derive(Debug, Clone)]
/// FeatureSelectionIsotonicRegression
pub struct FeatureSelectionIsotonicRegression<State = Untrained> {
    /// Isotonic regression parameters
    pub constraint: MonotonicityConstraint,
    /// y_min
    pub y_min: Option<Float>,
    /// y_max
    pub y_max: Option<Float>,
    /// out_of_bounds
    pub out_of_bounds: String,
    /// loss
    pub loss: LossFunction,

    /// Feature selection parameters
    pub selection_method: FeatureSelectionMethod,
    /// n_features_to_select
    pub n_features_to_select: Option<usize>,
    /// selection_threshold
    pub selection_threshold: Float,

    // Fitted attributes
    selected_features_: Option<Vec<usize>>,
    feature_scores_: Option<Array1<Float>>,
    isotonic_models_: Option<Vec<IsotonicRegression<Trained>>>,

    _state: PhantomData<State>,
}

impl FeatureSelectionIsotonicRegression<Untrained> {
    /// Create a new feature selection isotonic regression model
    ///
    /// # Returns
    ///
    /// A new `FeatureSelectionIsotonicRegression` instance with default parameters:
    /// - Global increasing monotonicity constraint
    /// - No bounds on output
    /// - Squared loss function
    /// - Univariate monotonic selection method
    /// - Selection threshold of 0.01
    pub fn new() -> Self {
        Self {
            constraint: MonotonicityConstraint::Global { increasing: true },
            y_min: None,
            y_max: None,
            out_of_bounds: "nan".to_string(),
            loss: LossFunction::SquaredLoss,
            selection_method: FeatureSelectionMethod::UnivariateMonotonic,
            n_features_to_select: None,
            selection_threshold: 0.01,
            selected_features_: None,
            feature_scores_: None,
            isotonic_models_: None,
            _state: PhantomData,
        }
    }

    /// Set the feature selection method
    ///
    /// # Arguments
    ///
    /// * `method` - Feature selection method to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sklears_isotonic::regularized::feature_selection_isotonic::*;
    /// let model = FeatureSelectionIsotonicRegression::new()
    ///     .selection_method(FeatureSelectionMethod::ForwardSelection);
    /// ```
    pub fn selection_method(mut self, method: FeatureSelectionMethod) -> Self {
        self.selection_method = method;
        self
    }

    /// Set the number of features to select
    ///
    /// If not specified, features will be selected based on the threshold.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features to select
    pub fn n_features_to_select(mut self, n_features: usize) -> Self {
        self.n_features_to_select = Some(n_features);
        self
    }

    /// Set the selection threshold
    ///
    /// Features with scores above this threshold will be selected
    /// (when `n_features_to_select` is not specified).
    ///
    /// # Arguments
    ///
    /// * `threshold` - Selection threshold
    pub fn selection_threshold(mut self, threshold: Float) -> Self {
        self.selection_threshold = threshold;
        self
    }

    /// Set monotonicity constraint
    ///
    /// # Arguments
    ///
    /// * `constraint` - Monotonicity constraint to apply
    pub fn constraint(mut self, constraint: MonotonicityConstraint) -> Self {
        self.constraint = constraint;
        self
    }

    /// Set loss function
    ///
    /// # Arguments
    ///
    /// * `loss` - Loss function for isotonic regression
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set lower bound on output values
    ///
    /// # Arguments
    ///
    /// * `y_min` - Lower bound
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set upper bound on output values
    ///
    /// # Arguments
    ///
    /// * `y_max` - Upper bound
    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }
}

impl Default for FeatureSelectionIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FeatureSelectionIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for FeatureSelectionIsotonicRegression<Trained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for FeatureSelectionIsotonicRegression<Untrained> {
    type Fitted = FeatureSelectionIsotonicRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "X and y cannot be empty".to_string(),
            ));
        }

        let _n_features = x.ncols();

        // Perform feature selection based on the specified method
        let (selected_features, feature_scores) = match self.selection_method {
            FeatureSelectionMethod::UnivariateMonotonic => {
                self.univariate_monotonic_selection(x, y)?
            }
            FeatureSelectionMethod::ForwardSelection => self.forward_selection(x, y)?,
            FeatureSelectionMethod::BackwardElimination => self.backward_elimination(x, y)?,
            FeatureSelectionMethod::RecursiveElimination => self.recursive_elimination(x, y)?,
            FeatureSelectionMethod::L1Regularization { alpha } => {
                self.l1_regularization_selection(x, y, alpha)?
            }
            FeatureSelectionMethod::MutualInformation => self.mutual_information_selection(x, y)?,
        };

        // Train isotonic models for each selected feature
        let mut isotonic_models = Vec::new();
        for &feature_idx in &selected_features {
            let feature_col = x.column(feature_idx).to_owned();
            let mut isotonic = IsotonicRegression::new()
                .constraint(self.constraint.clone())
                .loss(self.loss);

            if let Some(y_min) = self.y_min {
                isotonic = isotonic.y_min(y_min);
            }
            if let Some(y_max) = self.y_max {
                isotonic = isotonic.y_max(y_max);
            }

            let fitted_model = isotonic.fit(&feature_col, y)?;
            isotonic_models.push(fitted_model);
        }

        Ok(FeatureSelectionIsotonicRegression {
            constraint: self.constraint,
            y_min: self.y_min,
            y_max: self.y_max,
            out_of_bounds: self.out_of_bounds,
            loss: self.loss,
            selection_method: self.selection_method,
            n_features_to_select: self.n_features_to_select,
            selection_threshold: self.selection_threshold,
            selected_features_: Some(selected_features),
            feature_scores_: Some(feature_scores),
            isotonic_models_: Some(isotonic_models),
            _state: PhantomData,
        })
    }
}

impl FeatureSelectionIsotonicRegression<Untrained> {
    /// Helper function to extract subset of data with specific features
    fn extract_subset_with_features(
        &self,
        data: &Array2<Float>,
        sample_indices: &[usize],
        feature_indices: &[usize],
    ) -> Result<Array2<Float>> {
        if sample_indices.is_empty() || feature_indices.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        // Validate indices
        for &idx in sample_indices {
            if idx >= data.nrows() {
                return Err(SklearsError::InvalidInput(format!(
                    "Sample index {} is out of bounds for array with {} rows",
                    idx,
                    data.nrows()
                )));
            }
        }

        for &idx in feature_indices {
            if idx >= data.ncols() {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature index {} is out of bounds for array with {} columns",
                    idx,
                    data.ncols()
                )));
            }
        }

        let mut result = Array2::zeros((sample_indices.len(), feature_indices.len()));
        for (new_i, &old_i) in sample_indices.iter().enumerate() {
            for (new_j, &old_j) in feature_indices.iter().enumerate() {
                result[(new_i, new_j)] = data[(old_i, old_j)];
            }
        }

        Ok(result)
    }

    /// Univariate monotonic feature selection based on Spearman correlation
    ///
    /// This method computes the Spearman rank correlation between each feature
    /// and the target variable, selecting features with high absolute correlation.
    fn univariate_monotonic_selection(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Vec<usize>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut scores = Array1::zeros(n_features);

        // Calculate monotonic correlation for each feature
        for i in 0..n_features {
            let feature_col = x.column(i);
            scores[i] = spearman_correlation(&feature_col.to_owned(), y)?;
        }

        // Select features based on absolute correlation
        let abs_scores: Array1<Float> = scores.mapv(|x| x.abs());
        let mut selected_features = Vec::new();

        if let Some(n_select) = self.n_features_to_select {
            // Select top n features
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| abs_scores[b].partial_cmp(&abs_scores[a]).unwrap());
            selected_features = indices.into_iter().take(n_select).collect();
        } else {
            // Select features above threshold
            for i in 0..n_features {
                if abs_scores[i] >= self.selection_threshold {
                    selected_features.push(i);
                }
            }
        }

        if selected_features.is_empty() {
            // Select at least one feature (the best one)
            let best_idx = argmax(&abs_scores)?;
            selected_features.push(best_idx);
        }

        Ok((selected_features, scores))
    }

    /// Forward selection using cross-validation
    ///
    /// This method starts with no features and iteratively adds the feature
    /// that most improves the cross-validation score.
    fn forward_selection(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Vec<usize>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut selected_features = Vec::new();
        let mut scores = Array1::zeros(n_features);
        let max_features = self.n_features_to_select.unwrap_or(n_features.min(10));

        for _ in 0..max_features {
            let mut best_score = Float::NEG_INFINITY;
            let mut best_feature = None;

            // Try adding each remaining feature
            for candidate in 0..n_features {
                if selected_features.contains(&candidate) {
                    continue;
                }

                let mut test_features = selected_features.clone();
                test_features.push(candidate);

                // Evaluate this feature subset using cross-validation
                let score = self.cross_validate_feature_subset(x, y, &test_features)?;

                if score > best_score {
                    best_score = score;
                    best_feature = Some(candidate);
                }
            }

            if let Some(feature) = best_feature {
                if best_score > self.selection_threshold || selected_features.is_empty() {
                    selected_features.push(feature);
                    scores[feature] = best_score;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok((selected_features, scores))
    }

    /// Backward elimination using cross-validation
    ///
    /// This method starts with all features and iteratively removes the feature
    /// that least degrades the cross-validation score.
    fn backward_elimination(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Vec<usize>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut selected_features: Vec<usize> = (0..n_features).collect();
        let mut scores = Array1::zeros(n_features);

        // Start with all features and remove the worst ones
        while selected_features.len() > 1 {
            let mut worst_score = Float::INFINITY;
            let mut worst_feature_idx = None;

            // Try removing each feature
            for (idx, &_feature) in selected_features.iter().enumerate() {
                let mut test_features = selected_features.clone();
                test_features.remove(idx);

                if test_features.is_empty() {
                    break;
                }

                // Evaluate without this feature
                let score = self.cross_validate_feature_subset(x, y, &test_features)?;

                if score < worst_score {
                    worst_score = score;
                    worst_feature_idx = Some(idx);
                }
            }

            if let Some(idx) = worst_feature_idx {
                if worst_score < self.selection_threshold && selected_features.len() > 1 {
                    let removed_feature = selected_features.remove(idx);
                    scores[removed_feature] = worst_score;
                } else {
                    break;
                }
            } else {
                break;
            }

            // Stop if we've reached the target number of features
            if let Some(n_select) = self.n_features_to_select {
                if selected_features.len() <= n_select {
                    break;
                }
            }
        }

        // Fill scores for remaining features
        for &feature in &selected_features {
            if scores[feature] == 0.0 {
                scores[feature] = self.selection_threshold + 0.1;
            }
        }

        Ok((selected_features, scores))
    }

    /// Recursive feature elimination
    ///
    /// This method systematically ranks features by their importance and
    /// iteratively eliminates the least important ones.
    fn recursive_elimination(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Vec<usize>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut feature_mask = vec![true; n_features];
        let mut scores = Array1::zeros(n_features);
        let target_features = self.n_features_to_select.unwrap_or(n_features / 2);

        while feature_mask.iter().filter(|&&x| x).count() > target_features {
            let active_features: Vec<usize> = feature_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &active)| if active { Some(i) } else { None })
                .collect();

            if active_features.len() <= target_features {
                break;
            }

            // Evaluate each active feature's contribution
            let mut feature_importance = Vec::new();
            for &feature in &active_features {
                let mut test_features = active_features.clone();
                test_features.retain(|&x| x != feature);

                if test_features.is_empty() {
                    feature_importance.push((feature, Float::INFINITY));
                    continue;
                }

                let score_with = self.cross_validate_feature_subset(x, y, &active_features)?;
                let score_without = self.cross_validate_feature_subset(x, y, &test_features)?;
                let importance = score_with - score_without;

                feature_importance.push((feature, importance));
            }

            // Remove the least important feature
            feature_importance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if let Some((worst_feature, importance)) = feature_importance.first() {
                feature_mask[*worst_feature] = false;
                scores[*worst_feature] = *importance;
            } else {
                break;
            }
        }

        let selected_features: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &active)| if active { Some(i) } else { None })
            .collect();

        Ok((selected_features, scores))
    }

    /// L1 regularization for automatic feature selection
    ///
    /// This method uses L1 regularization penalties to automatically select features,
    /// similar to Lasso regularization.
    fn l1_regularization_selection(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
    ) -> Result<(Vec<usize>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut scores = Array1::zeros(n_features);

        // For L1 regularization feature selection, we use univariate analysis
        // with penalty-based scoring. This is a simplified approach that
        // penalizes features based on their correlation strength relative to alpha.
        for i in 0..n_features {
            let feature_col = x.column(i);
            let correlation = spearman_correlation(&feature_col.to_owned(), y)?;

            // Apply soft thresholding similar to L1 regularization
            let abs_corr = correlation.abs();
            if abs_corr > alpha {
                scores[i] = correlation.signum() * (abs_corr - alpha);
            } else {
                scores[i] = 0.0;
            }
        }

        // Select features with non-zero scores (survived L1 penalty)
        let mut selected_features = Vec::new();

        if let Some(n_select) = self.n_features_to_select {
            // Select top n features by absolute score
            let abs_scores = scores.mapv(|x| x.abs());
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| abs_scores[b].partial_cmp(&abs_scores[a]).unwrap());
            selected_features = indices.into_iter().take(n_select).collect();
        } else {
            // Select features above threshold
            for i in 0..n_features {
                if scores[i].abs() >= self.selection_threshold {
                    selected_features.push(i);
                }
            }
        }

        if selected_features.is_empty() {
            // Select the best feature even if it doesn't survive L1 penalty
            let abs_scores = scores.mapv(|x| x.abs());
            let best_idx = argmax(&abs_scores)?;
            selected_features.push(best_idx);
        }

        Ok((selected_features, scores))
    }

    /// Mutual information based feature selection
    ///
    /// This method selects features based on their mutual information with the target,
    /// which captures non-linear monotonic relationships.
    fn mutual_information_selection(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Vec<usize>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut scores = Array1::zeros(n_features);

        // Calculate mutual information for each feature
        for i in 0..n_features {
            let feature_col = x.column(i);
            scores[i] = mutual_information(&feature_col.to_owned(), y, 10)?; // Use 10 bins as default
        }

        // Select features based on mutual information
        let mut selected_features = Vec::new();

        if let Some(n_select) = self.n_features_to_select {
            // Select top n features
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
            selected_features = indices.into_iter().take(n_select).collect();
        } else {
            // Select features above threshold
            for i in 0..n_features {
                if scores[i] >= self.selection_threshold {
                    selected_features.push(i);
                }
            }
        }

        if selected_features.is_empty() {
            // Select at least one feature (the best one)
            let best_idx = argmax(&scores)?;
            selected_features.push(best_idx);
        }

        Ok((selected_features, scores))
    }

    /// Cross-validate feature subset performance
    ///
    /// This method evaluates the performance of a given feature subset using
    /// k-fold cross-validation.
    fn cross_validate_feature_subset(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        features: &[usize],
    ) -> Result<Float> {
        if features.is_empty() {
            return Ok(Float::NEG_INFINITY);
        }

        let n_samples = x.nrows();
        let n_folds = 5.min(n_samples);
        let fold_size = n_samples / n_folds;
        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_samples {
                if i >= test_start && i < test_end {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if train_indices.is_empty() || test_indices.is_empty() {
                continue;
            }

            // Extract training data for selected features
            let x_train = self.extract_subset_with_features(x, &train_indices, features)?;
            let y_train = extract_subset_labels(y, &train_indices)?;
            let x_test = self.extract_subset_with_features(x, &test_indices, features)?;
            let y_test = extract_subset_labels(y, &test_indices)?;

            // Fit additive isotonic model for multi-feature case
            if features.len() == 1 {
                let isotonic = IsotonicRegression::new()
                    .constraint(self.constraint.clone())
                    .loss(self.loss);

                let fitted = isotonic.fit(&x_train.column(0).to_owned(), &y_train)?;
                let predictions = fitted.predict(&x_test.column(0).to_owned())?;
                let score = -simd_mse(predictions.as_slice().unwrap(), y_test.as_slice().unwrap());
                scores.push(score);
            } else {
                let additive = AdditiveIsotonicRegression::new(features.len()).loss(self.loss);

                let fitted = additive.fit(&x_train, &y_train)?;
                let predictions = fitted.predict(&x_test)?;
                let score = -simd_mse(predictions.as_slice().unwrap(), y_test.as_slice().unwrap());
                scores.push(score);
            }
        }

        if scores.is_empty() {
            Ok(Float::NEG_INFINITY)
        } else {
            Ok(simd_mean(&scores))
        }
    }
}

impl Predict<Array2<Float>, Array1<Float>> for FeatureSelectionIsotonicRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let isotonic_models = self.isotonic_models_.as_ref().unwrap();

        if x.ncols() < *selected_features.iter().max().unwrap_or(&0) + 1 {
            return Err(SklearsError::InvalidInput(
                "Input has fewer features than expected".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        if selected_features.len() == 1 {
            // Single feature case
            let feature_idx = selected_features[0];
            let feature_col = x.column(feature_idx).to_owned();
            predictions = isotonic_models[0].predict(&feature_col)?;
        } else {
            // Multi-feature case: use additive model approach
            for (i, &feature_idx) in selected_features.iter().enumerate() {
                let feature_col = x.column(feature_idx).to_owned();
                let feature_predictions = isotonic_models[i].predict(&feature_col)?;
                predictions = predictions + feature_predictions;
            }
            // Average the predictions
            predictions /= selected_features.len() as Float;
        }

        Ok(predictions)
    }
}

impl FeatureSelectionIsotonicRegression<Trained> {
    /// Get the selected features
    ///
    /// Returns the indices of the features that were selected during fitting.
    pub fn selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get the feature scores
    ///
    /// Returns the scores computed for each feature during selection.
    pub fn feature_scores(&self) -> Option<&Array1<Float>> {
        self.feature_scores_.as_ref()
    }

    /// Get the number of selected features
    ///
    /// Returns the count of features that were selected.
    pub fn n_selected_features(&self) -> usize {
        self.selected_features_.as_ref().map_or(0, |f| f.len())
    }

    /// Check if a specific feature was selected
    ///
    /// # Arguments
    ///
    /// * `feature_idx` - Index of the feature to check
    ///
    /// # Returns
    ///
    /// True if the feature was selected, false otherwise.
    pub fn is_feature_selected(&self, feature_idx: usize) -> bool {
        self.selected_features_
            .as_ref()
            .is_some_and(|features| features.contains(&feature_idx))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_feature_selection_isotonic_regression() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0]
        ];
        let y = Array1::from(vec![1.0, 2.0, 2.5, 4.0, 5.0]);

        let model = FeatureSelectionIsotonicRegression::new()
            .selection_method(FeatureSelectionMethod::UnivariateMonotonic)
            .n_features_to_select(2);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 5);
        assert_eq!(fitted.n_selected_features(), 2);
    }

    #[test]
    fn test_univariate_monotonic_selection() {
        let x = array![
            [1.0, 5.0, 3.0],
            [2.0, 4.0, 1.0],
            [3.0, 3.0, 2.0],
            [4.0, 2.0, 4.0],
            [5.0, 1.0, 5.0]
        ];
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let model = FeatureSelectionIsotonicRegression::new()
            .selection_method(FeatureSelectionMethod::UnivariateMonotonic)
            .selection_threshold(0.5);

        let fitted = model.fit(&x, &y).unwrap();

        // Feature 0 should have high positive correlation
        // Feature 1 should have high negative correlation
        // Feature 2 should have moderate positive correlation
        let selected = fitted.selected_features().unwrap();
        assert!(selected.contains(&0)); // Should be selected (perfect correlation)
    }

    #[test]
    fn test_forward_selection() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0]
        ];
        let y = Array1::from(vec![1.0, 2.0, 2.5, 4.0, 5.0]);

        let model = FeatureSelectionIsotonicRegression::new()
            .selection_method(FeatureSelectionMethod::ForwardSelection)
            .n_features_to_select(2);

        let fitted = model.fit(&x, &y).unwrap();
        assert!(fitted.n_selected_features() <= 2);
    }

    #[test]
    fn test_l1_regularization_selection() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0]
        ];
        let y = Array1::from(vec![1.0, 2.0, 2.5, 4.0, 5.0]);

        let model = FeatureSelectionIsotonicRegression::new()
            .selection_method(FeatureSelectionMethod::L1Regularization { alpha: 0.5 })
            .selection_threshold(0.1);

        let fitted = model.fit(&x, &y).unwrap();
        assert!(fitted.n_selected_features() >= 1);
    }

    #[test]
    fn test_mutual_information_selection() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0]
        ];
        let y = Array1::from(vec![1.0, 2.0, 2.5, 4.0, 5.0]);

        let model = FeatureSelectionIsotonicRegression::new()
            .selection_method(FeatureSelectionMethod::MutualInformation)
            .n_features_to_select(2);

        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_selected_features(), 2);
    }

    #[test]
    fn test_feature_selection_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = Array1::from(vec![1.0, 2.0, 2.5]);

        // Test all selection methods
        let methods = vec![
            FeatureSelectionMethod::UnivariateMonotonic,
            FeatureSelectionMethod::ForwardSelection,
            FeatureSelectionMethod::BackwardElimination,
            FeatureSelectionMethod::RecursiveElimination,
            FeatureSelectionMethod::L1Regularization { alpha: 0.1 },
            FeatureSelectionMethod::MutualInformation,
        ];

        for method in methods {
            let model = FeatureSelectionIsotonicRegression::new()
                .selection_method(method)
                .n_features_to_select(1);

            let fitted = model.fit(&x, &y).unwrap();
            assert_eq!(fitted.n_selected_features(), 1);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<Float>::zeros((0, 2));
        let y = Array1::<Float>::zeros(0);

        let model = FeatureSelectionIsotonicRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_dimensions_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::from(vec![1.0]); // Wrong length

        let model = FeatureSelectionIsotonicRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_selected_method() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]];
        let y = Array1::from(vec![1.0, 2.0, 2.5]);

        let model = FeatureSelectionIsotonicRegression::new()
            .selection_method(FeatureSelectionMethod::UnivariateMonotonic)
            .n_features_to_select(2);

        let fitted = model.fit(&x, &y).unwrap();

        // Check that some features are selected and some are not
        let mut n_selected = 0;
        for i in 0..3 {
            if fitted.is_feature_selected(i) {
                n_selected += 1;
            }
        }
        assert_eq!(n_selected, 2);
    }
}
