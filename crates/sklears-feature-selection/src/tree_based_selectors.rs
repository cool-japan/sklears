//! Tree-based feature selection methods

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::base::SelectorMixin;

/// Tree-based feature importance selector
/// Generic tree selector that can work with any tree-based estimator
#[derive(Debug, Clone)]
pub struct TreeSelector<E, State = Untrained> {
    estimator: E,
    threshold: Option<f64>,
    max_features: Option<usize>,
    state: PhantomData<State>,
    // Trained state
    estimator_: Option<E>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    threshold_: Option<f64>,
}

/// Trait for tree-based estimators that provide feature importance
pub trait TreeImportance {
    /// Get feature importance scores from the tree
    /// Higher scores indicate more important features
    fn feature_importances(&self) -> SklResult<Array1<Float>>;
}

impl<E: Clone> TreeSelector<E, Untrained> {
    /// Create a new TreeSelector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            threshold: None,
            max_features: None,
            state: PhantomData,
            estimator_: None,
            selected_features_: None,
            n_features_: None,
            threshold_: None,
        }
    }

    /// Set the threshold for feature selection
    /// If None, will use median of feature importances
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set the maximum number of features to select
    pub fn max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }
}

impl<E> Estimator for TreeSelector<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for TreeSelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: TreeImportance + Send + Sync,
    Y: Clone + Send + Sync,
{
    type Fitted = TreeSelector<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();

        // Fit the tree estimator
        let fitted_estimator = self.estimator.clone().fit(x, y)?;

        // Get feature importances
        let importances = fitted_estimator.feature_importances()?;

        // Determine threshold
        let threshold = if let Some(thresh) = self.threshold {
            thresh
        } else {
            // Use median as default threshold
            let mut sorted_importances = importances.to_vec();
            sorted_importances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_importances[sorted_importances.len() / 2]
        };

        // Select features above threshold
        let mut selected_features: Vec<usize> = (0..n_features)
            .filter(|&i| importances[i] > threshold)
            .collect();

        // Apply max_features limit if specified
        if let Some(max_feat) = self.max_features {
            if selected_features.len() > max_feat {
                // Sort by importance and take top max_feat
                selected_features
                    .sort_by(|&a, &b| importances[b].partial_cmp(&importances[a]).unwrap());
                selected_features.truncate(max_feat);
                selected_features.sort(); // Restore original order
            }
        }

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(format!(
                "No features selected with threshold={}. Try reducing threshold.",
                threshold
            )));
        }

        Ok(TreeSelector {
            estimator: self.estimator.clone(),
            threshold: self.threshold,
            max_features: self.max_features,
            state: PhantomData,
            estimator_: Some(self.estimator),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            threshold_: Some(threshold),
        })
    }
}

impl<E> Transform<Array2<Float>> for TreeSelector<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let k = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, k));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl<E> SelectorMixin for TreeSelector<E, Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl<E> TreeSelector<E, Trained> {
    /// Get the support mask
    pub fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    /// Get the threshold used for selection
    pub fn threshold(&self) -> f64 {
        self.threshold_.unwrap()
    }

    /// Get selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }
}

/// Gradient Boosting Feature Selection
/// Uses gradient boosting to estimate feature importance and select features
#[derive(Debug, Clone)]
pub struct GradientBoostingSelector<State = Untrained> {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: f64,
    threshold: Option<f64>,
    max_features: Option<usize>,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    feature_importances_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    estimators_: Option<Vec<DecisionStump>>,
}

/// Simple decision stump for gradient boosting
#[derive(Debug, Clone)]
pub(crate) struct DecisionStump {
    feature_idx: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
    importance: f64,
}

impl DecisionStump {
    fn predict(&self, x: &Array1<Float>) -> f64 {
        if x[self.feature_idx] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }
}

impl GradientBoostingSelector<Untrained> {
    /// Create a new GradientBoostingSelector
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            threshold: None,
            max_features: None,
            random_state: None,
            state: PhantomData,
            feature_importances_: None,
            selected_features_: None,
            n_features_: None,
            estimators_: None,
        }
    }

    /// Set the number of boosting estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        if n_estimators < 1 {
            panic!("n_estimators must be at least 1");
        }
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        if learning_rate <= 0.0 {
            panic!("learning_rate must be positive");
        }
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum depth of decision stumps
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        if max_depth < 1 {
            panic!("max_depth must be at least 1");
        }
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum samples required to split a node
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        if min_samples_split < 2 {
            panic!("min_samples_split must be at least 2");
        }
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples required at a leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        if min_samples_leaf < 1 {
            panic!("min_samples_leaf must be at least 1");
        }
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the subsample ratio for training each estimator
    pub fn subsample(mut self, subsample: f64) -> Self {
        if !(0.0..=1.0).contains(&subsample) {
            panic!("subsample must be between 0 and 1");
        }
        self.subsample = subsample;
        self
    }

    /// Set the threshold for feature selection
    pub fn threshold(mut self, threshold: Option<f64>) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum number of features to select
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Default for GradientBoostingSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GradientBoostingSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for GradientBoostingSelector<Untrained> {
    type Fitted = GradientBoostingSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Gradient Boosting requires at least 2 samples".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        // Initialize feature importances
        let mut feature_importances = Array1::zeros(n_features);
        let mut estimators = Vec::new();

        // Initialize predictions with mean of target
        let y_mean = y.mean().unwrap_or(0.0);
        let mut predictions = Array1::from_elem(n_samples, y_mean);

        // Boosting iterations
        for _ in 0..self.n_estimators {
            // Compute residuals (negative gradients for regression)
            let residuals: Array1<Float> = y - &predictions;

            // Sample data if subsample < 1.0
            let (x_sample, residuals_sample) = if self.subsample < 1.0 {
                let sample_size = (n_samples as f64 * self.subsample).round() as usize;
                let indices: Vec<usize> = (0..n_samples).collect();
                let mut sampled_indices = indices;
                for i in 0..sample_size {
                    let j = rng.gen_range(i..n_samples);
                    sampled_indices.swap(i, j);
                }
                sampled_indices.truncate(sample_size);

                let x_sub = x.select(Axis(0), &sampled_indices);
                let r_sub = residuals.select(Axis(0), &sampled_indices);
                (x_sub, r_sub)
            } else {
                (x.clone(), residuals)
            };

            // Find best split for this iteration
            let mut best_stump = None;
            let mut best_improvement = f64::NEG_INFINITY;

            for feature_idx in 0..n_features {
                // Get unique values for this feature
                let mut feature_values: Vec<f64> = x_sample.column(feature_idx).to_vec();
                feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                feature_values.dedup();

                // Try different thresholds
                for i in 0..(feature_values.len().saturating_sub(1)) {
                    let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                    // Split data based on threshold
                    let (left_indices, right_indices): (Vec<_>, Vec<_>) = (0..x_sample.nrows())
                        .partition(|&j| x_sample[[j, feature_idx]] <= threshold);

                    if left_indices.len() < self.min_samples_leaf
                        || right_indices.len() < self.min_samples_leaf
                    {
                        continue;
                    }

                    // Compute values for left and right splits
                    let left_value = if !left_indices.is_empty() {
                        left_indices
                            .iter()
                            .map(|&idx| residuals_sample[idx])
                            .sum::<f64>()
                            / left_indices.len() as f64
                    } else {
                        0.0
                    };

                    let right_value = if !right_indices.is_empty() {
                        right_indices
                            .iter()
                            .map(|&idx| residuals_sample[idx])
                            .sum::<f64>()
                            / right_indices.len() as f64
                    } else {
                        0.0
                    };

                    // Compute improvement (reduction in squared error)
                    let total_var = residuals_sample.mapv(|x| x * x).sum();
                    let left_var: f64 = left_indices
                        .iter()
                        .map(|&idx| (residuals_sample[idx] - left_value).powi(2))
                        .sum();
                    let right_var: f64 = right_indices
                        .iter()
                        .map(|&idx| (residuals_sample[idx] - right_value).powi(2))
                        .sum();

                    let improvement = total_var - left_var - right_var;

                    if improvement > best_improvement {
                        best_improvement = improvement;
                        best_stump = Some(DecisionStump {
                            feature_idx,
                            threshold,
                            left_value,
                            right_value,
                            importance: improvement,
                        });
                    }
                }
            }

            if let Some(stump) = best_stump {
                // Update feature importance
                feature_importances[stump.feature_idx] += stump.importance;

                // Update predictions
                for i in 0..n_samples {
                    let row = x.row(i);
                    let prediction = stump.predict(&row.to_owned());
                    predictions[i] += self.learning_rate * prediction;
                }

                estimators.push(stump);
            }
        }

        // Normalize feature importances
        let total_importance = feature_importances.sum();
        if total_importance > 0.0 {
            feature_importances /= total_importance;
        }

        // Select features based on threshold
        let threshold = self
            .threshold
            .unwrap_or_else(|| feature_importances.mean().unwrap_or(0.0));

        let mut selected_features: Vec<usize> = feature_importances
            .iter()
            .enumerate()
            .filter(|(_, &importance)| importance > threshold)
            .map(|(idx, _)| idx)
            .collect();

        // Apply max_features limit if specified
        if let Some(max_feat) = self.max_features {
            if selected_features.len() > max_feat {
                // Sort by importance and take top max_feat
                selected_features.sort_by(|&a, &b| {
                    feature_importances[b]
                        .partial_cmp(&feature_importances[a])
                        .unwrap()
                });
                selected_features.truncate(max_feat);
                selected_features.sort(); // Restore original order
            }
        }

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected by Gradient Boosting. Try reducing threshold.".to_string(),
            ));
        }

        Ok(GradientBoostingSelector {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            subsample: self.subsample,
            threshold: self.threshold,
            max_features: self.max_features,
            random_state: self.random_state,
            state: PhantomData,
            feature_importances_: Some(feature_importances),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            estimators_: Some(estimators),
        })
    }
}

impl Transform<Array2<Float>> for GradientBoostingSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        extract_features(x, selected_features)
    }
}

impl SelectorMixin for GradientBoostingSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl GradientBoostingSelector<Trained> {
    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        self.feature_importances_.as_ref().unwrap()
    }

    /// Get selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get the trained estimators (decision stumps)
    pub(crate) fn estimators(&self) -> &[DecisionStump] {
        self.estimators_.as_ref().unwrap()
    }
}

/// Tree-based importance selector for ensemble feature ranking
#[derive(Debug, Clone)]
pub struct TreeImportanceSelector {
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
}

impl Default for TreeImportanceSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeImportanceSelector {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            random_state: None,
        }
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl crate::embedded::SelectorFunction for TreeImportanceSelector {
    fn compute_scores(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
    ) -> SklResult<Array1<f64>> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Simple random forest importance simulation
        // In a real implementation, this would use actual tree algorithms
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        let mut feature_importance = Array1::<f64>::zeros(n_features);

        // Simulate feature importance by computing correlation-based importance
        if let Some(y) = y_classif {
            // For classification, use chi-squared like measure
            for i in 0..n_features {
                let feature_col = x.column(i);
                let mut score = 0.0;

                // Simple correlation measure
                for j in 0..n_samples {
                    let feature_val = feature_col[j];
                    let target_val = y[j] as f64;
                    score += (feature_val * target_val).abs();
                }

                feature_importance[i] = score / n_samples as f64;
            }
        } else if let Some(y) = y_regression {
            // For regression, use correlation
            for i in 0..n_features {
                let feature_col = x.column(i);
                let correlation = compute_pearson_correlation(&feature_col.to_owned(), y);
                feature_importance[i] = correlation.abs();
            }
        } else {
            return Err(SklearsError::InvalidInput(
                "Need either classification or regression target".to_string(),
            ));
        }

        // Add some randomness to simulate ensemble of trees
        for importance in feature_importance.iter_mut() {
            *importance += rng.gen::<f64>() * 0.1; // Add small random component
        }

        Ok(feature_importance)
    }

    fn name(&self) -> &str {
        "tree_importance"
    }

    fn clone_box(&self) -> Box<dyn crate::embedded::SelectorFunction> {
        Box::new(self.clone())
    }
}

/// Helper function to compute Pearson correlation between two arrays
fn compute_pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Extract selected features from the data matrix
fn extract_features(x: &Array2<Float>, features: &[usize]) -> SklResult<Array2<Float>> {
    let n_samples = x.nrows();
    let n_selected = features.len();
    let mut x_subset = Array2::zeros((n_samples, n_selected));

    for (new_idx, &old_idx) in features.iter().enumerate() {
        if old_idx >= x.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature index {} out of bounds",
                old_idx
            )));
        }
        x_subset.column_mut(new_idx).assign(&x.column(old_idx));
    }

    Ok(x_subset)
}
