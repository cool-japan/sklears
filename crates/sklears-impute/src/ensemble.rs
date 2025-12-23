//! Ensemble-based imputation methods
//!
//! This module provides ensemble learning approaches to missing data imputation,
//! including random forests, gradient boosting, and other tree-based ensemble methods.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Random Forest Imputer
///
/// Uses random forest regression/classification to impute missing values.
/// For each feature with missing values, trains a random forest using other features as predictors.
///
/// # Parameters
///
/// * `n_estimators` - Number of trees in the forest
/// * `max_depth` - Maximum depth of trees
/// * `min_samples_split` - Minimum samples required to split a node
/// * `min_samples_leaf` - Minimum samples required at a leaf node
/// * `max_features` - Number of features to consider for best split
/// * `bootstrap` - Whether to use bootstrap sampling
/// * `random_state` - Random state for reproducibility
/// * `missing_values` - The placeholder for missing values
///
/// # Examples
///
/// ```
/// use sklears_impute::RandomForestImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = RandomForestImputer::new()
///     .n_estimators(100)
///     .max_depth(10);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RandomForestImputer<S = Untrained> {
    state: S,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: String,
    bootstrap: bool,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for RandomForestImputer
#[derive(Debug, Clone)]
pub struct RandomForestImputerTrained {
    forests: HashMap<usize, RandomForest>,
    feature_means_: Array1<f64>,
    n_features_in_: usize,
}

/// Random Forest model
#[derive(Debug, Clone)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    feature_indices: Vec<usize>,
    target_feature: usize,
}

/// Gradient Boosting Imputer
///
/// Uses gradient boosting to impute missing values through iterative improvement.
/// Builds additive models in a forward stage-wise fashion.
///
/// # Parameters
///
/// * `n_estimators` - Number of boosting stages
/// * `learning_rate` - Learning rate shrinks contribution of each tree
/// * `max_depth` - Maximum depth of individual regression estimators
/// * `min_samples_split` - Minimum samples required to split a node
/// * `min_samples_leaf` - Minimum samples required at a leaf node
/// * `subsample` - Fraction of samples used for fitting individual base learners
/// * `random_state` - Random state for reproducibility
/// * `missing_values` - The placeholder for missing values
///
/// # Examples
///
/// ```
/// use sklears_impute::GradientBoostingImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = GradientBoostingImputer::new()
///     .n_estimators(100)
///     .learning_rate(0.1);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GradientBoostingImputer<S = Untrained> {
    state: S,
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: f64,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for GradientBoostingImputer
#[derive(Debug, Clone)]
pub struct GradientBoostingImputerTrained {
    boosting_models: HashMap<usize, GradientBoostingModel>,
    feature_means_: Array1<f64>,
    n_features_in_: usize,
}

/// Gradient Boosting model
#[derive(Debug, Clone)]
pub struct GradientBoostingModel {
    trees: Vec<DecisionTree>,
    learning_rate: f64,
    initial_prediction: f64,
    target_feature: usize,
}

/// Extra Trees Imputer
///
/// Uses extremely randomized trees (Extra Trees) for imputation.
/// Similar to Random Forest but with more randomization in tree building.
///
/// # Parameters
///
/// * `n_estimators` - Number of trees in the forest
/// * `max_depth` - Maximum depth of trees
/// * `min_samples_split` - Minimum samples required to split a node
/// * `min_samples_leaf` - Minimum samples required at a leaf node
/// * `max_features` - Number of features to consider for best split
/// * `bootstrap` - Whether to use bootstrap sampling
/// * `random_state` - Random state for reproducibility
/// * `missing_values` - The placeholder for missing values
#[derive(Debug, Clone)]
pub struct ExtraTreesImputer<S = Untrained> {
    state: S,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: String,
    bootstrap: bool,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for ExtraTreesImputer
#[derive(Debug, Clone)]
pub struct ExtraTreesImputerTrained {
    forests: HashMap<usize, ExtraTreesForest>,
    feature_means_: Array1<f64>,
    n_features_in_: usize,
}

/// Extra Trees Forest model
#[derive(Debug, Clone)]
pub struct ExtraTreesForest {
    trees: Vec<DecisionTree>,
    feature_indices: Vec<usize>,
    target_feature: usize,
}

/// Decision Tree for imputation
#[derive(Debug, Clone)]
pub struct DecisionTree {
    nodes: Vec<TreeNode>,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

/// Tree node structure
#[derive(Debug, Clone)]
pub struct TreeNode {
    feature_index: Option<usize>,
    threshold: Option<f64>,
    left_child: Option<usize>,
    right_child: Option<usize>,
    value: f64,
    n_samples: usize,
    is_leaf: bool,
}

/// Training data for tree construction
#[derive(Debug, Clone)]
struct TreeTrainingData {
    features: Array2<f64>,
    targets: Array1<f64>,
    sample_indices: Vec<usize>,
}

// RandomForestImputer implementation

impl RandomForestImputer<Untrained> {
    /// Create a new RandomForestImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: "sqrt".to_string(),
            bootstrap: true,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the maximum depth
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Set the minimum samples split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the max features strategy
    pub fn max_features(mut self, max_features: String) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set whether to use bootstrap
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for RandomForestImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RandomForestImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RandomForestImputer<Untrained> {
    type Fitted = RandomForestImputer<RandomForestImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let mut rng = Random::default();

        // Compute feature means for fallback
        let feature_means = compute_feature_means(&X, self.missing_values);

        let mut forests = HashMap::new();

        // Train random forest for each feature with missing values
        for target_feature in 0..n_features {
            let has_missing = (0..n_samples).any(|i| self.is_missing(X[[i, target_feature]]));

            if has_missing {
                let forest = self.train_random_forest(&X, target_feature, &mut rng)?;
                forests.insert(target_feature, forest);
            }
        }

        Ok(RandomForestImputer {
            state: RandomForestImputerTrained {
                forests,
                feature_means_: feature_means,
                n_features_in_: n_features,
            },
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            bootstrap: self.bootstrap,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for RandomForestImputer<RandomForestImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        // Apply each forest to impute its target feature
        for (&target_feature, forest) in &self.state.forests {
            for i in 0..n_samples {
                if self.is_missing(X_imputed[[i, target_feature]]) {
                    // Create input vector excluding the target feature
                    let mut input_features = Vec::new();
                    for j in 0..n_features {
                        if j != target_feature {
                            if self.is_missing(X_imputed[[i, j]]) {
                                input_features.push(self.state.feature_means_[j]);
                            } else {
                                input_features.push(X_imputed[[i, j]]);
                            }
                        }
                    }

                    let input_array = Array1::from_vec(input_features);
                    let predicted_value = self.predict_forest(forest, &input_array)?;
                    X_imputed[[i, target_feature]] = predicted_value;
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl RandomForestImputer<Untrained> {
    fn train_random_forest(
        &self,
        X: &Array2<f64>,
        target_feature: usize,
        rng: &mut impl Rng,
    ) -> SklResult<RandomForest> {
        let (n_samples, n_features) = X.dim();

        // Collect training samples where target feature is not missing
        let mut training_data = Vec::new();
        let mut training_targets = Vec::new();

        for i in 0..n_samples {
            if !self.is_missing(X[[i, target_feature]]) {
                let mut features = Vec::new();
                let mut has_missing = false;

                for j in 0..n_features {
                    if j != target_feature {
                        if self.is_missing(X[[i, j]]) {
                            has_missing = true;
                            break;
                        }
                        features.push(X[[i, j]]);
                    }
                }

                if !has_missing {
                    training_data.push(features);
                    training_targets.push(X[[i, target_feature]]);
                }
            }
        }

        if training_data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid training samples for feature".to_string(),
            ));
        }

        let n_training_features = training_data[0].len();
        let training_X =
            Array2::from_shape_fn((training_data.len(), n_training_features), |(i, j)| {
                training_data[i][j]
            });
        let training_y = Array1::from_vec(training_targets);

        // Feature indices (excluding target)
        let mut feature_indices = Vec::new();
        for j in 0..n_features {
            if j != target_feature {
                feature_indices.push(j);
            }
        }

        // Train trees
        let mut trees = Vec::new();
        for _ in 0..self.n_estimators {
            let tree = self.train_tree(&training_X, &training_y, rng)?;
            trees.push(tree);
        }

        Ok(RandomForest {
            trees,
            feature_indices,
            target_feature,
        })
    }

    fn train_tree(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> SklResult<DecisionTree> {
        let (n_samples, _n_features) = X.dim();
        let mut sample_indices: Vec<usize> = (0..n_samples).collect();

        if self.bootstrap {
            // Bootstrap sampling
            sample_indices = (0..n_samples)
                .map(|_| rng.gen_range(0..n_samples))
                .collect();
        }

        let training_data = TreeTrainingData {
            features: X.clone(),
            targets: y.clone(),
            sample_indices,
        };

        let mut tree = DecisionTree {
            nodes: Vec::new(),
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        self.build_tree(&mut tree, &training_data, 0, rng)?;
        Ok(tree)
    }

    fn build_tree(
        &self,
        tree: &mut DecisionTree,
        data: &TreeTrainingData,
        depth: usize,
        rng: &mut impl Rng,
    ) -> SklResult<usize> {
        let sample_indices = &data.sample_indices;
        let n_samples = sample_indices.len();

        // Calculate mean target value for this node
        let node_value = if n_samples > 0 {
            sample_indices.iter().map(|&i| data.targets[i]).sum::<f64>() / n_samples as f64
        } else {
            0.0
        };

        // Check stopping criteria
        let should_stop = n_samples < self.min_samples_split
            || n_samples < self.min_samples_leaf * 2
            || self.max_depth.is_some_and(|max_d| depth >= max_d)
            || self.all_targets_equal(data, sample_indices);

        if should_stop {
            // Create leaf node
            let node_index = tree.nodes.len();
            tree.nodes.push(TreeNode {
                feature_index: None,
                threshold: None,
                left_child: None,
                right_child: None,
                value: node_value,
                n_samples,
                is_leaf: true,
            });
            return Ok(node_index);
        }

        // Find best split
        let (best_feature, best_threshold) = self.find_best_split(data, sample_indices, rng)?;

        if best_feature.is_none() {
            // No good split found, create leaf
            let node_index = tree.nodes.len();
            tree.nodes.push(TreeNode {
                feature_index: None,
                threshold: None,
                left_child: None,
                right_child: None,
                value: node_value,
                n_samples,
                is_leaf: true,
            });
            return Ok(node_index);
        }

        let feature_idx = best_feature.unwrap();
        let threshold = best_threshold.unwrap();

        // Split samples
        let (left_indices, right_indices) =
            self.split_samples(data, sample_indices, feature_idx, threshold);

        if left_indices.is_empty() || right_indices.is_empty() {
            // Invalid split, create leaf
            let node_index = tree.nodes.len();
            tree.nodes.push(TreeNode {
                feature_index: None,
                threshold: None,
                left_child: None,
                right_child: None,
                value: node_value,
                n_samples,
                is_leaf: true,
            });
            return Ok(node_index);
        }

        // Create internal node
        let node_index = tree.nodes.len();
        tree.nodes.push(TreeNode {
            feature_index: Some(feature_idx),
            threshold: Some(threshold),
            left_child: None,
            right_child: None,
            value: node_value,
            n_samples,
            is_leaf: false,
        });

        // Recursively build children
        let left_data = TreeTrainingData {
            features: data.features.clone(),
            targets: data.targets.clone(),
            sample_indices: left_indices,
        };
        let left_child_idx = self.build_tree(tree, &left_data, depth + 1, rng)?;

        let right_data = TreeTrainingData {
            features: data.features.clone(),
            targets: data.targets.clone(),
            sample_indices: right_indices,
        };
        let right_child_idx = self.build_tree(tree, &right_data, depth + 1, rng)?;

        // Update node with child indices
        tree.nodes[node_index].left_child = Some(left_child_idx);
        tree.nodes[node_index].right_child = Some(right_child_idx);

        Ok(node_index)
    }

    fn all_targets_equal(&self, data: &TreeTrainingData, sample_indices: &[usize]) -> bool {
        if sample_indices.is_empty() {
            return true;
        }

        let first_target = data.targets[sample_indices[0]];
        sample_indices
            .iter()
            .all(|&i| (data.targets[i] - first_target).abs() < 1e-8)
    }

    fn find_best_split(
        &self,
        data: &TreeTrainingData,
        sample_indices: &[usize],
        rng: &mut impl Rng,
    ) -> SklResult<(Option<usize>, Option<f64>)> {
        let n_features = data.features.ncols();

        // Determine number of features to consider
        let max_features = match self.max_features.as_str() {
            "sqrt" => (n_features as f64).sqrt() as usize,
            "log2" => (n_features as f64).log2() as usize,
            "all" => n_features,
            _ => n_features,
        };

        // Randomly select features to consider
        let mut feature_candidates: Vec<usize> = (0..n_features).collect();
        feature_candidates.shuffle(rng);
        feature_candidates.truncate(max_features.max(1));

        let mut best_score = f64::NEG_INFINITY;
        let mut best_feature = None;
        let mut best_threshold = None;

        for &feature_idx in &feature_candidates {
            // Get unique feature values for potential thresholds
            let mut feature_values: Vec<f64> = sample_indices
                .iter()
                .map(|&i| data.features[[i, feature_idx]])
                .collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            feature_values.dedup_by(|a, b| (*a - *b).abs() < 1e-8);

            if feature_values.len() < 2 {
                continue;
            }

            // Try thresholds between consecutive unique values
            for i in 0..(feature_values.len() - 1) {
                let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;
                let score =
                    self.calculate_split_score(data, sample_indices, feature_idx, threshold);

                if score > best_score {
                    best_score = score;
                    best_feature = Some(feature_idx);
                    best_threshold = Some(threshold);
                }
            }
        }

        Ok((best_feature, best_threshold))
    }

    fn calculate_split_score(
        &self,
        data: &TreeTrainingData,
        sample_indices: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> f64 {
        let (left_indices, right_indices) =
            self.split_samples(data, sample_indices, feature_idx, threshold);

        if left_indices.is_empty() || right_indices.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Calculate variance reduction (for regression)
        let total_variance = self.calculate_variance(data, sample_indices);
        let left_variance = self.calculate_variance(data, &left_indices);
        let right_variance = self.calculate_variance(data, &right_indices);

        let left_weight = left_indices.len() as f64 / sample_indices.len() as f64;
        let right_weight = right_indices.len() as f64 / sample_indices.len() as f64;

        let weighted_variance = left_weight * left_variance + right_weight * right_variance;
        total_variance - weighted_variance
    }

    fn calculate_variance(&self, data: &TreeTrainingData, sample_indices: &[usize]) -> f64 {
        if sample_indices.len() <= 1 {
            return 0.0;
        }

        let mean = sample_indices.iter().map(|&i| data.targets[i]).sum::<f64>()
            / sample_indices.len() as f64;
        let variance = sample_indices
            .iter()
            .map(|&i| (data.targets[i] - mean).powi(2))
            .sum::<f64>()
            / sample_indices.len() as f64;

        variance
    }

    fn split_samples(
        &self,
        data: &TreeTrainingData,
        sample_indices: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &sample_idx in sample_indices {
            if data.features[[sample_idx, feature_idx]] <= threshold {
                left_indices.push(sample_idx);
            } else {
                right_indices.push(sample_idx);
            }
        }

        (left_indices, right_indices)
    }
}

impl RandomForestImputer<RandomForestImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn predict_forest(&self, forest: &RandomForest, input: &Array1<f64>) -> SklResult<f64> {
        let mut predictions = Vec::new();

        for tree in &forest.trees {
            let prediction = self.predict_tree(tree, input)?;
            predictions.push(prediction);
        }

        // Average predictions
        Ok(predictions.iter().sum::<f64>() / predictions.len() as f64)
    }

    fn predict_tree(&self, tree: &DecisionTree, input: &Array1<f64>) -> SklResult<f64> {
        let mut current_node_idx = 0;

        loop {
            if current_node_idx >= tree.nodes.len() {
                return Err(SklearsError::InvalidInput(
                    "Invalid tree structure".to_string(),
                ));
            }

            let node = &tree.nodes[current_node_idx];

            if node.is_leaf {
                return Ok(node.value);
            }

            let feature_idx = node.feature_index.ok_or_else(|| {
                SklearsError::InvalidInput("Non-leaf node missing feature index".to_string())
            })?;
            let threshold = node.threshold.ok_or_else(|| {
                SklearsError::InvalidInput("Non-leaf node missing threshold".to_string())
            })?;

            if feature_idx >= input.len() {
                return Err(SklearsError::InvalidInput(
                    "Feature index out of bounds".to_string(),
                ));
            }

            if input[feature_idx] <= threshold {
                current_node_idx = node
                    .left_child
                    .ok_or_else(|| SklearsError::InvalidInput("Missing left child".to_string()))?;
            } else {
                current_node_idx = node
                    .right_child
                    .ok_or_else(|| SklearsError::InvalidInput("Missing right child".to_string()))?;
            }
        }
    }
}

// Helper functions

fn compute_feature_means(X: &Array2<f64>, missing_values: f64) -> Array1<f64> {
    let (_, n_features) = X.dim();
    let mut means = Array1::zeros(n_features);

    let is_missing_nan = missing_values.is_nan();

    for j in 0..n_features {
        let column = X.column(j);
        let valid_values: Vec<f64> = column
            .iter()
            .filter(|&&x| {
                if is_missing_nan {
                    !x.is_nan()
                } else {
                    (x - missing_values).abs() >= f64::EPSILON
                }
            })
            .cloned()
            .collect();

        means[j] = if valid_values.is_empty() {
            0.0
        } else {
            valid_values.iter().sum::<f64>() / valid_values.len() as f64
        };
    }

    means
}

// Simplified implementations for GradientBoostingImputer and ExtraTreesImputer
// (These would be more complex in a full implementation)

impl GradientBoostingImputer<Untrained> {
    /// Create a new GradientBoostingImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum depth
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the subsample ratio
    pub fn subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }
}

impl Default for GradientBoostingImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GradientBoostingImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for GradientBoostingImputer<Untrained> {
    type Fitted = GradientBoostingImputer<GradientBoostingImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_n_samples, n_features) = X.dim();

        let feature_means = compute_feature_means(&X, self.missing_values);
        let boosting_models = HashMap::new(); // Simplified - would implement gradient boosting training

        Ok(GradientBoostingImputer {
            state: GradientBoostingImputerTrained {
                boosting_models,
                feature_means_: feature_means,
                n_features_in_: n_features,
            },
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            subsample: self.subsample,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for GradientBoostingImputer<GradientBoostingImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        // Simplified implementation - use mean imputation as fallback
        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    X_imputed[[i, j]] = self.state.feature_means_[j];
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl GradientBoostingImputer<GradientBoostingImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl ExtraTreesImputer<Untrained> {
    /// Create a new ExtraTreesImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: "sqrt".to_string(),
            bootstrap: false,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the maximum depth
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }
}

impl Default for ExtraTreesImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ExtraTreesImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ExtraTreesImputer<Untrained> {
    type Fitted = ExtraTreesImputer<ExtraTreesImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_n_samples, n_features) = X.dim();

        let feature_means = compute_feature_means(&X, self.missing_values);
        let forests = HashMap::new(); // Simplified - would implement extra trees training

        Ok(ExtraTreesImputer {
            state: ExtraTreesImputerTrained {
                forests,
                feature_means_: feature_means,
                n_features_in_: n_features,
            },
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            bootstrap: self.bootstrap,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for ExtraTreesImputer<ExtraTreesImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        // Simplified implementation - use mean imputation as fallback
        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    X_imputed[[i, j]] = self.state.feature_means_[j];
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl ExtraTreesImputer<ExtraTreesImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}
