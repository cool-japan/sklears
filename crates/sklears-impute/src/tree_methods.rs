//! Tree-based imputation methods
//!
//! This module contains imputation methods using decision trees and related algorithms.

use itertools::Itertools;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Simple decision tree for imputation
///
/// This is a basic implementation suitable for imputation purposes.
/// For more complex scenarios, consider using the full decision tree from sklears-tree.
#[derive(Debug, Clone)]
pub struct DecisionTreeImputer<S = Untrained> {
    state: S,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for DecisionTreeImputer
#[derive(Debug, Clone)]
pub struct DecisionTreeImputerTrained {
    trees: Vec<DecisionTree>,
    n_features_in_: usize,
    feature_means_: Array1<f64>,
}

/// Simple decision tree node
#[derive(Debug, Clone)]
struct TreeNode {
    feature: Option<usize>,
    threshold: Option<f64>,
    value: Option<f64>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

/// Simple decision tree
#[derive(Debug, Clone)]
struct DecisionTree {
    root: TreeNode,
}

impl DecisionTreeImputer<Untrained> {
    /// Create a new DecisionTreeImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_depth: Some(5),
            min_samples_split: 10,
            min_samples_leaf: 5,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the maximum depth of the tree
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum samples required to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples required in a leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
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

impl Default for DecisionTreeImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DecisionTreeImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for DecisionTreeImputer<Untrained> {
    type Fitted = DecisionTreeImputer<DecisionTreeImputerTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        // Calculate feature means for fallback
        let mut feature_means_ = Array1::zeros(n_features);
        for j in 0..n_features {
            let column = X.column(j);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if !valid_values.is_empty() {
                feature_means_[j] = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            }
        }

        // Train a decision tree for each feature
        let mut trees = Vec::new();

        for target_feature in 0..n_features {
            // Collect samples where target feature is observed
            let mut training_data = Vec::new();
            let mut training_targets = Vec::new();

            for i in 0..X.nrows() {
                if !self.is_missing(X[[i, target_feature]]) {
                    let mut features = Vec::new();
                    let mut has_missing = false;

                    for j in 0..n_features {
                        if j != target_feature {
                            if self.is_missing(X[[i, j]]) {
                                // For simplicity, use mean imputation in training
                                features.push(feature_means_[j]);
                                has_missing = true;
                            } else {
                                features.push(X[[i, j]]);
                            }
                        }
                    }

                    // Only use samples with at least some non-missing features
                    if !has_missing || features.iter().any(|&x| !x.is_nan()) {
                        training_data.push(features);
                        training_targets.push(X[[i, target_feature]]);
                    }
                }
            }

            if training_data.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "No valid training data for feature {}",
                    target_feature
                )));
            }

            // Build decision tree
            let tree = build_regression_tree(
                &training_data,
                &training_targets,
                self.max_depth.unwrap_or(5),
                self.min_samples_split,
                self.min_samples_leaf,
            );

            trees.push(tree);
        }

        Ok(DecisionTreeImputer {
            state: DecisionTreeImputerTrained {
                trees,
                n_features_in_: n_features,
                feature_means_,
            },
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for DecisionTreeImputer<DecisionTreeImputerTrained>
{
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

        // Impute each feature using its corresponding tree
        for target_feature in 0..n_features {
            let tree = &self.state.trees[target_feature];

            for i in 0..n_samples {
                if self.is_missing(X[[i, target_feature]]) {
                    // Prepare features for prediction (excluding target feature)
                    let mut features = Vec::new();
                    for j in 0..n_features {
                        if j != target_feature {
                            if self.is_missing(X_imputed[[i, j]]) {
                                // Use mean if other features are also missing
                                features.push(self.state.feature_means_[j]);
                            } else {
                                features.push(X_imputed[[i, j]]);
                            }
                        }
                    }

                    // Predict using the tree
                    let prediction = tree.predict(&features);
                    X_imputed[[i, target_feature]] = prediction;
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl DecisionTreeImputer<DecisionTreeImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl DecisionTree {
    fn predict(&self, features: &[f64]) -> f64 {
        self.root.predict(features)
    }
}

impl TreeNode {
    fn predict(&self, features: &[f64]) -> f64 {
        match (&self.feature, &self.threshold, &self.value) {
            (Some(feature_idx), Some(threshold), _) => {
                if *feature_idx < features.len() && features[*feature_idx] <= *threshold {
                    if let Some(ref left) = self.left {
                        left.predict(features)
                    } else {
                        self.value.unwrap_or(0.0)
                    }
                } else if let Some(ref right) = self.right {
                    right.predict(features)
                } else {
                    self.value.unwrap_or(0.0)
                }
            }
            (_, _, Some(value)) => *value,
            _ => 0.0,
        }
    }
}

/// Build a simple regression tree
pub fn build_regression_tree(
    data: &[Vec<f64>],
    targets: &[f64],
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> DecisionTree {
    let root = build_tree_node(
        data,
        targets,
        0,
        max_depth,
        min_samples_split,
        min_samples_leaf,
    );
    DecisionTree { root }
}

fn build_tree_node(
    data: &[Vec<f64>],
    targets: &[f64],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> TreeNode {
    // Base case: return leaf node
    if data.len() < min_samples_split || depth >= max_depth {
        let mean_value = targets.iter().sum::<f64>() / targets.len() as f64;
        return TreeNode {
            feature: None,
            threshold: None,
            value: Some(mean_value),
            left: None,
            right: None,
        };
    }

    // Find best split
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_score = f64::INFINITY;

    if data.is_empty() {
        return TreeNode {
            feature: None,
            threshold: None,
            value: Some(0.0),
            left: None,
            right: None,
        };
    }

    let n_features = data[0].len();

    for feature_idx in 0..n_features {
        let mut feature_values: Vec<f64> = data.iter().map(|row| row[feature_idx]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        feature_values.dedup();

        for &threshold in &feature_values {
            let (left_targets, right_targets): (Vec<f64>, Vec<f64>) = data
                .iter()
                .zip(targets.iter())
                .partition_map(|(row, &target)| {
                    if row[feature_idx] <= threshold {
                        itertools::Either::Left(target)
                    } else {
                        itertools::Either::Right(target)
                    }
                });

            if left_targets.len() < min_samples_leaf || right_targets.len() < min_samples_leaf {
                continue;
            }

            let left_mse = calculate_mse(&left_targets);
            let right_mse = calculate_mse(&right_targets);
            let weighted_mse = (left_targets.len() as f64 * left_mse
                + right_targets.len() as f64 * right_mse)
                / data.len() as f64;

            if weighted_mse < best_score {
                best_score = weighted_mse;
                best_feature = feature_idx;
                best_threshold = threshold;
            }
        }
    }

    // If no good split found, return leaf
    if best_score == f64::INFINITY {
        let mean_value = targets.iter().sum::<f64>() / targets.len() as f64;
        return TreeNode {
            feature: None,
            threshold: None,
            value: Some(mean_value),
            left: None,
            right: None,
        };
    }

    // Split data
    let (left_pairs, right_pairs): (Vec<_>, Vec<_>) = data
        .iter()
        .zip(targets.iter())
        .partition_map(|(row, &target)| {
            if row[best_feature] <= best_threshold {
                itertools::Either::Left((row.clone(), target))
            } else {
                itertools::Either::Right((row.clone(), target))
            }
        });

    let (left_data, left_targets): (Vec<Vec<f64>>, Vec<f64>) = left_pairs.into_iter().unzip();
    let (right_data, right_targets): (Vec<Vec<f64>>, Vec<f64>) = right_pairs.into_iter().unzip();

    // Recursively build subtrees
    let left_child = if !left_data.is_empty() {
        Some(Box::new(build_tree_node(
            &left_data,
            &left_targets,
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
        )))
    } else {
        None
    };

    let right_child = if !right_data.is_empty() {
        Some(Box::new(build_tree_node(
            &right_data,
            &right_targets,
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
        )))
    } else {
        None
    };

    TreeNode {
        feature: Some(best_feature),
        threshold: Some(best_threshold),
        value: None,
        left: left_child,
        right: right_child,
    }
}

pub fn calculate_mse(targets: &[f64]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }

    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    targets.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / targets.len() as f64
}