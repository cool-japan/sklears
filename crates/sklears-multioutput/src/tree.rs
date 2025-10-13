//! Tree-based multi-output algorithms
//!
//! This module provides tree-based algorithms for multi-output prediction problems,
//! including decision trees, random forests, and structured predictors.

use crate::multi_label::{BinaryRelevance, BinaryRelevanceTrained};
// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

// Tree-related enums and helper structures

/// Classification criterion for decision trees
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationCriterion {
    /// Gini impurity
    Gini,
    /// Information gain / entropy
    Entropy,
}

/// DAG inference methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DAGInferenceMethod {
    /// Greedy inference following topological order
    Greedy,
    /// Belief propagation on the DAG
    BeliefPropagation,
    /// Integer linear programming for exact inference
    ExactILP,
}

#[derive(Debug, Clone)]
struct DecisionNode {
    is_leaf: bool,
    prediction: Option<Array1<Float>>, // Mean values for each target
    feature_idx: Option<usize>,
    threshold: Option<Float>,
    left: Option<Box<DecisionNode>>,
    right: Option<Box<DecisionNode>>,
    n_samples: usize,
    variance: Float, // Sum of variances across all targets
}

#[derive(Debug, Clone)]
struct ClassificationDecisionNode {
    is_leaf: bool,
    prediction: Option<Array1<i32>>, // Mode/majority class for each target
    probabilities: Option<Array2<Float>>, // Probability distributions per target
    feature_idx: Option<usize>,
    threshold: Option<Float>,
    left: Option<Box<ClassificationDecisionNode>>,
    right: Option<Box<ClassificationDecisionNode>>,
    n_samples: usize,
    impurity: Float, // Combined impurity across all targets
}

/// Multi-Target Regression Tree
///
/// A decision tree regressor that can handle multiple target variables simultaneously.
/// Uses joint variance reduction for optimal splits across all targets.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::MultiTargetRegressionTree;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 1.5], [4.5, 4.5]];
///
/// let tree = MultiTargetRegressionTree::new()
///     .max_depth(Some(3))
///     .min_samples_split(2);
/// let trained_tree = tree.fit(&X.view(), &y).unwrap();
/// let predictions = trained_tree.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiTargetRegressionTree<S = Untrained> {
    state: S,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MultiTargetRegressionTreeTrained {
    tree: DecisionNode,
    n_features: usize,
    n_targets: usize,
    feature_importances: Array1<Float>,
}

impl MultiTargetRegressionTree<Untrained> {
    /// Create a new MultiTargetRegressionTree instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_depth: Some(5),
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: None,
        }
    }

    /// Set the maximum depth of the tree
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split an internal node
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required to be at a leaf node
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Get the maximum depth of the tree
    pub fn get_max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Get the minimum number of samples required to split an internal node
    pub fn get_min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Get the minimum number of samples required to be at a leaf node
    pub fn get_min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Get the random state
    pub fn get_random_state(&self) -> Option<u64> {
        self.random_state
    }
}

impl Default for MultiTargetRegressionTree<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiTargetRegressionTree<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<Float>> for MultiTargetRegressionTree<Untrained> {
    type Fitted = MultiTargetRegressionTree<MultiTargetRegressionTreeTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<Float>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_targets = y.ncols();
        if n_targets == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one target".to_string(),
            ));
        }

        if n_samples < self.min_samples_split {
            return Err(SklearsError::InvalidInput(
                "Number of samples is less than min_samples_split".to_string(),
            ));
        }

        // Build the tree
        let indices: Vec<usize> = (0..n_samples).collect();
        let tree = self.build_tree(&X, y, &indices, 0)?;

        // Calculate feature importances (simplified)
        let mut feature_importances = Array1::<Float>::zeros(n_features);
        self.calculate_feature_importances(&tree, &mut feature_importances, n_samples as Float);

        // Normalize feature importances
        let sum_importances: Float = feature_importances.sum();
        if sum_importances > 0.0 {
            feature_importances /= sum_importances;
        }

        Ok(MultiTargetRegressionTree {
            state: MultiTargetRegressionTreeTrained {
                tree,
                n_features,
                n_targets,
                feature_importances,
            },
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            random_state: self.random_state,
        })
    }
}

impl MultiTargetRegressionTree<Untrained> {
    fn build_tree(
        &self,
        X: &Array2<Float>,
        y: &Array2<Float>,
        indices: &[usize],
        depth: usize,
    ) -> SklResult<DecisionNode> {
        let n_samples = indices.len();
        let n_targets = y.ncols();

        // Calculate current prediction (mean of targets)
        let mut prediction = Array1::<Float>::zeros(n_targets);
        for &idx in indices {
            for j in 0..n_targets {
                prediction[j] += y[[idx, j]];
            }
        }
        prediction /= n_samples as Float;

        // Calculate variance across all targets
        let mut variance = 0.0;
        for &idx in indices {
            for j in 0..n_targets {
                let diff = y[[idx, j]] - prediction[j];
                variance += diff * diff;
            }
        }
        variance /= n_samples as Float;

        // Check stopping criteria
        let should_stop = n_samples < self.min_samples_split
            || n_samples < self.min_samples_leaf
            || self.max_depth.is_some_and(|max_d| depth >= max_d)
            || variance < 1e-10;

        if should_stop {
            return Ok(DecisionNode {
                is_leaf: true,
                prediction: Some(prediction),
                feature_idx: None,
                threshold: None,
                left: None,
                right: None,
                n_samples,
                variance,
            });
        }

        // Find best split
        let (best_feature, best_threshold, best_variance_reduction) =
            self.find_best_split(X, y, indices)?;

        if best_variance_reduction <= 0.0 {
            return Ok(DecisionNode {
                is_leaf: true,
                prediction: Some(prediction),
                feature_idx: None,
                threshold: None,
                left: None,
                right: None,
                n_samples,
                variance,
            });
        }

        // Split the data
        let (left_indices, right_indices) =
            self.split_data(X, indices, best_feature, best_threshold);

        if left_indices.len() < self.min_samples_leaf || right_indices.len() < self.min_samples_leaf
        {
            return Ok(DecisionNode {
                is_leaf: true,
                prediction: Some(prediction),
                feature_idx: None,
                threshold: None,
                left: None,
                right: None,
                n_samples,
                variance,
            });
        }

        // Recursively build child nodes
        let left_child = self.build_tree(X, y, &left_indices, depth + 1)?;
        let right_child = self.build_tree(X, y, &right_indices, depth + 1)?;

        Ok(DecisionNode {
            is_leaf: false,
            prediction: None,
            feature_idx: Some(best_feature),
            threshold: Some(best_threshold),
            left: Some(Box::new(left_child)),
            right: Some(Box::new(right_child)),
            n_samples,
            variance,
        })
    }

    fn find_best_split(
        &self,
        X: &Array2<Float>,
        y: &Array2<Float>,
        indices: &[usize],
    ) -> SklResult<(usize, Float, Float)> {
        let n_features = X.ncols();
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_variance_reduction = 0.0;

        // Calculate current variance
        let current_variance = self.calculate_variance(y, indices);

        for feature_idx in 0..n_features {
            // Get unique feature values
            let mut feature_values: Vec<Float> =
                indices.iter().map(|&idx| X[[idx, feature_idx]]).collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            feature_values.dedup();

            for i in 0..feature_values.len().saturating_sub(1) {
                let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                let (left_indices, right_indices) =
                    self.split_data(X, indices, feature_idx, threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let left_variance = self.calculate_variance(y, &left_indices);
                let right_variance = self.calculate_variance(y, &right_indices);

                let weighted_variance = (left_indices.len() as Float * left_variance
                    + right_indices.len() as Float * right_variance)
                    / indices.len() as Float;

                let variance_reduction = current_variance - weighted_variance;

                if variance_reduction > best_variance_reduction {
                    best_variance_reduction = variance_reduction;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        Ok((best_feature, best_threshold, best_variance_reduction))
    }

    fn calculate_variance(&self, y: &Array2<Float>, indices: &[usize]) -> Float {
        if indices.is_empty() {
            return 0.0;
        }

        let n_targets = y.ncols();
        let n_samples = indices.len();

        // Calculate means
        let mut means = Array1::<Float>::zeros(n_targets);
        for &idx in indices {
            for j in 0..n_targets {
                means[j] += y[[idx, j]];
            }
        }
        means /= n_samples as Float;

        // Calculate variance
        let mut variance = 0.0;
        for &idx in indices {
            for j in 0..n_targets {
                let diff = y[[idx, j]] - means[j];
                variance += diff * diff;
            }
        }
        variance / n_samples as Float
    }

    fn split_data(
        &self,
        X: &Array2<Float>,
        indices: &[usize],
        feature_idx: usize,
        threshold: Float,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if X[[idx, feature_idx]] <= threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        (left_indices, right_indices)
    }

    fn calculate_feature_importances(
        &self,
        node: &DecisionNode,
        importances: &mut Array1<Float>,
        total_samples: Float,
    ) {
        if let (Some(feature_idx), Some(left), Some(right)) =
            (node.feature_idx, &node.left, &node.right)
        {
            let importance = (node.n_samples as Float / total_samples) * node.variance;
            importances[feature_idx] += importance;

            self.calculate_feature_importances(left, importances, total_samples);
            self.calculate_feature_importances(right, importances, total_samples);
        }
    }
}

impl MultiTargetRegressionTree<MultiTargetRegressionTreeTrained> {
    /// Get the feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        &self.state.feature_importances
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.state.n_features
    }

    /// Get the number of targets
    pub fn n_targets(&self) -> usize {
        self.state.n_targets
    }

    /// Predict using the fitted tree
    #[allow(non_snake_case)]
    pub fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = *X;
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_targets));

        for i in 0..n_samples {
            let sample = X.slice(s![i, ..]);
            let prediction = self.predict_single(&self.state.tree, &sample)?;
            for j in 0..self.state.n_targets {
                predictions[[i, j]] = prediction[j];
            }
        }

        Ok(predictions)
    }

    fn predict_single(
        &self,
        node: &DecisionNode,
        sample: &ArrayView1<'_, Float>,
    ) -> SklResult<Array1<Float>> {
        if node.is_leaf {
            if let Some(ref prediction) = node.prediction {
                Ok(prediction.clone())
            } else {
                Err(SklearsError::InvalidInput(
                    "Leaf node without prediction".to_string(),
                ))
            }
        } else {
            let feature_idx = node.feature_idx.ok_or(SklearsError::InvalidInput(
                "Non-leaf node without feature index".to_string(),
            ))?;
            let threshold = node.threshold.ok_or(SklearsError::InvalidInput(
                "Non-leaf node without threshold".to_string(),
            ))?;

            if sample[feature_idx] <= threshold {
                if let Some(ref left) = node.left {
                    self.predict_single(left, sample)
                } else {
                    Err(SklearsError::InvalidInput(
                        "Non-leaf node without left child".to_string(),
                    ))
                }
            } else if let Some(ref right) = node.right {
                self.predict_single(right, sample)
            } else {
                Err(SklearsError::InvalidInput(
                    "Non-leaf node without right child".to_string(),
                ))
            }
        }
    }
}

/// Multi-Target Decision Tree Classifier
///
/// A decision tree classifier that can handle multiple target variables simultaneously.
/// Uses joint entropy/gini reduction for optimal splits across all targets.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::MultiTargetDecisionTreeClassifier;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[0, 1], [1, 0], [1, 1], [0, 0]]; // Two binary classification targets
///
/// let tree = MultiTargetDecisionTreeClassifier::new()
///     .max_depth(Some(3));
/// let trained_tree = tree.fit(&X.view(), &y).unwrap();
/// let predictions = trained_tree.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiTargetDecisionTreeClassifier<S = Untrained> {
    state: S,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    criterion: ClassificationCriterion,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MultiTargetDecisionTreeClassifierTrained {
    tree: ClassificationDecisionNode,
    n_features: usize,
    n_targets: usize,
    feature_importances: Array1<Float>,
    classes_per_target: Vec<Vec<i32>>,
}

impl MultiTargetDecisionTreeClassifier<Untrained> {
    /// Create a new MultiTargetDecisionTreeClassifier instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_depth: Some(5),
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: ClassificationCriterion::Gini,
            random_state: None,
        }
    }

    /// Set the maximum depth of the tree
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split an internal node
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required to be at a leaf node
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the split criterion
    pub fn criterion(mut self, criterion: ClassificationCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for MultiTargetDecisionTreeClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiTargetDecisionTreeClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for MultiTargetDecisionTreeClassifier<Untrained> {
    type Fitted = MultiTargetDecisionTreeClassifier<MultiTargetDecisionTreeClassifierTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_targets = y.ncols();
        if n_targets == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one target".to_string(),
            ));
        }

        // Get unique classes for each target
        let mut classes_per_target = Vec::new();
        for target_idx in 0..n_targets {
            let target_column = y.column(target_idx);
            let mut unique_classes: Vec<i32> = target_column.iter().cloned().collect();
            unique_classes.sort_unstable();
            unique_classes.dedup();
            classes_per_target.push(unique_classes);
        }

        // Initialize feature importances
        let mut feature_importances = Array1::<Float>::zeros(n_features);

        // Build the tree
        let indices: Vec<usize> = (0..n_samples).collect();
        let tree = build_classification_tree(
            &X,
            y,
            &indices,
            &mut feature_importances,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.criterion,
            &classes_per_target,
        )?;

        // Normalize feature importances
        let importance_sum = feature_importances.sum();
        if importance_sum > 0.0 {
            feature_importances /= importance_sum;
        }

        let trained_state = MultiTargetDecisionTreeClassifierTrained {
            tree,
            n_features,
            n_targets,
            feature_importances,
            classes_per_target,
        };

        Ok(MultiTargetDecisionTreeClassifier {
            state: trained_state,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            criterion: self.criterion,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for MultiTargetDecisionTreeClassifier<MultiTargetDecisionTreeClassifierTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_targets));

        for i in 0..n_samples {
            let sample = X.row(i);
            let prediction = predict_classification_sample(&self.state.tree, &sample);
            for j in 0..self.state.n_targets {
                predictions[[i, j]] = prediction[j];
            }
        }

        Ok(predictions)
    }
}

impl MultiTargetDecisionTreeClassifier<MultiTargetDecisionTreeClassifierTrained> {
    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        &self.state.feature_importances
    }

    /// Predict class probabilities for each target
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Vec<Array2<Float>>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut all_probabilities = Vec::new();

        // Initialize probability arrays for each target
        for target_idx in 0..self.state.n_targets {
            let n_classes = self.state.classes_per_target[target_idx].len();
            all_probabilities.push(Array2::<Float>::zeros((n_samples, n_classes)));
        }

        for i in 0..n_samples {
            let sample = X.row(i);
            let probabilities = predict_classification_probabilities(
                &self.state.tree,
                &sample,
                &self.state.classes_per_target,
            );

            for (target_idx, target_probs) in probabilities.iter().enumerate() {
                for (class_idx, &prob) in target_probs.iter().enumerate() {
                    all_probabilities[target_idx][[i, class_idx]] = prob;
                }
            }
        }

        Ok(all_probabilities)
    }
}

/// Random Forest Multi-Output Extension
///
/// A random forest that can handle multiple output variables simultaneously.
/// This implementation creates multiple multi-target regression trees and
/// averages their predictions for robust multi-output regression.
///
/// # Mathematical Foundation
///
/// The random forest combines multiple multi-target regression trees:
/// - Each tree is trained on a bootstrap sample of the data
/// - Each tree considers only a random subset of features at each split
/// - Final prediction is the average of all tree predictions
/// - Feature importance is averaged across all trees
///
/// # Examples
///
/// ```
/// use sklears_multioutput::RandomForestMultiOutput;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 1.5], [4.5, 4.5]];
///
/// let forest = RandomForestMultiOutput::new()
///     .n_estimators(10)
///     .max_depth(Some(3));
/// let trained_forest = forest.fit(&X.view(), &y).unwrap();
/// let predictions = trained_forest.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RandomForestMultiOutput<S = Untrained> {
    state: S,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    bootstrap: bool,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct RandomForestMultiOutputTrained {
    trees: Vec<MultiTargetRegressionTree<MultiTargetRegressionTreeTrained>>,
    n_features: usize,
    n_targets: usize,
    feature_importances: Array1<Float>,
}

impl RandomForestMultiOutput<Untrained> {
    /// Create a new RandomForestMultiOutput instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_estimators: 10,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            bootstrap: true,
            random_state: None,
        }
    }

    /// Set the number of trees in the forest
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the maximum depth of the trees
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split an internal node
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required to be at a leaf node
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the number of features to consider when looking for the best split
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set whether to use bootstrap samples when building trees
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Get the number of trees in the forest
    pub fn get_n_estimators(&self) -> usize {
        self.n_estimators
    }

    /// Get the maximum depth of the trees
    pub fn get_max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Get the minimum number of samples required to split an internal node
    pub fn get_min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Get the minimum number of samples required to be at a leaf node
    pub fn get_min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Get the maximum number of features to consider when looking for the best split
    pub fn get_max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// Get whether bootstrap samples are used when building trees
    pub fn get_bootstrap(&self) -> bool {
        self.bootstrap
    }

    /// Get the random state
    pub fn get_random_state(&self) -> Option<u64> {
        self.random_state
    }
}

impl Default for RandomForestMultiOutput<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RandomForestMultiOutput<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<Float>> for RandomForestMultiOutput<Untrained> {
    type Fitted = RandomForestMultiOutput<RandomForestMultiOutputTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<Float>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_targets = y.ncols();
        if n_targets == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one target".to_string(),
            ));
        }

        let mut trees = Vec::new();
        let mut feature_importances = Array1::<Float>::zeros(n_features);

        for i in 0..self.n_estimators {
            // Create bootstrap sample if needed
            let (X_sample, y_sample) = if self.bootstrap {
                self.create_bootstrap_sample(&X, y, i)?
            } else {
                (X.clone(), y.clone())
            };

            // Create and train tree
            let tree = MultiTargetRegressionTree::new()
                .max_depth(self.max_depth)
                .min_samples_split(self.min_samples_split)
                .min_samples_leaf(self.min_samples_leaf)
                .random_state(self.random_state.map(|s| s.wrapping_add(i as u64)));

            let trained_tree = tree.fit(&X_sample.view(), &y_sample)?;

            // Accumulate feature importances
            feature_importances += trained_tree.feature_importances();

            trees.push(trained_tree);
        }

        // Average feature importances
        feature_importances /= self.n_estimators as Float;

        Ok(RandomForestMultiOutput {
            state: RandomForestMultiOutputTrained {
                trees,
                n_features,
                n_targets,
                feature_importances,
            },
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            bootstrap: self.bootstrap,
            random_state: self.random_state,
        })
    }
}

impl RandomForestMultiOutput<Untrained> {
    fn create_bootstrap_sample(
        &self,
        X: &Array2<Float>,
        y: &Array2<Float>,
        seed: usize,
    ) -> SklResult<(Array2<Float>, Array2<Float>)> {
        let n_samples = X.nrows();
        let mut rng_state = self.random_state.unwrap_or(42).wrapping_add(seed as u64);

        let mut X_sample = Array2::<Float>::zeros(X.raw_dim());
        let mut y_sample = Array2::<Float>::zeros(y.raw_dim());

        for i in 0..n_samples {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let idx = (rng_state / 65536) % (n_samples as u64);

            X_sample
                .slice_mut(s![i, ..])
                .assign(&X.slice(s![idx as usize, ..]));
            y_sample
                .slice_mut(s![i, ..])
                .assign(&y.slice(s![idx as usize, ..]));
        }

        Ok((X_sample, y_sample))
    }
}

impl RandomForestMultiOutput<RandomForestMultiOutputTrained> {
    /// Get the feature importances averaged across all trees
    pub fn feature_importances(&self) -> &Array1<Float> {
        &self.state.feature_importances
    }

    /// Get the number of estimators (trees)
    pub fn n_estimators(&self) -> usize {
        self.state.trees.len()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.state.n_features
    }

    /// Get the number of targets
    pub fn n_targets(&self) -> usize {
        self.state.n_targets
    }

    /// Predict using the forest (average of all tree predictions)
    #[allow(non_snake_case)]
    pub fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = *X;
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_targets));

        // Average predictions from all trees
        for tree in &self.state.trees {
            let tree_predictions = tree.predict(&X)?;
            predictions += &tree_predictions;
        }

        predictions /= self.state.trees.len() as Float;
        Ok(predictions)
    }
}

/// Tree-Structured Prediction
///
/// A structured prediction method for outputs that follow a tree structure.
/// Each internal node in the tree has a classifier that decides which branch to take,
/// enabling hierarchical multi-label classification.
///
/// # Examples
///
/// ```
/// use sklears_core::traits::{Predict, Fit};
/// use sklears_multioutput::TreeStructuredPredictor;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0]];
/// let y = array![[0, 1, 2], [1, 2, 0]]; // Tree paths
///
/// let tree_predictor = TreeStructuredPredictor::new()
///     .max_depth(3)
///     .branching_factor(3);
/// let trained_predictor = tree_predictor.fit(&X, &y).unwrap();
/// let predictions = trained_predictor.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TreeStructuredPredictor<State = Untrained> {
    max_depth: usize,
    branching_factor: usize,
    tree_structure: Vec<Vec<usize>>, // Adjacency list representation
    node_classifiers: HashMap<usize, String>,
    state: State,
}

/// Trained state for Tree-Structured Predictor
#[derive(Debug, Clone)]
pub struct TreeStructuredPredictorTrained {
    node_classifiers: HashMap<usize, BinaryRelevance<BinaryRelevanceTrained>>,
    tree_structure: Vec<Vec<usize>>,
    max_depth: usize,
    n_nodes: usize,
}

impl Default for TreeStructuredPredictor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeStructuredPredictor<Untrained> {
    /// Create new tree-structured predictor
    pub fn new() -> Self {
        Self {
            max_depth: 5,
            branching_factor: 2,
            tree_structure: Vec::new(),
            node_classifiers: HashMap::new(),
            state: Untrained,
        }
    }

    /// Set maximum tree depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set branching factor
    pub fn branching_factor(mut self, factor: usize) -> Self {
        self.branching_factor = factor;
        self
    }

    /// Set custom tree structure
    pub fn tree_structure(mut self, structure: Vec<Vec<usize>>) -> Self {
        self.tree_structure = structure;
        self
    }
}

impl Estimator for TreeStructuredPredictor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<i32>> for TreeStructuredPredictor<Untrained> {
    type Fitted = TreeStructuredPredictor<TreeStructuredPredictorTrained>;

    fn fit(self, X: &Array2<Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_samples, _n_features) = X.dim();
        let (y_samples, max_path_length) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Build tree structure if not provided
        let tree_structure = if self.tree_structure.is_empty() {
            self.build_default_tree_structure()?
        } else {
            self.tree_structure.clone()
        };

        let n_nodes = tree_structure.len();
        let mut node_classifiers = HashMap::new();

        // Train classifier for each internal node
        for node_id in 0..n_nodes {
            if !tree_structure[node_id].is_empty() {
                // Internal node
                // Create binary classification data for this node
                let (node_X, node_y) = self.create_node_training_data(
                    &X.view(),
                    &y.view(),
                    node_id,
                    &tree_structure,
                    max_path_length,
                )?;

                if !node_y.is_empty() {
                    let classifier = BinaryRelevance::new();
                    let trained_classifier = classifier.fit(&node_X.view(), &node_y)?;
                    node_classifiers.insert(node_id, trained_classifier);
                }
            }
        }

        Ok(TreeStructuredPredictor {
            max_depth: self.max_depth,
            branching_factor: self.branching_factor,
            tree_structure: tree_structure.clone(),
            node_classifiers: HashMap::new(),
            state: TreeStructuredPredictorTrained {
                node_classifiers,
                tree_structure,
                max_depth: self.max_depth,
                n_nodes,
            },
        })
    }
}

impl TreeStructuredPredictor<Untrained> {
    /// Build default tree structure
    fn build_default_tree_structure(&self) -> SklResult<Vec<Vec<usize>>> {
        let mut total_nodes = 0;
        for depth in 0..self.max_depth {
            total_nodes += self.branching_factor.pow(depth as u32);
        }

        let mut tree_structure = vec![Vec::new(); total_nodes];
        let mut node_id = 0;

        // Build complete tree
        for depth in 0..(self.max_depth - 1) {
            let nodes_at_depth = self.branching_factor.pow(depth as u32);

            for _ in 0..nodes_at_depth {
                for child in 0..self.branching_factor {
                    let child_id = node_id + nodes_at_depth + child;
                    if child_id < total_nodes {
                        tree_structure[node_id].push(child_id);
                    }
                }
                node_id += 1;
            }
        }

        Ok(tree_structure)
    }

    /// Create training data for a specific node
    fn create_node_training_data(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView2<i32>,
        node_id: usize,
        tree_structure: &Vec<Vec<usize>>,
        max_path_length: usize,
    ) -> SklResult<(Array2<Float>, Array2<i32>)> {
        let n_samples = X.nrows();
        let mut valid_samples = Vec::new();
        let mut node_labels = Vec::new();

        for sample_idx in 0..n_samples {
            let path = y.row(sample_idx);

            // Check if this sample's path passes through this node
            for pos in 0..max_path_length {
                if path[pos] as usize == node_id && pos + 1 < max_path_length {
                    // Sample passes through this node, determine which child to predict
                    let next_node = path[pos + 1] as usize;

                    // Find which child index this corresponds to
                    if let Some(child_idx) = tree_structure[node_id]
                        .iter()
                        .position(|&child| child == next_node)
                    {
                        valid_samples.push(sample_idx);
                        node_labels.push(child_idx as i32);
                        break;
                    }
                }
            }
        }

        // Create training data for this node
        let n_valid = valid_samples.len();
        if n_valid == 0 {
            return Ok((
                Array2::<Float>::zeros((0, X.ncols())),
                Array2::<i32>::zeros((0, 1)),
            ));
        }

        let mut node_X = Array2::<Float>::zeros((n_valid, X.ncols()));
        let mut node_y = Array2::<i32>::zeros((n_valid, 1));

        for (i, &sample_idx) in valid_samples.iter().enumerate() {
            for j in 0..X.ncols() {
                node_X[[i, j]] = X[[sample_idx, j]];
            }
            node_y[[i, 0]] = node_labels[i];
        }

        Ok((node_X, node_y))
    }
}

impl Predict<Array2<Float>, Array2<i32>>
    for TreeStructuredPredictor<TreeStructuredPredictorTrained>
{
    fn predict(&self, X: &Array2<Float>) -> SklResult<Array2<i32>> {
        let n_samples = X.nrows();
        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.max_depth));

        for sample_idx in 0..n_samples {
            let sample = X.row(sample_idx);
            let path = self.predict_tree_path(&sample)?;

            for (pos, &node) in path.iter().enumerate() {
                if pos < self.state.max_depth {
                    predictions[[sample_idx, pos]] = node as i32;
                }
            }
        }

        Ok(predictions)
    }
}

impl TreeStructuredPredictor<TreeStructuredPredictorTrained> {
    /// Predict tree path for a single sample
    fn predict_tree_path(&self, sample: &ArrayView1<Float>) -> SklResult<Vec<usize>> {
        let mut path = Vec::new();
        let mut current_node = 0; // Start at root
        path.push(current_node);

        while !self.state.tree_structure[current_node].is_empty() {
            // Internal node - predict which child to take
            if let Some(classifier) = self.state.node_classifiers.get(&current_node) {
                let sample_2d = sample.to_owned().insert_axis(scirs2_core::ndarray::Axis(0));
                let prediction = classifier.predict(&sample_2d.view())?;
                let child_idx = prediction[[0, 0]] as usize;

                if child_idx < self.state.tree_structure[current_node].len() {
                    current_node = self.state.tree_structure[current_node][child_idx];
                    path.push(current_node);
                } else {
                    break; // Invalid prediction
                }
            } else {
                break; // No classifier for this node
            }
        }

        Ok(path)
    }

    /// Get tree structure
    pub fn tree_structure(&self) -> &Vec<Vec<usize>> {
        &self.state.tree_structure
    }
}

// Supporting functions for tree algorithms

/// Build a classification decision tree recursively
pub fn build_classification_tree(
    X: &Array2<Float>,
    y: &Array2<i32>,
    indices: &[usize],
    feature_importances: &mut Array1<Float>,
    depth: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    criterion: ClassificationCriterion,
    classes_per_target: &[Vec<i32>],
) -> SklResult<ClassificationDecisionNode> {
    let n_samples = indices.len();

    // Calculate current impurity and predictions
    let (current_impurity, prediction, probabilities) =
        calculate_classification_metrics(y, indices, classes_per_target, criterion);

    // Check stopping criteria
    let should_stop = n_samples < min_samples_split
        || (max_depth.is_some() && depth >= max_depth.unwrap())
        || current_impurity == 0.0;

    if should_stop {
        return Ok(ClassificationDecisionNode {
            is_leaf: true,
            prediction: Some(prediction),
            probabilities: Some(probabilities),
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            n_samples,
            impurity: current_impurity,
        });
    }

    // Find best split
    let mut best_impurity_reduction = 0.0;
    let mut best_feature = None;
    let mut best_threshold = None;
    let mut best_left_indices = Vec::new();
    let mut best_right_indices = Vec::new();

    for feature_idx in 0..X.ncols() {
        // Get unique values for this feature in current samples
        let mut feature_values: Vec<Float> = indices.iter().map(|&i| X[[i, feature_idx]]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        feature_values.dedup();

        // Try splits between consecutive unique values
        for i in 0..feature_values.len().saturating_sub(1) {
            let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&idx| X[[idx, feature_idx]] <= threshold);

            // Check minimum samples per leaf
            if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
                continue;
            }

            // Calculate impurity reduction
            let (left_impurity, _, _) =
                calculate_classification_metrics(y, &left_indices, classes_per_target, criterion);
            let (right_impurity, _, _) =
                calculate_classification_metrics(y, &right_indices, classes_per_target, criterion);

            let weighted_impurity = (left_indices.len() as Float * left_impurity
                + right_indices.len() as Float * right_impurity)
                / n_samples as Float;
            let impurity_reduction = current_impurity - weighted_impurity;

            if impurity_reduction > best_impurity_reduction {
                best_impurity_reduction = impurity_reduction;
                best_feature = Some(feature_idx);
                best_threshold = Some(threshold);
                best_left_indices = left_indices;
                best_right_indices = right_indices;
            }
        }
    }

    // If no good split found, create leaf
    if best_feature.is_none() || best_impurity_reduction <= 0.0 {
        return Ok(ClassificationDecisionNode {
            is_leaf: true,
            prediction: Some(prediction),
            probabilities: Some(probabilities),
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            n_samples,
            impurity: current_impurity,
        });
    }

    // Update feature importance
    feature_importances[best_feature.unwrap()] += best_impurity_reduction * n_samples as Float;

    // Recursively build left and right subtrees
    let left_child = build_classification_tree(
        X,
        y,
        &best_left_indices,
        feature_importances,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        criterion,
        classes_per_target,
    )?;

    let right_child = build_classification_tree(
        X,
        y,
        &best_right_indices,
        feature_importances,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        criterion,
        classes_per_target,
    )?;

    Ok(ClassificationDecisionNode {
        is_leaf: false,
        prediction: Some(prediction),
        probabilities: Some(probabilities),
        feature_idx: best_feature,
        threshold: best_threshold,
        left: Some(Box::new(left_child)),
        right: Some(Box::new(right_child)),
        n_samples,
        impurity: current_impurity,
    })
}

/// Calculate classification metrics for a set of samples
pub fn calculate_classification_metrics(
    y: &Array2<i32>,
    indices: &[usize],
    classes_per_target: &[Vec<i32>],
    criterion: ClassificationCriterion,
) -> (Float, Array1<i32>, Array2<Float>) {
    let n_targets = y.ncols();
    let n_samples = indices.len();

    let mut prediction = Array1::<i32>::zeros(n_targets);
    let mut total_impurity = 0.0;

    // Calculate max number of classes across all targets for probability matrix
    let max_classes = classes_per_target
        .iter()
        .map(|classes| classes.len())
        .max()
        .unwrap_or(0);
    let mut probabilities = Array2::<Float>::zeros((n_targets, max_classes));

    for target_idx in 0..n_targets {
        let classes = &classes_per_target[target_idx];
        let n_classes = classes.len();

        // Count class frequencies
        let mut class_counts = vec![0; n_classes];
        for &sample_idx in indices {
            let class_label = y[[sample_idx, target_idx]];
            if let Some(class_idx) = classes.iter().position(|&c| c == class_label) {
                class_counts[class_idx] += 1;
            }
        }

        // Find majority class
        let majority_class_idx = class_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        prediction[target_idx] = classes[majority_class_idx];

        // Calculate probabilities and impurity
        let mut target_impurity = 0.0;
        for (class_idx, &count) in class_counts.iter().enumerate() {
            let prob = count as Float / n_samples as Float;
            probabilities[[target_idx, class_idx]] = prob;

            if prob > 0.0 {
                target_impurity += match criterion {
                    ClassificationCriterion::Gini => prob * (1.0 - prob),
                    ClassificationCriterion::Entropy => -prob * prob.ln(),
                };
            }
        }

        // For Gini, multiply by 2; for Entropy, it's already correct
        if matches!(criterion, ClassificationCriterion::Gini) {
            target_impurity *= 2.0;
        }

        total_impurity += target_impurity;
    }

    // Average impurity across targets
    total_impurity /= n_targets as Float;

    (total_impurity, prediction, probabilities)
}

/// Predict for a single classification sample
pub fn predict_classification_sample(
    node: &ClassificationDecisionNode,
    sample: &ArrayView1<Float>,
) -> Array1<i32> {
    if node.is_leaf {
        return node.prediction.as_ref().unwrap().clone();
    }

    let feature_value = sample[node.feature_idx.unwrap()];
    let threshold = node.threshold.unwrap();

    if feature_value <= threshold {
        predict_classification_sample(node.left.as_ref().unwrap(), sample)
    } else {
        predict_classification_sample(node.right.as_ref().unwrap(), sample)
    }
}

/// Predict probabilities for a single classification sample
pub fn predict_classification_probabilities(
    node: &ClassificationDecisionNode,
    sample: &ArrayView1<Float>,
    classes_per_target: &[Vec<i32>],
) -> Vec<Array1<Float>> {
    if node.is_leaf {
        let mut result = Vec::new();
        for target_idx in 0..classes_per_target.len() {
            let n_classes = classes_per_target[target_idx].len();
            let mut target_probs = Array1::<Float>::zeros(n_classes);
            for class_idx in 0..n_classes {
                target_probs[class_idx] =
                    node.probabilities.as_ref().unwrap()[[target_idx, class_idx]];
            }
            result.push(target_probs);
        }
        return result;
    }

    let feature_value = sample[node.feature_idx.unwrap()];
    let threshold = node.threshold.unwrap();

    if feature_value <= threshold {
        predict_classification_probabilities(
            node.left.as_ref().unwrap(),
            sample,
            classes_per_target,
        )
    } else {
        predict_classification_probabilities(
            node.right.as_ref().unwrap(),
            sample,
            classes_per_target,
        )
    }
}
