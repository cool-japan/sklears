//! Decision tree implementations for AdaBoost
//!
//! Provides weighted decision-stump classifiers and regressors used as base
//! learners inside AdaBoost.  For AdaBoost, the default depth is 1 (a single
//! decision stump), but arbitrary `max_depth` is also supported.

use super::types::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result,
    traits::{Fit, Trained, Untrained},
    types::{Float, Int},
};
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Impurity helpers
// ---------------------------------------------------------------------------

/// Gini impurity for a slice of integer class labels.
fn gini_impurity(labels: &[Int]) -> Float {
    let n = labels.len();
    if n == 0 {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<Int, usize> = std::collections::HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    let n_f = n as Float;
    1.0 - counts
        .values()
        .map(|&c| (c as Float / n_f).powi(2))
        .sum::<Float>()
}

/// Entropy impurity for a slice of integer class labels.
fn entropy_impurity(labels: &[Int]) -> Float {
    let n = labels.len();
    if n == 0 {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<Int, usize> = std::collections::HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    let n_f = n as Float;
    -counts
        .values()
        .map(|&c| {
            let p = c as Float / n_f;
            if p > 0.0 {
                p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<Float>()
}

/// Variance (MSE) for a slice of float targets — used in regression splits.
fn variance(targets: &[Float]) -> Float {
    let n = targets.len();
    if n == 0 {
        return 0.0;
    }
    let mean = targets.iter().sum::<Float>() / n as Float;
    targets.iter().map(|&v| (v - mean).powi(2)).sum::<Float>() / n as Float
}

/// Most frequent class label (ties broken by smallest label).
fn majority_class(labels: &[Int]) -> Int {
    let mut counts: std::collections::HashMap<Int, usize> = std::collections::HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(label, count)| (count, -(label as i64)))
        .map(|(label, _)| label)
        .unwrap_or(0)
}

/// Mean of a float slice.
fn mean_value(targets: &[Float]) -> Float {
    if targets.is_empty() {
        return 0.0;
    }
    targets.iter().sum::<Float>() / targets.len() as Float
}

// ---------------------------------------------------------------------------
// Classification tree builder
// ---------------------------------------------------------------------------

/// Bundled build parameters to avoid too-many-arguments clippy lint.
struct ClassifierBuildParams {
    criterion: SplitCriterion,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

fn build_classifier(
    x: &Array2<Float>,
    y: &[Int],
    indices: &[usize],
    params: &ClassifierBuildParams,
    depth: usize,
) -> ClassifierNode {
    let labels: Vec<Int> = indices.iter().map(|&i| y[i]).collect();
    let n = indices.len();

    // Base cases
    let all_same = labels.windows(2).all(|w| w[0] == w[1]);
    if n < params.min_samples_split || depth >= params.max_depth || all_same {
        return ClassifierNode::Leaf(majority_class(&labels));
    }

    let impurity_fn: fn(&[Int]) -> Float = match params.criterion {
        SplitCriterion::Gini => gini_impurity,
        SplitCriterion::Entropy => entropy_impurity,
    };
    let parent_impurity = impurity_fn(&labels);

    let n_features = x.ncols();
    let mut best_gain = Float::NEG_INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0_f64;

    for feat in 0..n_features {
        let mut vals: Vec<(Float, Int)> = indices.iter().map(|&i| (x[[i, feat]], y[i])).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for k in 0..vals.len().saturating_sub(1) {
            if (vals[k].0 - vals[k + 1].0).abs() < Float::EPSILON {
                continue;
            }
            let threshold = (vals[k].0 + vals[k + 1].0) / 2.0;

            let left_labels: Vec<Int> = vals[..=k].iter().map(|v| v.1).collect();
            let right_labels: Vec<Int> = vals[k + 1..].iter().map(|v| v.1).collect();

            if left_labels.len() < params.min_samples_leaf
                || right_labels.len() < params.min_samples_leaf
            {
                continue;
            }

            let n_l = left_labels.len() as Float;
            let n_r = right_labels.len() as Float;
            let n_f = n as Float;
            let gain = parent_impurity
                - (n_l / n_f) * impurity_fn(&left_labels)
                - (n_r / n_f) * impurity_fn(&right_labels);

            if gain > best_gain {
                best_gain = gain;
                best_feature = feat;
                best_threshold = threshold;
            }
        }
    }

    if best_gain <= 0.0 {
        return ClassifierNode::Leaf(majority_class(&labels));
    }

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| x[[i, best_feature]] <= best_threshold);

    let left = build_classifier(x, y, &left_idx, params, depth + 1);
    let right = build_classifier(x, y, &right_idx, params, depth + 1);

    ClassifierNode::Split {
        feature_index: best_feature,
        threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn predict_node(node: &ClassifierNode, x_row: &scirs2_core::ndarray::ArrayView1<Float>) -> Int {
    match node {
        ClassifierNode::Leaf(label) => *label,
        ClassifierNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            if x_row[*feature_index] <= *threshold {
                predict_node(left, x_row)
            } else {
                predict_node(right, x_row)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Regression tree builder
// ---------------------------------------------------------------------------

/// Bundled build parameters for regression trees.
struct RegressorBuildParams {
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

fn build_regressor(
    x: &Array2<Float>,
    y: &[Float],
    indices: &[usize],
    params: &RegressorBuildParams,
    depth: usize,
) -> RegressorNode {
    let targets: Vec<Float> = indices.iter().map(|&i| y[i]).collect();
    let n = indices.len();

    if n < params.min_samples_split || depth >= params.max_depth {
        return RegressorNode::Leaf(mean_value(&targets));
    }

    let parent_var = variance(&targets);
    let n_features = x.ncols();
    let mut best_gain = Float::NEG_INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0_f64;

    for feat in 0..n_features {
        let mut vals: Vec<(Float, Float)> = indices.iter().map(|&i| (x[[i, feat]], y[i])).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for k in 0..vals.len().saturating_sub(1) {
            if (vals[k].0 - vals[k + 1].0).abs() < Float::EPSILON {
                continue;
            }
            let threshold = (vals[k].0 + vals[k + 1].0) / 2.0;

            let left_targets: Vec<Float> = vals[..=k].iter().map(|v| v.1).collect();
            let right_targets: Vec<Float> = vals[k + 1..].iter().map(|v| v.1).collect();

            if left_targets.len() < params.min_samples_leaf
                || right_targets.len() < params.min_samples_leaf
            {
                continue;
            }

            let n_l = left_targets.len() as Float;
            let n_r = right_targets.len() as Float;
            let n_f = n as Float;
            let gain = parent_var
                - (n_l / n_f) * variance(&left_targets)
                - (n_r / n_f) * variance(&right_targets);

            if gain > best_gain {
                best_gain = gain;
                best_feature = feat;
                best_threshold = threshold;
            }
        }
    }

    if best_gain <= 0.0 {
        return RegressorNode::Leaf(mean_value(&targets));
    }

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| x[[i, best_feature]] <= best_threshold);

    let left = build_regressor(x, y, &left_idx, params, depth + 1);
    let right = build_regressor(x, y, &right_idx, params, depth + 1);

    RegressorNode::Split {
        feature_index: best_feature,
        threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn predict_reg_node(
    node: &RegressorNode,
    x_row: &scirs2_core::ndarray::ArrayView1<Float>,
) -> Float {
    match node {
        RegressorNode::Leaf(value) => *value,
        RegressorNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            if x_row[*feature_index] <= *threshold {
                predict_reg_node(left, x_row)
            } else {
                predict_reg_node(right, x_row)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeClassifier impl
// ---------------------------------------------------------------------------

impl Default for DecisionTreeClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeClassifier<Untrained> {
    pub fn new() -> Self {
        Self {
            criterion: SplitCriterion::Gini,
            max_depth: Some(1),
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: None,
            state: PhantomData,
            tree_: None,
        }
    }

    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Fit<Array2<Float>, Array1<Int>> for DecisionTreeClassifier<Untrained> {
    type Fitted = DecisionTreeClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Int>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let indices: Vec<usize> = (0..n_samples).collect();
        let y_slice: Vec<Int> = y.iter().cloned().collect();

        let params = ClassifierBuildParams {
            criterion: self.criterion,
            max_depth: self.max_depth.unwrap_or(usize::MAX),
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };
        let root = build_classifier(x, &y_slice, &indices, &params, 0);

        Ok(DecisionTreeClassifier {
            criterion: self.criterion,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            random_state: self.random_state,
            state: PhantomData::<Trained>,
            tree_: Some(DecisionTreeClassifierState { root, n_features }),
        })
    }
}

impl DecisionTreeClassifier<Trained> {
    pub fn predict(&self, x: &Array2<Float>) -> Result<Array1<Int>> {
        let tree = self
            .tree_
            .as_ref()
            .expect("tree_ must be set in trained state");

        let predictions: Vec<Int> = x
            .rows()
            .into_iter()
            .map(|row| predict_node(&tree.root, &row))
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeRegressor impl
// ---------------------------------------------------------------------------

impl Default for DecisionTreeRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeRegressor<Untrained> {
    pub fn new() -> Self {
        Self {
            criterion: SplitCriterion::Gini,
            max_depth: Some(1),
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: None,
            state: PhantomData,
            tree_: None,
        }
    }

    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Fit<Array2<Float>, Array1<Float>> for DecisionTreeRegressor<Untrained> {
    type Fitted = DecisionTreeRegressor<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let indices: Vec<usize> = (0..n_samples).collect();
        let y_slice: Vec<Float> = y.iter().cloned().collect();

        let params = RegressorBuildParams {
            max_depth: self.max_depth.unwrap_or(usize::MAX),
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };
        let root = build_regressor(x, &y_slice, &indices, &params, 0);

        Ok(DecisionTreeRegressor {
            criterion: self.criterion,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            random_state: self.random_state,
            state: PhantomData::<Trained>,
            tree_: Some(DecisionTreeRegressorState { root, n_features }),
        })
    }
}

impl DecisionTreeRegressor<Trained> {
    pub fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let tree = self
            .tree_
            .as_ref()
            .expect("tree_ must be set in trained state");

        let predictions: Vec<Float> = x
            .rows()
            .into_iter()
            .map(|row| predict_reg_node(&tree.root, &row))
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};
    use sklears_core::traits::Fit;

    // -------------------------------------------------------------------
    // Classifier tests
    // -------------------------------------------------------------------

    #[test]
    fn test_stump_perfect_separation() {
        // Feature 0 perfectly separates the classes at threshold 1.5.
        let x: Array2<Float> = array![[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]];
        let y: Array1<Int> = array![0, 0, 1, 1];
        let tree = DecisionTreeClassifier::new()
            .max_depth(1)
            .fit(&x, &y)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        for (i, &p) in preds.iter().enumerate() {
            assert_eq!(
                p, y[i],
                "mismatch at sample {}: predicted {}, expected {}",
                i, p, y[i]
            );
        }
    }

    #[test]
    fn test_stump_output_len() {
        let x: Array2<Float> = array![[1.0], [2.0], [3.0], [4.0]];
        let y: Array1<Int> = array![0, 0, 1, 1];
        let tree = DecisionTreeClassifier::new()
            .fit(&x, &y)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_stump_predictions_are_valid_labels() {
        let x: Array2<Float> = array![[1.0], [2.0], [3.0], [4.0]];
        let y: Array1<Int> = array![0, 0, 1, 1];
        let tree = DecisionTreeClassifier::new()
            .fit(&x, &y)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        for &p in preds.iter() {
            assert!(p == 0 || p == 1, "unexpected label {}", p);
        }
    }

    #[test]
    fn test_deeper_tree_classification() {
        let x: Array2<Float> = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [3.0, 2.0],
        ];
        let y: Array1<Int> = array![0, 0, 1, 1, 2, 2];
        let tree = DecisionTreeClassifier::new()
            .max_depth(3)
            .fit(&x, &y)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        assert_eq!(preds.len(), 6);
        for &p in preds.iter() {
            assert!(p == 0 || p == 1 || p == 2, "unexpected label {}", p);
        }
    }

    // -------------------------------------------------------------------
    // Regressor tests
    // -------------------------------------------------------------------

    #[test]
    fn test_stump_regressor_output_shape() {
        let x: Array2<Float> = array![[1.0], [2.0], [3.0], [4.0]];
        let y: Array1<Float> = array![1.0, 2.0, 3.0, 4.0];
        let tree = DecisionTreeRegressor::new()
            .max_depth(1)
            .fit(&x, &y)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_stump_regressor_constant_output() {
        // All targets the same → leaf predicts constant
        let x: Array2<Float> = array![[1.0], [2.0], [3.0]];
        let y: Array1<Float> = array![5.0, 5.0, 5.0];
        let tree = DecisionTreeRegressor::new()
            .max_depth(1)
            .fit(&x, &y)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        for &p in preds.iter() {
            assert!((p - 5.0).abs() < 1e-10, "expected 5.0, got {}", p);
        }
    }

    #[test]
    fn test_deeper_tree_regression() {
        let x: Array2<Float> = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y: Array1<Float> = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let tree = DecisionTreeRegressor::new()
            .max_depth(4)
            .fit(&x, &y)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        assert_eq!(preds.len(), 6);
        let mse: Float = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<Float>()
            / 6.0;
        assert!(mse < 4.0, "MSE {} too high", mse);
    }
}
