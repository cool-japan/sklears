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

/// The winner of the greedy split search at a single regression node.
///
/// A `gain` of [`Float::NEG_INFINITY`] means no admissible candidate existed at
/// all (every feature constant, or every cut point rejected by
/// `min_samples_leaf`); the caller then turns the node into a leaf.
#[derive(Debug, Clone, Copy)]
struct BestRegressionSplit {
    feature_index: usize,
    threshold: Float,
    gain: Float,
}

/// Scratch buffers reused by every feature of a single node, so the split
/// search performs zero per-candidate heap allocation.
struct SplitSearchBuffers {
    /// `(feature value, mean-centered target)`, sorted by feature value.
    pairs: Vec<(Float, Float)>,
    /// `prefix_sum[k] = sum_{j <= k} pairs[j].1`.
    prefix_sum: Vec<Float>,
    /// `prefix_sq[k] = sum_{j <= k} pairs[j].1 * pairs[j].1`.
    prefix_sq: Vec<Float>,
}

impl SplitSearchBuffers {
    fn with_capacity(n: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(n),
            prefix_sum: Vec::with_capacity(n),
            prefix_sq: Vec::with_capacity(n),
        }
    }
}

/// Exhaustive greedy variance-reduction split search over every feature, in
/// `O(F * n log n)` per node.
///
/// For each feature the `(value, target)` pairs are sorted once and prefix sums
/// of the mean-centered targets and of their squares are accumulated; each of
/// the `n - 1` candidate cut points is then scored in `O(1)` from
/// `SSE = sum(t^2) - (sum t)^2 / n`, instead of materializing two child `Vec`s
/// and re-running two `O(n)` [`variance`] passes per candidate (which made the
/// original search `O(F * n^2)` with `2 * (n - 1)` allocations per feature).
///
/// The result is a drop-in replacement for that original loop: the sort is the
/// same stable comparison on the feature value, candidates between equal
/// adjacent values are skipped by the same `Float::EPSILON` test, the threshold
/// is the same midpoint, `min_samples_leaf` is applied at the same point, the
/// criterion is the same weighted variance reduction, and the winner is still
/// the first candidate that *strictly* exceeds the running best.
///
/// Subtracting `node_mean` before accumulating is exact in real arithmetic
/// (sums of squared deviations are translation invariant) and makes the
/// incremental form at least as accurate as the two-pass one it replaces, by
/// removing the cancellation a raw `E[t^2] - E[t]^2` would suffer when the
/// targets are far from zero.
fn find_best_regression_split(
    x: &Array2<Float>,
    y: &[Float],
    indices: &[usize],
    parent_var: Float,
    node_mean: Float,
    min_samples_leaf: usize,
    buffers: &mut SplitSearchBuffers,
) -> BestRegressionSplit {
    let mut best = BestRegressionSplit {
        feature_index: 0,
        threshold: 0.0,
        gain: Float::NEG_INFINITY,
    };

    let n = indices.len();
    if n < 2 {
        return best;
    }
    let n_f = n as Float;

    let SplitSearchBuffers {
        pairs,
        prefix_sum,
        prefix_sq,
    } = buffers;

    for feat in 0..x.ncols() {
        pairs.clear();
        pairs.extend(indices.iter().map(|&i| (x[[i, feat]], y[i] - node_mean)));
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        prefix_sum.clear();
        prefix_sq.clear();
        let mut running_sum = 0.0;
        let mut running_sq = 0.0;
        for &(_, target) in pairs.iter() {
            running_sum += target;
            running_sq += target * target;
            prefix_sum.push(running_sum);
            prefix_sq.push(running_sq);
        }
        let total_sum = running_sum;
        let total_sq = running_sq;

        for k in 0..n - 1 {
            let value = pairs[k].0;
            let next_value = pairs[k + 1].0;
            if (value - next_value).abs() < Float::EPSILON {
                continue;
            }

            let n_left = k + 1;
            let n_right = n - n_left;
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let sum_left = prefix_sum[k];
            let sq_left = prefix_sq[k];
            let sum_right = total_sum - sum_left;
            let sq_right = total_sq - sq_left;

            let n_l = n_left as Float;
            let n_r = n_right as Float;
            // Population variance of each side. `max(0.0)` only absorbs the
            // rounding noise that can push a numerically-zero SSE slightly
            // negative; `variance()` is non-negative by construction.
            let var_left = (sq_left - sum_left * sum_left / n_l).max(0.0) / n_l;
            let var_right = (sq_right - sum_right * sum_right / n_r).max(0.0) / n_r;

            let gain = parent_var - (n_l / n_f) * var_left - (n_r / n_f) * var_right;

            if gain > best.gain {
                best = BestRegressionSplit {
                    feature_index: feat,
                    threshold: (value + next_value) / 2.0,
                    gain,
                };
            }
        }
    }

    best
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
    let node_mean = mean_value(&targets);
    let mut buffers = SplitSearchBuffers::with_capacity(n);
    let best = find_best_regression_split(
        x,
        y,
        indices,
        parent_var,
        node_mean,
        params.min_samples_leaf,
        &mut buffers,
    );

    if best.gain <= 0.0 {
        return RegressorNode::Leaf(mean_value(&targets));
    }

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| x[[i, best.feature_index]] <= best.threshold);

    let left = build_regressor(x, y, &left_idx, params, depth + 1);
    let right = build_regressor(x, y, &right_idx, params, depth + 1);

    RegressorNode::Split {
        feature_index: best.feature_index,
        threshold: best.threshold,
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

    // -------------------------------------------------------------------
    // Split-search equivalence: prefix-sum fast path vs naive O(n^2) oracle
    // -------------------------------------------------------------------

    /// Deterministic xorshift64* PRNG, so every case below is seeded and
    /// reproducible without adding a dependency.
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut z = self.0;
            z ^= z >> 12;
            z ^= z << 25;
            z ^= z >> 27;
            self.0 = z;
            z.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// Uniform in `[0, 1)`.
        fn next_f64(&mut self) -> Float {
            (self.next_u64() >> 11) as Float / (1u64 << 53) as Float
        }

        /// Uniform in `0..upper` (`upper` must be non-zero).
        fn next_usize(&mut self, upper: usize) -> usize {
            (self.next_u64() % upper as u64) as usize
        }
    }

    /// Verbatim retention of the original `O(F * n^2)` split search: for every
    /// candidate threshold it materializes both child target vectors and runs
    /// two full [`variance`] passes. Kept as the test-only oracle that
    /// [`find_best_regression_split`] must reproduce.
    fn naive_best_regression_split(
        x: &Array2<Float>,
        y: &[Float],
        indices: &[usize],
        parent_var: Float,
        min_samples_leaf: usize,
    ) -> BestRegressionSplit {
        let n = indices.len();
        let mut best_gain = Float::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0_f64;

        for feat in 0..x.ncols() {
            let mut vals: Vec<(Float, Float)> =
                indices.iter().map(|&i| (x[[i, feat]], y[i])).collect();
            vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            for k in 0..vals.len().saturating_sub(1) {
                if (vals[k].0 - vals[k + 1].0).abs() < Float::EPSILON {
                    continue;
                }
                let threshold = (vals[k].0 + vals[k + 1].0) / 2.0;

                let left_targets: Vec<Float> = vals[..=k].iter().map(|v| v.1).collect();
                let right_targets: Vec<Float> = vals[k + 1..].iter().map(|v| v.1).collect();

                if left_targets.len() < min_samples_leaf || right_targets.len() < min_samples_leaf {
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

        BestRegressionSplit {
            feature_index: best_feature,
            threshold: best_threshold,
            gain: best_gain,
        }
    }

    /// A random design matrix whose columns are a mix of continuous values,
    /// heavily duplicated low-cardinality values, and outright constants.
    fn random_dataset(
        rng: &mut TestRng,
        n: usize,
        n_features: usize,
    ) -> (Array2<Float>, Vec<Float>) {
        let mut columns: Vec<Vec<Float>> = Vec::with_capacity(n_features);
        for feat in 0..n_features {
            // Feature 0 is always continuous so at least one column can split;
            // the rest cycle through duplicated and constant personalities.
            let kind = if feat == 0 { 0 } else { rng.next_usize(3) };
            let column: Vec<Float> = match kind {
                0 => (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect(),
                1 => {
                    let levels = 1 + rng.next_usize(4);
                    (0..n).map(|_| rng.next_usize(levels) as Float).collect()
                }
                _ => {
                    let constant = rng.next_f64();
                    vec![constant; n]
                }
            };
            columns.push(column);
        }

        let mut data = Vec::with_capacity(n * n_features);
        for row in 0..n {
            for column in &columns {
                data.push(column[row]);
            }
        }
        let x = Array2::from_shape_vec((n, n_features), data)
            .expect("generated shape matches data length");
        let y: Vec<Float> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        (x, y)
    }

    /// `actual` must match `expected` to 1e-9 relative (with a small absolute
    /// floor), treating `-inf` (no admissible split) as an exact sentinel.
    fn assert_close(actual: Float, expected: Float, label: &str, case: usize) {
        if actual.is_infinite() || expected.is_infinite() {
            assert_eq!(
                actual, expected,
                "case {case}: {label} sentinel differs (fast {actual}, naive {expected})"
            );
            return;
        }
        let tolerance = 1e-9 * expected.abs() + 1e-12;
        assert!(
            (actual - expected).abs() <= tolerance,
            "case {case}: {label} differs (fast {actual}, naive {expected})"
        );
    }

    #[test]
    fn test_fast_split_search_matches_naive_reference() {
        let mut rng = TestRng::new(0x5EED_1234_ABCD_0001);

        for case in 0..60usize {
            let n = 2 + rng.next_usize(199); // 2..=200
            let n_features = 1 + rng.next_usize(8); // 1..=8
            let min_samples_leaf = 1 + rng.next_usize(3);
            let (x, y) = random_dataset(&mut rng, n, n_features);
            let indices: Vec<usize> = (0..n).collect();

            let targets: Vec<Float> = indices.iter().map(|&i| y[i]).collect();
            let parent_var = variance(&targets);
            let node_mean = mean_value(&targets);

            let naive = naive_best_regression_split(&x, &y, &indices, parent_var, min_samples_leaf);
            let mut buffers = SplitSearchBuffers::with_capacity(n);
            let fast = find_best_regression_split(
                &x,
                &y,
                &indices,
                parent_var,
                node_mean,
                min_samples_leaf,
                &mut buffers,
            );

            assert_eq!(
                fast.feature_index, naive.feature_index,
                "case {case} (n={n}, F={n_features}, msl={min_samples_leaf}): \
                 feature differs (fast {}, naive {})",
                fast.feature_index, naive.feature_index
            );
            assert_close(fast.threshold, naive.threshold, "threshold", case);
            assert_close(fast.gain, naive.gain, "gain", case);
        }
    }

    /// A node whose every column is constant offers no candidate at all, so the
    /// search must report the `-inf` sentinel (and the tree must become a leaf).
    #[test]
    fn test_split_search_on_all_constant_features_finds_nothing() {
        let x: Array2<Float> = array![[1.0, 7.0], [1.0, 7.0], [1.0, 7.0], [1.0, 7.0]];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let indices: Vec<usize> = (0..4).collect();
        let parent_var = variance(&y);
        let node_mean = mean_value(&y);
        let mut buffers = SplitSearchBuffers::with_capacity(4);

        let best =
            find_best_regression_split(&x, &y, &indices, parent_var, node_mean, 1, &mut buffers);
        assert_eq!(best.gain, Float::NEG_INFINITY);

        let y_arr: Array1<Float> = Array1::from_vec(y);
        let tree = DecisionTreeRegressor::new()
            .max_depth(3)
            .fit(&x, &y_arr)
            .expect("fit should succeed");
        let preds = tree.predict(&x).expect("predict should succeed");
        for &p in preds.iter() {
            assert!((p - 2.5).abs() < 1e-12, "expected the node mean, got {p}");
        }
    }

    /// Perf smoke test for the `O(n log n)` split search. Ignored by default;
    /// run with `cargo test -p sklears-ensemble --release -- --ignored`.
    #[test]
    #[ignore = "perf smoke test: 50k x 20 tree build, run explicitly and in release"]
    fn test_regressor_build_perf_smoke_50k_rows() {
        let mut rng = TestRng::new(0xC0FF_EE00_1234_5679);
        let n = 50_000usize;
        let n_features = 20usize;

        let mut data = Vec::with_capacity(n * n_features);
        for _ in 0..n * n_features {
            data.push(rng.next_f64() * 2.0 - 1.0);
        }
        let x = Array2::from_shape_vec((n, n_features), data)
            .expect("generated shape matches data length");
        let targets: Vec<Float> = (0..n).map(|_| rng.next_f64()).collect();
        let y = Array1::from_vec(targets);

        let start = std::time::Instant::now();
        let tree = DecisionTreeRegressor::new()
            .max_depth(3)
            .fit(&x, &y)
            .expect("fit should succeed");
        let elapsed = start.elapsed();
        println!("depth-3 tree on {n}x{n_features} built in {elapsed:?}");

        let preds = tree.predict(&x).expect("predict should succeed");
        assert_eq!(preds.len(), n);
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "single depth-3 tree on {n}x{n_features} took {elapsed:?}, expected well under 5s"
        );
    }
}
