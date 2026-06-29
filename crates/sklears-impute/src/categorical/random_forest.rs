//! Categorical Random Forest Imputer
#![allow(non_snake_case)]
//!
//! Imputation using Random Forest specifically designed for categorical data.
//! Uses ensemble of categorical decision trees for robust imputation via
//! the MissForest iterative algorithm.
//!
//! # Algorithm
//!
//! MissForest iterative imputation:
//! 1. Initial fill with column medians for all NaN entries.
//! 2. For each column `j` with any missing values, train a CART regression
//!    forest on the rows where `j` was observed.
//! 3. Iterate up to `max_iter` times, updating the imputed copy with forest
//!    predictions and checking for convergence.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Random, Rng};
use scirs2_core::RngExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Categorical Random Forest Imputer
///
/// Imputation using Random Forest specifically designed for categorical data.
/// Uses ensemble of categorical decision trees for robust imputation.
///
/// # Parameters
///
/// * `n_estimators` - Number of trees in the forest
/// * `max_depth` - Maximum depth of the individual trees
/// * `min_samples_split` - Minimum number of samples required to split a node
/// * `min_samples_leaf` - Minimum number of samples required at a leaf node
/// * `max_features` - Number of features to consider when looking for the best split
/// * `bootstrap` - Whether to use bootstrap sampling
/// * `max_iter` - Maximum number of iterations for iterative imputation
/// * `tol` - Tolerance for convergence
/// * `missing_values` - The placeholder for missing values
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::CategoricalRandomForestImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 2.0, 1.0], [2.0, f64::NAN, 3.0]];
///
/// let imputer = CategoricalRandomForestImputer::new()
///     .n_estimators(10)
///     .max_depth(Some(5));
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CategoricalRandomForestImputer<S = Untrained> {
    state: S,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    bootstrap: bool,
    max_iter: usize,
    tol: f64,
    missing_values: f64,
    random_state: Option<u64>,
}

/// Trained state for CategoricalRandomForestImputer
#[derive(Debug, Clone)]
pub struct CategoricalRandomForestImputerTrained {
    trees_: Vec<CategoricalTree>,
    feature_importances_: Array1<f64>,
    n_features_in_: usize,
    n_estimators_: usize,
}

/// Simple categorical decision tree for random forest
#[derive(Debug, Clone)]
pub struct CategoricalTree {
    feature: Option<usize>,
    value: Option<f64>,
    prediction: Option<f64>,
    left: Option<Box<CategoricalTree>>,
    right: Option<Box<CategoricalTree>>,
}

// ── Module-level helpers ────────────────────────────────────────────────────

/// Traverse a `CategoricalTree` and return the leaf prediction for `row`.
pub fn tree_predict(tree: &CategoricalTree, row: &[f64]) -> f64 {
    match (tree.feature, tree.value) {
        (Some(feat), Some(threshold)) => {
            let x = if feat < row.len() { row[feat] } else { 0.0 };
            if x <= threshold {
                match &tree.left {
                    Some(left) => tree_predict(left, row),
                    None => tree.prediction.unwrap_or(0.0),
                }
            } else {
                match &tree.right {
                    Some(right) => tree_predict(right, row),
                    None => tree.prediction.unwrap_or(0.0),
                }
            }
        }
        _ => tree.prediction.unwrap_or(0.0),
    }
}

/// Compute the mean of a slice; returns 0.0 for an empty slice.
#[inline]
fn slice_mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Compute the weighted MSE of two sub-splits.
fn weighted_mse(left_targets: &[f64], right_targets: &[f64]) -> f64 {
    let mse_part = |ts: &[f64]| -> f64 {
        if ts.is_empty() {
            return 0.0;
        }
        let m = slice_mean(ts);
        ts.iter().map(|&v| (v - m).powi(2)).sum::<f64>()
    };
    mse_part(left_targets) + mse_part(right_targets)
}

/// Build a CART regression tree using MSE split criterion.
///
/// * `rows`      – feature vectors (one per training sample)
/// * `targets`   – regression target for each row
/// * `depth`     – current recursion depth (0 at root)
/// * `max_depth` – optional depth cap
/// * `min_split` – minimum node size to attempt splitting
/// * `min_leaf`  – minimum sample count required in each child
pub fn build_tree(
    rows: &[Vec<f64>],
    targets: &[f64],
    depth: usize,
    max_depth: Option<usize>,
    min_split: usize,
    min_leaf: usize,
) -> CategoricalTree {
    let n = rows.len();
    let mean_pred = slice_mean(targets);

    // ── Base cases ──────────────────────────────────────────────────────────
    // (a) Too few samples to split
    if n < min_split || n < 2 * min_leaf {
        return CategoricalTree {
            feature: None,
            value: None,
            prediction: Some(mean_pred),
            left: None,
            right: None,
        };
    }
    // (b) Depth cap reached
    if max_depth.is_some_and(|md| depth >= md) {
        return CategoricalTree {
            feature: None,
            value: None,
            prediction: Some(mean_pred),
            left: None,
            right: None,
        };
    }
    // (c) All targets are identical
    let all_equal = targets
        .iter()
        .all(|&t| (t - targets[0]).abs() < f64::EPSILON);
    if all_equal {
        return CategoricalTree {
            feature: None,
            value: None,
            prediction: Some(mean_pred),
            left: None,
            right: None,
        };
    }

    // ── Find best split ─────────────────────────────────────────────────────
    let n_features = if n > 0 { rows[0].len() } else { 0 };
    let mut best_mse = f64::INFINITY;
    let mut best_feature: Option<usize> = None;
    let mut best_threshold: Option<f64> = None;

    for feat in 0..n_features {
        // Collect and sort distinct feature values
        let mut vals: Vec<f64> = rows.iter().map(|r| r[feat]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in training rows"));
        vals.dedup();

        // Try midpoints between adjacent distinct values as thresholds
        for w in vals.windows(2) {
            let threshold = (w[0] + w[1]) / 2.0;

            let mut left_t: Vec<f64> = Vec::new();
            let mut right_t: Vec<f64> = Vec::new();

            for (row, &tgt) in rows.iter().zip(targets.iter()) {
                if row[feat] <= threshold {
                    left_t.push(tgt);
                } else {
                    right_t.push(tgt);
                }
            }

            if left_t.len() < min_leaf || right_t.len() < min_leaf {
                continue;
            }

            let mse = weighted_mse(&left_t, &right_t);
            if mse < best_mse {
                best_mse = mse;
                best_feature = Some(feat);
                best_threshold = Some(threshold);
            }
        }
    }

    // ── Build subtrees ──────────────────────────────────────────────────────
    match (best_feature, best_threshold) {
        (Some(feat), Some(threshold)) => {
            let mut left_rows: Vec<Vec<f64>> = Vec::new();
            let mut left_tgts: Vec<f64> = Vec::new();
            let mut right_rows: Vec<Vec<f64>> = Vec::new();
            let mut right_tgts: Vec<f64> = Vec::new();

            for (row, &tgt) in rows.iter().zip(targets.iter()) {
                if row[feat] <= threshold {
                    left_rows.push(row.clone());
                    left_tgts.push(tgt);
                } else {
                    right_rows.push(row.clone());
                    right_tgts.push(tgt);
                }
            }

            let left = build_tree(
                &left_rows,
                &left_tgts,
                depth + 1,
                max_depth,
                min_split,
                min_leaf,
            );
            let right = build_tree(
                &right_rows,
                &right_tgts,
                depth + 1,
                max_depth,
                min_split,
                min_leaf,
            );

            CategoricalTree {
                feature: Some(feat),
                value: Some(threshold),
                prediction: Some(mean_pred),
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
            }
        }
        _ => {
            // No improvement found → leaf
            CategoricalTree {
                feature: None,
                value: None,
                prediction: Some(mean_pred),
                left: None,
                right: None,
            }
        }
    }
}

// ── Builder / configuration ─────────────────────────────────────────────────

impl CategoricalRandomForestImputer<Untrained> {
    /// Create a new CategoricalRandomForestImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_estimators: 10,
            max_depth: Some(5),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            bootstrap: true,
            max_iter: 10,
            tol: 1e-6,
            missing_values: f64::NAN,
            random_state: None,
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the maximum depth
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
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

    /// Set the maximum features
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set bootstrap sampling
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    #[inline]
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    // ── Private fit helpers ─────────────────────────────────────────────────

    /// Compute median over non-missing values in a column; returns 0.0 if all missing.
    fn column_median(&self, x: &Array2<f64>, col: usize) -> f64 {
        let mut vals: Vec<f64> = x
            .column(col)
            .iter()
            .filter(|&&v| !self.is_missing(v))
            .cloned()
            .collect();
        if vals.is_empty() {
            return 0.0;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN"));
        let mid = vals.len() / 2;
        if vals.len().is_multiple_of(2) {
            (vals[mid - 1] + vals[mid]) / 2.0
        } else {
            vals[mid]
        }
    }

    /// Build a bootstrap sample of (`rows`, `targets`) and fit one tree.
    fn fit_one_tree(
        &self,
        rows: &[Vec<f64>],
        targets: &[f64],
        rng: &mut (impl Rng + ?Sized),
    ) -> CategoricalTree {
        let n = rows.len();
        if n == 0 {
            return CategoricalTree {
                feature: None,
                value: None,
                prediction: Some(0.0),
                left: None,
                right: None,
            };
        }

        let (boot_rows, boot_tgts): (Vec<Vec<f64>>, Vec<f64>) = if self.bootstrap {
            let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let br: Vec<Vec<f64>> = indices.iter().map(|&i| rows[i].clone()).collect();
            let bt: Vec<f64> = indices.iter().map(|&i| targets[i]).collect();
            (br, bt)
        } else {
            (rows.to_vec(), targets.to_vec())
        };

        // Feature sub-sampling: choose sqrt(n_features) features if max_features is set
        let n_features = if boot_rows.is_empty() {
            0
        } else {
            boot_rows[0].len()
        };
        let selected_features: Vec<usize> = {
            let k = self
                .max_features
                .unwrap_or_else(|| ((n_features as f64).sqrt().ceil() as usize).max(1));
            let k = k.min(n_features);
            let mut all_feats: Vec<usize> = (0..n_features).collect();
            all_feats.shuffle(rng);
            all_feats.truncate(k);
            all_feats.sort_unstable();
            all_feats
        };

        // Project rows onto selected features
        let projected: Vec<Vec<f64>> = boot_rows
            .iter()
            .map(|r| selected_features.iter().map(|&f| r[f]).collect())
            .collect();

        let subtree = build_tree(
            &projected,
            &boot_tgts,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        );

        // Wrap: if only a subset of features was used, we must re-map feature indices
        // back to the original space by building a thin wrapper tree that routes via
        // the selected_features mapping.
        remap_tree_features(subtree, &selected_features)
    }
}

/// Recursively remap leaf/split feature indices from projected space to original space.
fn remap_tree_features(tree: CategoricalTree, feat_map: &[usize]) -> CategoricalTree {
    CategoricalTree {
        feature: tree
            .feature
            .map(|f| if f < feat_map.len() { feat_map[f] } else { f }),
        value: tree.value,
        prediction: tree.prediction,
        left: tree
            .left
            .map(|c| Box::new(remap_tree_features(*c, feat_map))),
        right: tree
            .right
            .map(|c| Box::new(remap_tree_features(*c, feat_map))),
    }
}

impl Default for CategoricalRandomForestImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CategoricalRandomForestImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

// ── Fit ─────────────────────────────────────────────────────────────────────

impl Fit<ArrayView2<'_, Float>, ()> for CategoricalRandomForestImputer<Untrained> {
    type Fitted = CategoricalRandomForestImputer<CategoricalRandomForestImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X_orig: Array2<f64> = X.mapv(|x| x);
        let (n_samples, n_features) = X_orig.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        // ── Determine which columns have missing values ───────────────────
        let missing_cols: Vec<usize> = (0..n_features)
            .filter(|&j| (0..n_samples).any(|i| self.is_missing(X_orig[[i, j]])))
            .collect();

        // ── Step 1: initial fill with column medians ──────────────────────
        let medians: Vec<f64> = (0..n_features)
            .map(|j| self.column_median(&X_orig, j))
            .collect();

        let mut X_work: Array2<f64> = X_orig.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_work[[i, j]]) {
                    X_work[[i, j]] = medians[j];
                }
            }
        }

        // ── Set up RNG ────────────────────────────────────────────────────
        let mut rng: Box<dyn Rng> = match self.random_state {
            Some(seed) => Box::new(Random::seed(seed)),
            None => Box::new(Random::default()),
        };

        // ── Step 2: build per-column forests and iteratively impute ───────
        // For each column with missing values, we maintain a single
        // "representative" tree (the first bootstrap tree from the last
        // training round).  All `n_estimators` trees are used for prediction
        // (averaged) during the iterative phase.

        // Per-column: Vec of n_estimators trees (one forest per missing col)
        let mut col_forests: Vec<Option<Vec<CategoricalTree>>> = vec![None; n_features];

        for _iter in 0..self.max_iter {
            let prev_X_work = X_work.clone();
            let _ = &prev_X_work; // suppress unused warning in first iter

            for &j in &missing_cols {
                // Collect training rows: rows where X_orig[i, j] was NOT missing
                let mut train_rows: Vec<Vec<f64>> = Vec::new();
                let mut train_tgts: Vec<f64> = Vec::new();

                for i in 0..n_samples {
                    if !self.is_missing(X_orig[[i, j]]) {
                        // Features: all columns except j (from working copy)
                        let features: Vec<f64> = (0..n_features)
                            .filter(|&k| k != j)
                            .map(|k| X_work[[i, k]])
                            .collect();
                        train_rows.push(features);
                        train_tgts.push(X_orig[[i, j]]);
                    }
                }

                if train_rows.is_empty() {
                    continue;
                }

                // Build forest of n_estimators trees
                let mut trees: Vec<CategoricalTree> = Vec::with_capacity(self.n_estimators);
                for _ in 0..self.n_estimators {
                    trees.push(self.fit_one_tree(&train_rows, &train_tgts, &mut *rng));
                }
                col_forests[j] = Some(trees);

                // Predict missing entries in column j using forest average
                if let Some(ref forest) = col_forests[j] {
                    for i in 0..n_samples {
                        if self.is_missing(X_orig[[i, j]]) {
                            let row: Vec<f64> = (0..n_features)
                                .filter(|&k| k != j)
                                .map(|k| X_work[[i, k]])
                                .collect();
                            let pred: f64 =
                                forest.iter().map(|t| tree_predict(t, &row)).sum::<f64>()
                                    / forest.len() as f64;
                            X_work[[i, j]] = pred;
                        }
                    }
                }
            }

            // ── Convergence check ──────────────────────────────────────────
            let mut change: f64 = 0.0;
            for &j in &missing_cols {
                for i in 0..n_samples {
                    if self.is_missing(X_orig[[i, j]]) {
                        let delta = (X_work[[i, j]] - prev_X_work[[i, j]]).abs();
                        if delta > change {
                            change = delta;
                        }
                    }
                }
            }

            if change < self.tol {
                break;
            }
        }

        // ── Step 4: collect one representative tree per missing column ─────
        // `trees_` is indexed by column; non-missing columns get a trivial leaf.
        let mut trees_: Vec<CategoricalTree> = (0..n_features)
            .map(|_| CategoricalTree {
                feature: None,
                value: None,
                prediction: Some(0.0),
                left: None,
                right: None,
            })
            .collect();

        for &j in &missing_cols {
            if let Some(Some(ref forest)) = col_forests.get(j) {
                if !forest.is_empty() {
                    trees_[j] = forest[0].clone();
                }
            }
        }

        let feature_importances_ = Array1::zeros(n_features);

        Ok(CategoricalRandomForestImputer {
            state: CategoricalRandomForestImputerTrained {
                trees_,
                feature_importances_,
                n_features_in_: n_features,
                n_estimators_: self.n_estimators,
            },
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            bootstrap: self.bootstrap,
            max_iter: self.max_iter,
            tol: self.tol,
            missing_values: self.missing_values,
            random_state: self.random_state,
        })
    }
}

// ── Transform ───────────────────────────────────────────────────────────────

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for CategoricalRandomForestImputer<CategoricalRandomForestImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X_in: Array2<f64> = X.mapv(|x| x);
        let (n_samples, n_features) = X_in.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_out = X_in.clone();

        // For each row, impute each missing feature using the representative tree
        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_out[[i, j]]) {
                    // Build the feature vector (all columns except j)
                    let row: Vec<f64> = (0..n_features)
                        .filter(|&k| k != j)
                        .map(|k| {
                            let v = X_out[[i, k]];
                            if self.is_missing(v) {
                                0.0 // fallback: zero for chained missing
                            } else {
                                v
                            }
                        })
                        .collect();

                    X_out[[i, j]] = tree_predict(&self.state.trees_[j], &row);
                }
            }
        }

        Ok(X_out.mapv(|x| x as Float))
    }
}

impl CategoricalRandomForestImputer<CategoricalRandomForestImputerTrained> {
    #[inline]
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    /// Number of features seen during `fit`
    pub fn n_features_in(&self) -> usize {
        self.state.n_features_in_
    }

    /// Placeholder feature importances (all zeros in this version)
    pub fn feature_importances(&self) -> &Array1<f64> {
        &self.state.feature_importances_
    }

    /// Number of estimators used during fit
    pub fn n_estimators_fitted(&self) -> usize {
        self.state.n_estimators_
    }
}

impl Default for CategoricalTree {
    fn default() -> Self {
        Self {
            feature: None,
            value: None,
            prediction: Some(0.0),
            left: None,
            right: None,
        }
    }
}
