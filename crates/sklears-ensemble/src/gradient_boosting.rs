//! Gradient boosting
//!
//! Binary [`GradientBoostingClassifier`] (binomial log-loss) and
//! [`GradientBoostingRegressor`] (squared-error loss), following Friedman's
//! "Greedy Function Approximation" (2001) and "Stochastic Gradient Boosting"
//! (2002), built on the exact-split regression trees of [`crate::adaboost`].
//!
//! Supported: shrinkage (`learning_rate`), tree size control (`max_depth`,
//! `min_samples_split`, `min_samples_leaf`), row subsampling (`subsample`),
//! gain/frequency/cover feature importances, and staged scoring so a caller can
//! choose a stage count without refitting.
//!
//! This is deliberately *not* an XGBoost / LightGBM / CatBoost port: there is no
//! histogram binning, no second-order (Newton) leaf estimation, no L1/L2 leaf
//! regularization, and no multiclass support. Several
//! [`GradientBoostingConfig`] fields are still accepted but ignored — each one
//! is marked `NO-OP` on the field itself and summarized in the `# Scope`
//! sections of [`TrainedGradientBoostingClassifier`] and
//! [`TrainedGradientBoostingRegressor`].

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained},
    types::Float,
};

use crate::adaboost::{DecisionTreeRegressor, RegressorNode};

/// Loss functions for gradient boosting
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFunction {
    /// Least squares loss for regression
    SquaredLoss,
    /// Absolute deviation loss for regression (robust)
    AbsoluteLoss,
    /// Huber loss for regression (robust)
    HuberLoss,
    /// Quantile loss for regression
    QuantileLoss,
    /// Logistic loss for binary classification
    LogisticLoss,
    /// Deviance loss for multiclass classification
    DevianceLoss,
    /// Exponential loss for AdaBoost
    ExponentialLoss,
    /// Modified Huber loss
    ModifiedHuberLoss,
    /// Pseudo-Huber loss (robust)
    PseudoHuber,
    /// Fair loss function (robust)
    Fair,
    /// LogCosh loss (smooth approximation of Huber)
    LogCosh,
    /// Epsilon-insensitive loss
    EpsilonInsensitive,
    /// Tukey's biweight loss (robust)
    Tukey,
    /// Cauchy loss (robust)
    Cauchy,
    /// Welsch loss (robust)
    Welsch,
}

impl LossFunction {
    /// Compute loss value
    pub fn loss(&self, y_true: Float, y_pred: Float) -> Float {
        match self {
            LossFunction::SquaredLoss => 0.5 * (y_true - y_pred).powi(2),
            LossFunction::AbsoluteLoss => (y_true - y_pred).abs(),
            LossFunction::HuberLoss => {
                let delta = 1.0;
                let residual = y_true - y_pred;
                if residual.abs() <= delta {
                    0.5 * residual.powi(2)
                } else {
                    delta * (residual.abs() - 0.5 * delta)
                }
            }
            LossFunction::LogisticLoss => {
                let z = y_true * y_pred;
                if z > 0.0 {
                    (1.0 + (-z).exp()).ln()
                } else {
                    -z + (1.0 + z.exp()).ln()
                }
            }
            LossFunction::PseudoHuber => {
                let delta: Float = 1.0;
                let residual = y_true - y_pred;
                delta.powi(2) * ((1.0 + (residual / delta).powi(2)).sqrt() - 1.0)
            }
            LossFunction::Fair => {
                let c = 1.0;
                let residual = y_true - y_pred;
                c * (residual.abs() / c - (1.0 + residual.abs() / c).ln())
            }
            LossFunction::LogCosh => {
                let residual = y_true - y_pred;
                residual.cosh().ln()
            }
            _ => (y_true - y_pred).powi(2), // Default to squared loss
        }
    }

    /// Compute gradient (negative derivative of loss w.r.t. prediction)
    pub fn gradient(&self, y_true: Float, y_pred: Float) -> Float {
        match self {
            LossFunction::SquaredLoss => y_pred - y_true,
            LossFunction::AbsoluteLoss => {
                if y_pred > y_true {
                    1.0
                } else if y_pred < y_true {
                    -1.0
                } else {
                    0.0
                }
            }
            LossFunction::HuberLoss => {
                let delta = 1.0;
                let residual = y_pred - y_true;
                if residual.abs() <= delta {
                    residual
                } else {
                    delta * residual.signum()
                }
            }
            LossFunction::LogisticLoss => {
                let z = y_true * y_pred;
                -y_true / (1.0 + z.exp())
            }
            LossFunction::PseudoHuber => {
                let delta = 1.0;
                let residual = y_pred - y_true;
                residual / (1.0 + (residual / delta).powi(2)).sqrt()
            }
            LossFunction::Fair => {
                let c = 1.0;
                let residual = y_pred - y_true;
                residual / (1.0 + residual.abs() / c)
            }
            LossFunction::LogCosh => {
                let residual = y_pred - y_true;
                residual.tanh()
            }
            _ => y_pred - y_true, // Default to squared loss gradient
        }
    }

    /// Compute Hessian (second derivative of loss w.r.t. prediction)
    pub fn hessian(&self, y_true: Float, y_pred: Float) -> Float {
        match self {
            LossFunction::SquaredLoss => 1.0,
            LossFunction::AbsoluteLoss => 0.0, // Not differentiable at residual = 0
            LossFunction::HuberLoss => {
                let delta = 1.0;
                let residual = y_pred - y_true;
                if residual.abs() <= delta {
                    1.0
                } else {
                    0.0
                }
            }
            LossFunction::LogisticLoss => {
                let z = y_true * y_pred;
                let exp_z = z.exp();
                y_true.powi(2) * exp_z / (1.0 + exp_z).powi(2)
            }
            LossFunction::PseudoHuber => {
                let delta = 1.0;
                let residual = y_pred - y_true;
                1.0 / (1.0 + (residual / delta).powi(2)).powf(1.5)
            }
            LossFunction::Fair => {
                let c = 1.0;
                let residual = y_pred - y_true;
                c / (c + residual.abs()).powi(2)
            }
            LossFunction::LogCosh => {
                let residual = y_pred - y_true;
                1.0 - residual.tanh().powi(2)
            }
            _ => 1.0, // Default to squared loss hessian
        }
    }

    /// Check if the loss function is robust to outliers
    pub fn is_robust(&self) -> bool {
        matches!(
            self,
            LossFunction::AbsoluteLoss
                | LossFunction::HuberLoss
                | LossFunction::PseudoHuber
                | LossFunction::Fair
                | LossFunction::Tukey
                | LossFunction::Cauchy
                | LossFunction::Welsch
        )
    }
}

/// Types of gradient boosting trees
#[derive(Debug, Clone)]
pub enum GradientBoostingTree {
    /// Decision tree weak learner
    DecisionTree,
    /// Histogram-based tree for efficiency
    HistogramTree,
    /// Neural network weak learner
    NeuralNetwork,
}

/// Gradient boosting configuration
///
/// Fields marked `NO-OP` are accepted for forward compatibility but have no
/// effect on fitting today; every other field is honored.
#[derive(Debug, Clone)]
pub struct GradientBoostingConfig {
    /// Number of boosting rounds (one regression tree each). Honored.
    pub n_estimators: usize,
    /// Shrinkage applied to every tree's contribution. Honored.
    pub learning_rate: Float,
    /// Maximum depth of each regression tree. Honored.
    pub max_depth: usize,
    /// Minimum node size required before a split is considered. Honored.
    pub min_samples_split: usize,
    /// Minimum number of samples required on each side of a split. Honored.
    pub min_samples_leaf: usize,
    /// Fraction of the training rows drawn *without replacement* for each
    /// boosting round (stochastic gradient boosting). Must lie in
    /// `(0.0, 1.0]`; `1.0` (the default) disables sampling entirely and leaves
    /// fitting bit-for-bit identical to the non-stochastic path. Honored.
    pub subsample: Float,
    /// NO-OP. The classifier always optimizes binomial log-loss and the
    /// regressor always optimizes squared error; this field is ignored.
    pub loss_function: LossFunction,
    /// Seed for the row-subsampling RNG. Honored whenever `subsample < 1.0`
    /// (fitting is deterministic regardless when `subsample == 1.0`). When
    /// `None`, a fixed default seed is used, so runs stay reproducible.
    pub random_state: Option<u64>,
    /// NO-OP. Only the exact-split decision-tree learner is implemented;
    /// this field is ignored.
    pub tree_type: GradientBoostingTree,
    /// NO-OP. No internal early stopping is performed. Use
    /// [`TrainedGradientBoostingClassifier::staged_decision_function`] (or the
    /// regressor's [`TrainedGradientBoostingRegressor::staged_predict`]) to
    /// choose a stage count externally without refitting.
    pub early_stopping: Option<usize>,
    /// NO-OP. No internal validation split is carved out; this field is
    /// ignored.
    pub validation_fraction: Float,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            loss_function: LossFunction::SquaredLoss,
            random_state: None,
            tree_type: GradientBoostingTree::DecisionTree,
            early_stopping: None,
            validation_fraction: 0.1,
        }
    }
}

/// Feature importance metrics
#[derive(Debug, Clone)]
pub struct FeatureImportanceMetrics {
    pub gain: Array1<Float>,
    pub frequency: Array1<Float>,
    pub cover: Array1<Float>,
}

impl FeatureImportanceMetrics {
    pub fn new(n_features: usize) -> Self {
        Self {
            gain: Array1::zeros(n_features),
            frequency: Array1::zeros(n_features),
            cover: Array1::zeros(n_features),
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient boosting internals (Friedman, "Greedy Function Approximation", 2001)
// ---------------------------------------------------------------------------

/// Numerically stable logistic sigmoid, shared by the classifier stages.
///
/// Safe over the whole real line in `f64`: for very negative `z`, `(-z).exp()`
/// saturates to `+inf` and the result is `0.0`; for very positive `z` it
/// underflows to `0.0` and the result is `1.0` — never `NaN`.
fn sigmoid(z: Float) -> Float {
    1.0 / (1.0 + (-z).exp())
}

/// Mean of a float slice (`0.0` for an empty slice).
///
/// Local reimplementation: `adaboost::decision_tree`'s own `mean_value` is
/// module-private and cannot be reused from here.
fn mean_value(values: &[Float]) -> Float {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<Float>() / values.len() as Float
}

/// Population variance of a float slice (`0.0` for an empty slice).
///
/// Matches the variance definition used inside `adaboost::decision_tree`'s
/// `build_regressor`, so the split gains recomputed by the importance walker
/// line up with the tree's own greedy criterion.
fn variance(values: &[Float]) -> Float {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mean = mean_value(values);
    values.iter().map(|&v| (v - mean).powi(2)).sum::<Float>() / n as Float
}

/// Seed used for the row-subsampling RNG when `random_state` is `None`, so that
/// even an unseeded stochastic fit is reproducible.
const DEFAULT_SUBSAMPLE_SEED: u64 = 0;

/// Reject a `subsample` outside `(0.0, 1.0]` before any work is done.
fn validate_subsample(subsample: Float) -> Result<()> {
    if subsample > 0.0 && subsample <= 1.0 {
        Ok(())
    } else {
        Err(SklearsError::InvalidParameter {
            name: "subsample".to_string(),
            reason: format!("subsample must lie in (0.0, 1.0], got {subsample}"),
        })
    }
}

/// Per-round row sampler for stochastic gradient boosting (Friedman 2002).
///
/// `state` stays `None` when `subsample == 1.0` (or a fraction that rounds up
/// to every row): the RNG is then never constructed and every round sees the
/// full training set, so the default configuration is bit-for-bit identical to
/// the previous non-stochastic implementation.
struct RowSampler {
    state: Option<SubsampleState>,
}

/// Sampling state used only when `subsample < 1.0`.
struct SubsampleState {
    rng: CoreRandom<StdRng>,
    /// Running permutation of `0..n_samples`. Each draw is a partial
    /// Fisher-Yates over it, which leaves it a permutation, so successive
    /// rounds are independent draws from the same seeded chain.
    permutation: Vec<usize>,
    /// Rows drawn per boosting round.
    n_rows: usize,
}

impl RowSampler {
    fn new(subsample: Float, n_samples: usize, random_state: Option<u64>) -> Self {
        if subsample >= 1.0 || n_samples == 0 {
            return Self { state: None };
        }
        let n_rows = ((subsample * n_samples as Float) as usize).clamp(1, n_samples);
        if n_rows == n_samples {
            return Self { state: None };
        }
        Self {
            state: Some(SubsampleState {
                rng: seeded_rng(random_state.unwrap_or(DEFAULT_SUBSAMPLE_SEED)),
                permutation: (0..n_samples).collect(),
                n_rows,
            }),
        }
    }

    /// Rows to fit the next round on, in ascending order. `None` means "every
    /// row", letting the caller skip building a sub-matrix altogether.
    fn next_rows(&mut self) -> Option<&[usize]> {
        let state = self.state.as_mut()?;
        let n_samples = state.permutation.len();
        let n_rows = state.n_rows;
        for i in 0..n_rows {
            let offset: usize = state.rng.random_range(0..(n_samples - i));
            state.permutation.swap(i, i + offset);
        }
        state.permutation[..n_rows].sort_unstable();
        Some(&state.permutation[..n_rows])
    }
}

/// Materialize the design matrix and target vector restricted to `rows`.
#[allow(non_snake_case)] // standard ML notation
fn gather_rows(
    X: &Array2<Float>,
    targets: &Array1<Float>,
    rows: &[usize],
) -> (Array2<Float>, Array1<Float>) {
    let x_sub = Array2::from_shape_fn((rows.len(), X.ncols()), |(r, c)| X[[rows[r], c]]);
    let targets_sub = Array1::from_shape_fn(rows.len(), |r| targets[rows[r]]);
    (x_sub, targets_sub)
}

/// Discover the class labels by sorting + deduplicating the raw values of `y`,
/// requiring exactly two (binary classification only).
///
/// Exact float equality is safe here because the returned values are the raw
/// labels themselves (never arithmetic results).
fn discover_binary_classes(y: &Array1<Float>) -> Result<Array1<Float>> {
    let mut classes: Vec<Float> = y.iter().copied().collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();
    if classes.len() != 2 {
        return Err(SklearsError::InvalidInput(format!(
            "GradientBoostingClassifier currently supports only binary classification, \
             but the targets contain {} distinct classes",
            classes.len()
        )));
    }
    Ok(Array1::from_vec(classes))
}

/// Running feature-importance accumulators, bundled together to keep the
/// recursive walker's signature small (mirrors `RegressorBuildParams`).
struct ImportanceAccumulator {
    gain: Array1<Float>,
    frequency: Array1<Float>,
    cover: Array1<Float>,
}

impl ImportanceAccumulator {
    fn new(n_features: usize) -> Self {
        Self {
            gain: Array1::zeros(n_features),
            frequency: Array1::zeros(n_features),
            cover: Array1::zeros(n_features),
        }
    }

    /// Normalize each importance vector to sum to 1 (a no-op when the running
    /// total is `0`, e.g. when every tree collapsed to a single leaf).
    fn into_normalized(mut self) -> FeatureImportanceMetrics {
        let total_gain = self.gain.sum();
        if total_gain > 0.0 {
            self.gain /= total_gain;
        }
        let total_frequency = self.frequency.sum();
        if total_frequency > 0.0 {
            self.frequency /= total_frequency;
        }
        let total_cover = self.cover.sum();
        if total_cover > 0.0 {
            self.cover /= total_cover;
        }
        FeatureImportanceMetrics {
            gain: self.gain,
            frequency: self.frequency,
            cover: self.cover,
        }
    }
}

/// Recursively walk one fitted regression tree, re-partitioning the training
/// rows that reach each split node exactly as `build_regressor` did, and
/// accumulating gain / frequency / cover from the real pseudo-residuals used to
/// fit that tree.
///
/// `indices` are the training rows reaching `node`; the root call passes all of
/// `0..n_samples`. `residuals` is the target vector the tree was fitted on.
fn accumulate_importance(
    node: &RegressorNode,
    x: &Array2<Float>,
    residuals: &Array1<Float>,
    indices: &[usize],
    acc: &mut ImportanceAccumulator,
) {
    let (feature_index, threshold, left, right) = match node {
        RegressorNode::Leaf(_) => return,
        RegressorNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => (*feature_index, *threshold, left, right),
    };

    let n = indices.len();
    if n == 0 {
        return;
    }

    let node_targets: Vec<Float> = indices.iter().map(|&i| residuals[i]).collect();
    let parent_var = variance(&node_targets);

    // Same partition rule the tree used: x[[i, feature_index]] <= threshold -> left.
    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| x[[i, feature_index]] <= threshold);

    let left_targets: Vec<Float> = left_idx.iter().map(|&i| residuals[i]).collect();
    let right_targets: Vec<Float> = right_idx.iter().map(|&i| residuals[i]).collect();

    let n_f = n as Float;
    let weighted_children_var = (left_targets.len() as Float / n_f) * variance(&left_targets)
        + (right_targets.len() as Float / n_f) * variance(&right_targets);

    acc.frequency[feature_index] += 1.0;
    acc.cover[feature_index] += n_f;
    acc.gain[feature_index] += (parent_var - weighted_children_var).max(0.0);

    accumulate_importance(left, x, residuals, &left_idx, acc);
    accumulate_importance(right, x, residuals, &right_idx, acc);
}

/// Gradient Boosting Classifier
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier {
    config: GradientBoostingConfig,
}

impl GradientBoostingClassifier {
    pub fn new(config: GradientBoostingConfig) -> Self {
        Self { config }
    }

    pub fn builder() -> GradientBoostingClassifierBuilder {
        GradientBoostingClassifierBuilder::default()
    }
}

/// Trained binary Gradient Boosting Classifier (Friedman 2001, binomial log-loss).
///
/// `F_0 = ln(p / (1 - p))` (with `p` the positive-class fraction) and each round
/// fits a regression tree to the pseudo-residuals `y - sigmoid(F_{m-1})`,
/// updating `F_m = F_{m-1} + learning_rate * h_m`.
///
/// With `subsample < 1.0` each round fits its tree on a fresh random subset of
/// the rows (drawn without replacement from the `random_state` chain) while
/// `F_m` is still updated for *every* row — standard stochastic gradient
/// boosting.
///
/// # Scope
/// Only binary classification with binomial log-loss is implemented. The
/// `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`,
/// `min_samples_leaf`, `subsample`, and `random_state` knobs are honored. The
/// `loss_function`, `tree_type`, `early_stopping`, and `validation_fraction`
/// knobs are accepted but have no effect: the loss and the learner are fixed,
/// and no internal early stopping or validation split is performed — use
/// [`Self::staged_decision_function`] / [`Self::decision_function_at`] to pick a
/// stage count externally without refitting.
#[derive(Debug, Clone)]
pub struct TrainedGradientBoostingClassifier {
    config: GradientBoostingConfig,
    feature_importance: FeatureImportanceMetrics,
    n_features: usize,
    classes: Array1<Float>,
    trees: Vec<DecisionTreeRegressor<Trained>>,
    init_logit: Float,
}

impl Fit<Array2<Float>, Array1<Float>> for GradientBoostingClassifier {
    type Fitted = TrainedGradientBoostingClassifier;

    #[allow(non_snake_case)] // standard ML notation
    fn fit(self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples == 0 || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit GradientBoostingClassifier on an empty dataset".to_string(),
            ));
        }
        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.nrows()={n_samples}"),
                actual: format!("y.len()={}", y.len()),
            });
        }
        if self.config.n_estimators == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_estimators".to_string(),
                reason: "Number of estimators must be positive".to_string(),
            });
        }
        validate_subsample(self.config.subsample)?;

        // Discover the two classes directly from the labels.
        let classes = discover_binary_classes(y)?;
        let negative_class = classes[0];

        // Map labels to {0, 1}; exact equality is safe because `classes` holds
        // the raw label values themselves (never arithmetic results).
        let y_binary = y.mapv(|v| if v == negative_class { 0.0 } else { 1.0 });

        // F_0 = ln(p / (1 - p)); both classes are present so 0 < p < 1 strictly.
        let p = y_binary.sum() / n_samples as Float;
        let init_logit = (p / (1.0 - p)).ln();
        let mut current = Array1::from_elem(n_samples, init_logit);

        let all_indices: Vec<usize> = (0..n_samples).collect();
        let mut trees: Vec<DecisionTreeRegressor<Trained>> =
            Vec::with_capacity(self.config.n_estimators);
        let mut importance = ImportanceAccumulator::new(n_features);
        let mut sampler =
            RowSampler::new(self.config.subsample, n_samples, self.config.random_state);

        for m in 0..self.config.n_estimators {
            // Vanilla gradient-boosting residuals for log-loss: r_i = y_i - p_i,
            // where p_i = sigmoid(F_{m-1}(x_i)). Note this is a plain difference,
            // NOT the LogitBoost working response divided by p*(1-p).
            let probabilities = current.mapv(sigmoid);
            let residuals = &y_binary - &probabilities;

            let untrained = DecisionTreeRegressor::new()
                .max_depth(self.config.max_depth)
                .min_samples_split(self.config.min_samples_split)
                .min_samples_leaf(self.config.min_samples_leaf)
                .random_state(self.config.random_state.map(|s| s + m as u64));

            // Stochastic gradient boosting: the tree sees only the sampled
            // rows, but `current` is updated for all of them below.
            let sampled_rows = sampler.next_rows();
            let tree = match sampled_rows {
                None => untrained.fit(X, &residuals)?,
                Some(rows) => {
                    let (x_sub, residuals_sub) = gather_rows(X, &residuals, rows);
                    untrained.fit(&x_sub, &residuals_sub)?
                }
            };

            let update = tree.predict(X)?;
            // F_m = F_{m-1} + learning_rate * h_m, for every row.
            for i in 0..n_samples {
                current[i] += self.config.learning_rate * update[i];
            }

            if let Some(state) = tree.tree_.as_ref() {
                // Replay the gains on exactly the rows the tree was fitted on.
                let importance_rows = sampled_rows.unwrap_or(&all_indices);
                accumulate_importance(&state.root, X, &residuals, importance_rows, &mut importance);
            }

            trees.push(tree);
        }

        Ok(TrainedGradientBoostingClassifier {
            config: self.config,
            feature_importance: importance.into_normalized(),
            n_features,
            classes,
            trees,
            init_logit,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for TrainedGradientBoostingClassifier {
    #[allow(non_snake_case)] // standard ML notation
    fn predict(&self, X: &Array2<Float>) -> Result<Array1<Float>> {
        // `decision_function` performs the feature-count check.
        let scores = self.decision_function(X)?;
        let negative = self.classes[0];
        let positive = self.classes[1];
        Ok(scores.mapv(|z| if sigmoid(z) > 0.5 { positive } else { negative }))
    }
}

impl TrainedGradientBoostingClassifier {
    /// Reject a design matrix whose column count differs from training.
    #[allow(non_snake_case)] // standard ML notation
    fn check_features(&self, X: &Array2<Float>) -> Result<()> {
        if X.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: X.ncols(),
            });
        }
        Ok(())
    }

    /// Number of boosting stages actually fitted (always `n_estimators`).
    pub fn n_stages(&self) -> usize {
        self.trees.len()
    }

    /// Raw cumulative log-odds `F(x) = init_logit + learning_rate * sum_m h_m(x)`.
    #[allow(non_snake_case)] // standard ML notation
    pub fn decision_function(&self, X: &Array2<Float>) -> Result<Array1<Float>> {
        self.decision_function_at(X, self.trees.len())
    }

    /// [`Self::decision_function`] truncated to the first `n_stages` boosting
    /// rounds, i.e. the score the model would produce if it had been fitted
    /// with `n_estimators = n_stages`.
    ///
    /// `n_stages == 0` returns the constant initial score; `n_stages` above
    /// [`Self::n_stages`] is an
    /// [`SklearsError::InvalidParameter`](sklears_core::error::SklearsError).
    /// Combined with a held-out set this gives early stopping without refitting.
    #[allow(non_snake_case)] // standard ML notation
    pub fn decision_function_at(
        &self,
        X: &Array2<Float>,
        n_stages: usize,
    ) -> Result<Array1<Float>> {
        self.check_features(X)?;
        if n_stages > self.trees.len() {
            return Err(SklearsError::InvalidParameter {
                name: "n_stages".to_string(),
                reason: format!(
                    "n_stages must not exceed the {} fitted stages, got {n_stages}",
                    self.trees.len()
                ),
            });
        }

        let n_rows = X.nrows();
        let mut scores = Array1::from_elem(n_rows, self.init_logit);
        for tree in self.trees.iter().take(n_stages) {
            let update = tree.predict(X)?;
            for i in 0..n_rows {
                scores[i] += self.config.learning_rate * update[i];
            }
        }
        Ok(scores)
    }

    /// The raw score after *each* boosting stage, in order: entry `m` is
    /// `decision_function_at(X, m + 1)`, and the last entry equals
    /// [`Self::decision_function`].
    ///
    /// Computed in a single pass with one running score vector (cloned once per
    /// stage), so scoring all stages costs the same tree traversals as scoring
    /// the full model once.
    #[allow(non_snake_case)] // standard ML notation
    pub fn staged_decision_function(&self, X: &Array2<Float>) -> Result<Vec<Array1<Float>>> {
        self.check_features(X)?;

        let n_rows = X.nrows();
        let mut running = Array1::from_elem(n_rows, self.init_logit);
        let mut stages = Vec::with_capacity(self.trees.len());
        for tree in &self.trees {
            let update = tree.predict(X)?;
            for i in 0..n_rows {
                running[i] += self.config.learning_rate * update[i];
            }
            stages.push(running.clone());
        }
        Ok(stages)
    }

    /// Probability of the positive class (`classes()[1]`) for each row,
    /// `sigmoid(decision_function(X))`.
    #[allow(non_snake_case)] // standard ML notation
    pub fn predict_proba_positive(&self, X: &Array2<Float>) -> Result<Array1<Float>> {
        Ok(self.decision_function(X)?.mapv(sigmoid))
    }

    /// [`Self::staged_decision_function`] pushed through the sigmoid — the
    /// positive-class probability after each boosting stage.
    #[allow(non_snake_case)] // standard ML notation
    pub fn staged_predict_proba_positive(&self, X: &Array2<Float>) -> Result<Vec<Array1<Float>>> {
        Ok(self
            .staged_decision_function(X)?
            .into_iter()
            .map(|scores| scores.mapv(sigmoid))
            .collect())
    }

    /// The two class labels discovered from the training targets (ascending).
    pub fn classes(&self) -> &Array1<Float> {
        &self.classes
    }

    pub fn feature_importances_gain(&self) -> &Array1<Float> {
        &self.feature_importance.gain
    }

    pub fn feature_importances_frequency(&self) -> &Array1<Float> {
        &self.feature_importance.frequency
    }

    pub fn feature_importances_cover(&self) -> &Array1<Float> {
        &self.feature_importance.cover
    }
}

/// Gradient Boosting Regressor
#[derive(Debug, Clone)]
pub struct GradientBoostingRegressor {
    config: GradientBoostingConfig,
}

impl GradientBoostingRegressor {
    pub fn new(config: GradientBoostingConfig) -> Self {
        Self { config }
    }

    pub fn builder() -> GradientBoostingRegressorBuilder {
        GradientBoostingRegressorBuilder::default()
    }
}

/// Trained Gradient Boosting Regressor (Friedman 2001, squared-error loss).
///
/// `F_0 = mean(y)` and each round fits a regression tree to the pseudo-residuals
/// `y - F_{m-1}`, updating `F_m = F_{m-1} + learning_rate * h_m`.
///
/// With `subsample < 1.0` each round fits its tree on a fresh random subset of
/// the rows (drawn without replacement from the `random_state` chain) while
/// `F_m` is still updated for *every* row — standard stochastic gradient
/// boosting.
///
/// # Scope
/// Only squared-error regression is implemented. The `n_estimators`,
/// `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
/// `subsample`, and `random_state` knobs are honored. The `loss_function`,
/// `tree_type`, `early_stopping`, and `validation_fraction` knobs are accepted
/// but have no effect: the loss and the learner are fixed, and no internal early
/// stopping or validation split is performed — use [`Self::staged_predict`] /
/// [`Self::predict_at`] to pick a stage count externally without refitting.
#[derive(Debug, Clone)]
pub struct TrainedGradientBoostingRegressor {
    config: GradientBoostingConfig,
    feature_importance: FeatureImportanceMetrics,
    n_features: usize,
    trees: Vec<DecisionTreeRegressor<Trained>>,
    init_prediction: Float,
}

impl Fit<Array2<Float>, Array1<Float>> for GradientBoostingRegressor {
    type Fitted = TrainedGradientBoostingRegressor;

    #[allow(non_snake_case)] // standard ML notation
    fn fit(self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples == 0 || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit GradientBoostingRegressor on an empty dataset".to_string(),
            ));
        }
        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.nrows()={n_samples}"),
                actual: format!("y.len()={}", y.len()),
            });
        }
        if self.config.n_estimators == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_estimators".to_string(),
                reason: "Number of estimators must be positive".to_string(),
            });
        }
        validate_subsample(self.config.subsample)?;

        // F_0(x) = mean(y).
        let init_prediction = y.sum() / n_samples as Float;
        let mut current = Array1::from_elem(n_samples, init_prediction);

        let all_indices: Vec<usize> = (0..n_samples).collect();
        let mut trees: Vec<DecisionTreeRegressor<Trained>> =
            Vec::with_capacity(self.config.n_estimators);
        let mut importance = ImportanceAccumulator::new(n_features);
        let mut sampler =
            RowSampler::new(self.config.subsample, n_samples, self.config.random_state);

        for m in 0..self.config.n_estimators {
            // Pseudo-residuals for squared-error loss: r_i = y_i - F_{m-1}(x_i).
            let residuals = y - &current;

            let untrained = DecisionTreeRegressor::new()
                .max_depth(self.config.max_depth)
                .min_samples_split(self.config.min_samples_split)
                .min_samples_leaf(self.config.min_samples_leaf)
                .random_state(self.config.random_state.map(|s| s + m as u64));

            // Stochastic gradient boosting: the tree sees only the sampled
            // rows, but `current` is updated for all of them below.
            let sampled_rows = sampler.next_rows();
            let tree = match sampled_rows {
                None => untrained.fit(X, &residuals)?,
                Some(rows) => {
                    let (x_sub, residuals_sub) = gather_rows(X, &residuals, rows);
                    untrained.fit(&x_sub, &residuals_sub)?
                }
            };

            let update = tree.predict(X)?;
            // F_m = F_{m-1} + learning_rate * h_m, for every row.
            for i in 0..n_samples {
                current[i] += self.config.learning_rate * update[i];
            }

            if let Some(state) = tree.tree_.as_ref() {
                // Replay the gains on exactly the rows the tree was fitted on.
                let importance_rows = sampled_rows.unwrap_or(&all_indices);
                accumulate_importance(&state.root, X, &residuals, importance_rows, &mut importance);
            }

            trees.push(tree);
        }

        Ok(TrainedGradientBoostingRegressor {
            config: self.config,
            feature_importance: importance.into_normalized(),
            n_features,
            trees,
            init_prediction,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for TrainedGradientBoostingRegressor {
    #[allow(non_snake_case)] // standard ML notation
    fn predict(&self, X: &Array2<Float>) -> Result<Array1<Float>> {
        // Replay the additive model: F(x) = init_prediction + lr * sum_m h_m(x).
        self.predict_at(X, self.trees.len())
    }
}

impl TrainedGradientBoostingRegressor {
    /// Reject a design matrix whose column count differs from training.
    #[allow(non_snake_case)] // standard ML notation
    fn check_features(&self, X: &Array2<Float>) -> Result<()> {
        if X.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: X.ncols(),
            });
        }
        Ok(())
    }

    /// Number of boosting stages actually fitted (always `n_estimators`).
    pub fn n_stages(&self) -> usize {
        self.trees.len()
    }

    /// [`Predict::predict`] truncated to the first `n_stages` boosting rounds,
    /// i.e. the prediction the model would produce if it had been fitted with
    /// `n_estimators = n_stages`.
    ///
    /// `n_stages == 0` returns the constant initial prediction; `n_stages`
    /// above [`Self::n_stages`] is an
    /// [`SklearsError::InvalidParameter`](sklears_core::error::SklearsError).
    #[allow(non_snake_case)] // standard ML notation
    pub fn predict_at(&self, X: &Array2<Float>, n_stages: usize) -> Result<Array1<Float>> {
        self.check_features(X)?;
        if n_stages > self.trees.len() {
            return Err(SklearsError::InvalidParameter {
                name: "n_stages".to_string(),
                reason: format!(
                    "n_stages must not exceed the {} fitted stages, got {n_stages}",
                    self.trees.len()
                ),
            });
        }

        let n_rows = X.nrows();
        let mut predictions = Array1::from_elem(n_rows, self.init_prediction);
        for tree in self.trees.iter().take(n_stages) {
            let update = tree.predict(X)?;
            for i in 0..n_rows {
                predictions[i] += self.config.learning_rate * update[i];
            }
        }
        Ok(predictions)
    }

    /// The prediction after *each* boosting stage, in order: entry `m` is
    /// `predict_at(X, m + 1)`, and the last entry equals
    /// [`Predict::predict`].
    ///
    /// Computed in a single pass with one running prediction vector (cloned
    /// once per stage).
    #[allow(non_snake_case)] // standard ML notation
    pub fn staged_predict(&self, X: &Array2<Float>) -> Result<Vec<Array1<Float>>> {
        self.check_features(X)?;

        let n_rows = X.nrows();
        let mut running = Array1::from_elem(n_rows, self.init_prediction);
        let mut stages = Vec::with_capacity(self.trees.len());
        for tree in &self.trees {
            let update = tree.predict(X)?;
            for i in 0..n_rows {
                running[i] += self.config.learning_rate * update[i];
            }
            stages.push(running.clone());
        }
        Ok(stages)
    }

    pub fn feature_importances_gain(&self) -> &Array1<Float> {
        &self.feature_importance.gain
    }

    pub fn feature_importances_frequency(&self) -> &Array1<Float> {
        &self.feature_importance.frequency
    }

    pub fn feature_importances_cover(&self) -> &Array1<Float> {
        &self.feature_importance.cover
    }
}

/// Builder for GradientBoostingClassifier
#[derive(Debug, Default)]
pub struct GradientBoostingClassifierBuilder {
    config: GradientBoostingConfig,
}

impl GradientBoostingClassifierBuilder {
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    pub fn loss_function(mut self, loss_function: LossFunction) -> Self {
        self.config.loss_function = loss_function;
        self
    }

    pub fn tree_type(mut self, tree_type: GradientBoostingTree) -> Self {
        self.config.tree_type = tree_type;
        self
    }

    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Fraction of the training rows each boosting round is fitted on, drawn
    /// without replacement (stochastic gradient boosting).
    ///
    /// Must lie in `(0.0, 1.0]`; values outside that range are rejected at
    /// [`Fit::fit`] time. `1.0` (the default) disables sampling. Pair with
    /// [`Self::random_state`] for reproducible stochastic fits.
    pub fn subsample(mut self, subsample: Float) -> Self {
        self.config.subsample = subsample;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    pub fn build(self) -> GradientBoostingClassifier {
        GradientBoostingClassifier::new(self.config)
    }
}

/// Builder for GradientBoostingRegressor
#[derive(Debug, Default)]
pub struct GradientBoostingRegressorBuilder {
    config: GradientBoostingConfig,
}

impl GradientBoostingRegressorBuilder {
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    pub fn loss_function(mut self, loss_function: LossFunction) -> Self {
        self.config.loss_function = loss_function;
        self
    }

    pub fn tree_type(mut self, tree_type: GradientBoostingTree) -> Self {
        self.config.tree_type = tree_type;
        self
    }

    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Fraction of the training rows each boosting round is fitted on, drawn
    /// without replacement (stochastic gradient boosting).
    ///
    /// Must lie in `(0.0, 1.0]`; values outside that range are rejected at
    /// [`Fit::fit`] time. `1.0` (the default) disables sampling. Pair with
    /// [`Self::random_state`] for reproducible stochastic fits.
    pub fn subsample(mut self, subsample: Float) -> Self {
        self.config.subsample = subsample;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    pub fn build(self) -> GradientBoostingRegressor {
        GradientBoostingRegressor::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};
    use sklears_core::error::SklearsError;
    use sklears_core::traits::{Fit, Predict};

    fn mse(pred: &Array1<Float>, target: &Array1<Float>) -> Float {
        pred.iter()
            .zip(target.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<Float>()
            / pred.len() as Float
    }

    /// `y = 2*x1 - x2` over a 5x4 integer grid (20 rows, 2 features).
    fn linear_grid() -> (Array2<Float>, Array1<Float>) {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for x1 in 0..5 {
            for x2 in 0..4 {
                features.push(x1 as Float);
                features.push(x2 as Float);
                targets.push(2.0 * x1 as Float - x2 as Float);
            }
        }
        let x = Array2::from_shape_vec((20, 2), features).expect("grid shape matches data length");
        let y = Array1::from_vec(targets);
        (x, y)
    }

    /// Two well-separated 2D blobs: lower-left labelled `2.0`, upper-right `5.0`
    /// (deliberately not `0/1`, so class discovery is genuinely exercised).
    fn two_blobs() -> (Array2<Float>, Array1<Float>) {
        let lower = [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [0.0, 2.0],
        ];
        let upper = [
            [8.0, 9.0],
            [9.0, 8.0],
            [9.0, 10.0],
            [10.0, 9.0],
            [8.0, 8.0],
            [10.0, 10.0],
            [9.0, 9.0],
            [8.0, 10.0],
        ];
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for row in lower.iter() {
            features.push(row[0]);
            features.push(row[1]);
            targets.push(2.0);
        }
        for row in upper.iter() {
            features.push(row[0]);
            features.push(row[1]);
            targets.push(5.0);
        }
        let x = Array2::from_shape_vec((16, 2), features).expect("blob shape matches data length");
        let y = Array1::from_vec(targets);
        (x, y)
    }

    // -- Regressor -------------------------------------------------------------

    /// Fits `y = 2*x1 - x2` and asserts predictions vary, are not the old
    /// all-zero fabrication, and reach a low training MSE.
    #[test]
    fn test_regressor_learns_linear_function() {
        let (x, y) = linear_grid();
        let model = GradientBoostingRegressor::builder()
            .n_estimators(50)
            .learning_rate(0.2)
            .max_depth(3)
            .build()
            .fit(&x, &y)
            .expect("regressor fit should succeed");
        let pred = model.predict(&x).expect("regressor predict should succeed");

        let max = pred.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        let min = pred.iter().copied().fold(Float::INFINITY, Float::min);
        assert!(
            max - min > 5.0,
            "predictions look constant: spread {}",
            max - min
        );
        assert!(
            pred.iter().any(|&p| p.abs() > 1.0),
            "predictions are essentially all zero"
        );

        let train_mse = mse(&pred, &y);
        assert!(train_mse < 0.5, "training MSE too high: {train_mse}");
    }

    /// The test that would have caught the original fabrication: more boosting
    /// rounds must strictly reduce training MSE.
    #[test]
    fn test_regressor_mse_decreases_with_more_estimators() {
        let (x, y) = linear_grid();
        let few = GradientBoostingRegressor::builder()
            .n_estimators(1)
            .learning_rate(0.2)
            .max_depth(3)
            .build()
            .fit(&x, &y)
            .expect("fit with 1 estimator should succeed");
        let many = GradientBoostingRegressor::builder()
            .n_estimators(50)
            .learning_rate(0.2)
            .max_depth(3)
            .build()
            .fit(&x, &y)
            .expect("fit with 50 estimators should succeed");

        let mse_few = mse(&few.predict(&x).expect("predict"), &y);
        let mse_many = mse(&many.predict(&x).expect("predict"), &y);
        assert!(
            mse_many < mse_few,
            "MSE did not decrease: 1 est = {mse_few}, 50 est = {mse_many}"
        );
    }

    /// Real (non-zero) feature importances that normalize to sum 1.
    #[test]
    fn test_regressor_feature_importances_frequency_normalized() {
        let (x, y) = linear_grid();
        let model = GradientBoostingRegressor::builder()
            .n_estimators(50)
            .learning_rate(0.2)
            .max_depth(3)
            .build()
            .fit(&x, &y)
            .expect("fit should succeed");
        let freq = model.feature_importances_frequency();
        let total: Float = freq.sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "frequency should sum to 1, got {total}"
        );
        assert!(freq.iter().any(|&v| v > 0.0), "frequency is all zero");
    }

    /// Shape mismatch on fit and feature mismatch on predict both error.
    #[test]
    fn test_regressor_shape_and_feature_mismatch_errors() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape matches data length");
        let y_bad = Array1::from_vec(vec![1.0, 2.0]);
        let fit_err = GradientBoostingRegressor::builder()
            .n_estimators(5)
            .build()
            .fit(&x, &y_bad);
        assert!(matches!(fit_err, Err(SklearsError::ShapeMismatch { .. })));

        let y_ok = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let model = GradientBoostingRegressor::builder()
            .n_estimators(5)
            .build()
            .fit(&x, &y_ok)
            .expect("fit should succeed");
        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape matches data length");
        let pred_err = model.predict(&x_wrong);
        assert!(matches!(
            pred_err,
            Err(SklearsError::FeatureMismatch { .. })
        ));
    }

    // -- Classifier ------------------------------------------------------------

    /// Separable blobs: both class labels are predicted, `classes()` returns the
    /// real discovered labels, and training accuracy is high.
    #[test]
    fn test_classifier_separable_blobs_accuracy_and_classes() {
        let (x, y) = two_blobs();
        let model = GradientBoostingClassifier::builder()
            .n_estimators(50)
            .learning_rate(0.3)
            .max_depth(2)
            .build()
            .fit(&x, &y)
            .expect("classifier fit should succeed");

        assert_eq!(model.classes(), &Array1::from_vec(vec![2.0, 5.0]));

        let pred = model
            .predict(&x)
            .expect("classifier predict should succeed");
        assert!(
            pred.iter().any(|&p| p == 2.0),
            "no negative-class predictions"
        );
        assert!(
            pred.iter().any(|&p| p == 5.0),
            "no positive-class predictions"
        );

        let correct = pred.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        let accuracy = correct as Float / y.len() as Float;
        assert!(accuracy >= 0.9, "accuracy too low: {accuracy}");
    }

    /// Mean log-loss (via `predict_proba_positive`) strictly decreases with more
    /// rounds. Accuracy saturates on separable data, so log-loss is used.
    #[test]
    fn test_classifier_log_loss_decreases_with_more_estimators() {
        let (x, y) = two_blobs();
        let y_binary: Vec<Float> = y
            .iter()
            .map(|&t| if t == 5.0 { 1.0 } else { 0.0 })
            .collect();

        let log_loss = |proba: &Array1<Float>| -> Float {
            proba
                .iter()
                .zip(y_binary.iter())
                .map(|(&p, &yb)| {
                    let p = p.clamp(1e-15, 1.0 - 1e-15);
                    -(yb * p.ln() + (1.0 - yb) * (1.0 - p).ln())
                })
                .sum::<Float>()
                / proba.len() as Float
        };

        let few = GradientBoostingClassifier::builder()
            .n_estimators(1)
            .learning_rate(0.2)
            .max_depth(2)
            .build()
            .fit(&x, &y)
            .expect("fit with 1 estimator should succeed");
        let many = GradientBoostingClassifier::builder()
            .n_estimators(50)
            .learning_rate(0.2)
            .max_depth(2)
            .build()
            .fit(&x, &y)
            .expect("fit with 50 estimators should succeed");

        let loss_few = log_loss(&few.predict_proba_positive(&x).expect("proba"));
        let loss_many = log_loss(&many.predict_proba_positive(&x).expect("proba"));
        assert!(
            loss_many < loss_few,
            "log-loss did not decrease: 1 est = {loss_few}, 50 est = {loss_many}"
        );
    }

    /// Non-binary targets (3 distinct values) must error at fit time.
    #[test]
    fn test_classifier_non_binary_targets_error() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape matches data length");
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let result = GradientBoostingClassifier::builder()
            .n_estimators(5)
            .build()
            .fit(&x, &y);
        assert!(matches!(result, Err(SklearsError::InvalidInput(_))));
    }

    /// Feature mismatch on predict must error.
    #[test]
    fn test_classifier_feature_mismatch_on_predict_errors() {
        let (x, y) = two_blobs();
        let model = GradientBoostingClassifier::builder()
            .n_estimators(10)
            .build()
            .fit(&x, &y)
            .expect("fit should succeed");
        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape matches data length");
        let result = model.predict(&x_wrong);
        assert!(matches!(result, Err(SklearsError::FeatureMismatch { .. })));
    }

    // -- Subsampling and staged scoring ---------------------------------------

    /// A deterministic, mildly overlapping binary problem with 120 rows and 3
    /// features — large enough that row subsampling actually changes the fit.
    fn noisy_binary_problem() -> (Array2<Float>, Array1<Float>) {
        let n = 120usize;
        let n_features = 3usize;
        let mut state: u64 = 0x2024_ABCD_1234_5678;
        let mut next_f64 = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as Float) / ((1u64 << 53) as Float)
        };

        let mut data = Vec::with_capacity(n * n_features);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let f0 = next_f64() * 2.0 - 1.0;
            let f1 = next_f64() * 2.0 - 1.0;
            let f2 = next_f64() * 2.0 - 1.0;
            data.push(f0);
            data.push(f1);
            data.push(f2);
            let score = 1.5 * f0 + f1 - 0.5 * f2;
            targets.push(if score > 0.0 { 1.0 } else { 0.0 });
        }

        let x = Array2::from_shape_vec((n, n_features), data)
            .expect("generated shape matches data length");
        (x, Array1::from_vec(targets))
    }

    /// Mean binomial log-loss of `proba` against `{0, 1}` targets.
    fn binary_log_loss(proba: &Array1<Float>, y_binary: &Array1<Float>) -> Float {
        proba
            .iter()
            .zip(y_binary.iter())
            .map(|(&p, &yb)| {
                let p = p.clamp(1e-15, 1.0 - 1e-15);
                -(yb * p.ln() + (1.0 - yb) * (1.0 - p).ln())
            })
            .sum::<Float>()
            / proba.len() as Float
    }

    /// A seeded `subsample = 0.5` fit is reproducible run-to-run and is a
    /// genuinely different model from the full-data fit.
    #[test]
    fn test_classifier_subsample_is_reproducible_and_differs_from_full() {
        let (x, y) = noisy_binary_problem();
        let fit = |subsample: Float| {
            GradientBoostingClassifier::builder()
                .n_estimators(30)
                .learning_rate(0.2)
                .max_depth(2)
                .subsample(subsample)
                .random_state(7)
                .build()
                .fit(&x, &y)
                .expect("fit should succeed")
        };

        let first = fit(0.5).predict_proba_positive(&x).expect("proba");
        let second = fit(0.5).predict_proba_positive(&x).expect("proba");
        let full = fit(1.0).predict_proba_positive(&x).expect("proba");

        for (i, (&p, &q)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(
                p, q,
                "subsample=0.5 with a fixed seed is not reproducible at row {i}"
            );
        }
        assert!(
            first
                .iter()
                .zip(full.iter())
                .any(|(&p, &q)| (p - q).abs() > 1e-9),
            "subsample=0.5 produced exactly the same model as subsample=1.0"
        );
    }

    /// Different seeds must draw different subsamples (so `random_state` really
    /// drives the sampler).
    #[test]
    fn test_classifier_subsample_depends_on_seed() {
        let (x, y) = noisy_binary_problem();
        let fit = |seed: u64| {
            GradientBoostingClassifier::builder()
                .n_estimators(30)
                .learning_rate(0.2)
                .max_depth(2)
                .subsample(0.5)
                .random_state(seed)
                .build()
                .fit(&x, &y)
                .expect("fit should succeed")
                .predict_proba_positive(&x)
                .expect("proba")
        };

        let a = fit(1);
        let b = fit(2);
        assert!(
            a.iter().zip(b.iter()).any(|(&p, &q)| (p - q).abs() > 1e-9),
            "two different seeds produced identical subsampled models"
        );
    }

    /// `subsample` outside `(0.0, 1.0]` is rejected at fit time.
    #[test]
    fn test_classifier_invalid_subsample_errors() {
        let (x, y) = two_blobs();
        for bad in [0.0, -0.5, 1.5] {
            let result = GradientBoostingClassifier::builder()
                .n_estimators(5)
                .subsample(bad)
                .build()
                .fit(&x, &y);
            assert!(
                matches!(result, Err(SklearsError::InvalidParameter { .. })),
                "subsample={bad} should have been rejected"
            );
        }
    }

    /// Staged scores have one entry per boosting round, the last entry is
    /// exactly `decision_function`, and every prefix matches
    /// `decision_function_at`.
    #[test]
    fn test_classifier_staged_decision_function_matches_full_and_truncated() {
        let (x, y) = two_blobs();
        let n_estimators = 12usize;
        let model = GradientBoostingClassifier::builder()
            .n_estimators(n_estimators)
            .learning_rate(0.2)
            .max_depth(2)
            .build()
            .fit(&x, &y)
            .expect("fit should succeed");

        assert_eq!(model.n_stages(), n_estimators);
        let staged = model.staged_decision_function(&x).expect("staged scores");
        assert_eq!(staged.len(), n_estimators);

        let full = model.decision_function(&x).expect("decision_function");
        for (i, (&s, &f)) in staged[n_estimators - 1].iter().zip(full.iter()).enumerate() {
            assert_eq!(
                s, f,
                "final staged score differs from decision_function at row {i}"
            );
        }

        for m in 0..=n_estimators {
            let truncated = model
                .decision_function_at(&x, m)
                .expect("decision_function_at");
            if m == 0 {
                let init = truncated[0];
                assert!(
                    truncated.iter().all(|&v| v == init),
                    "stage 0 must be the constant initial score"
                );
            } else {
                for (&s, &t) in staged[m - 1].iter().zip(truncated.iter()) {
                    assert_eq!(s, t, "stage {m} disagrees with decision_function_at");
                }
            }
        }

        let staged_proba = model
            .staged_predict_proba_positive(&x)
            .expect("staged proba");
        assert_eq!(staged_proba.len(), n_estimators);
        let full_proba = model.predict_proba_positive(&x).expect("proba");
        for (&s, &f) in staged_proba[n_estimators - 1].iter().zip(full_proba.iter()) {
            assert_eq!(s, f, "final staged probability differs from predict_proba");
        }

        assert!(matches!(
            model.decision_function_at(&x, n_estimators + 1),
            Err(SklearsError::InvalidParameter { .. })
        ));
    }

    /// Boosting still works under subsampling: more rounds at `subsample = 0.8`
    /// beat the 10-round baseline on log-loss.
    #[test]
    fn test_classifier_log_loss_decreases_with_subsampling() {
        let (x, y) = noisy_binary_problem();
        let fit = |n_estimators: usize| {
            GradientBoostingClassifier::builder()
                .n_estimators(n_estimators)
                .learning_rate(0.2)
                .max_depth(2)
                .subsample(0.8)
                .random_state(11)
                .build()
                .fit(&x, &y)
                .expect("fit should succeed")
                .predict_proba_positive(&x)
                .expect("proba")
        };

        let loss_baseline = binary_log_loss(&fit(10), &y);
        let loss_boosted = binary_log_loss(&fit(60), &y);
        assert!(
            loss_boosted < loss_baseline,
            "log-loss did not decrease under subsample=0.8: 10 est = {loss_baseline}, \
             60 est = {loss_boosted}"
        );
    }

    /// The regressor mirrors the classifier: staged predictions line up with the
    /// full model, and a seeded `subsample` is reproducible yet distinct.
    #[test]
    fn test_regressor_staged_predict_and_subsample() {
        let (x, y) = linear_grid();
        let n_estimators = 15usize;
        let model = GradientBoostingRegressor::builder()
            .n_estimators(n_estimators)
            .learning_rate(0.2)
            .max_depth(3)
            .build()
            .fit(&x, &y)
            .expect("fit should succeed");

        assert_eq!(model.n_stages(), n_estimators);
        let staged = model.staged_predict(&x).expect("staged predictions");
        assert_eq!(staged.len(), n_estimators);

        let full = model.predict(&x).expect("predict");
        for (&s, &f) in staged[n_estimators - 1].iter().zip(full.iter()) {
            assert_eq!(s, f, "final staged prediction differs from predict");
        }
        let zero = model.predict_at(&x, 0).expect("predict_at(0)");
        let init = zero[0];
        assert!(
            zero.iter().all(|&v| v == init),
            "stage 0 must be the constant initial prediction"
        );

        let fit_sub = |subsample: Float| {
            GradientBoostingRegressor::builder()
                .n_estimators(n_estimators)
                .learning_rate(0.2)
                .max_depth(3)
                .subsample(subsample)
                .random_state(5)
                .build()
                .fit(&x, &y)
                .expect("fit should succeed")
                .predict(&x)
                .expect("predict")
        };
        let sub_first = fit_sub(0.5);
        let sub_second = fit_sub(0.5);
        for (&p, &q) in sub_first.iter().zip(sub_second.iter()) {
            assert_eq!(p, q, "seeded regressor subsampling is not reproducible");
        }
        assert!(
            sub_first
                .iter()
                .zip(full.iter())
                .any(|(&p, &q)| (p - q).abs() > 1e-9),
            "subsample=0.5 produced exactly the same regressor as subsample=1.0"
        );
    }
}
