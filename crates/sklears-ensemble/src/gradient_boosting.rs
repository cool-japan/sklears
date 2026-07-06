//! Gradient Boosting implementation
//!
//! This module provides comprehensive gradient boosting algorithms including XGBoost, LightGBM,
//! and CatBoost-compatible implementations with histogram-based tree building, ensemble methods,
//! and advanced boosting strategies.

use scirs2_core::ndarray::{Array1, Array2};
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
#[derive(Debug, Clone)]
pub struct GradientBoostingConfig {
    pub n_estimators: usize,
    pub learning_rate: Float,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub subsample: Float,
    pub loss_function: LossFunction,
    pub random_state: Option<u64>,
    pub tree_type: GradientBoostingTree,
    pub early_stopping: Option<usize>,
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
/// # Scope
/// Only binary classification with binomial log-loss is implemented. The
/// `loss_function`, `tree_type`, `early_stopping`, `validation_fraction`, and
/// `subsample` configuration knobs are accepted but currently have no effect.
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

        for m in 0..self.config.n_estimators {
            // Vanilla gradient-boosting residuals for log-loss: r_i = y_i - p_i,
            // where p_i = sigmoid(F_{m-1}(x_i)). Note this is a plain difference,
            // NOT the LogitBoost working response divided by p*(1-p).
            let probabilities = current.mapv(sigmoid);
            let residuals = &y_binary - &probabilities;

            let tree = DecisionTreeRegressor::new()
                .max_depth(self.config.max_depth)
                .min_samples_split(self.config.min_samples_split)
                .min_samples_leaf(self.config.min_samples_leaf)
                .random_state(self.config.random_state.map(|s| s + m as u64))
                .fit(X, &residuals)?;

            let update = tree.predict(X)?;
            // F_m = F_{m-1} + learning_rate * h_m.
            for i in 0..n_samples {
                current[i] += self.config.learning_rate * update[i];
            }

            if let Some(state) = tree.tree_.as_ref() {
                accumulate_importance(&state.root, X, &residuals, &all_indices, &mut importance);
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
    /// Raw cumulative log-odds `F(x) = init_logit + learning_rate * sum_m h_m(x)`.
    #[allow(non_snake_case)] // standard ML notation
    pub fn decision_function(&self, X: &Array2<Float>) -> Result<Array1<Float>> {
        if X.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: X.ncols(),
            });
        }

        let n_rows = X.nrows();
        let mut scores = Array1::from_elem(n_rows, self.init_logit);
        for tree in &self.trees {
            let update = tree.predict(X)?;
            for i in 0..n_rows {
                scores[i] += self.config.learning_rate * update[i];
            }
        }
        Ok(scores)
    }

    /// Probability of the positive class (`classes()[1]`) for each row,
    /// `sigmoid(decision_function(X))`.
    #[allow(non_snake_case)] // standard ML notation
    pub fn predict_proba_positive(&self, X: &Array2<Float>) -> Result<Array1<Float>> {
        Ok(self.decision_function(X)?.mapv(sigmoid))
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
/// # Scope
/// Only squared-error regression is implemented. The `loss_function`,
/// `tree_type`, `early_stopping`, `validation_fraction`, and `subsample`
/// configuration knobs are accepted but currently have no effect.
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

        // F_0(x) = mean(y).
        let init_prediction = y.sum() / n_samples as Float;
        let mut current = Array1::from_elem(n_samples, init_prediction);

        let all_indices: Vec<usize> = (0..n_samples).collect();
        let mut trees: Vec<DecisionTreeRegressor<Trained>> =
            Vec::with_capacity(self.config.n_estimators);
        let mut importance = ImportanceAccumulator::new(n_features);

        for m in 0..self.config.n_estimators {
            // Pseudo-residuals for squared-error loss: r_i = y_i - F_{m-1}(x_i).
            let residuals = y - &current;

            let tree = DecisionTreeRegressor::new()
                .max_depth(self.config.max_depth)
                .min_samples_split(self.config.min_samples_split)
                .min_samples_leaf(self.config.min_samples_leaf)
                .random_state(self.config.random_state.map(|s| s + m as u64))
                .fit(X, &residuals)?;

            let update = tree.predict(X)?;
            // F_m = F_{m-1} + learning_rate * h_m.
            for i in 0..n_samples {
                current[i] += self.config.learning_rate * update[i];
            }

            if let Some(state) = tree.tree_.as_ref() {
                accumulate_importance(&state.root, X, &residuals, &all_indices, &mut importance);
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
        if X.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: X.ncols(),
            });
        }

        // Replay the additive model: F(x) = init_prediction + lr * sum_m h_m(x).
        let n_rows = X.nrows();
        let mut predictions = Array1::from_elem(n_rows, self.init_prediction);
        for tree in &self.trees {
            let update = tree.predict(X)?;
            for i in 0..n_rows {
                predictions[i] += self.config.learning_rate * update[i];
            }
        }
        Ok(predictions)
    }
}

impl TrainedGradientBoostingRegressor {
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
        assert!(matches!(pred_err, Err(SklearsError::FeatureMismatch { .. })));
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

        let pred = model.predict(&x).expect("classifier predict should succeed");
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
        let y_binary: Vec<Float> = y.iter().map(|&t| if t == 5.0 { 1.0 } else { 0.0 }).collect();

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
}
