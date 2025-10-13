//! Penalized Discriminant Analysis
//!
//! This module implements Penalized Discriminant Analysis (PDA) with various penalty functions
//! including L1, L2, elastic net, group lasso, SCAD, and MCP penalties.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};

/// Types of penalty functions
#[derive(Debug, Clone)]
pub enum PenaltyType {
    /// L1 penalty (Lasso)
    L1 { lambda: Float },
    /// L2 penalty (Ridge)
    L2 { lambda: Float },
    /// Elastic Net penalty (combination of L1 and L2)
    ElasticNet { lambda: Float, alpha: Float },
    /// Group Lasso penalty
    GroupLasso {
        lambda: Float,

        groups: Vec<Vec<usize>>,
    },
    /// SCAD penalty (Smoothly Clipped Absolute Deviation)
    SCAD { lambda: Float, a: Float },
    /// MCP penalty (Minimax Concave Penalty)
    MCP { lambda: Float, gamma: Float },
    /// Adaptive Lasso penalty
    AdaptiveLasso {
        lambda: Float,
        weights: Array1<Float>,
    },
    /// Fused Lasso penalty for ordered features
    FusedLasso { lambda1: Float, lambda2: Float },
}

/// Configuration for Penalized Discriminant Analysis
#[derive(Debug, Clone)]
pub struct PenalizedDiscriminantAnalysisConfig {
    /// Type of penalty to apply
    pub penalty: PenaltyType,
    /// Maximum number of iterations for optimization
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Learning rate for gradient descent
    pub learning_rate: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to standardize features
    pub standardize: bool,
    /// Number of components for dimensionality reduction
    pub n_components: Option<usize>,
    /// Optimization algorithm
    pub algorithm: String,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Line search parameters for optimization
    pub line_search: bool,
    /// Backtracking line search parameter
    pub beta: Float,
    /// Armijo condition parameter
    pub sigma: Float,
}

impl Default for PenalizedDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            penalty: PenaltyType::L1 { lambda: 0.01 },
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.01,
            fit_intercept: true,
            standardize: true,
            n_components: None,
            algorithm: "proximal_gradient".to_string(),
            random_state: None,
            line_search: false,
            beta: 0.5,
            sigma: 0.1,
        }
    }
}

/// Penalized Discriminant Analysis
pub struct PenalizedDiscriminantAnalysis {
    config: PenalizedDiscriminantAnalysisConfig,
}

impl PenalizedDiscriminantAnalysis {
    /// Create a new Penalized Discriminant Analysis instance
    pub fn new() -> Self {
        Self {
            config: PenalizedDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the penalty type
    pub fn penalty(mut self, penalty: PenaltyType) -> Self {
        self.config.penalty = penalty;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to standardize features
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.config.standardize = standardize;
        self
    }

    /// Set number of components
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set optimization algorithm
    pub fn algorithm(mut self, algorithm: &str) -> Self {
        self.config.algorithm = algorithm.to_string();
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Enable line search
    pub fn line_search(mut self, line_search: bool) -> Self {
        self.config.line_search = line_search;
        self
    }

    /// Standardize features
    fn standardize_features(
        &self,
        x: &Array2<Float>,
    ) -> (Array2<Float>, Array1<Float>, Array1<Float>) {
        if !self.config.standardize {
            let means = Array1::zeros(x.ncols());
            let stds = Array1::ones(x.ncols());
            return (x.clone(), means, stds);
        }

        let means = x.mean_axis(Axis(0)).unwrap();
        let mut stds = Array1::zeros(x.ncols());

        for j in 0..x.ncols() {
            let col = x.column(j);
            let var = col
                .iter()
                .map(|&val| (val - means[j]).powi(2))
                .sum::<Float>()
                / (x.nrows() - 1) as Float;
            stds[j] = var.sqrt().max(1e-8); // Avoid division by zero
        }

        let mut x_std = x.clone();
        for mut row in x_std.axis_iter_mut(Axis(0)) {
            for (j, val) in row.iter_mut().enumerate() {
                *val = (*val - means[j]) / stds[j];
            }
        }

        (x_std, means, stds)
    }

    /// Compute penalty value
    fn compute_penalty(&self, w: &Array2<Float>) -> Float {
        match &self.config.penalty {
            PenaltyType::L1 { lambda } => lambda * w.iter().map(|&x| x.abs()).sum::<Float>(),
            PenaltyType::L2 { lambda } => lambda * w.iter().map(|&x| x * x).sum::<Float>(),
            PenaltyType::ElasticNet { lambda, alpha } => {
                let l1_term = w.iter().map(|&x| x.abs()).sum::<Float>();
                let l2_term = w.iter().map(|&x| x * x).sum::<Float>();
                lambda * (alpha * l1_term + (1.0 - alpha) * l2_term)
            }
            PenaltyType::GroupLasso { lambda, groups } => {
                let mut penalty = 0.0;
                for group in groups {
                    let group_norm: Float = group
                        .iter()
                        .map(|&idx| w.iter().nth(idx).unwrap_or(&0.0).powi(2))
                        .sum::<Float>()
                        .sqrt();
                    penalty += group_norm;
                }
                lambda * penalty
            }
            PenaltyType::SCAD { lambda, a } => {
                let mut penalty = 0.0;
                for &x in w.iter() {
                    let abs_x = x.abs();
                    if abs_x <= *lambda {
                        penalty += lambda * abs_x;
                    } else if abs_x <= a * lambda {
                        penalty += (2.0 * a * lambda * abs_x - abs_x * abs_x - lambda * lambda)
                            / (2.0 * (a - 1.0));
                    } else {
                        penalty += lambda * lambda * (a + 1.0) / 2.0;
                    }
                }
                penalty
            }
            PenaltyType::MCP { lambda, gamma } => {
                let mut penalty = 0.0;
                for &x in w.iter() {
                    let abs_x = x.abs();
                    if abs_x <= gamma * lambda {
                        penalty += lambda * abs_x - abs_x * abs_x / (2.0 * gamma);
                    } else {
                        penalty += gamma * lambda * lambda / 2.0;
                    }
                }
                penalty
            }
            PenaltyType::AdaptiveLasso { lambda, weights } => {
                lambda
                    * w.iter()
                        .zip(weights.iter().cycle())
                        .map(|(&x, &weight)| weight * x.abs())
                        .sum::<Float>()
            }
            PenaltyType::FusedLasso { lambda1, lambda2 } => {
                let l1_term = w.iter().map(|&x| x.abs()).sum::<Float>();
                let mut fusion_term = 0.0;
                for i in 1..w.len() {
                    let diff = w.iter().nth(i).unwrap() - w.iter().nth(i - 1).unwrap();
                    fusion_term += diff.abs();
                }
                lambda1 * l1_term + lambda2 * fusion_term
            }
        }
    }

    /// Compute penalty gradient
    fn compute_penalty_gradient(&self, w: &Array2<Float>) -> Array2<Float> {
        let mut grad = Array2::zeros(w.dim());

        match &self.config.penalty {
            PenaltyType::L1 { lambda } => {
                for (i, &val) in w.iter().enumerate() {
                    grad.as_slice_mut().unwrap()[i] = lambda * val.signum();
                }
            }
            PenaltyType::L2 { lambda } => {
                grad = w * (2.0 * lambda);
            }
            PenaltyType::ElasticNet { lambda, alpha } => {
                for (i, &val) in w.iter().enumerate() {
                    grad.as_slice_mut().unwrap()[i] =
                        lambda * (alpha * val.signum() + (1.0 - alpha) * 2.0 * val);
                }
            }
            PenaltyType::GroupLasso { lambda, groups } => {
                for group in groups {
                    let group_norm: Float = group
                        .iter()
                        .map(|&idx| w.iter().nth(idx).unwrap_or(&0.0).powi(2))
                        .sum::<Float>()
                        .sqrt();

                    if group_norm > 0.0 {
                        for &idx in group {
                            if let Some(val) = w.iter().nth(idx) {
                                grad.as_slice_mut().unwrap()[idx] = lambda * val / group_norm;
                            }
                        }
                    }
                }
            }
            PenaltyType::SCAD { lambda, a } => {
                for (i, &val) in w.iter().enumerate() {
                    let abs_val = val.abs();
                    let grad_val = if abs_val <= *lambda {
                        lambda * val.signum()
                    } else if abs_val <= a * lambda {
                        (a * lambda * val.signum() - val) / (a - 1.0)
                    } else {
                        0.0
                    };
                    grad.as_slice_mut().unwrap()[i] = grad_val;
                }
            }
            PenaltyType::MCP { lambda, gamma } => {
                for (i, &val) in w.iter().enumerate() {
                    let abs_val = val.abs();
                    let grad_val = if abs_val <= gamma * lambda {
                        lambda * val.signum() - val / gamma
                    } else {
                        0.0
                    };
                    grad.as_slice_mut().unwrap()[i] = grad_val;
                }
            }
            PenaltyType::AdaptiveLasso { lambda, weights } => {
                for (i, &val) in w.iter().enumerate() {
                    let weight = weights.get(i % weights.len()).unwrap_or(&1.0);
                    grad.as_slice_mut().unwrap()[i] = lambda * weight * val.signum();
                }
            }
            PenaltyType::FusedLasso { lambda1, lambda2 } => {
                // L1 gradient
                for (i, &val) in w.iter().enumerate() {
                    grad.as_slice_mut().unwrap()[i] += lambda1 * val.signum();
                }

                // Fusion gradient
                let w_vec: Vec<Float> = w.iter().cloned().collect();
                for i in 0..w_vec.len() {
                    let mut fusion_grad = 0.0;

                    // Gradient from (w[i] - w[i-1])
                    if i > 0 {
                        let diff = w_vec[i] - w_vec[i - 1];
                        fusion_grad += lambda2 * diff.signum();
                    }

                    // Gradient from (w[i+1] - w[i])
                    if i < w_vec.len() - 1 {
                        let diff = w_vec[i + 1] - w_vec[i];
                        fusion_grad -= lambda2 * diff.signum();
                    }

                    grad.as_slice_mut().unwrap()[i] += fusion_grad;
                }
            }
        }

        grad
    }

    /// Proximal operator for different penalties
    fn proximal_operator(&self, w: &Array2<Float>, step_size: Float) -> Array2<Float> {
        match &self.config.penalty {
            PenaltyType::L1 { lambda } => {
                let threshold = lambda * step_size;
                w.mapv(|x| {
                    if x.abs() <= threshold {
                        0.0
                    } else {
                        x - threshold * x.signum()
                    }
                })
            }
            PenaltyType::L2 { lambda } => {
                let shrinkage = 1.0 / (1.0 + 2.0 * lambda * step_size);
                w * shrinkage
            }
            PenaltyType::ElasticNet { lambda, alpha } => {
                let l1_threshold = alpha * lambda * step_size;
                let l2_shrinkage = 1.0 / (1.0 + 2.0 * (1.0 - alpha) * lambda * step_size);

                w.mapv(|x| {
                    let soft_thresholded = if x.abs() <= l1_threshold {
                        0.0
                    } else {
                        x - l1_threshold * x.signum()
                    };
                    soft_thresholded * l2_shrinkage
                })
            }
            _ => {
                // For complex penalties, use gradient descent step
                let grad = self.compute_penalty_gradient(w);
                w - &(grad * step_size)
            }
        }
    }

    /// Optimize using proximal gradient descent
    fn optimize_proximal_gradient(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_classes = classes.len();

        // Initialize weights and intercept
        let mut w = Array2::zeros((n_features, n_classes - 1));
        let mut intercept = Array1::zeros(n_classes - 1);

        // Convert labels to one-hot encoding (for binary classification, use single column)
        let mut y_encoded = Array2::zeros((n_samples, n_classes - 1));
        for (i, &label) in y.iter().enumerate() {
            let class_idx = classes.iter().position(|&c| c == label).unwrap();
            if class_idx < n_classes - 1 {
                y_encoded[[i, class_idx]] = 1.0;
            }
        }

        let mut learning_rate = self.config.learning_rate;

        for iter in 0..self.config.max_iter {
            // Compute predictions
            let mut predictions = x.dot(&w);
            if self.config.fit_intercept {
                for mut row in predictions.axis_iter_mut(Axis(0)) {
                    row += &intercept;
                }
            }

            // Compute loss and gradients using logistic regression
            let (loss, w_grad, intercept_grad) =
                self.compute_logistic_loss_and_gradients(&predictions, &y_encoded, x)?;

            // Add penalty to loss for monitoring
            let penalty = self.compute_penalty(&w);
            let total_loss = loss + penalty;

            // Line search if enabled
            if self.config.line_search {
                learning_rate = self.backtracking_line_search(
                    x,
                    &y_encoded,
                    &w,
                    &intercept,
                    &w_grad,
                    &intercept_grad,
                    learning_rate,
                )?;
            }

            // Update weights using proximal operator
            let w_new = &w - &(&w_grad * learning_rate);
            w = self.proximal_operator(&w_new, learning_rate);

            // Update intercept
            if self.config.fit_intercept {
                intercept = &intercept - &(&intercept_grad * learning_rate);
            }

            // Check convergence
            if iter > 0
                && (w_grad.iter().map(|&x| x.abs()).sum::<Float>()
                    + intercept_grad.iter().map(|&x| x.abs()).sum::<Float>())
                    < self.config.tol
            {
                break;
            }

            // Update learning rate for next iteration
            learning_rate *= 0.99;
        }

        Ok((w, intercept))
    }

    /// Compute logistic loss and gradients
    fn compute_logistic_loss_and_gradients(
        &self,
        predictions: &Array2<Float>,
        y_true: &Array2<Float>,
        x: &Array2<Float>,
    ) -> Result<(Float, Array2<Float>, Array1<Float>)> {
        let n_samples = predictions.nrows();

        // Compute softmax probabilities
        let mut probas = Array2::zeros(predictions.dim());
        for (i, pred_row) in predictions.axis_iter(Axis(0)).enumerate() {
            let max_pred = pred_row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_preds: Vec<Float> = pred_row.iter().map(|&x| (x - max_pred).exp()).collect();
            let sum_exp = exp_preds.iter().sum::<Float>() + 1.0; // +1 for reference class

            for (j, &exp_pred) in exp_preds.iter().enumerate() {
                probas[[i, j]] = exp_pred / sum_exp;
            }
        }

        // Compute cross-entropy loss
        let mut loss = 0.0;
        for i in 0..n_samples {
            for j in 0..predictions.ncols() {
                let p = probas[[i, j]].max(1e-15); // Avoid log(0)
                loss -= y_true[[i, j]] * p.ln();
            }
            // Add reference class
            let ref_prob = 1.0 - probas.row(i).sum();
            let ref_true = 1.0 - y_true.row(i).sum();
            if ref_prob > 1e-15 {
                loss -= ref_true * ref_prob.ln();
            }
        }
        loss /= n_samples as Float;

        // Compute gradients
        let residuals = &probas - y_true;
        let w_grad = x.t().dot(&residuals) / n_samples as Float;
        let intercept_grad = residuals.mean_axis(Axis(0)).unwrap();

        Ok((loss, w_grad, intercept_grad))
    }

    /// Backtracking line search
    fn backtracking_line_search(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        w: &Array2<Float>,
        intercept: &Array1<Float>,
        w_grad: &Array2<Float>,
        intercept_grad: &Array1<Float>,
        initial_step: Float,
    ) -> Result<Float> {
        let mut step_size = initial_step;
        let max_backtracks = 20;

        // Current objective value
        let mut current_pred = x.dot(w);
        if self.config.fit_intercept {
            for mut row in current_pred.axis_iter_mut(Axis(0)) {
                row += intercept;
            }
        }
        let (current_loss, _, _) = self.compute_logistic_loss_and_gradients(&current_pred, y, x)?;
        let current_penalty = self.compute_penalty(w);
        let current_obj = current_loss + current_penalty;

        // Directional derivative
        let directional_deriv = w_grad
            .iter()
            .zip(w_grad.iter())
            .map(|(&g, &g2)| g * g2)
            .sum::<Float>()
            + intercept_grad
                .iter()
                .zip(intercept_grad.iter())
                .map(|(&g, &g2)| g * g2)
                .sum::<Float>();

        for _ in 0..max_backtracks {
            // Try step
            let w_new = w - &(w_grad * step_size);
            let intercept_new = intercept - &(intercept_grad * step_size);

            let mut new_pred = x.dot(&w_new);
            if self.config.fit_intercept {
                for mut row in new_pred.axis_iter_mut(Axis(0)) {
                    row += &intercept_new;
                }
            }

            let (new_loss, _, _) = self.compute_logistic_loss_and_gradients(&new_pred, y, x)?;
            let new_penalty = self.compute_penalty(&w_new);
            let new_obj = new_loss + new_penalty;

            // Armijo condition
            if new_obj <= current_obj - self.config.sigma * step_size * directional_deriv {
                return Ok(step_size);
            }

            step_size *= self.config.beta;
        }

        Ok(step_size)
    }
}

/// Trained Penalized Discriminant Analysis model
pub struct TrainedPenalizedDiscriminantAnalysis {
    config: PenalizedDiscriminantAnalysisConfig,
    weights: Array2<Float>,
    intercept: Array1<Float>,
    classes: Vec<i32>,
    feature_means: Array1<Float>,
    feature_stds: Array1<Float>,
    components: Option<Array2<Float>>,
}

impl TrainedPenalizedDiscriminantAnalysis {
    /// Get the learned weights
    pub fn weights(&self) -> &Array2<Float> {
        &self.weights
    }

    /// Get the intercept
    pub fn intercept(&self) -> &Array1<Float> {
        &self.intercept
    }

    /// Get the classes
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get the transformation components
    pub fn components(&self) -> Option<&Array2<Float>> {
        self.components.as_ref()
    }

    /// Get feature importance (absolute weights)
    pub fn feature_importance(&self) -> Array1<Float> {
        self.weights
            .map_axis(Axis(1), |row| row.iter().map(|&x| x.abs()).sum())
    }

    /// Get sparsity level (fraction of zero weights)
    pub fn sparsity(&self) -> Float {
        let total_weights = self.weights.len() as Float;
        let zero_weights = self.weights.iter().filter(|&&x| x.abs() < 1e-10).count() as Float;
        zero_weights / total_weights
    }
}

impl Estimator for PenalizedDiscriminantAnalysis {
    type Config = PenalizedDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for PenalizedDiscriminantAnalysis {
    type Fitted = TrainedPenalizedDiscriminantAnalysis;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidData {
                reason: "Number of samples in X and y must match".to_string(),
            });
        }

        // Get unique classes
        let classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        if classes.len() < 2 {
            return Err(SklearsError::InvalidData {
                reason: "Need at least 2 classes for classification".to_string(),
            });
        }

        // Standardize features
        let (x_std, feature_means, feature_stds) = self.standardize_features(x);

        // Optimize weights
        let (weights, intercept) = match self.config.algorithm.as_str() {
            "proximal_gradient" => self.optimize_proximal_gradient(&x_std, y, &classes)?,
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "parameter".to_string(),
                    reason: format!("Unknown algorithm: {}", self.config.algorithm),
                })
            }
        };

        // Compute components for dimensionality reduction if requested
        let components = if let Some(n_comp) = self.config.n_components {
            let max_components = weights.ncols(); // For binary: 1, for multiclass: n_classes-1
            let comp_dim = n_comp.min(max_components);
            if comp_dim > 0 {
                // Take all features and the available components
                Some(weights.slice(s![.., ..comp_dim]).to_owned())
            } else {
                None
            }
        } else {
            None
        };

        Ok(TrainedPenalizedDiscriminantAnalysis {
            config: self.config,
            weights,
            intercept,
            classes,
            feature_means,
            feature_stds,
            components,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedPenalizedDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, proba_row) in probas.axis_iter(Axis(0)).enumerate() {
            let max_idx = proba_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedPenalizedDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.feature_means.len() {
            return Err(SklearsError::InvalidData {
                reason: "Number of features in X does not match training data".to_string(),
            });
        }

        // Standardize input
        let mut x_std = x.clone();
        if self.config.standardize {
            for mut row in x_std.axis_iter_mut(Axis(0)) {
                for (j, val) in row.iter_mut().enumerate() {
                    *val = (*val - self.feature_means[j]) / self.feature_stds[j];
                }
            }
        }

        // Compute linear predictions
        let mut predictions = x_std.dot(&self.weights);
        if self.config.fit_intercept {
            for mut row in predictions.axis_iter_mut(Axis(0)) {
                row += &self.intercept;
            }
        }

        // Convert to probabilities using softmax
        let n_samples = predictions.nrows();
        let n_classes = self.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (i, pred_row) in predictions.axis_iter(Axis(0)).enumerate() {
            let max_pred = pred_row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 1.0; // For reference class

            for (j, &pred) in pred_row.iter().enumerate() {
                let exp_pred = (pred - max_pred).exp();
                probas[[i, j]] = exp_pred;
                exp_sum += exp_pred;
            }

            // Normalize probabilities
            for j in 0..pred_row.len() {
                probas[[i, j]] /= exp_sum;
            }

            // Set reference class probability
            if n_classes > pred_row.len() {
                probas[[i, n_classes - 1]] = 1.0 / exp_sum;
            }
        }

        Ok(probas)
    }
}

impl Transform<Array2<Float>> for TrainedPenalizedDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if let Some(ref components) = self.components {
            if x.ncols() != self.feature_means.len() {
                return Err(SklearsError::InvalidData {
                    reason: "Number of features in X does not match training data".to_string(),
                });
            }

            // Standardize input
            let mut x_std = x.clone();
            if self.config.standardize {
                for mut row in x_std.axis_iter_mut(Axis(0)) {
                    for (j, val) in row.iter_mut().enumerate() {
                        *val = (*val - self.feature_means[j]) / self.feature_stds[j];
                    }
                }
            }

            Ok(x_std.dot(components))
        } else {
            Err(SklearsError::NotFitted {
                operation: "transform - no components available. Set n_components during fitting"
                    .to_string(),
            })
        }
    }
}

impl Default for PenalizedDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_penalized_discriminant_analysis_l1() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let pda = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::L1 { lambda: 0.1 })
            .max_iter(100);

        let fitted = pda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
        assert!(fitted.sparsity() >= 0.0);
        assert!(fitted.sparsity() <= 1.0);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        let probas = fitted.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_penalized_discriminant_analysis_elastic_net() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let pda = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::ElasticNet {
                lambda: 0.1,
                alpha: 0.5,
            })
            .max_iter(50);

        let fitted = pda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (4, 2));
    }

    #[test]
    fn test_penalized_discriminant_analysis_scad() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let pda = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::SCAD {
                lambda: 0.1,
                a: 3.7,
            })
            .max_iter(50);

        let fitted = pda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_penalized_discriminant_analysis_mcp() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let pda = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::MCP {
                lambda: 0.1,
                gamma: 3.0,
            })
            .max_iter(50);

        let fitted = pda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_penalized_discriminant_analysis_group_lasso() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let groups = vec![vec![0, 1], vec![2, 3]];
        let pda = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::GroupLasso {
                lambda: 0.1,
                groups,
            })
            .max_iter(50);

        let fitted = pda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_penalized_discriminant_analysis_with_components() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let pda = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::L1 { lambda: 0.1 })
            .n_components(Some(1)) // For binary classification, max 1 component
            .max_iter(50);

        let fitted = pda.fit(&x, &y).unwrap();

        assert!(fitted.components().is_some());

        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed.dim(), (4, 1)); // 4 samples, 1 component
    }

    #[test]
    fn test_feature_importance() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let pda = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::L1 { lambda: 0.1 })
            .max_iter(50);

        let fitted = pda.fit(&x, &y).unwrap();

        let importance = fitted.feature_importance();
        assert_eq!(importance.len(), 3);
        assert!(importance.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_sparsity_computation() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        // High penalty should lead to sparser solution
        let pda_sparse = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::L1 { lambda: 1.0 })
            .max_iter(100);

        let fitted_sparse = pda_sparse.fit(&x, &y).unwrap();

        // Low penalty should lead to less sparse solution
        let pda_dense = PenalizedDiscriminantAnalysis::new()
            .penalty(PenaltyType::L1 { lambda: 0.001 })
            .max_iter(100);

        let fitted_dense = pda_dense.fit(&x, &y).unwrap();

        // Sparse model should have higher sparsity
        assert!(fitted_sparse.sparsity() >= fitted_dense.sparsity());
    }
}
