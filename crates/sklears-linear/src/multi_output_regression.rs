//! Multi-output regression models
//!
//! This module provides multi-output regression capabilities including:
//! - Multi-output Ridge and Lasso regression
//! - Chain rule for structured outputs
//! - Target correlation modeling
//! - Output space dimensionality reduction
//!
//! These models can handle multiple target variables simultaneously,
//! potentially leveraging correlations between targets for improved performance.

// The nalgebra-tests feature is intentionally not declared in Cargo.toml —
// it exists only as a gate for legacy test code that predates the ndarray migration.
#![allow(unexpected_cfgs)]

use crate::errors::{LinearModelError, OptimizationError, OptimizationErrorKind};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::compat::svd;

/// Helper function to create OptimizationError instances
fn optimization_error(
    kind: OptimizationErrorKind,
    algorithm: &str,
    message: &str,
) -> LinearModelError {
    LinearModelError::OptimizationError(OptimizationError {
        kind,
        algorithm: algorithm.to_string(),
        iteration: None,
        max_iterations: None,
        convergence_info: None,
        suggestions: vec![message.to_string()],
    })
}

/// Strategy for handling multiple outputs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultiOutputStrategy {
    /// Independent models for each output (no correlation modeling)
    Independent,
    /// Joint modeling with shared regularization
    Joint,
    /// Chain rule modeling (ordered dependency)
    Chain,
    /// Reduced rank modeling with dimensionality reduction
    ReducedRank,
}

/// Configuration for multi-output regression
#[derive(Debug, Clone)]
pub struct MultiOutputConfig {
    /// Regularization parameter (alpha)
    pub alpha: f64,
    /// L1 ratio for elastic net (0.0 = Ridge, 1.0 = Lasso)
    pub l1_ratio: f64,
    /// Multi-output strategy
    pub strategy: MultiOutputStrategy,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Rank for reduced rank modeling (None = auto)
    pub rank: Option<usize>,
    /// Enable target correlation modeling
    pub model_correlations: bool,
    /// Chain order for chain rule (None = auto)
    pub chain_order: Option<Vec<usize>>,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for MultiOutputConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.0, // Ridge by default
            strategy: MultiOutputStrategy::Independent,
            max_iter: 1000,
            tolerance: 1e-4,
            rank: None,
            model_correlations: false,
            chain_order: None,
            fit_intercept: true,
            random_state: None,
        }
    }
}

/// Result of multi-output regression training
#[derive(Debug, Clone)]
pub struct MultiOutputResult {
    /// Coefficient matrix (features x targets)
    pub coefficients: Array2<f64>,
    /// Intercept vector (if fitted)
    pub intercept: Option<Array1<f64>>,
    /// Target correlation matrix (if modeled)
    pub target_correlations: Option<Array2<f64>>,
    /// Reduced rank factors (if using reduced rank)
    pub rank_factors: Option<(Array2<f64>, Array2<f64>)>,
    /// Chain order used (if chain strategy)
    pub chain_order: Option<Vec<usize>>,
    /// Final training loss
    pub training_loss: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
}

/// Multi-output regression model
pub struct MultiOutputRegression {
    config: MultiOutputConfig,
    result: Option<MultiOutputResult>,
    n_features: Option<usize>,
    n_targets: Option<usize>,
}

impl MultiOutputRegression {
    /// Create a new multi-output regression model
    pub fn new(config: MultiOutputConfig) -> Self {
        Self {
            config,
            result: None,
            n_features: None,
            n_targets: None,
        }
    }

    /// Create a Ridge multi-output model
    pub fn ridge(alpha: f64) -> Self {
        let config = MultiOutputConfig {
            alpha,
            l1_ratio: 0.0,
            strategy: MultiOutputStrategy::Joint,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a Lasso multi-output model
    pub fn lasso(alpha: f64) -> Self {
        let config = MultiOutputConfig {
            alpha,
            l1_ratio: 1.0,
            strategy: MultiOutputStrategy::Joint,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create an elastic net multi-output model
    pub fn elastic_net(alpha: f64, l1_ratio: f64) -> Self {
        let config = MultiOutputConfig {
            alpha,
            l1_ratio,
            strategy: MultiOutputStrategy::Joint,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a reduced rank model
    pub fn reduced_rank(alpha: f64, rank: usize) -> Self {
        let config = MultiOutputConfig {
            alpha,
            strategy: MultiOutputStrategy::ReducedRank,
            rank: Some(rank),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a chain rule model
    pub fn chain_rule(alpha: f64, chain_order: Option<Vec<usize>>) -> Self {
        let config = MultiOutputConfig {
            alpha,
            strategy: MultiOutputStrategy::Chain,
            chain_order,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Fit the model to training data
    #[allow(non_snake_case)]
    #[allow(clippy::result_large_err)]
    pub fn fit(&mut self, X: &Array2<f64>, Y: &Array2<f64>) -> Result<(), LinearModelError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_targets = Y.ncols();

        if X.nrows() != Y.nrows() {
            return Err(optimization_error(
                OptimizationErrorKind::InvalidProblemDimensions,
                "MultiOutputRegression",
                "Number of samples in X and Y must match",
            ));
        }

        self.n_features = Some(n_features);
        self.n_targets = Some(n_targets);

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean_opt, y_mean_opt) = if self.config.fit_intercept {
            let x_mean = X.mean_axis(Axis(0)).ok_or_else(|| {
                optimization_error(
                    OptimizationErrorKind::InvalidProblemDimensions,
                    "MultiOutputRegression",
                    "Cannot compute column means of X (empty matrix?)",
                )
            })?;
            let y_mean = Y.mean_axis(Axis(0)).ok_or_else(|| {
                optimization_error(
                    OptimizationErrorKind::InvalidProblemDimensions,
                    "MultiOutputRegression",
                    "Cannot compute column means of Y (empty matrix?)",
                )
            })?;

            // Broadcast mean rows across all samples
            let x_mean_mat = x_mean
                .view()
                .insert_axis(Axis(0))
                .broadcast((n_samples, n_features))
                .ok_or_else(|| {
                    optimization_error(
                        OptimizationErrorKind::InvalidProblemDimensions,
                        "MultiOutputRegression",
                        "Broadcast of X mean failed",
                    )
                })?
                .to_owned();
            let y_mean_mat = y_mean
                .view()
                .insert_axis(Axis(0))
                .broadcast((n_samples, n_targets))
                .ok_or_else(|| {
                    optimization_error(
                        OptimizationErrorKind::InvalidProblemDimensions,
                        "MultiOutputRegression",
                        "Broadcast of Y mean failed",
                    )
                })?
                .to_owned();

            let xc = X - &x_mean_mat;
            let yc = Y - &y_mean_mat;
            (xc, yc, Some(x_mean), Some(y_mean))
        } else {
            (X.clone(), Y.clone(), None, None)
        };

        // Fit model based on strategy
        let result = match self.config.strategy {
            MultiOutputStrategy::Independent => self.fit_independent(&x_centered, &y_centered)?,
            MultiOutputStrategy::Joint => self.fit_joint(&x_centered, &y_centered)?,
            MultiOutputStrategy::Chain => self.fit_chain(&x_centered, &y_centered)?,
            MultiOutputStrategy::ReducedRank => self.fit_reduced_rank(&x_centered, &y_centered)?,
        };

        // Compute intercept if needed
        // intercept = y_mean - W^T * x_mean
        let intercept = if let (Some(xm), Some(ym)) = (x_mean_opt, y_mean_opt) {
            // result.coefficients is (n_features x n_targets)
            // W^T * x_mean → shape (n_targets,)
            let pred_mean = result.coefficients.t().dot(&xm);
            Some(ym - pred_mean)
        } else {
            None
        };

        self.result = Some(MultiOutputResult {
            coefficients: result.coefficients,
            intercept,
            target_correlations: result.target_correlations,
            rank_factors: result.rank_factors,
            chain_order: result.chain_order,
            training_loss: result.training_loss,
            iterations: result.iterations,
            converged: result.converged,
        });

        Ok(())
    }

    /// Predict using the fitted model
    #[allow(non_snake_case)]
    #[allow(clippy::result_large_err)]
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array2<f64>, LinearModelError> {
        let result = self.result.as_ref().ok_or_else(|| {
            optimization_error(
                OptimizationErrorKind::ModelNotFitted,
                "MultiOutputRegression",
                "Model must be fitted before prediction",
            )
        })?;

        let n_features = self.n_features.ok_or_else(|| {
            optimization_error(
                OptimizationErrorKind::ModelNotFitted,
                "MultiOutputRegression",
                "Model features not initialized",
            )
        })?;

        if X.ncols() != n_features {
            return Err(optimization_error(
                OptimizationErrorKind::InvalidProblemDimensions,
                "MultiOutputRegression",
                "Number of features in X must match training data",
            ));
        }

        // X is (n_samples x n_features), coefficients is (n_features x n_targets)
        // predictions is (n_samples x n_targets)
        let mut predictions = X.dot(&result.coefficients);

        if let Some(intercept) = &result.intercept {
            for i in 0..predictions.nrows() {
                for j in 0..predictions.ncols() {
                    predictions[[i, j]] += intercept[j];
                }
            }
        }

        Ok(predictions)
    }

    /// Get the model coefficients
    pub fn coefficients(&self) -> Option<&Array2<f64>> {
        self.result.as_ref().map(|r| &r.coefficients)
    }

    /// Get the model intercept
    pub fn intercept(&self) -> Option<&Array1<f64>> {
        self.result.as_ref().and_then(|r| r.intercept.as_ref())
    }

    /// Get target correlations (if modeled)
    pub fn target_correlations(&self) -> Option<&Array2<f64>> {
        self.result
            .as_ref()
            .and_then(|r| r.target_correlations.as_ref())
    }

    /// Fit independent models for each target
    #[allow(clippy::result_large_err)]
    fn fit_independent(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<MultiOutputResult, LinearModelError> {
        let n_features = x.ncols();
        let n_targets = y.ncols();
        let mut coefficients = Array2::<f64>::zeros((n_features, n_targets));
        let mut total_loss = 0.0;
        let mut total_iterations = 0;
        let mut all_converged = true;

        for target_idx in 0..n_targets {
            let y_col = y.column(target_idx).to_owned();
            let (coef, loss, iters, converged) = self.fit_single_target(x, &y_col)?;
            coefficients.column_mut(target_idx).assign(&coef);
            total_loss += loss;
            total_iterations += iters;
            all_converged &= converged;
        }

        Ok(MultiOutputResult {
            coefficients,
            intercept: None,
            target_correlations: None,
            rank_factors: None,
            chain_order: None,
            training_loss: total_loss / n_targets as f64,
            iterations: total_iterations / n_targets,
            converged: all_converged,
        })
    }

    /// Fit joint model with shared regularization
    #[allow(clippy::result_large_err)]
    fn fit_joint(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<MultiOutputResult, LinearModelError> {
        let n_features = x.ncols();
        let n_targets = y.ncols();

        // Vectorize the problem: vec(Y) = kron(I, X) * vec(W)
        // Where W is the coefficient matrix (features x targets)

        // For joint modeling, we solve: min ||Y - XW||_F^2 + alpha * R(W)
        // where R(W) is the regularization term

        let mut w = Array2::<f64>::zeros((n_features, n_targets));
        let mut converged = false;
        let mut iteration = 0;
        let mut prev_loss = f64::INFINITY;

        // Use coordinate descent for joint optimization
        while iteration < self.config.max_iter && !converged {
            // Update each coefficient
            for j in 0..n_features {
                for k in 0..n_targets {
                    // Compute residual without current coefficient
                    let mut residual = y.column(k).to_owned();
                    for i in 0..n_features {
                        if i != j {
                            let xi = x.column(i).to_owned();
                            let scale = w[[i, k]];
                            residual.scaled_add(-scale, &xi);
                        }
                    }

                    // Compute correlation with feature j
                    let x_j = x.column(j);
                    let correlation = x_j.dot(&residual);
                    let norm_sq = x_j.dot(&x_j);

                    // Apply soft thresholding for Lasso component
                    let lasso_penalty = self.config.alpha * self.config.l1_ratio;
                    let ridge_penalty = self.config.alpha * (1.0 - self.config.l1_ratio);

                    let new_coef = if norm_sq > 0.0 {
                        let threshold = lasso_penalty;
                        let denominator = norm_sq + ridge_penalty;

                        if correlation > threshold {
                            (correlation - threshold) / denominator
                        } else if correlation < -threshold {
                            (correlation + threshold) / denominator
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    w[[j, k]] = new_coef;
                }
            }

            // Compute loss: 0.5 * ||Y - X*W||_F^2
            let predictions = x.dot(&w);
            let residuals = y - &predictions;
            let current_loss_base: f64 = residuals.iter().map(|v| v * v).sum::<f64>() * 0.5;

            // Add regularization terms
            let l1_penalty =
                self.config.alpha * self.config.l1_ratio * w.iter().map(|x| x.abs()).sum::<f64>();
            let l2_penalty = 0.5
                * self.config.alpha
                * (1.0 - self.config.l1_ratio)
                * w.iter().map(|v| v * v).sum::<f64>();
            let current_loss = current_loss_base + l1_penalty + l2_penalty;

            // Check convergence
            let loss_change = (prev_loss - current_loss).abs() / prev_loss.max(1.0);
            converged = loss_change < self.config.tolerance;
            prev_loss = current_loss;
            iteration += 1;
        }

        // Compute target correlations if requested
        let target_correlations = if self.config.model_correlations {
            Some(self.compute_target_correlations(y)?)
        } else {
            None
        };

        Ok(MultiOutputResult {
            coefficients: w,
            intercept: None,
            target_correlations,
            rank_factors: None,
            chain_order: None,
            training_loss: prev_loss,
            iterations: iteration,
            converged,
        })
    }

    /// Fit chain rule model
    #[allow(clippy::result_large_err)]
    fn fit_chain(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<MultiOutputResult, LinearModelError> {
        let n_targets = y.ncols();

        // Determine chain order
        let chain_order = if let Some(order) = &self.config.chain_order {
            order.clone()
        } else {
            // Auto-determine order based on target correlations
            self.auto_determine_chain_order(y)?
        };

        if chain_order.len() != n_targets {
            return Err(optimization_error(
                OptimizationErrorKind::InvalidProblemDimensions,
                "MultiOutputRegression",
                "Chain order must include all targets",
            ));
        }

        let n_features = x.ncols();
        let n_samples = x.nrows();
        let mut coefficients = Array2::<f64>::zeros((n_features, n_targets));
        let mut total_loss = 0.0;
        let mut total_iterations = 0;
        let mut all_converged = true;

        // Fit targets in chain order
        for (chain_idx, &target_idx) in chain_order.iter().enumerate() {
            let y_col = y.column(target_idx).to_owned();

            // For targets after the first, include previous targets as features
            let current_x: Array2<f64> = if chain_idx > 0 {
                // Build extended feature matrix: original X + previous predicted targets
                let extra_cols = chain_idx;
                let mut extended_x = Array2::<f64>::zeros((n_samples, n_features + extra_cols));
                extended_x
                    .slice_mut(scirs2_core::ndarray::s![.., ..n_features])
                    .assign(x);
                for (prev_idx, &prev_target_idx) in chain_order[..chain_idx].iter().enumerate() {
                    let prev_col = y.column(prev_target_idx).to_owned();
                    extended_x
                        .column_mut(n_features + prev_idx)
                        .assign(&prev_col);
                }
                extended_x
            } else {
                x.clone()
            };

            let (coef, loss, iters, converged) = self.fit_single_target(&current_x, &y_col)?;

            // Store only the original feature coefficients (not the augmented target features)
            coefficients
                .column_mut(target_idx)
                .assign(&coef.slice(scirs2_core::ndarray::s![..n_features]));

            total_loss += loss;
            total_iterations += iters;
            all_converged &= converged;
        }

        Ok(MultiOutputResult {
            coefficients,
            intercept: None,
            target_correlations: None,
            rank_factors: None,
            chain_order: Some(chain_order),
            training_loss: total_loss / n_targets as f64,
            iterations: total_iterations / n_targets,
            converged: all_converged,
        })
    }

    /// Fit reduced rank model
    #[allow(clippy::result_large_err)]
    fn fit_reduced_rank(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<MultiOutputResult, LinearModelError> {
        let n_features = x.ncols();
        let n_targets = y.ncols();

        // Determine rank
        let rank = self
            .config
            .rank
            .unwrap_or_else(|| (n_features.min(n_targets) / 2).max(1));

        if rank >= n_features.min(n_targets) {
            return Err(optimization_error(
                OptimizationErrorKind::InvalidProblemDimensions,
                "MultiOutputRegression",
                "Rank must be less than min(n_features, n_targets)",
            ));
        }

        // First fit full rank model
        let full_result = self.fit_joint(x, y)?;
        let w_full = full_result.coefficients;

        // Apply SVD to get reduced rank approximation: W = U * S * V^T
        // svd returns (U, singular_values, Vt) where Vt is already V^T
        let (u, singular_values, vt) = svd(&w_full.view(), true).map_err(|e| {
            optimization_error(
                OptimizationErrorKind::ConvergenceFailed,
                "MultiOutputRegression",
                &format!("SVD failed: {e}"),
            )
        })?;

        // Truncate to desired rank
        let u_truncated = u.slice(scirs2_core::ndarray::s![.., ..rank]).to_owned();
        let sv_truncated = singular_values
            .slice(scirs2_core::ndarray::s![..rank])
            .to_owned();
        let vt_truncated = vt.slice(scirs2_core::ndarray::s![..rank, ..]).to_owned();

        // Build diagonal S matrix for rank factors storage
        let mut s_mat = Array2::<f64>::zeros((rank, rank));
        for i in 0..rank {
            s_mat[[i, i]] = sv_truncated[i];
        }

        // Reconstruct coefficient matrix: W_r = U_r * S_r * Vt_r
        // (n_features x rank) * (rank x rank) * (rank x n_targets)
        let us = u_truncated.dot(&s_mat);
        let coefficients = us.dot(&vt_truncated);

        // Compute loss with reduced rank model
        let predictions = x.dot(&coefficients);
        let residuals = y - &predictions;
        let training_loss: f64 = residuals.iter().map(|v| v * v).sum::<f64>() * 0.5;

        // rank_factors = (U_r, S_r * Vt_r)
        let sv_vt = s_mat.dot(&vt_truncated);

        Ok(MultiOutputResult {
            coefficients,
            intercept: None,
            target_correlations: None,
            rank_factors: Some((u_truncated, sv_vt)),
            chain_order: None,
            training_loss,
            iterations: full_result.iterations,
            converged: full_result.converged,
        })
    }

    /// Fit a single target variable
    #[allow(clippy::result_large_err)]
    fn fit_single_target(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64, usize, bool), LinearModelError> {
        let n_features = x.ncols();
        let mut coef = Array1::<f64>::zeros(n_features);
        let mut converged = false;
        let mut iteration = 0;
        let mut prev_loss = f64::INFINITY;

        // Use coordinate descent
        while iteration < self.config.max_iter && !converged {
            for j in 0..n_features {
                // Compute residual without current coefficient
                let mut residual = y.clone();
                for i in 0..n_features {
                    if i != j {
                        let xi = x.column(i).to_owned();
                        let scale = coef[i];
                        residual.scaled_add(-scale, &xi);
                    }
                }

                // Compute correlation with feature j
                let x_j = x.column(j);
                let correlation = x_j.dot(&residual);
                let norm_sq = x_j.dot(&x_j);

                // Apply soft thresholding for Lasso component
                let lasso_penalty = self.config.alpha * self.config.l1_ratio;
                let ridge_penalty = self.config.alpha * (1.0 - self.config.l1_ratio);

                coef[j] = if norm_sq > 0.0 {
                    let threshold = lasso_penalty;
                    let denominator = norm_sq + ridge_penalty;

                    if correlation > threshold {
                        (correlation - threshold) / denominator
                    } else if correlation < -threshold {
                        (correlation + threshold) / denominator
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }

            // Compute loss: 0.5 * ||y - X*coef||^2
            let predictions = x.dot(&coef);
            let residuals = y - &predictions;
            let current_loss_base: f64 = residuals.iter().map(|v| v * v).sum::<f64>() * 0.5;

            // Add regularization terms
            let l1_penalty = self.config.alpha
                * self.config.l1_ratio
                * coef.iter().map(|x| x.abs()).sum::<f64>();
            let l2_penalty = 0.5
                * self.config.alpha
                * (1.0 - self.config.l1_ratio)
                * coef.iter().map(|v| v * v).sum::<f64>();
            let current_loss = current_loss_base + l1_penalty + l2_penalty;

            // Check convergence
            let loss_change = (prev_loss - current_loss).abs() / prev_loss.max(1.0);
            converged = loss_change < self.config.tolerance;
            prev_loss = current_loss;
            iteration += 1;
        }

        Ok((coef, prev_loss, iteration, converged))
    }

    /// Compute target correlations
    #[allow(clippy::result_large_err)]
    fn compute_target_correlations(
        &self,
        y: &Array2<f64>,
    ) -> Result<Array2<f64>, LinearModelError> {
        let n_targets = y.ncols();
        let mut correlations = Array2::<f64>::zeros((n_targets, n_targets));

        for i in 0..n_targets {
            for j in 0..n_targets {
                if i == j {
                    correlations[[i, j]] = 1.0;
                } else {
                    let yi = y.column(i);
                    let yj = y.column(j);

                    let n = yi.len() as f64;
                    let mean_i = yi.iter().sum::<f64>() / n;
                    let mean_j = yj.iter().sum::<f64>() / n;

                    let numerator: f64 = yi
                        .iter()
                        .zip(yj.iter())
                        .map(|(a, b)| (a - mean_i) * (b - mean_j))
                        .sum();

                    let denom_i: f64 = yi.iter().map(|x| (x - mean_i).powi(2)).sum::<f64>().sqrt();
                    let denom_j: f64 = yj.iter().map(|x| (x - mean_j).powi(2)).sum::<f64>().sqrt();

                    let correlation = if denom_i > 0.0 && denom_j > 0.0 {
                        numerator / (denom_i * denom_j)
                    } else {
                        0.0
                    };

                    correlations[[i, j]] = correlation;
                }
            }
        }

        Ok(correlations)
    }

    /// Auto-determine chain order based on target correlations
    #[allow(clippy::result_large_err)]
    fn auto_determine_chain_order(&self, y: &Array2<f64>) -> Result<Vec<usize>, LinearModelError> {
        let correlations = self.compute_target_correlations(y)?;
        let n_targets = y.ncols();

        // Use a greedy approach to order targets by correlation strength
        let mut order = Vec::new();
        let mut used = vec![false; n_targets];

        // Start with the target that has the highest average correlation with others
        let mut best_start = 0;
        let mut best_avg_corr = -1.0;

        for i in 0..n_targets {
            let avg_corr: f64 = correlations
                .row(i)
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, corr)| corr.abs())
                .sum::<f64>()
                / (n_targets - 1) as f64;

            if avg_corr > best_avg_corr {
                best_avg_corr = avg_corr;
                best_start = i;
            }
        }

        order.push(best_start);
        used[best_start] = true;

        // Greedily add remaining targets based on correlation with already selected
        while order.len() < n_targets {
            let mut best_next = 0;
            let mut best_corr = -1.0;

            for i in 0..n_targets {
                if used[i] {
                    continue;
                }

                // Find maximum correlation with already selected targets
                let max_corr = order
                    .iter()
                    .map(|&j| correlations[[i, j]].abs())
                    .fold(0.0, f64::max);

                if max_corr > best_corr {
                    best_corr = max_corr;
                    best_next = i;
                }
            }

            order.push(best_next);
            used[best_next] = true;
        }

        Ok(order)
    }
}

/// Builder for multi-output regression models
pub struct MultiOutputRegressionBuilder {
    config: MultiOutputConfig,
}

impl MultiOutputRegressionBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MultiOutputConfig::default(),
        }
    }

    /// Set regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set L1 ratio for elastic net
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.config.l1_ratio = l1_ratio;
        self
    }

    /// Set multi-output strategy
    pub fn strategy(mut self, strategy: MultiOutputStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Set rank for reduced rank modeling
    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = Some(rank);
        self
    }

    /// Enable target correlation modeling
    pub fn model_correlations(mut self, model_correlations: bool) -> Self {
        self.config.model_correlations = model_correlations;
        self
    }

    /// Set chain order for chain strategy
    pub fn chain_order(mut self, chain_order: Vec<usize>) -> Self {
        self.config.chain_order = Some(chain_order);
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Build the model
    pub fn build(self) -> MultiOutputRegression {
        MultiOutputRegression::new(self.config)
    }
}

impl Default for MultiOutputRegressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: These tests use nalgebra while the rest of sklears uses ndarray
// The tests need to be rewritten to use ndarray or the whole module needs migration
#[cfg(all(test, feature = "nalgebra-tests"))]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn create_test_data() -> (Array2<f64>, Array2<f64>) {
        let X = DMatrix::from_row_slice(
            5,
            3,
            &[
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0,
            ],
        );

        let Y = DMatrix::from_row_slice(5, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);

        (X, Y)
    }

    #[test]
    fn test_multi_output_ridge() {
        let (X, Y) = create_test_data();
        let mut model = MultiOutputRegression::ridge(0.1);

        let result = model.fit(&X, &Y);
        assert!(result.is_ok());

        let predictions = model.predict(&X);
        assert!(predictions.is_ok());

        let pred = predictions.expect("operation should succeed");
        assert_eq!(pred.nrows(), X.nrows());
        assert_eq!(pred.ncols(), Y.ncols());
    }

    #[test]
    fn test_multi_output_lasso() {
        let (X, Y) = create_test_data();
        let mut model = MultiOutputRegression::lasso(0.1);

        let result = model.fit(&X, &Y);
        assert!(result.is_ok());

        let predictions = model.predict(&X);
        assert!(predictions.is_ok());
    }

    #[test]
    fn test_multi_output_elastic_net() {
        let (X, Y) = create_test_data();
        let mut model = MultiOutputRegression::elastic_net(0.1, 0.5);

        let result = model.fit(&X, &Y);
        assert!(result.is_ok());

        let coefficients = model.coefficients();
        assert!(coefficients.is_some());
    }

    #[test]
    fn test_independent_strategy() {
        let (X, Y) = create_test_data();
        let config = MultiOutputConfig {
            strategy: MultiOutputStrategy::Independent,
            alpha: 0.1,
            ..Default::default()
        };
        let mut model = MultiOutputRegression::new(config);

        let result = model.fit(&X, &Y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_chain_strategy() {
        let (X, Y) = create_test_data();
        let config = MultiOutputConfig {
            strategy: MultiOutputStrategy::Chain,
            alpha: 0.1,
            chain_order: Some(vec![0, 1]),
            ..Default::default()
        };
        let mut model = MultiOutputRegression::new(config);

        let result = model.fit(&X, &Y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduced_rank_strategy() {
        let (X, Y) = create_test_data();
        let config = MultiOutputConfig {
            strategy: MultiOutputStrategy::ReducedRank,
            alpha: 0.1,
            rank: Some(1),
            ..Default::default()
        };
        let mut model = MultiOutputRegression::new(config);

        let result = model.fit(&X, &Y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_target_correlations() {
        let (X, Y) = create_test_data();
        let config = MultiOutputConfig {
            model_correlations: true,
            alpha: 0.1,
            ..Default::default()
        };
        let mut model = MultiOutputRegression::new(config);

        model.fit(&X, &Y).expect("model fitting should succeed");
        let correlations = model.target_correlations();
        assert!(correlations.is_some());

        let corr_matrix = correlations.expect("operation should succeed");
        assert_eq!(corr_matrix.nrows(), Y.ncols());
        assert_eq!(corr_matrix.ncols(), Y.ncols());

        // Diagonal should be 1.0
        for i in 0..corr_matrix.nrows() {
            assert!((corr_matrix[(i, i)] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_builder_pattern() {
        let model = MultiOutputRegressionBuilder::new()
            .alpha(0.1)
            .l1_ratio(0.5)
            .strategy(MultiOutputStrategy::Joint)
            .max_iter(500)
            .tolerance(1e-5)
            .fit_intercept(true)
            .model_correlations(true)
            .build();

        assert_eq!(model.config.alpha, 0.1);
        assert_eq!(model.config.l1_ratio, 0.5);
        assert_eq!(model.config.strategy, MultiOutputStrategy::Joint);
        assert_eq!(model.config.max_iter, 500);
        assert!((model.config.tolerance - 1e-5).abs() < 1e-10);
        assert!(model.config.fit_intercept);
        assert!(model.config.model_correlations);
    }

    #[test]
    fn test_prediction_dimensions() {
        let (X, Y) = create_test_data();
        let mut model = MultiOutputRegression::ridge(0.1);

        model.fit(&X, &Y).expect("model fitting should succeed");

        // Test with same dimensions
        let predictions = model.predict(&X).expect("prediction should succeed");
        assert_eq!(predictions.nrows(), X.nrows());
        assert_eq!(predictions.ncols(), Y.ncols());

        // Test with different number of samples
        let X_new = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
        let predictions_new = model.predict(&X_new).expect("prediction should succeed");
        assert_eq!(predictions_new.nrows(), 3);
        assert_eq!(predictions_new.ncols(), Y.ncols());
    }

    #[test]
    fn test_invalid_dimensions() {
        let (X, Y) = create_test_data();
        let mut model = MultiOutputRegression::ridge(0.1);

        // Different number of samples
        let Y_bad = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);

        let result = model.fit(&X, &Y_bad);
        assert!(result.is_err());
    }
}
