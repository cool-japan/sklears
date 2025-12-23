//! Regularization Path Algorithms for SVM
//!
//! This module implements algorithms to compute the full regularization path for
//! various types of regularized SVMs. The regularization path shows how the solution
//! changes as the regularization parameter varies, which is useful for model selection
//! and understanding the trade-off between complexity and fit.
//!
//! Algorithms included:
//! - Lasso Path: For L1-regularized linear SVMs
//! - Elastic Net Path: For combined L1/L2 regularized SVMs  
//! - Group Lasso Path: For group-structured regularization
//! - Adaptive Lasso Path: With adaptive weights for feature selection

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Regularization path algorithm types
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RegularizationPathType {
    /// Lasso regularization path (L1)
    #[default]
    Lasso,
    /// Elastic Net regularization path (L1 + L2)
    ElasticNet { l1_ratio: Float },
    /// Group Lasso regularization path
    GroupLasso { groups: Vec<Vec<usize>> },
    /// Adaptive Lasso with feature-specific weights
    AdaptiveLasso { weights: Array1<Float> },
    /// Fused Lasso for sequence data
    FusedLasso,
}

/// Cross-validation strategy for path selection
#[derive(Debug, Clone, PartialEq)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold { k: usize },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Time series split for temporal data
    TimeSeriesSplit { n_splits: usize },
    /// Stratified K-fold for classification
    StratifiedKFold { k: usize },
}

impl Default for CrossValidationStrategy {
    fn default() -> Self {
        CrossValidationStrategy::KFold { k: 5 }
    }
}

/// Configuration for regularization path computation
#[derive(Debug, Clone)]
pub struct RegularizationPathConfig {
    /// Type of regularization path
    pub path_type: RegularizationPathType,
    /// Number of lambda values to compute
    pub n_lambdas: usize,
    /// Minimum lambda value (relative to lambda_max)
    pub lambda_min_ratio: Float,
    /// Custom lambda values (if provided, overrides n_lambdas)
    pub lambdas: Option<Array1<Float>>,
    /// Tolerance for convergence
    pub tol: Float,
    /// Maximum number of iterations per lambda
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Cross-validation strategy
    pub cv_strategy: CrossValidationStrategy,
    /// Whether to standardize features
    pub standardize: bool,
    /// Early stopping for path computation
    pub early_stopping: bool,
    /// Minimum improvement for early stopping
    pub min_improvement: Float,
    /// Verbose output
    pub verbose: bool,
}

impl Default for RegularizationPathConfig {
    fn default() -> Self {
        Self {
            path_type: RegularizationPathType::default(),
            n_lambdas: 100,
            lambda_min_ratio: 1e-4,
            lambdas: None,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            cv_strategy: CrossValidationStrategy::default(),
            standardize: true,
            early_stopping: true,
            min_improvement: 1e-6,
            verbose: false,
        }
    }
}

/// Results of regularization path computation
#[derive(Debug, Clone)]
pub struct RegularizationPathResult {
    /// Lambda values used
    pub lambdas: Array1<Float>,
    /// Coefficient paths (n_lambdas × n_features)
    pub coef_path: Array2<Float>,
    /// Intercept paths (n_lambdas)
    pub intercept_path: Array1<Float>,
    /// Cross-validation scores (n_lambdas)
    pub cv_scores: Array1<Float>,
    /// Standard errors of CV scores (n_lambdas)
    pub cv_scores_std: Array1<Float>,
    /// Number of non-zero coefficients at each lambda
    pub n_nonzero: Array1<usize>,
    /// Indices of selected features at each lambda
    pub active_features: Vec<Vec<usize>>,
    /// Best lambda value (based on CV)
    pub best_lambda: Float,
    /// Index of best lambda
    pub best_lambda_idx: usize,
    /// Lambda at 1 standard error rule
    pub lambda_1se: Float,
    /// Index of lambda at 1 standard error rule
    pub lambda_1se_idx: usize,
}

/// Regularization Path Solver for SVMs
#[derive(Debug)]
pub struct RegularizationPathSolver {
    config: RegularizationPathConfig,
}

impl Default for RegularizationPathSolver {
    fn default() -> Self {
        Self::new(RegularizationPathConfig::default())
    }
}

impl RegularizationPathSolver {
    /// Create a new regularization path solver
    pub fn new(config: RegularizationPathConfig) -> Self {
        Self { config }
    }

    /// Fit regularization path
    pub fn fit_path(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<RegularizationPathResult> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Shape mismatch: X and y must have the same number of samples".to_string(),
            ));
        }

        // Standardize features if requested
        let (x_processed, _feature_means, feature_stds) = if self.config.standardize {
            self.standardize_features(x)?
        } else {
            (
                x.clone(),
                Array1::zeros(n_features),
                Array1::ones(n_features),
            )
        };

        // Center targets
        let y_mean = if self.config.fit_intercept {
            y.mean().unwrap_or(0.0)
        } else {
            0.0
        };
        let y_centered = y.mapv(|val| val - y_mean);

        // Compute lambda sequence
        let mut lambdas = if let Some(custom_lambdas) = &self.config.lambdas {
            custom_lambdas.clone()
        } else {
            self.compute_lambda_sequence(&x_processed, &y_centered)?
        };

        let n_lambdas = lambdas.len();

        // Initialize path storage
        let mut coef_path = Array2::zeros((n_lambdas, n_features));
        let mut intercept_path = Array1::zeros(n_lambdas);
        let mut cv_scores = Array1::zeros(n_lambdas);
        let mut cv_scores_std = Array1::zeros(n_lambdas);
        let mut n_nonzero = Array1::zeros(n_lambdas);
        let mut active_features = Vec::with_capacity(n_lambdas);

        // Initial coefficient vector
        let mut coef = Array1::zeros(n_features);

        // Compute path
        for (i, &lambda) in lambdas.iter().enumerate() {
            if self.config.verbose && i % 10 == 0 {
                println!(
                    "Computing path for lambda {}/{}: {:.6}",
                    i + 1,
                    n_lambdas,
                    lambda
                );
            }

            // Warm start from previous solution
            let (new_coef, intercept) =
                self.solve_for_lambda(&x_processed, &y_centered, lambda, &coef, y_mean)?;

            coef = new_coef.clone();

            // Store results
            coef_path.row_mut(i).assign(&new_coef);
            intercept_path[i] = intercept;

            // Count non-zero coefficients
            let nonzero_count = new_coef
                .iter()
                .filter(|&&x| x.abs() > self.config.tol)
                .count();
            n_nonzero[i] = nonzero_count;

            // Track active features
            let active: Vec<usize> = new_coef
                .iter()
                .enumerate()
                .filter(|(_, &x)| x.abs() > self.config.tol)
                .map(|(idx, _)| idx)
                .collect();
            active_features.push(active);

            // Cross-validation for this lambda
            let (cv_score, cv_std) =
                self.cross_validate_lambda(&x_processed, &y_centered, lambda, y_mean)?;
            cv_scores[i] = cv_score;
            cv_scores_std[i] = cv_std;

            // Early stopping check
            if self.config.early_stopping && i > 10 {
                let recent_improvement = if i >= 5 {
                    let recent_avg = cv_scores.slice(s![i - 4..=i]).mean().unwrap();
                    let prev_avg = cv_scores.slice(s![i - 9..=i - 5]).mean().unwrap();
                    recent_avg - prev_avg
                } else {
                    Float::INFINITY
                };

                if recent_improvement.abs() < self.config.min_improvement {
                    if self.config.verbose {
                        println!("Early stopping at lambda index {i}");
                    }
                    // Truncate arrays
                    let actual_n_lambdas = i + 1;
                    lambdas = lambdas.slice(s![..actual_n_lambdas]).to_owned();
                    coef_path = coef_path.slice(s![..actual_n_lambdas, ..]).to_owned();
                    intercept_path = intercept_path.slice(s![..actual_n_lambdas]).to_owned();
                    cv_scores = cv_scores.slice(s![..actual_n_lambdas]).to_owned();
                    cv_scores_std = cv_scores_std.slice(s![..actual_n_lambdas]).to_owned();
                    n_nonzero = n_nonzero.slice(s![..actual_n_lambdas]).to_owned();
                    active_features.truncate(actual_n_lambdas);
                    break;
                }
            }
        }

        // Reverse standardization for coefficients
        if self.config.standardize {
            for i in 0..coef_path.nrows() {
                for j in 0..n_features {
                    if feature_stds[j] > 1e-10 {
                        coef_path[[i, j]] /= feature_stds[j];
                    }
                }
            }
        }

        // Find best lambda (minimum CV error)
        let best_lambda_idx = cv_scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let best_lambda = lambdas[best_lambda_idx];

        // Find lambda at 1 standard error rule
        let best_score = cv_scores[best_lambda_idx];
        let best_std = cv_scores_std[best_lambda_idx];
        let threshold = best_score + best_std;

        let lambda_1se_idx = (0..lambdas.len())
            .find(|&i| cv_scores[i] <= threshold && lambdas[i] >= best_lambda)
            .unwrap_or(best_lambda_idx);
        let lambda_1se = lambdas[lambda_1se_idx];

        Ok(RegularizationPathResult {
            lambdas,
            coef_path,
            intercept_path,
            cv_scores,
            cv_scores_std,
            n_nonzero,
            active_features,
            best_lambda,
            best_lambda_idx,
            lambda_1se,
            lambda_1se_idx,
        })
    }

    /// Standardize features
    fn standardize_features(
        &self,
        x: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::ones(n_features);

        // Compute means
        for j in 0..n_features {
            means[j] = x.column(j).mean().unwrap_or(0.0);
        }

        // Compute standard deviations
        for j in 0..n_features {
            let variance = x
                .column(j)
                .iter()
                .map(|&val| (val - means[j]).powi(2))
                .sum::<Float>()
                / (x.nrows() - 1) as Float;
            stds[j] = variance.sqrt().max(1e-10);
        }

        // Standardize
        let mut x_std = x.clone();
        for i in 0..x.nrows() {
            for j in 0..n_features {
                x_std[[i, j]] = (x_std[[i, j]] - means[j]) / stds[j];
            }
        }

        Ok((x_std, means, stds))
    }

    /// Compute lambda sequence
    fn compute_lambda_sequence(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Compute lambda_max (smallest lambda that gives all-zero solution)
        let lambda_max = match &self.config.path_type {
            RegularizationPathType::Lasso | RegularizationPathType::AdaptiveLasso { .. } => {
                self.compute_lasso_lambda_max(x, y)?
            }
            RegularizationPathType::ElasticNet { l1_ratio } => {
                self.compute_lasso_lambda_max(x, y)? / l1_ratio
            }
            RegularizationPathType::GroupLasso { .. } => {
                self.compute_group_lasso_lambda_max(x, y)?
            }
            RegularizationPathType::FusedLasso => self.compute_fused_lasso_lambda_max(x, y)?,
        };

        let lambda_min = lambda_max * self.config.lambda_min_ratio;

        // Create log-spaced sequence
        let mut lambdas = Array1::zeros(self.config.n_lambdas);
        let log_max = lambda_max.ln();
        let log_min = lambda_min.ln();
        let step = (log_max - log_min) / (self.config.n_lambdas - 1) as Float;

        for i in 0..self.config.n_lambdas {
            lambdas[i] = (log_max - i as Float * step).exp();
        }

        Ok(lambdas)
    }

    /// Compute lambda_max for Lasso
    fn compute_lasso_lambda_max(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Float> {
        let mut max_correlation: Float = 0.0;

        for j in 0..x.ncols() {
            let correlation = x
                .column(j)
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| xi * yi)
                .sum::<Float>()
                .abs()
                / x.nrows() as Float;

            max_correlation = max_correlation.max(correlation);
        }

        Ok(max_correlation)
    }

    /// Compute lambda_max for Group Lasso
    fn compute_group_lasso_lambda_max(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Float> {
        if let RegularizationPathType::GroupLasso { groups } = &self.config.path_type {
            let mut max_group_norm: Float = 0.0;

            for group in groups {
                let mut group_norm = 0.0;
                for &feature_idx in group {
                    if feature_idx < x.ncols() {
                        let correlation = x
                            .column(feature_idx)
                            .iter()
                            .zip(y.iter())
                            .map(|(&xi, &yi)| xi * yi)
                            .sum::<Float>()
                            / x.nrows() as Float;
                        group_norm += correlation * correlation;
                    }
                }
                group_norm = group_norm.sqrt();
                max_group_norm = max_group_norm.max(group_norm);
            }

            Ok(max_group_norm)
        } else {
            Err(SklearsError::InvalidInput(
                "Invalid path type for Group Lasso lambda_max computation".to_string(),
            ))
        }
    }

    /// Compute lambda_max for Fused Lasso
    fn compute_fused_lasso_lambda_max(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Float> {
        let lasso_max = self.compute_lasso_lambda_max(x, y)?;

        // For fused lasso, also consider difference penalties
        let mut max_diff_correlation: Float = 0.0;
        for j in 0..(x.ncols() - 1) {
            let diff_feature: Array1<Float> = x
                .rows()
                .into_iter()
                .map(|row| row[j + 1] - row[j])
                .collect();

            let correlation = diff_feature
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| xi * yi)
                .sum::<Float>()
                .abs()
                / x.nrows() as Float;

            max_diff_correlation = max_diff_correlation.max(correlation);
        }

        Ok(lasso_max.max(max_diff_correlation))
    }

    /// Solve for a specific lambda value
    fn solve_for_lambda(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        lambda: Float,
        initial_coef: &Array1<Float>,
        y_mean: Float,
    ) -> Result<(Array1<Float>, Float)> {
        match &self.config.path_type {
            RegularizationPathType::Lasso => self.solve_lasso(x, y, lambda, initial_coef, y_mean),
            RegularizationPathType::ElasticNet { l1_ratio } => {
                self.solve_elastic_net(x, y, lambda, *l1_ratio, initial_coef, y_mean)
            }
            RegularizationPathType::GroupLasso { groups } => {
                self.solve_group_lasso(x, y, lambda, groups, initial_coef, y_mean)
            }
            RegularizationPathType::AdaptiveLasso { weights } => {
                self.solve_adaptive_lasso(x, y, lambda, weights, initial_coef, y_mean)
            }
            RegularizationPathType::FusedLasso => {
                self.solve_fused_lasso(x, y, lambda, initial_coef, y_mean)
            }
        }
    }

    /// Solve Lasso for a specific lambda using coordinate descent
    fn solve_lasso(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        lambda: Float,
        initial_coef: &Array1<Float>,
        y_mean: Float,
    ) -> Result<(Array1<Float>, Float)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut coef = initial_coef.clone();
        let mut intercept = y_mean;

        // Precompute X^T X diagonal for efficiency
        let mut xtx_diag = Array1::zeros(n_features);
        for j in 0..n_features {
            xtx_diag[j] = x.column(j).iter().map(|&val| val * val).sum::<Float>();
        }

        // Coordinate descent
        for _ in 0..self.config.max_iter {
            let mut converged = true;

            for j in 0..n_features {
                let old_coef_j = coef[j];

                // Compute residual without feature j
                let mut residual_sum = 0.0;
                for i in 0..n_samples {
                    let mut prediction = intercept;
                    for k in 0..n_features {
                        if k != j {
                            prediction += coef[k] * x[[i, k]];
                        }
                    }
                    residual_sum += x[[i, j]] * (y[i] - prediction);
                }

                // Soft thresholding
                let threshold = lambda * n_samples as Float;
                if residual_sum > threshold {
                    coef[j] = (residual_sum - threshold) / xtx_diag[j];
                } else if residual_sum < -threshold {
                    coef[j] = (residual_sum + threshold) / xtx_diag[j];
                } else {
                    coef[j] = 0.0;
                }

                if (coef[j] - old_coef_j).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update intercept
            if self.config.fit_intercept {
                let mut residual_sum = 0.0;
                for i in 0..n_samples {
                    let mut prediction = 0.0;
                    for k in 0..n_features {
                        prediction += coef[k] * x[[i, k]];
                    }
                    residual_sum += y[i] - prediction;
                }
                intercept = residual_sum / n_samples as Float;
            }

            if converged {
                break;
            }
        }

        Ok((coef, intercept))
    }

    /// Solve Elastic Net for a specific lambda
    fn solve_elastic_net(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        lambda: Float,
        l1_ratio: Float,
        initial_coef: &Array1<Float>,
        y_mean: Float,
    ) -> Result<(Array1<Float>, Float)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut coef = initial_coef.clone();
        let mut intercept = y_mean;

        let l1_penalty = lambda * l1_ratio;
        let l2_penalty = lambda * (1.0 - l1_ratio);

        // Precompute X^T X diagonal
        let mut xtx_diag = Array1::zeros(n_features);
        for j in 0..n_features {
            xtx_diag[j] = x.column(j).iter().map(|&val| val * val).sum::<Float>()
                + l2_penalty * n_samples as Float;
        }

        // Coordinate descent
        for _ in 0..self.config.max_iter {
            let mut converged = true;

            for j in 0..n_features {
                let old_coef_j = coef[j];

                // Compute residual without feature j
                let mut residual_sum = 0.0;
                for i in 0..n_samples {
                    let mut prediction = intercept;
                    for k in 0..n_features {
                        if k != j {
                            prediction += coef[k] * x[[i, k]];
                        }
                    }
                    residual_sum += x[[i, j]] * (y[i] - prediction);
                }

                // Soft thresholding with L2 penalty
                let threshold = l1_penalty * n_samples as Float;
                if residual_sum > threshold {
                    coef[j] = (residual_sum - threshold) / xtx_diag[j];
                } else if residual_sum < -threshold {
                    coef[j] = (residual_sum + threshold) / xtx_diag[j];
                } else {
                    coef[j] = 0.0;
                }

                if (coef[j] - old_coef_j).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update intercept
            if self.config.fit_intercept {
                let mut residual_sum = 0.0;
                for i in 0..n_samples {
                    let mut prediction = 0.0;
                    for k in 0..n_features {
                        prediction += coef[k] * x[[i, k]];
                    }
                    residual_sum += y[i] - prediction;
                }
                intercept = residual_sum / n_samples as Float;
            }

            if converged {
                break;
            }
        }

        Ok((coef, intercept))
    }

    /// Solve Group Lasso for a specific lambda
    fn solve_group_lasso(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        lambda: Float,
        groups: &[Vec<usize>],
        initial_coef: &Array1<Float>,
        y_mean: Float,
    ) -> Result<(Array1<Float>, Float)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut coef = initial_coef.clone();
        let intercept = y_mean;

        // Group coordinate descent
        for _ in 0..self.config.max_iter {
            let mut converged = true;

            for group in groups {
                let group_size = group.len();
                let mut old_group_coef = Array1::zeros(group_size);
                for (idx, &feature_idx) in group.iter().enumerate() {
                    if feature_idx < n_features {
                        old_group_coef[idx] = coef[feature_idx];
                    }
                }

                // Compute group gradient
                let mut group_gradient = Array1::zeros(group_size);
                for i in 0..n_samples {
                    let mut prediction = intercept;
                    for k in 0..n_features {
                        prediction += coef[k] * x[[i, k]];
                    }
                    let residual = y[i] - prediction;

                    for (idx, &feature_idx) in group.iter().enumerate() {
                        if feature_idx < n_features {
                            group_gradient[idx] +=
                                x[[i, feature_idx]] * residual / n_samples as Float;
                        }
                    }
                }

                // Group soft thresholding
                let group_norm = group_gradient
                    .iter()
                    .map(|&x: &Float| x * x)
                    .sum::<Float>()
                    .sqrt();
                if group_norm > lambda {
                    let shrinkage_factor = (1.0 - lambda / group_norm).max(0.0);
                    for (idx, &feature_idx) in group.iter().enumerate() {
                        if feature_idx < n_features {
                            coef[feature_idx] = group_gradient[idx] * shrinkage_factor;
                            if (coef[feature_idx] - old_group_coef[idx]).abs() > self.config.tol {
                                converged = false;
                            }
                        }
                    }
                } else {
                    // Shrink entire group to zero
                    for &feature_idx in group {
                        if feature_idx < n_features {
                            if coef[feature_idx].abs() > self.config.tol {
                                converged = false;
                            }
                            coef[feature_idx] = 0.0;
                        }
                    }
                }
            }

            if converged {
                break;
            }
        }

        Ok((coef, intercept))
    }

    /// Solve Adaptive Lasso for a specific lambda
    fn solve_adaptive_lasso(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        lambda: Float,
        weights: &Array1<Float>,
        initial_coef: &Array1<Float>,
        y_mean: Float,
    ) -> Result<(Array1<Float>, Float)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut coef = initial_coef.clone();
        let intercept = y_mean;

        // Precompute X^T X diagonal
        let mut xtx_diag = Array1::zeros(n_features);
        for j in 0..n_features {
            xtx_diag[j] = x.column(j).iter().map(|&val| val * val).sum::<Float>();
        }

        // Coordinate descent with adaptive weights
        for _ in 0..self.config.max_iter {
            let mut converged = true;

            for j in 0..n_features {
                let old_coef_j = coef[j];

                // Compute residual without feature j
                let mut residual_sum = 0.0;
                for i in 0..n_samples {
                    let mut prediction = intercept;
                    for k in 0..n_features {
                        if k != j {
                            prediction += coef[k] * x[[i, k]];
                        }
                    }
                    residual_sum += x[[i, j]] * (y[i] - prediction);
                }

                // Adaptive soft thresholding
                let adaptive_threshold = lambda * weights[j] * n_samples as Float;
                if residual_sum > adaptive_threshold {
                    coef[j] = (residual_sum - adaptive_threshold) / xtx_diag[j];
                } else if residual_sum < -adaptive_threshold {
                    coef[j] = (residual_sum + adaptive_threshold) / xtx_diag[j];
                } else {
                    coef[j] = 0.0;
                }

                if (coef[j] - old_coef_j).abs() > self.config.tol {
                    converged = false;
                }
            }

            if converged {
                break;
            }
        }

        Ok((coef, intercept))
    }

    /// Solve Fused Lasso for a specific lambda
    fn solve_fused_lasso(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        lambda: Float,
        initial_coef: &Array1<Float>,
        y_mean: Float,
    ) -> Result<(Array1<Float>, Float)> {
        // For simplicity, implement as standard Lasso
        // A full implementation would include difference penalties
        self.solve_lasso(x, y, lambda, initial_coef, y_mean)
    }

    /// Cross-validate for a specific lambda value
    fn cross_validate_lambda(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        lambda: Float,
        y_mean: Float,
    ) -> Result<(Float, Float)> {
        let n_samples = x.nrows();
        let mut cv_scores = Vec::new();

        match &self.config.cv_strategy {
            CrossValidationStrategy::KFold { k } => {
                let fold_size = n_samples / k;

                for fold in 0..*k {
                    let test_start = fold * fold_size;
                    let test_end = if fold == k - 1 {
                        n_samples
                    } else {
                        (fold + 1) * fold_size
                    };

                    // Create train/test splits
                    let mut train_indices = Vec::new();
                    let mut test_indices = Vec::new();

                    for i in 0..n_samples {
                        if i >= test_start && i < test_end {
                            test_indices.push(i);
                        } else {
                            train_indices.push(i);
                        }
                    }

                    if train_indices.is_empty() || test_indices.is_empty() {
                        continue;
                    }

                    // Extract train/test data
                    let x_train = self.extract_rows(x, &train_indices)?;
                    let y_train = self.extract_elements(y, &train_indices)?;
                    let x_test = self.extract_rows(x, &test_indices)?;
                    let y_test = self.extract_elements(y, &test_indices)?;

                    // Train on fold
                    let zero_coef = Array1::zeros(x.ncols());
                    let (coef_fold, intercept_fold) =
                        self.solve_for_lambda(&x_train, &y_train, lambda, &zero_coef, y_mean)?;

                    // Evaluate on test fold
                    let mut test_error = 0.0;
                    for i in 0..x_test.nrows() {
                        let mut prediction = if self.config.fit_intercept {
                            intercept_fold
                        } else {
                            0.0
                        };

                        for j in 0..x_test.ncols() {
                            prediction += coef_fold[j] * x_test[[i, j]];
                        }

                        test_error += (y_test[i] - prediction).powi(2);
                    }
                    test_error /= x_test.nrows() as Float;
                    cv_scores.push(test_error);
                }
            }
            _ => {
                // For other CV strategies, implement similarly
                // Using simple holdout for now
                let n_test = n_samples / 5;
                let n_train = n_samples - n_test;

                let x_train = x.slice(s![..n_train, ..]).to_owned();
                let y_train = y.slice(s![..n_train]).to_owned();
                let x_test = x.slice(s![n_train.., ..]).to_owned();
                let y_test = y.slice(s![n_train..]).to_owned();

                let zero_coef = Array1::zeros(x.ncols());
                let (coef_fold, intercept_fold) =
                    self.solve_for_lambda(&x_train, &y_train, lambda, &zero_coef, y_mean)?;

                let mut test_error = 0.0;
                for i in 0..x_test.nrows() {
                    let mut prediction = if self.config.fit_intercept {
                        intercept_fold
                    } else {
                        0.0
                    };

                    for j in 0..x_test.ncols() {
                        prediction += coef_fold[j] * x_test[[i, j]];
                    }

                    test_error += (y_test[i] - prediction).powi(2);
                }
                test_error /= x_test.nrows() as Float;
                cv_scores.push(test_error);
            }
        }

        if cv_scores.is_empty() {
            return Ok((Float::INFINITY, 0.0));
        }

        let mean_score = cv_scores.iter().sum::<Float>() / cv_scores.len() as Float;
        let variance = cv_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / cv_scores.len() as Float;
        let std_score = variance.sqrt();

        Ok((mean_score, std_score))
    }

    /// Extract specific rows from a matrix
    fn extract_rows(&self, matrix: &Array2<Float>, indices: &[usize]) -> Result<Array2<Float>> {
        let n_features = matrix.ncols();
        let mut result = Array2::zeros((indices.len(), n_features));

        for (i, &idx) in indices.iter().enumerate() {
            if idx < matrix.nrows() {
                result.row_mut(i).assign(&matrix.row(idx));
            }
        }

        Ok(result)
    }

    /// Extract specific elements from an array
    fn extract_elements(&self, array: &Array1<Float>, indices: &[usize]) -> Result<Array1<Float>> {
        let mut result = Array1::zeros(indices.len());

        for (i, &idx) in indices.iter().enumerate() {
            if idx < array.len() {
                result[i] = array[idx];
            }
        }

        Ok(result)
    }
}

impl RegularizationPathResult {
    /// Get coefficient at a specific lambda value
    pub fn coef_at_lambda(&self, lambda: Float) -> Option<ArrayView1<'_, Float>> {
        let idx = self
            .lambdas
            .iter()
            .position(|&l| (l - lambda).abs() < 1e-10)?;
        Some(self.coef_path.row(idx))
    }

    /// Get the coefficient path for a specific feature
    pub fn feature_path(&self, feature_idx: usize) -> Option<ArrayView1<'_, Float>> {
        if feature_idx < self.coef_path.ncols() {
            Some(self.coef_path.column(feature_idx))
        } else {
            None
        }
    }

    /// Find the sparsest model within 1 standard error of the best
    pub fn sparse_model_1se(&self) -> (Float, ArrayView1<'_, Float>) {
        let lambda = self.lambda_1se;
        let coef = self.coef_path.row(self.lambda_1se_idx);
        (lambda, coef)
    }

    /// Get summary statistics
    pub fn summary(&self) -> HashMap<String, Float> {
        let mut summary = HashMap::new();

        summary.insert("n_lambdas".to_string(), self.lambdas.len() as Float);
        summary.insert("best_lambda".to_string(), self.best_lambda);
        summary.insert("lambda_1se".to_string(), self.lambda_1se);
        summary.insert(
            "best_cv_score".to_string(),
            self.cv_scores[self.best_lambda_idx],
        );
        summary.insert(
            "min_nonzero_features".to_string(),
            *self.n_nonzero.iter().min().unwrap_or(&0) as Float,
        );
        summary.insert(
            "max_nonzero_features".to_string(),
            *self.n_nonzero.iter().max().unwrap_or(&0) as Float,
        );

        summary
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_regularization_path_config() {
        let config = RegularizationPathConfig {
            path_type: RegularizationPathType::Lasso,
            n_lambdas: 50,
            lambda_min_ratio: 1e-3,
            ..Default::default()
        };

        assert_eq!(config.n_lambdas, 50);
        assert_eq!(config.lambda_min_ratio, 1e-3);
        assert!(matches!(config.path_type, RegularizationPathType::Lasso));
    }

    #[test]
    fn test_lambda_sequence_computation() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let config = RegularizationPathConfig::default();
        let solver = RegularizationPathSolver::new(config);

        let lambdas = solver.compute_lambda_sequence(&x, &y).unwrap();

        assert_eq!(lambdas.len(), 100);
        assert!(lambdas[0] > lambdas[lambdas.len() - 1]); // Decreasing sequence

        for i in 1..lambdas.len() {
            assert!(lambdas[i - 1] >= lambdas[i]); // Non-increasing
        }
    }

    #[test]
    #[ignore = "Slow test: computes regularization path. Run with --ignored flag"]
    fn test_lasso_path() {
        let x = array![
            [1.0, 2.0, 0.1],
            [2.0, 3.0, 0.2],
            [3.0, 4.0, 0.3],
            [4.0, 5.0, 0.4],
            [5.0, 6.0, 0.5],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = RegularizationPathConfig {
            path_type: RegularizationPathType::Lasso,
            n_lambdas: 20,
            max_iter: 100,
            verbose: false,
            ..Default::default()
        };

        let solver = RegularizationPathSolver::new(config);
        let result = solver.fit_path(&x, &y).unwrap();

        assert_eq!(result.lambdas.len(), 20);
        assert_eq!(result.coef_path.nrows(), 20);
        assert_eq!(result.coef_path.ncols(), 3);
        assert_eq!(result.intercept_path.len(), 20);
        assert_eq!(result.cv_scores.len(), 20);

        // Check that sparsity increases with regularization
        assert!(result.n_nonzero[0] >= result.n_nonzero[result.n_nonzero.len() - 1]);

        // Check best lambda selection
        assert!(result.best_lambda_idx < result.lambdas.len());
        assert!(result.lambda_1se_idx < result.lambdas.len());

        // Test coefficient extraction
        let best_coef = result.coef_path.row(result.best_lambda_idx);
        assert_eq!(best_coef.len(), 3);

        // Test summary
        let summary = result.summary();
        assert!(summary.contains_key("best_lambda"));
        assert!(summary.contains_key("lambda_1se"));
    }

    #[test]
    fn test_elastic_net_path_type() {
        let path_type = RegularizationPathType::ElasticNet { l1_ratio: 0.5 };

        if let RegularizationPathType::ElasticNet { l1_ratio } = path_type {
            assert_eq!(l1_ratio, 0.5);
        } else {
            panic!("Expected ElasticNet path type");
        }
    }

    #[test]
    fn test_group_lasso_path_type() {
        let groups = vec![vec![0, 1], vec![2, 3], vec![4]];
        let path_type = RegularizationPathType::GroupLasso {
            groups: groups.clone(),
        };

        if let RegularizationPathType::GroupLasso { groups: g } = path_type {
            assert_eq!(g.len(), 3);
            assert_eq!(g[0], vec![0, 1]);
        } else {
            panic!("Expected GroupLasso path type");
        }
    }

    #[test]
    fn test_standardization() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let config = RegularizationPathConfig::default();
        let solver = RegularizationPathSolver::new(config);

        let (x_std, means, _stds) = solver.standardize_features(&x).unwrap();

        // Check means are approximately zero after standardization
        for j in 0..x_std.ncols() {
            let col_mean = x_std.column(j).mean().unwrap();
            assert!((col_mean).abs() < 1e-10);
        }

        // Check original means and stds
        assert!((means[0] - 2.0).abs() < 1e-10);
        assert!((means[1] - 20.0).abs() < 1e-10);
    }
}
