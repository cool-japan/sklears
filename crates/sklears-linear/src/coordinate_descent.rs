//! Coordinate Descent solver for Lasso and ElasticNet regression

#[cfg(feature = "early-stopping")]
use crate::early_stopping::{train_validation_split, EarlyStopping, EarlyStoppingConfig};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Soft thresholding operator for L1 regularization
#[inline]
fn soft_threshold(x: Float, lambda: Float) -> Float {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Coordinate Descent solver for Lasso and ElasticNet
pub struct CoordinateDescentSolver {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to use cyclic or random selection
    pub cyclic: bool,
    /// Early stopping configuration (optional)
    #[cfg(feature = "early-stopping")]
    pub early_stopping_config: Option<EarlyStoppingConfig>,
}

impl Default for CoordinateDescentSolver {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-4,
            cyclic: true,
            #[cfg(feature = "early-stopping")]
            early_stopping_config: None,
        }
    }
}

impl CoordinateDescentSolver {
    /// Solve Lasso regression using coordinate descent
    ///
    /// Minimizes: (1/2n) ||y - Xβ||² + α||β||₁
    pub fn solve_lasso(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
        fit_intercept: bool,
    ) -> Result<(Array1<Float>, Option<Float>)> {
        self.solve_lasso_with_warm_start(x, y, alpha, fit_intercept, None, None)
    }

    /// Solve Lasso regression using coordinate descent with warm start
    ///
    /// Minimizes: (1/2n) ||y - Xβ||² + α||β||₁
    pub fn solve_lasso_with_warm_start(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
        fit_intercept: bool,
        initial_coef: Option<&Array1<Float>>,
        initial_intercept: Option<Float>,
    ) -> Result<(Array1<Float>, Option<Float>)> {
        let n_samples = x.nrows() as Float;
        let n_features = x.ncols();

        // Initialize coefficients (with warm start if provided)
        let mut coef = match initial_coef {
            Some(init_coef) => {
                if init_coef.len() != n_features {
                    return Err(SklearsError::FeatureMismatch {
                        expected: n_features,
                        actual: init_coef.len(),
                    });
                }
                init_coef.clone()
            }
            None => Array1::zeros(n_features),
        };

        let mut intercept = if fit_intercept {
            initial_intercept.unwrap_or_else(|| y.mean().unwrap_or(0.0))
        } else {
            0.0
        };

        // Precompute norms for each feature
        let feature_norms: Array1<Float> = x
            .axis_iter(Axis(1))
            .map(|col| col.dot(&col) / n_samples)
            .collect();

        // Coordinate descent iterations
        let mut converged = false;
        for _iter in 0..self.max_iter {
            let old_coef = coef.clone();

            // Update intercept if needed
            if fit_intercept {
                let residuals = y - &x.dot(&coef) - intercept;
                intercept = residuals.mean().unwrap_or(0.0);
            }

            // Update each coordinate
            for j in 0..n_features {
                // Skip if feature norm is zero
                if feature_norms[j] == 0.0 {
                    coef[j] = 0.0;
                    continue;
                }

                // Compute partial residual (excluding j-th feature)
                let mut residuals = y - &x.dot(&coef);
                if fit_intercept {
                    residuals -= intercept;
                }
                residuals = residuals + x.column(j).to_owned() * coef[j];

                // Compute gradient for j-th feature
                let gradient = x.column(j).dot(&residuals) / n_samples;

                // Apply soft thresholding
                coef[j] = soft_threshold(gradient, alpha) / feature_norms[j];
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(Float::abs).sum();
            if coef_change < self.tol {
                converged = true;
                break;
            }
        }

        if !converged {
            eprintln!(
                "Warning: Coordinate descent did not converge. Consider increasing max_iter."
            );
        }

        let intercept_opt = if fit_intercept { Some(intercept) } else { None };
        Ok((coef, intercept_opt))
    }

    /// Solve ElasticNet regression using coordinate descent
    ///
    /// Minimizes: (1/2n) ||y - Xβ||² + α * ρ * ||β||₁ + α * (1-ρ)/2 * ||β||²
    /// where ρ is l1_ratio
    pub fn solve_elastic_net(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
        l1_ratio: Float,
        fit_intercept: bool,
    ) -> Result<(Array1<Float>, Option<Float>)> {
        self.solve_elastic_net_with_warm_start(x, y, alpha, l1_ratio, fit_intercept, None, None)
    }

    /// Solve ElasticNet regression using coordinate descent with warm start
    ///
    /// Minimizes: (1/2n) ||y - Xβ||² + α * ρ * ||β||₁ + α * (1-ρ)/2 * ||β||²
    /// where ρ is l1_ratio
    pub fn solve_elastic_net_with_warm_start(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
        l1_ratio: Float,
        fit_intercept: bool,
        initial_coef: Option<&Array1<Float>>,
        initial_intercept: Option<Float>,
    ) -> Result<(Array1<Float>, Option<Float>)> {
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(SklearsError::InvalidParameter {
                name: "l1_ratio".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }

        let n_samples = x.nrows() as Float;
        let n_features = x.ncols();

        // Regularization parameters
        let l1_reg = alpha * l1_ratio;
        let l2_reg = alpha * (1.0 - l1_ratio);

        // Initialize coefficients (with warm start if provided)
        let mut coef = match initial_coef {
            Some(init_coef) => {
                if init_coef.len() != n_features {
                    return Err(SklearsError::FeatureMismatch {
                        expected: n_features,
                        actual: init_coef.len(),
                    });
                }
                init_coef.clone()
            }
            None => Array1::zeros(n_features),
        };

        let mut intercept = if fit_intercept {
            initial_intercept.unwrap_or_else(|| y.mean().unwrap_or(0.0))
        } else {
            0.0
        };

        // Precompute norms for each feature (including L2 penalty)
        let feature_norms: Array1<Float> = x
            .axis_iter(Axis(1))
            .map(|col| col.dot(&col) / n_samples + l2_reg)
            .collect();

        // Coordinate descent iterations
        let mut converged = false;
        for _iter in 0..self.max_iter {
            let old_coef = coef.clone();

            // Update intercept if needed
            if fit_intercept {
                let residuals = y - &x.dot(&coef) - intercept;
                intercept = residuals.mean().unwrap_or(0.0);
            }

            // Update each coordinate
            for j in 0..n_features {
                // Skip if feature norm is zero
                if feature_norms[j] == 0.0 {
                    coef[j] = 0.0;
                    continue;
                }

                // Compute partial residual (excluding j-th feature)
                let mut residuals = y - &x.dot(&coef);
                if fit_intercept {
                    residuals -= intercept;
                }
                residuals = residuals + x.column(j).to_owned() * coef[j];

                // Compute gradient for j-th feature
                let gradient = x.column(j).dot(&residuals) / n_samples;

                // Apply soft thresholding for L1 and account for L2
                coef[j] = soft_threshold(gradient, l1_reg) / feature_norms[j];
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(Float::abs).sum();
            if coef_change < self.tol {
                converged = true;
                break;
            }
        }

        if !converged {
            eprintln!(
                "Warning: Coordinate descent did not converge. Consider increasing max_iter."
            );
        }

        let intercept_opt = if fit_intercept { Some(intercept) } else { None };
        Ok((coef, intercept_opt))
    }

    /// Configure early stopping for the solver
    pub fn with_early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping_config = Some(config);
        self
    }

    /// Solve Lasso regression with early stopping based on validation metrics
    ///
    /// This method automatically splits the data into training and validation sets
    /// and uses validation performance to determine when to stop training.
    pub fn solve_lasso_with_early_stopping(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
        fit_intercept: bool,
    ) -> Result<(Array1<Float>, Option<Float>, ValidationInfo)> {
        let early_stopping_config = self.early_stopping_config.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "Early stopping config not set. Use with_early_stopping() first.".to_string(),
            )
        })?;

        // Split data into training and validation sets
        let (x_train, y_train, x_val, y_val) = train_validation_split(
            x,
            y,
            early_stopping_config.validation_split,
            early_stopping_config.shuffle,
            early_stopping_config.random_state,
        )?;

        self.solve_lasso_with_early_stopping_split(
            &x_train,
            &y_train,
            &x_val,
            &y_val,
            alpha,
            fit_intercept,
        )
    }

    /// Solve Lasso regression with early stopping using pre-split data
    pub fn solve_lasso_with_early_stopping_split(
        &self,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
        x_val: &Array2<Float>,
        y_val: &Array1<Float>,
        alpha: Float,
        fit_intercept: bool,
    ) -> Result<(Array1<Float>, Option<Float>, ValidationInfo)> {
        let early_stopping_config = self.early_stopping_config.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "Early stopping config not set. Use with_early_stopping() first.".to_string(),
            )
        })?;

        let mut early_stopping = EarlyStopping::new(early_stopping_config.clone());

        let n_samples = x_train.nrows() as Float;
        let n_features = x_train.ncols();

        // Initialize coefficients
        let mut coef = Array1::zeros(n_features);
        let mut intercept = if fit_intercept {
            y_train.mean().unwrap_or(0.0)
        } else {
            0.0
        };

        // Store best coefficients if restore_best_weights is enabled
        let mut best_coef = coef.clone();
        let mut best_intercept = intercept;

        // Precompute norms for each feature
        let feature_norms: Array1<Float> = x_train
            .axis_iter(Axis(1))
            .map(|col| col.dot(&col) / n_samples)
            .collect();

        let mut validation_scores = Vec::new();
        let mut converged = false;

        // Coordinate descent iterations with early stopping
        for iter in 0..self.max_iter {
            let old_coef = coef.clone();

            // Update intercept if needed
            if fit_intercept {
                let residuals = y_train - &x_train.dot(&coef) - intercept;
                intercept = residuals.mean().unwrap_or(0.0);
            }

            // Update each coordinate
            for j in 0..n_features {
                if feature_norms[j] == 0.0 {
                    coef[j] = 0.0;
                    continue;
                }

                // Compute partial residual (excluding j-th feature)
                let mut residuals = y_train - &x_train.dot(&coef);
                if fit_intercept {
                    residuals -= intercept;
                }
                residuals = residuals + x_train.column(j).to_owned() * coef[j];

                // Compute gradient for j-th feature
                let gradient = x_train.column(j).dot(&residuals) / n_samples;

                // Apply soft thresholding
                coef[j] = soft_threshold(gradient, alpha) / feature_norms[j];
            }

            // Check convergence based on coefficient change
            let coef_change = (&coef - &old_coef).mapv(Float::abs).sum();
            if coef_change < self.tol {
                converged = true;
            }

            // Compute validation score (R² score)
            let val_predictions = if fit_intercept {
                x_val.dot(&coef) + intercept
            } else {
                x_val.dot(&coef)
            };

            let r2_score = compute_r2_score(&val_predictions, y_val);
            validation_scores.push(r2_score);

            // Check early stopping
            let should_continue = early_stopping.update(r2_score);

            // Store best weights if this is the best iteration so far
            if early_stopping_config.restore_best_weights
                && early_stopping.best_iteration() == iter + 1
            {
                best_coef = coef.clone();
                best_intercept = intercept;
            }

            if !should_continue || converged {
                break;
            }
        }

        // Restore best weights if configured
        let (final_coef, final_intercept) = if early_stopping_config.restore_best_weights {
            (best_coef, best_intercept)
        } else {
            (coef, intercept)
        };

        let validation_info = ValidationInfo {
            validation_scores,
            best_score: early_stopping.best_score(),
            best_iteration: early_stopping.best_iteration(),
            stopped_early: early_stopping.should_stop(),
            converged,
        };

        let intercept_opt = if fit_intercept {
            Some(final_intercept)
        } else {
            None
        };
        Ok((final_coef, intercept_opt, validation_info))
    }

    /// Solve ElasticNet regression with early stopping based on validation metrics
    pub fn solve_elastic_net_with_early_stopping(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
        l1_ratio: Float,
        fit_intercept: bool,
    ) -> Result<(Array1<Float>, Option<Float>, ValidationInfo)> {
        let early_stopping_config = self.early_stopping_config.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "Early stopping config not set. Use with_early_stopping() first.".to_string(),
            )
        })?;

        // Split data into training and validation sets
        let (x_train, y_train, x_val, y_val) = train_validation_split(
            x,
            y,
            early_stopping_config.validation_split,
            early_stopping_config.shuffle,
            early_stopping_config.random_state,
        )?;

        self.solve_elastic_net_with_early_stopping_split(
            &x_train,
            &y_train,
            &x_val,
            &y_val,
            alpha,
            l1_ratio,
            fit_intercept,
        )
    }

    /// Solve ElasticNet regression with early stopping using pre-split data
    pub fn solve_elastic_net_with_early_stopping_split(
        &self,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
        x_val: &Array2<Float>,
        y_val: &Array1<Float>,
        alpha: Float,
        l1_ratio: Float,
        fit_intercept: bool,
    ) -> Result<(Array1<Float>, Option<Float>, ValidationInfo)> {
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(SklearsError::InvalidParameter {
                name: "l1_ratio".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }

        let early_stopping_config = self.early_stopping_config.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "Early stopping config not set. Use with_early_stopping() first.".to_string(),
            )
        })?;

        let mut early_stopping = EarlyStopping::new(early_stopping_config.clone());

        let n_samples = x_train.nrows() as Float;
        let n_features = x_train.ncols();

        // Regularization parameters
        let l1_reg = alpha * l1_ratio;
        let l2_reg = alpha * (1.0 - l1_ratio);

        // Initialize coefficients
        let mut coef = Array1::zeros(n_features);
        let mut intercept = if fit_intercept {
            y_train.mean().unwrap_or(0.0)
        } else {
            0.0
        };

        // Store best coefficients if restore_best_weights is enabled
        let mut best_coef = coef.clone();
        let mut best_intercept = intercept;

        // Precompute norms for each feature (including L2 penalty)
        let feature_norms: Array1<Float> = x_train
            .axis_iter(Axis(1))
            .map(|col| col.dot(&col) / n_samples + l2_reg)
            .collect();

        let mut validation_scores = Vec::new();
        let mut converged = false;

        // Coordinate descent iterations with early stopping
        for iter in 0..self.max_iter {
            let old_coef = coef.clone();

            // Update intercept if needed
            if fit_intercept {
                let residuals = y_train - &x_train.dot(&coef) - intercept;
                intercept = residuals.mean().unwrap_or(0.0);
            }

            // Update each coordinate
            for j in 0..n_features {
                if feature_norms[j] == 0.0 {
                    coef[j] = 0.0;
                    continue;
                }

                // Compute partial residual (excluding j-th feature)
                let mut residuals = y_train - &x_train.dot(&coef);
                if fit_intercept {
                    residuals -= intercept;
                }
                residuals = residuals + x_train.column(j).to_owned() * coef[j];

                // Compute gradient for j-th feature
                let gradient = x_train.column(j).dot(&residuals) / n_samples;

                // Apply soft thresholding for L1 and account for L2
                coef[j] = soft_threshold(gradient, l1_reg) / feature_norms[j];
            }

            // Check convergence based on coefficient change
            let coef_change = (&coef - &old_coef).mapv(Float::abs).sum();
            if coef_change < self.tol {
                converged = true;
            }

            // Compute validation score (R² score)
            let val_predictions = if fit_intercept {
                x_val.dot(&coef) + intercept
            } else {
                x_val.dot(&coef)
            };

            let r2_score = compute_r2_score(&val_predictions, y_val);
            validation_scores.push(r2_score);

            // Check early stopping
            let should_continue = early_stopping.update(r2_score);

            // Store best weights if this is the best iteration so far
            if early_stopping_config.restore_best_weights
                && early_stopping.best_iteration() == iter + 1
            {
                best_coef = coef.clone();
                best_intercept = intercept;
            }

            if !should_continue || converged {
                break;
            }
        }

        // Restore best weights if configured
        let (final_coef, final_intercept) = if early_stopping_config.restore_best_weights {
            (best_coef, best_intercept)
        } else {
            (coef, intercept)
        };

        let validation_info = ValidationInfo {
            validation_scores,
            best_score: early_stopping.best_score(),
            best_iteration: early_stopping.best_iteration(),
            stopped_early: early_stopping.should_stop(),
            converged,
        };

        let intercept_opt = if fit_intercept {
            Some(final_intercept)
        } else {
            None
        };
        Ok((final_coef, intercept_opt, validation_info))
    }
}

/// Information about the validation process during early stopping
#[derive(Debug, Clone)]
pub struct ValidationInfo {
    /// Validation scores for each iteration
    pub validation_scores: Vec<Float>,
    /// Best validation score achieved
    pub best_score: Option<Float>,
    /// Iteration where best score was achieved
    pub best_iteration: usize,
    /// Whether training was stopped early
    pub stopped_early: bool,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Compute R² score for regression validation
pub fn compute_r2_score(y_pred: &Array1<Float>, y_true: &Array1<Float>) -> Float {
    let y_mean = y_true.mean().unwrap_or(0.0);
    let ss_res = (y_pred - y_true).mapv(|x| x * x).sum();
    let ss_tot = y_true.mapv(|yi| (yi - y_mean).powi(2)).sum();

    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - (ss_res / ss_tot)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(2.0, 1.0), 1.0);
        assert_eq!(soft_threshold(-2.0, 1.0), -1.0);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.5, 1.0), 0.0);
    }

    #[test]
    fn test_lasso_simple() {
        // Simple test case: y = 2x (no noise)
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let solver = CoordinateDescentSolver::default();

        // With small alpha, should be close to OLS solution
        let (coef, intercept) = solver.solve_lasso(&x, &y, 0.01, false).unwrap();
        assert_abs_diff_eq!(coef[0], 2.0, epsilon = 0.1);
        assert_eq!(intercept, None);

        // With large alpha, coefficient should shrink
        let (coef, _intercept) = solver.solve_lasso(&x, &y, 1.0, false).unwrap();
        assert!(coef[0] < 2.0);
        assert!(coef[0] > 0.0);
    }

    #[test]
    fn test_elastic_net() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let solver = CoordinateDescentSolver::default();

        // ElasticNet with l1_ratio=0.5
        let (coef, _) = solver.solve_elastic_net(&x, &y, 0.1, 0.5, false).unwrap();

        // Should be between Lasso and Ridge solutions
        let (lasso_coef, _) = solver.solve_lasso(&x, &y, 0.1, false).unwrap();

        // ElasticNet coefficient should be different from pure Lasso
        assert!(coef[0] != lasso_coef[0]);
    }

    #[test]
    fn test_early_stopping_lasso() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        // Create synthetic data with more samples for meaningful validation split
        let n_samples = 100;
        let n_features = 5;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        // Generate simple linear relationship
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = (i * j + 1) as Float / 10.0;
            }
            y[i] = 2.0 * x[[i, 0]] + 1.5 * x[[i, 1]] + 0.5 * (i as Float % 3.0);
        }

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(5),
            validation_split: 0.2,
            shuffle: true,
            random_state: Some(42),
            higher_is_better: true,
            min_iterations: 3,
            restore_best_weights: true,
        };

        let solver = CoordinateDescentSolver::default().with_early_stopping(early_stopping_config);

        let result = solver.solve_lasso_with_early_stopping(&x, &y, 0.01, true);
        assert!(result.is_ok());

        let (coef, intercept, validation_info) = result.unwrap();

        // Check that we have reasonable results
        assert_eq!(coef.len(), n_features);
        assert!(intercept.is_some());
        assert!(!validation_info.validation_scores.is_empty());
        assert!(validation_info.best_score.is_some());

        // Early stopping should have triggered or converged
        assert!(validation_info.stopped_early || validation_info.converged);
    }

    #[test]
    fn test_early_stopping_elastic_net() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        // Create synthetic data
        let n_samples = 80;
        let n_features = 4;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = (i + j) as Float / 10.0;
            }
            y[i] = 1.0 * x[[i, 0]] + 2.0 * x[[i, 1]] + 0.1 * (i as Float);
        }

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::TolerancePatience {
                tolerance: 0.01,
                patience: 3,
            },
            validation_split: 0.25,
            shuffle: false,
            random_state: None,
            higher_is_better: true,
            min_iterations: 2,
            restore_best_weights: false,
        };

        let solver = CoordinateDescentSolver::default().with_early_stopping(early_stopping_config);

        let result = solver.solve_elastic_net_with_early_stopping(&x, &y, 0.1, 0.5, true);
        assert!(result.is_ok());

        let (coef, intercept, validation_info) = result.unwrap();

        // Check results
        assert_eq!(coef.len(), n_features);
        assert!(intercept.is_some());
        assert!(!validation_info.validation_scores.is_empty());
        assert!(validation_info.best_score.is_some());

        // Validation info should be meaningful
        assert!(validation_info.best_iteration > 0);
    }

    #[test]
    fn test_early_stopping_with_presplit_data() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        // Create training and validation data
        let x_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y_train = array![3.0, 5.0, 7.0, 9.0]; // y = x1 + x2
        let x_val = array![[5.0, 6.0], [6.0, 7.0]];
        let y_val = array![11.0, 13.0];

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::TargetScore(0.8),
            validation_split: 0.2, // Not used since we provide pre-split data
            shuffle: false,
            random_state: None,
            higher_is_better: true,
            min_iterations: 1,
            restore_best_weights: true,
        };

        let solver = CoordinateDescentSolver::default().with_early_stopping(early_stopping_config);

        let result = solver
            .solve_lasso_with_early_stopping_split(&x_train, &y_train, &x_val, &y_val, 0.001, true);
        assert!(result.is_ok());

        let (coef, intercept, validation_info) = result.unwrap();

        assert_eq!(coef.len(), 2);
        assert!(intercept.is_some());
        assert!(!validation_info.validation_scores.is_empty());
    }

    #[test]
    fn test_validation_info_structure() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(2),
            validation_split: 0.25,
            shuffle: false,
            random_state: Some(123),
            higher_is_better: true,
            min_iterations: 1,
            restore_best_weights: true,
        };

        let solver = CoordinateDescentSolver {
            max_iter: 10,
            tol: 1e-6,
            cyclic: true,
            early_stopping_config: Some(early_stopping_config),
        };

        let result = solver.solve_lasso_with_early_stopping(&x, &y, 0.01, false);
        assert!(result.is_ok());

        let (_coef, _intercept, validation_info) = result.unwrap();

        // Validation info should have meaningful structure
        assert!(!validation_info.validation_scores.is_empty());
        assert!(validation_info.best_score.is_some());
        assert!(validation_info.best_iteration >= 1);

        // Should have stopped early or converged
        assert!(validation_info.stopped_early || validation_info.converged);

        // Validation scores should be finite
        for score in &validation_info.validation_scores {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_r2_score_computation() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];

        let r2 = compute_r2_score(&y_pred, &y_true);
        assert!(r2 > 0.9); // Should be high for good predictions
        assert!(r2 <= 1.0);

        // Perfect predictions should give R² = 1
        let perfect_pred = y_true.clone();
        let r2_perfect = compute_r2_score(&perfect_pred, &y_true);
        assert!((r2_perfect - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_early_stopping_without_config() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let solver = CoordinateDescentSolver::default(); // No early stopping config

        let result = solver.solve_lasso_with_early_stopping(&x, &y, 0.1, false);
        assert!(result.is_err());

        let error = result.unwrap_err();
        match error {
            SklearsError::InvalidInput(msg) => {
                assert!(msg.contains("Early stopping config not set"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }
}
