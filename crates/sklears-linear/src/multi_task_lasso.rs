//! Multi-task Lasso model implementation
//!
//! Multi-task Lasso jointly solves multiple regression problems with shared sparsity patterns.
//! It minimizes: (1/2n) ||Y - XW||²_F + α||W||_{2,1}
//! where ||W||_{2,1} = Σ_i ||w_i||_2 encourages entire rows of W to be zero.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

/// Configuration for Multi-task Lasso
#[derive(Debug, Clone)]
pub struct MultiTaskLassoConfig {
    /// L1 regularization parameter
    pub alpha: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to use warm start
    pub warm_start: bool,
    /// Whether to use precomputed Gram matrix
    pub precompute: bool,
}

impl Default for MultiTaskLassoConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            warm_start: false,
            precompute: false,
        }
    }
}

/// Multi-task Lasso model
#[derive(Debug, Clone)]
pub struct MultiTaskLasso<State = Untrained> {
    config: MultiTaskLassoConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    n_features_: Option<usize>,
    n_tasks_: Option<usize>,
    n_iter_: Option<usize>,
}

impl MultiTaskLasso<Untrained> {
    /// Create a new Multi-task Lasso model
    pub fn new() -> Self {
        Self {
            config: MultiTaskLassoConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_tasks_: None,
            n_iter_: None,
        }
    }

    /// Create with specific alpha
    pub fn with_alpha(alpha: Float) -> Self {
        Self::new().alpha(alpha)
    }

    /// Set alpha parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
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

    /// Builder method to create from config
    pub fn builder() -> MultiTaskLassoBuilder {
        MultiTaskLassoBuilder::default()
    }
}

/// Builder for MultiTaskLasso
#[derive(Debug, Default)]
pub struct MultiTaskLassoBuilder {
    config: MultiTaskLassoConfig,
}

impl MultiTaskLassoBuilder {
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn build(self) -> MultiTaskLasso<Untrained> {
        MultiTaskLasso {
            config: self.config,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_tasks_: None,
            n_iter_: None,
        }
    }
}

impl Default for MultiTaskLasso<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for MultiTaskLasso<State> {
    type Float = Float;
    type Config = MultiTaskLassoConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array2<Float>> for MultiTaskLasso<Untrained> {
    type Fitted = MultiTaskLasso<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_tasks = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == Y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, Y.shape[0]={}", n_samples, y.nrows()),
            });
        }

        if n_samples < n_features {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be >= number of features".to_string(),
            ));
        }

        // Initialize coefficients
        let mut coef: Array2<Float> = Array2::zeros((n_features, n_tasks));
        let intercept = if self.config.fit_intercept {
            y.mean_axis(Axis(0)).ok_or_else(|| {
                SklearsError::NumericalError("Failed to compute mean of Y".to_string())
            })?
        } else {
            Array1::zeros(n_tasks)
        };

        // Center data if fitting intercept
        let (x_centered, y_centered) = if self.config.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).ok_or_else(|| {
                SklearsError::NumericalError("Failed to compute mean of X".to_string())
            })?;
            let x_c = x - &x_mean.insert_axis(Axis(0));
            let y_c = y - &intercept.clone().insert_axis(Axis(0));
            (x_c, y_c)
        } else {
            (x.clone(), y.clone())
        };

        // Coordinate descent for multi-task lasso
        let mut n_iter = 0;
        let n_samples_f = n_samples as Float;

        for iter in 0..self.config.max_iter {
            let old_coef = coef.clone();

            // Update each feature
            for j in 0..n_features {
                // Compute residuals without j-th feature
                let mut residuals = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        let x_col = x_centered.column(k);
                        let coef_row = coef.row(k);
                        for t in 0..n_tasks {
                            let mut res_col = residuals.column_mut(t);
                            res_col.scaled_add(-coef_row[t], &x_col);
                        }
                    }
                }

                // Compute gradient for j-th feature across all tasks
                let x_j = x_centered.column(j);
                let mut gradient: Array1<Float> = Array1::zeros(n_tasks);
                for t in 0..n_tasks {
                    gradient[t] = x_j.dot(&residuals.column(t)) / n_samples_f;
                }

                // Compute L2 norm of gradient
                let gradient_norm = gradient.dot(&gradient).sqrt();

                // Apply block soft thresholding
                if gradient_norm > self.config.alpha {
                    let scale = 1.0 - self.config.alpha / gradient_norm;
                    let x_j_norm_sq = x_j.dot(&x_j) / n_samples_f;

                    for t in 0..n_tasks {
                        coef[[j, t]] = scale * gradient[t] / x_j_norm_sq;
                    }
                } else {
                    // Set entire row to zero
                    for t in 0..n_tasks {
                        coef[[j, t]] = 0.0;
                    }
                }
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(Float::abs).sum();
            if coef_change < self.config.tol {
                n_iter = iter + 1;
                break;
            }
            n_iter = iter + 1;
        }

        if n_iter == self.config.max_iter {
            eprintln!("Warning: MultiTaskLasso did not converge. Consider increasing max_iter.");
        }

        Ok(MultiTaskLasso {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            n_features_: Some(n_features),
            n_tasks_: Some(n_tasks),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<Float>, Array2<Float>> for MultiTaskLasso<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Validate input
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let coef = self.coef_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        let intercept = self
            .intercept_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        // Compute predictions: Y = X @ W + intercept
        let mut predictions = x.dot(coef);

        // Add intercept
        for i in 0..predictions.nrows() {
            let mut row = predictions.row_mut(i);
            row += intercept;
        }

        Ok(predictions)
    }
}

impl MultiTaskLasso<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> Result<&Array2<Float>> {
        self.coef_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "coef()".to_string(),
        })
    }

    /// Get the intercept
    pub fn intercept(&self) -> Result<&Array1<Float>> {
        self.intercept_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "intercept()".to_string(),
            })
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> Result<usize> {
        self.n_iter_.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_iter()".to_string(),
        })
    }

    /// Get the number of features
    pub fn n_features(&self) -> Result<usize> {
        self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_features()".to_string(),
        })
    }

    /// Get the number of tasks
    pub fn n_tasks(&self) -> Result<usize> {
        self.n_tasks_.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_tasks()".to_string(),
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_task_lasso_simple() {
        // Simple test: Y = X @ W where W has some zero rows
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        // Two tasks with true coefficients
        // First feature: [2.0, 3.0]
        // Second feature: [0.0, 0.0] (sparse)
        // Third feature: [1.0, -1.0]
        let y = array![
            [2.0, 3.0],  // x[0] @ w
            [0.0, 0.0],  // x[1] @ w
            [1.0, -1.0], // x[2] @ w
            [2.0, 3.0],  // x[3] @ w
            [3.0, 2.0],  // x[4] @ w
            [1.0, -1.0], // x[5] @ w
        ];

        let model = MultiTaskLasso::new().alpha(0.01).fit_intercept(false);

        let fitted = model.fit(&x, &y).unwrap();
        let coef = fitted.coef().unwrap();

        // Check that coefficients are approximately correct
        assert_abs_diff_eq!(coef[[0, 0]], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(coef[[0, 1]], 3.0, epsilon = 0.1);

        // Second feature should be sparse (close to zero)
        assert_abs_diff_eq!(coef[[1, 0]], 0.0, epsilon = 0.1);
        assert_abs_diff_eq!(coef[[1, 1]], 0.0, epsilon = 0.1);

        assert_abs_diff_eq!(coef[[2, 0]], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(coef[[2, 1]], -1.0, epsilon = 0.1);
    }

    #[test]
    fn test_multi_task_lasso_with_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];

        let model = MultiTaskLasso::new().alpha(0.1).fit_intercept(true);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 2]);

        // Should have non-zero intercepts
        let intercept = fitted.intercept().unwrap();
        assert_eq!(intercept.len(), 2);
    }

    #[test]
    fn test_multi_task_lasso_high_alpha() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        // High alpha should make all coefficients zero
        let model = MultiTaskLasso::new().alpha(100.0).fit_intercept(false);

        let fitted = model.fit(&x, &y).unwrap();
        let coef = fitted.coef().unwrap();

        // All coefficients should be zero
        for &c in coef.iter() {
            assert_abs_diff_eq!(c, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_builder_pattern() {
        let model = MultiTaskLasso::builder()
            .alpha(0.5)
            .max_iter(500)
            .tol(1e-3)
            .fit_intercept(true)
            .build();

        assert_eq!(model.config.alpha, 0.5);
        assert_eq!(model.config.max_iter, 500);
        assert_eq!(model.config.tol, 1e-3);
        assert!(model.config.fit_intercept);
    }
}
