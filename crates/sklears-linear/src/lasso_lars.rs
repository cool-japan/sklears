//! Lasso LARS (Least Angle Regression with Lasso modification) implementation

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::solve;
use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::Float,
};

/// Configuration for LassoLars
#[derive(Debug, Clone)]
pub struct LassoLarsConfig {
    /// L1 regularization parameter
    pub alpha: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Whether to normalize/standardize features before fitting
    pub normalize: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for stopping criterion
    pub eps: Float,
}

impl Default for LassoLarsConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            normalize: true,
            max_iter: 500,
            eps: Float::EPSILON.sqrt(),
        }
    }
}

/// Lasso LARS model
#[derive(Debug, Clone)]
pub struct LassoLars<State = Untrained> {
    config: LassoLarsConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_: Option<usize>,
    active_: Option<Vec<usize>>,
    alphas_: Option<Array1<Float>>,
    n_iter_: Option<usize>,
}

impl LassoLars<Untrained> {
    /// Create a new LassoLars model
    pub fn new() -> Self {
        Self {
            config: LassoLarsConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            active_: None,
            alphas_: None,
            n_iter_: None,
        }
    }

    /// Set the L1 regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to normalize features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn eps(mut self, eps: Float) -> Self {
        self.config.eps = eps;
        self
    }
}

impl Default for LassoLars<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LassoLars<Untrained> {
    type Float = Float;
    type Config = LassoLarsConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for LassoLars<Untrained> {
    type Fitted = LassoLars<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Center X and y
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = x - &x_mean;

        let y_mean = if self.config.fit_intercept {
            y.mean().unwrap_or(0.0)
        } else {
            0.0
        };
        let y_centered = y - y_mean;

        // Normalize X if requested
        let x_scale = if self.config.normalize {
            let mut scale = Array1::zeros(n_features);
            for j in 0..n_features {
                let col = x_centered.column(j);
                scale[j] = col.dot(&col).sqrt();
                if scale[j] > self.config.eps {
                    x_centered.column_mut(j).mapv_inplace(|x| x / scale[j]);
                } else {
                    scale[j] = 1.0;
                }
            }
            scale
        } else {
            Array1::ones(n_features)
        };

        // Initialize Lasso LARS algorithm
        let mut coef: Array1<Float> = Array1::zeros(n_features);
        let mut active: Vec<usize> = Vec::new();
        let mut alphas = Vec::new();
        let mut sign_active: Vec<Float> = Vec::new();

        // Compute initial correlations
        let mut residual = y_centered.clone();
        let mut correlations = x_centered.t().dot(&residual);

        // Scale alpha for normalized data
        let alpha_scaled = if self.config.normalize {
            self.config.alpha * n_samples as Float
        } else {
            self.config.alpha
        };

        let mut n_iter = 0;

        for _ in 0..self.config.max_iter {
            // Find the predictor with largest correlation
            let mut max_corr = 0.0;
            let mut best_idx = 0;

            for j in 0..n_features {
                if !active.contains(&j) {
                    let corr = correlations[j].abs();
                    if corr > max_corr {
                        max_corr = corr;
                        best_idx = j;
                    }
                }
            }

            // Check if we've reached the regularization level
            if max_corr < alpha_scaled {
                break;
            }

            // Add variable to active set
            active.push(best_idx);
            sign_active.push(correlations[best_idx].signum());
            alphas.push(max_corr);

            // Compute the direction of the step
            let n_active = active.len();
            let mut x_active = Array2::zeros((n_samples, n_active));
            for (i, &j) in active.iter().enumerate() {
                x_active.column_mut(i).assign(&x_centered.column(j));
                x_active.column_mut(i).mapv_inplace(|x| x * sign_active[i]);
            }

            // Gram matrix of active set
            let gram = x_active.t().dot(&x_active);

            // Direction vector
            let ones = Array1::ones(n_active);

            // Add small regularization to avoid singular matrix
            let mut gram_reg = gram.clone();
            for i in 0..n_active {
                gram_reg[[i, i]] += 1e-10;
            }

            let gram_inv_ones = &solve(&gram_reg.view(), &ones.view(), None)
                .map_err(|e| SklearsError::NumericalError(format!("Failed to solve: {}", e)))?;

            let normalization = 1.0 / ones.dot(gram_inv_ones).sqrt();
            let direction = gram_inv_ones * normalization;

            // Equiangular direction in full feature space
            let equiangular = x_active.dot(&direction);

            // Compute step size
            let mut gamma = max_corr;

            // Check how far we can go before another variable enters
            for j in 0..n_features {
                if !active.contains(&j) {
                    let a_j = x_centered.column(j).dot(&equiangular);
                    let c_j = correlations[j];

                    // Two possible directions
                    let gamma_plus = (max_corr - c_j) / (normalization - a_j + self.config.eps);
                    let gamma_minus = (max_corr + c_j) / (normalization + a_j + self.config.eps);

                    if gamma_plus > 0.0 && gamma_plus < gamma {
                        gamma = gamma_plus;
                    }
                    if gamma_minus > 0.0 && gamma_minus < gamma {
                        gamma = gamma_minus;
                    }
                }
            }

            // Also check for sign changes (Lasso modification)
            let mut min_gamma_sign = gamma;
            let mut sign_change_idx = None;

            for (i, &j) in active.iter().enumerate() {
                let gamma_sign = -coef[j] / (direction[i] * sign_active[i]);
                if gamma_sign > 0.0 && gamma_sign < min_gamma_sign {
                    min_gamma_sign = gamma_sign;
                    sign_change_idx = Some(i);
                }
            }

            gamma = min_gamma_sign;

            // Update coefficients
            for (i, &j) in active.iter().enumerate() {
                coef[j] += gamma * direction[i] * sign_active[i];
            }

            // Update residual and correlations
            residual = residual - gamma * equiangular;
            correlations = x_centered.t().dot(&residual);

            // Remove variable if sign changed
            if let Some(idx) = sign_change_idx {
                coef[active[idx]] = 0.0;
                active.remove(idx);
                sign_active.remove(idx);
            }

            n_iter += 1;

            // Check if we've reached the target alpha
            let current_alpha = correlations
                .mapv(Float::abs)
                .fold(0.0 as Float, |acc: Float, &x| acc.max(x));
            if current_alpha <= alpha_scaled {
                break;
            }
        }

        // Rescale coefficients if we normalized
        if self.config.normalize {
            for j in 0..n_features {
                if x_scale[j] > 0.0 {
                    coef[j] /= x_scale[j];
                }
            }
        }

        // Compute intercept if needed
        let intercept = if self.config.fit_intercept {
            Some(y_mean - x_mean.dot(&coef))
        } else {
            None
        };

        Ok(LassoLars {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: intercept,
            n_features_: Some(n_features),
            active_: Some(active),
            alphas_: Some(Array1::from(alphas)),
            n_iter_: Some(n_iter),
        })
    }
}

impl LassoLars<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().expect("Model is trained")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }

    /// Get the indices of active features
    pub fn active(&self) -> &[usize] {
        self.active_.as_ref().expect("Model is trained")
    }

    /// Get the values of the correlation at each iteration
    pub fn alphas(&self) -> &Array1<Float> {
        self.alphas_.as_ref().expect("Model is trained")
    }

    /// Get the number of iterations run
    pub fn n_iter(&self) -> usize {
        self.n_iter_.expect("Model is trained")
    }
}

impl Predict<Array2<Float>, Array1<Float>> for LassoLars<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let coef = self.coef_.as_ref().expect("Model is trained");
        let mut predictions = x.dot(coef);

        if let Some(intercept) = self.intercept_ {
            predictions += intercept;
        }

        Ok(predictions)
    }
}

impl Score<Array2<Float>, Array1<Float>> for LassoLars<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate R² score
        let ss_res = (&predictions - y).mapv(|x| x * x).sum();
        let y_mean = y.mean().unwrap_or(0.0);
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            return Ok(1.0);
        }

        Ok(1.0 - (ss_res / ss_tot))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lasso_lars_simple() {
        // Simple test case
        let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]];
        let y = array![3.0, 6.0, 9.0, 12.0, 15.0]; // y = x1 + x2

        let model = LassoLars::new()
            .alpha(0.01)
            .fit_intercept(false)
            .normalize(false)
            .fit(&x, &y)
            .unwrap();

        // Should select one of the perfectly correlated features
        let coef = model.coef();
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();
        assert!(n_nonzero >= 1);

        // Predictions should be accurate
        let predictions = model.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(predictions[i], y[i], epsilon = 0.1);
        }
    }

    #[test]
    fn test_lasso_lars_sparsity() {
        // Create data where only first feature is relevant
        let n_samples = 20;
        let mut x = Array2::zeros((n_samples, 5));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            x[[i, 0]] = i as f64;
            x[[i, 1]] = (i as f64) * 0.1; // weak feature
                                          // Add deterministic noise
            x[[i, 2]] = ((i * 7) % 10) as f64 / 10.0;
            x[[i, 3]] = ((i * 13) % 10) as f64 / 10.0;
            x[[i, 4]] = ((i * 17) % 10) as f64 / 10.0;
            y[i] = 2.0 * x[[i, 0]] + 0.1 * (i % 3) as f64;
        }

        // With moderate regularization, should select only the most important features
        let model = LassoLars::new()
            .alpha(1.0)
            .fit_intercept(false)
            .normalize(true)
            .fit(&x, &y)
            .unwrap();

        let coef = model.coef();

        // Count non-zero coefficients
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();

        // Should have sparse solution (less than all features)
        assert!(
            n_nonzero < 5,
            "Solution should be sparse, got {} non-zero coefficients",
            n_nonzero
        );

        // At least one coefficient should be selected
        assert!(
            n_nonzero >= 1,
            "At least one coefficient should be selected"
        );
    }

    #[test]
    fn test_lasso_lars_with_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1

        let model = LassoLars::new()
            .alpha(0.01)
            .fit_intercept(true)
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(model.intercept().unwrap(), 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_lasso_lars_strong_regularization() {
        let x = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];

        // With very strong regularization, all coefficients should be zero
        let model = LassoLars::new()
            .alpha(100.0)
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        let coef = model.coef();
        for &c in coef.iter() {
            assert_abs_diff_eq!(c, 0.0, epsilon = 1e-10);
        }
    }
}
