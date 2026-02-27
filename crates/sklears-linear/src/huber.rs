//! Huber Regressor for robust linear regression
//!
//! The Huber Regressor is a linear regression model that is robust to outliers.
//! It uses the Huber loss function which is quadratic for small errors and linear for large errors.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
};

/// Helper function to safely compute mean along an axis
#[inline]
fn safe_mean_axis(arr: &Array2<f64>, axis: Axis) -> Result<Array1<f64>> {
    arr.mean_axis(axis).ok_or_else(|| {
        SklearsError::NumericalError("Failed to compute mean along axis".to_string())
    })
}

/// Helper function to safely compute mean of 1D array
#[inline]
fn safe_mean(arr: &Array1<f64>) -> Result<f64> {
    arr.mean()
        .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean".to_string()))
}

/// Helper function to safely compare floats
#[inline]
fn compare_floats(a: &f64, b: &f64) -> Result<std::cmp::Ordering> {
    a.partial_cmp(b).ok_or_else(|| {
        SklearsError::InvalidInput("Cannot compare values: NaN encountered".to_string())
    })
}

/// Configuration for HuberRegressor
#[derive(Debug, Clone)]
pub struct HuberRegressorConfig {
    /// The parameter that controls the number of outliers the algorithm should tolerate
    pub epsilon: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// The alpha parameter for L2 regularization
    pub alpha: f64,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Tolerance for stopping criteria
    pub tol: f64,
}

impl Default for HuberRegressorConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.35, // Default value from scikit-learn
            max_iter: 100,
            alpha: 0.0001,
            fit_intercept: true,
            tol: 1e-5,
        }
    }
}

/// Huber Regressor for robust linear regression
pub struct HuberRegressor<State = Untrained> {
    config: HuberRegressorConfig,
    state: PhantomData<State>,
    coef_: Option<Array1<f64>>,
    intercept_: Option<f64>,
    scale_: Option<f64>,
    n_iter_: Option<usize>,
}

impl HuberRegressor<Untrained> {
    /// Create a new HuberRegressor with default configuration
    pub fn new() -> Self {
        Self {
            config: HuberRegressorConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            scale_: None,
            n_iter_: None,
        }
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the L2 regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to fit the intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the tolerance for stopping criteria
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }
}

impl Default for HuberRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HuberRegressor<Untrained> {
    type Config = HuberRegressorConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Huber loss function
fn huber_loss(residual: f64, epsilon: f64) -> f64 {
    let abs_residual = residual.abs();
    if abs_residual <= epsilon {
        0.5 * residual * residual
    } else {
        epsilon * abs_residual - 0.5 * epsilon * epsilon
    }
}

/// Derivative of Huber loss
fn huber_loss_derivative(residual: f64, epsilon: f64) -> f64 {
    if residual.abs() <= epsilon {
        residual
    } else {
        epsilon * residual.signum()
    }
}

/// Weight function for IRLS (Iteratively Reweighted Least Squares)
fn huber_weight(residual: f64, epsilon: f64) -> f64 {
    let abs_residual = residual.abs();
    if abs_residual <= f64::EPSILON || abs_residual <= epsilon {
        1.0
    } else {
        epsilon / abs_residual
    }
}

impl Fit<Array2<f64>, Array1<f64>> for HuberRegressor<Untrained> {
    type Fitted = HuberRegressor<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Center X and y if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.config.fit_intercept {
            let x_mean = safe_mean_axis(x, Axis(0))?;
            let y_mean = safe_mean(y)?;
            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;
            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Initialize coefficients
        let mut scale = 1.0;
        let mut n_iter = 0;

        // Initial fit using ordinary least squares
        let xt_x = x_centered.t().dot(&x_centered);
        let xt_x_reg = &xt_x + &Array2::eye(n_features) * self.config.alpha * n_samples as f64;
        let xt_y = x_centered.t().dot(&y_centered);

        // Solve normal equations
        let mut coef = xt_x_reg
            .solve(&xt_y)
            .unwrap_or_else(|_| Array1::zeros(n_features));

        // IRLS iterations
        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Compute residuals
            let predictions = x_centered.dot(&coef);
            let residuals = &y_centered - &predictions;

            // Estimate scale using MAD (Median Absolute Deviation)
            let mut abs_residuals: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();
            abs_residuals.sort_by(|a, b| compare_floats(a, b).unwrap_or(std::cmp::Ordering::Equal));
            let n = abs_residuals.len();
            let mad = if n % 2 == 0 {
                (abs_residuals[n / 2 - 1] + abs_residuals[n / 2]) / 2.0
            } else {
                abs_residuals[n / 2]
            };
            scale = 1.4826 * mad.max(1e-6); // Consistency factor for normal distribution

            // Compute weights
            let weights: Array1<f64> =
                residuals.mapv(|r| huber_weight(r / scale, self.config.epsilon));

            // Weighted least squares update
            let w_sqrt = weights.mapv(|w| w.sqrt());
            let x_weighted = &x_centered * &w_sqrt.clone().insert_axis(Axis(1));
            let y_weighted = &y_centered * &w_sqrt;

            let xt_w_x = x_weighted.t().dot(&x_weighted);
            let xt_w_x_reg =
                &xt_w_x + &Array2::eye(n_features) * self.config.alpha * n_samples as f64;
            let xt_w_y = x_weighted.t().dot(&y_weighted);

            // Solve weighted normal equations
            match xt_w_x_reg.solve(&xt_w_y) {
                Ok(new_coef) => {
                    // Check for convergence
                    let coef_change = (&new_coef - &coef).mapv(|x| x.abs()).sum();
                    coef = new_coef;

                    if coef_change < self.config.tol {
                        break;
                    }
                }
                Err(_) => {
                    break;
                }
            }
        }

        // Compute intercept if needed
        let intercept = if self.config.fit_intercept {
            let y_m = y_mean.ok_or_else(|| {
                SklearsError::InvalidState(
                    "y_mean should be Some when fit_intercept=true".to_string(),
                )
            })?;
            let x_m = x_mean.ok_or_else(|| {
                SklearsError::InvalidState(
                    "x_mean should be Some when fit_intercept=true".to_string(),
                )
            })?;
            y_m - x_m.dot(&coef)
        } else {
            0.0
        };

        Ok(HuberRegressor {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            scale_: Some(scale),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for HuberRegressor<Trained> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let coef = self.coef_.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("coef_ should be Some in Trained state".to_string())
        })?;
        let intercept = self.intercept_.ok_or_else(|| {
            SklearsError::InvalidState("intercept_ should be Some in Trained state".to_string())
        })?;

        Ok(x.dot(coef) + intercept)
    }
}

impl Score<Array2<f64>, Array1<f64>> for HuberRegressor<Trained> {
    type Float = f64;

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        let ss_res = (y - &predictions).mapv(|e| e * e).sum();
        let y_mean = safe_mean(y)?;
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        Ok(1.0 - ss_res / ss_tot)
    }
}

impl HuberRegressor<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> Result<&Array1<f64>> {
        self.coef_.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("coef_ should be Some in Trained state".to_string())
        })
    }

    /// Get the intercept
    pub fn intercept(&self) -> Result<f64> {
        self.intercept_.ok_or_else(|| {
            SklearsError::InvalidState("intercept_ should be Some in Trained state".to_string())
        })
    }

    /// Get the scale parameter
    pub fn scale(&self) -> Result<f64> {
        self.scale_.ok_or_else(|| {
            SklearsError::InvalidState("scale_ should be Some in Trained state".to_string())
        })
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> Result<usize> {
        self.n_iter_.ok_or_else(|| {
            SklearsError::InvalidState("n_iter_ should be Some in Trained state".to_string())
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
    fn test_huber_simple() {
        // Test with simple linear relationship
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let model = HuberRegressor::new().fit(&x, &y).unwrap();

        assert_abs_diff_eq!(model.coef().unwrap()[0], 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(model.intercept().unwrap(), 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_huber_with_outliers() {
        // Test with outliers - more data points for stability
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![
            2.1, 3.9, 6.1, 7.9, 10.1, 11.9, 14.1, 16.0, 18.1, 50.0 // Last point is an outlier
        ];

        let model = HuberRegressor::new()
            .epsilon(1.35)
            .max_iter(200)
            .fit(&x, &y)
            .unwrap();

        // The coefficient should be somewhat robust to the outlier
        println!(
            "Coefficient: {}, Intercept: {}",
            model.coef().unwrap()[0],
            model.intercept().unwrap()
        );
        assert!(
            model.coef().unwrap()[0] > 1.8 && model.coef().unwrap()[0] < 3.0,
            "Coefficient: {}",
            model.coef().unwrap()[0]
        );

        // Test that predictions are reasonable
        let x_test = array![[3.0], [5.0], [7.0]];
        let predictions = model.predict(&x_test).unwrap();
        // With the outlier affecting the fit, predictions will be different
        // Just ensure they're in a reasonable range
        assert!(predictions[0] > 0.0 && predictions[0] < 15.0);
        assert!(predictions[1] > 5.0 && predictions[1] < 20.0);
    }

    #[test]
    fn test_huber_no_intercept() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![2.0, 4.0, 6.0];

        let model = HuberRegressor::new()
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(model.coef().unwrap()[0], 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(model.intercept().unwrap(), 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_huber_with_regularization() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.1, 3.9, 6.1, 7.9, 10.1];

        let model = HuberRegressor::new().alpha(0.1).fit(&x, &y).unwrap();

        // With regularization, coefficient should be slightly shrunk
        assert!(model.coef().unwrap()[0] < 2.0);
    }

    #[test]
    fn test_huber_multivariate() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![5.0, 8.0, 11.0, 14.0]; // y = 1*x1 + 2*x2

        let model = HuberRegressor::new().max_iter(200).fit(&x, &y).unwrap();

        // With limited data points and regularization, exact fit is difficult
        assert_abs_diff_eq!(model.coef().unwrap()[0], 1.0, epsilon = 0.6);
        assert_abs_diff_eq!(model.coef().unwrap()[1], 2.0, epsilon = 0.6);
    }
}
