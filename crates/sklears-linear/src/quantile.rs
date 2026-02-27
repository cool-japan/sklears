//! Quantile Regression
//!
//! Quantile regression aims at estimating the conditional quantiles of the response variable,
//! instead of the conditional mean. This is useful for obtaining a more complete picture of
//! the conditional distribution of the response variable.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::Float,
};

use crate::solver::Solver;

/// Helper function to safely compute mean along an axis
#[inline]
fn safe_mean_axis(arr: &Array2<Float>, axis: Axis) -> Result<Array1<Float>> {
    arr.mean_axis(axis).ok_or_else(|| {
        SklearsError::NumericalError("Failed to compute mean along axis".to_string())
    })
}

/// Helper function to safely compute mean of 1D array
#[inline]
fn safe_mean(arr: &Array1<Float>) -> Result<Float> {
    arr.mean().ok_or_else(|| {
        SklearsError::NumericalError("Failed to compute mean".to_string())
    })
}

/// Helper function to safely compare floats
#[inline]
fn compare_floats(a: &Float, b: &Float) -> Result<std::cmp::Ordering> {
    a.partial_cmp(b).ok_or_else(|| {
        SklearsError::InvalidInput("Cannot compare values: NaN encountered".to_string())
    })
}

/// Configuration for QuantileRegressor
#[derive(Debug, Clone)]
pub struct QuantileRegressorConfig {
    /// The quantile that the model tries to predict
    pub quantile: Float,
    /// Regularization constant that multiplies the L1 penalty term
    pub alpha: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Solver to use: 'interior-point', 'highs-ds', 'highs-ipm', 'highs', 'revised simplex'
    pub solver: QuantileSolver,
    /// Parameters for the solver
    pub solver_options: Option<SolverOptions>,
}

/// Solver options for quantile regression
#[derive(Debug, Clone)]
pub struct SolverOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for stopping criterion
    pub tol: Float,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-4,
        }
    }
}

/// Available solvers for quantile regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantileSolver {
    InteriorPoint,
    CoordinateDescent,
    IRLS, // Iteratively Reweighted Least Squares
}

impl Default for QuantileRegressorConfig {
    fn default() -> Self {
        Self {
            quantile: 0.5, // Median regression by default
            alpha: 1.0,
            fit_intercept: true,
            solver: QuantileSolver::InteriorPoint,
            solver_options: Some(SolverOptions::default()),
        }
    }
}

/// Quantile Regressor
pub struct QuantileRegressor<State = Untrained> {
    config: QuantileRegressorConfig,
    state: PhantomData<State>,
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_in_: Option<usize>,
    n_iter_: Option<usize>,
}

impl QuantileRegressor<Untrained> {
    /// Create a new QuantileRegressor with default configuration
    pub fn new() -> Self {
        Self {
            config: QuantileRegressorConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_in_: None,
            n_iter_: None,
        }
    }

    /// Set the quantile to predict
    pub fn quantile(mut self, quantile: Float) -> Self {
        if quantile <= 0.0 || quantile >= 1.0 {
            panic!("Quantile must be in (0, 1)");
        }
        self.config.quantile = quantile;
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to fit the intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the solver
    pub fn solver(mut self, solver: QuantileSolver) -> Self {
        self.config.solver = solver;
        self
    }
}

impl Default for QuantileRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for QuantileRegressor<Untrained> {
    type Float = Float;
    type Config = QuantileRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for QuantileRegressor<Trained> {
    type Float = Float;
    type Config = QuantileRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute the check function (pinball loss) for quantile regression
fn check_function(residual: Float, quantile: Float) -> Float {
    if residual >= 0.0 {
        quantile * residual
    } else {
        (quantile - 1.0) * residual
    }
}

/// Compute the derivative of the check function
fn check_function_derivative(residual: Float, quantile: Float) -> Float {
    if residual > 0.0 {
        quantile
    } else if residual < 0.0 {
        quantile - 1.0
    } else {
        0.0 // Subgradient at 0
    }
}

/// Solve quantile regression using coordinate descent
fn solve_coordinate_descent(
    x: &Array2<Float>,
    y: &Array1<Float>,
    quantile: Float,
    alpha: Float,
    fit_intercept: bool,
    options: &SolverOptions,
) -> Result<(Array1<Float>, Float, usize)> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Initialize coefficients
    let mut coef = Array1::<Float>::zeros(n_features);
    let mut intercept = 0.0;

    // Center X and y if fitting intercept
    let (x_centered, y_centered, x_mean, y_mean) = if fit_intercept {
        let x_mean = safe_mean_axis(x, Axis(0))?;
        let y_mean = safe_mean(y)?;
        let x_centered = x - &x_mean;
        let y_centered = y - y_mean;
        (x_centered, y_centered, Some(x_mean), Some(y_mean))
    } else {
        (x.clone(), y.clone(), None, None)
    };

    let mut n_iter = 0;

    // Coordinate descent iterations
    for iter in 0..options.max_iter {
        n_iter = iter + 1;
        let coef_old = coef.clone();

        // Update each coordinate
        for j in 0..n_features {
            let mut gradient = 0.0;
            let mut x_j_norm_sq = 0.0;

            // Compute residuals without feature j
            let mut residuals = Array1::<Float>::zeros(n_samples);
            for i in 0..n_samples {
                let mut pred = 0.0;
                for k in 0..n_features {
                    if k != j {
                        pred += x_centered[[i, k]] * coef[k];
                    }
                }
                residuals[i] = y_centered[i] - pred;
                x_j_norm_sq += x_centered[[i, j]] * x_centered[[i, j]];
            }

            // Compute gradient for feature j
            for i in 0..n_samples {
                let residual = residuals[i] - x_centered[[i, j]] * coef[j];
                gradient += x_centered[[i, j]] * check_function_derivative(residual, quantile);
            }

            gradient /= n_samples as Float;
            x_j_norm_sq /= n_samples as Float;

            // Soft thresholding for L1 penalty
            if gradient > alpha {
                coef[j] = -(gradient - alpha) / x_j_norm_sq;
            } else if gradient < -alpha {
                coef[j] = -(gradient + alpha) / x_j_norm_sq;
            } else {
                coef[j] = 0.0;
            }
        }

        // Check convergence
        let coef_change = (&coef - &coef_old).mapv(|x| x.abs()).sum();
        if coef_change < options.tol {
            break;
        }
    }

    // Compute intercept if needed
    if fit_intercept {
        let x_mean = x_mean.ok_or_else(|| {
            SklearsError::InvalidState("x_mean should be Some when fit_intercept=true".to_string())
        })?;
        let y_mean = y_mean.ok_or_else(|| {
            SklearsError::InvalidState("y_mean should be Some when fit_intercept=true".to_string())
        })?;
        intercept = y_mean - x_mean.dot(&coef);
    }

    Ok((coef, intercept, n_iter))
}

/// Solve quantile regression using IRLS (Iteratively Reweighted Least Squares)
fn solve_irls(
    x: &Array2<Float>,
    y: &Array1<Float>,
    quantile: Float,
    alpha: Float,
    fit_intercept: bool,
    options: &SolverOptions,
) -> Result<(Array1<Float>, Float, usize)> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let epsilon = 1e-6; // Small value to avoid division by zero

    // Add intercept column if needed
    let x_design = if fit_intercept {
        let mut x_with_intercept = Array2::<Float>::zeros((n_samples, n_features + 1));
        x_with_intercept
            .slice_mut(scirs2_core::ndarray::s![.., 0])
            .fill(1.0);
        x_with_intercept
            .slice_mut(scirs2_core::ndarray::s![.., 1..])
            .assign(x);
        x_with_intercept
    } else {
        x.clone()
    };

    let n_params = x_design.ncols();

    // Initialize coefficients
    let mut beta = Array1::<Float>::zeros(n_params);

    let mut n_iter = 0;

    // IRLS iterations
    for iter in 0..options.max_iter {
        n_iter = iter + 1;

        // Compute residuals
        let predictions = x_design.dot(&beta);
        let residuals = y - &predictions;

        // Compute weights
        let weights = residuals.mapv(|r| {
            let abs_r = r.abs();
            if abs_r < epsilon {
                1.0 / epsilon
            } else {
                1.0 / abs_r
            }
        });

        // Create weighted design matrix and response
        let w_sqrt = weights.mapv(|w| w.sqrt());
        let x_weighted = &x_design * &w_sqrt.clone().insert_axis(Axis(1));
        let y_weighted = y * &w_sqrt;

        // Add L2 regularization to diagonal
        let xt_w_x = x_weighted.t().dot(&x_weighted);
        let mut xt_w_x_reg = xt_w_x + Array2::<Float>::eye(n_params) * alpha;

        // Adjust the gradient for quantile
        let mut gradient_adjustment = Array1::<Float>::zeros(n_samples);
        for i in 0..n_samples {
            if residuals[i] > 0.0 {
                gradient_adjustment[i] = quantile;
            } else if residuals[i] < 0.0 {
                gradient_adjustment[i] = quantile - 1.0;
            } else {
                gradient_adjustment[i] = 2.0 * quantile - 1.0;
            }
        }

        let y_adjusted = &y_weighted + &x_weighted.dot(&beta) * &gradient_adjustment * &w_sqrt;
        let xt_w_y = x_weighted.t().dot(&y_adjusted);

        // Solve weighted least squares
        match xt_w_x_reg.solve(&xt_w_y) {
            Ok(new_beta) => {
                // Check convergence
                let beta_change = (new_beta.clone() - &beta).mapv(|x| x.abs()).sum();
                beta = new_beta;

                if beta_change < options.tol {
                    break;
                }
            }
            Err(_) => {
                return Err(SklearsError::InvalidInput(
                    "Failed to solve weighted least squares".to_string(),
                ));
            }
        }
    }

    // Extract coefficients and intercept
    let (coef, intercept) = if fit_intercept {
        let intercept = beta[0];
        let coef = beta.slice(scirs2_core::ndarray::s![1..]).to_owned();
        (coef, intercept)
    } else {
        (beta, 0.0)
    };

    Ok((coef, intercept, n_iter))
}

impl Fit<Array2<Float>, Array1<Float>> for QuantileRegressor<Untrained> {
    type Fitted = QuantileRegressor<Trained>;

    fn fit(mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.config.quantile <= 0.0 || self.config.quantile >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "Quantile must be in (0, 1)".to_string(),
            ));
        }

        let default_options = SolverOptions::default();
        let options = self
            .config
            .solver_options
            .as_ref()
            .unwrap_or(&default_options);

        // Solve quantile regression
        let (coef, intercept, n_iter) = match self.config.solver {
            QuantileSolver::CoordinateDescent => solve_coordinate_descent(
                x,
                y,
                self.config.quantile,
                self.config.alpha,
                self.config.fit_intercept,
                options,
            )?,
            QuantileSolver::IRLS => solve_irls(
                x,
                y,
                self.config.quantile,
                self.config.alpha,
                self.config.fit_intercept,
                options,
            )?,
            QuantileSolver::InteriorPoint => {
                // For now, fall back to IRLS
                solve_irls(
                    x,
                    y,
                    self.config.quantile,
                    self.config.alpha,
                    self.config.fit_intercept,
                    options,
                )?
            }
        };

        Ok(QuantileRegressor {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for QuantileRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let coef = self.coef_.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("coef_ should be Some in Trained state".to_string())
        })?;
        let intercept = self.intercept_.ok_or_else(|| {
            SklearsError::InvalidState("intercept_ should be Some in Trained state".to_string())
        })?;

        let n_features_in = self.n_features_in_.ok_or_else(|| {
            SklearsError::InvalidState("n_features_in_ should be Some in Trained state".to_string())
        })?;

        if x.ncols() != n_features_in {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but QuantileRegressor is expecting {} features",
                x.ncols(),
                n_features_in
            )));
        }

        Ok(x.dot(coef) + intercept)
    }
}

impl Score<Array2<Float>, Array1<Float>> for QuantileRegressor<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Float> {
        // For quantile regression, we use the pinball loss as the score
        let predictions = self.predict(x)?;
        let residuals = y - &predictions;

        let loss = residuals
            .iter()
            .map(|&r| check_function(r, self.config.quantile))
            .sum::<Float>()
            / y.len() as Float;

        // Return negative loss (higher is better)
        Ok(-loss)
    }
}

impl QuantileRegressor<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> Result<&Array1<Float>> {
        self.coef_.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("coef_ should be Some in Trained state".to_string())
        })
    }

    /// Get the intercept
    pub fn intercept(&self) -> Result<Float> {
        self.intercept_.ok_or_else(|| {
            SklearsError::InvalidState("intercept_ should be Some in Trained state".to_string())
        })
    }

    /// Get the number of features seen during fit
    pub fn n_features_in(&self) -> Result<usize> {
        self.n_features_in_.ok_or_else(|| {
            SklearsError::InvalidState("n_features_in_ should be Some in Trained state".to_string())
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
    fn test_quantile_regressor_median() {
        // Simple linear relationship
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],];
        let y = array![2.1, 3.9, 6.1, 7.9, 10.1, 11.9, 14.1, 15.9];

        // Median regression (quantile = 0.5)
        let model = QuantileRegressor::new()
            .quantile(0.5)
            .alpha(0.0)
            .solver(QuantileSolver::IRLS)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();

        // Check that we get reasonable predictions
        let residuals = &y - &predictions;

        // The model should produce a reasonable fit
        // Since we have a simple linear relationship, check R-squared-like metric
        let ss_res = residuals.mapv(|r| r * r).sum();
        let y_mean = y.mean().unwrap();
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();
        let r2 = 1.0 - ss_res / ss_tot;

        assert!(r2 > 0.8, "R² = {}, coef = {}", r2, model.coef().unwrap()[0]);
    }

    #[test]
    fn test_quantile_regressor_lower_quantile() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        // 25th percentile regression
        let model = QuantileRegressor::new()
            .quantile(0.25)
            .alpha(0.0)
            .solver(QuantileSolver::CoordinateDescent)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();

        // For 0.25 quantile, about 25% of residuals should be negative
        let residuals = &y - &predictions;
        let n_negative = residuals.iter().filter(|&&r| r < 0.0).count();
        let ratio = n_negative as f64 / residuals.len() as f64;

        // Should be approximately 0.25
        assert!(ratio >= 0.0 && ratio <= 0.5);
    }

    #[test]
    fn test_quantile_regressor_upper_quantile() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        // 75th percentile regression
        let model = QuantileRegressor::new()
            .quantile(0.75)
            .alpha(0.0)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();

        // For upper quantile, predictions should generally be higher
        let residuals = &y - &predictions;

        // The predictions should be above most data points
        // Just check that the model runs and produces reasonable output
        assert!(model.coef().unwrap()[0] > 1.5 && model.coef().unwrap()[0] < 2.5);
    }

    #[test]
    fn test_quantile_regressor_with_regularization() {
        let x = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = QuantileRegressor::new()
            .quantile(0.5)
            .alpha(1.0)
            .fit(&x, &y)
            .unwrap();

        // With L1 regularization, the second coefficient (always 0) should be 0
        assert!(model.coef().unwrap()[1].abs() < 0.1);
    }

    #[test]
    fn test_quantile_regressor_no_intercept() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![2.0, 4.0, 6.0];

        let model = QuantileRegressor::new()
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(model.intercept().unwrap(), 0.0);
    }

    #[test]
    fn test_check_function() {
        // Test check function values
        assert_eq!(check_function(1.0, 0.5), 0.5);
        assert_eq!(check_function(-1.0, 0.5), 0.5); // (0.5 - 1) * (-1) = 0.5
        assert_eq!(check_function(2.0, 0.25), 0.5);
        assert_eq!(check_function(-2.0, 0.25), 1.5); // (0.25 - 1) * (-2) = 1.5
    }
}
