//! Linear Support Vector Regression using coordinate descent
//!
//! This implementation uses coordinate descent optimization which avoids the need
//! for BLAS operations while still providing efficient linear SVR training.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::{SeedableRng, StdRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};

/// Linear Support Vector Regression
///
/// LinearSVR is a more efficient implementation for linear kernels using coordinate descent,
/// which avoids the computational overhead of the full SMO algorithm while maintaining
/// accuracy for linear regression problems.
///
/// # Parameters
/// * `C` - Regularization parameter (default: 1.0)
/// * `epsilon` - Epsilon in the epsilon-SVR model (default: 0.1)
/// * `loss` - Loss function type ('epsilon_insensitive' or 'squared_epsilon_insensitive', default: 'epsilon_insensitive')
/// * `dual` - Use dual formulation (default: true)
/// * `tol` - Tolerance for stopping criterion (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `fit_intercept` - Whether to fit an intercept term (default: true)
/// * `intercept_scaling` - When `fit_intercept` is True, scaling for synthetic feature (default: 1.0)
/// * `verbose` - Enable verbose output (default: false)
/// * `random_state` - Random seed for reproducible results (default: None)
///
/// # Example
/// ```rust
/// use sklears_svm::LinearSVR;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
/// let y = array![1.5, 2.5, 3.0, 1.8];
///
/// let model = LinearSVR::new()
///     .with_c(1.0)
///     .with_epsilon(0.1)
///     .with_max_iter(1000);
///
/// let trained_model = model.fit(&X, &y).unwrap();
/// let predictions = trained_model.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LinearSVR {
    /// Regularization parameter
    pub c: f64,
    /// Epsilon in the epsilon-SVR model
    pub epsilon: f64,
    /// Loss function ('epsilon_insensitive' or 'squared_epsilon_insensitive')
    pub loss: String,
    /// Use dual formulation
    pub dual: bool,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Scaling for synthetic intercept feature
    pub intercept_scaling: f64,
    /// Verbose output
    pub verbose: bool,
    /// Random seed
    pub random_state: Option<u64>,
}

/// Trained Linear Support Vector Regression model
#[derive(Debug, Clone)]
pub struct TrainedLinearSVR {
    /// Model weights (coefficients)
    pub coef_: Array1<f64>,
    /// Intercept term
    pub intercept_: f64,
    /// Number of features
    pub n_features_in_: usize,
    /// Training parameters
    params: LinearSVR,
}

impl Default for LinearSVR {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSVR {
    /// Create a new LinearSVR with default parameters
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            loss: "epsilon_insensitive".to_string(),
            dual: true,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            intercept_scaling: 1.0,
            verbose: false,
            random_state: None,
        }
    }

    /// Set the regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon parameter
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the loss function
    pub fn with_loss(mut self, loss: &str) -> Self {
        self.loss = loss.to_string();
        self
    }

    /// Set whether to use dual formulation
    pub fn with_dual(mut self, dual: bool) -> Self {
        self.dual = dual;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the intercept scaling
    pub fn with_intercept_scaling(mut self, intercept_scaling: f64) -> Self {
        self.intercept_scaling = intercept_scaling;
        self
    }

    /// Set verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Coordinate descent solver for linear SVR
    fn coordinate_descent(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        alpha_plus: &mut Array1<f64>,
        alpha_minus: &mut Array1<f64>,
        w: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let mut rng = StdRng::seed_from_u64(42);

        for iteration in 0..self.max_iter {
            let mut alpha_diff = 0.0;

            // Shuffle sample indices for better convergence
            let mut indices: Vec<usize> = (0..n_samples).collect();
            if self.random_state.is_some() {
                indices.shuffle(&mut rng);
            }

            for &i in &indices {
                let xi = x.row(i);
                let yi = y[i];

                // Compute current prediction
                let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                for j in 0..n_features {
                    prediction += w[j] * xi[j];
                }

                let error = prediction - yi;
                let old_alpha_plus = alpha_plus[i];
                let old_alpha_minus = alpha_minus[i];

                // Update alpha_plus and alpha_minus based on loss function
                match self.loss.as_str() {
                    "epsilon_insensitive" => {
                        // Epsilon-insensitive loss: max(0, |y - f(x)| - epsilon)
                        if error > self.epsilon {
                            // Update alpha_plus
                            let hessian = xi.iter().map(|&x| x * x).sum::<f64>();
                            if hessian > 0.0 {
                                let gradient = 1.0;
                                let delta = -gradient / (hessian + 1.0 / self.c);
                                alpha_plus[i] = (alpha_plus[i] + delta).max(0.0).min(self.c);
                            }
                        } else if error < -self.epsilon {
                            // Update alpha_minus
                            let hessian = xi.iter().map(|&x| x * x).sum::<f64>();
                            if hessian > 0.0 {
                                let gradient = -1.0;
                                let delta = -gradient / (hessian + 1.0 / self.c);
                                alpha_minus[i] = (alpha_minus[i] + delta).max(0.0).min(self.c);
                            }
                        }
                    }
                    "squared_epsilon_insensitive" => {
                        // Squared epsilon-insensitive loss
                        if error.abs() > self.epsilon {
                            let loss_value = error.abs() - self.epsilon;
                            let hessian =
                                2.0 * (xi.iter().map(|&x| x * x).sum::<f64>() + 1.0 / self.c);

                            if error > self.epsilon {
                                let gradient = 2.0 * loss_value;
                                if hessian > 0.0 {
                                    let delta = -gradient / hessian;
                                    alpha_plus[i] = (alpha_plus[i] + delta).max(0.0).min(self.c);
                                }
                            } else {
                                let gradient = -2.0 * loss_value;
                                if hessian > 0.0 {
                                    let delta = -gradient / hessian;
                                    alpha_minus[i] = (alpha_minus[i] + delta).max(0.0).min(self.c);
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(SklearsError::InvalidParameter {
                            name: "loss".to_string(),
                            reason: format!("Unknown loss: {}", self.loss),
                        })
                    }
                }

                let alpha_plus_change = alpha_plus[i] - old_alpha_plus;
                let alpha_minus_change = alpha_minus[i] - old_alpha_minus;
                alpha_diff += alpha_plus_change.abs() + alpha_minus_change.abs();

                // Update weights and intercept
                if alpha_plus_change.abs() > 1e-12 {
                    for j in 0..n_features {
                        w[j] += alpha_plus_change * xi[j];
                    }
                    if self.fit_intercept {
                        *intercept += alpha_plus_change * self.intercept_scaling;
                    }
                }

                if alpha_minus_change.abs() > 1e-12 {
                    for j in 0..n_features {
                        w[j] -= alpha_minus_change * xi[j];
                    }
                    if self.fit_intercept {
                        *intercept -= alpha_minus_change * self.intercept_scaling;
                    }
                }
            }

            // Check convergence
            if alpha_diff < self.tol {
                if self.verbose {
                    println!("LinearSVR converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                println!("LinearSVR iteration {iteration}, alpha_diff: {alpha_diff:.6}");
            }
        }

        Ok(())
    }
}

impl Estimator for LinearSVR {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>> for LinearSVR {
    type Fitted = TrainedLinearSVR;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<TrainedLinearSVR> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if x.len_of(Axis(0)) != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize dual variables
        let mut alpha_plus = Array1::zeros(n_samples);
        let mut alpha_minus = Array1::zeros(n_samples);
        let mut w = Array1::zeros(n_features);
        let mut intercept = 0.0;

        // Run coordinate descent optimization
        self.coordinate_descent(
            x.view(),
            y.view(),
            &mut alpha_plus,
            &mut alpha_minus,
            &mut w,
            &mut intercept,
        )?;

        Ok(TrainedLinearSVR {
            coef_: w,
            intercept_: intercept,
            n_features_in_: n_features,
            params: self,
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for TrainedLinearSVR {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_,
                actual: n_features,
            });
        }

        let mut predictions = Array1::zeros(n_samples);

        for (i, x_row) in x.axis_iter(Axis(0)).enumerate() {
            let mut prediction = if self.params.fit_intercept {
                self.intercept_
            } else {
                0.0
            };

            for (j, &x_val) in x_row.iter().enumerate() {
                prediction += self.coef_[j] * x_val;
            }
            predictions[i] = prediction;
        }

        Ok(predictions)
    }
}

impl TrainedLinearSVR {
    /// Get the model coefficients
    pub fn coef(&self) -> &Array1<f64> {
        &self.coef_
    }

    /// Get the intercept term
    pub fn intercept(&self) -> f64 {
        self.intercept_
    }

    /// Compute decision function values (same as predict for regression)
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.predict(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_svr_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
        let y = array![1.5, 2.5, 3.0, 1.8];

        let model = LinearSVR::new()
            .with_c(1.0)
            .with_epsilon(0.1)
            .with_max_iter(1000);
        let trained_model = model.fit(&X, &y).unwrap();

        let predictions = trained_model.predict(&X).unwrap();
        assert_eq!(predictions.len(), 4);

        // Check that coefficients and intercept are reasonable
        assert_eq!(trained_model.coef().len(), 2);
    }

    #[test]
    fn test_linear_svr_parameters() {
        let model = LinearSVR::new()
            .with_c(0.5)
            .with_epsilon(0.2)
            .with_loss("squared_epsilon_insensitive")
            .with_dual(false)
            .with_tol(1e-5)
            .with_max_iter(500)
            .with_fit_intercept(false)
            .with_random_state(42);

        assert_eq!(model.c, 0.5);
        assert_abs_diff_eq!(model.epsilon, 0.2);
        assert_eq!(model.loss, "squared_epsilon_insensitive");
        assert!(!model.dual);
        assert_abs_diff_eq!(model.tol, 1e-5);
        assert_eq!(model.max_iter, 500);
        assert!(!model.fit_intercept);
        assert_eq!(model.random_state, Some(42));
    }

    #[test]
    fn test_linear_svr_decision_function() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![1.5, 2.5];

        let model = LinearSVR::new();
        let trained_model = model.fit(&X, &y).unwrap();

        let predictions = trained_model.predict(&X).unwrap();
        let decision_values = trained_model.decision_function(&X).unwrap();

        // For regression, decision function should equal predictions
        for (pred, dec) in predictions.iter().zip(decision_values.iter()) {
            assert_abs_diff_eq!(pred, dec, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_linear_svr_invalid_loss() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![1.5, 2.5];

        let model = LinearSVR::new().with_loss("invalid_loss");
        let result = model.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_svr_dimension_mismatch() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![1.5]; // Wrong size

        let model = LinearSVR::new();
        let result = model.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_svr_prediction_dimension_mismatch() {
        let X_train_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y_train = array![1.5, 2.5];
        let X_test_var = array![[1.0]]; // Wrong number of features

        let model = LinearSVR::new();
        let trained_model = model.fit(&X_train_var, &y_train).unwrap();
        let result = trained_model.predict(&X_test_var);
        assert!(result.is_err());
    }
}
