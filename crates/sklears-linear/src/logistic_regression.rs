//! Optimized Logistic Regression implementation using scirs2's optimization capabilities

use std::marker::PhantomData;

use scirs2_core::ndarray::{array, s, Array, Axis};
use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Score, Trained, Untrained},
    types::{Array1, Array2, Float},
};

use crate::{
    optimizer::{proximal, SagOptimizer, SagaOptimizer},
    Penalty, Solver,
};

/// Configuration for Logistic Regression
#[derive(Debug, Clone)]
pub struct LogisticRegressionConfig {
    /// Regularization penalty
    pub penalty: Penalty,
    /// Solver to use
    pub solver: Solver,
    /// Maximum iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Random state for stochastic solvers
    pub random_state: Option<u64>,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            penalty: Penalty::L2(1.0),
            solver: Solver::Auto,
            max_iter: 100,
            tol: 1e-4,
            fit_intercept: true,
            random_state: None,
        }
    }
}

/// Logistic Regression classifier
#[derive(Debug, Clone)]
pub struct LogisticRegression<State = Untrained> {
    config: LogisticRegressionConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_: Option<usize>,
    classes_: Option<Array1<Float>>,
}

impl LogisticRegression<Untrained> {
    /// Create a new Logistic Regression model
    pub fn new() -> Self {
        Self {
            config: LogisticRegressionConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            classes_: None,
        }
    }

    /// Set penalty
    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.config.penalty = penalty;
        self
    }

    /// Set solver
    pub fn solver(mut self, solver: Solver) -> Self {
        self.config.solver = solver;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl Default for LogisticRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LogisticRegression<Untrained> {
    type Config = LogisticRegressionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// Binary classification implementation
impl Fit<Array2<Float>, Array1<Float>> for LogisticRegression<Untrained> {
    type Fitted = LogisticRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        // Get unique classes (for now, assume binary: 0 and 1)
        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();

        if classes.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Logistic regression currently only supports binary classification".to_string(),
            ));
        }

        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Initialize parameters
        let n_params = if self.config.fit_intercept {
            n_features + 1
        } else {
            n_features
        };
        let initial_params = Array::zeros(n_params);

        // Clone data for closure capture
        let x_data = x.clone();
        let y_data = y.clone();

        // Choose solver based on configuration
        let params = match self.config.solver {
            Solver::Lbfgs | Solver::Auto => {
                // Use scirs2's L-BFGS for smooth objectives
                self.fit_lbfgs(&x_data, &y_data, n_samples, initial_params)?
            }
            Solver::Sag => {
                // Use our SAG implementation
                self.fit_sag(x, y, n_samples, n_params, initial_params)?
            }
            Solver::Saga => {
                // Use our SAGA implementation for non-smooth penalties
                self.fit_saga(x, y, n_samples, n_params, initial_params)?
            }
            _ => {
                // Fall back to simple gradient descent
                return Err(SklearsError::Other(format!(
                    "Solver {:?} not yet implemented for logistic regression",
                    self.config.solver
                )));
            }
        };

        // Extract final parameters
        let (coef_, intercept_) = if self.config.fit_intercept {
            (params.slice(s![1..]).to_owned(), Some(params[0]))
        } else {
            (params, None)
        };

        Ok(LogisticRegression {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef_),
            intercept_,
            n_features_: Some(n_features),
            classes_: Some(Array::from_vec(classes)),
        })
    }
}

// Solver implementations
impl LogisticRegression<Untrained> {
    /// Fit using scirs2's L-BFGS optimizer
    fn fit_lbfgs(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        n_samples: usize,
        initial_params: Array1<Float>,
    ) -> Result<Array1<Float>> {
        // For now, use our own L-BFGS implementation since scirs2's API is unclear
        use crate::optimizer::LbfgsOptimizer;

        let config = self.config.clone();
        let x_clone = x.clone();
        let y_clone = y.clone();

        // Define objective function
        let f = |params: &Array1<Float>| -> Float {
            let (w, b) = if config.fit_intercept {
                (params.slice(s![1..]).to_owned(), params[0])
            } else {
                (params.to_owned(), 0.0)
            };

            // Compute predictions
            let linear_pred = x_clone.dot(&w) + b;
            let predictions = linear_pred.mapv(|z| 1.0 / (1.0 + (-z).exp()));

            // Negative log-likelihood
            let epsilon = 1e-7;
            let mut loss = 0.0;
            for i in 0..n_samples {
                let p = predictions[i].clamp(epsilon, 1.0 - epsilon);
                loss -= y_clone[i] * p.ln() + (1.0 - y_clone[i]) * (1.0 - p).ln();
            }
            loss /= n_samples as f64;

            // Add regularization
            let reg_loss = match config.penalty {
                Penalty::None => 0.0,
                Penalty::L2(alpha) => alpha * w.mapv(|wi| wi * wi).sum() / (2.0 * n_samples as f64),
                _ => 0.0, // L1 and ElasticNet need special handling
            };
            loss + reg_loss
        };

        // Define gradient function
        let grad_f = |params: &Array1<Float>| -> Array1<Float> {
            let (w, b) = if config.fit_intercept {
                (params.slice(s![1..]).to_owned(), params[0])
            } else {
                (params.to_owned(), 0.0)
            };

            // Compute predictions
            let linear_pred = x_clone.dot(&w) + b;
            let predictions = linear_pred.mapv(|z| 1.0 / (1.0 + (-z).exp()));

            // Compute gradient
            let mut grad = Array::zeros(params.len());

            // Gradient of negative log-likelihood
            for i in 0..n_samples {
                let error = (predictions[i] - y_clone[i]) / n_samples as Float;
                if config.fit_intercept {
                    grad[0] += error;
                    let xi = x_clone.row(i);
                    for j in 0..xi.len() {
                        grad[j + 1] += error * xi[j];
                    }
                } else {
                    let xi = x_clone.row(i);
                    for j in 0..xi.len() {
                        grad[j] += error * xi[j];
                    }
                }
            }

            // Add regularization gradient
            if let Penalty::L2(alpha) = config.penalty {
                let start = if config.fit_intercept { 1 } else { 0 };
                for j in start..params.len() {
                    grad[j] += alpha * params[j] / n_samples as Float;
                }
            }

            grad
        };

        // Use our L-BFGS optimizer
        let optimizer = LbfgsOptimizer {
            max_iter: self.config.max_iter,
            tol: self.config.tol,
            ..Default::default()
        };

        optimizer
            .minimize(f, grad_f, initial_params)
            .map_err(|e| SklearsError::NumericalError(format!("L-BFGS optimization failed: {}", e)))
    }

    /// Fit using SAG optimizer
    fn fit_sag(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        n_samples: usize,
        n_params: usize,
        initial_params: Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Individual loss function
        let f_i = |params: &Array1<Float>, i: usize| -> Float {
            let (w, b) = if self.config.fit_intercept {
                (params.slice(s![1..]).to_owned(), params[0])
            } else {
                (params.clone(), 0.0)
            };

            let xi = x.row(i);
            let linear_pred = xi.dot(&w) + b;
            let p = 1.0 / (1.0 + (-linear_pred).exp());

            // Binary cross-entropy for sample i
            let epsilon = 1e-7;
            let p_clamped = p.clamp(epsilon, 1.0 - epsilon);
            let loss_i = -y[i] * p_clamped.ln() - (1.0 - y[i]) * (1.0 - p_clamped).ln();

            // Add regularization contribution
            match self.config.penalty {
                Penalty::L2(alpha) => {
                    loss_i + alpha * w.mapv(|wi| wi * wi).sum() / (2.0 * n_samples as Float)
                }
                _ => loss_i,
            }
        };

        // Gradient for individual sample
        let grad_f_i = |params: &Array1<Float>, i: usize| -> Array1<Float> {
            let (w, b) = if self.config.fit_intercept {
                (params.slice(s![1..]).to_owned(), params[0])
            } else {
                (params.clone(), 0.0)
            };

            let xi = x.row(i);
            let linear_pred = xi.dot(&w) + b;
            let p = 1.0 / (1.0 + (-linear_pred).exp());
            let error = p - y[i];

            let mut grad = Array::zeros(n_params);
            if self.config.fit_intercept {
                grad[0] = error;
                grad.slice_mut(s![1..]).assign(&(error * &xi));
            } else {
                grad.assign(&(error * &xi));
            }

            // Add L2 regularization gradient
            if let Penalty::L2(alpha) = self.config.penalty {
                let start = if self.config.fit_intercept { 1 } else { 0 };
                for j in start..n_params {
                    grad[j] += alpha * params[j] / n_samples as Float;
                }
            }

            grad
        };

        let optimizer = SagOptimizer {
            max_epochs: self.config.max_iter,
            tol: self.config.tol,
            random_state: self.config.random_state,
            ..Default::default()
        };

        optimizer
            .minimize(f_i, grad_f_i, initial_params, n_samples)
            .map_err(|e| SklearsError::NumericalError(format!("SAG failed: {}", e)))
    }

    /// Fit using SAGA optimizer (supports L1 penalty)
    fn fit_saga(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        n_samples: usize,
        n_params: usize,
        initial_params: Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Smooth part gradient (without penalty)
        let grad_f_i = |params: &Array1<Float>, i: usize| -> Array1<Float> {
            let (w, b) = if self.config.fit_intercept {
                (params.slice(s![1..]).to_owned(), params[0])
            } else {
                (params.clone(), 0.0)
            };

            let xi = x.row(i);
            let linear_pred = xi.dot(&w) + b;
            let p = 1.0 / (1.0 + (-linear_pred).exp());
            let error = p - y[i];

            let mut grad = Array::zeros(n_params);
            if self.config.fit_intercept {
                grad[0] = error;
                grad.slice_mut(s![1..]).assign(&(error * &xi));
            } else {
                grad.assign(&(error * &xi));
            }

            grad
        };

        // Dummy f_i (not used in SAGA but required by interface)
        let f_i = |_: &Array1<Float>, _: usize| -> Float { 0.0 };

        // Proximal operator based on penalty
        let prox_g = |z: &Array1<Float>, step_size: Float| -> Array1<Float> {
            match self.config.penalty {
                Penalty::L1(alpha) => {
                    // Apply soft thresholding, but not to intercept
                    let mut result = z.clone();
                    let start = if self.config.fit_intercept { 1 } else { 0 };
                    for i in start..n_params {
                        result[i] = proximal::prox_l1(
                            &array![z[i]],
                            step_size * alpha / n_samples as Float,
                        )[0];
                    }
                    result
                }
                Penalty::ElasticNet { l1_ratio, alpha } => {
                    // Apply elastic net proximal operator
                    let mut result = z.clone();
                    let start = if self.config.fit_intercept { 1 } else { 0 };
                    for i in start..n_params {
                        result[i] = proximal::prox_elastic_net(
                            &array![z[i]],
                            step_size * alpha / n_samples as Float,
                            l1_ratio,
                        )[0];
                    }
                    result
                }
                _ => z.clone(), // No proximal step for L2 or no penalty
            }
        };

        let optimizer = SagaOptimizer {
            max_epochs: self.config.max_iter,
            tol: self.config.tol,
            random_state: self.config.random_state,
            ..Default::default()
        };

        optimizer
            .minimize(f_i, grad_f_i, prox_g, initial_params, n_samples)
            .map_err(|e| SklearsError::NumericalError(format!("SAGA failed: {}", e)))
    }
}

impl LogisticRegression<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().expect("Model is trained")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Float> {
        self.classes_.as_ref().expect("Model is trained")
    }
}

impl Predict<Array2<Float>, Array1<Float>> for LogisticRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().expect("Model is trained");

        // Return class with highest probability
        Ok(probas.map_axis(Axis(1), |row| {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            classes[max_idx]
        }))
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for LogisticRegression<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let coef = self.coef_.as_ref().expect("Model is trained");
        let mut linear_pred = x.dot(coef);

        if let Some(intercept) = self.intercept_ {
            linear_pred += intercept;
        }

        // Apply sigmoid
        let proba_class1 = linear_pred.mapv(|z| 1.0 / (1.0 + (-z).exp()));

        // Return probabilities for both classes
        let n_samples = x.nrows();
        let mut probas = Array::zeros((n_samples, 2));
        probas.slice_mut(s![.., 0]).assign(&(1.0 - &proba_class1));
        probas.slice_mut(s![.., 1]).assign(&proba_class1);

        Ok(probas)
    }
}

impl Score<Array2<Float>, Array1<Float>> for LogisticRegression<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate accuracy
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_val)| (*pred - *true_val).abs() < 1e-6)
            .count();

        Ok(correct as f64 / y.len() as f64)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_logistic_regression_lbfgs() {
        // Simple linearly separable data
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
        ];
        let y = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let model = LogisticRegression::new()
            .solver(Solver::Lbfgs)
            .penalty(Penalty::L2(0.1))
            .fit(&x, &y)
            .unwrap();

        // Should achieve perfect classification on training data
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.99);

        // Check predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_logistic_regression_sag() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
        ];
        let y = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let model = LogisticRegression::new()
            .solver(Solver::Sag)
            .penalty(Penalty::L2(0.1))
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.95);
    }

    #[test]
    fn test_logistic_regression_saga_l1() {
        // Test with L1 penalty for sparsity
        let x = array![
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 3.0],
            [3.0, 0.0, 4.0],
            [-1.0, 0.0, -2.0],
            [-2.0, 0.0, -3.0],
            [-3.0, 0.0, -4.0],
        ];
        let y = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let model = LogisticRegression::new()
            .solver(Solver::Saga)
            .penalty(Penalty::L1(1.0))
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        let coef = model.coef();

        // Middle feature should be zero due to L1 penalty
        assert_abs_diff_eq!(coef[1], 0.0, epsilon = 0.01);
    }
}
