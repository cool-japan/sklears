//! Linear Regression implementation

use std::marker::PhantomData;

use scirs2_core::ndarray::{s, Array};
use scirs2_linalg::solve;
use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Array1, Array2, Float},
};

use crate::{Penalty, Solver};

#[cfg(feature = "coordinate-descent")]
use crate::coordinate_descent::CoordinateDescentSolver;

#[cfg(feature = "coordinate-descent")]
use crate::coordinate_descent::ValidationInfo;

#[cfg(feature = "early-stopping")]
use crate::early_stopping::EarlyStoppingConfig;

/// Configuration for Linear Regression
#[derive(Debug, Clone)]
pub struct LinearRegressionConfig {
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Regularization penalty
    pub penalty: Penalty,
    /// Solver to use
    pub solver: Solver,
    /// Maximum iterations for iterative solvers
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Whether to use warm start (reuse previous solution as initialization)
    pub warm_start: bool,
    /// Enable GPU acceleration if available
    #[cfg(feature = "gpu")]
    pub use_gpu: bool,
    /// Minimum problem size to use GPU acceleration
    #[cfg(feature = "gpu")]
    pub gpu_min_size: usize,
}

impl Default for LinearRegressionConfig {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            penalty: Penalty::None,
            solver: Solver::Auto,
            max_iter: 1000,
            tol: 1e-4,
            warm_start: false,
            #[cfg(feature = "gpu")]
            use_gpu: true,
            #[cfg(feature = "gpu")]
            gpu_min_size: 1000,
        }
    }
}

/// Linear Regression model
#[derive(Debug, Clone)]
pub struct LinearRegression<State = Untrained> {
    config: LinearRegressionConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_: Option<usize>,
}

impl LinearRegression<Untrained> {
    /// Create a new Linear Regression model
    pub fn new() -> Self {
        Self {
            config: LinearRegressionConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
        }
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization (Ridge/L2)
    pub fn regularization(mut self, alpha: f64) -> Self {
        self.config.penalty = Penalty::L2(alpha);
        self
    }

    /// Create a Lasso regression model (L1 penalty)
    pub fn lasso(alpha: f64) -> Self {
        Self::new()
            .penalty(Penalty::L1(alpha))
            .solver(Solver::CoordinateDescent)
    }

    /// Create an ElasticNet regression model (L1 + L2 penalty)
    pub fn elastic_net(alpha: f64, l1_ratio: f64) -> Self {
        Self::new()
            .penalty(Penalty::ElasticNet { l1_ratio, alpha })
            .solver(Solver::CoordinateDescent)
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

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Enable or disable GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Set minimum problem size for GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn gpu_min_size(mut self, min_size: usize) -> Self {
        self.config.gpu_min_size = min_size;
        self
    }
}

impl Default for LinearRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LinearRegression<Untrained> {
    type Config = LinearRegressionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for LinearRegression<Untrained> {
    type Fitted = LinearRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Add intercept column if needed
        let (x_with_intercept, n_params) = if self.config.fit_intercept {
            let mut x_new = Array::ones((n_samples, n_features + 1));
            x_new.slice_mut(s![.., 1..]).assign(x);
            (x_new, n_features + 1)
        } else {
            (x.clone(), n_features)
        };

        // Solve based on penalty type
        let params = match self.config.penalty {
            Penalty::None => {
                // Check if we should use GPU acceleration
                #[cfg(feature = "gpu")]
                if self.config.use_gpu && n_samples * n_features >= self.config.gpu_min_size {
                    // Try GPU-accelerated OLS
                    match self.solve_ols_gpu(&x_with_intercept, y) {
                        Ok(params) => params,
                        Err(_) => {
                            // Fallback to CPU if GPU fails
                            self.solve_ols_cpu(&x_with_intercept, y)?
                        }
                    }
                } else {
                    self.solve_ols_cpu(&x_with_intercept, y)?
                }

                #[cfg(not(feature = "gpu"))]
                self.solve_ols_cpu(&x_with_intercept, y)?
            }
            Penalty::L2(alpha) => {
                // Ridge regression
                // (X^T X + αI) β = X^T y
                let xtx = x_with_intercept.t().dot(&x_with_intercept);
                let xty = x_with_intercept.t().dot(y);

                // Add regularization to diagonal (except intercept if present)
                let mut regularized = xtx.clone();
                let start_idx = if self.config.fit_intercept { 1 } else { 0 };
                for i in start_idx..n_params {
                    regularized[[i, i]] += alpha;
                }

                solve(&regularized.view(), &xty.view(), None).map_err(|e| {
                    SklearsError::NumericalError(format!("Failed to solve ridge regression: {}", e))
                })?
            }
            Penalty::L1(alpha) => {
                // Lasso regression using coordinate descent
                #[cfg(feature = "coordinate-descent")]
                {
                    let cd_solver = CoordinateDescentSolver {
                        max_iter: self.config.max_iter,
                        tol: self.config.tol,
                        cyclic: true,
                        #[cfg(feature = "early-stopping")]
                        early_stopping_config: None,
                    };

                    let (coef, intercept) = cd_solver
                        .solve_lasso(x, y, alpha, self.config.fit_intercept)
                        .map_err(|e| {
                            SklearsError::NumericalError(format!(
                                "Coordinate descent failed: {}",
                                e
                            ))
                        })?;

                    if self.config.fit_intercept {
                        // Need to add intercept to beginning of params for consistency
                        let mut params = Array::zeros(coef.len() + 1);
                        params[0] = intercept.unwrap_or(0.0);
                        params.slice_mut(s![1..]).assign(&coef);
                        params
                    } else {
                        coef
                    }
                }
                #[cfg(not(feature = "coordinate-descent"))]
                {
                    return Err(SklearsError::InvalidParameter {
                        name: "penalty".to_string(),
                        reason:
                            "L1 regularization (Lasso) requires the 'coordinate-descent' feature"
                                .to_string(),
                    });
                }
            }
            Penalty::ElasticNet { l1_ratio, alpha } => {
                // ElasticNet regression using coordinate descent
                #[cfg(feature = "coordinate-descent")]
                {
                    let cd_solver = CoordinateDescentSolver {
                        max_iter: self.config.max_iter,
                        tol: self.config.tol,
                        cyclic: true,
                        #[cfg(feature = "early-stopping")]
                        early_stopping_config: None,
                    };

                    let (coef, intercept) = cd_solver
                        .solve_elastic_net(x, y, alpha, l1_ratio, self.config.fit_intercept)
                        .map_err(|e| {
                            SklearsError::NumericalError(format!(
                                "Coordinate descent failed: {}",
                                e
                            ))
                        })?;

                    if self.config.fit_intercept {
                        // Need to add intercept to beginning of params for consistency
                        let mut params = Array::zeros(coef.len() + 1);
                        params[0] = intercept.unwrap_or(0.0);
                        params.slice_mut(s![1..]).assign(&coef);
                        params
                    } else {
                        coef
                    }
                }
                #[cfg(not(feature = "coordinate-descent"))]
                {
                    return Err(SklearsError::InvalidParameter {
                        name: "penalty".to_string(),
                        reason:
                            "ElasticNet regularization requires the 'coordinate-descent' feature"
                                .to_string(),
                    });
                }
            }
        };

        // Extract coefficients and intercept
        let (coef_, intercept_) = if self.config.fit_intercept {
            let intercept = params[0];
            let coef = params.slice(s![1..]).to_owned();
            (coef, Some(intercept))
        } else {
            (params, None)
        };

        Ok(LinearRegression {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef_),
            intercept_,
            n_features_: Some(n_features),
        })
    }
}

impl LinearRegression<Untrained> {
    /// CPU-based OLS solver
    fn solve_ols_cpu(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        // Ordinary Least Squares using scirs2
        // X^T X β = X^T y
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);

        // Use scirs2's linear solver
        solve(&xtx.view(), &xty.view(), None).map_err(|e| {
            SklearsError::NumericalError(format!("Failed to solve linear system: {}", e))
        })
    }

    /// GPU-based OLS solver
    #[cfg(feature = "gpu")]
    fn solve_ols_gpu(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        use crate::gpu_acceleration::{GpuConfig, GpuLinearOps};

        // Initialize GPU operations
        let gpu_config = GpuConfig {
            device_id: 0,
            use_pinned_memory: true,
            min_problem_size: self.config.gpu_min_size,
            ..Default::default()
        };

        let gpu_ops = GpuLinearOps::new(gpu_config).map_err(|e| {
            SklearsError::NumericalError(format!("Failed to initialize GPU operations: {}", e))
        })?;

        // Check if GPU is available
        if !gpu_ops.is_gpu_available() {
            return Err(SklearsError::NumericalError(
                "GPU not available, falling back to CPU".to_string(),
            ));
        }

        // Compute X^T X using GPU
        let xt = gpu_ops.matrix_transpose(x)?;
        let xtx = gpu_ops.matrix_multiply(&xt, x)?;

        // Compute X^T y using GPU
        let xty = gpu_ops.matrix_vector_multiply(&xt, y)?;

        // Solve linear system using GPU
        gpu_ops.solve_linear_system(&xtx, &xty)
    }

    /// Fit the linear regression model with warm start
    ///
    /// Uses the provided coefficients and intercept as initialization for iterative solvers
    pub fn fit_with_warm_start(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        initial_coef: Option<&Array1<Float>>,
        initial_intercept: Option<Float>,
    ) -> Result<LinearRegression<Trained>> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();

        // For warm start, we only support ElasticNet/Lasso methods (coordinate descent)
        let params: Array1<Float> = match self.config.penalty {
            Penalty::L1(_)
            | Penalty::L2(_)
            | Penalty::ElasticNet {
                alpha: _,
                l1_ratio: _,
            } => {
                #[cfg(feature = "coordinate-descent")]
                {
                    let (alpha_val, l1_ratio) = match self.config.penalty {
                        Penalty::L1(alpha) => (alpha, 1.0),
                        Penalty::L2(alpha) => (alpha, 0.0),
                        Penalty::ElasticNet { alpha, l1_ratio } => (alpha, l1_ratio),
                        _ => unreachable!(),
                    };

                    let cd_solver = CoordinateDescentSolver {
                        max_iter: self.config.max_iter,
                        tol: self.config.tol,
                        cyclic: true,
                        #[cfg(feature = "early-stopping")]
                        early_stopping_config: None,
                    };

                    let (coef, intercept) = cd_solver
                        .solve_elastic_net_with_warm_start(
                            x,
                            y,
                            alpha_val,
                            l1_ratio,
                            self.config.fit_intercept,
                            initial_coef,
                            initial_intercept,
                        )
                        .map_err(|e| {
                            SklearsError::NumericalError(format!(
                                "Coordinate descent failed: {}",
                                e
                            ))
                        })?;

                    if self.config.fit_intercept {
                        // Need to add intercept to beginning of params for consistency
                        let mut params = Array::zeros(coef.len() + 1);
                        params[0] = intercept.unwrap_or(0.0);
                        params.slice_mut(s![1..]).assign(&coef);
                        params
                    } else {
                        coef
                    }
                }
                #[cfg(not(feature = "coordinate-descent"))]
                {
                    return Err(SklearsError::InvalidParameter {
                        name: "penalty".to_string(),
                        reason: "Warm start requires the 'coordinate-descent' feature".to_string(),
                    });
                }
            }
            Penalty::None => {
                return Err(SklearsError::InvalidParameter {
                    name: "penalty".to_string(),
                    reason:
                        "Warm start only supported for regularized methods (L1, L2, ElasticNet)"
                            .to_string(),
                });
            }
        };

        // Extract coefficients and intercept
        let (coef_, intercept_) = if self.config.fit_intercept {
            let intercept = params[0];
            let coef = params.slice(s![1..]).to_owned();
            (coef, Some(intercept))
        } else {
            (params, None)
        };

        Ok(LinearRegression {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef_),
            intercept_,
            n_features_: Some(n_features),
        })
    }
}

impl LinearRegression<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().expect("Model is trained")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }
}

impl Predict<Array2<Float>, Array1<Float>> for LinearRegression<Trained> {
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

impl Score<Array2<Float>, Array1<Float>> for LinearRegression<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate R² score using scirs2 metrics
        let ss_res = (&predictions - y).mapv(|x| x * x).sum();
        let y_mean = y.mean().unwrap_or(0.0);
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            return Ok(1.0);
        }

        Ok(1.0 - (ss_res / ss_tot))
    }
}

impl LinearRegression<Untrained> {
    /// Fit the linear regression model with early stopping based on validation metrics
    ///
    /// This method is particularly useful for regularized methods (Lasso, ElasticNet)
    /// where early stopping can prevent overfitting.
    #[cfg(all(feature = "coordinate-descent", feature = "early-stopping"))]
    pub fn fit_with_early_stopping(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        early_stopping_config: EarlyStoppingConfig,
    ) -> Result<(LinearRegression<Trained>, ValidationInfo)> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();

        // Early stopping is most beneficial for regularized methods
        match self.config.penalty {
            Penalty::L1(alpha) => {
                let cd_solver = CoordinateDescentSolver {
                    max_iter: self.config.max_iter,
                    tol: self.config.tol,
                    cyclic: true,
                    early_stopping_config: Some(early_stopping_config),
                };

                let (coef, intercept, validation_info) = cd_solver
                    .solve_lasso_with_early_stopping(x, y, alpha, self.config.fit_intercept)?;

                let intercept_ = if self.config.fit_intercept {
                    intercept
                } else {
                    None
                };

                let fitted_model = LinearRegression {
                    config: self.config,
                    state: PhantomData,
                    coef_: Some(coef),
                    intercept_,
                    n_features_: Some(n_features),
                };

                Ok((fitted_model, validation_info))
            }
            Penalty::ElasticNet { l1_ratio, alpha } => {
                let cd_solver = CoordinateDescentSolver {
                    max_iter: self.config.max_iter,
                    tol: self.config.tol,
                    cyclic: true,
                    early_stopping_config: Some(early_stopping_config),
                };

                let (coef, intercept, validation_info) = cd_solver
                    .solve_elastic_net_with_early_stopping(
                        x,
                        y,
                        alpha,
                        l1_ratio,
                        self.config.fit_intercept,
                    )?;

                let intercept_ = if self.config.fit_intercept {
                    intercept
                } else {
                    None
                };

                let fitted_model = LinearRegression {
                    config: self.config,
                    state: PhantomData,
                    coef_: Some(coef),
                    intercept_,
                    n_features_: Some(n_features),
                };

                Ok((fitted_model, validation_info))
            }
            Penalty::L2(_alpha) => {
                // For Ridge regression, we can use iterative solver with early stopping
                // For now, fall back to regular fit and provide minimal validation info
                let fitted_model = self.fit(x, y)?;
                let validation_info = ValidationInfo {
                    validation_scores: vec![1.0], // Dummy score
                    best_score: Some(1.0),
                    best_iteration: 1,
                    stopped_early: false,
                    converged: true,
                };
                Ok((fitted_model, validation_info))
            }
            Penalty::None => {
                // For OLS, early stopping doesn't make much sense since it's a direct solution
                let fitted_model = self.fit(x, y)?;
                let validation_info = ValidationInfo {
                    validation_scores: vec![1.0], // Dummy score
                    best_score: Some(1.0),
                    best_iteration: 1,
                    stopped_early: false,
                    converged: true,
                };
                Ok((fitted_model, validation_info))
            }
        }
    }

    /// Fit the linear regression model with early stopping using pre-split validation data
    ///
    /// This gives you more control over the train/validation split compared to
    /// `fit_with_early_stopping` which automatically splits the data.
    #[cfg(all(feature = "coordinate-descent", feature = "early-stopping"))]
    pub fn fit_with_early_stopping_split(
        self,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
        x_val: &Array2<Float>,
        y_val: &Array1<Float>,
        early_stopping_config: EarlyStoppingConfig,
    ) -> Result<(LinearRegression<Trained>, ValidationInfo)> {
        // Validate inputs
        validate::check_consistent_length(x_train, y_train)?;
        validate::check_consistent_length(x_val, y_val)?;

        let n_features = x_train.ncols();
        if x_val.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x_val.ncols(),
            });
        }

        // Early stopping is most beneficial for regularized methods
        match self.config.penalty {
            Penalty::L1(alpha) => {
                let cd_solver = CoordinateDescentSolver {
                    max_iter: self.config.max_iter,
                    tol: self.config.tol,
                    cyclic: true,
                    early_stopping_config: Some(early_stopping_config),
                };

                let (coef, intercept, validation_info) = cd_solver
                    .solve_lasso_with_early_stopping_split(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        alpha,
                        self.config.fit_intercept,
                    )?;

                let intercept_ = if self.config.fit_intercept {
                    intercept
                } else {
                    None
                };

                let fitted_model = LinearRegression {
                    config: self.config,
                    state: PhantomData,
                    coef_: Some(coef),
                    intercept_,
                    n_features_: Some(n_features),
                };

                Ok((fitted_model, validation_info))
            }
            Penalty::ElasticNet { l1_ratio, alpha } => {
                let cd_solver = CoordinateDescentSolver {
                    max_iter: self.config.max_iter,
                    tol: self.config.tol,
                    cyclic: true,
                    early_stopping_config: Some(early_stopping_config),
                };

                let (coef, intercept, validation_info) = cd_solver
                    .solve_elastic_net_with_early_stopping_split(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        alpha,
                        l1_ratio,
                        self.config.fit_intercept,
                    )?;

                let intercept_ = if self.config.fit_intercept {
                    intercept
                } else {
                    None
                };

                let fitted_model = LinearRegression {
                    config: self.config,
                    state: PhantomData,
                    coef_: Some(coef),
                    intercept_,
                    n_features_: Some(n_features),
                };

                Ok((fitted_model, validation_info))
            }
            Penalty::L2(_alpha) => {
                // For Ridge regression, compute validation score manually
                let fitted_model = LinearRegression::new()
                    .penalty(self.config.penalty)
                    .fit_intercept(self.config.fit_intercept)
                    .fit(x_train, y_train)?;

                // Compute validation R² score
                let val_predictions = fitted_model.predict(x_val)?;
                let r2_score = crate::coordinate_descent::compute_r2_score(&val_predictions, y_val);

                let validation_info = ValidationInfo {
                    validation_scores: vec![r2_score],
                    best_score: Some(r2_score),
                    best_iteration: 1,
                    stopped_early: false,
                    converged: true,
                };

                Ok((fitted_model, validation_info))
            }
            Penalty::None => {
                // For OLS, compute validation score manually
                let fitted_model = LinearRegression::new()
                    .fit_intercept(self.config.fit_intercept)
                    .fit(x_train, y_train)?;

                // Compute validation R² score
                let val_predictions = fitted_model.predict(x_val)?;
                let r2_score = crate::coordinate_descent::compute_r2_score(&val_predictions, y_val);

                let validation_info = ValidationInfo {
                    validation_scores: vec![r2_score],
                    best_score: Some(r2_score),
                    best_iteration: 1,
                    stopped_early: false,
                    converged: true,
                };

                Ok((fitted_model, validation_info))
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_regression_simple() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = LinearRegression::new()
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 1e-10);

        let predictions = model.predict(&array![[5.0]]).unwrap();
        assert_abs_diff_eq!(predictions[0], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_regression_with_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0]; // y = 2x + 1

        let model = LinearRegression::new()
            .fit_intercept(true)
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(model.intercept().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ridge_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = LinearRegression::new()
            .fit_intercept(false)
            .regularization(0.1)
            .fit(&x, &y)
            .unwrap();

        // With regularization, coefficient should be slightly less than 2.0
        assert!(model.coef()[0] < 2.0);
        assert!(model.coef()[0] > 1.9);
    }

    #[test]
    fn test_lasso_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        // Test with small alpha
        let model = LinearRegression::lasso(0.01)
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        // Should be close to OLS solution (coef = 2.0)
        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 0.1);

        // Test with larger alpha
        let model = LinearRegression::lasso(0.5)
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        // Coefficient should be shrunk
        assert!(model.coef()[0] < 2.0);
        assert!(model.coef()[0] > 1.0);
    }

    #[test]
    fn test_elastic_net_regression() {
        let x = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let y = array![3.0, 6.0, 9.0, 12.0]; // y = 2*x1 + 2*x2

        let model = LinearRegression::elastic_net(0.1, 0.5)
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        // Both coefficients should be shrunk but non-zero
        println!(
            "ElasticNet coef[0] = {}, coef[1] = {}",
            model.coef()[0],
            model.coef()[1]
        );
        assert!(model.coef()[0] > 0.0);
        assert!(model.coef()[0] < 3.0); // More lenient bound for weak regularization
        assert!(model.coef()[1] > 0.0);
        assert!(model.coef()[1] < 3.0); // More lenient bound for weak regularization
    }

    #[test]
    fn test_lasso_sparsity() {
        // Create data where only first feature is relevant
        let n_samples = 20;
        let mut x = Array2::zeros((n_samples, 5));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            x[[i, 0]] = i as f64;
            x[[i, 1]] = (i as f64) * 0.1; // weak feature
                                          // Add deterministic noise instead of random
            x[[i, 2]] = ((i * 7) % 10) as f64 / 10.0; // pseudo-random noise
            x[[i, 3]] = ((i * 13) % 10) as f64 / 10.0; // pseudo-random noise
            x[[i, 4]] = ((i * 17) % 10) as f64 / 10.0; // pseudo-random noise
            y[i] = 2.0 * x[[i, 0]] + 0.05 * (i % 3) as f64;
        }

        // With strong L1 penalty, should select only the first feature
        let model = LinearRegression::lasso(1.0)
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        let coef = model.coef();

        // First coefficient should be non-zero
        assert!(coef[0] > 0.5);

        // Other coefficients should be zero or very small
        for i in 2..5 {
            assert_abs_diff_eq!(coef[i], 0.0, epsilon = 0.01);
        }
    }

    #[test]
    #[cfg(all(feature = "coordinate-descent", feature = "early-stopping"))]
    fn test_linear_regression_early_stopping_lasso() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        // Create larger dataset for meaningful validation split
        let n_samples = 100;
        let n_features = 8;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        // Generate synthetic data with linear relationship
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = (i * j + 1) as f64 / 20.0;
            }
            // Only first few features are relevant
            y[i] = 2.0 * x[[i, 0]] + 1.5 * x[[i, 1]] + 0.8 * x[[i, 2]] + 0.1 * (i as f64 % 5.0);
        }

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(10),
            validation_split: 0.25,
            shuffle: true,
            random_state: Some(42),
            higher_is_better: true,
            min_iterations: 5,
            restore_best_weights: true,
        };

        let model = LinearRegression::lasso(0.1);
        let result = model.fit_with_early_stopping(&x, &y, early_stopping_config);

        assert!(result.is_ok());
        let (fitted_model, validation_info) = result.unwrap();

        // Check model properties
        assert_eq!(fitted_model.coef().len(), n_features);
        assert!(fitted_model.intercept().is_some());

        // Check validation info
        assert!(!validation_info.validation_scores.is_empty());
        assert!(validation_info.best_score.is_some());
        assert!(validation_info.best_iteration >= 1);

        // Predictions should work
        let predictions = fitted_model.predict(&x);
        assert!(predictions.is_ok());
        assert_eq!(predictions.unwrap().len(), n_samples);
    }

    #[test]
    #[cfg(all(feature = "coordinate-descent", feature = "early-stopping"))]
    fn test_linear_regression_early_stopping_elastic_net() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.0],
            [5.0, 6.0, 2.5],
            [6.0, 7.0, 3.0],
            [7.0, 8.0, 3.5],
            [8.0, 9.0, 4.0]
        ];
        let y = array![4.5, 7.0, 9.5, 12.0, 14.5, 17.0, 19.5, 22.0]; // y ≈ 1.5*x1 + x2 + x3

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::TolerancePatience {
                tolerance: 0.005,
                patience: 3,
            },
            validation_split: 0.25,
            shuffle: false,
            random_state: Some(123),
            higher_is_better: true,
            min_iterations: 2,
            restore_best_weights: true,
        };

        let model = LinearRegression::elastic_net(0.1, 0.7);
        let result = model.fit_with_early_stopping(&x, &y, early_stopping_config);

        assert!(result.is_ok());
        let (fitted_model, validation_info) = result.unwrap();

        assert_eq!(fitted_model.coef().len(), 3);
        assert!(fitted_model.intercept().is_some());
        assert!(!validation_info.validation_scores.is_empty());
        assert!(validation_info.best_score.is_some());
    }

    #[test]
    #[cfg(all(feature = "coordinate-descent", feature = "early-stopping"))]
    fn test_linear_regression_early_stopping_with_split() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        // Training data
        let x_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y_train = array![5.0, 8.0, 11.0, 14.0, 17.0]; // y = 2*x1 + x2

        // Validation data
        let x_val = array![[6.0, 7.0], [7.0, 8.0]];
        let y_val = array![20.0, 23.0];

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::TargetScore(0.9),
            validation_split: 0.2, // Ignored since we provide split data
            shuffle: false,
            random_state: None,
            higher_is_better: true,
            min_iterations: 1,
            restore_best_weights: false,
        };

        let model = LinearRegression::lasso(0.01);
        let result = model.fit_with_early_stopping_split(
            &x_train,
            &y_train,
            &x_val,
            &y_val,
            early_stopping_config,
        );

        assert!(result.is_ok());
        let (fitted_model, validation_info) = result.unwrap();

        assert_eq!(fitted_model.coef().len(), 2);
        assert!(fitted_model.intercept().is_some());
        assert!(!validation_info.validation_scores.is_empty());

        // Coefficients should be close to true values [2, 1] with small regularization
        let coef = fitted_model.coef();
        assert!((coef[0] - 2.0).abs() < 0.5);
        assert!((coef[1] - 1.0).abs() < 0.5);
    }

    #[test]
    #[cfg(all(feature = "coordinate-descent", feature = "early-stopping"))]
    fn test_linear_regression_early_stopping_ols() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]; // y = 2*x + 1

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(5),
            validation_split: 0.33,
            shuffle: false,
            random_state: None,
            higher_is_better: true,
            min_iterations: 1,
            restore_best_weights: true,
        };

        // For OLS (no penalty), early stopping returns dummy validation info
        let model = LinearRegression::new().fit_intercept(true);
        let result = model.fit_with_early_stopping(&x, &y, early_stopping_config);

        assert!(result.is_ok());
        let (fitted_model, validation_info) = result.unwrap();

        assert_eq!(fitted_model.coef().len(), 1);
        assert!(fitted_model.intercept().is_some());

        // For OLS, validation info indicates no early stopping occurred
        assert!(!validation_info.stopped_early);
        assert!(validation_info.converged);
        assert_eq!(validation_info.best_iteration, 1);

        // Model should still work correctly
        assert_abs_diff_eq!(fitted_model.coef()[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted_model.intercept().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(all(feature = "coordinate-descent", feature = "early-stopping"))]
    fn test_linear_regression_early_stopping_ridge() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        let x = array![
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 3.0]
        ];
        let y = array![2.5, 4.0, 5.5, 7.0, 8.5, 10.0]; // y ≈ 1.5*x1 + x2

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(3),
            validation_split: 0.33,
            shuffle: true,
            random_state: Some(456),
            higher_is_better: true,
            min_iterations: 1,
            restore_best_weights: false,
        };

        // For Ridge regression, early stopping currently returns dummy validation info
        let model = LinearRegression::new()
            .regularization(0.1)
            .fit_intercept(true);
        let result = model.fit_with_early_stopping(&x, &y, early_stopping_config);

        assert!(result.is_ok());
        let (fitted_model, validation_info) = result.unwrap();

        assert_eq!(fitted_model.coef().len(), 2);
        assert!(fitted_model.intercept().is_some());

        // For Ridge, early stopping is not fully implemented yet, so it should indicate convergence
        assert!(!validation_info.stopped_early);
        assert!(validation_info.converged);
    }
}
