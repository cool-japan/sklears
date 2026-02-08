//! Gaussian Process Regression implementations

// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

use crate::kernels::Kernel;

/// Configuration for Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct GaussianProcessRegressorConfig {
    /// alpha
    pub alpha: f64,
    /// optimizer
    pub optimizer: Option<String>,
    /// n_restarts_optimizer
    pub n_restarts_optimizer: usize,
    /// normalize_y
    pub normalize_y: bool,
    /// copy_x_train
    pub copy_x_train: bool,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for GaussianProcessRegressorConfig {
    fn default() -> Self {
        Self {
            alpha: 1e-10,
            optimizer: Some("fmin_l_bfgs_b".to_string()),
            n_restarts_optimizer: 0,
            normalize_y: false,
            copy_x_train: true,
            random_state: None,
        }
    }
}

///
/// let X = array![[1.0], [2.0], [3.0], [4.0]];
/// let y = array![1.0, 4.0, 9.0, 16.0];
///
/// let kernel = RBF::new(1.0);
/// let gpr = GaussianProcessRegressor::new().kernel(Box::new(kernel));
/// let fitted = gpr.fit(&X, &y).unwrap();
/// let (mean, std) = fitted.predict_with_std(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GaussianProcessRegressor<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    config: GaussianProcessRegressorConfig,
}

/// Trained state for Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct GprTrained {
    /// X_train
    pub X_train: Option<Array2<f64>>, // Training inputs
    /// y_train
    pub y_train: Array1<f64>, // Training outputs
    /// y_mean
    pub y_mean: f64, // Mean of training outputs
    /// y_std
    pub y_std: f64, // Standard deviation of training outputs
    /// alpha
    pub alpha: Array1<f64>, // Solved coefficients
    /// L
    pub L: Array2<f64>, // Cholesky decomposition of K + alpha*I
    /// K
    pub K: Array2<f64>, // Kernel matrix
    /// kernel
    pub kernel: Box<dyn Kernel>, // Kernel function
    /// log_marginal_likelihood_value
    pub log_marginal_likelihood_value: f64, // Log marginal likelihood
}

impl GaussianProcessRegressor<Untrained> {
    /// Create a new GaussianProcessRegressor instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            config: GaussianProcessRegressorConfig::default(),
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the noise level
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set the optimizer
    pub fn optimizer(mut self, optimizer: Option<String>) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Set the number of optimizer restarts
    pub fn n_restarts_optimizer(mut self, n_restarts: usize) -> Self {
        self.config.n_restarts_optimizer = n_restarts;
        self
    }

    /// Set whether to normalize the target values
    pub fn normalize_y(mut self, normalize_y: bool) -> Self {
        self.config.normalize_y = normalize_y;
        self
    }

    /// Set whether to copy X during training
    pub fn copy_x_train(mut self, copy_x_train: bool) -> Self {
        self.config.copy_x_train = copy_x_train;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Estimator for GaussianProcessRegressor<Untrained> {
    type Config = GaussianProcessRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for GaussianProcessRegressor<GprTrained> {
    type Config = GaussianProcessRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for GaussianProcessRegressor<Untrained> {
    type Fitted = GaussianProcessRegressor<GprTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;

        // Normalize y if requested
        let (y_normalized, y_mean, y_std) = if self.config.normalize_y {
            let mean = y.mean().unwrap_or(0.0);
            let std = y.std(0.0);
            let std = if std == 0.0 { 1.0 } else { std };
            let y_norm = (y - mean) / std;
            (y_norm, mean, std)
        } else {
            (y.to_owned(), 0.0, 1.0)
        };

        // Compute kernel matrix
        let K = kernel.compute_kernel_matrix(X, None)?;

        // Add noise regularization
        let mut K_reg = K.clone();
        for i in 0..K_reg.nrows() {
            K_reg[[i, i]] += self.config.alpha;
        }

        // Robust Cholesky decomposition
        let L = crate::utils::robust_cholesky(&K_reg)?;

        // Solve K * alpha = y
        let alpha = crate::utils::triangular_solve(&L, &y_normalized)?;

        // Compute log marginal likelihood
        let log_marginal_likelihood_value =
            crate::utils::log_marginal_likelihood(&L, &alpha, &y_normalized);

        let X_train = if self.config.copy_x_train {
            Some(X.to_owned())
        } else {
            None
        };

        Ok(GaussianProcessRegressor {
            state: GprTrained {
                X_train,
                y_train: y_normalized,
                y_mean,
                y_std,
                alpha,
                L,
                K: K_reg,
                kernel,
                log_marginal_likelihood_value,
            },
            kernel: None,
            config: self.config,
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for GaussianProcessRegressor<GprTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
        let X_train = self
            .state
            .X_train
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        // Compute cross-kernel and mean directly — no variance computation needed
        let K_star = self.state.kernel.compute_kernel_matrix(X_train, Some(X))?;
        let mean_normalized = K_star.t().dot(&self.state.alpha);

        if self.config.normalize_y {
            Ok(mean_normalized * self.state.y_std + self.state.y_mean)
        } else {
            Ok(mean_normalized)
        }
    }
}

impl GaussianProcessRegressor<GprTrained> {
    /// Predict with uncertainty estimates
    #[allow(non_snake_case)]
    pub fn predict_with_std(&self, X: &Array2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let X_train = self
            .state
            .X_train
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_with_std".to_string(),
            })?;

        // Compute kernel between test and training points
        let K_star = self.state.kernel.compute_kernel_matrix(X_train, Some(X))?;

        // Predict mean
        let mean_normalized = K_star.t().dot(&self.state.alpha);

        // Denormalize if needed
        let mean = if self.config.normalize_y {
            mean_normalized * self.state.y_std + self.state.y_mean
        } else {
            mean_normalized
        };

        // Compute predictive variance
        // Solve L * v = K_star as a single batch operation
        let v = crate::utils::forward_solve_matrix(&self.state.L, &K_star)?;

        let K_star_star_diag = self.state.kernel.diagonal(X)?;

        let var_normalized = K_star_star_diag - v.map_axis(Axis(0), |col| col.dot(&col));

        // Ensure non-negative variance
        let var_normalized = var_normalized.mapv(|x| x.max(0.0));

        let std = if self.config.normalize_y {
            var_normalized.mapv(|x| (x * self.state.y_std.powi(2)).sqrt())
        } else {
            var_normalized.mapv(|x| x.sqrt())
        };

        Ok((mean, std))
    }

    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.state.log_marginal_likelihood_value
    }
}

impl Default for GaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}
