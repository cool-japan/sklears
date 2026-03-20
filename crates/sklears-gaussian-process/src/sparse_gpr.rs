//! Sparse Gaussian Process Regression implementations

// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

use crate::kernels::Kernel;
use crate::utils::{robust_cholesky, triangular_solve};

/// Configuration for Sparse Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct SparseGaussianProcessRegressorConfig {
    /// n_inducing
    pub n_inducing: usize,
    /// inducing_init
    pub inducing_init: InducingPointInit,
    /// alpha
    pub alpha: f64,
    /// optimizer
    pub optimizer: Option<String>,
    /// n_restarts_optimizer
    pub n_restarts_optimizer: usize,
    /// copy_x_train
    pub copy_x_train: bool,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for SparseGaussianProcessRegressorConfig {
    fn default() -> Self {
        Self {
            n_inducing: 10,
            inducing_init: InducingPointInit::Kmeans,
            alpha: 1e-10,
            optimizer: Some("fmin_l_bfgs_b".to_string()),
            n_restarts_optimizer: 0,
            copy_x_train: true,
            random_state: None,
        }
    }
}

/// # Examples
///
/// ```ignore
/// let X = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
/// let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
///
/// let kernel = RBF::new(1.0);
/// let sgpr = SparseGaussianProcessRegressor::new()
///     .kernel(Box::new(kernel))
///     .n_inducing(4);
/// let fitted = sgpr.fit(&X, &y).unwrap();
/// let predictions = fitted.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseGaussianProcessRegressor<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    config: SparseGaussianProcessRegressorConfig,
}

/// Methods for initializing inducing points
#[derive(Debug, Clone)]
pub enum InducingPointInit {
    /// Random
    Random,
    /// Uniform
    Uniform,
    /// Kmeans
    Kmeans,
}

/// Trained state for Sparse Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct SgprTrained {
    /// X_train
    pub X_train: Option<Array2<f64>>, // Training inputs
    /// y_train
    pub y_train: Array1<f64>, // Training outputs
    /// Z
    pub Z: Array2<f64>, // Inducing points
    /// alpha
    pub alpha: Array1<f64>, // Solved coefficients
    /// Kmm
    pub Kmm: Array2<f64>, // Kernel matrix between inducing points
    /// Knm
    pub Knm: Array2<f64>, // Kernel matrix between training and inducing points
    /// L
    pub L: Array2<f64>, // Cholesky decomposition
    /// kernel
    pub kernel: Box<dyn Kernel>, // Kernel function
    /// sigma_n
    pub sigma_n: f64, // Noise level
    /// log_marginal_likelihood_value
    pub log_marginal_likelihood_value: f64, // Log marginal likelihood
}

impl SparseGaussianProcessRegressor<Untrained> {
    /// Create a new SparseGaussianProcessRegressor instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            config: SparseGaussianProcessRegressorConfig::default(),
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the number of inducing points
    pub fn n_inducing(mut self, n_inducing: usize) -> Self {
        self.config.n_inducing = n_inducing;
        self
    }

    /// Set the inducing point initialization method
    pub fn inducing_init(mut self, inducing_init: InducingPointInit) -> Self {
        self.config.inducing_init = inducing_init;
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

impl Estimator for SparseGaussianProcessRegressor<Untrained> {
    type Config = SparseGaussianProcessRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for SparseGaussianProcessRegressor<SgprTrained> {
    type Config = SparseGaussianProcessRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for SparseGaussianProcessRegressor<Untrained> {
    type Fitted = SparseGaussianProcessRegressor<SgprTrained>;

    fn fit(self, X: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;

        // Initialize inducing points
        let Z = match self.config.inducing_init {
            InducingPointInit::Random => crate::utils::random_inducing_points(
                &X.view(),
                self.config.n_inducing,
                self.config.random_state,
            )?,
            InducingPointInit::Uniform => crate::utils::uniform_inducing_points(
                &X.view(),
                self.config.n_inducing,
                self.config.random_state,
            )?,
            InducingPointInit::Kmeans => crate::utils::kmeans_inducing_points(
                &X.view(),
                self.config.n_inducing,
                self.config.random_state,
            )?,
        };

        // Compute kernel matrices
        let Kmm = kernel.compute_kernel_matrix(&Z, None)?;
        let Knm = kernel.compute_kernel_matrix(X, Some(&Z))?;

        // Add regularization to Kmm
        let mut Kmm_reg = Kmm.clone();
        for i in 0..Kmm_reg.nrows() {
            Kmm_reg[[i, i]] += 1e-8; // Small jitter for numerical stability
        }

        // Cholesky decomposition of Kmm
        let L_mm = robust_cholesky(&Kmm_reg)?;

        // Solve for Q = Knm * Kmm^{-1}
        let mut Q = Array2::<f64>::zeros(Knm.raw_dim());
        for i in 0..Knm.nrows() {
            let knm_i = Knm.row(i).to_owned();
            let q_i = triangular_solve(&L_mm, &knm_i)?;
            Q.row_mut(i).assign(&q_i);
        }

        // Compute diagonal of Knn - Q * Q^T
        let knn_diag = X
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, x)| {
                let knn_ii = kernel.kernel(&x, &x);
                let q_i = Q.row(i);
                knn_ii - q_i.dot(&q_i)
            })
            .collect::<Array1<f64>>();

        // Add noise
        let sigma_n_sq = self.config.alpha;
        let Lambda_diag = knn_diag.mapv(|x| x + sigma_n_sq);

        // Construct A = I + Q^T * Lambda^{-1} * Q
        let mut A = Array2::<f64>::eye(self.config.n_inducing);
        for i in 0..X.nrows() {
            let q_i = Q.row(i);
            let lambda_inv = 1.0 / Lambda_diag[i];
            for j in 0..self.config.n_inducing {
                for k in 0..self.config.n_inducing {
                    A[[j, k]] += q_i[j] * q_i[k] * lambda_inv;
                }
            }
        }

        // Cholesky decomposition of A
        let L_A = robust_cholesky(&A)?;

        // Solve for alpha
        let b = Q.t().dot(&(y / &Lambda_diag));
        let alpha = triangular_solve(&L_A, &b)?;

        // Compute log marginal likelihood (simplified)
        let log_marginal_likelihood_value = {
            let log_det_A = 2.0 * L_A.diag().mapv(|x| x.ln()).sum();
            let log_det_Lambda = Lambda_diag.mapv(|x| x.ln()).sum();
            let quadratic = y.dot(&(y / &Lambda_diag)) - alpha.dot(&b);
            -0.5 * (quadratic
                + log_det_A
                + log_det_Lambda
                + y.len() as f64 * (2.0 * std::f64::consts::PI).ln())
        };

        let X_train = if self.config.copy_x_train {
            Some(X.to_owned())
        } else {
            None
        };

        Ok(SparseGaussianProcessRegressor {
            state: SgprTrained {
                // X_train
                X_train,
                y_train: y.to_owned(),
                Z,
                alpha,
                Kmm: Kmm_reg,
                // Knm
                Knm,
                L: L_A,
                kernel,
                sigma_n: self.config.alpha.sqrt(),
                log_marginal_likelihood_value,
            },
            kernel: None,
            config: self.config,
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for SparseGaussianProcessRegressor<SgprTrained> {
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (mean, _) = self.predict_with_std(X)?;
        Ok(mean)
    }
}

impl SparseGaussianProcessRegressor<SgprTrained> {
    /// Predict with uncertainty estimates
    #[allow(non_snake_case)]
    pub fn predict_with_std(&self, X: &Array2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        // Compute kernel between test points and inducing points
        let K_star_m = self
            .state
            .kernel
            .compute_kernel_matrix(X, Some(&self.state.Z.to_owned()))?;

        // Solve for q_star = K_star_m * Kmm^{-1}
        let L_mm = robust_cholesky(&self.state.Kmm)?;
        let mut Q_star = Array2::<f64>::zeros(K_star_m.raw_dim());
        for i in 0..K_star_m.nrows() {
            let k_star_i = K_star_m.row(i).to_owned();
            let q_star_i = triangular_solve(&L_mm, &k_star_i)?;
            Q_star.row_mut(i).assign(&q_star_i);
        }

        // Predict mean
        let mean = Q_star.dot(&self.state.alpha);

        // Predict variance (simplified)
        let k_star_star_diag = X
            .axis_iter(Axis(0))
            .map(|x| self.state.kernel.kernel(&x, &x))
            .collect::<Array1<f64>>();

        let var = k_star_star_diag.clone(); // Simplified variance computation
        let std = var.mapv(|x| (x + self.state.sigma_n.powi(2)).sqrt().max(0.0));

        Ok((mean, std))
    }

    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.state.log_marginal_likelihood_value
    }

    /// Get the inducing points
    pub fn inducing_points(&self) -> &Array2<f64> {
        &self.state.Z
    }
}

impl Default for SparseGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}
