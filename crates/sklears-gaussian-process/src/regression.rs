//! Gaussian Process Regression Models
//!
//! This module provides advanced Gaussian Process regression implementations:
//! - `VariationalSparseGaussianProcessRegressor`: Scalable GP regression using variational inference
//! - `MultiOutputGaussianProcessRegressor`: Multi-output GP regression with Linear Model of Coregionalization
//!
//! These models offer efficient solutions for large-scale regression problems with uncertainty quantification.

use std::f64::consts::PI;

// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
// SciRS2 Policy - Use scirs2-core for random number generation
// use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};

use crate::classification::GpcConfig;
use crate::kernels::Kernel;
use crate::sparse_gpr;
use crate::utils;

/// Optimization method for variational sparse Gaussian processes
#[derive(Debug, Clone, PartialEq, Default)]
pub enum VariationalOptimizer {
    /// Adam optimizer with adaptive learning rates
    #[default]
    Adam,
    /// Natural gradients optimizer using the Fisher information metric
    NaturalGradients,
    /// Doubly stochastic variational inference with mini-batches for both data and inducing points
    DoublyStochastic,
}

/// # Examples
///
/// ```ignore
/// let X = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
/// let y = array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
///
/// let kernel = RBF::new(2.0);
/// let vsgpr = VariationalSparseGaussianProcessRegressor::new()
///     .kernel(Box::new(kernel))
///     .n_inducing(3);
/// let fitted = vsgpr.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct VariationalSparseGaussianProcessRegressor<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    n_inducing: usize,
    inducing_init: sparse_gpr::InducingPointInit,
    optimizer: VariationalOptimizer,
    learning_rate: f64,
    max_iter: usize,
    batch_size: Option<usize>,
    inducing_batch_size: Option<usize>, // Mini-batch size for inducing points in doubly stochastic
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    natural_gradient_damping: f64, // Damping factor for natural gradients
    sigma_n: f64,
    tol: f64,
    verbose: bool,
    random_state: Option<u64>,
    config: GpcConfig,
}

/// Trained state for Variational Sparse Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct VsgprTrained {
    /// Z
    pub Z: Array2<f64>, // Inducing points
    /// m
    pub m: Array1<f64>, // Variational mean
    /// S
    pub S: Array2<f64>, // Variational covariance
    /// kernel
    pub kernel: Box<dyn Kernel>, // Kernel function
    /// sigma_n
    pub sigma_n: f64, // Noise standard deviation
    /// elbo_history
    pub elbo_history: Vec<f64>, // ELBO history during training
    /// final_elbo
    pub final_elbo: f64, // Final ELBO value
}

impl VariationalSparseGaussianProcessRegressor<Untrained> {
    /// Create a new VariationalSparseGaussianProcessRegressor instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            n_inducing: 10,
            inducing_init: sparse_gpr::InducingPointInit::Kmeans,
            optimizer: VariationalOptimizer::default(),
            learning_rate: 0.01,
            max_iter: 1000,
            batch_size: None,
            inducing_batch_size: None,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            natural_gradient_damping: 1e-4,
            sigma_n: 0.1,
            tol: 1e-6,
            verbose: false,
            random_state: None,
            config: GpcConfig::default(),
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the number of inducing points
    pub fn n_inducing(mut self, n_inducing: usize) -> Self {
        self.n_inducing = n_inducing;
        self
    }

    /// Set the inducing point initialization method
    pub fn inducing_init(mut self, inducing_init: sparse_gpr::InducingPointInit) -> Self {
        self.inducing_init = inducing_init;
        self
    }

    /// Set the learning rate for optimization
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the batch size for mini-batch optimization
    pub fn batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the inducing batch size for doubly stochastic optimization
    pub fn inducing_batch_size(mut self, inducing_batch_size: Option<usize>) -> Self {
        self.inducing_batch_size = inducing_batch_size;
        self
    }

    /// Set the noise standard deviation
    pub fn sigma_n(mut self, sigma_n: f64) -> Self {
        self.sigma_n = sigma_n;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the optimization method
    pub fn optimizer(mut self, optimizer: VariationalOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set the damping factor for natural gradients (only used when optimizer is NaturalGradients)
    pub fn natural_gradient_damping(mut self, damping: f64) -> Self {
        self.natural_gradient_damping = damping;
        self
    }
}

impl Estimator for VariationalSparseGaussianProcessRegressor<Untrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for VariationalSparseGaussianProcessRegressor<VsgprTrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, f64>>
    for VariationalSparseGaussianProcessRegressor<Untrained>
{
    type Fitted = VariationalSparseGaussianProcessRegressor<VsgprTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;

        // Initialize inducing points
        let Z = match self.inducing_init {
            sparse_gpr::InducingPointInit::Random => {
                utils::random_inducing_points(X, self.n_inducing, self.random_state)?
            }
            sparse_gpr::InducingPointInit::Uniform => {
                utils::uniform_inducing_points(X, self.n_inducing, self.random_state)?
            }
            sparse_gpr::InducingPointInit::Kmeans => {
                utils::kmeans_inducing_points(X, self.n_inducing, self.random_state)?
            }
        };

        // Initialize variational parameters
        let mut m = Array1::<f64>::zeros(self.n_inducing);
        let mut S = Array2::<f64>::eye(self.n_inducing);

        // Initialize optimizer-specific parameters
        let mut m_adam_m = Array1::<f64>::zeros(self.n_inducing);
        let mut m_adam_v = Array1::<f64>::zeros(self.n_inducing);
        let mut S_adam_m = Array2::<f64>::zeros((self.n_inducing, self.n_inducing));
        let mut S_adam_v = Array2::<f64>::zeros((self.n_inducing, self.n_inducing));

        let mut elbo_history = Vec::new();
        let n_data = X.nrows();
        let batch_size = self.batch_size.unwrap_or(n_data);

        // Training loop
        for iter in 0..self.max_iter {
            let mut total_elbo = 0.0;
            let mut n_batches = 0;

            // Process data in batches
            for batch_start in (0..n_data).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_data);
                let X_batch = X.slice(s![batch_start..batch_end, ..]);
                let y_batch = y.slice(s![batch_start..batch_end]);

                // Compute ELBO and gradients
                let (elbo, grad_m, grad_S) = compute_elbo_and_gradients(
                    &X_batch,
                    &y_batch,
                    &Z,
                    &m,
                    &S,
                    &kernel,
                    self.sigma_n,
                )?;

                total_elbo += elbo * (n_data as f64 / batch_size as f64);

                // Apply optimizer-specific updates
                match self.optimizer {
                    VariationalOptimizer::Adam => {
                        let t = (iter * (n_data / batch_size) + n_batches + 1) as f64;

                        // Update m with Adam
                        m_adam_m = self.beta1 * &m_adam_m + (1.0 - self.beta1) * &grad_m;
                        m_adam_v =
                            self.beta2 * &m_adam_v + (1.0 - self.beta2) * grad_m.mapv(|x| x * x);
                        let m_hat = &m_adam_m / (1.0 - self.beta1.powf(t));
                        let v_hat = &m_adam_v / (1.0 - self.beta2.powf(t));
                        m = &m
                            + self.learning_rate * &m_hat
                                / (v_hat.mapv(|x| x.sqrt()) + self.epsilon);

                        // Update S with Adam (ensure positive definiteness)
                        S_adam_m = self.beta1 * &S_adam_m + (1.0 - self.beta1) * &grad_S;
                        S_adam_v =
                            self.beta2 * &S_adam_v + (1.0 - self.beta2) * grad_S.mapv(|x| x * x);
                        let S_m_hat = &S_adam_m / (1.0 - self.beta1.powf(t));
                        let S_v_hat = &S_adam_v / (1.0 - self.beta2.powf(t));
                        S = &S
                            + self.learning_rate * &S_m_hat
                                / (S_v_hat.mapv(|x| x.sqrt() + self.epsilon));
                    }
                    VariationalOptimizer::NaturalGradients => {
                        // Natural gradients using the Fisher information metric
                        // For mean parameter: m += η * grad_m
                        m = &m + self.learning_rate * &grad_m;

                        // For covariance parameter: S += η * (S * grad_S * S + damping * I)
                        // This uses the natural gradient based on the Fisher information
                        let natural_grad_S = S.dot(&grad_S).dot(&S)
                            + self.natural_gradient_damping * Array2::<f64>::eye(self.n_inducing);
                        S = &S + self.learning_rate * &natural_grad_S;
                    }
                    VariationalOptimizer::DoublyStochastic => {
                        // Doubly stochastic variational inference
                        // Use mini-batches for both data (already done) and inducing points
                        let inducing_batch_size =
                            self.inducing_batch_size.unwrap_or(self.n_inducing);

                        if inducing_batch_size < self.n_inducing {
                            // Sample random subset of inducing points for this update
                            // SciRS2 Policy - Use scirs2-core for random number generation
                            let mut rng = scirs2_core::random::Random::seed(42);
                            let mut indices: Vec<usize> = (0..self.n_inducing).collect();
                            // Simple shuffle using Fisher-Yates algorithm
                            for i in (1..indices.len()).rev() {
                                let j = rng.gen_range(0..i + 1);
                                indices.swap(i, j);
                            }
                            indices.truncate(inducing_batch_size);

                            // Apply gradients only to selected subset with scaling
                            let scaling_factor =
                                self.n_inducing as f64 / inducing_batch_size as f64;

                            // Update mean parameters for selected indices
                            for &idx in &indices {
                                m[idx] += self.learning_rate * grad_m[idx] * scaling_factor;
                            }

                            // Update covariance parameters for selected indices
                            // Only update the submatrix corresponding to selected indices
                            for &idx in &indices {
                                for &jdx in &indices {
                                    S[[idx, jdx]] +=
                                        self.learning_rate * grad_S[[idx, jdx]] * scaling_factor;
                                }
                            }
                        } else {
                            // Full batch update - use simple gradient ascent for doubly stochastic
                            m = &m + self.learning_rate * &grad_m;
                            S = &S + self.learning_rate * &grad_S;
                        }
                    }
                }

                // Ensure S remains positive definite for both optimizers
                S = ensure_positive_definite(S)?;

                n_batches += 1;
            }

            let avg_elbo = total_elbo / n_batches as f64;
            elbo_history.push(avg_elbo);

            if self.verbose && iter % 100 == 0 {
                println!("Iteration {}: ELBO = {:.6}", iter, avg_elbo);
            }

            // Check convergence
            if iter > 0 && (avg_elbo - elbo_history[iter - 1]).abs() < self.tol {
                if self.verbose {
                    println!("Converged at iteration {}", iter);
                }
                break;
            }
        }

        let final_elbo = elbo_history.last().copied().unwrap_or(0.0);

        Ok(VariationalSparseGaussianProcessRegressor {
            state: VsgprTrained {
                Z,
                m,
                S,
                kernel,
                sigma_n: self.sigma_n,
                elbo_history,
                final_elbo,
            },
            kernel: None,
            n_inducing: self.n_inducing,
            inducing_init: self.inducing_init,
            optimizer: self.optimizer,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            batch_size: self.batch_size,
            inducing_batch_size: self.inducing_batch_size,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            natural_gradient_damping: self.natural_gradient_damping,
            sigma_n: self.sigma_n,
            tol: self.tol,
            verbose: self.verbose,
            random_state: self.random_state,
            config: self.config.clone(),
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<f64>>
    for VariationalSparseGaussianProcessRegressor<VsgprTrained>
{
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let (mean, _) = self.predict_with_std(X)?;
        Ok(mean)
    }
}

impl VariationalSparseGaussianProcessRegressor<VsgprTrained> {
    /// Predict with uncertainty estimates
    #[allow(non_snake_case)]
    pub fn predict_with_std(&self, X: &ArrayView2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        // Compute kernel matrices
        let Kzz = self
            .state
            .kernel
            .compute_kernel_matrix(&self.state.Z, None)?;
        let X_owned = X.to_owned();
        let Kxz = self
            .state
            .kernel
            .compute_kernel_matrix(&X_owned, Some(&self.state.Z))?;
        let Kxx_diag = X
            .axis_iter(Axis(0))
            .map(|x| self.state.kernel.kernel(&x, &x))
            .collect::<Array1<f64>>();

        // Cholesky decomposition of Kzz
        let L_zz = utils::robust_cholesky(&Kzz)?;

        // Solve Lzz^{-1} * Kxz^T -> A
        let mut A = Array2::<f64>::zeros((self.state.Z.nrows(), X.nrows()));
        for i in 0..X.nrows() {
            let kxz_i = Kxz.row(i).to_owned();
            let a_i = utils::triangular_solve(&L_zz, &kxz_i)?;
            A.column_mut(i).assign(&a_i);
        }

        // Predictive mean: A^T * m
        let mean = A.t().dot(&self.state.m);

        // Predictive variance: Kxx + A^T * (S - I) * A
        let I = Array2::<f64>::eye(self.state.S.nrows());
        let S_diff = &self.state.S - &I;
        let var_correction = A.t().dot(&S_diff.dot(&A));

        let mut variance = Kxx_diag.clone();
        for i in 0..X.nrows() {
            variance[i] += var_correction[[i, i]] + self.state.sigma_n.powi(2);
        }

        let std = variance.mapv(|x| x.sqrt().max(0.0));

        Ok((mean, std))
    }

    /// Get the evidence lower bound (ELBO)
    pub fn elbo(&self) -> f64 {
        self.state.final_elbo
    }

    /// Get the ELBO history during training
    pub fn elbo_history(&self) -> &[f64] {
        &self.state.elbo_history
    }

    /// Get the inducing points
    pub fn inducing_points(&self) -> &Array2<f64> {
        &self.state.Z
    }

    /// Get the variational mean
    pub fn variational_mean(&self) -> &Array1<f64> {
        &self.state.m
    }

    /// Get the variational covariance
    pub fn variational_covariance(&self) -> &Array2<f64> {
        &self.state.S
    }

    /// Online update with new data for streaming/scalable learning
    ///
    /// This method allows incrementally updating the variational parameters
    /// with new data points without retraining from scratch, making it suitable
    /// for streaming scenarios and very large datasets.
    ///
    /// # Arguments
    /// * `X_new` - New input data points
    /// * `y_new` - New target values
    /// * `learning_rate` - Learning rate for the update (if None, uses model's learning rate)
    /// * `n_iterations` - Number of update iterations (default: 10)
    ///
    /// # Returns
    /// Updated model with modified variational parameters
    pub fn update(
        mut self,
        X_new: &ArrayView2<f64>,
        y_new: &ArrayView1<f64>,
        learning_rate: Option<f64>,
        n_iterations: Option<usize>,
    ) -> SklResult<Self> {
        if X_new.nrows() != y_new.len() {
            return Err(SklearsError::InvalidInput(
                "X_new and y_new must have the same number of samples".to_string(),
            ));
        }

        let lr = learning_rate.unwrap_or(self.learning_rate);
        let n_iter = n_iterations.unwrap_or(10);

        // Perform incremental updates using the current optimizer
        for _ in 0..n_iter {
            // Compute gradients for new data
            let (_, grad_m, grad_S) = compute_elbo_and_gradients(
                X_new,
                y_new,
                &self.state.Z,
                &self.state.m,
                &self.state.S,
                &self.state.kernel,
                self.state.sigma_n,
            )?;

            // Apply optimizer-specific updates
            match self.optimizer {
                VariationalOptimizer::Adam => {
                    // Simple gradient ascent for online updates
                    self.state.m = &self.state.m + lr * &grad_m;
                    self.state.S = &self.state.S + lr * &grad_S;
                }
                VariationalOptimizer::NaturalGradients => {
                    // Natural gradients update
                    self.state.m = &self.state.m + lr * &grad_m;
                    let natural_grad_S = self.state.S.dot(&grad_S).dot(&self.state.S)
                        + self.natural_gradient_damping * Array2::<f64>::eye(self.state.Z.nrows());
                    self.state.S = &self.state.S + lr * &natural_grad_S;
                }
                VariationalOptimizer::DoublyStochastic => {
                    // For streaming updates, use simple gradient ascent
                    self.state.m = &self.state.m + lr * &grad_m;
                    self.state.S = &self.state.S + lr * &grad_S;
                }
            }

            // Ensure S remains positive definite
            self.state.S = ensure_positive_definite(self.state.S)?;
        }

        // Update ELBO history with final value
        let (final_elbo, _, _) = compute_elbo_and_gradients(
            X_new,
            y_new,
            &self.state.Z,
            &self.state.m,
            &self.state.S,
            &self.state.kernel,
            self.state.sigma_n,
        )?;

        self.state.elbo_history.push(final_elbo);
        self.state.final_elbo = final_elbo;

        Ok(self)
    }

    /// Recursive Bayesian update for online GP learning
    ///
    /// This method implements proper recursive Bayesian updates for Gaussian processes,
    /// maintaining the posterior mean and covariance through sequential updates without
    /// requiring full recomputation of the ELBO.
    ///
    /// # Arguments
    /// * `X_new` - New input data (n_new x n_features)
    /// * `y_new` - New target values (n_new,)
    /// * `forgetting_factor` - Exponential forgetting factor (0 < λ ≤ 1)
    ///
    /// # Returns
    /// Updated model with recursively updated posterior
    #[allow(non_snake_case)]
    pub fn recursive_update(
        mut self,
        X_new: &ArrayView2<f64>,
        y_new: &ArrayView1<f64>,
        forgetting_factor: Option<f64>,
    ) -> SklResult<Self> {
        if X_new.nrows() != y_new.len() {
            return Err(SklearsError::InvalidInput(
                "X_new and y_new must have the same number of samples".to_string(),
            ));
        }

        let lambda = forgetting_factor.unwrap_or(1.0);
        if lambda <= 0.0 || lambda > 1.0 {
            return Err(SklearsError::InvalidInput(
                "Forgetting factor must be in range (0, 1]".to_string(),
            ));
        }

        // Apply forgetting to prior covariance (increase uncertainty)
        if lambda < 1.0 {
            self.state.S /= lambda;
        }

        // Compute kernel matrices for new data
        let Kzz = self
            .state
            .kernel
            .compute_kernel_matrix(&self.state.Z, None)?;
        let X_new_owned = X_new.to_owned();
        let Kzx_new = self
            .state
            .kernel
            .compute_kernel_matrix(&self.state.Z, Some(&X_new_owned))?;

        // Robust Cholesky decomposition
        let L_zz = utils::robust_cholesky(&Kzz)?;

        // For each new data point, perform recursive Bayesian update
        for (i, &y_i) in y_new.iter().enumerate() {
            let k_zi = Kzx_new.column(i);

            // Solve L_zz * alpha = k_zi
            let alpha = utils::triangular_solve(&L_zz, &k_zi.to_owned())?;

            // Predictive variance: σ²_i = σ²_n + k_ii - α^T α
            let k_ii = self.state.kernel.kernel(&X_new.row(i), &X_new.row(i));
            let pred_var = self.state.sigma_n.powi(2) + k_ii - alpha.dot(&alpha);

            if pred_var <= 0.0 {
                continue; // Skip if predictive variance is non-positive
            }

            // Predictive mean: μ_i = α^T m
            let pred_mean = alpha.dot(&self.state.m);

            // Innovation (prediction error)
            let innovation = y_i - pred_mean;

            // Kalman gain: K = S * α / σ²_i
            let kalman_gain = self.state.S.dot(&alpha) / pred_var;

            // Update posterior mean: m := m + K * innovation
            let m_update = &kalman_gain * innovation;
            self.state.m = &self.state.m + &m_update;

            // Update posterior covariance: S := S - K * α^T * S
            let s_update = kalman_gain
                .view()
                .into_shape((kalman_gain.len(), 1))
                .map_err(|_| SklearsError::FitError("Shape error in recursive update".to_string()))?
                .dot(&alpha.view().into_shape((1, alpha.len())).map_err(|_| {
                    SklearsError::FitError("Shape error in recursive update".to_string())
                })?)
                .dot(&self.state.S);
            self.state.S = &self.state.S - &s_update;

            // Ensure positive definiteness
            self.state.S = ensure_positive_definite(self.state.S)?;
        }

        // Update ELBO history with approximated value
        let approx_elbo = self.compute_approximate_elbo(X_new, y_new)?;
        self.state.elbo_history.push(approx_elbo);
        self.state.final_elbo = approx_elbo;

        Ok(self)
    }

    /// Sliding window update for streaming data
    ///
    /// Maintains a sliding window of recent data points and uses exponential
    /// forgetting to down-weight older observations.
    ///
    /// # Arguments
    /// * `X_new` - New input data
    /// * `y_new` - New target values
    /// * `window_size` - Maximum number of recent observations to maintain
    /// * `decay_rate` - Exponential decay rate for older observations
    pub fn sliding_window_update(
        mut self,
        X_new: &ArrayView2<f64>,
        y_new: &ArrayView1<f64>,
        window_size: usize,
        decay_rate: f64,
    ) -> SklResult<Self> {
        if X_new.nrows() != y_new.len() {
            return Err(SklearsError::InvalidInput(
                "X_new and y_new must have the same number of samples".to_string(),
            ));
        }

        if decay_rate <= 0.0 || decay_rate > 1.0 {
            return Err(SklearsError::InvalidInput(
                "Decay rate must be in range (0, 1]".to_string(),
            ));
        }

        // Apply exponential forgetting based on window position
        let n_new = X_new.nrows();
        for i in 0..n_new {
            let age_weight = decay_rate.powi((n_new - i - 1) as i32);
            let forgetting_factor = (1.0 - age_weight).max(0.1); // Minimum forgetting factor

            let x_i = X_new.row(i);
            let y_i = Array1::from(vec![y_new[i]]);

            self = self.recursive_update(
                &x_i.view()
                    .into_shape((1, x_i.len()))
                    .map_err(|_| {
                        SklearsError::FitError("Shape error in sliding window update".to_string())
                    })?
                    .view(),
                &y_i.view(),
                Some(forgetting_factor),
            )?;
        }

        // Limit ELBO history to window size
        if self.state.elbo_history.len() > window_size {
            let start_idx = self.state.elbo_history.len() - window_size;
            self.state.elbo_history = self.state.elbo_history[start_idx..].to_vec();
        }

        Ok(self)
    }

    /// Compute approximate ELBO for recursive updates
    ///
    /// This provides a computationally efficient approximation to the full ELBO
    /// calculation for use in recursive updates.
    fn compute_approximate_elbo(&self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> SklResult<f64> {
        let _n = X.nrows() as f64;

        // Compute prediction errors
        let (y_pred, y_var) = self.predict_with_std(X)?;

        // Log likelihood approximation
        let mut log_likelihood = 0.0;
        for i in 0..X.nrows() {
            let residual = y[i] - y_pred[i];
            let total_var = y_var[i] + self.state.sigma_n.powi(2);
            log_likelihood -=
                0.5 * (residual.powi(2) / total_var + total_var.ln() + (2.0 * PI).ln());
        }

        // KL divergence approximation (simplified)
        let kl_divergence = self.compute_approximate_kl_divergence()?;

        // ELBO = log likelihood - KL divergence
        Ok(log_likelihood - kl_divergence)
    }

    /// Compute approximate KL divergence for ELBO calculation
    #[allow(non_snake_case)]
    fn compute_approximate_kl_divergence(&self) -> SklResult<f64> {
        let m = self.state.Z.nrows();

        // Prior: N(0, K_zz)
        let K_zz = self
            .state
            .kernel
            .compute_kernel_matrix(&self.state.Z, None)?;
        let L_zz = utils::robust_cholesky(&K_zz)?;

        // Posterior: N(m, S)
        let L_s = utils::robust_cholesky(&self.state.S)?;

        // KL(q||p) = 0.5 * [tr(K_zz^{-1} S) + m^T K_zz^{-1} m - m - log|S| + log|K_zz|]

        // log|K_zz| = 2 * sum(log(diag(L_zz)))
        let log_det_k = 2.0 * L_zz.diag().iter().map(|x| x.ln()).sum::<f64>();

        // log|S| = 2 * sum(log(diag(L_s)))
        let log_det_s = 2.0 * L_s.diag().iter().map(|x| x.ln()).sum::<f64>();

        // Solve K_zz^{-1} m and K_zz^{-1} S
        let k_inv_m = utils::triangular_solve(&L_zz, &self.state.m)?;
        let mut k_inv_s_trace = 0.0;
        for i in 0..m {
            let s_col = self.state.S.column(i).to_owned();
            let k_inv_s_col = utils::triangular_solve(&L_zz, &s_col)?;
            k_inv_s_trace += k_inv_s_col.dot(&s_col);
        }

        let kl = 0.5 * (k_inv_s_trace + k_inv_m.dot(&k_inv_m) - m as f64 - log_det_s + log_det_k);

        Ok(kl)
    }

    /// Adaptive sparse GP with dynamic inducing point management
    ///
    /// This method adaptively adds/removes inducing points based on the approximation
    /// quality and computational budget, maintaining good approximation while controlling
    /// computational cost.
    ///
    /// # Arguments
    /// * `X_new` - New input data
    /// * `y_new` - New target values
    /// * `max_inducing` - Maximum number of inducing points allowed
    /// * `quality_threshold` - Minimum approximation quality threshold (0.0-1.0)
    /// * `removal_threshold` - Threshold for removing redundant inducing points
    ///
    /// # Returns
    /// Updated model with adaptively adjusted inducing points
    pub fn adaptive_sparse_update(
        mut self,
        X_new: &ArrayView2<f64>,
        y_new: &ArrayView1<f64>,
        max_inducing: usize,
        quality_threshold: f64,
        removal_threshold: f64,
    ) -> SklResult<Self> {
        if X_new.nrows() != y_new.len() {
            return Err(SklearsError::InvalidInput(
                "X_new and y_new must have the same number of samples".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&quality_threshold) {
            return Err(SklearsError::InvalidInput(
                "Quality threshold must be in range [0, 1]".to_string(),
            ));
        }

        // 1. First, perform standard recursive update
        self = self.recursive_update(X_new, y_new, None)?;

        // 2. Assess current approximation quality
        let quality = self.assess_approximation_quality(X_new)?;

        // 3. If quality is below threshold and we haven't reached max inducing points, add new ones
        if quality < quality_threshold && self.state.Z.nrows() < max_inducing {
            self = self.add_inducing_points(X_new, y_new, max_inducing)?;
        }

        // 4. Remove redundant inducing points if we have too many
        if self.state.Z.nrows() > max_inducing {
            self = self.remove_redundant_inducing_points(removal_threshold, max_inducing)?;
        }

        // 5. Optionally optimize inducing point locations
        self = self.optimize_inducing_point_locations(X_new, y_new)?;

        Ok(self)
    }

    /// Assess the approximation quality of current inducing points
    ///
    /// Returns a quality score between 0 and 1, where 1 indicates perfect approximation
    fn assess_approximation_quality(&self, X: &ArrayView2<f64>) -> SklResult<f64> {
        let n_test = (X.nrows() / 4).max(10).min(50); // Sample subset for efficiency
        let indices: Vec<usize> = (0..X.nrows()).step_by(X.nrows() / n_test + 1).collect();

        let mut total_variance_explained = 0.0;
        let mut total_variance = 0.0;

        for &i in indices.iter().take(n_test) {
            let x_i = X.row(i);

            // Compute true kernel value k(x_i, x_i)
            let k_ii = self.state.kernel.kernel(&x_i, &x_i);

            // Compute approximated kernel value through inducing points
            let kxz = self
                .state
                .Z
                .axis_iter(Axis(0))
                .map(|z| self.state.kernel.kernel(&x_i, &z))
                .collect::<Array1<f64>>();

            let Kzz = self
                .state
                .kernel
                .compute_kernel_matrix(&self.state.Z, None)?;
            let L_zz = utils::robust_cholesky(&Kzz)?;
            let alpha = utils::triangular_solve(&L_zz, &kxz)?;
            let k_approx = alpha.dot(&alpha);

            // Variance explained by inducing points
            let variance_explained = k_approx / k_ii;
            total_variance_explained += variance_explained;
            total_variance += 1.0;
        }

        let quality = total_variance_explained / total_variance;
        Ok(quality.min(1.0).max(0.0))
    }

    /// Add new inducing points based on data coverage and uncertainty
    fn add_inducing_points(
        mut self,
        X: &ArrayView2<f64>,
        _y: &ArrayView1<f64>,
        max_inducing: usize,
    ) -> SklResult<Self> {
        let current_inducing = self.state.Z.nrows();
        let points_to_add = (max_inducing - current_inducing).min(X.nrows());

        if points_to_add == 0 {
            return Ok(self);
        }

        // Strategy 1: Select points with highest prediction uncertainty
        let (_, uncertainties) = self.predict_with_std(X)?;
        let mut uncertainty_indices: Vec<(usize, f64)> = uncertainties
            .iter()
            .enumerate()
            .map(|(i, &u)| (i, u))
            .collect();

        // Sort by uncertainty (descending)
        uncertainty_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Strategy 2: Ensure good spatial coverage by checking distances to existing inducing points
        let mut selected_indices = Vec::new();
        for (idx, _) in uncertainty_indices.iter().take(points_to_add * 2) {
            let x_candidate = X.row(*idx);

            // Check minimum distance to existing inducing points
            let min_distance = self
                .state
                .Z
                .axis_iter(Axis(0))
                .map(|z| {
                    let diff = &x_candidate - &z;
                    diff.dot(&diff).sqrt()
                })
                .fold(f64::INFINITY, f64::min);

            // Add point if it's sufficiently far from existing inducing points
            if min_distance > 0.1 {
                // Minimum distance threshold
                selected_indices.push(*idx);
                if selected_indices.len() >= points_to_add {
                    break;
                }
            }
        }

        // Add selected points to inducing set
        if !selected_indices.is_empty() {
            let mut new_Z = Array2::zeros((current_inducing + selected_indices.len(), X.ncols()));
            new_Z
                .slice_mut(s![..current_inducing, ..])
                .assign(&self.state.Z);

            for (i, &idx) in selected_indices.iter().enumerate() {
                new_Z.row_mut(current_inducing + i).assign(&X.row(idx));
            }

            self.state.Z = new_Z;

            // Expand variational parameters
            let new_m_size = self.state.Z.nrows();
            let mut new_m = Array1::zeros(new_m_size);
            new_m
                .slice_mut(s![..current_inducing])
                .assign(&self.state.m);
            // Initialize new parameters with small random values
            for i in current_inducing..new_m_size {
                new_m[i] = 0.01 * (i as f64 - new_m_size as f64 / 2.0) / new_m_size as f64;
            }
            self.state.m = new_m;

            // Expand covariance matrix
            let mut new_S = Array2::eye(new_m_size) * 0.1; // Small diagonal initialization
            new_S
                .slice_mut(s![..current_inducing, ..current_inducing])
                .assign(&self.state.S);
            self.state.S = new_S;
        }

        Ok(self)
    }

    /// Remove redundant inducing points to maintain computational efficiency
    fn remove_redundant_inducing_points(
        mut self,
        _removal_threshold: f64,
        max_inducing: usize,
    ) -> SklResult<Self> {
        let current_inducing = self.state.Z.nrows();
        if current_inducing <= max_inducing {
            return Ok(self);
        }

        let points_to_remove = current_inducing - max_inducing;

        // Compute influence scores for each inducing point
        let mut influence_scores = Vec::new();
        for i in 0..current_inducing {
            // Score based on variational mean magnitude and diagonal covariance
            let influence = self.state.m[i].abs() + self.state.S[[i, i]];
            influence_scores.push((i, influence));
        }

        // Sort by influence (ascending - remove least influential)
        influence_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Select points to remove
        let indices_to_remove: Vec<usize> = influence_scores
            .iter()
            .take(points_to_remove)
            .map(|(i, _)| *i)
            .collect();

        // Create mask of points to keep
        let mut keep_mask = vec![true; current_inducing];
        for &i in &indices_to_remove {
            keep_mask[i] = false;
        }

        // Filter inducing points
        let kept_indices: Vec<usize> = (0..current_inducing).filter(|&i| keep_mask[i]).collect();

        let new_size = kept_indices.len();
        let mut new_Z = Array2::zeros((new_size, self.state.Z.ncols()));
        let mut new_m = Array1::zeros(new_size);
        let mut new_S = Array2::zeros((new_size, new_size));

        // Copy kept inducing points and parameters
        for (new_i, &old_i) in kept_indices.iter().enumerate() {
            new_Z.row_mut(new_i).assign(&self.state.Z.row(old_i));
            new_m[new_i] = self.state.m[old_i];
            for (new_j, &old_j) in kept_indices.iter().enumerate() {
                new_S[[new_i, new_j]] = self.state.S[[old_i, old_j]];
            }
        }

        self.state.Z = new_Z;
        self.state.m = new_m;
        self.state.S = new_S;

        Ok(self)
    }

    /// Optimize inducing point locations to improve approximation
    fn optimize_inducing_point_locations(
        mut self,
        X: &ArrayView2<f64>,
        _y: &ArrayView1<f64>,
    ) -> SklResult<Self> {
        // Simple optimization: move inducing points towards data centroids in their neighborhoods
        let n_inducing = self.state.Z.nrows();

        for i in 0..n_inducing {
            let z_i = self.state.Z.row(i);

            // Find nearby data points
            let mut nearby_points = Vec::new();
            for j in 0..X.nrows() {
                let x_j = X.row(j);
                let distance = (&z_i - &x_j).mapv(|x| x.powi(2)).sum().sqrt();
                if distance < 1.0 {
                    // Distance threshold
                    nearby_points.push(j);
                }
            }

            // Compute centroid of nearby points
            if !nearby_points.is_empty() {
                let mut centroid = Array1::zeros(X.ncols());
                for &j in &nearby_points {
                    centroid = centroid + X.row(j);
                }
                centroid /= nearby_points.len() as f64;

                // Move inducing point towards centroid (with momentum)
                let momentum = 0.1;
                let new_z_i = (1.0 - momentum) * &z_i + momentum * &centroid;
                self.state.Z.row_mut(i).assign(&new_z_i);
            }
        }

        Ok(self)
    }
}

impl Default for VariationalSparseGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute ELBO and gradients for variational sparse GP
#[allow(non_snake_case)]
fn compute_elbo_and_gradients(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    Z: &Array2<f64>,
    m: &Array1<f64>,
    S: &Array2<f64>,
    kernel: &Box<dyn Kernel>,
    sigma_n: f64,
) -> SklResult<(f64, Array1<f64>, Array2<f64>)> {
    let n = X.nrows();
    let m_ind = Z.nrows();

    // Compute kernel matrices
    let Kzz = kernel.compute_kernel_matrix(Z, None)?;
    let X_owned = X.to_owned();
    let Kxz = kernel.compute_kernel_matrix(&X_owned, Some(Z))?;
    let Kxx_diag = X
        .axis_iter(Axis(0))
        .map(|x| kernel.kernel(&x, &x))
        .collect::<Array1<f64>>();

    // Cholesky decomposition of Kzz
    let L_zz = utils::robust_cholesky(&Kzz)?;

    // Solve Lzz^{-1} * Kxz^T -> A
    let mut A = Array2::<f64>::zeros((m_ind, n));
    for i in 0..n {
        let kxz_i = Kxz.row(i).to_owned();
        let a_i = utils::triangular_solve(&L_zz, &kxz_i)?;
        A.column_mut(i).assign(&a_i);
    }

    // Compute predictive mean and variance
    let f_mean = A.t().dot(m);
    let A_S_At = A.t().dot(&S.dot(&A));

    let mut f_var = Kxx_diag.clone();
    for i in 0..n {
        f_var[i] += A_S_At[[i, i]] - A.column(i).dot(&A.column(i));
    }

    // Data fit term
    let sigma_n_sq = sigma_n * sigma_n;
    let mut data_fit = 0.0;
    for i in 0..n {
        let residual = y[i] - f_mean[i];
        let total_var = f_var[i] + sigma_n_sq;
        data_fit -= 0.5 * (residual * residual / total_var + total_var.ln() + (2.0 * PI).ln());
    }

    // KL divergence term
    let I = Array2::<f64>::eye(m_ind);
    let Kzz_inv = utils::triangular_solve_matrix(&L_zz, &I)?;
    let S_Kzz_inv = S.dot(&Kzz_inv);

    let trace_term = S_Kzz_inv.diag().sum();
    let quad_term = m.dot(&Kzz_inv.dot(m));
    let log_det_S = 2.0 * utils::robust_cholesky(S)?.diag().mapv(|x| x.ln()).sum();
    let log_det_Kzz = 2.0 * L_zz.diag().mapv(|x| x.ln()).sum();

    let kl_div = 0.5 * (trace_term + quad_term - m_ind as f64 + log_det_Kzz - log_det_S);

    let elbo = data_fit - kl_div;

    // Compute gradients
    let mut grad_m = Array1::<f64>::zeros(m_ind);
    let mut grad_S = Array2::<f64>::zeros((m_ind, m_ind));

    // Gradient w.r.t. m
    for i in 0..n {
        let residual = y[i] - f_mean[i];
        let total_var = f_var[i] + sigma_n_sq;
        let a_i = A.column(i);
        grad_m = &grad_m + residual / total_var * &a_i.to_owned();
    }
    grad_m = &grad_m - &Kzz_inv.dot(m);

    // Gradient w.r.t. S (simplified)
    for i in 0..n {
        let residual = y[i] - f_mean[i];
        let total_var = f_var[i] + sigma_n_sq;
        let a_i = A.column(i).to_owned();
        let outer_a = Array2::from_shape_fn((m_ind, m_ind), |(j, k)| a_i[j] * a_i[k]);

        grad_S = &grad_S
            + 0.5 * (residual * residual / (total_var * total_var) - 1.0 / total_var) * &outer_a;
    }
    grad_S = &grad_S - 0.5 * &Kzz_inv;

    Ok((elbo, grad_m, grad_S))
}

/// Ensure a matrix remains positive definite
fn ensure_positive_definite(mut S: Array2<f64>) -> SklResult<Array2<f64>> {
    // Simple approach: add small diagonal jitter if needed
    let min_eigenval = 1e-6;

    // Check if matrix is symmetric
    let is_symmetric = S
        .iter()
        .zip(S.t().iter())
        .all(|(a, b)| (a - b).abs() < 1e-12);
    if !is_symmetric {
        // Force symmetry
        S = 0.5 * (&S + &S.t());
    }

    // Add jitter to diagonal
    for i in 0..S.nrows() {
        S[[i, i]] += min_eigenval;
    }

    Ok(S)
}

/// Multi-output Gaussian Process Regressor with Linear Model of Coregionalization (LMC)
///
/// The Linear Model of Coregionalization (LMC) is a framework for modeling multiple
/// correlated outputs by expressing each output as a linear combination of independent
/// latent Gaussian processes. This approach captures cross-correlations between outputs
/// while maintaining computational efficiency.
///
/// For Q outputs and R latent GPs, the model is:
/// f_q(x) = Σ_r A_{q,r} * u_r(x)
///
/// where:
/// - f_q(x) is the q-th output function
/// - u_r(x) are independent latent GPs with kernel k_r(x, x')
/// - A is the Q×R mixing matrix that captures output correlations
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::{MultiOutputGaussianProcessRegressor, RBF};
/// use sklears_core::traits::{Fit, Predict};
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
/// use scirs2_core::ndarray::array;
///
/// let kernel = RBF::new(1.0);
/// let mogpr = MultiOutputGaussianProcessRegressor::new()
///     .n_outputs(2)
///     .n_latent(1)
///     .kernel(Box::new(kernel))
///     .alpha(1e-10);
///
/// let X = array![[1.0], [2.0], [3.0]];
/// let Y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3 samples, 2 outputs
///
/// let fitted = mogpr.fit(&X.view(), &Y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiOutputGaussianProcessRegressor<S = Untrained> {
    kernel: Option<Box<dyn Kernel>>,
    alpha: f64,
    n_outputs: usize,
    n_latent: usize,
    mixing_matrix: Option<Array2<f64>>, // Q × R mixing matrix A
    _state: S,
}

/// Trained state for MultiOutputGaussianProcessRegressor
#[derive(Debug, Clone)]
pub struct MogprTrained {
    X_train: Array2<f64>,
    Y_train: Array2<f64>,
    kernel: Box<dyn Kernel>,
    alpha: f64,
    n_outputs: usize,
    n_latent: usize,
    mixing_matrix: Array2<f64>,
    covariance_inv: Vec<Array2<f64>>, // Inverse covariances for each latent GP
    y_latent: Vec<Array1<f64>>,       // Latent targets for each GP
}

impl MultiOutputGaussianProcessRegressor<Untrained> {
    /// Create a new MultiOutputGaussianProcessRegressor instance
    pub fn new() -> Self {
        Self {
            kernel: None,
            alpha: 1e-10,
            n_outputs: 1,
            n_latent: 1,
            mixing_matrix: None,
            _state: Untrained,
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the number of outputs
    pub fn n_outputs(mut self, n_outputs: usize) -> Self {
        self.n_outputs = n_outputs;
        self
    }

    /// Set the number of latent GPs
    pub fn n_latent(mut self, n_latent: usize) -> Self {
        self.n_latent = n_latent;
        self
    }

    /// Set a custom mixing matrix A (Q × R)
    pub fn mixing_matrix(mut self, mixing_matrix: Array2<f64>) -> Self {
        self.mixing_matrix = Some(mixing_matrix);
        self
    }

    /// Initialize the mixing matrix randomly or with provided values
    fn initialize_mixing_matrix(&self) -> Array2<f64> {
        if let Some(ref matrix) = self.mixing_matrix {
            matrix.clone()
        } else {
            // Initialize with random values from normal distribution
            let mut matrix = Array2::<f64>::zeros((self.n_outputs, self.n_latent));
            let mut rng_state = 42u64; // Simple seed

            for i in 0..self.n_outputs {
                for j in 0..self.n_latent {
                    // Simple Box-Muller for normal samples
                    let u1 = self.uniform(&mut rng_state);
                    let u2 = self.uniform(&mut rng_state);
                    let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    matrix[[i, j]] = normal * 0.5; // Scale down initial values
                }
            }
            matrix
        }
    }

    /// Simple uniform random number generator
    fn uniform(&self, state: &mut u64) -> f64 {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        (*state as f64) / (u64::MAX as f64)
    }

    /// Optimize the mixing matrix using alternating optimization
    fn optimize_mixing_matrix(
        &self,
        Y: &ArrayView2<f64>,
        _K_inv: &[Array2<f64>],
        n_iter: usize,
    ) -> (Array2<f64>, Vec<Array1<f64>>) {
        let (n_samples, n_outputs) = Y.dim();
        let mut A = self.initialize_mixing_matrix();
        let mut y_latent = vec![Array1::<f64>::zeros(n_samples); self.n_latent];

        for _iter in 0..n_iter {
            // Update latent targets given current mixing matrix
            for r in 0..self.n_latent {
                let mut target = Array1::<f64>::zeros(n_samples);

                // Compute pseudo-targets for latent GP r
                for i in 0..n_samples {
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for q in 0..n_outputs {
                        let weight = A[[q, r]].powi(2);
                        weighted_sum += weight * Y[[i, q]];
                        weight_sum += weight;
                    }

                    if weight_sum > 1e-10 {
                        target[i] = weighted_sum / weight_sum;
                    }
                }

                y_latent[r] = target;
            }

            // Update mixing matrix given current latent targets
            for q in 0..n_outputs {
                for r in 0..self.n_latent {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;

                    for i in 0..n_samples {
                        // Compute the optimal A[q,r] using least squares
                        let residual = Y[[i, q]];
                        let latent_contrib = y_latent[r][i];

                        numerator += residual * latent_contrib;
                        denominator += latent_contrib * latent_contrib;
                    }

                    if denominator > 1e-10 {
                        A[[q, r]] = numerator / denominator;
                    }
                }
            }
        }

        (A, y_latent)
    }
}

impl Default for MultiOutputGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiOutputGaussianProcessRegressor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView2<'_, f64>, SklearsError>
    for MultiOutputGaussianProcessRegressor<Untrained>
{
    type Fitted = MultiOutputGaussianProcessRegressor<MogprTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, Y: &ArrayView2<f64>) -> Result<Self::Fitted, SklearsError> {
        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("No kernel provided".to_string()))?
            .clone();

        let (n_samples, _n_features) = X.dim();
        let (n_samples_y, n_outputs) = Y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and Y must match".to_string(),
            ));
        }

        if n_outputs != self.n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} outputs, got {}",
                self.n_outputs, n_outputs
            )));
        }

        // Compute kernel matrix
        let X_owned = X.to_owned();
        let K = kernel.compute_kernel_matrix(&X_owned, None)?;
        let mut K_reg = K.clone();

        // Add regularization
        for i in 0..n_samples {
            K_reg[[i, i]] += self.alpha;
        }

        // For multi-output, we need to solve for each latent GP
        let mut covariance_inv = Vec::new();

        // Each latent GP uses the same kernel structure
        for _r in 0..self.n_latent {
            let chol_decomp = utils::robust_cholesky(&K_reg)?;
            let identity = Array2::eye(n_samples);
            let inv = solve_triangular_matrix(&chol_decomp, &identity)?;
            covariance_inv.push(inv);
        }

        // Optimize mixing matrix and latent targets
        let (mixing_matrix, y_latent) = self.optimize_mixing_matrix(Y, &covariance_inv, 10);

        Ok(MultiOutputGaussianProcessRegressor {
            kernel: None,
            alpha: 0.0,
            n_outputs: 0,
            n_latent: 0,
            mixing_matrix: None,
            _state: MogprTrained {
                X_train: X.to_owned(),
                Y_train: Y.to_owned(),
                kernel,
                alpha: self.alpha,
                n_outputs: self.n_outputs,
                n_latent: self.n_latent,
                mixing_matrix,
                covariance_inv,
                y_latent,
            },
        })
    }
}

impl MultiOutputGaussianProcessRegressor<MogprTrained> {
    /// Access the trained state
    pub fn trained_state(&self) -> &MogprTrained {
        &self._state
    }

    /// Get the learned mixing matrix
    pub fn mixing_matrix(&self) -> &Array2<f64> {
        &self._state.mixing_matrix
    }

    /// Get the log marginal likelihood for model selection
    #[allow(non_snake_case)]
    pub fn log_marginal_likelihood(&self) -> SklResult<f64> {
        let mut total_ll = 0.0;
        let n_samples = self._state.X_train.nrows();

        for r in 0..self._state.n_latent {
            // Compute kernel matrix
            let K = self
                ._state
                .kernel
                .compute_kernel_matrix(&self._state.X_train, None)?;
            let mut K_reg = K.clone();

            // Add regularization
            for i in 0..n_samples {
                K_reg[[i, i]] += self._state.alpha;
            }

            let chol_decomp = utils::robust_cholesky(&K_reg)?;
            let y = &self._state.y_latent[r];

            // Compute log determinant
            let log_det = chol_decomp.diag().iter().map(|x| x.ln()).sum::<f64>() * 2.0;

            // Solve for alpha = K^{-1} * y
            let alpha = utils::triangular_solve(&chol_decomp, y)?;
            let data_fit = y.dot(&alpha);

            // Log marginal likelihood: -0.5 * (y^T K^{-1} y + log|K| + n*log(2π))
            let ll =
                -0.5 * (data_fit + log_det + n_samples as f64 * (2.0 * std::f64::consts::PI).ln());
            total_ll += ll;
        }

        Ok(total_ll)
    }
}

impl Predict<ArrayView2<'_, f64>, Array2<f64>>
    for MultiOutputGaussianProcessRegressor<MogprTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let (n_test, _) = X.dim();
        let mut predictions = Array2::<f64>::zeros((n_test, self._state.n_outputs));

        // Predict for each latent GP and combine using mixing matrix
        for r in 0..self._state.n_latent {
            // Compute cross-covariance between test and training points
            let X_test_owned = X.to_owned();
            let K_star = self
                ._state
                .kernel
                .compute_kernel_matrix(&self._state.X_train, Some(&X_test_owned))?;

            // Compute predictions for latent GP r
            let y_latent = &self._state.y_latent[r];
            let alpha = utils::triangular_solve(
                &utils::robust_cholesky(&{
                    let K = self
                        ._state
                        .kernel
                        .compute_kernel_matrix(&self._state.X_train, None)?;
                    let mut K_reg = K;
                    for i in 0..self._state.X_train.nrows() {
                        K_reg[[i, i]] += self._state.alpha;
                    }
                    K_reg
                })?,
                y_latent,
            )?;

            let latent_pred = K_star.t().dot(&alpha);

            // Combine predictions using mixing matrix
            for q in 0..self._state.n_outputs {
                let weight = self._state.mixing_matrix[[q, r]];
                for i in 0..n_test {
                    predictions[[i, q]] += weight * latent_pred[i];
                }
            }
        }

        Ok(predictions)
    }
}

/// Solve triangular system for multiple right-hand sides
fn solve_triangular_matrix(L: &Array2<f64>, B: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = L.nrows();
    let m = B.ncols();
    let mut X = Array2::<f64>::zeros((n, m));

    for j in 0..m {
        let b = B.column(j);
        let x = utils::triangular_solve(L, &b.to_owned())?;
        X.column_mut(j).assign(&x);
    }

    Ok(X)
}
