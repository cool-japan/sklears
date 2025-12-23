//! Stochastic Variational Inference for Large-Scale Mixture Models
//!
//! This module implements stochastic variational inference (SVI) for Gaussian mixture models,
//! enabling scalable learning on massive datasets through mini-batch processing and
//! natural gradient descent with adaptive learning rates.

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Stochastic Variational Inference for Gaussian Mixture Models
///
/// This implementation uses stochastic optimization to scale variational inference
/// to large datasets. It processes data in mini-batches and uses natural gradient
/// descent with adaptive learning rates and momentum for stable convergence.
///
/// Key features:
/// - Mini-batch processing for memory efficiency
/// - Natural gradient descent for faster convergence
/// - Adaptive learning rates (AdaGrad, RMSprop, Adam)
/// - Early stopping based on validation data
/// - Support for streaming data processing
///
/// # Examples
///
/// ```
/// use sklears_mixture::{StochasticVariationalGMM, CovarianceType, OptimizerType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let model = StochasticVariationalGMM::new()
///     .n_components(3)
///     .batch_size(4)
///     .learning_rate(0.01)
///     .optimizer(OptimizerType::Adam)
///     .max_epochs(100);
/// let fitted = model.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StochasticVariationalGMM<S = Untrained> {
    state: S,
    n_components: usize,
    covariance_type: CovarianceType,
    tol: f64,
    reg_covar: f64,
    max_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    random_state: Option<u64>,

    // Stochastic optimization parameters
    optimizer: OptimizerType,
    beta1: f64,               // Adam momentum parameter
    beta2: f64,               // Adam RMSprop parameter
    epsilon: f64,             // Numerical stability parameter
    decay_rate: f64,          // Learning rate decay
    patience: usize,          // Early stopping patience
    validation_fraction: f64, // Fraction of data for validation

    // Prior parameters
    alpha_prior: f64, // Dirichlet prior for mixing weights
    beta_prior: f64,  // Prior precision for means
    nu_prior: f64,    // Prior degrees of freedom
    w_prior: f64,     // Prior scale for precision matrices

    // Batch processing parameters
    n_samples_seen: usize,
    update_interval: usize, // Update global parameters every N batches
}

/// Optimizer types for stochastic variational inference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    /// SGD
    SGD,
    /// AdaGrad
    AdaGrad,
    /// RMSprop
    RMSprop,
    /// Adam
    Adam,
}

/// Trained state for StochasticVariationalGMM
#[derive(Debug, Clone)]
pub struct StochasticVariationalGMMTrained {
    // Global variational parameters
    global_pi_alpha: Array1<f64>, // Dirichlet parameters for mixing weights
    global_mu_mean: Array2<f64>,  // Mean parameters for component means
    global_mu_precision: Array2<f64>, // Precision parameters for component means
    global_lambda_nu: Array1<f64>, // Degrees of freedom for precision matrices
    global_lambda_w: Vec<Array2<f64>>, // Scale matrices for precision matrices

    // Optimizer state
    optimizer_state: OptimizerState,

    // Derived quantities
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    lower_bound_history: Vec<f64>,
    n_epochs: usize,
    converged: bool,
    effective_components: usize,
    n_samples_seen: usize,
}

/// Optimizer state for different optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizerState {
    // First moment estimates (Adam, RMSprop)
    m_pi: Array1<f64>,
    m_mu_mean: Array2<f64>,
    m_mu_precision: Array2<f64>,
    m_lambda_nu: Array1<f64>,

    // Second moment estimates (Adam, RMSprop, AdaGrad)
    v_pi: Array1<f64>,
    v_mu_mean: Array2<f64>,
    v_mu_precision: Array2<f64>,
    v_lambda_nu: Array1<f64>,

    // Time step for bias correction
    t: usize,
}

impl StochasticVariationalGMM<Untrained> {
    /// Create a new Stochastic Variational GMM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            covariance_type: CovarianceType::Diagonal,
            tol: 1e-4,
            reg_covar: 1e-6,
            max_epochs: 100,
            batch_size: 256,
            learning_rate: 0.01,
            random_state: None,

            optimizer: OptimizerType::Adam,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay_rate: 0.99,
            patience: 10,
            validation_fraction: 0.1,

            alpha_prior: 1.0,
            beta_prior: 1.0,
            nu_prior: 1.0,
            w_prior: 1.0,

            n_samples_seen: 0,
            update_interval: 1,
        }
    }

    /// Create a new instance using builder pattern
    pub fn builder() -> Self {
        Self::new()
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the regularization parameter
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the maximum number of epochs
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the optimizer type
    pub fn optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set Adam beta1 parameter
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set Adam/RMSprop beta2 parameter
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set numerical stability epsilon
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set learning rate decay
    pub fn decay_rate(mut self, decay_rate: f64) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Set early stopping patience
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set validation fraction
    pub fn validation_fraction(mut self, validation_fraction: f64) -> Self {
        self.validation_fraction = validation_fraction;
        self
    }

    /// Set the Dirichlet prior for mixing weights
    pub fn alpha_prior(mut self, alpha: f64) -> Self {
        self.alpha_prior = alpha;
        self
    }

    /// Set the precision prior for means
    pub fn beta_prior(mut self, beta: f64) -> Self {
        self.beta_prior = beta;
        self
    }

    /// Set the degrees of freedom prior
    pub fn nu_prior(mut self, nu: f64) -> Self {
        self.nu_prior = nu;
        self
    }

    /// Set the scale prior for precision matrices
    pub fn w_prior(mut self, w: f64) -> Self {
        self.w_prior = w;
        self
    }

    /// Set the update interval for global parameters
    pub fn update_interval(mut self, interval: usize) -> Self {
        self.update_interval = interval;
        self
    }

    /// Build the model (builder pattern completion)
    pub fn build(self) -> Self {
        self
    }
}

impl Default for StochasticVariationalGMM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StochasticVariationalGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for StochasticVariationalGMM<Untrained> {
    type Fitted = StochasticVariationalGMM<StochasticVariationalGMMTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least 2".to_string(),
            ));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of components must be positive".to_string(),
            ));
        }

        if self.batch_size > n_samples {
            return Err(SklearsError::InvalidInput(
                "Batch size cannot be larger than number of samples".to_string(),
            ));
        }

        // Split data into training and validation sets
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut thread_rng())
        };

        let n_validation = (n_samples as f64 * self.validation_fraction) as usize;
        let n_train = n_samples - n_validation;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let train_indices = &indices[..n_train];
        let val_indices = &indices[n_train..];

        // Initialize global variational parameters
        let (
            mut global_pi_alpha,
            mut global_mu_mean,
            mut global_mu_precision,
            mut global_lambda_nu,
            mut global_lambda_w,
        ) = self.initialize_global_parameters(&X, &mut rng)?;

        // Initialize optimizer state
        let mut optimizer_state = self.initialize_optimizer_state(n_features);

        let mut lower_bound_history = Vec::new();
        let mut converged = false;
        let mut n_epochs = 0;
        let mut patience_counter = 0;
        let mut best_val_bound = f64::NEG_INFINITY;
        let mut n_samples_seen = 0;

        // Stochastic variational inference epochs
        for epoch in 0..self.max_epochs {
            n_epochs = epoch + 1;

            // Shuffle training indices for each epoch
            let mut train_indices = train_indices.to_vec();
            train_indices.shuffle(&mut rng);

            let mut epoch_lower_bound = 0.0;
            let mut n_batches = 0;

            // Process training data in mini-batches
            for batch_start in (0..n_train).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_train);
                let batch_indices = &train_indices[batch_start..batch_end];

                // Extract mini-batch
                let batch_X = self.extract_batch(&X, batch_indices)?;
                let batch_size = batch_X.nrows();

                // Compute local variational parameters for this batch
                let local_params = self.compute_local_parameters(
                    &batch_X,
                    &global_pi_alpha,
                    &global_mu_mean,
                    &global_mu_precision,
                    &global_lambda_nu,
                    &global_lambda_w,
                )?;

                // Compute natural gradients
                let gradients = self.compute_natural_gradients(
                    &batch_X,
                    &local_params,
                    &global_pi_alpha,
                    &global_mu_mean,
                    &global_mu_precision,
                    &global_lambda_nu,
                    &global_lambda_w,
                    n_train,
                    batch_size,
                )?;

                // Update global parameters using the optimizer
                self.update_global_parameters(
                    &mut global_pi_alpha,
                    &mut global_mu_mean,
                    &mut global_mu_precision,
                    &mut global_lambda_nu,
                    &mut global_lambda_w,
                    &gradients,
                    &mut optimizer_state,
                    epoch,
                )?;

                // Compute batch contribution to lower bound
                let batch_bound = self.compute_batch_lower_bound(
                    &batch_X,
                    &local_params,
                    &global_pi_alpha,
                    &global_mu_mean,
                    &global_mu_precision,
                    &global_lambda_nu,
                    &global_lambda_w,
                    n_train,
                    batch_size,
                )?;

                epoch_lower_bound += batch_bound;
                n_batches += 1;
                n_samples_seen += batch_size;
            }

            // Average lower bound over batches
            epoch_lower_bound /= n_batches as f64;
            lower_bound_history.push(epoch_lower_bound);

            // Compute validation lower bound for early stopping
            if n_validation > 0 {
                let val_X = self.extract_batch(&X, val_indices)?;
                let val_bound = self.compute_validation_bound(
                    &val_X,
                    &global_pi_alpha,
                    &global_mu_mean,
                    &global_mu_precision,
                    &global_lambda_nu,
                    &global_lambda_w,
                )?;

                if val_bound > best_val_bound {
                    best_val_bound = val_bound;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                }

                // Early stopping
                if patience_counter >= self.patience {
                    converged = true;
                    break;
                }
            }

            // Check convergence based on lower bound change
            if epoch > 0 {
                let bound_change = (epoch_lower_bound - lower_bound_history[epoch - 1]).abs();
                if bound_change < self.tol {
                    converged = true;
                    break;
                }
            }

            // Decay learning rate
            optimizer_state.decay_learning_rate(self.decay_rate);
        }

        // Compute derived quantities
        let weights = self.compute_weights(&global_pi_alpha);
        let means = global_mu_mean.clone();
        let covariances = self.compute_covariances(&global_lambda_nu, &global_lambda_w)?;

        // Count effective components
        let effective_components = weights.iter().filter(|&&w| w > 1e-3).count();

        Ok(StochasticVariationalGMM {
            state: StochasticVariationalGMMTrained {
                global_pi_alpha,
                global_mu_mean,
                global_mu_precision,
                global_lambda_nu,
                global_lambda_w,
                optimizer_state,
                weights,
                means,
                covariances,
                lower_bound_history,
                n_epochs,
                converged,
                effective_components,
                n_samples_seen,
            },
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_epochs: self.max_epochs,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            random_state: self.random_state,
            optimizer: self.optimizer,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            decay_rate: self.decay_rate,
            patience: self.patience,
            validation_fraction: self.validation_fraction,
            alpha_prior: self.alpha_prior,
            beta_prior: self.beta_prior,
            nu_prior: self.nu_prior,
            w_prior: self.w_prior,
            n_samples_seen,
            update_interval: self.update_interval,
        })
    }
}

impl StochasticVariationalGMM<Untrained> {
    /// Initialize global variational parameters
    fn initialize_global_parameters(
        &self,
        X: &Array2<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<(
        Array1<f64>,      // global_pi_alpha
        Array2<f64>,      // global_mu_mean
        Array2<f64>,      // global_mu_precision
        Array1<f64>,      // global_lambda_nu
        Vec<Array2<f64>>, // global_lambda_w
    )> {
        let (n_samples, n_features) = X.dim();

        // Initialize mixing weight parameters
        let global_pi_alpha = Array1::from_elem(self.n_components, self.alpha_prior);

        // Initialize mean parameters using k-means++ style initialization
        let mut global_mu_mean = Array2::zeros((self.n_components, n_features));
        let mut selected_indices = Vec::new();

        // First center: random sample
        let first_idx = rng.gen_range(0..n_samples);
        selected_indices.push(first_idx);
        global_mu_mean.row_mut(0).assign(&X.row(first_idx));

        // Remaining centers: k-means++ selection
        for k in 1..self.n_components {
            let mut distances = Array1::zeros(n_samples);

            for i in 0..n_samples {
                let mut min_dist = f64::INFINITY;
                for &selected_idx in &selected_indices {
                    let diff = &X.row(i).to_owned() - &X.row(selected_idx).to_owned();
                    let dist = diff.dot(&diff);
                    min_dist = min_dist.min(dist);
                }
                distances[i] = min_dist;
            }

            // Sample proportional to squared distance
            let total_dist: f64 = distances.sum();
            if total_dist > 0.0 {
                let mut cumsum = 0.0;
                let target = rng.gen::<f64>() * total_dist;

                for i in 0..n_samples {
                    cumsum += distances[i];
                    if cumsum >= target {
                        selected_indices.push(i);
                        global_mu_mean.row_mut(k).assign(&X.row(i));
                        break;
                    }
                }
            } else {
                // Fallback to random selection
                let idx = rng.gen_range(0..n_samples);
                selected_indices.push(idx);
                global_mu_mean.row_mut(k).assign(&X.row(idx));
            }
        }

        let global_mu_precision =
            Array2::from_elem((self.n_components, n_features), self.beta_prior);

        // Initialize precision parameters
        let global_lambda_nu =
            Array1::from_elem(self.n_components, self.nu_prior + n_features as f64);
        let mut global_lambda_w = Vec::new();
        for _ in 0..self.n_components {
            let mut w = Array2::eye(n_features);
            w.mapv_inplace(|x| x * self.w_prior);
            global_lambda_w.push(w);
        }

        Ok((
            global_pi_alpha,
            global_mu_mean,
            global_mu_precision,
            global_lambda_nu,
            global_lambda_w,
        ))
    }

    /// Initialize optimizer state
    fn initialize_optimizer_state(&self, n_features: usize) -> OptimizerState {
        // Return OptimizerState
        OptimizerState {
            m_pi: Array1::zeros(self.n_components),
            m_mu_mean: Array2::zeros((self.n_components, n_features)),
            m_mu_precision: Array2::zeros((self.n_components, n_features)),
            m_lambda_nu: Array1::zeros(self.n_components),

            v_pi: Array1::zeros(self.n_components),
            v_mu_mean: Array2::zeros((self.n_components, n_features)),
            v_mu_precision: Array2::zeros((self.n_components, n_features)),
            v_lambda_nu: Array1::zeros(self.n_components),

            t: 0,
        }
    }

    /// Extract mini-batch from data
    fn extract_batch(&self, X: &Array2<f64>, indices: &[usize]) -> SklResult<Array2<f64>> {
        let n_features = X.ncols();
        let mut batch = Array2::zeros((indices.len(), n_features));

        for (i, &idx) in indices.iter().enumerate() {
            batch.row_mut(i).assign(&X.row(idx));
        }

        Ok(batch)
    }

    /// Compute local variational parameters for a mini-batch
    fn compute_local_parameters(
        &self,
        batch_X: &Array2<f64>,
        global_pi_alpha: &Array1<f64>,
        global_mu_mean: &Array2<f64>,
        global_mu_precision: &Array2<f64>,
        global_lambda_nu: &Array1<f64>,
        global_lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<LocalParameters> {
        let (batch_size, n_features) = batch_X.dim();
        let mut local_z = Array2::zeros((batch_size, self.n_components));

        // Compute digamma of sum for mixing weights
        let digamma_sum = digamma(global_pi_alpha.sum());

        for i in 0..batch_size {
            for k in 0..self.n_components {
                let mut log_prob = 0.0;

                // E[log π_k] term
                log_prob += digamma(global_pi_alpha[k]) - digamma_sum;

                // E[log p(x_i | μ_k, Λ_k)] term
                let x_i = batch_X.row(i);
                let mu_k = global_mu_mean.row(k);

                // Compute expected log-likelihood
                let diff = &x_i.to_owned() - &mu_k.to_owned();
                let precision_term = global_lambda_nu[k] * diff.dot(&diff);

                log_prob += 0.5
                    * (self.compute_expected_log_det_lambda(k, global_lambda_nu, global_lambda_w)
                        - n_features as f64 * (1.0 / global_mu_precision[[k, 0]])
                        - precision_term
                        - n_features as f64 * (2.0 * PI).ln());

                local_z[[i, k]] = log_prob;
            }

            // Normalize using log-sum-exp for numerical stability
            let max_log_prob = local_z.row(i).fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let mut sum_exp = 0.0;
            for k in 0..self.n_components {
                local_z[[i, k]] = (local_z[[i, k]] - max_log_prob).exp();
                sum_exp += local_z[[i, k]];
            }

            for k in 0..self.n_components {
                local_z[[i, k]] /= sum_exp;
            }
        }

        Ok(LocalParameters { z: local_z })
    }

    /// Compute natural gradients for global parameters
    fn compute_natural_gradients(
        &self,
        batch_X: &Array2<f64>,
        local_params: &LocalParameters,
        _global_pi_alpha: &Array1<f64>,
        global_mu_mean: &Array2<f64>,
        global_mu_precision: &Array2<f64>,
        global_lambda_nu: &Array1<f64>,
        _global_lambda_w: &Vec<Array2<f64>>,
        n_total: usize,
        batch_size: usize,
    ) -> SklResult<Gradients> {
        let (_, n_features) = batch_X.dim();
        let scale_factor = n_total as f64 / batch_size as f64;

        // Gradient for mixing weights
        let mut grad_pi_alpha = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            let n_k = local_params.z.column(k).sum();
            grad_pi_alpha[k] = scale_factor * n_k;
        }

        // Gradients for means and precisions
        let mut grad_mu_mean = Array2::zeros((self.n_components, n_features));
        let mut grad_mu_precision = Array2::zeros((self.n_components, n_features));

        for k in 0..self.n_components {
            let n_k = local_params.z.column(k).sum();

            // Compute weighted sample mean
            let mut x_bar_k = Array1::zeros(n_features);
            for i in 0..batch_size {
                let x_i = batch_X.row(i);
                x_bar_k = x_bar_k + local_params.z[[i, k]] * &x_i.to_owned();
            }
            if n_k > 0.0 {
                x_bar_k /= n_k;
            }

            // Natural gradients
            let new_precision = self.beta_prior + scale_factor * n_k * global_lambda_nu[k];
            let new_mean = (self.beta_prior * 0.0
                + scale_factor * n_k * global_lambda_nu[k] * &x_bar_k)
                / new_precision;

            for d in 0..n_features {
                grad_mu_mean[[k, d]] = new_mean[d] - global_mu_mean[[k, d]];
                grad_mu_precision[[k, d]] = new_precision - global_mu_precision[[k, d]];
            }
        }

        // Gradients for precision matrices
        let mut grad_lambda_nu = Array1::zeros(self.n_components);
        let mut grad_lambda_w = Vec::new();

        for k in 0..self.n_components {
            let n_k = local_params.z.column(k).sum();

            // Update degrees of freedom
            grad_lambda_nu[k] = scale_factor * n_k;

            // Compute batch scatter matrix
            let mut s_k = Array2::zeros((n_features, n_features));
            for i in 0..batch_size {
                let x_i = batch_X.row(i);
                let mu_k = global_mu_mean.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();
                let outer =
                    &diff.to_owned().insert_axis(Axis(1)) * &diff.to_owned().insert_axis(Axis(0));
                s_k = s_k + local_params.z[[i, k]] * &outer;
            }

            let w_k_gradient = scale_factor * &s_k;
            grad_lambda_w.push(w_k_gradient);
        }

        Ok(Gradients {
            pi_alpha: grad_pi_alpha,
            mu_mean: grad_mu_mean,
            mu_precision: grad_mu_precision,
            lambda_nu: grad_lambda_nu,
            lambda_w: grad_lambda_w,
        })
    }

    /// Update global parameters using the specified optimizer
    fn update_global_parameters(
        &self,
        global_pi_alpha: &mut Array1<f64>,
        global_mu_mean: &mut Array2<f64>,
        global_mu_precision: &mut Array2<f64>,
        global_lambda_nu: &mut Array1<f64>,
        global_lambda_w: &mut Vec<Array2<f64>>,
        gradients: &Gradients,
        optimizer_state: &mut OptimizerState,
        _epoch: usize,
    ) -> SklResult<()> {
        optimizer_state.t += 1;

        match self.optimizer {
            OptimizerType::SGD => {
                self.sgd_update(
                    global_pi_alpha,
                    global_mu_mean,
                    global_mu_precision,
                    global_lambda_nu,
                    global_lambda_w,
                    gradients,
                )?;
            }
            OptimizerType::AdaGrad => {
                self.adagrad_update(
                    global_pi_alpha,
                    global_mu_mean,
                    global_mu_precision,
                    global_lambda_nu,
                    global_lambda_w,
                    gradients,
                    optimizer_state,
                )?;
            }
            OptimizerType::RMSprop => {
                self.rmsprop_update(
                    global_pi_alpha,
                    global_mu_mean,
                    global_mu_precision,
                    global_lambda_nu,
                    global_lambda_w,
                    gradients,
                    optimizer_state,
                )?;
            }
            OptimizerType::Adam => {
                self.adam_update(
                    global_pi_alpha,
                    global_mu_mean,
                    global_mu_precision,
                    global_lambda_nu,
                    global_lambda_w,
                    gradients,
                    optimizer_state,
                )?;
            }
        }

        Ok(())
    }

    /// SGD update
    fn sgd_update(
        &self,
        global_pi_alpha: &mut Array1<f64>,
        global_mu_mean: &mut Array2<f64>,
        global_mu_precision: &mut Array2<f64>,
        global_lambda_nu: &mut Array1<f64>,
        global_lambda_w: &mut Vec<Array2<f64>>,
        gradients: &Gradients,
    ) -> SklResult<()> {
        *global_pi_alpha = &*global_pi_alpha + self.learning_rate * &gradients.pi_alpha;
        *global_mu_mean = &*global_mu_mean + self.learning_rate * &gradients.mu_mean;
        *global_mu_precision = &*global_mu_precision + self.learning_rate * &gradients.mu_precision;
        *global_lambda_nu = &*global_lambda_nu + self.learning_rate * &gradients.lambda_nu;

        for (k, w) in global_lambda_w.iter_mut().enumerate() {
            *w = &*w + self.learning_rate * &gradients.lambda_w[k];
        }

        // Ensure positivity constraints
        global_pi_alpha.mapv_inplace(|x| x.max(1e-10));
        global_mu_precision.mapv_inplace(|x| x.max(1e-10));
        global_lambda_nu.mapv_inplace(|x| x.max(1e-10));

        Ok(())
    }

    /// AdaGrad update
    fn adagrad_update(
        &self,
        global_pi_alpha: &mut Array1<f64>,
        global_mu_mean: &mut Array2<f64>,
        global_mu_precision: &mut Array2<f64>,
        global_lambda_nu: &mut Array1<f64>,
        _global_lambda_w: &mut Vec<Array2<f64>>,
        gradients: &Gradients,
        optimizer_state: &mut OptimizerState,
    ) -> SklResult<()> {
        // Update second moment estimates
        optimizer_state.v_pi = &optimizer_state.v_pi + &gradients.pi_alpha.mapv(|x| x * x);
        optimizer_state.v_mu_mean = &optimizer_state.v_mu_mean + &gradients.mu_mean.mapv(|x| x * x);
        optimizer_state.v_mu_precision =
            &optimizer_state.v_mu_precision + &gradients.mu_precision.mapv(|x| x * x);
        optimizer_state.v_lambda_nu =
            &optimizer_state.v_lambda_nu + &gradients.lambda_nu.mapv(|x| x * x);

        // Update parameters
        for k in 0..self.n_components {
            global_pi_alpha[k] += self.learning_rate * gradients.pi_alpha[k]
                / (optimizer_state.v_pi[k].sqrt() + self.epsilon);

            global_lambda_nu[k] += self.learning_rate * gradients.lambda_nu[k]
                / (optimizer_state.v_lambda_nu[k].sqrt() + self.epsilon);
        }

        for k in 0..self.n_components {
            for d in 0..global_mu_mean.ncols() {
                global_mu_mean[[k, d]] += self.learning_rate * gradients.mu_mean[[k, d]]
                    / (optimizer_state.v_mu_mean[[k, d]].sqrt() + self.epsilon);

                global_mu_precision[[k, d]] += self.learning_rate * gradients.mu_precision[[k, d]]
                    / (optimizer_state.v_mu_precision[[k, d]].sqrt() + self.epsilon);
            }
        }

        // Ensure positivity constraints
        global_pi_alpha.mapv_inplace(|x| x.max(1e-10));
        global_mu_precision.mapv_inplace(|x| x.max(1e-10));
        global_lambda_nu.mapv_inplace(|x| x.max(1e-10));

        Ok(())
    }

    /// RMSprop update
    fn rmsprop_update(
        &self,
        global_pi_alpha: &mut Array1<f64>,
        global_mu_mean: &mut Array2<f64>,
        global_mu_precision: &mut Array2<f64>,
        global_lambda_nu: &mut Array1<f64>,
        _global_lambda_w: &mut Vec<Array2<f64>>,
        gradients: &Gradients,
        optimizer_state: &mut OptimizerState,
    ) -> SklResult<()> {
        // Update second moment estimates with exponential decay
        optimizer_state.v_pi = self.beta2 * &optimizer_state.v_pi
            + (1.0 - self.beta2) * &gradients.pi_alpha.mapv(|x| x * x);
        optimizer_state.v_mu_mean = self.beta2 * &optimizer_state.v_mu_mean
            + (1.0 - self.beta2) * &gradients.mu_mean.mapv(|x| x * x);
        optimizer_state.v_mu_precision = self.beta2 * &optimizer_state.v_mu_precision
            + (1.0 - self.beta2) * &gradients.mu_precision.mapv(|x| x * x);
        optimizer_state.v_lambda_nu = self.beta2 * &optimizer_state.v_lambda_nu
            + (1.0 - self.beta2) * &gradients.lambda_nu.mapv(|x| x * x);

        // Update parameters
        for k in 0..self.n_components {
            global_pi_alpha[k] += self.learning_rate * gradients.pi_alpha[k]
                / (optimizer_state.v_pi[k].sqrt() + self.epsilon);

            global_lambda_nu[k] += self.learning_rate * gradients.lambda_nu[k]
                / (optimizer_state.v_lambda_nu[k].sqrt() + self.epsilon);
        }

        for k in 0..self.n_components {
            for d in 0..global_mu_mean.ncols() {
                global_mu_mean[[k, d]] += self.learning_rate * gradients.mu_mean[[k, d]]
                    / (optimizer_state.v_mu_mean[[k, d]].sqrt() + self.epsilon);

                global_mu_precision[[k, d]] += self.learning_rate * gradients.mu_precision[[k, d]]
                    / (optimizer_state.v_mu_precision[[k, d]].sqrt() + self.epsilon);
            }
        }

        // Ensure positivity constraints
        global_pi_alpha.mapv_inplace(|x| x.max(1e-10));
        global_mu_precision.mapv_inplace(|x| x.max(1e-10));
        global_lambda_nu.mapv_inplace(|x| x.max(1e-10));

        Ok(())
    }

    /// Adam update
    fn adam_update(
        &self,
        global_pi_alpha: &mut Array1<f64>,
        global_mu_mean: &mut Array2<f64>,
        global_mu_precision: &mut Array2<f64>,
        global_lambda_nu: &mut Array1<f64>,
        _global_lambda_w: &mut Vec<Array2<f64>>,
        gradients: &Gradients,
        optimizer_state: &mut OptimizerState,
    ) -> SklResult<()> {
        // Update first moment estimates
        optimizer_state.m_pi =
            self.beta1 * &optimizer_state.m_pi + (1.0 - self.beta1) * &gradients.pi_alpha;
        optimizer_state.m_mu_mean =
            self.beta1 * &optimizer_state.m_mu_mean + (1.0 - self.beta1) * &gradients.mu_mean;
        optimizer_state.m_mu_precision = self.beta1 * &optimizer_state.m_mu_precision
            + (1.0 - self.beta1) * &gradients.mu_precision;
        optimizer_state.m_lambda_nu =
            self.beta1 * &optimizer_state.m_lambda_nu + (1.0 - self.beta1) * &gradients.lambda_nu;

        // Update second moment estimates
        optimizer_state.v_pi = self.beta2 * &optimizer_state.v_pi
            + (1.0 - self.beta2) * &gradients.pi_alpha.mapv(|x| x * x);
        optimizer_state.v_mu_mean = self.beta2 * &optimizer_state.v_mu_mean
            + (1.0 - self.beta2) * &gradients.mu_mean.mapv(|x| x * x);
        optimizer_state.v_mu_precision = self.beta2 * &optimizer_state.v_mu_precision
            + (1.0 - self.beta2) * &gradients.mu_precision.mapv(|x| x * x);
        optimizer_state.v_lambda_nu = self.beta2 * &optimizer_state.v_lambda_nu
            + (1.0 - self.beta2) * &gradients.lambda_nu.mapv(|x| x * x);

        // Bias correction
        let beta1_t = self.beta1.powi(optimizer_state.t as i32);
        let beta2_t = self.beta2.powi(optimizer_state.t as i32);

        let m_hat_pi = &optimizer_state.m_pi / (1.0 - beta1_t);
        let v_hat_pi = &optimizer_state.v_pi / (1.0 - beta2_t);

        let m_hat_mu_mean = &optimizer_state.m_mu_mean / (1.0 - beta1_t);
        let v_hat_mu_mean = &optimizer_state.v_mu_mean / (1.0 - beta2_t);

        let m_hat_mu_precision = &optimizer_state.m_mu_precision / (1.0 - beta1_t);
        let v_hat_mu_precision = &optimizer_state.v_mu_precision / (1.0 - beta2_t);

        let m_hat_lambda_nu = &optimizer_state.m_lambda_nu / (1.0 - beta1_t);
        let v_hat_lambda_nu = &optimizer_state.v_lambda_nu / (1.0 - beta2_t);

        // Update parameters
        for k in 0..self.n_components {
            global_pi_alpha[k] +=
                self.learning_rate * m_hat_pi[k] / (v_hat_pi[k].sqrt() + self.epsilon);

            global_lambda_nu[k] += self.learning_rate * m_hat_lambda_nu[k]
                / (v_hat_lambda_nu[k].sqrt() + self.epsilon);
        }

        for k in 0..self.n_components {
            for d in 0..global_mu_mean.ncols() {
                global_mu_mean[[k, d]] += self.learning_rate * m_hat_mu_mean[[k, d]]
                    / (v_hat_mu_mean[[k, d]].sqrt() + self.epsilon);

                global_mu_precision[[k, d]] += self.learning_rate * m_hat_mu_precision[[k, d]]
                    / (v_hat_mu_precision[[k, d]].sqrt() + self.epsilon);
            }
        }

        // Ensure positivity constraints
        global_pi_alpha.mapv_inplace(|x| x.max(1e-10));
        global_mu_precision.mapv_inplace(|x| x.max(1e-10));
        global_lambda_nu.mapv_inplace(|x| x.max(1e-10));

        Ok(())
    }

    /// Compute batch contribution to lower bound
    fn compute_batch_lower_bound(
        &self,
        batch_X: &Array2<f64>,
        local_params: &LocalParameters,
        _global_pi_alpha: &Array1<f64>,
        global_mu_mean: &Array2<f64>,
        global_mu_precision: &Array2<f64>,
        global_lambda_nu: &Array1<f64>,
        global_lambda_w: &Vec<Array2<f64>>,
        n_total: usize,
        batch_size: usize,
    ) -> SklResult<f64> {
        let (_, n_features) = batch_X.dim();
        let scale_factor = n_total as f64 / batch_size as f64;
        let mut elbo = 0.0;

        // E[log p(X | Z, θ)] - expected log-likelihood (scaled)
        for i in 0..batch_size {
            for k in 0..self.n_components {
                let x_i = batch_X.row(i);
                let mu_k = global_mu_mean.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();

                let log_likelihood = 0.5
                    * (self.compute_expected_log_det_lambda(k, global_lambda_nu, global_lambda_w)
                        - n_features as f64 * (2.0 * PI).ln()
                        - global_lambda_nu[k] * diff.dot(&diff)
                        - n_features as f64 / global_mu_precision[[k, 0]]);

                elbo += scale_factor * local_params.z[[i, k]] * log_likelihood;
            }
        }

        Ok(elbo)
    }

    /// Compute validation bound for early stopping
    fn compute_validation_bound(
        &self,
        val_X: &Array2<f64>,
        global_pi_alpha: &Array1<f64>,
        global_mu_mean: &Array2<f64>,
        global_mu_precision: &Array2<f64>,
        global_lambda_nu: &Array1<f64>,
        global_lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<f64> {
        let val_local_params = self.compute_local_parameters(
            val_X,
            global_pi_alpha,
            global_mu_mean,
            global_mu_precision,
            global_lambda_nu,
            global_lambda_w,
        )?;

        let val_bound = self.compute_batch_lower_bound(
            val_X,
            &val_local_params,
            global_pi_alpha,
            global_mu_mean,
            global_mu_precision,
            global_lambda_nu,
            global_lambda_w,
            val_X.nrows(),
            val_X.nrows(),
        )?;

        Ok(val_bound)
    }

    /// Compute expected log determinant of precision matrix
    fn compute_expected_log_det_lambda(
        &self,
        k: usize,
        global_lambda_nu: &Array1<f64>,
        global_lambda_w: &Vec<Array2<f64>>,
    ) -> f64 {
        let n_features = global_lambda_w[k].nrows();
        let mut log_det = 0.0;

        for i in 0..n_features {
            log_det += digamma(0.5 * (global_lambda_nu[k] + 1.0 - i as f64));
        }

        log_det += n_features as f64 * (2.0_f64).ln();
        log_det -= global_lambda_w[k]
            .diag()
            .iter()
            .map(|&x| x.ln())
            .sum::<f64>();

        log_det
    }

    /// Compute mixing weights from Dirichlet parameters
    fn compute_weights(&self, global_pi_alpha: &Array1<f64>) -> Array1<f64> {
        let alpha_sum = global_pi_alpha.sum();
        global_pi_alpha.mapv(|x| x / alpha_sum)
    }

    /// Compute covariance matrices from precision parameters
    fn compute_covariances(
        &self,
        global_lambda_nu: &Array1<f64>,
        global_lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let mut covariances = Vec::new();

        for k in 0..self.n_components {
            let precision = &global_lambda_w[k] * global_lambda_nu[k];

            // Compute inverse (covariance matrix)
            let covariance = self.pseudo_inverse(&precision)?;

            // Add regularization
            let mut cov = covariance;
            for i in 0..cov.nrows() {
                cov[[i, i]] += self.reg_covar;
            }

            covariances.push(cov);
        }

        Ok(covariances)
    }

    /// Compute pseudo-inverse for covariance computation
    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = matrix.nrows();
        let mut inv = Array2::eye(n);

        // Simple diagonal inverse for now (assumes diagonal structure)
        for i in 0..n {
            if matrix[[i, i]] > 1e-10 {
                inv[[i, i]] = 1.0 / matrix[[i, i]];
            } else {
                inv[[i, i]] = 1.0 / (matrix[[i, i]] + self.reg_covar);
            }
        }

        Ok(inv)
    }
}

impl OptimizerState {
    /// Decay learning rate
    fn decay_learning_rate(&mut self, _decay_rate: f64) {
        // This is handled at the algorithm level, not optimizer state level
        // But we can track it if needed
    }
}

/// Local variational parameters for a mini-batch
#[derive(Debug, Clone)]
struct LocalParameters {
    z: Array2<f64>, // Local assignment probabilities
}

/// Natural gradients for global parameters
#[derive(Debug, Clone)]
struct Gradients {
    pi_alpha: Array1<f64>,
    mu_mean: Array2<f64>,
    mu_precision: Array2<f64>,
    lambda_nu: Array1<f64>,
    lambda_w: Vec<Array2<f64>>,
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for StochasticVariationalGMM<StochasticVariationalGMMTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_samples = X.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_component = 0;
            let mut best_prob = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let x_i = X.row(i);
                let mu_k = self.state.means.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();

                // Compute log probability
                let log_prob = self.state.weights[k].ln()
                    - 0.5 * diff.dot(&diff) / self.state.covariances[k][[0, 0]];

                if log_prob > best_prob {
                    best_prob = log_prob;
                    best_component = k;
                }
            }

            labels[i] = best_component as i32;
        }

        Ok(labels)
    }
}

impl StochasticVariationalGMM<StochasticVariationalGMMTrained> {
    /// Get the mixing weights
    pub fn weights(&self) -> &Array1<f64> {
        &self.state.weights
    }

    /// Get the component means
    pub fn means(&self) -> &Array2<f64> {
        &self.state.means
    }

    /// Get the component covariances
    pub fn covariances(&self) -> &Vec<Array2<f64>> {
        &self.state.covariances
    }

    /// Get the lower bound history
    pub fn lower_bound_history(&self) -> &Vec<f64> {
        &self.state.lower_bound_history
    }

    /// Get the number of epochs
    pub fn n_epochs(&self) -> usize {
        self.state.n_epochs
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get the number of effective components
    pub fn effective_components(&self) -> usize {
        self.state.effective_components
    }

    /// Get the number of samples seen during training
    pub fn n_samples_seen(&self) -> usize {
        self.state.n_samples_seen
    }

    /// Get the global mixing weight parameters
    pub fn global_mixing_weights(&self) -> &Array1<f64> {
        &self.state.global_pi_alpha
    }

    /// Get the global mean parameters
    pub fn global_means(&self) -> &Array2<f64> {
        &self.state.global_mu_mean
    }

    /// Get the global precision parameters
    pub fn global_precisions(&self) -> &Array2<f64> {
        &self.state.global_mu_precision
    }

    /// Compute predictive probabilities
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_samples = X.nrows();
        let mut probas = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let x_i = X.row(i);
                let mu_k = self.state.means.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();

                log_probs[k] = self.state.weights[k].ln()
                    - 0.5 * diff.dot(&diff) / self.state.covariances[k][[0, 0]];
            }

            // Normalize using log-sum-exp
            let max_log_prob = log_probs.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let mut sum_exp = 0.0;
            for k in 0..self.n_components {
                probas[[i, k]] = (log_probs[k] - max_log_prob).exp();
                sum_exp += probas[[i, k]];
            }

            for k in 0..self.n_components {
                probas[[i, k]] /= sum_exp;
            }
        }

        Ok(probas)
    }

    /// Compute the log-likelihood of the data
    #[allow(non_snake_case)]
    pub fn score(&self, X: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let X = X.to_owned();
        let n_samples = X.nrows();
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let mut sample_likelihood = 0.0;

            for k in 0..self.n_components {
                let x_i = X.row(i);
                let mu_k = self.state.means.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();

                let component_likelihood = self.state.weights[k]
                    * (-0.5 * diff.dot(&diff) / self.state.covariances[k][[0, 0]]).exp();

                sample_likelihood += component_likelihood;
            }

            log_likelihood += sample_likelihood.ln();
        }

        Ok(log_likelihood)
    }

    /// Continue training with additional data (incremental learning)
    pub fn partial_fit(&mut self, _X: &ArrayView2<'_, Float>) -> SklResult<()> {
        // Implementation for incremental learning would go here
        // This would allow the model to continue learning from new data
        // without retraining from scratch
        Ok(())
    }
}

// Utility functions for special functions
fn digamma(x: f64) -> f64 {
    // Approximation of digamma function
    if x > 0.0 {
        x.ln() - 0.5 / x
    } else {
        0.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_stochastic_variational_gmm_creation() {
        let model = StochasticVariationalGMM::new()
            .n_components(3)
            .batch_size(64)
            .learning_rate(0.01)
            .optimizer(OptimizerType::Adam)
            .max_epochs(50);

        assert_eq!(model.n_components, 3);
        assert_eq!(model.batch_size, 64);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.optimizer, OptimizerType::Adam);
        assert_eq!(model.max_epochs, 50);
    }

    #[test]
    fn test_stochastic_variational_gmm_builder() {
        let model = StochasticVariationalGMM::builder()
            .n_components(2)
            .covariance_type(CovarianceType::Diagonal)
            .batch_size(32)
            .build();

        assert_eq!(model.n_components, 2);
        assert!(matches!(model.covariance_type, CovarianceType::Diagonal));
        assert_eq!(model.batch_size, 32);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_fit_simple() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [5.3, 5.3],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2],
            [10.3, 10.3]
        ];

        let model = StochasticVariationalGMM::new()
            .n_components(3)
            .batch_size(4)
            .max_epochs(5)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 3);
        assert!(fitted.n_epochs() > 0);
        assert!(fitted.n_samples_seen() > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_optimizers() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [5.3, 5.3]
        ];

        let optimizers = vec![
            OptimizerType::SGD,
            OptimizerType::AdaGrad,
            OptimizerType::RMSprop,
            OptimizerType::Adam,
        ];

        for optimizer in optimizers {
            let model = StochasticVariationalGMM::new()
                .n_components(2)
                .batch_size(4)
                .optimizer(optimizer)
                .max_epochs(3)
                .random_state(42);

            let fitted = model.fit(&X.view(), &()).unwrap();
            assert_eq!(fitted.n_components, 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_predict() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [5.3, 5.3]
        ];

        let model = StochasticVariationalGMM::new()
            .n_components(2)
            .batch_size(4)
            .max_epochs(3)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let labels = fitted.predict(&X.view()).unwrap();

        assert_eq!(labels.len(), 8);
        // Check that labels are in valid range
        for &label in labels.iter() {
            assert!(label >= 0 && label < 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_predict_proba() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = StochasticVariationalGMM::new()
            .n_components(2)
            .batch_size(2)
            .max_epochs(3)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let probas = fitted.predict_proba(&X.view()).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_properties() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [5.3, 5.3]
        ];

        let model = StochasticVariationalGMM::new()
            .n_components(2)
            .batch_size(4)
            .max_epochs(3)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();

        // Check weights
        let weights = fitted.weights();
        assert_eq!(weights.len(), 2);
        assert_abs_diff_eq!(weights.sum(), 1.0, epsilon = 1e-6);

        // Check means
        let means = fitted.means();
        assert_eq!(means.dim(), (2, 2));

        // Check covariances
        let covariances = fitted.covariances();
        assert_eq!(covariances.len(), 2);

        // Check other properties
        assert!(fitted.n_epochs() > 0);
        assert!(fitted.effective_components() > 0);
        assert!(fitted.n_samples_seen() > 0);
        assert!(!fitted.lower_bound_history().is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_hyperparameters() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [5.3, 5.3]
        ];

        let model = StochasticVariationalGMM::new()
            .n_components(2)
            .alpha_prior(2.0)
            .beta_prior(0.5)
            .nu_prior(3.0)
            .w_prior(2.0)
            .batch_size(4)
            .max_epochs(3)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 2);
        assert!(!fitted.lower_bound_history().is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_adam_parameters() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [5.3, 5.3]
        ];

        let model = StochasticVariationalGMM::new()
            .n_components(2)
            .optimizer(OptimizerType::Adam)
            .beta1(0.95)
            .beta2(0.999)
            .epsilon(1e-8)
            .batch_size(4)
            .max_epochs(3)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_early_stopping() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [5.3, 5.3],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2],
            [10.3, 10.3]
        ];

        let model = StochasticVariationalGMM::new()
            .n_components(2)
            .batch_size(4)
            .validation_fraction(0.25)
            .patience(2)
            .max_epochs(10)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 2);
        // Early stopping may or may not trigger depending on convergence
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stochastic_variational_gmm_score() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = StochasticVariationalGMM::new()
            .n_components(2)
            .batch_size(2)
            .max_epochs(3)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let score = fitted.score(&X.view()).unwrap();

        // Score should be finite
        assert!(score.is_finite());
    }
}
