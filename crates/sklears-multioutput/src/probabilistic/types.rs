//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use super::functions::{ArrayDeterminant, ArrayInverse};
pub use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
pub use scirs2_core::random::thread_rng;
pub use scirs2_core::random::{RandNormal, Rng};
pub use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
pub use std::marker::PhantomData;

/// Bayesian Multi-Output Model configuration
#[derive(Debug, Clone)]
pub struct BayesianMultiOutputConfig {
    /// Prior distribution for weights
    pub weight_prior: PriorDistribution,
    /// Prior distribution for bias
    pub bias_prior: PriorDistribution,
    /// Prior distribution for noise variance
    pub noise_prior: PriorDistribution,
    /// Inference method
    pub inference_method: InferenceMethod,
    /// Number of MCMC samples
    pub n_samples: usize,
    /// Burn-in period for MCMC
    pub burn_in: usize,
    /// Thinning interval for MCMC
    pub thin: usize,
    /// Maximum number of iterations for variational inference
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}
/// Ensemble strategy for combining Bayesian models
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleStrategy {
    /// Bayesian Model Averaging - weight models by their marginal likelihood
    BayesianAveraging,
    /// Equal weighting of all models
    EqualWeight,
    /// Product of Experts - multiply predictions (for independent experts)
    ProductOfExperts,
    /// Mixture of Experts - learn gating function
    MixtureOfExperts,
    /// Committee Machine - robust aggregation
    CommitteeMachine,
}
/// Trained state for Ensemble Bayesian Model
#[derive(Debug, Clone)]
pub struct EnsembleBayesianModelTrained {
    /// Trained individual models
    pub models: Vec<BayesianMultiOutputModelTrained>,
    /// Model weights (based on marginal likelihood or learned)
    pub model_weights: Array1<Float>,
    /// Log marginal likelihood for the ensemble
    pub ensemble_log_likelihood: Float,
    /// Number of features
    pub n_features: usize,
    /// Number of outputs
    pub n_outputs: usize,
    /// Configuration
    pub config: EnsembleBayesianConfig,
}
/// Prior distribution types for Bayesian models
#[derive(Debug, Clone, PartialEq)]
pub enum PriorDistribution {
    /// Normal distribution with mean and variance
    Normal(Float, Float),
    /// Laplace distribution with location and scale
    Laplace(Float, Float),
    /// Gamma distribution with shape and rate
    Gamma(Float, Float),
    /// Uniform distribution with lower and upper bounds
    Uniform(Float, Float),
    /// Hierarchical prior
    Hierarchical,
}
/// Inference method for Bayesian models
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceMethod {
    /// Variational Bayesian inference
    Variational,
    /// Markov Chain Monte Carlo
    MCMC,
    /// Expectation-Maximization
    EM,
    /// Laplace approximation
    Laplace,
    /// Exact inference (when possible)
    Exact,
}
/// Kernel function types for Gaussian Processes
#[derive(Debug, Clone, PartialEq)]
pub enum KernelFunction {
    /// RBF (Gaussian) kernel with length scale
    RBF(Float),
    /// Matern kernel with length scale and nu parameter
    Matern(Float, Float),
    /// Linear kernel
    Linear,
    /// Polynomial kernel with degree and offset
    Polynomial(usize, Float),
    /// Rational Quadratic kernel with alpha and length scale
    RationalQuadratic(Float, Float),
}
/// Configuration for ensemble Bayesian methods
#[derive(Debug, Clone)]
pub struct EnsembleBayesianConfig {
    /// Number of models in ensemble
    pub n_models: usize,
    /// Strategy for combining predictions
    pub strategy: EnsembleStrategy,
    /// Bootstrap sample ratio (1.0 = same size as training data)
    pub bootstrap_ratio: Float,
    /// Configuration for individual models
    pub base_config: BayesianMultiOutputConfig,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Prior weight on each model (for Bayesian averaging)
    pub model_prior: Vec<Float>,
}
/// Gaussian Process Multi-Output Model
///
/// Implements Gaussian Process regression for multi-output problems using
/// various covariance functions and supports uncertainty quantification.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::probabilistic::{GaussianProcessMultiOutput, KernelFunction};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
///
/// let gp = GaussianProcessMultiOutput::new()
///     .kernel(KernelFunction::RBF(1.0))
///     .noise_level(0.1)
///     .random_state(Some(42));
/// ```
#[derive(Debug, Clone)]
pub struct GaussianProcessMultiOutput<S = Untrained> {
    pub(crate) state: S,
    pub(crate) kernel: KernelFunction,
    pub(crate) noise_level: Float,
    pub(crate) normalize_y: bool,
    pub(crate) random_state: Option<u64>,
}
impl GaussianProcessMultiOutput<Untrained> {
    /// Create a new GaussianProcessMultiOutput instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: KernelFunction::RBF(1.0),
            noise_level: 1e-10,
            normalize_y: false,
            random_state: None,
        }
    }
    /// Set the kernel function
    pub fn kernel(mut self, kernel: KernelFunction) -> Self {
        self.kernel = kernel;
        self
    }
    /// Set the noise level
    pub fn noise_level(mut self, noise: Float) -> Self {
        self.noise_level = noise;
        self
    }
    /// Set whether to normalize target values
    pub fn normalize_y(mut self, normalize: bool) -> Self {
        self.normalize_y = normalize;
        self
    }
    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }
    /// Fit the Gaussian Process model
    #[allow(non_snake_case)]
    pub fn fit(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView2<Float>,
    ) -> SklResult<GaussianProcessMultiOutput<GaussianProcessMultiOutputTrained>> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_outputs = y.ncols();
        let X_train = X.to_owned();
        let mut y_train = y.to_owned();
        let (y_mean, y_std) = if self.normalize_y {
            let mean = y.mean_axis(Axis(0)).unwrap();
            let std = y.std_axis(Axis(0), 0.0);
            for i in 0..n_samples {
                for j in 0..n_outputs {
                    y_train[[i, j]] = (y_train[[i, j]] - mean[j]) / std[j];
                }
            }
            (mean, std)
        } else {
            (Array1::<Float>::zeros(n_outputs), Array1::ones(n_outputs))
        };
        let mut K = self.compute_kernel_matrix(&X_train.view(), &X_train.view());
        for i in 0..n_samples {
            K[[i, i]] += self.noise_level;
        }
        let K_inv = self.compute_kernel_inverse(&K)?;
        Ok(GaussianProcessMultiOutput {
            state: GaussianProcessMultiOutputTrained {
                X_train,
                y_train,
                K_inv,
                y_mean,
                y_std,
                kernel: self.kernel.clone(),
                noise_level: self.noise_level,
                normalize_y: self.normalize_y,
                n_features,
                n_outputs,
            },
            kernel: self.kernel.clone(),
            noise_level: self.noise_level,
            normalize_y: self.normalize_y,
            random_state: self.random_state,
        })
    }
    /// Compute kernel matrix between two sets of points
    fn compute_kernel_matrix(
        &self,
        X1: &ArrayView2<Float>,
        X2: &ArrayView2<Float>,
    ) -> Array2<Float> {
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<Float>::zeros((n1, n2));
        for i in 0..n1 {
            for j in 0..n2 {
                K[[i, j]] = self.kernel_function(&X1.row(i), &X2.row(j));
            }
        }
        K
    }
    /// Compute kernel function between two points
    fn kernel_function(&self, x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> Float {
        match &self.kernel {
            KernelFunction::RBF(length_scale) => {
                let dist_sq = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>();
                (-dist_sq / (2.0 * length_scale.powi(2))).exp()
            }
            KernelFunction::Matern(length_scale, nu) => {
                let dist = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();
                if dist == 0.0 {
                    1.0
                } else {
                    let sqrt_2nu = (2.0 * nu).sqrt();
                    let arg = sqrt_2nu * dist / length_scale;
                    if (nu - 0.5).abs() < 1e-10 {
                        (-arg).exp()
                    } else if (nu - 1.5).abs() < 1e-10 {
                        (1.0 + arg) * (-arg).exp()
                    } else if (nu - 2.5).abs() < 1e-10 {
                        (1.0 + arg + arg.powi(2) / 3.0) * (-arg).exp()
                    } else {
                        (-arg).exp()
                    }
                }
            }
            KernelFunction::Linear => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum(),
            KernelFunction::Polynomial(degree, offset) => {
                let dot_product = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<Float>();
                (dot_product + offset).powi(*degree as i32)
            }
            KernelFunction::RationalQuadratic(alpha, length_scale) => {
                let dist_sq = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>();
                (1.0 + dist_sq / (2.0 * alpha * length_scale.powi(2))).powf(-alpha)
            }
        }
    }
    /// Compute inverse of kernel matrix
    fn compute_kernel_inverse(&self, K: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n = K.nrows();
        let mut K_inv = Array2::eye(n);
        let mut K_work = K.clone();
        for i in 0..n {
            let pivot = K_work[[i, i]];
            if pivot.abs() < 1e-12 {
                return Err(SklearsError::InvalidInput(
                    "Kernel matrix is singular".to_string(),
                ));
            }
            for j in 0..n {
                K_work[[i, j]] /= pivot;
                K_inv[[i, j]] /= pivot;
            }
            for k in 0..n {
                if k != i {
                    let factor = K_work[[k, i]];
                    for j in 0..n {
                        K_work[[k, j]] -= factor * K_work[[i, j]];
                        K_inv[[k, j]] -= factor * K_inv[[i, j]];
                    }
                }
            }
        }
        Ok(K_inv)
    }
}
impl GaussianProcessMultiOutput<GaussianProcessMultiOutputTrained> {
    /// Predict using the Gaussian Process model
    pub fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }
        let (mean, _) = self.predict_with_variance(X)?;
        Ok(mean)
    }
    /// Predict with uncertainty estimation
    #[allow(non_snake_case)]
    pub fn predict_with_variance(
        &self,
        X: &ArrayView2<Float>,
    ) -> SklResult<(Array2<Float>, Array2<Float>)> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }
        let n_test = X.nrows();
        let n_outputs = self.state.n_outputs;
        let K_star = self.compute_kernel_matrix(X, &self.state.X_train.view());
        let K_star_star = self.compute_kernel_matrix(X, X);
        let alpha = K_star.dot(&self.state.K_inv.dot(&self.state.y_train));
        let mut mean = alpha;
        let v = K_star.dot(&self.state.K_inv);
        let mut variance = Array2::<Float>::zeros((n_test, n_outputs));
        for i in 0..n_test {
            let pred_var = K_star_star[[i, i]] - v.row(i).dot(&K_star.row(i));
            for j in 0..n_outputs {
                variance[[i, j]] = pred_var.max(self.state.noise_level);
            }
        }
        if self.state.normalize_y {
            for i in 0..n_test {
                for j in 0..n_outputs {
                    mean[[i, j]] = mean[[i, j]] * self.state.y_std[j] + self.state.y_mean[j];
                    variance[[i, j]] *= self.state.y_std[j].powi(2);
                }
            }
        }
        Ok((mean, variance))
    }
    /// Compute kernel matrix between two sets of points
    fn compute_kernel_matrix(
        &self,
        X1: &ArrayView2<Float>,
        X2: &ArrayView2<Float>,
    ) -> Array2<Float> {
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::<Float>::zeros((n1, n2));
        for i in 0..n1 {
            for j in 0..n2 {
                K[[i, j]] = self.kernel_function(&X1.row(i), &X2.row(j));
            }
        }
        K
    }
    /// Compute kernel function between two points
    fn kernel_function(&self, x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> Float {
        match &self.state.kernel {
            KernelFunction::RBF(length_scale) => {
                let dist_sq = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>();
                (-dist_sq / (2.0 * length_scale.powi(2))).exp()
            }
            KernelFunction::Matern(length_scale, nu) => {
                let dist = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();
                if dist == 0.0 {
                    1.0
                } else {
                    let sqrt_2nu = (2.0 * nu).sqrt();
                    let arg = sqrt_2nu * dist / length_scale;
                    if (nu - 0.5).abs() < 1e-10 {
                        (-arg).exp()
                    } else if (nu - 1.5).abs() < 1e-10 {
                        (1.0 + arg) * (-arg).exp()
                    } else if (nu - 2.5).abs() < 1e-10 {
                        (1.0 + arg + arg.powi(2) / 3.0) * (-arg).exp()
                    } else {
                        (-arg).exp()
                    }
                }
            }
            KernelFunction::Linear => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum(),
            KernelFunction::Polynomial(degree, offset) => {
                let dot_product = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<Float>();
                (dot_product + offset).powi(*degree as i32)
            }
            KernelFunction::RationalQuadratic(alpha, length_scale) => {
                let dist_sq = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>();
                (1.0 + dist_sq / (2.0 * alpha * length_scale.powi(2))).powf(-alpha)
            }
        }
    }
    /// Get the training data
    pub fn training_data(&self) -> (&Array2<Float>, &Array2<Float>) {
        (&self.state.X_train, &self.state.y_train)
    }
    /// Get the kernel function
    pub fn kernel(&self) -> &KernelFunction {
        &self.state.kernel
    }
}
/// Bayesian Multi-Output Model
#[derive(Debug, Clone)]
pub struct BayesianMultiOutputModel<S = Untrained> {
    pub(crate) state: S,
    pub(crate) config: BayesianMultiOutputConfig,
}
impl BayesianMultiOutputModel<Untrained> {
    /// Create a new Bayesian Multi-Output Model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: BayesianMultiOutputConfig::default(),
        }
    }
    /// Set the configuration
    pub fn config(mut self, config: BayesianMultiOutputConfig) -> Self {
        self.config = config;
        self
    }
    /// Set the weight prior distribution
    pub fn weight_prior(mut self, weight_prior: PriorDistribution) -> Self {
        self.config.weight_prior = weight_prior;
        self
    }
    /// Set the bias prior distribution
    pub fn bias_prior(mut self, bias_prior: PriorDistribution) -> Self {
        self.config.bias_prior = bias_prior;
        self
    }
    /// Set the noise prior distribution
    pub fn noise_prior(mut self, noise_prior: PriorDistribution) -> Self {
        self.config.noise_prior = noise_prior;
        self
    }
    /// Set the inference method
    pub fn inference_method(mut self, inference_method: InferenceMethod) -> Self {
        self.config.inference_method = inference_method;
        self
    }
    /// Set the number of MCMC samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.config.n_samples = n_samples;
        self
    }
    /// Set the burn-in period
    pub fn burn_in(mut self, burn_in: usize) -> Self {
        self.config.burn_in = burn_in;
        self
    }
    /// Set the thinning interval
    pub fn thin(mut self, thin: usize) -> Self {
        self.config.thin = thin;
        self
    }
    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }
    /// Set the convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }
    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}
impl BayesianMultiOutputModel<Untrained> {
    /// Variational Bayesian inference
    pub(crate) fn variational_inference(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(
        PosteriorDistribution,
        PosteriorDistribution,
        PosteriorDistribution,
        Float,
        Vec<Float>,
    )> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        let mut weight_mean = Array2::<Float>::zeros((n_features, n_outputs));
        for i in 0..n_features {
            for j in 0..n_outputs {
                weight_mean[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut weight_log_var = Array2::<Float>::zeros((n_features, n_outputs));
        let mut bias_mean = Array1::<Float>::zeros(n_outputs);
        let mut bias_log_var = Array1::<Float>::zeros(n_outputs);
        let mut noise_log_var = Array1::<Float>::zeros(n_outputs);
        let mut elbo_history = Vec::new();
        for iteration in 0..self.config.max_iter {
            let elbo = self.compute_elbo(
                X,
                y,
                &weight_mean,
                &weight_log_var,
                &bias_mean,
                &bias_log_var,
                &noise_log_var,
            )?;
            elbo_history.push(elbo);
            self.update_variational_parameters(
                X,
                y,
                &mut weight_mean,
                &mut weight_log_var,
                &mut bias_mean,
                &mut bias_log_var,
                &mut noise_log_var,
            )?;
            if iteration > 0 && (elbo - elbo_history[iteration - 1]).abs() < self.config.tol {
                break;
            }
        }
        let weight_covariance_flat = weight_log_var
            .mapv(|x| x.exp())
            .into_shape((n_features * n_outputs,))
            .unwrap();
        let weight_covariance = Array2::from_diag(&weight_covariance_flat);
        let bias_covariance = Array2::from_diag(&bias_log_var.mapv(|x| x.exp()));
        let noise_covariance = Array2::from_diag(&noise_log_var.mapv(|x| x.exp()));
        let weight_posterior = PosteriorDistribution {
            mean: weight_mean.clone(),
            covariance: weight_covariance,
            samples: None,
        };
        let bias_posterior = PosteriorDistribution {
            mean: bias_mean.into_shape((n_outputs, 1)).unwrap(),
            covariance: bias_covariance,
            samples: None,
        };
        let noise_posterior = PosteriorDistribution {
            mean: noise_log_var
                .mapv(|x| x.exp())
                .into_shape((n_outputs, 1))
                .unwrap(),
            covariance: noise_covariance,
            samples: None,
        };
        let log_marginal_likelihood = elbo_history.last().cloned().unwrap_or(0.0);
        Ok((
            weight_posterior,
            bias_posterior,
            noise_posterior,
            log_marginal_likelihood,
            elbo_history,
        ))
    }
    /// MCMC inference using Hamiltonian Monte Carlo
    pub(crate) fn mcmc_inference(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(
        PosteriorDistribution,
        PosteriorDistribution,
        PosteriorDistribution,
        Float,
        Vec<Float>,
    )> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();
        let mut weight_samples =
            Array3::<Float>::zeros((self.config.n_samples, n_features, n_outputs));
        let mut bias_samples = Array2::<Float>::zeros((self.config.n_samples, n_outputs));
        let mut noise_samples = Array2::<Float>::zeros((self.config.n_samples, n_outputs));
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        let mut current_weights = Array2::<Float>::zeros((n_features, n_outputs));
        for i in 0..n_features {
            for j in 0..n_outputs {
                current_weights[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut current_bias = Array1::<Float>::zeros(n_outputs);
        let mut current_noise = Array1::ones(n_outputs);
        let mut acceptance_count = 0;
        let mut log_likelihood_history = Vec::new();
        for sample_idx in 0..self.config.n_samples + self.config.burn_in {
            let (new_weights, new_bias, new_noise, accepted) =
                self.hmc_step(X, y, &current_weights, &current_bias, &current_noise, rng)?;
            if accepted {
                current_weights = new_weights;
                current_bias = new_bias;
                current_noise = new_noise;
                acceptance_count += 1;
            }
            if sample_idx >= self.config.burn_in
                && (sample_idx - self.config.burn_in) % self.config.thin == 0
            {
                let store_idx = (sample_idx - self.config.burn_in) / self.config.thin;
                if store_idx < self.config.n_samples {
                    weight_samples
                        .slice_mut(s![store_idx, .., ..])
                        .assign(&current_weights);
                    bias_samples
                        .slice_mut(s![store_idx, ..])
                        .assign(&current_bias);
                    noise_samples
                        .slice_mut(s![store_idx, ..])
                        .assign(&current_noise);
                }
            }
            let log_likelihood =
                self.compute_log_likelihood(X, y, &current_weights, &current_bias, &current_noise)?;
            log_likelihood_history.push(log_likelihood);
        }
        let weight_mean = weight_samples.mean_axis(Axis(0)).unwrap();
        let bias_mean = bias_samples.mean_axis(Axis(0)).unwrap();
        let noise_mean = noise_samples.mean_axis(Axis(0)).unwrap();
        let weight_covariance = Array2::<Float>::eye(n_features * n_outputs);
        let bias_covariance = Array2::<Float>::eye(n_outputs);
        let noise_covariance = Array2::<Float>::eye(n_outputs);
        let weight_posterior = PosteriorDistribution {
            mean: weight_mean,
            covariance: weight_covariance,
            samples: Some(weight_samples),
        };
        let bias_posterior = PosteriorDistribution {
            mean: bias_mean.into_shape((n_outputs, 1)).unwrap(),
            covariance: bias_covariance,
            samples: Some(
                bias_samples
                    .into_shape((self.config.n_samples, n_outputs, 1))
                    .unwrap(),
            ),
        };
        let noise_posterior = PosteriorDistribution {
            mean: noise_mean.into_shape((n_outputs, 1)).unwrap(),
            covariance: noise_covariance,
            samples: Some(
                noise_samples
                    .into_shape((self.config.n_samples, n_outputs, 1))
                    .unwrap(),
            ),
        };
        let log_marginal_likelihood =
            log_likelihood_history.iter().sum::<Float>() / log_likelihood_history.len() as Float;
        Ok((
            weight_posterior,
            bias_posterior,
            noise_posterior,
            log_marginal_likelihood,
            log_likelihood_history,
        ))
    }
    /// Expectation-Maximization inference
    pub(crate) fn em_inference(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(
        PosteriorDistribution,
        PosteriorDistribution,
        PosteriorDistribution,
        Float,
        Vec<Float>,
    )> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        let mut weights = Array2::<Float>::zeros((n_features, n_outputs));
        for i in 0..n_features {
            for j in 0..n_outputs {
                weights[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut bias = Array1::<Float>::zeros(n_outputs);
        let mut noise_variance = Array1::ones(n_outputs);
        let mut likelihood_history = Vec::new();
        for iteration in 0..self.config.max_iter {
            let predictions = X.dot(&weights) + &bias;
            let residuals = y - &predictions;
            let xTx = X.t().dot(X);
            let xTy = X.t().dot(y);
            let regularization: Array2<Float> = Array2::eye(n_features) * 0.01;
            let weights_new = (xTx + regularization).inv().unwrap().dot(&xTy);
            weights = weights_new;
            bias = y.mean_axis(Axis(0)).unwrap() - X.mean_axis(Axis(0)).unwrap().dot(&weights);
            for output_idx in 0..n_outputs {
                let residual_col = residuals.column(output_idx);
                noise_variance[output_idx] = residual_col.mapv(|x| x * x).mean().unwrap();
            }
            let log_likelihood =
                self.compute_log_likelihood(X, y, &weights, &bias, &noise_variance)?;
            likelihood_history.push(log_likelihood);
            if iteration > 0
                && (log_likelihood - likelihood_history[iteration - 1]).abs() < self.config.tol
            {
                break;
            }
        }
        let weight_covariance = Array2::<Float>::eye(n_features * n_outputs) * 0.01;
        let bias_covariance = Array2::<Float>::eye(n_outputs) * 0.01;
        let noise_covariance = Array2::<Float>::eye(n_outputs) * 0.01;
        let weight_posterior = PosteriorDistribution {
            mean: weights,
            covariance: weight_covariance,
            samples: None,
        };
        let bias_posterior = PosteriorDistribution {
            mean: bias.into_shape((n_outputs, 1)).unwrap(),
            covariance: bias_covariance,
            samples: None,
        };
        let noise_posterior = PosteriorDistribution {
            mean: noise_variance.into_shape((n_outputs, 1)).unwrap(),
            covariance: noise_covariance,
            samples: None,
        };
        let log_marginal_likelihood = likelihood_history.last().cloned().unwrap_or(0.0);
        Ok((
            weight_posterior,
            bias_posterior,
            noise_posterior,
            log_marginal_likelihood,
            likelihood_history,
        ))
    }
    /// Laplace approximation inference
    pub(crate) fn laplace_inference(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(
        PosteriorDistribution,
        PosteriorDistribution,
        PosteriorDistribution,
        Float,
        Vec<Float>,
    )> {
        let (map_weights, map_bias, map_noise) = self.find_map_estimate(X, y, rng)?;
        let hessian = self.compute_hessian(X, y, &map_weights, &map_bias, &map_noise)?;
        let posterior_covariance = hessian.inv().unwrap();
        let (n_features, n_outputs) = map_weights.dim();
        let weight_posterior = PosteriorDistribution {
            mean: map_weights.clone(),
            covariance: posterior_covariance.clone(),
            samples: None,
        };
        let bias_posterior = PosteriorDistribution {
            mean: map_bias.clone().into_shape((n_outputs, 1)).unwrap(),
            covariance: Array2::<Float>::eye(n_outputs) * 0.01,
            samples: None,
        };
        let noise_posterior = PosteriorDistribution {
            mean: map_noise.clone().into_shape((n_outputs, 1)).unwrap(),
            covariance: Array2::<Float>::eye(n_outputs) * 0.01,
            samples: None,
        };
        let log_marginal_likelihood =
            self.compute_log_likelihood(X, y, &map_weights, &map_bias, &map_noise)?;
        Ok((
            weight_posterior,
            bias_posterior,
            noise_posterior,
            log_marginal_likelihood,
            vec![log_marginal_likelihood],
        ))
    }
    /// Exact inference (for conjugate priors)
    pub(crate) fn exact_inference(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        _rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(
        PosteriorDistribution,
        PosteriorDistribution,
        PosteriorDistribution,
        Float,
        Vec<Float>,
    )> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();
        let prior_precision: Float = 1.0;
        let prior_mean: Array2<Float> = Array2::<Float>::zeros((n_features, n_outputs));
        let posterior_precision =
            X.t().to_owned().dot(X) + Array2::<Float>::eye(n_features) * prior_precision;
        let posterior_mean = posterior_precision
            .inv()
            .unwrap()
            .dot(&(X.t().dot(y) + &prior_mean * prior_precision));
        let posterior_covariance = posterior_precision.inv().unwrap();
        let weight_posterior = PosteriorDistribution {
            mean: posterior_mean,
            covariance: posterior_covariance,
            samples: None,
        };
        let bias_posterior = PosteriorDistribution {
            mean: Array2::<Float>::zeros((n_outputs, 1)),
            covariance: Array2::<Float>::eye(n_outputs),
            samples: None,
        };
        let noise_posterior = PosteriorDistribution {
            mean: Array2::<Float>::ones((n_outputs, 1)),
            covariance: Array2::<Float>::eye(n_outputs),
            samples: None,
        };
        let log_marginal_likelihood = self.compute_log_marginal_likelihood(X, y)?;
        Ok((
            weight_posterior,
            bias_posterior,
            noise_posterior,
            log_marginal_likelihood,
            vec![log_marginal_likelihood],
        ))
    }
    /// Compute Evidence Lower Bound (ELBO) for variational inference
    fn compute_elbo(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        weight_mean: &Array2<Float>,
        weight_log_var: &Array2<Float>,
        bias_mean: &Array1<Float>,
        bias_log_var: &Array1<Float>,
        noise_log_var: &Array1<Float>,
    ) -> SklResult<Float> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();
        let predictions = X.dot(weight_mean) + bias_mean;
        let residuals = y - &predictions;
        let noise_var = noise_log_var.mapv(|x| x.exp());
        let likelihood_term = -0.5
            * residuals
                .iter()
                .zip(noise_var.iter())
                .map(|(r, nv)| (r * r) / nv + nv.ln())
                .sum::<Float>();
        let weight_kl = self.compute_kl_divergence_normal(weight_mean, weight_log_var)?;
        let bias_kl = self.compute_kl_divergence_normal(
            &bias_mean.clone().into_shape((n_outputs, 1)).unwrap(),
            &bias_log_var.clone().into_shape((n_outputs, 1)).unwrap(),
        )?;
        let noise_kl = self.compute_kl_divergence_gamma(noise_log_var)?;
        let elbo = likelihood_term - weight_kl - bias_kl - noise_kl;
        Ok(elbo)
    }
    /// Update variational parameters
    fn update_variational_parameters(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        weight_mean: &mut Array2<Float>,
        weight_log_var: &mut Array2<Float>,
        bias_mean: &mut Array1<Float>,
        bias_log_var: &mut Array1<Float>,
        noise_log_var: &mut Array1<Float>,
    ) -> SklResult<()> {
        let learning_rate = 0.01;
        let predictions = X.dot(weight_mean) + &*bias_mean;
        let residuals = y - &predictions;
        let weight_gradient = X.t().dot(&residuals);
        *weight_mean = weight_mean.clone() + learning_rate * weight_gradient;
        let bias_gradient = residuals.mean_axis(Axis(0)).unwrap();
        *bias_mean = bias_mean.clone() + learning_rate * bias_gradient;
        for i in 0..noise_log_var.len() {
            let residual_variance = residuals.column(i).mapv(|x| x * x).mean().unwrap();
            noise_log_var[i] = residual_variance.ln();
        }
        Ok(())
    }
    /// Compute KL divergence for Normal distribution
    fn compute_kl_divergence_normal(
        &self,
        mean: &Array2<Float>,
        log_var: &Array2<Float>,
    ) -> SklResult<Float> {
        let kl = 0.5 * (log_var.mapv(|x| x.exp()) + mean.mapv(|x| x * x) - log_var - 1.0).sum();
        Ok(kl)
    }
    /// Compute KL divergence for Gamma distribution
    fn compute_kl_divergence_gamma(&self, log_var: &Array1<Float>) -> SklResult<Float> {
        let kl = log_var.mapv(|x| x.exp() - x - 1.0).sum();
        Ok(kl)
    }
    /// Hamiltonian Monte Carlo step
    fn hmc_step(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        current_weights: &Array2<Float>,
        current_bias: &Array1<Float>,
        current_noise: &Array1<Float>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>, bool)> {
        let step_size = 0.01;
        let n_leapfrog_steps = 10;
        let normal_dist = RandNormal::new(0.0, 1.0).unwrap();
        let mut momentum_weights = Array2::<Float>::zeros(current_weights.raw_dim());
        for i in 0..current_weights.nrows() {
            for j in 0..current_weights.ncols() {
                momentum_weights[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut momentum_bias = Array1::<Float>::zeros(current_bias.len());
        for i in 0..current_bias.len() {
            momentum_bias[i] = rng.sample(normal_dist);
        }
        let mut momentum_noise = Array1::<Float>::zeros(current_noise.len());
        for i in 0..current_noise.len() {
            momentum_noise[i] = rng.sample(normal_dist);
        }
        let mut new_weights = current_weights.clone();
        let mut new_bias = current_bias.clone();
        let mut new_noise = current_noise.clone();
        let mut new_momentum_weights = momentum_weights.clone();
        let mut new_momentum_bias = momentum_bias.clone();
        let mut new_momentum_noise = momentum_noise.clone();
        for _ in 0..n_leapfrog_steps {
            let (weight_grad, bias_grad, noise_grad) =
                self.compute_gradients(X, y, &new_weights, &new_bias, &new_noise)?;
            new_momentum_weights = new_momentum_weights + step_size * 0.5 * weight_grad;
            new_momentum_bias = new_momentum_bias + step_size * 0.5 * bias_grad;
            new_momentum_noise = new_momentum_noise + step_size * 0.5 * noise_grad;
            new_weights = new_weights + step_size * &new_momentum_weights;
            new_bias = new_bias + step_size * &new_momentum_bias;
            new_noise = new_noise + step_size * &new_momentum_noise;
            let (weight_grad, bias_grad, noise_grad) =
                self.compute_gradients(X, y, &new_weights, &new_bias, &new_noise)?;
            new_momentum_weights = new_momentum_weights + step_size * 0.5 * weight_grad;
            new_momentum_bias = new_momentum_bias + step_size * 0.5 * bias_grad;
            new_momentum_noise = new_momentum_noise + step_size * 0.5 * noise_grad;
        }
        let current_energy = self.compute_energy(
            X,
            y,
            current_weights,
            current_bias,
            current_noise,
            &momentum_weights,
            &momentum_bias,
            &momentum_noise,
        )?;
        let new_energy = self.compute_energy(
            X,
            y,
            &new_weights,
            &new_bias,
            &new_noise,
            &new_momentum_weights,
            &new_momentum_bias,
            &new_momentum_noise,
        )?;
        let acceptance_prob = (-new_energy + current_energy).exp().min(1.0);
        let accepted = rng.gen::<Float>() < acceptance_prob;
        if accepted {
            Ok((new_weights, new_bias, new_noise, true))
        } else {
            Ok((
                current_weights.clone(),
                current_bias.clone(),
                current_noise.clone(),
                false,
            ))
        }
    }
    /// Compute energy for HMC
    fn compute_energy(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
        noise: &Array1<Float>,
        momentum_weights: &Array2<Float>,
        momentum_bias: &Array1<Float>,
        momentum_noise: &Array1<Float>,
    ) -> SklResult<Float> {
        let log_likelihood = self.compute_log_likelihood(X, y, weights, bias, noise)?;
        let log_prior = self.compute_log_prior(weights, bias, noise)?;
        let kinetic_energy = 0.5
            * (momentum_weights.mapv(|x| x * x).sum()
                + momentum_bias.mapv(|x| x * x).sum()
                + momentum_noise.mapv(|x| x * x).sum());
        Ok(-log_likelihood - log_prior + kinetic_energy)
    }
    /// Compute gradients for HMC
    fn compute_gradients(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
        noise: &Array1<Float>,
    ) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let predictions = X.dot(weights) + bias;
        let residuals = y - &predictions;
        let weight_grad = X.t().dot(&residuals);
        let bias_grad = residuals.mean_axis(Axis(0)).unwrap();
        let noise_grad = Array1::<Float>::zeros(noise.len());
        let weight_prior_grad = -weights.clone();
        let bias_prior_grad = -bias.clone();
        let noise_prior_grad = Array1::<Float>::zeros(noise.len());
        Ok((
            weight_grad + weight_prior_grad,
            bias_grad + bias_prior_grad,
            noise_grad + noise_prior_grad,
        ))
    }
    /// Compute log likelihood
    fn compute_log_likelihood(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
        noise: &Array1<Float>,
    ) -> SklResult<Float> {
        let predictions = X.dot(weights) + bias;
        let residuals = y - &predictions;
        let mut log_likelihood = 0.0;
        for (output_idx, &noise_var) in noise.iter().enumerate() {
            let residual_col = residuals.column(output_idx);
            let ll = -0.5 * residual_col.mapv(|x| x * x).sum() / noise_var
                - 0.5 * residual_col.len() as Float * noise_var.ln();
            log_likelihood += ll;
        }
        Ok(log_likelihood)
    }
    /// Compute log prior
    fn compute_log_prior(
        &self,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
        noise: &Array1<Float>,
    ) -> SklResult<Float> {
        let weight_log_prior = match &self.config.weight_prior {
            PriorDistribution::Normal(mean, var) => {
                -0.5 * weights.mapv(|x| (x - mean) * (x - mean) / var).sum()
            }
            _ => 0.0,
        };
        let bias_log_prior = match &self.config.bias_prior {
            PriorDistribution::Normal(mean, var) => {
                -0.5 * bias.mapv(|x| (x - mean) * (x - mean) / var).sum()
            }
            _ => 0.0,
        };
        let noise_log_prior = match &self.config.noise_prior {
            PriorDistribution::Gamma(shape, rate) => {
                noise.mapv(|x| (shape - 1.0) * x.ln() - rate * x).sum()
            }
            _ => 0.0,
        };
        Ok(weight_log_prior + bias_log_prior + noise_log_prior)
    }
    /// Find MAP estimate
    fn find_map_estimate(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        let mut weights = Array2::<Float>::zeros((n_features, n_outputs));
        for i in 0..n_features {
            for j in 0..n_outputs {
                weights[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut bias = Array1::<Float>::zeros(n_outputs);
        let mut noise = Array1::ones(n_outputs);
        let learning_rate = 0.01;
        for _iteration in 0..self.config.max_iter {
            let (weight_grad, bias_grad, noise_grad) =
                self.compute_gradients(X, y, &weights, &bias, &noise)?;
            weights = weights + learning_rate * weight_grad;
            bias = bias + learning_rate * bias_grad;
            noise = noise + learning_rate * noise_grad;
            noise = noise.mapv(|x| x.max(1e-6));
        }
        Ok((weights, bias, noise))
    }
    /// Compute Hessian matrix
    fn compute_hessian(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
        noise: &Array1<Float>,
    ) -> SklResult<Array2<Float>> {
        let (n_features, n_outputs) = weights.dim();
        let total_params = n_features * n_outputs + n_outputs + n_outputs;
        let mut hessian = Array2::<Float>::eye(total_params);
        let xTx = X.t().dot(X);
        for i in 0..n_features {
            for j in 0..n_features {
                hessian[(i, j)] = xTx[(i, j)];
            }
        }
        for i in 0..total_params {
            hessian[(i, i)] += 1.0;
        }
        Ok(hessian)
    }
    /// Compute log marginal likelihood
    fn compute_log_marginal_likelihood(
        &self,
        X: &ArrayView2<'_, Float>,
        y: &ArrayView2<'_, Float>,
    ) -> SklResult<Float> {
        let (n_samples, n_features) = X.dim();
        let n_outputs = y.ncols();
        let prior_precision: Float = 1.0;
        let posterior_precision = X.t().dot(X) + Array2::<Float>::eye(n_features) * prior_precision;
        let log_det_prior = n_features as Float * prior_precision.ln();
        let log_det_posterior = posterior_precision.det().unwrap().ln();
        let log_marginal_likelihood = 0.5 * (log_det_prior - log_det_posterior);
        Ok(log_marginal_likelihood)
    }
}
impl BayesianMultiOutputModel<BayesianMultiOutputModelTrained> {
    /// Predict with uncertainty quantification
    pub fn predict_with_uncertainty(
        &self,
        X: &ArrayView2<'_, Float>,
    ) -> SklResult<PredictionWithUncertainty> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features, n_features
            )));
        }
        let mean = self.predict(X)?;
        let mut variance = Array2::<Float>::zeros((n_samples, self.state.n_outputs));
        for i in 0..n_samples {
            let x_i = X.slice(s![i, ..]);
            let weight_diag = self.state.weight_posterior.covariance.diag().to_owned();
            let weight_diag_reshaped = weight_diag
                .into_shape((self.state.n_features, self.state.n_outputs))
                .unwrap();
            for j in 0..self.state.n_outputs {
                let weight_var_j = weight_diag_reshaped.column(j);
                let pred_var = x_i.dot(&weight_var_j)
                    + self.state.noise_posterior.mean[[j, 0]]
                        * self.state.noise_posterior.mean[[j, 0]];
                variance[[i, j]] = pred_var;
            }
        }
        let z_score = 1.96;
        let mut confidence_intervals = Array3::<Float>::zeros((n_samples, self.state.n_outputs, 2));
        for i in 0..n_samples {
            for j in 0..self.state.n_outputs {
                let std_dev = variance[(i, j)].sqrt();
                confidence_intervals[(i, j, 0)] = mean[(i, j)] - z_score * std_dev;
                confidence_intervals[(i, j, 1)] = mean[(i, j)] + z_score * std_dev;
            }
        }
        let n_pred_samples = 100;
        let mut samples = Array3::<Float>::zeros((n_samples, self.state.n_outputs, n_pred_samples));
        if let Some(ref weight_samples) = self.state.weight_posterior.samples {
            for sample_idx in 0..n_pred_samples.min(weight_samples.shape()[0]) {
                let weight_sample = weight_samples.slice(s![sample_idx, .., ..]);
                let pred_sample = X.dot(&weight_sample) + self.state.bias_posterior.mean.column(0);
                samples
                    .slice_mut(s![.., .., sample_idx])
                    .assign(&pred_sample);
            }
        } else {
            for sample_idx in 0..n_pred_samples {
                let pred_sample = mean.clone();
                samples
                    .slice_mut(s![.., .., sample_idx])
                    .assign(&pred_sample);
            }
        }
        Ok(PredictionWithUncertainty {
            mean,
            variance,
            confidence_intervals,
            samples,
        })
    }
    /// Get the posterior distribution over weights
    pub fn weight_posterior(&self) -> &PosteriorDistribution {
        &self.state.weight_posterior
    }
    /// Get the posterior distribution over bias
    pub fn bias_posterior(&self) -> &PosteriorDistribution {
        &self.state.bias_posterior
    }
    /// Get the posterior distribution over noise
    pub fn noise_posterior(&self) -> &PosteriorDistribution {
        &self.state.noise_posterior
    }
    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> Float {
        self.state.log_marginal_likelihood
    }
    /// Get the ELBO history (for variational inference)
    pub fn elbo_history(&self) -> &[Float] {
        &self.state.elbo_history
    }
}
/// Prediction with uncertainty
#[derive(Debug, Clone)]
pub struct PredictionWithUncertainty {
    /// Predicted mean
    pub mean: Array2<Float>,
    /// Predicted variance
    pub variance: Array2<Float>,
    /// Confidence intervals
    pub confidence_intervals: Array3<Float>,
    /// Predictive samples
    pub samples: Array3<Float>,
}
/// Posterior distribution representation
#[derive(Debug, Clone)]
pub struct PosteriorDistribution {
    /// Posterior mean
    pub mean: Array2<Float>,
    /// Posterior covariance
    pub covariance: Array2<Float>,
    /// Posterior samples (if MCMC)
    pub samples: Option<Array3<Float>>,
}
/// Trained state for Bayesian Multi-Output Model
#[derive(Debug, Clone)]
pub struct BayesianMultiOutputModelTrained {
    /// Posterior distribution over weights
    pub weight_posterior: PosteriorDistribution,
    /// Posterior distribution over bias
    pub bias_posterior: PosteriorDistribution,
    /// Posterior distribution over noise variance
    pub noise_posterior: PosteriorDistribution,
    /// Number of features
    pub n_features: usize,
    /// Number of outputs
    pub n_outputs: usize,
    /// Log marginal likelihood
    pub log_marginal_likelihood: Float,
    /// Evidence lower bound (ELBO) for variational inference
    pub elbo_history: Vec<Float>,
    /// Configuration used for training
    pub config: BayesianMultiOutputConfig,
}
#[derive(Debug, Clone)]
pub struct GaussianProcessMultiOutputTrained {
    X_train: Array2<Float>,
    y_train: Array2<Float>,
    K_inv: Array2<Float>,
    y_mean: Array1<Float>,
    y_std: Array1<Float>,
    kernel: KernelFunction,
    noise_level: Float,
    normalize_y: bool,
    n_features: usize,
    n_outputs: usize,
}
/// Ensemble Bayesian Multi-Output Model
///
/// Combines multiple Bayesian models using various ensemble strategies including
/// Bayesian Model Averaging, Product of Experts, and Committee Machines.
///
/// # Examples
///
/// ```rust
/// use sklears_multioutput::probabilistic::{EnsembleBayesianModel, EnsembleBayesianConfig, EnsembleStrategy};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Predict};
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
///
/// let config = EnsembleBayesianConfig {
///     n_models: 3,
///     strategy: EnsembleStrategy::BayesianAveraging,
///     random_state: Some(42),
///     ..Default::default()
/// };
///
/// let model = EnsembleBayesianModel::new().config(config);
/// let trained = model.fit(&X.view(), &y.view()).unwrap();
/// let predictions = trained.predict(&X.view()).unwrap();
/// assert_eq!(predictions.dim(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct EnsembleBayesianModel<S = Untrained> {
    pub(crate) state: S,
    pub(crate) config: EnsembleBayesianConfig,
}
impl EnsembleBayesianModel<Untrained> {
    /// Create a new Ensemble Bayesian Model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: EnsembleBayesianConfig::default(),
        }
    }
    /// Set the configuration
    pub fn config(mut self, config: EnsembleBayesianConfig) -> Self {
        self.config = config;
        self
    }
    /// Set the number of models
    pub fn n_models(mut self, n_models: usize) -> Self {
        self.config.n_models = n_models;
        self
    }
    /// Set the ensemble strategy
    pub fn strategy(mut self, strategy: EnsembleStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }
    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}
impl EnsembleBayesianModel<Untrained> {
    /// Compute model weights based on ensemble strategy
    pub(crate) fn compute_model_weights(
        &self,
        log_likelihoods: &[Float],
    ) -> SklResult<Array1<Float>> {
        let n_models = log_likelihoods.len();
        match self.config.strategy {
            EnsembleStrategy::BayesianAveraging => {
                let max_log_lik = log_likelihoods
                    .iter()
                    .cloned()
                    .fold(Float::NEG_INFINITY, Float::max);
                let exp_likelihoods: Vec<Float> = log_likelihoods
                    .iter()
                    .map(|&log_lik| (log_lik - max_log_lik).exp())
                    .collect();
                let sum: Float = exp_likelihoods.iter().sum();
                let weights: Vec<Float> = exp_likelihoods
                    .iter()
                    .map(|&exp_lik| exp_lik / sum)
                    .collect();
                Ok(Array1::from_vec(weights))
            }
            EnsembleStrategy::EqualWeight => {
                let weight = 1.0 / n_models as Float;
                Ok(Array1::from_elem(n_models, weight))
            }
            EnsembleStrategy::ProductOfExperts
            | EnsembleStrategy::CommitteeMachine
            | EnsembleStrategy::MixtureOfExperts => {
                let weight = 1.0 / n_models as Float;
                Ok(Array1::from_elem(n_models, weight))
            }
        }
    }
    /// Compute ensemble log marginal likelihood
    pub(crate) fn compute_ensemble_log_likelihood(
        &self,
        log_likelihoods: &[Float],
        weights: &Array1<Float>,
    ) -> Float {
        log_likelihoods
            .iter()
            .zip(weights.iter())
            .map(|(&log_lik, &weight)| weight * log_lik)
            .sum()
    }
}
impl EnsembleBayesianModel<EnsembleBayesianModelTrained> {
    /// Predict with uncertainty quantification
    ///
    /// Returns mean prediction, variance, and confidence intervals
    pub fn predict_with_uncertainty(
        &self,
        X: &ArrayView2<Float>,
        confidence_level: Float,
    ) -> SklResult<PredictionWithUncertainty> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                X.ncols()
            )));
        }
        let n_samples = X.nrows();
        let n_outputs = self.state.n_outputs;
        let mut all_predictions = Vec::new();
        for model in &self.state.models {
            let model_wrapper = BayesianMultiOutputModel {
                state: model.clone(),
                config: self.state.config.base_config.clone(),
            };
            all_predictions.push(model_wrapper.predict(X)?);
        }
        let mean = self.predict(X)?;
        let mut variance = Array2::zeros((n_samples, n_outputs));
        for i in 0..n_samples {
            for j in 0..n_outputs {
                let values: Vec<Float> = all_predictions.iter().map(|pred| pred[[i, j]]).collect();
                let mean_val = mean[[i, j]];
                let var: Float = values
                    .iter()
                    .map(|&v| (v - mean_val).powi(2))
                    .sum::<Float>()
                    / values.len() as Float;
                variance[[i, j]] = var;
            }
        }
        let z_score = if (confidence_level - 0.95).abs() < 1e-6 {
            1.96
        } else if (confidence_level - 0.99).abs() < 1e-6 {
            2.576
        } else {
            1.96
        };
        let mut confidence_intervals = Array3::zeros((n_samples, n_outputs, 2));
        for i in 0..n_samples {
            for j in 0..n_outputs {
                let std_dev = variance[[i, j]].sqrt();
                confidence_intervals[[i, j, 0]] = mean[[i, j]] - z_score * std_dev;
                confidence_intervals[[i, j, 1]] = mean[[i, j]] + z_score * std_dev;
            }
        }
        let n_ensemble_samples = 100;
        let mut samples = Array3::zeros((n_ensemble_samples, n_samples, n_outputs));
        let mut rng = scirs2_core::random::thread_rng();
        for s in 0..n_ensemble_samples {
            let model_idx = self.sample_model_index(&mut rng);
            samples
                .slice_mut(s![s, .., ..])
                .assign(&all_predictions[model_idx]);
        }
        Ok(PredictionWithUncertainty {
            mean,
            variance,
            confidence_intervals,
            samples,
        })
    }
    /// Sample a model index based on model weights
    fn sample_model_index(&self, rng: &mut impl Rng) -> usize {
        let r: Float = rng.gen();
        let mut cumsum = 0.0;
        for (i, &weight) in self.state.model_weights.iter().enumerate() {
            cumsum += weight;
            if r <= cumsum {
                return i;
            }
        }
        self.state.models.len() - 1
    }
    /// Get ensemble model weights
    pub fn model_weights(&self) -> &Array1<Float> {
        &self.state.model_weights
    }
    /// Get ensemble log marginal likelihood
    pub fn ensemble_log_likelihood(&self) -> Float {
        self.state.ensemble_log_likelihood
    }
    /// Get number of models in ensemble
    pub fn n_models(&self) -> usize {
        self.state.models.len()
    }
}
