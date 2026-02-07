//! Random Fourier Features for approximating RBF kernels
//!
//! This module provides implementations of Random Fourier Features (RFF) for scalable
//! Gaussian Process regression. The features include:
//!
//! - `RandomFourierFeatures`: Basic RFF transformer for approximating RBF kernels
//! - `RandomFourierFeaturesGPR`: Complete GP regressor using RFF with uncertainty quantification
//!
//! These implementations reduce computational complexity from O(n³) to O(nD) where D is
//! the number of random features, enabling scalable Gaussian Process modeling.

// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};

use crate::{classification::GpcConfig, utils};

/// # Examples
///
/// ```ignore
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let mut rff = RandomFourierFeatures::new(100, 1.0, Some(42));
/// rff.fit(&X.view()).unwrap();
/// let features = rff.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RandomFourierFeatures {
    n_components: usize,
    gamma: f64,
    random_state: Option<u64>,
    omega: Option<Array2<f64>>, // Random frequencies
    phi: Option<Array1<f64>>,   // Random phases
    fitted: bool,
}

impl RandomFourierFeatures {
    /// Create a new RandomFourierFeatures instance
    ///
    /// # Parameters
    /// * `n_components` - Number of random features to generate
    /// * `gamma` - Parameter of the RBF kernel (gamma = 1 / (2 * sigma^2))
    /// * `random_state` - Random seed for reproducibility
    pub fn new(n_components: usize, gamma: f64, random_state: Option<u64>) -> Self {
        Self {
            n_components,
            gamma,
            random_state,
            omega: None,
            phi: None,
            fitted: false,
        }
    }

    /// Fit the random Fourier features to the training data
    pub fn fit(&mut self, X: &ArrayView2<f64>) -> SklResult<()> {
        let (_, n_features) = X.dim();

        // Use provided random state or create a simple pseudo-random generator
        let mut rng_state = self.random_state.unwrap_or(42);

        // Generate random frequencies from normal distribution
        // For RBF kernel with parameter gamma, frequencies are drawn from N(0, 2*gamma)
        let mut omega = Array2::<f64>::zeros((self.n_components, n_features));
        let std_dev = (2.0 * self.gamma).sqrt();

        for i in 0..self.n_components {
            for j in 0..n_features {
                // Simple Box-Muller transform for normal distribution
                let (u1, u2) = self.uniform_pair(&mut rng_state);
                let normal_sample =
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                omega[[i, j]] = std_dev * normal_sample;
            }
        }

        // Generate random phases from uniform distribution [0, 2π]
        let mut phi = Array1::<f64>::zeros(self.n_components);
        for i in 0..self.n_components {
            let u = self.uniform(&mut rng_state);
            phi[i] = 2.0 * std::f64::consts::PI * u;
        }

        self.omega = Some(omega);
        self.phi = Some(phi);
        self.fitted = true;

        Ok(())
    }

    /// Transform the input data to the random feature space
    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "RandomFourierFeatures must be fitted before transform".to_string(),
            ));
        }

        let omega = self.omega.as_ref().unwrap();
        let phi = self.phi.as_ref().unwrap();

        let (n_samples, _) = X.dim();
        let mut features = Array2::<f64>::zeros((n_samples, self.n_components));

        // Compute features: sqrt(2/D) * cos(X * omega^T + phi)
        let normalization = (2.0 / self.n_components as f64).sqrt();

        for i in 0..n_samples {
            for j in 0..self.n_components {
                let dot_product: f64 = X
                    .row(i)
                    .iter()
                    .zip(omega.row(j).iter())
                    .map(|(x, w)| x * w)
                    .sum();
                features[[i, j]] = normalization * (dot_product + phi[j]).cos();
            }
        }

        Ok(features)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        self.fit(X)?;
        self.transform(X)
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the gamma parameter
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Check if the transformer is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Simple linear congruential generator for uniform random numbers
    fn uniform(&self, rng_state: &mut u64) -> f64 {
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (*rng_state & 0x7fffffff) as f64 / 0x7fffffff as f64
    }

    /// Generate a pair of uniform random numbers
    fn uniform_pair(&self, rng_state: &mut u64) -> (f64, f64) {
        let u1 = self.uniform(rng_state);
        let u2 = self.uniform(rng_state);
        (u1, u2)
    }
}

impl Default for RandomFourierFeatures {
    fn default() -> Self {
        Self::new(100, 1.0, Some(42))
    }
}

/// Random Fourier Features Gaussian Process Regressor
///
/// This implementation combines Random Fourier Features with linear regression
/// to approximate Gaussian Process regression at O(nD) computational complexity.
/// The approach maintains uncertainty estimates through Bayesian linear regression
/// in the feature space.
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::{RandomFourierFeaturesGPR, kernels::RBF};
/// use sklears_core::traits::{Fit, Predict};
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
/// let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
///
/// let rff_gpr = RandomFourierFeaturesGPR::new()
///     .n_components(50)
///     .gamma(1.0)
///     .alpha(1e-3);
/// let fitted = rff_gpr.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RandomFourierFeaturesGPR<S = Untrained> {
    state: S,
    n_components: usize,
    gamma: f64,
    alpha: f64,
    random_state: Option<u64>,
    config: GpcConfig,
}

/// Trained state for Random Fourier Features Gaussian Process Regressor
#[derive(Debug, Clone)]
pub struct RffGprTrained {
    /// rff
    pub rff: RandomFourierFeatures, // Fitted RFF transformer
    /// weights
    pub weights: Array1<f64>, // Linear regression weights
    /// weights_cov
    pub weights_cov: Array2<f64>, // Posterior covariance of weights
    /// alpha
    pub alpha: f64, // Noise precision
    /// y_mean
    pub y_mean: f64, // Training target mean
    /// log_marginal_likelihood_value
    pub log_marginal_likelihood_value: f64, // Log marginal likelihood
}

impl RandomFourierFeaturesGPR<Untrained> {
    /// Create a new RandomFourierFeaturesGPR instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 100,
            gamma: 1.0,
            alpha: 1e-3,
            random_state: Some(42),
            config: GpcConfig::default(),
        }
    }

    /// Set the number of random features
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the gamma parameter of the RBF kernel
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the noise precision parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Estimator for RandomFourierFeaturesGPR<Untrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for RandomFourierFeaturesGPR<RffGprTrained> {
    type Config = GpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, f64>> for RandomFourierFeaturesGPR<Untrained> {
    type Fitted = RandomFourierFeaturesGPR<RffGprTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize and fit Random Fourier Features
        let mut rff = RandomFourierFeatures::new(self.n_components, self.gamma, self.random_state);
        let features = rff.fit_transform(X)?;

        // Center the targets
        let y_mean = y.mean().unwrap_or(0.0);
        let y_centered = y.mapv(|yi| yi - y_mean);

        // Bayesian linear regression in feature space
        // Posterior: p(w|X,y) = N(mu_w, Sigma_w)
        // mu_w = Sigma_w * Phi^T * y / sigma_n^2
        // Sigma_w = (Phi^T * Phi / sigma_n^2 + I / sigma_p^2)^{-1}

        let n_features = features.ncols();
        let sigma_n_sq = 1.0 / self.alpha;
        let sigma_p_sq = 1.0; // Prior variance for weights

        // Compute Phi^T * Phi / sigma_n^2 + I / sigma_p^2
        let mut gram_matrix = features.t().dot(&features) / sigma_n_sq;
        for i in 0..n_features {
            gram_matrix[[i, i]] += 1.0 / sigma_p_sq;
        }

        // Compute Cholesky decomposition for efficient solving
        let L = utils::robust_cholesky(&gram_matrix)?;

        // Compute posterior mean: mu_w = Sigma_w * Phi^T * y / sigma_n^2
        let phi_t_y = features.t().dot(&y_centered) / sigma_n_sq;
        let weights = utils::triangular_solve(&L, &phi_t_y)?;

        // Compute posterior covariance: Sigma_w = (L * L^T)^{-1}
        let I = Array2::<f64>::eye(n_features);
        let weights_cov = utils::triangular_solve_matrix(&L, &I)?;

        // Compute log marginal likelihood
        let predictions = features.dot(&weights);
        let residuals = &y_centered - &predictions;
        let data_fit = -0.5 * residuals.mapv(|r| r * r).sum() * self.alpha;
        let complexity_penalty = -0.5 * weights.mapv(|w| w * w).sum() / sigma_p_sq;
        let normalization = -0.5 * y.len() as f64 * (2.0 * std::f64::consts::PI / self.alpha).ln();
        let log_marginal_likelihood_value = data_fit + complexity_penalty + normalization;

        Ok(RandomFourierFeaturesGPR {
            state: RffGprTrained {
                rff,
                weights,
                weights_cov,
                alpha: self.alpha,
                y_mean,
                log_marginal_likelihood_value,
            },
            n_components: self.n_components,
            gamma: self.gamma,
            alpha: self.alpha,
            random_state: self.random_state,
            config: self.config.clone(),
        })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<f64>> for RandomFourierFeaturesGPR<RffGprTrained> {
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let (mean, _) = self.predict_with_std(X)?;
        Ok(mean)
    }
}

impl RandomFourierFeaturesGPR<RffGprTrained> {
    /// Predict with uncertainty estimates
    pub fn predict_with_std(&self, X: &ArrayView2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        // Transform input to feature space
        let features = self.state.rff.transform(X)?;

        // Predictive mean: Phi * mu_w + y_mean
        let mean_centered = features.dot(&self.state.weights);
        let mean = mean_centered.mapv(|m| m + self.state.y_mean);

        // Predictive variance: Phi * Sigma_w * Phi^T + sigma_n^2
        let feature_uncertainty = features.dot(&self.state.weights_cov);
        let mut variance = Array1::<f64>::zeros(X.nrows());

        for i in 0..X.nrows() {
            let phi_i = features.row(i);
            let var_from_weights = phi_i.dot(&feature_uncertainty.row(i));
            let noise_var = 1.0 / self.state.alpha;
            variance[i] = var_from_weights + noise_var;
        }

        let std = variance.mapv(|v| v.sqrt().max(0.0));

        Ok((mean, std))
    }

    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.state.log_marginal_likelihood_value
    }

    /// Get the posterior weights
    pub fn weights(&self) -> &Array1<f64> {
        &self.state.weights
    }

    /// Get the posterior covariance of weights
    pub fn weights_covariance(&self) -> &Array2<f64> {
        &self.state.weights_cov
    }

    /// Get the fitted Random Fourier Features transformer
    pub fn rff_transformer(&self) -> &RandomFourierFeatures {
        &self.state.rff
    }
}

impl Default for RandomFourierFeaturesGPR<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}
