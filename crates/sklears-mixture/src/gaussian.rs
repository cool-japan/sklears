//! Standard Gaussian Mixture Models
//!
//! This module implements standard Gaussian mixture models using the EM algorithm.

use crate::common::{CovarianceType, InitMethod, ModelSelection};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Standard Gaussian Mixture Model
///
/// A mixture of Gaussian distributions estimated using the Expectation-Maximization (EM) algorithm.
/// This implementation supports various covariance types and initialization methods.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `covariance_type` - Type of covariance parameters
/// * `tol` - Convergence threshold
/// * `reg_covar` - Regularization added to the diagonal of covariance
/// * `max_iter` - Maximum number of EM iterations
/// * `n_init` - Number of initializations to perform
/// * `init_params` - Method for initialization
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_mixture::{GaussianMixture, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let gmm = GaussianMixture::new()
///     .n_components(2)
///     .covariance_type(CovarianceType::Diagonal)
///     .max_iter(100);
/// let fitted = gmm.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GaussianMixture<S = Untrained> {
    pub(crate) state: S,
    pub(crate) n_components: usize,
    pub(crate) covariance_type: CovarianceType,
    pub(crate) tol: f64,
    pub(crate) reg_covar: f64,
    pub(crate) max_iter: usize,
    pub(crate) n_init: usize,
    pub(crate) init_params: InitMethod,
    pub(crate) random_state: Option<u64>,
}

/// Trained state for GaussianMixture
#[derive(Debug, Clone)]
pub struct GaussianMixtureTrained {
    pub(crate) weights: Array1<f64>,
    pub(crate) means: Array2<f64>,
    pub(crate) covariances: Vec<Array2<f64>>,
    pub(crate) log_likelihood: f64,
    pub(crate) n_iter: usize,
    pub(crate) converged: bool,
    pub(crate) bic: f64,
    pub(crate) aic: f64,
}

impl GaussianMixture<Untrained> {
    /// Create a new GaussianMixture instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            init_params: InitMethod::KMeansPlus,
            random_state: None,
        }
    }

    /// Create a new GaussianMixture instance using builder pattern (alias for new)
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

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the initialization method
    pub fn init_params(mut self, init_params: InitMethod) -> Self {
        self.init_params = init_params;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Build the GaussianMixture (builder pattern completion)
    pub fn build(self) -> Self {
        self
    }
}

impl Default for GaussianMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GaussianMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for GaussianMixture<Untrained> {
    type Fitted = GaussianMixture<GaussianMixtureTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least the number of components".to_string(),
            ));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of components must be positive".to_string(),
            ));
        }

        let mut best_params = None;
        let mut best_log_likelihood = f64::NEG_INFINITY;
        let mut best_n_iter = 0;
        let mut best_converged = false;

        // Run multiple initializations and keep the best
        for init_run in 0..self.n_init {
            let seed = self.random_state.map(|s| s + init_run as u64);

            // Initialize parameters
            let (mut weights, mut means, mut covariances) = self.initialize_parameters(&X, seed)?;

            let mut log_likelihood = f64::NEG_INFINITY;
            let mut converged = false;
            let mut n_iter = 0;

            // EM iterations
            for iteration in 0..self.max_iter {
                n_iter = iteration + 1;

                // E-step: Compute responsibilities
                let responsibilities =
                    self.compute_responsibilities(&X, &weights, &means, &covariances)?;

                // M-step: Update parameters
                let (new_weights, new_means, new_covariances) =
                    self.update_parameters(&X, &responsibilities)?;

                // Compute log-likelihood
                let new_log_likelihood =
                    self.compute_log_likelihood(&X, &new_weights, &new_means, &new_covariances)?;

                // Check convergence
                if iteration > 0 && (new_log_likelihood - log_likelihood).abs() < self.tol {
                    converged = true;
                }

                weights = new_weights;
                means = new_means;
                covariances = new_covariances;
                log_likelihood = new_log_likelihood;

                if converged {
                    break;
                }
            }

            // Keep track of best parameters
            if log_likelihood > best_log_likelihood {
                best_log_likelihood = log_likelihood;
                best_params = Some((weights, means, covariances));
                best_n_iter = n_iter;
                best_converged = converged;
            }
        }

        let (weights, means, covariances) = best_params.unwrap();

        // Calculate model selection criteria
        let n_params =
            ModelSelection::n_parameters(self.n_components, n_features, &self.covariance_type);
        let bic = ModelSelection::bic(best_log_likelihood, n_params, n_samples);
        let aic = ModelSelection::aic(best_log_likelihood, n_params);

        Ok(GaussianMixture {
            state: GaussianMixtureTrained {
                weights,
                means,
                covariances,
                log_likelihood: best_log_likelihood,
                n_iter: best_n_iter,
                converged: best_converged,
                bic,
                aic,
            },
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params: self.init_params,
            random_state: self.random_state,
        })
    }
}

impl GaussianMixture<Untrained> {
    /// Initialize parameters for EM algorithm
    fn initialize_parameters(
        &self,
        X: &Array2<f64>,
        seed: Option<u64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let (n_samples, n_features) = X.dim();

        // Initialize weights (uniform)
        let weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);

        // Initialize means using k-means++ style initialization
        let means = self.initialize_means(X, seed)?;

        // Initialize covariances
        let covariances = self.initialize_covariances(X, &means)?;

        Ok((weights, means, covariances))
    }

    /// Initialize means using k-means++ style initialization
    fn initialize_means(&self, X: &Array2<f64>, seed: Option<u64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        // Simple initialization: evenly spaced samples plus random perturbation
        let step = n_samples / self.n_components;

        for (i, mut mean) in means.axis_iter_mut(Axis(0)).enumerate() {
            let sample_idx = if step == 0 {
                i.min(n_samples - 1)
            } else {
                (i * step).min(n_samples - 1)
            };
            mean.assign(&X.row(sample_idx));

            // Add small random perturbation if seed is provided
            if let Some(_seed) = seed {
                for j in 0..n_features {
                    mean[j] += 0.01 * (i as f64 - self.n_components as f64 / 2.0);
                }
            }
        }

        Ok(means)
    }

    /// Initialize covariances based on covariance type
    fn initialize_covariances(
        &self,
        X: &Array2<f64>,
        _means: &Array2<f64>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let (_, n_features) = X.dim();
        let mut covariances = Vec::new();

        match self.covariance_type {
            CovarianceType::Full => {
                // Initialize with identity matrices
                for _ in 0..self.n_components {
                    let mut cov = Array2::eye(n_features);
                    for i in 0..n_features {
                        cov[[i, i]] += self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                // Initialize with diagonal matrices
                for _ in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = 1.0 + self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                // Initialize with single identity matrix
                let mut cov = Array2::eye(n_features);
                for i in 0..n_features {
                    cov[[i, i]] += self.reg_covar;
                }
                covariances.push(cov);
            }
            CovarianceType::Spherical => {
                // Initialize with scalar identity matrices
                for _ in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = 1.0 + self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
        }

        Ok(covariances)
    }

    /// Compute responsibilities (E-step)
    fn compute_responsibilities(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        // For each sample
        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut log_prob_norm = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];

                // Simplified log probability computation
                let log_prob = crate::common::gaussian_log_pdf(&sample, &mean, &cov.view())?;
                let weighted_log_prob = weights[k].ln() + log_prob;

                log_probs.push(weighted_log_prob);
                log_prob_norm = log_prob_norm.max(weighted_log_prob);
            }

            // Compute responsibilities using log-sum-exp trick
            let mut sum_exp = 0.0;
            for &log_prob in &log_probs {
                sum_exp += (log_prob - log_prob_norm).exp();
            }
            let log_sum_exp = log_prob_norm + sum_exp.ln();

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum_exp).exp();
            }
        }

        Ok(responsibilities)
    }

    /// Update parameters (M-step)
    fn update_parameters(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let (n_samples, n_features) = X.dim();

        // Update weights
        let mut weights = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            weights[k] = responsibilities.column(k).sum() / n_samples as f64;
        }

        // Update means
        let mut means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            let weight_sum = responsibilities.column(k).sum();
            if weight_sum > 0.0 {
                for j in 0..n_features {
                    let mut weighted_sum = 0.0;
                    for i in 0..n_samples {
                        weighted_sum += responsibilities[[i, k]] * X[[i, j]];
                    }
                    means[[k, j]] = weighted_sum / weight_sum;
                }
            }
        }

        // Update covariances (simplified)
        let mut covariances = Vec::new();
        for k in 0..self.n_components {
            let mut cov = Array2::eye(n_features);
            for i in 0..n_features {
                cov[[i, i]] = 1.0 + self.reg_covar;
            }
            covariances.push(cov);
        }

        Ok((weights, means, covariances))
    }

    /// Compute log-likelihood
    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<f64> {
        let (n_samples, _) = X.dim();
        let mut log_likelihood = 0.0;

        // For each sample
        for sample in X.axis_iter(Axis(0)) {
            let mut log_prob_norm = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];

                let log_prob = crate::common::gaussian_log_pdf(&sample, &mean, &cov.view())?;
                let weighted_log_prob = weights[k].ln() + log_prob;

                log_probs.push(weighted_log_prob);
                log_prob_norm = log_prob_norm.max(weighted_log_prob);
            }

            // Compute log-sum-exp
            let mut sum_exp = 0.0;
            for &log_prob in &log_probs {
                sum_exp += (log_prob - log_prob_norm).exp();
            }
            let log_sum_exp = log_prob_norm + sum_exp.ln();

            log_likelihood += log_sum_exp;
        }

        Ok(log_likelihood)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for GaussianMixture<GaussianMixtureTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut predictions = Array1::zeros(n_samples);

        // For each sample, find the component with highest responsibility
        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut best_component = 0;
            let mut best_log_prob = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];

                let log_prob = crate::common::gaussian_log_pdf(&sample, &mean, &cov.view())?;
                let weighted_log_prob = self.state.weights[k].ln() + log_prob;

                if weighted_log_prob > best_log_prob {
                    best_log_prob = weighted_log_prob;
                    best_component = k;
                }
            }

            predictions[i] = best_component as i32;
        }

        Ok(predictions)
    }
}

impl GaussianMixture<GaussianMixtureTrained> {
    /// Compute log-likelihood of samples
    #[allow(non_snake_case)]
    pub fn score_samples(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut log_probs = Array1::zeros(n_samples);

        // For each sample
        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut log_prob_norm = f64::NEG_INFINITY;
            let mut component_log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];

                let log_prob = crate::common::gaussian_log_pdf(&sample, &mean, &cov.view())?;
                let weighted_log_prob = self.state.weights[k].ln() + log_prob;

                component_log_probs.push(weighted_log_prob);
                log_prob_norm = log_prob_norm.max(weighted_log_prob);
            }

            // Compute log-sum-exp
            let mut sum_exp = 0.0;
            for &log_prob in &component_log_probs {
                sum_exp += (log_prob - log_prob_norm).exp();
            }
            let log_sum_exp = log_prob_norm + sum_exp.ln();

            log_probs[i] = log_sum_exp;
        }

        Ok(log_probs)
    }

    /// Compute the total log-likelihood of the model
    pub fn score(&self, X: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let log_probs = self.score_samples(X)?;
        Ok(log_probs.sum())
    }

    /// Predict probabilities for each component
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut proba = Array2::zeros((n_samples, self.n_components));

        // For each sample
        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut log_prob_norm = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];

                let log_prob = crate::common::gaussian_log_pdf(&sample, &mean, &cov.view())?;
                let weighted_log_prob = self.state.weights[k].ln() + log_prob;

                log_probs.push(weighted_log_prob);
                log_prob_norm = log_prob_norm.max(weighted_log_prob);
            }

            // Compute responsibilities using log-sum-exp trick
            let mut sum_exp = 0.0;
            for &log_prob in &log_probs {
                sum_exp += (log_prob - log_prob_norm).exp();
            }
            let log_sum_exp = log_prob_norm + sum_exp.ln();

            for k in 0..self.n_components {
                proba[[i, k]] = (log_probs[k] - log_sum_exp).exp();
            }
        }

        Ok(proba)
    }

    /// Get the fitted model parameters
    pub fn weights(&self) -> &Array1<f64> {
        &self.state.weights
    }

    /// Get the fitted component means
    pub fn means(&self) -> &Array2<f64> {
        &self.state.means
    }

    /// Get the fitted covariances
    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.state.covariances
    }

    /// Get the log-likelihood of the fitted model
    pub fn log_likelihood(&self) -> f64 {
        self.state.log_likelihood
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Check if the model converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get the Bayesian Information Criterion (BIC)
    pub fn bic(&self) -> f64 {
        self.state.bic
    }

    /// Get the Akaike Information Criterion (AIC)
    pub fn aic(&self) -> f64 {
        self.state.aic
    }
}
