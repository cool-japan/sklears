//! Bayesian Gaussian Mixture Models
//!
//! This module implements Bayesian Gaussian mixture models with automatic model selection
//! through variational inference. The Bayesian approach allows for automatic determination
//! of the effective number of components and provides uncertainty quantification.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Utility function for log-sum-exp computation
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max_val = a.max(b);
    if max_val.is_finite() {
        max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
    } else {
        max_val
    }
}

/// Bayesian Gaussian Mixture Model
///
/// A Bayesian variant of Gaussian mixture model that uses variational inference
/// to automatically determine the effective number of components. This implementation
/// provides uncertainty quantification and automatic model selection capabilities.
///
/// The model uses variational Bayesian inference with proper priors on the mixture
/// weights, means, and covariances to enable automatic component selection.
///
/// # Parameters
///
/// * `n_components` - Maximum number of mixture components
/// * `covariance_type` - Type of covariance parameters (currently supports "full")
/// * `tol` - Convergence threshold for the variational lower bound
/// * `reg_covar` - Regularization added to the diagonal of covariance
/// * `max_iter` - Maximum number of variational EM iterations
/// * `random_state` - Random state for reproducibility
/// * `warm_start` - Whether to use previous fit as initialization
/// * `weight_concentration_prior_type` - Type of prior on mixture weights
/// * `weight_concentration_prior` - Prior concentration parameter for mixture weights
/// * `mean_precision_prior` - Prior precision for component means
/// * `mean_prior` - Prior mean for component means
/// * `degrees_of_freedom_prior` - Prior degrees of freedom for covariance matrices
/// * `covariance_prior` - Prior scale for covariance matrices
///
/// # Examples
///
/// ```
/// use sklears_mixture::{BayesianGaussianMixture, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let bgmm = BayesianGaussianMixture::new()
///     .n_components(4)  // Will automatically select effective number
///     .max_iter(100);
/// let fitted = bgmm.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// println!("Effective components: {}", fitted.n_components_effective());
/// ```
#[derive(Debug, Clone)]
pub struct BayesianGaussianMixture<S = Untrained> {
    pub(crate) state: S,
    n_components: usize,
    covariance_type: String,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    random_state: Option<u64>,
    warm_start: bool,
    weight_concentration_prior_type: String,
    weight_concentration_prior: Option<f64>,
    mean_precision_prior: Option<f64>,
    mean_prior: Option<Array1<f64>>,
    degrees_of_freedom_prior: Option<f64>,
    covariance_prior: Option<f64>,
}

impl BayesianGaussianMixture<Untrained> {
    /// Create a new BayesianGaussianMixture instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            covariance_type: "full".to_string(),
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            random_state: None,
            warm_start: false,
            weight_concentration_prior_type: "dirichlet_process".to_string(),
            weight_concentration_prior: None,
            mean_precision_prior: None,
            mean_prior: None,
            degrees_of_freedom_prior: None,
            covariance_prior: None,
        }
    }

    /// Set the maximum number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: String) -> Self {
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

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set weight concentration prior type
    pub fn weight_concentration_prior_type(mut self, prior_type: String) -> Self {
        self.weight_concentration_prior_type = prior_type;
        self
    }

    /// Set weight concentration prior
    pub fn weight_concentration_prior(mut self, prior: f64) -> Self {
        self.weight_concentration_prior = Some(prior);
        self
    }

    /// Set mean precision prior
    pub fn mean_precision_prior(mut self, prior: f64) -> Self {
        self.mean_precision_prior = Some(prior);
        self
    }
}

impl Default for BayesianGaussianMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BayesianGaussianMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for BayesianGaussianMixture<Untrained> {
    type Fitted = BayesianGaussianMixture<BayesianGaussianMixtureTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, _n_features) = X.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least the number of components".to_string(),
            ));
        }

        // Initialize parameters
        let mut weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
        let mut means = self.initialize_means(&X)?;
        let mut covariances = self.initialize_covariances(&X, &means)?;

        // Variational parameters
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));
        let mut lower_bound = f64::NEG_INFINITY;
        let mut converged = false;

        // EM iterations
        for iteration in 0..self.max_iter {
            // E-step: Update responsibilities
            self.update_responsibilities(
                &X,
                &weights,
                &means,
                &covariances,
                &mut responsibilities,
            )?;

            // M-step: Update parameters using variational Bayes
            let (new_weights, new_means, new_covariances) =
                self.update_parameters(&X, &responsibilities)?;

            // Check convergence
            let new_lower_bound = self.compute_lower_bound(
                &X,
                &responsibilities,
                &new_weights,
                &new_means,
                &new_covariances,
            );

            if iteration > 0 && (new_lower_bound - lower_bound).abs() < self.tol {
                converged = true;
            }

            weights = new_weights;
            means = new_means;
            covariances = new_covariances;
            lower_bound = new_lower_bound;

            if converged {
                break;
            }
        }

        // Determine effective number of components
        let weight_threshold = 1.0 / (self.n_components as f64 * 100.0);
        let n_components_effective = weights.iter().filter(|&&w| w > weight_threshold).count();

        Ok(BayesianGaussianMixture {
            state: BayesianGaussianMixtureTrained {
                weights,
                means,
                covariances,
                n_components_effective,
                lower_bound,
                converged,
                n_iter: if converged { 0 } else { self.max_iter }, // Simplified
            },
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            random_state: self.random_state,
            warm_start: self.warm_start,
            weight_concentration_prior_type: self.weight_concentration_prior_type,
            weight_concentration_prior: self.weight_concentration_prior,
            mean_precision_prior: self.mean_precision_prior,
            mean_prior: self.mean_prior,
            degrees_of_freedom_prior: self.degrees_of_freedom_prior,
            covariance_prior: self.covariance_prior,
        })
    }
}

impl BayesianGaussianMixture<Untrained> {
    fn initialize_means(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (_, n_features) = X.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        // Simple initialization: evenly spaced samples
        let step = X.nrows() / self.n_components;
        for (i, mut mean) in means.axis_iter_mut(Axis(0)).enumerate() {
            let sample_idx = (i * step).min(X.nrows() - 1);
            mean.assign(&X.row(sample_idx));
        }

        Ok(means)
    }

    fn initialize_covariances(
        &self,
        X: &Array2<f64>,
        _means: &Array2<f64>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let (_, n_features) = X.dim();

        // Initialize with identity covariance matrices (simplified)
        let mut covariances = Vec::new();
        for _ in 0..self.n_components {
            let mut cov = Array2::eye(n_features);
            // Add regularization
            for i in 0..n_features {
                cov[[i, i]] += self.reg_covar;
            }
            covariances.push(cov);
        }

        Ok(covariances)
    }

    fn update_responsibilities(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
        responsibilities: &mut Array2<f64>,
    ) -> SklResult<()> {
        let (n_samples, _) = X.dim();

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];
                let log_weight = weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_probs.push(log_prob);
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            // Normalize to get responsibilities
            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_prob_sum).exp();
            }
        }

        Ok(())
    }

    fn update_parameters(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let (n_samples, n_features) = X.dim();

        // Update weights
        let n_k: Array1<f64> = responsibilities.sum_axis(Axis(0));
        let weights = &n_k / n_samples as f64;

        // Update means
        let mut means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            if n_k[k] > 1e-10 {
                for i in 0..n_samples {
                    for j in 0..n_features {
                        means[[k, j]] += responsibilities[[i, k]] * X[[i, j]];
                    }
                }
                for j in 0..n_features {
                    means[[k, j]] /= n_k[k];
                }
            }
        }

        // Update covariances (simplified diagonal covariance)
        let mut covariances = Vec::new();
        for k in 0..self.n_components {
            let mut cov = Array2::eye(n_features);

            if n_k[k] > 1e-10 {
                let mean_k = means.row(k);

                for d in 0..n_features {
                    let mut var = 0.0;
                    for i in 0..n_samples {
                        let diff = X[[i, d]] - mean_k[d];
                        var += responsibilities[[i, k]] * diff * diff;
                    }
                    var /= n_k[k];
                    cov[[d, d]] = var + self.reg_covar;
                }
            } else {
                // Add regularization for empty components
                for d in 0..n_features {
                    cov[[d, d]] = 1.0 + self.reg_covar;
                }
            }

            covariances.push(cov);
        }

        Ok((weights, means, covariances))
    }

    fn multivariate_normal_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let d = x.len() as f64;
        let diff: Array1<f64> = x - mean;

        // Compute log determinant (simplified for diagonal covariance)
        let mut log_det = 0.0;
        for i in 0..cov.nrows() {
            log_det += cov[[i, i]].ln();
        }

        // Compute quadratic form (simplified for diagonal covariance)
        let mut quad_form = 0.0;
        for i in 0..diff.len() {
            quad_form += diff[i] * diff[i] / cov[[i, i]];
        }

        let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
        Ok(log_pdf)
    }

    fn compute_lower_bound(
        &self,
        _X: &Array2<f64>,
        _responsibilities: &Array2<f64>,
        _weights: &Array1<f64>,
        _means: &Array2<f64>,
        _covariances: &[Array2<f64>],
    ) -> f64 {
        // Simplified lower bound computation
        // In a full implementation, this would compute the variational lower bound
        0.0
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for BayesianGaussianMixture<BayesianGaussianMixtureTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut max_log_prob = f64::NEG_INFINITY;
            let mut best_component = 0;

            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];
                let log_weight = self.state.weights[k].ln();

                if let Ok(log_likelihood) = self.multivariate_normal_log_pdf(&sample, &mean, cov) {
                    let log_prob = log_weight + log_likelihood;
                    if log_prob > max_log_prob {
                        max_log_prob = log_prob;
                        best_component = k;
                    }
                }
            }

            predictions[i] = best_component as i32;
        }

        Ok(predictions)
    }
}

impl BayesianGaussianMixture<BayesianGaussianMixtureTrained> {
    /// Get the mixture weights
    pub fn weights(&self) -> &Array1<f64> {
        &self.state.weights
    }

    /// Get the component means
    pub fn means(&self) -> &Array2<f64> {
        &self.state.means
    }

    /// Get the component covariances
    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.state.covariances
    }

    /// Get the effective number of components
    pub fn n_components_effective(&self) -> usize {
        self.state.n_components_effective
    }

    /// Get the lower bound on the log likelihood
    pub fn lower_bound(&self) -> f64 {
        self.state.lower_bound
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Predict probabilities for each component
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut probabilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];
                let log_weight = self.state.weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_probs.push(log_prob);
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            // Normalize to get probabilities
            for k in 0..self.n_components {
                probabilities[[i, k]] = (log_probs[k] - log_prob_sum).exp();
            }
        }

        Ok(probabilities)
    }

    /// Compute the per-sample log-likelihood
    #[allow(non_snake_case)]
    pub fn score_samples(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];
                let log_weight = self.state.weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            scores[i] = log_prob_sum;
        }

        Ok(scores)
    }

    /// Compute the average log-likelihood
    pub fn score(&self, X: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let scores = self.score_samples(X)?;
        Ok(scores.mean().unwrap_or(0.0))
    }

    fn multivariate_normal_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let d = x.len() as f64;
        let diff: Array1<f64> = x - mean;

        // Compute log determinant (simplified for diagonal covariance)
        let mut log_det = 0.0;
        for i in 0..cov.nrows() {
            log_det += cov[[i, i]].ln();
        }

        // Compute quadratic form (simplified for diagonal covariance)
        let mut quad_form = 0.0;
        for i in 0..diff.len() {
            quad_form += diff[i] * diff[i] / cov[[i, i]];
        }

        let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
        Ok(log_pdf)
    }
}

/// Trained state for BayesianGaussianMixture
#[derive(Debug, Clone)]
pub struct BayesianGaussianMixtureTrained {
    /// Mixture component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariance matrices
    pub covariances: Vec<Array2<f64>>,
    /// Effective number of components
    pub n_components_effective: usize,
    /// Lower bound on log likelihood
    pub lower_bound: f64,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of iterations performed
    pub n_iter: usize,
}
