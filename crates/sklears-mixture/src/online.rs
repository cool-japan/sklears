//! Online Gaussian Mixture Models
//!
//! This module implements online Gaussian mixture models that can be updated incrementally
//! as new data arrives. This is useful for streaming data and large datasets that
//! don't fit in memory.

use crate::common::{CovarianceType, ModelSelection};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
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

/// Online Gaussian Mixture Model
///
/// An online version of Gaussian mixture model that can be updated incrementally
/// as new data arrives. This is useful for streaming data and large datasets that
/// don't fit in memory.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `covariance_type` - Type of covariance parameters
/// * `tol` - Convergence threshold
/// * `reg_covar` - Regularization added to the diagonal of covariance
/// * `max_iter` - Maximum number of EM iterations for initial fitting
/// * `learning_rate` - Initial learning rate for online updates
/// * `decay_rate` - Decay rate for learning rate
/// * `batch_size` - Size of mini-batches for online updates
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_mixture::{OnlineGaussianMixture, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X_initial = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
/// let X_new = array![[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let mut ogmm = OnlineGaussianMixture::new()
///     .n_components(2)
///     .covariance_type(CovarianceType::Diagonal)
///     .learning_rate(0.1)
///     .max_iter(50);
///
/// let mut fitted = ogmm.fit(&X_initial.view(), &()).unwrap();
/// fitted = fitted.partial_fit(&X_new.view()).unwrap(); // Update with new data
/// let labels = fitted.predict(&X_new.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OnlineGaussianMixture<S = Untrained> {
    pub(crate) state: S,
    n_components: usize,
    covariance_type: CovarianceType,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    learning_rate: f64,
    decay_rate: f64,
    batch_size: usize,
    random_state: Option<u64>,
}

impl OnlineGaussianMixture<Untrained> {
    /// Create a new OnlineGaussianMixture instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            learning_rate: 0.01,
            decay_rate: 0.9,
            batch_size: 100,
            random_state: None,
        }
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

    /// Set the initial learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the decay rate for learning rate
    pub fn decay_rate(mut self, decay_rate: f64) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Set the batch size for online updates
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for OnlineGaussianMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OnlineGaussianMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for OnlineGaussianMixture<Untrained> {
    type Fitted = OnlineGaussianMixture<OnlineGaussianMixtureTrained>;

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

        // Initialize parameters using standard EM
        let (mut weights, mut means, mut covariances) = self.initialize_parameters(&X)?;

        let mut log_likelihood = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        // Initial batch EM fitting
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

        // Calculate model selection criteria
        let n_params =
            ModelSelection::n_parameters(self.n_components, n_features, &self.covariance_type);
        let bic = ModelSelection::bic(log_likelihood, n_params, n_samples);
        let aic = ModelSelection::aic(log_likelihood, n_params);

        Ok(OnlineGaussianMixture {
            state: OnlineGaussianMixtureTrained {
                weights,
                means,
                covariances,
                log_likelihood,
                n_iter,
                converged,
                bic,
                aic,
                update_count: 0,
                current_learning_rate: self.learning_rate,
                total_samples_seen: n_samples,
            },
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            decay_rate: self.decay_rate,
            batch_size: self.batch_size,
            random_state: self.random_state,
        })
    }
}

impl OnlineGaussianMixture<Untrained> {
    /// Initialize parameters for EM algorithm
    fn initialize_parameters(
        &self,
        X: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        // Initialize weights (uniform)
        let weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);

        // Initialize means using evenly spaced samples
        let means = self.initialize_means(X)?;

        // Initialize covariances
        let covariances = self.initialize_covariances(X, &means)?;

        Ok((weights, means, covariances))
    }

    /// Initialize means using evenly spaced samples
    fn initialize_means(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        let step = n_samples / self.n_components;

        for (i, mut mean) in means.axis_iter_mut(Axis(0)).enumerate() {
            let sample_idx = if step == 0 {
                i.min(n_samples - 1)
            } else {
                (i * step).min(n_samples - 1)
            };
            mean.assign(&X.row(sample_idx));
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
                for _ in 0..self.n_components {
                    let mut cov = Array2::eye(n_features);
                    for i in 0..n_features {
                        cov[[i, i]] += self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                for _ in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = 1.0 + self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                let mut cov = Array2::eye(n_features);
                for i in 0..n_features {
                    cov[[i, i]] += self.reg_covar;
                }
                for _ in 0..self.n_components {
                    covariances.push(cov.clone());
                }
            }
            CovarianceType::Spherical => {
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

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];
                let log_weight = weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_probs.push(log_prob);
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_prob_sum).exp();
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

        // Update covariances
        let covariances = self.update_covariances(X, responsibilities, &means, &n_k)?;

        Ok((weights, means, covariances))
    }

    /// Update covariances based on covariance type
    fn update_covariances(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        means: &Array2<f64>,
        n_k: &Array1<f64>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let (n_samples, n_features) = X.dim();
        let mut covariances = Vec::new();

        match self.covariance_type {
            CovarianceType::Full => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = means.row(k);

                        for i in 0..n_samples {
                            let sample = X.row(i);
                            let diff = &sample - &mean_k;

                            for d1 in 0..n_features {
                                for d2 in 0..n_features {
                                    cov[[d1, d2]] += responsibilities[[i, k]] * diff[d1] * diff[d2];
                                }
                            }
                        }

                        for d1 in 0..n_features {
                            for d2 in 0..n_features {
                                cov[[d1, d2]] /= n_k[k];
                            }
                        }

                        for d in 0..n_features {
                            cov[[d, d]] += self.reg_covar;
                        }
                    } else {
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar;
                        }
                    }

                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

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
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar;
                        }
                    }

                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                let mut cov = Array2::zeros((n_features, n_features));
                let total_responsibility: f64 = n_k.sum();

                if total_responsibility > 1e-10 {
                    for k in 0..self.n_components {
                        let mean_k = means.row(k);

                        for i in 0..n_samples {
                            let sample = X.row(i);
                            let diff = &sample - &mean_k;

                            for d1 in 0..n_features {
                                for d2 in 0..n_features {
                                    cov[[d1, d2]] += responsibilities[[i, k]] * diff[d1] * diff[d2];
                                }
                            }
                        }
                    }

                    for d1 in 0..n_features {
                        for d2 in 0..n_features {
                            cov[[d1, d2]] /= total_responsibility;
                        }
                    }

                    for d in 0..n_features {
                        cov[[d, d]] += self.reg_covar;
                    }
                } else {
                    for d in 0..n_features {
                        cov[[d, d]] = 1.0 + self.reg_covar;
                    }
                }

                for _ in 0..self.n_components {
                    covariances.push(cov.clone());
                }
            }
            CovarianceType::Spherical => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = means.row(k);
                        let mut total_var = 0.0;

                        for i in 0..n_samples {
                            for d in 0..n_features {
                                let diff = X[[i, d]] - mean_k[d];
                                total_var += responsibilities[[i, k]] * diff * diff;
                            }
                        }

                        total_var /= n_k[k] * n_features as f64;
                        let variance = total_var + self.reg_covar;

                        for d in 0..n_features {
                            cov[[d, d]] = variance;
                        }
                    } else {
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar;
                        }
                    }

                    covariances.push(cov);
                }
            }
        }

        Ok(covariances)
    }

    /// Compute log-likelihood of the data
    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<f64> {
        let (n_samples, _) = X.dim();
        let mut total_log_likelihood = 0.0;

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];
                let log_weight = weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            total_log_likelihood += log_prob_sum;
        }

        Ok(total_log_likelihood)
    }

    /// Compute multivariate normal log probability density function
    fn multivariate_normal_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let d = x.len() as f64;
        let diff: Array1<f64> = x - mean;

        match self.covariance_type {
            CovarianceType::Full => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..cov.nrows() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
            CovarianceType::Diagonal | CovarianceType::Tied | CovarianceType::Spherical => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..diff.len() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
        }
    }
}

/// Trained state for OnlineGaussianMixture
#[derive(Debug, Clone)]
pub struct OnlineGaussianMixtureTrained {
    /// Mixture component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariance matrices or parameters
    pub covariances: Vec<Array2<f64>>,
    /// Log likelihood of the fitted model
    pub log_likelihood: f64,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Number of online updates performed
    pub update_count: usize,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Total number of samples seen during training
    pub total_samples_seen: usize,
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for OnlineGaussianMixture<OnlineGaussianMixtureTrained>
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

impl OnlineGaussianMixture<OnlineGaussianMixtureTrained> {
    /// Partial fit - update the model with new data
    #[allow(non_snake_case)]
    pub fn partial_fit(mut self, X: &ArrayView2<'_, Float>) -> SklResult<Self> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();

        if n_samples == 0 {
            return Ok(self);
        }

        // Update learning rate with decay
        self.state.current_learning_rate *= self.decay_rate;
        self.state.update_count += 1;
        self.state.total_samples_seen += n_samples;

        // Process data in mini-batches
        let mut start_idx = 0;
        while start_idx < n_samples {
            let end_idx = (start_idx + self.batch_size).min(n_samples);
            let batch_X = X.slice(s![start_idx..end_idx, ..]).to_owned();

            self = self.update_with_batch(&batch_X)?;
            start_idx = end_idx;
        }

        Ok(self)
    }

    /// Update model parameters with a single batch
    fn update_with_batch(mut self, batch_X: &Array2<f64>) -> SklResult<Self> {
        // E-step: Compute responsibilities for the batch
        let responsibilities = self.compute_responsibilities(
            batch_X,
            &self.state.weights,
            &self.state.means,
            &self.state.covariances,
        )?;

        // Online M-step: Update parameters with learning rate
        let (batch_weights, batch_means, batch_covariances) =
            self.compute_batch_parameters(batch_X, &responsibilities)?;

        // Blend old and new parameters using learning rate
        let lr = self.state.current_learning_rate;
        let old_weight = 1.0 - lr;

        // Update weights
        for k in 0..self.n_components {
            self.state.weights[k] = old_weight * self.state.weights[k] + lr * batch_weights[k];
        }

        // Update means
        for k in 0..self.n_components {
            for j in 0..self.state.means.ncols() {
                self.state.means[[k, j]] =
                    old_weight * self.state.means[[k, j]] + lr * batch_means[[k, j]];
            }
        }

        // Update covariances
        for k in 0..self.n_components {
            let n_features = self.state.covariances[k].nrows();
            for i in 0..n_features {
                for j in 0..n_features {
                    self.state.covariances[k][[i, j]] = old_weight
                        * self.state.covariances[k][[i, j]]
                        + lr * batch_covariances[k][[i, j]];
                }
            }
        }

        // Update log-likelihood (approximate)
        let batch_log_likelihood = self.compute_log_likelihood(
            batch_X,
            &self.state.weights,
            &self.state.means,
            &self.state.covariances,
        )?;
        self.state.log_likelihood =
            old_weight * self.state.log_likelihood + lr * batch_log_likelihood;

        Ok(self)
    }

    /// Compute parameters for a batch (used in online updates)
    fn compute_batch_parameters(
        &self,
        batch_X: &Array2<f64>,
        responsibilities: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let (batch_size, n_features) = batch_X.dim();

        // Compute batch weights
        let n_k: Array1<f64> = responsibilities.sum_axis(Axis(0));
        let batch_weights = &n_k / batch_size as f64;

        // Compute batch means
        let mut batch_means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            if n_k[k] > 1e-10 {
                for i in 0..batch_size {
                    for j in 0..n_features {
                        batch_means[[k, j]] += responsibilities[[i, k]] * batch_X[[i, j]];
                    }
                }
                for j in 0..n_features {
                    batch_means[[k, j]] /= n_k[k];
                }
            }
        }

        // Compute batch covariances
        let batch_covariances =
            self.compute_batch_covariances(batch_X, responsibilities, &batch_means, &n_k)?;

        Ok((batch_weights, batch_means, batch_covariances))
    }

    /// Compute batch covariances based on covariance type
    fn compute_batch_covariances(
        &self,
        batch_X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        batch_means: &Array2<f64>,
        n_k: &Array1<f64>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let (batch_size, n_features) = batch_X.dim();
        let mut covariances = Vec::new();

        match self.covariance_type {
            CovarianceType::Full => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = batch_means.row(k);

                        for i in 0..batch_size {
                            let sample = batch_X.row(i);
                            let diff = &sample - &mean_k;

                            for d1 in 0..n_features {
                                for d2 in 0..n_features {
                                    cov[[d1, d2]] += responsibilities[[i, k]] * diff[d1] * diff[d2];
                                }
                            }
                        }

                        for d1 in 0..n_features {
                            for d2 in 0..n_features {
                                cov[[d1, d2]] /= n_k[k];
                            }
                        }

                        for d in 0..n_features {
                            cov[[d, d]] += self.reg_covar;
                        }
                    } else {
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar;
                        }
                    }

                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = batch_means.row(k);

                        for d in 0..n_features {
                            let mut var = 0.0;
                            for i in 0..batch_size {
                                let diff = batch_X[[i, d]] - mean_k[d];
                                var += responsibilities[[i, k]] * diff * diff;
                            }
                            var /= n_k[k];
                            cov[[d, d]] = var + self.reg_covar;
                        }
                    } else {
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar;
                        }
                    }

                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                let mut cov = Array2::zeros((n_features, n_features));
                let total_responsibility: f64 = n_k.sum();

                if total_responsibility > 1e-10 {
                    for k in 0..self.n_components {
                        let mean_k = batch_means.row(k);

                        for i in 0..batch_size {
                            let sample = batch_X.row(i);
                            let diff = &sample - &mean_k;

                            for d1 in 0..n_features {
                                for d2 in 0..n_features {
                                    cov[[d1, d2]] += responsibilities[[i, k]] * diff[d1] * diff[d2];
                                }
                            }
                        }
                    }

                    for d1 in 0..n_features {
                        for d2 in 0..n_features {
                            cov[[d1, d2]] /= total_responsibility;
                        }
                    }

                    for d in 0..n_features {
                        cov[[d, d]] += self.reg_covar;
                    }
                } else {
                    for d in 0..n_features {
                        cov[[d, d]] = 1.0 + self.reg_covar;
                    }
                }

                for _ in 0..self.n_components {
                    covariances.push(cov.clone());
                }
            }
            CovarianceType::Spherical => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = batch_means.row(k);
                        let mut total_var = 0.0;

                        for i in 0..batch_size {
                            for d in 0..n_features {
                                let diff = batch_X[[i, d]] - mean_k[d];
                                total_var += responsibilities[[i, k]] * diff * diff;
                            }
                        }

                        total_var /= n_k[k] * n_features as f64;
                        let variance = total_var + self.reg_covar;

                        for d in 0..n_features {
                            cov[[d, d]] = variance;
                        }
                    } else {
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar;
                        }
                    }

                    covariances.push(cov);
                }
            }
        }

        Ok(covariances)
    }

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

    /// Get the log likelihood of the fitted model
    pub fn log_likelihood(&self) -> f64 {
        self.state.log_likelihood
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get the Bayesian Information Criterion
    pub fn bic(&self) -> f64 {
        self.state.bic
    }

    /// Get the Akaike Information Criterion
    pub fn aic(&self) -> f64 {
        self.state.aic
    }

    /// Get the number of online updates performed
    pub fn update_count(&self) -> usize {
        self.state.update_count
    }

    /// Get the current learning rate
    pub fn current_learning_rate(&self) -> f64 {
        self.state.current_learning_rate
    }

    /// Get the total number of samples seen
    pub fn total_samples_seen(&self) -> usize {
        self.state.total_samples_seen
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

            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];
                let log_weight = self.state.weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_probs.push(log_prob);
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

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

    fn compute_responsibilities(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];
                let log_weight = weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_probs.push(log_prob);
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_prob_sum).exp();
            }
        }

        Ok(responsibilities)
    }

    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<f64> {
        let (n_samples, _) = X.dim();
        let mut total_log_likelihood = 0.0;

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];
                let log_weight = weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            total_log_likelihood += log_prob_sum;
        }

        Ok(total_log_likelihood)
    }

    fn multivariate_normal_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let d = x.len() as f64;
        let diff: Array1<f64> = x - mean;

        match self.covariance_type {
            CovarianceType::Full => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..cov.nrows() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
            CovarianceType::Diagonal | CovarianceType::Tied | CovarianceType::Spherical => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..diff.len() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
        }
    }
}
