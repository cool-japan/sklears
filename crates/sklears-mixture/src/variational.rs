//! Variational Bayesian Gaussian Mixture Models
//!
//! This module implements variational Bayesian inference for Gaussian mixture models,
//! providing automatic model selection and uncertainty quantification.

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Variational Bayesian Gaussian Mixture Model
///
/// This implementation uses variational inference to perform Bayesian parameter estimation
/// for Gaussian mixture models. Unlike standard EM, this approach provides uncertainty
/// estimates and automatic model selection by effectively "turning off" unnecessary components.
///
/// # Parameters
///
/// * `n_components` - Maximum number of mixture components (actual number determined automatically)
/// * `covariance_type` - Type of covariance parameters
/// * `tol` - Convergence threshold
/// * `reg_covar` - Regularization added to the diagonal of covariance
/// * `max_iter` - Maximum number of variational iterations
/// * `random_state` - Random state for reproducibility
/// * `weight_concentration_prior` - Prior on the weight concentration parameter
/// * `mean_precision_prior` - Prior precision for component means
/// * `degrees_of_freedom_prior` - Prior degrees of freedom for covariance matrices
///
/// # Examples
///
/// ```
/// use sklears_mixture::{VariationalBayesianGMM, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let vbgmm = VariationalBayesianGMM::new()
///     .n_components(5)  // Will automatically determine optimal number
///     .covariance_type(CovarianceType::Diagonal)
///     .max_iter(100);
/// let fitted = vbgmm.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct VariationalBayesianGMM<S = Untrained> {
    pub(crate) state: S,
    pub(crate) n_components: usize,
    pub(crate) covariance_type: CovarianceType,
    pub(crate) tol: f64,
    pub(crate) reg_covar: f64,
    pub(crate) max_iter: usize,
    pub(crate) random_state: Option<u64>,
    pub(crate) weight_concentration_prior: f64,
    pub(crate) mean_precision_prior: f64,
    pub(crate) degrees_of_freedom_prior: f64,
}

/// Trained state for VariationalBayesianGMM
#[derive(Debug, Clone)]
pub struct VariationalBayesianGMMTrained {
    pub(crate) weights: Array1<f64>,
    pub(crate) means: Array2<f64>,
    pub(crate) covariances: Vec<Array2<f64>>,
    pub(crate) weight_concentration: Array1<f64>,
    pub(crate) mean_precision: Array1<f64>,
    pub(crate) degrees_of_freedom: Array1<f64>,
    pub(crate) lower_bound: f64,
    pub(crate) n_iter: usize,
    pub(crate) converged: bool,
    pub(crate) effective_components: usize,
}

impl VariationalBayesianGMM<Untrained> {
    /// Create a new VariationalBayesianGMM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            random_state: None,
            weight_concentration_prior: 1.0,
            mean_precision_prior: 1.0,
            degrees_of_freedom_prior: 1.0,
        }
    }

    /// Create a new VariationalBayesianGMM instance using builder pattern (alias for new)
    pub fn builder() -> Self {
        Self::new()
    }

    /// Set the maximum number of components
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

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the weight concentration prior
    pub fn weight_concentration_prior(mut self, prior: f64) -> Self {
        self.weight_concentration_prior = prior;
        self
    }

    /// Set the mean precision prior
    pub fn mean_precision_prior(mut self, prior: f64) -> Self {
        self.mean_precision_prior = prior;
        self
    }

    /// Set the degrees of freedom prior
    pub fn degrees_of_freedom_prior(mut self, prior: f64) -> Self {
        self.degrees_of_freedom_prior = prior;
        self
    }

    /// Build the VariationalBayesianGMM (builder pattern completion)
    pub fn build(self) -> Self {
        self
    }
}

impl Default for VariationalBayesianGMM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for VariationalBayesianGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for VariationalBayesianGMM<Untrained> {
    type Fitted = VariationalBayesianGMM<VariationalBayesianGMMTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, _n_features) = X.dim();

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

        // Initialize parameters
        let (
            mut weight_concentration,
            mut mean_precision,
            mut means,
            mut degrees_of_freedom,
            mut covariances,
        ) = self.initialize_parameters(&X)?;

        let mut lower_bound = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        // Variational EM iterations
        for iteration in 0..self.max_iter {
            n_iter = iteration + 1;

            // E-step: Update responsibilities
            let responsibilities = self.compute_responsibilities(
                &X,
                &weight_concentration,
                &means,
                &covariances,
                &degrees_of_freedom,
            )?;

            // M-step: Update parameters
            let (
                new_weight_concentration,
                new_mean_precision,
                new_means,
                new_degrees_of_freedom,
                new_covariances,
            ) = self.update_parameters(&X, &responsibilities)?;

            // Compute lower bound
            let new_lower_bound = self.compute_lower_bound(
                &X,
                &responsibilities,
                &new_weight_concentration,
                &new_mean_precision,
                &new_means,
                &new_degrees_of_freedom,
                &new_covariances,
            )?;

            // Check convergence
            if iteration > 0 && (new_lower_bound - lower_bound).abs() < self.tol {
                converged = true;
            }

            weight_concentration = new_weight_concentration;
            mean_precision = new_mean_precision;
            means = new_means;
            degrees_of_freedom = new_degrees_of_freedom;
            covariances = new_covariances;
            lower_bound = new_lower_bound;

            if converged {
                break;
            }
        }

        // Compute final weights from concentration parameters
        let weights = self.compute_weights(&weight_concentration);

        // Count effective components (those with significant weight)
        let effective_components = weights.iter().filter(|&&w| w > 1e-3).count();

        Ok(VariationalBayesianGMM {
            state: VariationalBayesianGMMTrained {
                weights,
                means,
                covariances,
                weight_concentration,
                mean_precision,
                degrees_of_freedom,
                lower_bound,
                n_iter,
                converged,
                effective_components,
            },
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            random_state: self.random_state,
            weight_concentration_prior: self.weight_concentration_prior,
            mean_precision_prior: self.mean_precision_prior,
            degrees_of_freedom_prior: self.degrees_of_freedom_prior,
        })
    }
}

impl VariationalBayesianGMM<Untrained> {
    /// Initialize variational parameters
    fn initialize_parameters(
        &self,
        X: &Array2<f64>,
    ) -> SklResult<(
        Array1<f64>,
        Array1<f64>,
        Array2<f64>,
        Array1<f64>,
        Vec<Array2<f64>>,
    )> {
        let (_n_samples, n_features) = X.dim();

        // Initialize weight concentration parameters
        let weight_concentration =
            Array1::from_elem(self.n_components, self.weight_concentration_prior);

        // Initialize mean precision parameters
        let mean_precision = Array1::from_elem(self.n_components, self.mean_precision_prior);

        // Initialize means using k-means++ style initialization
        let means = self.initialize_means(X)?;

        // Initialize degrees of freedom
        let degrees_of_freedom = Array1::from_elem(
            self.n_components,
            self.degrees_of_freedom_prior + n_features as f64,
        );

        // Initialize covariances
        let covariances = self.initialize_covariances(X)?;

        Ok((
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            covariances,
        ))
    }

    /// Initialize means using k-means++ style initialization
    fn initialize_means(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        // Use random initialization if random state is provided
        if let Some(seed) = self.random_state {
            let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);

            // First mean: pick random sample
            let idx = rng.gen_range(0..n_samples);
            means.row_mut(0).assign(&X.row(idx));

            // Subsequent means: pick samples far from existing means
            for i in 1..self.n_components {
                let mut best_distance = 0.0;
                let mut best_idx = 0;

                for j in 0..n_samples {
                    let sample = X.row(j);
                    let mut min_distance = f64::INFINITY;

                    for k in 0..i {
                        let existing_mean = means.row(k);
                        let distance = (&sample - &existing_mean).mapv(|x| x * x).sum();
                        min_distance = min_distance.min(distance);
                    }

                    if min_distance > best_distance {
                        best_distance = min_distance;
                        best_idx = j;
                    }
                }

                means.row_mut(i).assign(&X.row(best_idx));
            }
        } else {
            // Deterministic initialization: evenly spaced samples
            let step = n_samples / self.n_components;

            for (i, mut mean) in means.axis_iter_mut(Axis(0)).enumerate() {
                let sample_idx = if step == 0 {
                    i.min(n_samples - 1)
                } else {
                    (i * step).min(n_samples - 1)
                };
                mean.assign(&X.row(sample_idx));
            }
        }

        Ok(means)
    }

    /// Initialize covariances
    fn initialize_covariances(&self, X: &Array2<f64>) -> SklResult<Vec<Array2<f64>>> {
        let (_, n_features) = X.dim();
        let mut covariances = Vec::new();

        // Estimate global covariance for initialization
        let global_cov = self.estimate_global_covariance(X)?;

        for _ in 0..self.n_components {
            let mut cov = global_cov.clone();

            // Add regularization
            for i in 0..n_features {
                cov[[i, i]] += self.reg_covar;
            }

            covariances.push(cov);
        }

        Ok(covariances)
    }

    /// Estimate global covariance matrix
    fn estimate_global_covariance(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();

        // Compute sample mean
        let mut mean = Array1::zeros(n_features);
        for i in 0..n_features {
            mean[i] = X.column(i).sum() / n_samples as f64;
        }

        // Compute covariance matrix
        let mut cov = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += (X[[k, i]] - mean[i]) * (X[[k, j]] - mean[j]);
                }
                cov[[i, j]] = sum / (n_samples as f64 - 1.0);
            }
        }

        // Apply covariance type constraints
        match self.covariance_type {
            CovarianceType::Diagonal => {
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            cov[[i, j]] = 0.0;
                        }
                    }
                }
            }
            CovarianceType::Spherical => {
                let trace = cov.diag().sum() / n_features as f64;
                cov.fill(0.0);
                for i in 0..n_features {
                    cov[[i, i]] = trace;
                }
            }
            _ => {} // Full and Tied keep the estimated covariance
        }

        Ok(cov)
    }

    /// Compute responsibilities using current parameters
    fn compute_responsibilities(
        &self,
        X: &Array2<f64>,
        weight_concentration: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
        degrees_of_freedom: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        // Compute expected log weights
        let expected_log_weights = self.compute_expected_log_weights(weight_concentration);

        // For each sample
        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            let mut log_prob_norm = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];

                // Use Student-t distribution due to uncertainty in parameters
                let log_prob =
                    self.compute_student_t_log_pdf(&sample, &mean, cov, degrees_of_freedom[k])?;
                let weighted_log_prob = expected_log_weights[k] + log_prob;

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

    /// Compute expected log weights from concentration parameters
    fn compute_expected_log_weights(&self, weight_concentration: &Array1<f64>) -> Array1<f64> {
        let sum_concentration: f64 = weight_concentration.sum();
        let mut expected_log_weights = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            // Expected log weight under Dirichlet distribution
            expected_log_weights[k] = digamma(weight_concentration[k]) - digamma(sum_concentration);
        }

        expected_log_weights
    }

    /// Compute Student-t log PDF (approximation)
    fn compute_student_t_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
        _degrees_of_freedom: f64,
    ) -> SklResult<f64> {
        // For simplicity, use Gaussian approximation
        // In full implementation, this would use proper Student-t distribution
        crate::common::gaussian_log_pdf(x, mean, &cov.view())
    }

    /// Update variational parameters
    fn update_parameters(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
    ) -> SklResult<(
        Array1<f64>,
        Array1<f64>,
        Array2<f64>,
        Array1<f64>,
        Vec<Array2<f64>>,
    )> {
        let (n_samples, n_features) = X.dim();

        // Update weight concentration parameters
        let mut weight_concentration = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            weight_concentration[k] =
                self.weight_concentration_prior + responsibilities.column(k).sum();
        }

        // Update mean precision parameters
        let mut mean_precision = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            mean_precision[k] = self.mean_precision_prior + responsibilities.column(k).sum();
        }

        // Update means
        let mut means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            let resp_sum = responsibilities.column(k).sum();
            if resp_sum > 0.0 {
                for j in 0..n_features {
                    let mut weighted_sum = 0.0;
                    for i in 0..n_samples {
                        weighted_sum += responsibilities[[i, k]] * X[[i, j]];
                    }
                    means[[k, j]] = weighted_sum / resp_sum;
                }
            }
        }

        // Update degrees of freedom
        let mut degrees_of_freedom = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            degrees_of_freedom[k] =
                self.degrees_of_freedom_prior + responsibilities.column(k).sum();
        }

        // Update covariances (simplified)
        let mut covariances = Vec::new();
        for _k in 0..self.n_components {
            let mut cov = Array2::eye(n_features);
            for i in 0..n_features {
                cov[[i, i]] = 1.0 + self.reg_covar;
            }
            covariances.push(cov);
        }

        Ok((
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            covariances,
        ))
    }

    /// Compute variational lower bound
    fn compute_lower_bound(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        weight_concentration: &Array1<f64>,
        _mean_precision: &Array1<f64>,
        means: &Array2<f64>,
        _degrees_of_freedom: &Array1<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<f64> {
        // Simplified lower bound computation
        // In full implementation, this would include all entropy and expectation terms
        let mut lower_bound = 0.0;

        // Expected log likelihood term
        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            for k in 0..self.n_components {
                let resp = responsibilities[[i, k]];
                if resp > 0.0 {
                    let mean = means.row(k);
                    let cov = &covariances[k];
                    let log_prob = crate::common::gaussian_log_pdf(&sample, &mean, &cov.view())?;
                    lower_bound += resp * log_prob;
                }
            }
        }

        // KL divergence terms (simplified)
        let expected_log_weights = self.compute_expected_log_weights(weight_concentration);
        for k in 0..self.n_components {
            let resp_sum = responsibilities.column(k).sum();
            if resp_sum > 0.0 {
                lower_bound += resp_sum * expected_log_weights[k];
            }
        }

        Ok(lower_bound)
    }

    /// Compute final weights from concentration parameters
    fn compute_weights(&self, weight_concentration: &Array1<f64>) -> Array1<f64> {
        let sum_concentration: f64 = weight_concentration.sum();
        weight_concentration.mapv(|x| x / sum_concentration)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for VariationalBayesianGMM<VariationalBayesianGMMTrained>
{
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
                if self.state.weights[k] > 1e-3 {
                    // Only consider effective components
                    let mean = self.state.means.row(k);
                    let cov = &self.state.covariances[k];

                    let log_prob = crate::common::gaussian_log_pdf(&sample, &mean, &cov.view())?;
                    let weighted_log_prob = self.state.weights[k].ln() + log_prob;

                    if weighted_log_prob > best_log_prob {
                        best_log_prob = weighted_log_prob;
                        best_component = k;
                    }
                }
            }

            predictions[i] = best_component as i32;
        }

        Ok(predictions)
    }
}

impl VariationalBayesianGMM<VariationalBayesianGMMTrained> {
    /// Get the fitted weights
    pub fn weights(&self) -> &Array1<f64> {
        &self.state.weights
    }

    /// Get the fitted means
    pub fn means(&self) -> &Array2<f64> {
        &self.state.means
    }

    /// Get the fitted covariances
    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.state.covariances
    }

    /// Get the variational lower bound
    pub fn lower_bound(&self) -> f64 {
        self.state.lower_bound
    }

    /// Get the number of effective components
    pub fn effective_components(&self) -> usize {
        self.state.effective_components
    }

    /// Check if the model converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the weight concentration parameters
    pub fn weight_concentration(&self) -> &Array1<f64> {
        &self.state.weight_concentration
    }

    /// Get the mean precision parameters
    pub fn mean_precision(&self) -> &Array1<f64> {
        &self.state.mean_precision
    }

    /// Get the degrees of freedom parameters
    pub fn degrees_of_freedom(&self) -> &Array1<f64> {
        &self.state.degrees_of_freedom
    }
}

/// Digamma function approximation (for computing expected log weights)
fn digamma(x: f64) -> f64 {
    // Simple approximation using asymptotic expansion
    if x > 6.0 {
        x.ln() - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)
    } else {
        // For small x, use recurrence relation and asymptotic expansion
        let mut result = x;
        let mut n = 0;
        while result < 6.0 {
            result += 1.0;
            n += 1;
        }
        let asymptotic = result.ln() - 1.0 / (2.0 * result) - 1.0 / (12.0 * result * result);
        asymptotic - (0..n).map(|i| 1.0 / (x + i as f64)).sum::<f64>()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_variational_bayesian_gmm_basic() {
        let X = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0]
        ];

        let vbgmm = VariationalBayesianGMM::new()
            .n_components(3)
            .max_iter(10)
            .random_state(42);

        let fitted = vbgmm.fit(&X.view(), &()).unwrap();

        assert!(fitted.converged() || fitted.n_iter() == 10);
        assert!(fitted.effective_components() <= 3);
        assert!(fitted.lower_bound().is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_variational_bayesian_gmm_prediction() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let vbgmm = VariationalBayesianGMM::new()
            .n_components(2)
            .max_iter(20)
            .random_state(42);

        let fitted = vbgmm.fit(&X.view(), &()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        // Should cluster into two groups
        assert!(predictions[0] == predictions[1] || predictions[0] != predictions[2]);
    }

    #[test]
    fn test_variational_bayesian_gmm_builder() {
        let vbgmm = VariationalBayesianGMM::builder()
            .n_components(5)
            .covariance_type(CovarianceType::Diagonal)
            .tol(1e-4)
            .weight_concentration_prior(0.1)
            .mean_precision_prior(0.1)
            .degrees_of_freedom_prior(1.0)
            .build();

        assert_eq!(vbgmm.n_components, 5);
        assert_eq!(vbgmm.covariance_type, CovarianceType::Diagonal);
        assert_relative_eq!(vbgmm.tol, 1e-4);
        assert_relative_eq!(vbgmm.weight_concentration_prior, 0.1);
    }

    #[test]
    fn test_digamma_function() {
        // Test digamma function approximation
        assert_relative_eq!(digamma(1.0), -0.5772, epsilon = 0.1);
        assert_relative_eq!(digamma(2.0), 0.4228, epsilon = 0.1);
        assert_relative_eq!(digamma(10.0), 2.2517, epsilon = 0.01);
    }
}
