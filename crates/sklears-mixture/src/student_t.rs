//! Student-t Mixture Models
//!
//! This module provides Student-t mixture models for robust clustering and density estimation.
//! Student-t distributions have heavier tails than Gaussian distributions, making them more
//! robust to outliers and non-Gaussian noise.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

use crate::common::{CovarianceType, ModelSelection};

/// Student-t Mixture Model
///
/// A mixture of multivariate Student-t distributions that provides robustness to outliers
/// compared to Gaussian mixture models. The Student-t distribution has heavier tails,
/// making it more suitable for data with outliers or non-Gaussian noise.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `degrees_of_freedom` - Degrees of freedom for each component (must be > 2.0)
/// * `covariance_type` - Type of covariance parameters
/// * `tol` - Convergence threshold
/// * `reg_covar` - Regularization added to the diagonal of covariance
/// * `max_iter` - Maximum number of EM iterations
/// * `n_init` - Number of initializations to perform
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_mixture::{StudentTMixture, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let model = StudentTMixture::new()
///     .n_components(2)
///     .degrees_of_freedom(vec![3.0, 3.0])  // Heavy tails for robustness
///     .covariance_type(CovarianceType::Diagonal)
///     .max_iter(100);
/// let fitted = model.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StudentTMixture<S = Untrained> {
    state: S,
    n_components: usize,
    degrees_of_freedom: Vec<f64>,
    covariance_type: CovarianceType,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    random_state: Option<u64>,
}

impl StudentTMixture<Untrained> {
    /// Create a new StudentTMixture instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            degrees_of_freedom: vec![4.0], // Default to 4 degrees of freedom
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        // Adjust degrees of freedom vector to match
        self.degrees_of_freedom.resize(n_components, 4.0);
        self
    }

    /// Set the degrees of freedom for each component
    pub fn degrees_of_freedom(mut self, degrees_of_freedom: Vec<f64>) -> Self {
        if degrees_of_freedom.iter().any(|&df| df <= 2.0) {
            panic!("Degrees of freedom must be > 2.0 for finite variance");
        }
        self.degrees_of_freedom = degrees_of_freedom;
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

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Initialize parameters using K-means++ like initialization
    fn initialize_parameters(
        &self,
        X: &Array2<f64>,
        _seed: Option<u64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let (n_samples, n_features) = X.dim();

        // Initialize weights uniformly
        let weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);

        // Initialize means by selecting random data points
        let mut means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            let idx = (k * n_samples / self.n_components).min(n_samples - 1);
            means.row_mut(k).assign(&X.row(idx));
        }

        // Initialize covariances based on covariance type
        let mut covariances = Vec::new();
        match self.covariance_type {
            CovarianceType::Full => {
                let sample_cov = self.compute_sample_covariance(X)?;
                for _ in 0..self.n_components {
                    covariances.push(sample_cov.clone());
                }
            }
            CovarianceType::Diagonal => {
                let sample_var = self.compute_sample_variance(X)?;
                for _ in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for d in 0..n_features {
                        cov[[d, d]] = sample_var[d];
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                let sample_cov = self.compute_sample_covariance(X)?;
                for _ in 0..self.n_components {
                    covariances.push(sample_cov.clone());
                }
            }
            CovarianceType::Spherical => {
                let mean_var = self.compute_sample_variance(X)?.mean().unwrap_or(1.0);
                for _ in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for d in 0..n_features {
                        cov[[d, d]] = mean_var;
                    }
                    covariances.push(cov);
                }
            }
        }

        Ok((weights, means, covariances))
    }

    fn compute_sample_covariance(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mean = X.mean_axis(Axis(0)).unwrap();

        let mut cov = Array2::zeros((n_features, n_features));
        for i in 0..n_samples {
            let diff = &X.row(i) - &mean;
            for j in 0..n_features {
                for k in 0..n_features {
                    cov[[j, k]] += diff[j] * diff[k];
                }
            }
        }
        cov /= (n_samples - 1) as f64;

        // Add regularization
        for d in 0..n_features {
            cov[[d, d]] += self.reg_covar;
        }

        Ok(cov)
    }

    fn compute_sample_variance(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (n_samples, n_features) = X.dim();
        let mean = X.mean_axis(Axis(0)).unwrap();

        let mut var = Array1::zeros(n_features);
        for i in 0..n_samples {
            let diff = &X.row(i) - &mean;
            for j in 0..n_features {
                var[j] += diff[j] * diff[j];
            }
        }
        var /= (n_samples - 1) as f64;

        // Add regularization
        var += self.reg_covar;

        Ok(var)
    }

    /// Compute responsibilities (E-step) for Student-t mixture
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
            let mut log_probs = Vec::new();

            for k in 0..self.n_components {
                let log_weight = weights[k].ln();
                let log_pdf = self.student_t_log_pdf(
                    &sample,
                    &means.row(k),
                    &covariances[k],
                    self.degrees_of_freedom[k],
                )?;
                log_probs.push(log_weight + log_pdf);
            }

            // Numerically stable computation using log-sum-exp
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = max_log_prob
                + log_probs
                    .iter()
                    .map(|&lp| (lp - max_log_prob).exp())
                    .sum::<f64>()
                    .ln();

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum_exp).exp();
            }
        }

        Ok(responsibilities)
    }

    /// Compute log probability density for multivariate Student-t distribution
    fn student_t_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
        df: f64,
    ) -> SklResult<f64> {
        let d = x.len() as f64;

        // Simple approach: use determinant and inverse directly
        let det = self.matrix_determinant(cov)?;
        if det <= 0.0 {
            return Err(SklearsError::NumericalError(
                "Covariance matrix is not positive definite".to_string(),
            ));
        }

        let log_det = det.ln();
        let inv_cov = self.matrix_inverse(cov)?;

        // Compute Mahalanobis distance
        let diff = x - mean;
        let mut mahal_dist = 0.0;
        for i in 0..diff.len() {
            for j in 0..diff.len() {
                mahal_dist += diff[i] * inv_cov[[i, j]] * diff[j];
            }
        }

        // Student-t log PDF
        let gamma_term = Self::log_gamma((df + d) / 2.0) - Self::log_gamma(df / 2.0);
        let const_term = gamma_term - 0.5 * d * (df * PI).ln() - 0.5 * log_det;
        let density_term = -0.5 * (df + d) * (1.0 + mahal_dist / df).ln();

        Ok(const_term + density_term)
    }

    /// Update parameters (M-step) for Student-t mixture
    fn update_parameters(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let (n_samples, n_features) = X.dim();

        // Update weights
        let mut new_weights = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            new_weights[k] = responsibilities.column(k).sum() / n_samples as f64;
        }

        // Compute latent variables for Student-t (weights for each sample)
        let mut latent_weights = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for k in 0..self.n_components {
                let sample = X.row(i);
                let mean = means.row(k);
                let diff = &sample - &mean;

                // Mahalanobis distance
                let inv_cov = self.matrix_inverse(&covariances[k])?;
                let mut mahal_dist = 0.0;
                for j in 0..n_features {
                    for l in 0..n_features {
                        mahal_dist += diff[j] * inv_cov[[j, l]] * diff[l];
                    }
                }

                let df = self.degrees_of_freedom[k];
                latent_weights[[i, k]] = (df + n_features as f64) / (df + mahal_dist);
            }
        }

        // Update means
        let mut new_means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            let mut weighted_sum = Array1::zeros(n_features);
            let mut weight_sum = 0.0;

            for i in 0..n_samples {
                let weight = responsibilities[[i, k]] * latent_weights[[i, k]];
                weighted_sum = &weighted_sum + &(&X.row(i) * weight);
                weight_sum += weight;
            }

            if weight_sum > 1e-10 {
                new_means.row_mut(k).assign(&(&weighted_sum / weight_sum));
            } else {
                new_means.row_mut(k).assign(&means.row(k));
            }
        }

        // Update covariances
        let mut new_covariances = Vec::new();
        match self.covariance_type {
            CovarianceType::Full => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        let weight = responsibilities[[i, k]] * latent_weights[[i, k]];
                        let diff = &X.row(i) - &new_means.row(k);

                        for j in 0..n_features {
                            for l in 0..n_features {
                                cov[[j, l]] += weight * diff[j] * diff[l];
                            }
                        }
                        weight_sum += weight;
                    }

                    if weight_sum > 1e-10 {
                        cov /= weight_sum;
                    } else {
                        cov = covariances[k].clone();
                    }

                    // Add regularization
                    for d in 0..n_features {
                        cov[[d, d]] += self.reg_covar;
                    }

                    new_covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        let weight = responsibilities[[i, k]] * latent_weights[[i, k]];
                        let diff = &X.row(i) - &new_means.row(k);

                        for j in 0..n_features {
                            cov[[j, j]] += weight * diff[j] * diff[j];
                        }
                        weight_sum += weight;
                    }

                    if weight_sum > 1e-10 {
                        for d in 0..n_features {
                            cov[[d, d]] /= weight_sum;
                            cov[[d, d]] += self.reg_covar;
                        }
                    } else {
                        cov = covariances[k].clone();
                    }

                    new_covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                let mut tied_cov = Array2::zeros((n_features, n_features));
                let mut total_weight = 0.0;

                for k in 0..self.n_components {
                    for i in 0..n_samples {
                        let weight = responsibilities[[i, k]] * latent_weights[[i, k]];
                        let diff = &X.row(i) - &new_means.row(k);

                        for j in 0..n_features {
                            for l in 0..n_features {
                                tied_cov[[j, l]] += weight * diff[j] * diff[l];
                            }
                        }
                        total_weight += weight;
                    }
                }

                if total_weight > 1e-10 {
                    tied_cov /= total_weight;
                } else {
                    tied_cov = covariances[0].clone();
                }

                // Add regularization
                for d in 0..n_features {
                    tied_cov[[d, d]] += self.reg_covar;
                }

                for _ in 0..self.n_components {
                    new_covariances.push(tied_cov.clone());
                }
            }
            CovarianceType::Spherical => {
                for k in 0..self.n_components {
                    let mut var = 0.0;
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        let weight = responsibilities[[i, k]] * latent_weights[[i, k]];
                        let diff = &X.row(i) - &new_means.row(k);

                        for j in 0..n_features {
                            var += weight * diff[j] * diff[j];
                        }
                        weight_sum += weight;
                    }

                    if weight_sum > 1e-10 {
                        var = var / (weight_sum * n_features as f64) + self.reg_covar;
                    } else {
                        var = covariances[k][[0, 0]];
                    }

                    let mut cov = Array2::zeros((n_features, n_features));
                    for d in 0..n_features {
                        cov[[d, d]] = var;
                    }

                    new_covariances.push(cov);
                }
            }
        }

        Ok((new_weights, new_means, new_covariances))
    }

    /// Compute log-likelihood of the model
    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
    ) -> SklResult<f64> {
        let (n_samples, _) = X.dim();
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut sample_likelihood = 0.0;

            for k in 0..self.n_components {
                let log_pdf = self.student_t_log_pdf(
                    &sample,
                    &means.row(k),
                    &covariances[k],
                    self.degrees_of_freedom[k],
                )?;
                sample_likelihood += weights[k] * log_pdf.exp();
            }

            if sample_likelihood > 0.0 {
                log_likelihood += sample_likelihood.ln();
            } else {
                return Err(SklearsError::NumericalError(
                    "Zero likelihood encountered".to_string(),
                ));
            }
        }

        Ok(log_likelihood)
    }

    /// Simple matrix determinant for small matrices
    fn matrix_determinant(&self, A: &Array2<f64>) -> SklResult<f64> {
        let n = A.dim().0;
        if n == 1 {
            return Ok(A[[0, 0]]);
        }
        if n == 2 {
            return Ok(A[[0, 0]] * A[[1, 1]] - A[[0, 1]] * A[[1, 0]]);
        }

        // For larger matrices, use LU decomposition
        let mut L = Array2::zeros((n, n));
        let mut U = Array2::zeros((n, n));

        // Simple LU decomposition
        for i in 0..n {
            // Upper triangular
            for k in i..n {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += L[[i, j]] * U[[j, k]];
                }
                U[[i, k]] = A[[i, k]] - sum;
            }

            // Lower triangular
            for k in i..n {
                if i == k {
                    L[[i, i]] = 1.0;
                } else {
                    let mut sum = 0.0;
                    for j in 0..i {
                        sum += L[[k, j]] * U[[j, i]];
                    }
                    if U[[i, i]].abs() < 1e-10 {
                        return Err(SklearsError::NumericalError(
                            "Matrix is singular".to_string(),
                        ));
                    }
                    L[[k, i]] = (A[[k, i]] - sum) / U[[i, i]];
                }
            }
        }

        // Determinant is product of diagonal elements of U
        let mut det = 1.0;
        for i in 0..n {
            det *= U[[i, i]];
        }

        Ok(det)
    }

    /// Simple matrix inverse using Gauss-Jordan elimination
    fn matrix_inverse(&self, A: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = A.dim().0;
        let mut aug = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = A[[i, j]];
                aug[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..2 * n {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[i, i]].abs() < 1e-10 {
                return Err(SklearsError::NumericalError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = aug[[i, i]];
            for j in 0..2 * n {
                aug[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug[[k, i]];
                    for j in 0..2 * n {
                        aug[[k, j]] -= factor * aug[[i, j]];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inv = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, j + n]];
            }
        }

        Ok(inv)
    }

    /// Simple log-gamma approximation using Stirling's approximation
    fn log_gamma(x: f64) -> f64 {
        if x < 12.0 {
            // Use reflection formula for small x
            if x < 0.5 {
                return (PI / ((PI * x).sin())).ln() - Self::log_gamma(1.0 - x);
            } else {
                return Self::log_gamma(x + 1.0) - x.ln();
            }
        }

        // Stirling's approximation for large x
        0.5 * (2.0 * PI).ln() - 0.5 * x.ln() + x * x.ln() - x
    }
}

impl Default for StudentTMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StudentTMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for StudentTMixture<Untrained> {
    type Fitted = StudentTMixture<StudentTMixtureTrained>;

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

        if self.degrees_of_freedom.len() != self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of degrees of freedom must match number of components".to_string(),
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
                    self.update_parameters(&X, &responsibilities, &means, &covariances)?;

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

        Ok(StudentTMixture {
            state: StudentTMixtureTrained {
                weights,
                means,
                covariances,
                degrees_of_freedom: self.degrees_of_freedom.clone(),
                log_likelihood: best_log_likelihood,
                n_iter: best_n_iter,
                converged: best_converged,
                bic,
                aic,
                n_components: self.n_components,
                covariance_type: self.covariance_type.clone(),
            },
            n_components: self.n_components,
            degrees_of_freedom: self.degrees_of_freedom,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            random_state: self.random_state,
        })
    }
}

/// Trained state for StudentTMixture
#[derive(Debug, Clone)]
pub struct StudentTMixtureTrained {
    /// Mixture component weights
    pub weights: Array1<f64>,
    /// Component means (location parameters)
    pub means: Array2<f64>,
    /// Component covariance matrices (scale parameters)
    pub covariances: Vec<Array2<f64>>,
    /// Degrees of freedom for each component
    pub degrees_of_freedom: Vec<f64>,
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
    /// Number of components
    pub n_components: usize,
    /// Covariance type used
    pub covariance_type: CovarianceType,
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for StudentTMixture<StudentTMixtureTrained> {
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
                let log_weight = self.state.weights[k].ln();
                let log_pdf = self.student_t_log_pdf(
                    &sample,
                    &self.state.means.row(k),
                    &self.state.covariances[k],
                    self.state.degrees_of_freedom[k],
                )?;
                let log_prob = log_weight + log_pdf;

                if log_prob > max_log_prob {
                    max_log_prob = log_prob;
                    best_component = k;
                }
            }

            predictions[i] = best_component as i32;
        }

        Ok(predictions)
    }
}

impl StudentTMixture<StudentTMixtureTrained> {
    /// Predict class probabilities for samples
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut probabilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_probs = Vec::new();

            for k in 0..self.n_components {
                let log_weight = self.state.weights[k].ln();
                let log_pdf = self.student_t_log_pdf(
                    &sample,
                    &self.state.means.row(k),
                    &self.state.covariances[k],
                    self.state.degrees_of_freedom[k],
                )?;
                log_probs.push(log_weight + log_pdf);
            }

            // Numerically stable computation using log-sum-exp
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = max_log_prob
                + log_probs
                    .iter()
                    .map(|&lp| (lp - max_log_prob).exp())
                    .sum::<f64>()
                    .ln();

            for k in 0..self.n_components {
                probabilities[[i, k]] = (log_probs[k] - log_sum_exp).exp();
            }
        }

        Ok(probabilities)
    }

    /// Score samples using the log-likelihood
    #[allow(non_snake_case)]
    pub fn score_samples(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut sample_likelihood = 0.0;

            for k in 0..self.n_components {
                let log_pdf = self.student_t_log_pdf(
                    &sample,
                    &self.state.means.row(k),
                    &self.state.covariances[k],
                    self.state.degrees_of_freedom[k],
                )?;
                sample_likelihood += self.state.weights[k] * log_pdf.exp();
            }

            scores[i] = if sample_likelihood > 0.0 {
                sample_likelihood.ln()
            } else {
                f64::NEG_INFINITY
            };
        }

        Ok(scores)
    }

    /// Compute the log-likelihood score of the samples
    pub fn score(&self, X: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let scores = self.score_samples(X)?;
        Ok(scores.sum())
    }

    /// Get the effective degrees of freedom (accounting for estimation uncertainty)
    pub fn effective_degrees_of_freedom(&self) -> Vec<f64> {
        // Simple heuristic: reduce by number of estimated parameters per component
        let n_features = self.state.means.dim().1;
        let param_penalty = match self.state.covariance_type {
            CovarianceType::Full => n_features * (n_features + 1) / 2 + n_features,
            CovarianceType::Diagonal => 2 * n_features,
            CovarianceType::Tied => n_features * (n_features + 1) / 2 + n_features,
            CovarianceType::Spherical => n_features + 1,
        };

        self.state
            .degrees_of_freedom
            .iter()
            .map(|&df| (df - param_penalty as f64 * 0.1).max(2.1)) // Keep above 2.0
            .collect()
    }

    /// Student-t log PDF implementation (shared between trained and untrained)
    fn student_t_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
        df: f64,
    ) -> SklResult<f64> {
        let d = x.len() as f64;

        // Simple approach: use determinant and inverse directly
        let det = self.matrix_determinant(cov)?;
        if det <= 0.0 {
            return Err(SklearsError::NumericalError(
                "Covariance matrix is not positive definite".to_string(),
            ));
        }

        let log_det = det.ln();
        let inv_cov = self.matrix_inverse(cov)?;

        // Compute Mahalanobis distance
        let diff = x - mean;
        let mut mahal_dist = 0.0;
        for i in 0..diff.len() {
            for j in 0..diff.len() {
                mahal_dist += diff[i] * inv_cov[[i, j]] * diff[j];
            }
        }

        // Student-t log PDF
        let gamma_term = Self::log_gamma((df + d) / 2.0) - Self::log_gamma(df / 2.0);
        let const_term = gamma_term - 0.5 * d * (df * PI).ln() - 0.5 * log_det;
        let density_term = -0.5 * (df + d) * (1.0 + mahal_dist / df).ln();

        Ok(const_term + density_term)
    }

    /// Simple matrix determinant for small matrices
    fn matrix_determinant(&self, A: &Array2<f64>) -> SklResult<f64> {
        let n = A.dim().0;
        if n == 1 {
            return Ok(A[[0, 0]]);
        }
        if n == 2 {
            return Ok(A[[0, 0]] * A[[1, 1]] - A[[0, 1]] * A[[1, 0]]);
        }

        // For larger matrices, use LU decomposition
        let mut L = Array2::zeros((n, n));
        let mut U = Array2::zeros((n, n));

        // Simple LU decomposition
        for i in 0..n {
            // Upper triangular
            for k in i..n {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += L[[i, j]] * U[[j, k]];
                }
                U[[i, k]] = A[[i, k]] - sum;
            }

            // Lower triangular
            for k in i..n {
                if i == k {
                    L[[i, i]] = 1.0;
                } else {
                    let mut sum = 0.0;
                    for j in 0..i {
                        sum += L[[k, j]] * U[[j, i]];
                    }
                    if U[[i, i]].abs() < 1e-10 {
                        return Err(SklearsError::NumericalError(
                            "Matrix is singular".to_string(),
                        ));
                    }
                    L[[k, i]] = (A[[k, i]] - sum) / U[[i, i]];
                }
            }
        }

        // Determinant is product of diagonal elements of U
        let mut det = 1.0;
        for i in 0..n {
            det *= U[[i, i]];
        }

        Ok(det)
    }

    /// Simple matrix inverse using Gauss-Jordan elimination
    fn matrix_inverse(&self, A: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = A.dim().0;
        let mut aug = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = A[[i, j]];
                aug[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..2 * n {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[i, i]].abs() < 1e-10 {
                return Err(SklearsError::NumericalError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = aug[[i, i]];
            for j in 0..2 * n {
                aug[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug[[k, i]];
                    for j in 0..2 * n {
                        aug[[k, j]] -= factor * aug[[i, j]];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inv = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, j + n]];
            }
        }

        Ok(inv)
    }

    /// Simple log-gamma approximation using Stirling's approximation
    fn log_gamma(x: f64) -> f64 {
        if x < 12.0 {
            // Use reflection formula for small x
            if x < 0.5 {
                return (PI / ((PI * x).sin())).ln() - Self::log_gamma(1.0 - x);
            } else {
                return Self::log_gamma(x + 1.0) - x.ln();
            }
        }

        // Stirling's approximation for large x
        0.5 * (2.0 * PI).ln() - 0.5 * x.ln() + x * x.ln() - x
    }
}
