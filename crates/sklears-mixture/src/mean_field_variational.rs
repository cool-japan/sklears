//! Mean-Field Variational Inference for Mixture Models
//!
//! This module implements mean-field variational inference for Gaussian mixture models
//! with explicit factorization assumptions. Unlike the general variational Bayesian GMM,
//! this implementation explicitly leverages the mean-field approximation where the
//! posterior distribution is assumed to factorize as a product of independent distributions.

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Mean-Field Variational Inference for Gaussian Mixture Models
///
/// This implementation uses explicit mean-field approximation where the posterior
/// distribution q(θ) factorizes as:
/// q(θ) = q(z)q(π)q(μ)q(Λ)
/// where z are latent assignments, π are mixture weights, μ are means, and Λ are precisions.
///
/// # Examples
///
/// ```
/// use sklears_mixture::{MeanFieldVariationalGMM, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let model = MeanFieldVariationalGMM::new()
///     .n_components(3)
///     .covariance_type(CovarianceType::Diagonal)
///     .max_iter(100);
/// let fitted = model.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MeanFieldVariationalGMM<S = Untrained> {
    state: S,
    n_components: usize,
    covariance_type: CovarianceType,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    random_state: Option<u64>,
    // Mean-field specific parameters
    alpha_prior: f64, // Prior for mixing weights (Dirichlet)
    beta_prior: f64,  // Prior precision for means
    nu_prior: f64,    // Prior degrees of freedom for precision
    w_prior: f64,     // Prior scale for precision (Wishart)
    // Variational parameters update rates
    learning_rate: f64,
    momentum: f64,
}

/// Trained state for MeanFieldVariationalGMM
#[derive(Debug, Clone)]
pub struct MeanFieldVariationalGMMTrained {
    // Posterior parameters for mean-field approximation
    q_z: Array2<f64>,             // Variational posterior for latent assignments
    q_pi_alpha: Array1<f64>,      // Dirichlet parameters for mixing weights
    q_mu_mean: Array2<f64>,       // Mean parameters for component means
    q_mu_precision: Array2<f64>,  // Precision parameters for component means
    q_lambda_nu: Array1<f64>,     // Degrees of freedom for precision matrices
    q_lambda_w: Vec<Array2<f64>>, // Scale matrices for precision matrices

    // Derived quantities
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    lower_bound: f64,
    n_iter: usize,
    converged: bool,
    effective_components: usize,
}

impl MeanFieldVariationalGMM<Untrained> {
    /// Create a new Mean-Field Variational GMM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            covariance_type: CovarianceType::Full,
            tol: 1e-4,
            reg_covar: 1e-6,
            max_iter: 100,
            random_state: None,
            alpha_prior: 1.0,
            beta_prior: 1.0,
            nu_prior: 1.0,
            w_prior: 1.0,
            learning_rate: 0.1,
            momentum: 0.9,
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

    /// Set the learning rate for variational updates
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the momentum for variational updates
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Build the model (builder pattern completion)
    pub fn build(self) -> Self {
        self
    }
}

impl Default for MeanFieldVariationalGMM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MeanFieldVariationalGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MeanFieldVariationalGMM<Untrained> {
    type Fitted = MeanFieldVariationalGMM<MeanFieldVariationalGMMTrained>;

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

        // Initialize variational parameters
        let (
            mut q_z,
            mut q_pi_alpha,
            mut q_mu_mean,
            mut q_mu_precision,
            mut q_lambda_nu,
            mut q_lambda_w,
        ) = self.initialize_variational_parameters(&X)?;

        let mut lower_bound = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        // Momentum terms for natural gradient descent
        let mut momentum_z = Array2::zeros((n_samples, self.n_components));
        let mut momentum_pi = Array1::zeros(self.n_components);
        let mut momentum_mu_mean = Array2::zeros((self.n_components, n_features));
        let mut momentum_mu_precision = Array2::zeros((self.n_components, n_features));

        // Mean-field variational inference iterations
        for iteration in 0..self.max_iter {
            n_iter = iteration + 1;

            // Update q(z) - latent assignments
            let (new_q_z, grad_z) = self.update_q_z(
                &X,
                &q_pi_alpha,
                &q_mu_mean,
                &q_mu_precision,
                &q_lambda_nu,
                &q_lambda_w,
            )?;

            // Apply momentum to q(z) updates
            momentum_z = self.momentum * &momentum_z + (1.0 - self.momentum) * &grad_z;
            q_z = &q_z - self.learning_rate * &momentum_z;

            // Normalize q(z)
            for i in 0..n_samples {
                let sum = q_z.row(i).sum();
                if sum > 0.0 {
                    q_z.row_mut(i).mapv_inplace(|x| x / sum);
                }
            }

            // Update q(π) - mixing weights
            let (new_q_pi_alpha, grad_pi) = self.update_q_pi(&q_z)?;
            momentum_pi = self.momentum * &momentum_pi + (1.0 - self.momentum) * &grad_pi;
            q_pi_alpha = &q_pi_alpha - self.learning_rate * &momentum_pi;

            // Ensure positivity
            q_pi_alpha.mapv_inplace(|x| x.max(1e-10));

            // Update q(μ) - component means
            let (new_q_mu_mean, new_q_mu_precision, grad_mu_mean, grad_mu_precision) =
                self.update_q_mu(&X, &q_z, &q_lambda_nu, &q_lambda_w)?;

            momentum_mu_mean =
                self.momentum * &momentum_mu_mean + (1.0 - self.momentum) * &grad_mu_mean;
            momentum_mu_precision =
                self.momentum * &momentum_mu_precision + (1.0 - self.momentum) * &grad_mu_precision;

            q_mu_mean = &q_mu_mean - self.learning_rate * &momentum_mu_mean;
            q_mu_precision = &q_mu_precision - self.learning_rate * &momentum_mu_precision;

            // Ensure positivity for precision
            q_mu_precision.mapv_inplace(|x| x.max(1e-10));

            // Update q(Λ) - precision matrices
            let (new_q_lambda_nu, new_q_lambda_w) =
                self.update_q_lambda(&X, &q_z, &q_mu_mean, &q_mu_precision)?;
            q_lambda_nu = new_q_lambda_nu;
            q_lambda_w = new_q_lambda_w;

            // Compute variational lower bound
            let new_lower_bound = self.compute_variational_lower_bound(
                &X,
                &q_z,
                &q_pi_alpha,
                &q_mu_mean,
                &q_mu_precision,
                &q_lambda_nu,
                &q_lambda_w,
            )?;

            // Check convergence
            if iteration > 0 && (new_lower_bound - lower_bound).abs() < self.tol {
                converged = true;
            }

            lower_bound = new_lower_bound;

            if converged {
                break;
            }
        }

        // Compute derived quantities
        let weights = self.compute_weights(&q_pi_alpha);
        let means = q_mu_mean.clone();
        let covariances = self.compute_covariances(&q_lambda_nu, &q_lambda_w)?;

        // Count effective components
        let effective_components = weights.iter().filter(|&&w| w > 1e-3).count();

        Ok(MeanFieldVariationalGMM {
            state: MeanFieldVariationalGMMTrained {
                q_z,
                q_pi_alpha,
                q_mu_mean,
                q_mu_precision,
                q_lambda_nu,
                q_lambda_w,
                weights,
                means,
                covariances,
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
            alpha_prior: self.alpha_prior,
            beta_prior: self.beta_prior,
            nu_prior: self.nu_prior,
            w_prior: self.w_prior,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
        })
    }
}

impl MeanFieldVariationalGMM<Untrained> {
    /// Initialize variational parameters for mean-field approximation
    fn initialize_variational_parameters(
        &self,
        X: &Array2<f64>,
    ) -> SklResult<(
        Array2<f64>,      // q_z
        Array1<f64>,      // q_pi_alpha
        Array2<f64>,      // q_mu_mean
        Array2<f64>,      // q_mu_precision
        Array1<f64>,      // q_lambda_nu
        Vec<Array2<f64>>, // q_lambda_w
    )> {
        let (n_samples, n_features) = X.dim();
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut thread_rng())
        };

        // Initialize q(z) - latent assignments (responsibilities)
        let mut q_z = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for k in 0..self.n_components {
                q_z[[i, k]] = rng.gen::<f64>();
            }
            // Normalize
            let sum = q_z.row(i).sum();
            q_z.row_mut(i).mapv_inplace(|x| x / sum);
        }

        // Initialize q(π) - mixing weights
        let q_pi_alpha = Array1::from_elem(self.n_components, self.alpha_prior);

        // Initialize q(μ) - component means
        let q_mu_mean = Array2::zeros((self.n_components, n_features));
        let q_mu_precision = Array2::from_elem((self.n_components, n_features), self.beta_prior);

        // Initialize q(Λ) - precision matrices
        let q_lambda_nu = Array1::from_elem(self.n_components, self.nu_prior + n_features as f64);
        let mut q_lambda_w = Vec::new();
        for _ in 0..self.n_components {
            let mut w = Array2::eye(n_features);
            w.mapv_inplace(|x| x * self.w_prior);
            q_lambda_w.push(w);
        }

        Ok((
            q_z,
            q_pi_alpha,
            q_mu_mean,
            q_mu_precision,
            q_lambda_nu,
            q_lambda_w,
        ))
    }

    /// Update q(z) - latent assignments using mean-field approximation
    fn update_q_z(
        &self,
        X: &Array2<f64>,
        q_pi_alpha: &Array1<f64>,
        q_mu_mean: &Array2<f64>,
        q_mu_precision: &Array2<f64>,
        q_lambda_nu: &Array1<f64>,
        q_lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<(Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = X.dim();
        let mut q_z = Array2::zeros((n_samples, self.n_components));
        let mut gradients = Array2::zeros((n_samples, self.n_components));

        // Compute digamma of sum for mixing weights
        let digamma_sum = digamma(q_pi_alpha.sum());

        for i in 0..n_samples {
            for k in 0..self.n_components {
                let mut log_prob = 0.0;

                // E[log π_k] term
                log_prob += digamma(q_pi_alpha[k]) - digamma_sum;

                // E[log p(x_i | μ_k, Λ_k)] term
                let x_i = X.row(i);
                let mu_k = q_mu_mean.row(k);

                // Compute expected log-likelihood
                let diff = &x_i.to_owned() - &mu_k.to_owned();
                let precision_term = q_lambda_nu[k] * diff.dot(&diff);

                log_prob += 0.5
                    * (self.compute_expected_log_det_lambda(k, q_lambda_nu, q_lambda_w)
                        - n_features as f64 * (1.0 / q_mu_precision[[k, 0]])
                        - precision_term
                        - n_features as f64 * (2.0 * PI).ln());

                q_z[[i, k]] = log_prob;
            }

            // Normalize using log-sum-exp for numerical stability
            let max_log_prob = q_z.row(i).fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let mut sum_exp = 0.0;
            for k in 0..self.n_components {
                q_z[[i, k]] = (q_z[[i, k]] - max_log_prob).exp();
                sum_exp += q_z[[i, k]];
            }

            for k in 0..self.n_components {
                q_z[[i, k]] /= sum_exp;
                gradients[[i, k]] = q_z[[i, k]] - 1.0 / self.n_components as f64;
                // Gradient w.r.t. natural parameters
            }
        }

        Ok((q_z, gradients))
    }

    /// Update q(π) - mixing weights
    fn update_q_pi(&self, q_z: &Array2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let n_samples = q_z.nrows();
        let mut q_pi_alpha = Array1::from_elem(self.n_components, self.alpha_prior);
        let mut gradients = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let n_k = q_z.column(k).sum();
            q_pi_alpha[k] = self.alpha_prior + n_k;
            gradients[k] = n_k - self.alpha_prior; // Natural gradient
        }

        Ok((q_pi_alpha, gradients))
    }

    /// Update q(μ) - component means
    fn update_q_mu(
        &self,
        X: &Array2<f64>,
        q_z: &Array2<f64>,
        q_lambda_nu: &Array1<f64>,
        q_lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = X.dim();
        let mut q_mu_mean = Array2::zeros((self.n_components, n_features));
        let mut q_mu_precision =
            Array2::from_elem((self.n_components, n_features), self.beta_prior);
        let mut grad_mean = Array2::zeros((self.n_components, n_features));
        let mut grad_precision = Array2::zeros((self.n_components, n_features));

        for k in 0..self.n_components {
            let n_k = q_z.column(k).sum();

            // Compute weighted sample mean
            let mut x_bar_k = Array1::zeros(n_features);
            for i in 0..n_samples {
                let x_i = X.row(i);
                x_bar_k = x_bar_k + q_z[[i, k]] * &x_i.to_owned();
            }
            if n_k > 0.0 {
                x_bar_k /= n_k;
            }

            // Update precision
            let new_precision = self.beta_prior + n_k * q_lambda_nu[k];

            // Update mean
            let new_mean =
                (self.beta_prior * 0.0 + n_k * q_lambda_nu[k] * &x_bar_k) / new_precision;

            for d in 0..n_features {
                q_mu_mean[[k, d]] = new_mean[d];
                q_mu_precision[[k, d]] = new_precision;

                // Compute gradients
                grad_mean[[k, d]] = new_mean[d] - q_mu_mean[[k, d]];
                grad_precision[[k, d]] = new_precision - q_mu_precision[[k, d]];
            }
        }

        Ok((q_mu_mean, q_mu_precision, grad_mean, grad_precision))
    }

    /// Update q(Λ) - precision matrices
    fn update_q_lambda(
        &self,
        X: &Array2<f64>,
        q_z: &Array2<f64>,
        q_mu_mean: &Array2<f64>,
        q_mu_precision: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Vec<Array2<f64>>)> {
        let (n_samples, n_features) = X.dim();
        let mut q_lambda_nu = Array1::zeros(self.n_components);
        let mut q_lambda_w = Vec::new();

        for k in 0..self.n_components {
            let n_k = q_z.column(k).sum();

            // Update degrees of freedom
            q_lambda_nu[k] = self.nu_prior + n_k;

            // Compute scatter matrix
            let mut s_k: Array2<Float> = Array2::zeros((n_features, n_features));
            for i in 0..n_samples {
                let x_i = X.row(i);
                let mu_k = q_mu_mean.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();
                let outer =
                    &diff.to_owned().insert_axis(Axis(1)) * &diff.to_owned().insert_axis(Axis(0));
                s_k = s_k + q_z[[i, k]] * &outer;
            }

            // Add regularization and prior
            let mut w_k = Array2::eye(n_features) * self.w_prior;
            w_k = w_k + s_k;

            // Add uncertainty in mean estimates
            for d in 0..n_features {
                w_k[[d, d]] += 1.0 / q_mu_precision[[k, d]];
            }

            q_lambda_w.push(w_k);
        }

        Ok((q_lambda_nu, q_lambda_w))
    }

    /// Compute the variational lower bound (ELBO)
    fn compute_variational_lower_bound(
        &self,
        X: &Array2<f64>,
        q_z: &Array2<f64>,
        q_pi_alpha: &Array1<f64>,
        q_mu_mean: &Array2<f64>,
        q_mu_precision: &Array2<f64>,
        q_lambda_nu: &Array1<f64>,
        q_lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = X.dim();
        let mut elbo = 0.0;

        // E[log p(X | Z, θ)] - expected log-likelihood
        for i in 0..n_samples {
            for k in 0..self.n_components {
                let x_i = X.row(i);
                let mu_k = q_mu_mean.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();

                let log_likelihood = 0.5
                    * (self.compute_expected_log_det_lambda(k, q_lambda_nu, q_lambda_w)
                        - n_features as f64 * (2.0 * PI).ln()
                        - q_lambda_nu[k] * diff.dot(&diff)
                        - n_features as f64 / q_mu_precision[[k, 0]]);

                elbo += q_z[[i, k]] * log_likelihood;
            }
        }

        // E[log p(Z | π)] - expected log-prior for assignments
        let digamma_sum = digamma(q_pi_alpha.sum());
        for i in 0..n_samples {
            for k in 0..self.n_components {
                let expected_log_pi = digamma(q_pi_alpha[k]) - digamma_sum;
                elbo += q_z[[i, k]] * expected_log_pi;
            }
        }

        // E[log p(π)] - expected log-prior for mixing weights
        let alpha_sum = self.alpha_prior * self.n_components as f64;
        elbo += log_gamma(alpha_sum) - self.n_components as f64 * log_gamma(self.alpha_prior);
        for k in 0..self.n_components {
            elbo += (self.alpha_prior - 1.0) * (digamma(q_pi_alpha[k]) - digamma_sum);
        }

        // E[log p(μ)] - expected log-prior for means
        for k in 0..self.n_components {
            for d in 0..n_features {
                elbo += -0.5
                    * ((2.0 * PI).ln() - q_lambda_nu[k].ln()
                        + self.beta_prior * q_mu_precision[[k, d]] * q_mu_mean[[k, d]].powi(2));
            }
        }

        // E[log p(Λ)] - expected log-prior for precision matrices
        for k in 0..self.n_components {
            // Wishart prior contribution
            elbo += 0.5
                * (self.nu_prior - n_features as f64 - 1.0)
                * self.compute_expected_log_det_lambda(k, q_lambda_nu, q_lambda_w);
        }

        // Entropy terms H[q(Z)]
        for i in 0..n_samples {
            for k in 0..self.n_components {
                if q_z[[i, k]] > 1e-10 {
                    elbo -= q_z[[i, k]] * q_z[[i, k]].ln();
                }
            }
        }

        // Entropy terms H[q(π)]
        elbo -= log_gamma(q_pi_alpha.sum()) - q_pi_alpha.iter().map(|&x| log_gamma(x)).sum::<f64>();
        for k in 0..self.n_components {
            elbo -= (q_pi_alpha[k] - 1.0) * (digamma(q_pi_alpha[k]) - digamma_sum);
        }

        // Entropy terms H[q(μ)] and H[q(Λ)]
        for k in 0..self.n_components {
            for d in 0..n_features {
                elbo -= -0.5 * (1.0 + q_mu_precision[[k, d]].ln());
            }

            // Wishart entropy
            elbo -= 0.5 * q_lambda_nu[k] * n_features as f64 * (2.0 * PI).ln();
        }

        Ok(elbo)
    }

    /// Compute expected log determinant of precision matrix
    fn compute_expected_log_det_lambda(
        &self,
        k: usize,
        q_lambda_nu: &Array1<f64>,
        q_lambda_w: &Vec<Array2<f64>>,
    ) -> f64 {
        let n_features = q_lambda_w[k].nrows();
        let mut log_det = 0.0;

        for i in 0..n_features {
            log_det += digamma(0.5 * (q_lambda_nu[k] + 1.0 - i as f64));
        }

        log_det += n_features as f64 * (2.0_f64).ln();
        log_det -= q_lambda_w[k].diag().iter().map(|&x| x.ln()).sum::<f64>();

        log_det
    }

    /// Compute mixing weights from Dirichlet parameters
    fn compute_weights(&self, q_pi_alpha: &Array1<f64>) -> Array1<f64> {
        let alpha_sum = q_pi_alpha.sum();
        q_pi_alpha.mapv(|x| x / alpha_sum)
    }

    /// Compute covariance matrices from precision parameters
    fn compute_covariances(
        &self,
        q_lambda_nu: &Array1<f64>,
        q_lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let mut covariances = Vec::new();

        for k in 0..self.n_components {
            let precision = &q_lambda_w[k] * q_lambda_nu[k];

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

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for MeanFieldVariationalGMM<MeanFieldVariationalGMMTrained>
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

impl MeanFieldVariationalGMM<MeanFieldVariationalGMMTrained> {
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

    /// Get the variational lower bound
    pub fn lower_bound(&self) -> f64 {
        self.state.lower_bound
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get the number of effective components
    pub fn effective_components(&self) -> usize {
        self.state.effective_components
    }

    /// Get the posterior assignments q(z)
    pub fn posterior_assignments(&self) -> &Array2<f64> {
        &self.state.q_z
    }

    /// Get the posterior mixing weight parameters
    pub fn posterior_mixing_weights(&self) -> &Array1<f64> {
        &self.state.q_pi_alpha
    }

    /// Get the posterior mean parameters
    pub fn posterior_means(&self) -> &Array2<f64> {
        &self.state.q_mu_mean
    }

    /// Get the posterior precision parameters
    pub fn posterior_precisions(&self) -> &Array2<f64> {
        &self.state.q_mu_precision
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
            let mut log_likelihoods = Vec::new();

            for k in 0..self.n_components {
                let x_i = X.row(i);
                let mu_k = self.state.means.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();

                // Use numerically stable computation
                let log_weight = self.state.weights[k].ln();
                let log_gaussian =
                    -0.5 * diff.dot(&diff) / self.state.covariances[k][[0, 0]].max(1e-10);

                log_likelihoods.push(log_weight + log_gaussian);
            }

            // Use log-sum-exp trick for numerical stability
            let max_log_likelihood = log_likelihoods
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if max_log_likelihood.is_finite() {
                let sum_exp = log_likelihoods
                    .iter()
                    .map(|&x| (x - max_log_likelihood).exp())
                    .sum::<f64>();
                log_likelihood += max_log_likelihood + sum_exp.ln();
            } else {
                // Handle edge case where all log likelihoods are -∞
                log_likelihood += f64::NEG_INFINITY;
            }
        }

        Ok(log_likelihood)
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

fn log_gamma(x: f64) -> f64 {
    // Approximation of log gamma function
    if x > 0.0 {
        (x - 1.0) * x.ln() - x + 0.5 * (2.0 * PI / x).ln()
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
    fn test_mean_field_variational_gmm_creation() {
        let model = MeanFieldVariationalGMM::new()
            .n_components(3)
            .covariance_type(CovarianceType::Diagonal)
            .max_iter(50)
            .tol(1e-3)
            .alpha_prior(1.0)
            .beta_prior(1.0)
            .learning_rate(0.1)
            .momentum(0.9);

        assert_eq!(model.n_components, 3);
        assert_eq!(model.max_iter, 50);
        assert_eq!(model.tol, 1e-3);
        assert_eq!(model.alpha_prior, 1.0);
        assert_eq!(model.beta_prior, 1.0);
        assert_eq!(model.learning_rate, 0.1);
        assert_eq!(model.momentum, 0.9);
    }

    #[test]
    fn test_mean_field_variational_gmm_builder() {
        let model = MeanFieldVariationalGMM::builder()
            .n_components(2)
            .covariance_type(CovarianceType::Full)
            .build();

        assert_eq!(model.n_components, 2);
        assert!(matches!(model.covariance_type, CovarianceType::Full));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_fit_simple() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .max_iter(10)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 2);
        assert!(fitted.converged() || fitted.n_iter() == 10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_predict() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .max_iter(10)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let labels = fitted.predict(&X.view()).unwrap();

        assert_eq!(labels.len(), 4);
        // Check that labels are in valid range
        for &label in labels.iter() {
            assert!(label >= 0 && label < 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_predict_proba() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .max_iter(10)
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
    fn test_mean_field_variational_gmm_score() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .max_iter(10)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let score = fitted.score(&X.view()).unwrap();

        // Score should be finite
        assert!(score.is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_properties() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .max_iter(10)
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
        assert!(fitted.lower_bound().is_finite());
        assert!(fitted.n_iter() > 0);
        assert!(fitted.effective_components() > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_posterior_access() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .max_iter(10)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();

        // Check posterior assignments
        let q_z = fitted.posterior_assignments();
        assert_eq!(q_z.dim(), (4, 2));

        // Check that assignments sum to 1 for each sample
        for i in 0..4 {
            let sum: f64 = q_z.row(i).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }

        // Check posterior mixing weights
        let q_pi = fitted.posterior_mixing_weights();
        assert_eq!(q_pi.len(), 2);

        // Check posterior means
        let q_mu = fitted.posterior_means();
        assert_eq!(q_mu.dim(), (2, 2));

        // Check posterior precisions
        let q_precision = fitted.posterior_precisions();
        assert_eq!(q_precision.dim(), (2, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_different_covariance_types() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let covariance_types = vec![
            CovarianceType::Full,
            CovarianceType::Diagonal,
            CovarianceType::Tied,
            CovarianceType::Spherical,
        ];

        for cov_type in covariance_types {
            let model = MeanFieldVariationalGMM::new()
                .n_components(2)
                .covariance_type(cov_type)
                .max_iter(10)
                .random_state(42);

            let fitted = model.fit(&X.view(), &()).unwrap();
            assert_eq!(fitted.n_components, 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_hyperparameters() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .alpha_prior(2.0)
            .beta_prior(0.5)
            .nu_prior(3.0)
            .w_prior(2.0)
            .max_iter(10)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 2);
        assert!(fitted.lower_bound().is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mean_field_variational_gmm_optimization_params() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = MeanFieldVariationalGMM::new()
            .n_components(2)
            .learning_rate(0.01)
            .momentum(0.95)
            .max_iter(10)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 2);
        assert!(fitted.lower_bound().is_finite());
    }
}
