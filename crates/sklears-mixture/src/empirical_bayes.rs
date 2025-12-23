//! Empirical Bayes Methods for Mixture Models
//!
//! This module implements empirical Bayes estimation for Gaussian mixture models,
//! where hyperparameters of the prior distributions are estimated from the data
//! using maximum marginal likelihood (Type-II ML) or expectation-maximization.

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Empirical Bayes Gaussian Mixture Model
///
/// This implementation estimates the hyperparameters of the prior distributions
/// using empirical Bayes methods. Instead of fixing the priors, this approach
/// learns them from the data using maximum marginal likelihood estimation.
///
/// Key features:
/// - Automatic hyperparameter estimation
/// - Type-II maximum likelihood
/// - Hierarchical model structure
/// - Evidence optimization
/// - Multiple estimation strategies
///
/// # Examples
///
/// ```
/// use sklears_mixture::{EmpiricalBayesGMM, CovarianceType, EmpiricalBayesMethod};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let model = EmpiricalBayesGMM::new()
///     .n_components(3)
///     .method(EmpiricalBayesMethod::TypeIIML)
///     .covariance_type(CovarianceType::Diagonal)
///     .max_iter(100);
/// let fitted = model.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct EmpiricalBayesGMM<S = Untrained> {
    state: S,
    n_components: usize,
    covariance_type: CovarianceType,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    random_state: Option<u64>,

    // Empirical Bayes parameters
    method: EmpiricalBayesMethod,
    max_hyperparameter_iter: usize,
    hyperparameter_tol: f64,

    // Initial hyperparameter values (will be optimized)
    initial_alpha_concentration: f64,
    initial_mean_precision: f64,
    initial_degrees_of_freedom: f64,
    initial_scale_matrix: f64,

    // Hyperparameter bounds
    alpha_concentration_bounds: (f64, f64),
    mean_precision_bounds: (f64, f64),
    degrees_of_freedom_bounds: (f64, f64),
    scale_matrix_bounds: (f64, f64),

    // Optimization parameters
    learning_rate_hyperparams: f64,
    momentum_hyperparams: f64,
}

/// Empirical Bayes estimation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmpiricalBayesMethod {
    /// Type-II Maximum Likelihood (evidence maximization)
    TypeIIML,
    /// Expectation-Maximization for hyperparameters
    EM,
    /// Cross-validation based selection
    CrossValidation,
    /// Marginal likelihood maximization with gradient ascent
    GradientAscent,
}

/// Trained state for EmpiricalBayesGMM
#[derive(Debug, Clone)]
pub struct EmpiricalBayesGMMTrained {
    // Model parameters
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,

    // Estimated hyperparameters
    estimated_alpha_concentration: f64,
    estimated_mean_precision: f64,
    estimated_degrees_of_freedom: f64,
    estimated_scale_matrix: f64,

    // Variational parameters (if applicable)
    variational_pi_alpha: Array1<f64>,
    variational_mu_mean: Array2<f64>,
    variational_mu_precision: Array2<f64>,
    variational_lambda_nu: Array1<f64>,
    variational_lambda_w: Vec<Array2<f64>>,

    // Training information
    marginal_log_likelihood: f64,
    evidence_history: Vec<f64>,
    hyperparameter_history: Vec<HyperparameterState>,
    n_iter: usize,
    converged: bool,
    effective_components: usize,
}

/// Hyperparameter state for tracking optimization
#[derive(Debug, Clone)]
pub struct HyperparameterState {
    alpha_concentration: f64,
    mean_precision: f64,
    degrees_of_freedom: f64,
    scale_matrix: f64,
    marginal_likelihood: f64,
}

impl EmpiricalBayesGMM<Untrained> {
    /// Create a new Empirical Bayes GMM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            covariance_type: CovarianceType::Diagonal,
            tol: 1e-4,
            reg_covar: 1e-6,
            max_iter: 100,
            random_state: None,

            method: EmpiricalBayesMethod::TypeIIML,
            max_hyperparameter_iter: 50,
            hyperparameter_tol: 1e-3,

            initial_alpha_concentration: 1.0,
            initial_mean_precision: 1.0,
            initial_degrees_of_freedom: 1.0,
            initial_scale_matrix: 1.0,

            alpha_concentration_bounds: (0.1, 10.0),
            mean_precision_bounds: (0.01, 100.0),
            degrees_of_freedom_bounds: (0.1, 50.0),
            scale_matrix_bounds: (0.01, 100.0),

            learning_rate_hyperparams: 0.01,
            momentum_hyperparams: 0.9,
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

    /// Set the empirical Bayes method
    pub fn method(mut self, method: EmpiricalBayesMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the maximum number of hyperparameter iterations
    pub fn max_hyperparameter_iter(mut self, max_iter: usize) -> Self {
        self.max_hyperparameter_iter = max_iter;
        self
    }

    /// Set the hyperparameter convergence tolerance
    pub fn hyperparameter_tol(mut self, tol: f64) -> Self {
        self.hyperparameter_tol = tol;
        self
    }

    /// Set initial alpha concentration
    pub fn initial_alpha_concentration(mut self, alpha: f64) -> Self {
        self.initial_alpha_concentration = alpha;
        self
    }

    /// Set initial mean precision
    pub fn initial_mean_precision(mut self, precision: f64) -> Self {
        self.initial_mean_precision = precision;
        self
    }

    /// Set initial degrees of freedom
    pub fn initial_degrees_of_freedom(mut self, dof: f64) -> Self {
        self.initial_degrees_of_freedom = dof;
        self
    }

    /// Set initial scale matrix parameter
    pub fn initial_scale_matrix(mut self, scale: f64) -> Self {
        self.initial_scale_matrix = scale;
        self
    }

    /// Set bounds for alpha concentration
    pub fn alpha_concentration_bounds(mut self, bounds: (f64, f64)) -> Self {
        self.alpha_concentration_bounds = bounds;
        self
    }

    /// Set bounds for mean precision
    pub fn mean_precision_bounds(mut self, bounds: (f64, f64)) -> Self {
        self.mean_precision_bounds = bounds;
        self
    }

    /// Set bounds for degrees of freedom
    pub fn degrees_of_freedom_bounds(mut self, bounds: (f64, f64)) -> Self {
        self.degrees_of_freedom_bounds = bounds;
        self
    }

    /// Set bounds for scale matrix
    pub fn scale_matrix_bounds(mut self, bounds: (f64, f64)) -> Self {
        self.scale_matrix_bounds = bounds;
        self
    }

    /// Set learning rate for hyperparameter optimization
    pub fn learning_rate_hyperparams(mut self, lr: f64) -> Self {
        self.learning_rate_hyperparams = lr;
        self
    }

    /// Set momentum for hyperparameter optimization
    pub fn momentum_hyperparams(mut self, momentum: f64) -> Self {
        self.momentum_hyperparams = momentum;
        self
    }

    /// Build the model (builder pattern completion)
    pub fn build(self) -> Self {
        self
    }
}

impl Default for EmpiricalBayesGMM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for EmpiricalBayesGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for EmpiricalBayesGMM<Untrained> {
    type Fitted = EmpiricalBayesGMM<EmpiricalBayesGMMTrained>;

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

        // Initialize hyperparameters
        let mut current_hyperparams = HyperparameterState {
            alpha_concentration: self.initial_alpha_concentration,
            mean_precision: self.initial_mean_precision,
            degrees_of_freedom: self.initial_degrees_of_freedom,
            scale_matrix: self.initial_scale_matrix,
            marginal_likelihood: f64::NEG_INFINITY,
        };

        let mut evidence_history = Vec::new();
        let mut hyperparameter_history = Vec::new();
        let mut converged = false;
        let mut n_iter = 0;

        // Initialize momentum terms for gradient-based methods
        let mut momentum_alpha = 0.0;
        let mut momentum_mean_precision = 0.0;
        let mut momentum_dof = 0.0;
        let mut momentum_scale = 0.0;

        // Empirical Bayes optimization loop
        for iteration in 0..self.max_hyperparameter_iter {
            n_iter = iteration + 1;

            // Estimate model parameters given current hyperparameters
            let (_weights, _means, _covariances, variational_params, marginal_likelihood) =
                self.estimate_model_parameters(&X, &current_hyperparams)?;

            current_hyperparams.marginal_likelihood = marginal_likelihood;
            evidence_history.push(marginal_likelihood);
            hyperparameter_history.push(current_hyperparams.clone());

            // Update hyperparameters based on the chosen method
            let new_hyperparams = match self.method {
                EmpiricalBayesMethod::TypeIIML => {
                    self.type_ii_ml_update(&X, &variational_params, &current_hyperparams)?
                }
                EmpiricalBayesMethod::EM => {
                    self.em_hyperparameter_update(&X, &variational_params, &current_hyperparams)?
                }
                EmpiricalBayesMethod::CrossValidation => {
                    self.cross_validation_update(&X, &current_hyperparams)?
                }
                EmpiricalBayesMethod::GradientAscent => self.gradient_ascent_update(
                    &X,
                    &variational_params,
                    &current_hyperparams,
                    &mut momentum_alpha,
                    &mut momentum_mean_precision,
                    &mut momentum_dof,
                    &mut momentum_scale,
                )?,
            };

            // Check convergence
            if iteration > 0 {
                let hyperparam_change =
                    self.compute_hyperparameter_change(&current_hyperparams, &new_hyperparams);
                if hyperparam_change < self.hyperparameter_tol {
                    converged = true;
                }
            }

            current_hyperparams = new_hyperparams;

            if converged {
                break;
            }
        }

        // Final model estimation with optimal hyperparameters
        let (
            final_weights,
            final_means,
            final_covariances,
            final_variational_params,
            final_marginal_likelihood,
        ) = self.estimate_model_parameters(&X, &current_hyperparams)?;

        // Count effective components
        let effective_components = final_weights.iter().filter(|&&w| w > 1e-3).count();

        Ok(EmpiricalBayesGMM {
            state: EmpiricalBayesGMMTrained {
                weights: final_weights,
                means: final_means,
                covariances: final_covariances,

                estimated_alpha_concentration: current_hyperparams.alpha_concentration,
                estimated_mean_precision: current_hyperparams.mean_precision,
                estimated_degrees_of_freedom: current_hyperparams.degrees_of_freedom,
                estimated_scale_matrix: current_hyperparams.scale_matrix,

                variational_pi_alpha: final_variational_params.pi_alpha,
                variational_mu_mean: final_variational_params.mu_mean,
                variational_mu_precision: final_variational_params.mu_precision,
                variational_lambda_nu: final_variational_params.lambda_nu,
                variational_lambda_w: final_variational_params.lambda_w,

                marginal_log_likelihood: final_marginal_likelihood,
                evidence_history,
                hyperparameter_history,
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
            method: self.method,
            max_hyperparameter_iter: self.max_hyperparameter_iter,
            hyperparameter_tol: self.hyperparameter_tol,
            initial_alpha_concentration: self.initial_alpha_concentration,
            initial_mean_precision: self.initial_mean_precision,
            initial_degrees_of_freedom: self.initial_degrees_of_freedom,
            initial_scale_matrix: self.initial_scale_matrix,
            alpha_concentration_bounds: self.alpha_concentration_bounds,
            mean_precision_bounds: self.mean_precision_bounds,
            degrees_of_freedom_bounds: self.degrees_of_freedom_bounds,
            scale_matrix_bounds: self.scale_matrix_bounds,
            learning_rate_hyperparams: self.learning_rate_hyperparams,
            momentum_hyperparams: self.momentum_hyperparams,
        })
    }
}

/// Variational parameters for internal computation
#[derive(Debug, Clone)]
struct VariationalParameters {
    pi_alpha: Array1<f64>,
    mu_mean: Array2<f64>,
    mu_precision: Array2<f64>,
    lambda_nu: Array1<f64>,
    lambda_w: Vec<Array2<f64>>,
    responsibilities: Array2<f64>,
}

impl EmpiricalBayesGMM<Untrained> {
    /// Estimate model parameters given current hyperparameters
    fn estimate_model_parameters(
        &self,
        X: &Array2<f64>,
        hyperparams: &HyperparameterState,
    ) -> SklResult<(
        Array1<f64>,      // weights
        Array2<f64>,      // means
        Vec<Array2<f64>>, // covariances
        VariationalParameters,
        f64, // marginal likelihood
    )> {
        let (n_samples, _n_features) = X.dim();

        // Initialize variational parameters with current hyperparameters
        let (mut pi_alpha, mut mu_mean, mut mu_precision, mut lambda_nu, mut lambda_w) =
            self.initialize_variational_parameters(X, hyperparams)?;

        let mut marginal_likelihood = f64::NEG_INFINITY;

        // Variational EM iterations
        for _iter in 0..self.max_iter {
            // E-step: Update responsibilities
            let responsibilities = self.compute_responsibilities(
                X,
                &pi_alpha,
                &mu_mean,
                &mu_precision,
                &lambda_nu,
                &lambda_w,
            )?;

            // M-step: Update variational parameters
            let (new_pi_alpha, new_mu_mean, new_mu_precision, new_lambda_nu, new_lambda_w) =
                self.update_variational_parameters(X, &responsibilities, hyperparams)?;

            // Compute marginal likelihood (evidence)
            let new_marginal_likelihood = self.compute_marginal_likelihood(
                X,
                &responsibilities,
                &new_pi_alpha,
                &new_mu_mean,
                &new_mu_precision,
                &new_lambda_nu,
                &new_lambda_w,
                hyperparams,
            )?;

            // Check convergence
            if (new_marginal_likelihood - marginal_likelihood).abs() < self.tol {
                break;
            }

            pi_alpha = new_pi_alpha;
            mu_mean = new_mu_mean;
            mu_precision = new_mu_precision;
            lambda_nu = new_lambda_nu;
            lambda_w = new_lambda_w;
            marginal_likelihood = new_marginal_likelihood;
        }

        // Compute final model parameters
        let weights = self.compute_weights(&pi_alpha);
        let means = mu_mean.clone();
        let covariances = self.compute_covariances(&lambda_nu, &lambda_w)?;

        let variational_params = VariationalParameters {
            pi_alpha,
            mu_mean,
            mu_precision,
            lambda_nu,
            lambda_w,
            responsibilities: Array2::zeros((n_samples, self.n_components)), // Will be computed if needed
        };

        Ok((
            weights,
            means,
            covariances,
            variational_params,
            marginal_likelihood,
        ))
    }

    /// Initialize variational parameters with current hyperparameters
    fn initialize_variational_parameters(
        &self,
        X: &Array2<f64>,
        hyperparams: &HyperparameterState,
    ) -> SklResult<(
        Array1<f64>,      // pi_alpha
        Array2<f64>,      // mu_mean
        Array2<f64>,      // mu_precision
        Array1<f64>,      // lambda_nu
        Vec<Array2<f64>>, // lambda_w
    )> {
        let (n_samples, n_features) = X.dim();
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut thread_rng())
        };

        // Initialize using hyperparameters
        let pi_alpha = Array1::from_elem(self.n_components, hyperparams.alpha_concentration);

        // Initialize means with k-means++ style initialization
        let mut mu_mean: Array2<Float> = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            let idx = rng.gen_range(0..n_samples);
            mu_mean.row_mut(k).assign(&X.row(idx));
        }

        let mu_precision =
            Array2::from_elem((self.n_components, n_features), hyperparams.mean_precision);
        let lambda_nu = Array1::from_elem(
            self.n_components,
            hyperparams.degrees_of_freedom + n_features as f64,
        );

        let mut lambda_w = Vec::new();
        for _ in 0..self.n_components {
            let mut w = Array2::eye(n_features);
            w.mapv_inplace(|x| x * hyperparams.scale_matrix);
            lambda_w.push(w);
        }

        Ok((pi_alpha, mu_mean, mu_precision, lambda_nu, lambda_w))
    }

    /// Compute responsibilities (E-step)
    fn compute_responsibilities(
        &self,
        X: &Array2<f64>,
        pi_alpha: &Array1<f64>,
        mu_mean: &Array2<f64>,
        mu_precision: &Array2<f64>,
        lambda_nu: &Array1<f64>,
        lambda_w: &Vec<Array2<f64>>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut responsibilities: Array2<Float> = Array2::zeros((n_samples, self.n_components));

        // Compute digamma of sum for mixing weights
        let digamma_sum = digamma(pi_alpha.sum());

        for i in 0..n_samples {
            for k in 0..self.n_components {
                let mut log_prob = 0.0;

                // E[log π_k] term
                log_prob += digamma(pi_alpha[k]) - digamma_sum;

                // E[log p(x_i | μ_k, Λ_k)] term
                let x_i = X.row(i);
                let mu_k = mu_mean.row(k);

                let diff = &x_i.to_owned() - &mu_k.to_owned();
                let precision_term = lambda_nu[k] * diff.dot(&diff);

                log_prob += 0.5
                    * (self.compute_expected_log_det_lambda(k, lambda_nu, lambda_w)
                        - n_features as f64 * (1.0 / mu_precision[[k, 0]])
                        - precision_term
                        - n_features as f64 * (2.0 * PI).ln());

                responsibilities[[i, k]] = log_prob;
            }

            // Normalize using log-sum-exp
            let max_log_prob = responsibilities
                .row(i)
                .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let mut sum_exp = 0.0;
            for k in 0..self.n_components {
                responsibilities[[i, k]] = (responsibilities[[i, k]] - max_log_prob).exp();
                sum_exp += responsibilities[[i, k]];
            }

            for k in 0..self.n_components {
                responsibilities[[i, k]] /= sum_exp;
            }
        }

        Ok(responsibilities)
    }

    /// Update variational parameters (M-step)
    fn update_variational_parameters(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        hyperparams: &HyperparameterState,
    ) -> SklResult<(
        Array1<f64>,      // pi_alpha
        Array2<f64>,      // mu_mean
        Array2<f64>,      // mu_precision
        Array1<f64>,      // lambda_nu
        Vec<Array2<f64>>, // lambda_w
    )> {
        let (n_samples, n_features) = X.dim();

        // Update π parameters
        let mut pi_alpha = Array1::from_elem(self.n_components, hyperparams.alpha_concentration);
        for k in 0..self.n_components {
            let n_k = responsibilities.column(k).sum();
            pi_alpha[k] = hyperparams.alpha_concentration + n_k;
        }

        // Update μ parameters
        let mut mu_mean: Array2<Float> = Array2::zeros((self.n_components, n_features));
        let mut mu_precision: Array2<Float> = Array2::zeros((self.n_components, n_features));

        for k in 0..self.n_components {
            let n_k = responsibilities.column(k).sum();

            // Compute weighted sample mean
            let mut x_bar_k = Array1::zeros(n_features);
            for i in 0..n_samples {
                let x_i = X.row(i);
                x_bar_k = x_bar_k + responsibilities[[i, k]] * &x_i.to_owned();
            }
            if n_k > 0.0 {
                x_bar_k /= n_k;
            }

            // Update precision and mean
            let new_precision = hyperparams.mean_precision + n_k;
            let new_mean = (hyperparams.mean_precision * 0.0 + n_k * &x_bar_k) / new_precision;

            for d in 0..n_features {
                mu_mean[[k, d]] = new_mean[d];
                mu_precision[[k, d]] = new_precision;
            }
        }

        // Update Λ parameters
        let mut lambda_nu = Array1::zeros(self.n_components);
        let mut lambda_w = Vec::new();

        for k in 0..self.n_components {
            let n_k = responsibilities.column(k).sum();

            // Update degrees of freedom
            lambda_nu[k] = hyperparams.degrees_of_freedom + n_k;

            // Compute scatter matrix
            let mut s_k: Array2<Float> = Array2::zeros((n_features, n_features));
            for i in 0..n_samples {
                let x_i = X.row(i);
                let mu_k = mu_mean.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();
                let outer =
                    &diff.to_owned().insert_axis(Axis(1)) * &diff.to_owned().insert_axis(Axis(0));
                s_k = s_k + responsibilities[[i, k]] * &outer;
            }

            // Update scale matrix
            let mut w_k = Array2::eye(n_features) * hyperparams.scale_matrix;
            w_k = w_k + s_k;

            // Add uncertainty in mean estimates
            for d in 0..n_features {
                w_k[[d, d]] += 1.0 / mu_precision[[k, d]];
            }

            lambda_w.push(w_k);
        }

        Ok((pi_alpha, mu_mean, mu_precision, lambda_nu, lambda_w))
    }

    /// Compute marginal likelihood (evidence)
    fn compute_marginal_likelihood(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        pi_alpha: &Array1<f64>,
        mu_mean: &Array2<f64>,
        mu_precision: &Array2<f64>,
        lambda_nu: &Array1<f64>,
        lambda_w: &Vec<Array2<f64>>,
        hyperparams: &HyperparameterState,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = X.dim();
        let mut log_evidence = 0.0;

        // E[log p(X | Z, θ)] - expected log-likelihood
        for i in 0..n_samples {
            for k in 0..self.n_components {
                let x_i = X.row(i);
                let mu_k = mu_mean.row(k);
                let diff = &x_i.to_owned() - &mu_k.to_owned();

                let log_likelihood = 0.5
                    * (self.compute_expected_log_det_lambda(k, lambda_nu, lambda_w)
                        - n_features as f64 * (2.0 * PI).ln()
                        - lambda_nu[k] * diff.dot(&diff)
                        - n_features as f64 / mu_precision[[k, 0]]);

                log_evidence += responsibilities[[i, k]] * log_likelihood;
            }
        }

        // Prior terms
        log_evidence += self.compute_prior_log_likelihood(hyperparams);

        // Entropy terms (variational lower bound)
        log_evidence += self.compute_entropy_terms(
            responsibilities,
            pi_alpha,
            mu_precision,
            lambda_nu,
            lambda_w,
        );

        Ok(log_evidence)
    }

    /// Compute prior log likelihood contribution
    fn compute_prior_log_likelihood(&self, hyperparams: &HyperparameterState) -> f64 {
        let mut log_prior = 0.0;

        // Dirichlet prior for mixing weights
        let alpha_sum = hyperparams.alpha_concentration * self.n_components as f64;
        log_prior += log_gamma(alpha_sum)
            - self.n_components as f64 * log_gamma(hyperparams.alpha_concentration);

        // Wishart prior for precision matrices
        log_prior += self.n_components as f64
            * (0.5 * hyperparams.degrees_of_freedom * hyperparams.scale_matrix.ln()
                - 0.5 * hyperparams.degrees_of_freedom * (2.0 * PI).ln());

        log_prior
    }

    /// Compute entropy terms for variational lower bound
    fn compute_entropy_terms(
        &self,
        responsibilities: &Array2<f64>,
        pi_alpha: &Array1<f64>,
        mu_precision: &Array2<f64>,
        _lambda_nu: &Array1<f64>,
        _lambda_w: &Vec<Array2<f64>>,
    ) -> f64 {
        let mut entropy = 0.0;

        // Entropy of responsibilities
        for i in 0..responsibilities.nrows() {
            for k in 0..self.n_components {
                if responsibilities[[i, k]] > 1e-10 {
                    entropy -= responsibilities[[i, k]] * responsibilities[[i, k]].ln();
                }
            }
        }

        // Entropy of Dirichlet distribution
        entropy -= log_gamma(pi_alpha.sum()) - pi_alpha.iter().map(|&x| log_gamma(x)).sum::<f64>();

        // Entropy of Gaussian distributions for means
        for k in 0..self.n_components {
            for d in 0..mu_precision.ncols() {
                entropy -= -0.5 * (1.0 + mu_precision[[k, d]].ln());
            }
        }

        entropy
    }

    /// Type-II Maximum Likelihood hyperparameter update
    fn type_ii_ml_update(
        &self,
        X: &Array2<f64>,
        variational_params: &VariationalParameters,
        current_hyperparams: &HyperparameterState,
    ) -> SklResult<HyperparameterState> {
        let (_n_samples, n_features) = X.dim();

        // Compute sufficient statistics
        let _total_responsibility: f64 = variational_params.pi_alpha.sum()
            - self.n_components as f64 * current_hyperparams.alpha_concentration;

        // Update alpha concentration (Dirichlet parameter)
        let new_alpha = self.optimize_alpha_concentration(
            &variational_params.pi_alpha,
            current_hyperparams.alpha_concentration,
        )?;

        // Update mean precision
        let mean_squared_deviations: f64 = variational_params.mu_mean.iter().map(|&x| x * x).sum();
        let new_mean_precision =
            self.n_components as f64 * n_features as f64 / mean_squared_deviations.max(1e-10);
        let bounded_mean_precision = new_mean_precision
            .max(self.mean_precision_bounds.0)
            .min(self.mean_precision_bounds.1);

        // Update degrees of freedom
        let expected_log_det_sum: f64 = (0..self.n_components)
            .map(|k| {
                self.compute_expected_log_det_lambda(
                    k,
                    &variational_params.lambda_nu,
                    &variational_params.lambda_w,
                )
            })
            .sum();
        let new_dof = expected_log_det_sum / self.n_components as f64;
        let bounded_dof = new_dof
            .max(self.degrees_of_freedom_bounds.0)
            .min(self.degrees_of_freedom_bounds.1);

        // Update scale matrix
        let trace_sum: f64 = variational_params
            .lambda_w
            .iter()
            .map(|w| w.diag().sum())
            .sum();
        let new_scale = trace_sum / (self.n_components as f64 * n_features as f64 * bounded_dof);
        let bounded_scale = new_scale
            .max(self.scale_matrix_bounds.0)
            .min(self.scale_matrix_bounds.1);

        Ok(HyperparameterState {
            alpha_concentration: new_alpha,
            mean_precision: bounded_mean_precision,
            degrees_of_freedom: bounded_dof,
            scale_matrix: bounded_scale,
            marginal_likelihood: current_hyperparams.marginal_likelihood,
        })
    }

    /// Optimize alpha concentration using Newton's method
    fn optimize_alpha_concentration(
        &self,
        pi_alpha: &Array1<f64>,
        current_alpha: f64,
    ) -> SklResult<f64> {
        let mut alpha = current_alpha;
        let digamma_sum = digamma(pi_alpha.sum());

        for _iter in 0..10 {
            let gradient = self.n_components as f64 * (digamma(alpha) - digamma_sum);
            let hessian = self.n_components as f64 * trigamma(alpha);

            if hessian.abs() < 1e-10 {
                break;
            }

            let update = gradient / hessian;
            alpha = (alpha - update)
                .max(self.alpha_concentration_bounds.0)
                .min(self.alpha_concentration_bounds.1);

            if update.abs() < 1e-6 {
                break;
            }
        }

        Ok(alpha)
    }

    /// EM hyperparameter update
    fn em_hyperparameter_update(
        &self,
        X: &Array2<f64>,
        _variational_params: &VariationalParameters,
        current_hyperparams: &HyperparameterState,
    ) -> SklResult<HyperparameterState> {
        // For EM approach, use sample moments to update hyperparameters
        let (_n_samples, n_features) = X.dim();

        // Use sample statistics to estimate hyperparameters
        let sample_var = X.var_axis(Axis(0), 0.0);
        let mean_sample_var = sample_var.mean().unwrap_or(1.0);

        let new_alpha = (current_hyperparams.alpha_concentration
            + 0.1 * (1.0 - current_hyperparams.alpha_concentration))
            .max(self.alpha_concentration_bounds.0)
            .min(self.alpha_concentration_bounds.1);

        let new_mean_precision = (1.0 / mean_sample_var)
            .max(self.mean_precision_bounds.0)
            .min(self.mean_precision_bounds.1);

        let new_dof = (n_features as f64 + 2.0)
            .max(self.degrees_of_freedom_bounds.0)
            .min(self.degrees_of_freedom_bounds.1);

        let new_scale = mean_sample_var
            .max(self.scale_matrix_bounds.0)
            .min(self.scale_matrix_bounds.1);

        Ok(HyperparameterState {
            alpha_concentration: new_alpha,
            mean_precision: new_mean_precision,
            degrees_of_freedom: new_dof,
            scale_matrix: new_scale,
            marginal_likelihood: current_hyperparams.marginal_likelihood,
        })
    }

    /// Cross-validation hyperparameter update
    fn cross_validation_update(
        &self,
        X: &Array2<f64>,
        current_hyperparams: &HyperparameterState,
    ) -> SklResult<HyperparameterState> {
        // Simple grid search over hyperparameters using cross-validation
        let alpha_candidates = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let precision_candidates = vec![0.01, 0.1, 1.0, 10.0];

        let mut best_hyperparams = current_hyperparams.clone();
        let mut best_score = f64::NEG_INFINITY;

        for &alpha in &alpha_candidates {
            for &precision in &precision_candidates {
                let test_hyperparams = HyperparameterState {
                    alpha_concentration: alpha,
                    mean_precision: precision,
                    degrees_of_freedom: current_hyperparams.degrees_of_freedom,
                    scale_matrix: current_hyperparams.scale_matrix,
                    marginal_likelihood: 0.0,
                };

                let score = self.evaluate_hyperparameters(X, &test_hyperparams)?;

                if score > best_score {
                    best_score = score;
                    best_hyperparams = test_hyperparams;
                }
            }
        }

        best_hyperparams.marginal_likelihood = best_score;
        Ok(best_hyperparams)
    }

    /// Gradient ascent hyperparameter update
    fn gradient_ascent_update(
        &self,
        X: &Array2<f64>,
        variational_params: &VariationalParameters,
        current_hyperparams: &HyperparameterState,
        momentum_alpha: &mut f64,
        momentum_mean_precision: &mut f64,
        momentum_dof: &mut f64,
        momentum_scale: &mut f64,
    ) -> SklResult<HyperparameterState> {
        // Compute gradients of marginal likelihood w.r.t. hyperparameters
        let gradients =
            self.compute_hyperparameter_gradients(X, variational_params, current_hyperparams)?;

        // Update momentum terms
        *momentum_alpha = self.momentum_hyperparams * *momentum_alpha
            + (1.0 - self.momentum_hyperparams) * gradients.0;
        *momentum_mean_precision = self.momentum_hyperparams * *momentum_mean_precision
            + (1.0 - self.momentum_hyperparams) * gradients.1;
        *momentum_dof = self.momentum_hyperparams * *momentum_dof
            + (1.0 - self.momentum_hyperparams) * gradients.2;
        *momentum_scale = self.momentum_hyperparams * *momentum_scale
            + (1.0 - self.momentum_hyperparams) * gradients.3;

        // Update hyperparameters
        let new_alpha = (current_hyperparams.alpha_concentration
            + self.learning_rate_hyperparams * *momentum_alpha)
            .max(self.alpha_concentration_bounds.0)
            .min(self.alpha_concentration_bounds.1);

        let new_mean_precision = (current_hyperparams.mean_precision
            + self.learning_rate_hyperparams * *momentum_mean_precision)
            .max(self.mean_precision_bounds.0)
            .min(self.mean_precision_bounds.1);

        let new_dof = (current_hyperparams.degrees_of_freedom
            + self.learning_rate_hyperparams * *momentum_dof)
            .max(self.degrees_of_freedom_bounds.0)
            .min(self.degrees_of_freedom_bounds.1);

        let new_scale = (current_hyperparams.scale_matrix
            + self.learning_rate_hyperparams * *momentum_scale)
            .max(self.scale_matrix_bounds.0)
            .min(self.scale_matrix_bounds.1);

        Ok(HyperparameterState {
            alpha_concentration: new_alpha,
            mean_precision: new_mean_precision,
            degrees_of_freedom: new_dof,
            scale_matrix: new_scale,
            marginal_likelihood: current_hyperparams.marginal_likelihood,
        })
    }

    /// Compute gradients of marginal likelihood w.r.t. hyperparameters
    fn compute_hyperparameter_gradients(
        &self,
        X: &Array2<f64>,
        _variational_params: &VariationalParameters,
        hyperparams: &HyperparameterState,
    ) -> SklResult<(f64, f64, f64, f64)> {
        let eps = 1e-6;

        // Numerical gradients (finite differences)
        let baseline_score = self.evaluate_hyperparameters(X, hyperparams)?;

        // Gradient w.r.t. alpha
        let mut perturbed_hyperparams = hyperparams.clone();
        perturbed_hyperparams.alpha_concentration += eps;
        let alpha_score = self.evaluate_hyperparameters(X, &perturbed_hyperparams)?;
        let grad_alpha = (alpha_score - baseline_score) / eps;

        // Gradient w.r.t. mean precision
        perturbed_hyperparams = hyperparams.clone();
        perturbed_hyperparams.mean_precision += eps;
        let precision_score = self.evaluate_hyperparameters(X, &perturbed_hyperparams)?;
        let grad_precision = (precision_score - baseline_score) / eps;

        // Gradient w.r.t. degrees of freedom
        perturbed_hyperparams = hyperparams.clone();
        perturbed_hyperparams.degrees_of_freedom += eps;
        let dof_score = self.evaluate_hyperparameters(X, &perturbed_hyperparams)?;
        let grad_dof = (dof_score - baseline_score) / eps;

        // Gradient w.r.t. scale matrix
        perturbed_hyperparams = hyperparams.clone();
        perturbed_hyperparams.scale_matrix += eps;
        let scale_score = self.evaluate_hyperparameters(X, &perturbed_hyperparams)?;
        let grad_scale = (scale_score - baseline_score) / eps;

        Ok((grad_alpha, grad_precision, grad_dof, grad_scale))
    }

    /// Evaluate hyperparameters using marginal likelihood
    fn evaluate_hyperparameters(
        &self,
        X: &Array2<f64>,
        hyperparams: &HyperparameterState,
    ) -> SklResult<f64> {
        let (_, _, _, _, marginal_likelihood) = self.estimate_model_parameters(X, hyperparams)?;
        Ok(marginal_likelihood)
    }

    /// Compute hyperparameter change for convergence checking
    fn compute_hyperparameter_change(
        &self,
        old_hyperparams: &HyperparameterState,
        new_hyperparams: &HyperparameterState,
    ) -> f64 {
        let alpha_change =
            (new_hyperparams.alpha_concentration - old_hyperparams.alpha_concentration).abs();
        let precision_change =
            (new_hyperparams.mean_precision - old_hyperparams.mean_precision).abs();
        let dof_change =
            (new_hyperparams.degrees_of_freedom - old_hyperparams.degrees_of_freedom).abs();
        let scale_change = (new_hyperparams.scale_matrix - old_hyperparams.scale_matrix).abs();

        alpha_change + precision_change + dof_change + scale_change
    }

    /// Compute expected log determinant of precision matrix
    fn compute_expected_log_det_lambda(
        &self,
        k: usize,
        lambda_nu: &Array1<f64>,
        lambda_w: &[Array2<f64>],
    ) -> f64 {
        let n_features = lambda_w[k].nrows();
        let mut log_det = 0.0;

        for i in 0..n_features {
            log_det += digamma(0.5 * (lambda_nu[k] + 1.0 - i as f64));
        }

        log_det += n_features as f64 * (2.0_f64).ln();
        log_det -= lambda_w[k].diag().iter().map(|&x| x.ln()).sum::<f64>();

        log_det
    }

    /// Compute mixing weights from Dirichlet parameters
    fn compute_weights(&self, pi_alpha: &Array1<f64>) -> Array1<f64> {
        let alpha_sum = pi_alpha.sum();
        pi_alpha.mapv(|x| x / alpha_sum)
    }

    /// Compute covariance matrices from precision parameters
    fn compute_covariances(
        &self,
        lambda_nu: &Array1<f64>,
        lambda_w: &[Array2<f64>],
    ) -> SklResult<Vec<Array2<f64>>> {
        let mut covariances = Vec::new();

        for k in 0..self.n_components {
            let precision = &lambda_w[k] * lambda_nu[k];

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

        // Simple diagonal inverse for now
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

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for EmpiricalBayesGMM<EmpiricalBayesGMMTrained> {
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

impl EmpiricalBayesGMM<EmpiricalBayesGMMTrained> {
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

    /// Get the estimated hyperparameters
    pub fn estimated_hyperparameters(&self) -> HyperparameterState {
        HyperparameterState {
            alpha_concentration: self.state.estimated_alpha_concentration,
            mean_precision: self.state.estimated_mean_precision,
            degrees_of_freedom: self.state.estimated_degrees_of_freedom,
            scale_matrix: self.state.estimated_scale_matrix,
            marginal_likelihood: self.state.marginal_log_likelihood,
        }
    }

    /// Get the marginal log-likelihood (evidence)
    pub fn marginal_log_likelihood(&self) -> f64 {
        self.state.marginal_log_likelihood
    }

    /// Get the evidence history
    pub fn evidence_history(&self) -> &Vec<f64> {
        &self.state.evidence_history
    }

    /// Get the hyperparameter optimization history
    pub fn hyperparameter_history(&self) -> &Vec<HyperparameterState> {
        &self.state.hyperparameter_history
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

    /// Compute predictive probabilities
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_samples = X.nrows();
        let mut probas: Array2<Float> = Array2::zeros((n_samples, self.n_components));

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

    /// Get model evidence (marginal likelihood)
    pub fn model_evidence(&self) -> f64 {
        self.state.marginal_log_likelihood
    }

    /// Get Bayesian Information Criterion incorporating evidence
    pub fn bayesian_information_criterion(&self, X: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let n_samples = X.nrows() as f64;
        let n_params = self.count_parameters();
        let log_likelihood = self.score(X)?;

        Ok(-2.0 * log_likelihood + n_params as f64 * n_samples.ln())
    }

    /// Count the number of parameters in the model
    fn count_parameters(&self) -> usize {
        let n_features = self.state.means.ncols();

        // Mixing weights (K-1 free parameters)
        let weight_params = self.n_components - 1;

        // Means (K * D parameters)
        let mean_params = self.n_components * n_features;

        // Covariances (depends on covariance type)
        let cov_params = match self.covariance_type {
            CovarianceType::Full => self.n_components * n_features * (n_features + 1) / 2,
            CovarianceType::Diagonal => self.n_components * n_features,
            CovarianceType::Tied => n_features * (n_features + 1) / 2,
            CovarianceType::Spherical => self.n_components,
        };

        weight_params + mean_params + cov_params
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

fn trigamma(x: f64) -> f64 {
    // Approximation of trigamma function (derivative of digamma)
    if x > 0.0 {
        1.0 / x + 0.5 / (x * x)
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
    fn test_empirical_bayes_gmm_creation() {
        let model = EmpiricalBayesGMM::new()
            .n_components(3)
            .method(EmpiricalBayesMethod::TypeIIML)
            .max_hyperparameter_iter(20)
            .hyperparameter_tol(1e-3);

        assert_eq!(model.n_components, 3);
        assert_eq!(model.method, EmpiricalBayesMethod::TypeIIML);
        assert_eq!(model.max_hyperparameter_iter, 20);
        assert_eq!(model.hyperparameter_tol, 1e-3);
    }

    #[test]
    fn test_empirical_bayes_gmm_builder() {
        let model = EmpiricalBayesGMM::builder()
            .n_components(2)
            .covariance_type(CovarianceType::Diagonal)
            .method(EmpiricalBayesMethod::EM)
            .build();

        assert_eq!(model.n_components, 2);
        assert!(matches!(model.covariance_type, CovarianceType::Diagonal));
        assert_eq!(model.method, EmpiricalBayesMethod::EM);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_fit_simple() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .method(EmpiricalBayesMethod::TypeIIML)
            .max_iter(5)
            .max_hyperparameter_iter(3)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        assert_eq!(fitted.n_components, 2);
        assert!(fitted.n_iter() > 0);
        assert!(fitted.marginal_log_likelihood().is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_methods() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let methods = vec![
            EmpiricalBayesMethod::TypeIIML,
            EmpiricalBayesMethod::EM,
            EmpiricalBayesMethod::CrossValidation,
            EmpiricalBayesMethod::GradientAscent,
        ];

        for method in methods {
            let model = EmpiricalBayesGMM::new()
                .n_components(2)
                .method(method)
                .max_iter(3)
                .max_hyperparameter_iter(2)
                .random_state(42);

            let fitted = model.fit(&X.view(), &()).unwrap();
            assert_eq!(fitted.n_components, 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_predict() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .max_iter(3)
            .max_hyperparameter_iter(2)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let labels = fitted.predict(&X.view()).unwrap();

        assert_eq!(labels.len(), 6);
        // Check that labels are in valid range
        for &label in labels.iter() {
            assert!(label >= 0 && label < 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_predict_proba() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .max_iter(3)
            .max_hyperparameter_iter(2)
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
    fn test_empirical_bayes_gmm_hyperparameters() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .initial_alpha_concentration(2.0)
            .initial_mean_precision(0.5)
            .max_iter(3)
            .max_hyperparameter_iter(2)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();

        let hyperparams = fitted.estimated_hyperparameters();
        assert!(hyperparams.alpha_concentration > 0.0);
        assert!(hyperparams.mean_precision > 0.0);
        assert!(hyperparams.degrees_of_freedom > 0.0);
        assert!(hyperparams.scale_matrix > 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_properties() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .max_iter(3)
            .max_hyperparameter_iter(2)
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
        assert!(fitted.marginal_log_likelihood().is_finite());
        assert!(!fitted.evidence_history().is_empty());
        assert!(!fitted.hyperparameter_history().is_empty());
        assert!(fitted.n_iter() > 0);
        assert!(fitted.effective_components() > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_bounds() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .alpha_concentration_bounds((0.5, 5.0))
            .mean_precision_bounds((0.1, 10.0))
            .max_iter(3)
            .max_hyperparameter_iter(2)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();

        let hyperparams = fitted.estimated_hyperparameters();
        assert!(hyperparams.alpha_concentration >= 0.5);
        assert!(hyperparams.alpha_concentration <= 5.0);
        assert!(hyperparams.mean_precision >= 0.1);
        assert!(hyperparams.mean_precision <= 10.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_score() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .max_iter(3)
            .max_hyperparameter_iter(2)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let score = fitted.score(&X.view()).unwrap();

        // Score should be finite
        assert!(score.is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empirical_bayes_gmm_bic() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let model = EmpiricalBayesGMM::new()
            .n_components(2)
            .max_iter(3)
            .max_hyperparameter_iter(2)
            .random_state(42);

        let fitted = model.fit(&X.view(), &()).unwrap();
        let bic = fitted.bayesian_information_criterion(&X.view()).unwrap();

        // BIC should be finite
        assert!(bic.is_finite());
    }
}
