//! Exponential Family Mixture Models
//!
//! This module provides mixture models for exponential family distributions,
//! supporting various distribution types including Poisson, Exponential, Gamma,
//! Bernoulli, and Multinomial distributions using natural parameter representation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

use crate::common::ModelSelection;
use std::f64::consts::PI;

/// Exponential family distribution types
#[derive(Debug, Clone, PartialEq)]
pub enum ExponentialFamilyType {
    /// Poisson distribution for count data
    Poisson,
    /// Exponential distribution for waiting times
    Exponential,
    /// Gamma distribution for positive continuous data
    Gamma,
    /// Bernoulli distribution for binary data
    Bernoulli,
    /// Multinomial distribution for categorical data
    Multinomial(usize), // number of categories
}

/// Exponential Family Mixture Model
///
/// A mixture model for exponential family distributions using natural parameter representation.
/// Supports multiple distribution types with proper EM algorithm implementation.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `family_type` - Type of exponential family distribution
/// * `tol` - Convergence threshold
/// * `max_iter` - Maximum number of EM iterations
/// * `n_init` - Number of initializations to perform
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_mixture::{ExponentialFamilyMixture, ExponentialFamilyType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// // For Poisson count data
/// let X = array![[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]];
/// let model = ExponentialFamilyMixture::new()
///     .n_components(2)
///     .family_type(ExponentialFamilyType::Poisson)
///     .max_iter(100);
/// let fitted = model.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialFamilyMixture<S = Untrained> {
    state: S,
    n_components: usize,
    family_type: ExponentialFamilyType,
    tol: f64,
    max_iter: usize,
    n_init: usize,
    random_state: Option<u64>,
}

impl ExponentialFamilyMixture<Untrained> {
    /// Create a new ExponentialFamilyMixture instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            family_type: ExponentialFamilyType::Poisson,
            tol: 1e-3,
            max_iter: 100,
            n_init: 1,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the exponential family type
    pub fn family_type(mut self, family_type: ExponentialFamilyType) -> Self {
        self.family_type = family_type;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
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

    /// Validate data for the specific distribution type
    fn validate_data(&self, X: &Array2<f64>) -> SklResult<()> {
        match &self.family_type {
            ExponentialFamilyType::Poisson => {
                // Check for non-negative integers
                for value in X.iter() {
                    if *value < 0.0 || value.fract() != 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Poisson distribution requires non-negative integer data".to_string(),
                        ));
                    }
                }
            }
            ExponentialFamilyType::Exponential => {
                // Check for positive values
                for value in X.iter() {
                    if *value <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Exponential distribution requires positive data".to_string(),
                        ));
                    }
                }
            }
            ExponentialFamilyType::Gamma => {
                // Check for positive values
                for value in X.iter() {
                    if *value <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Gamma distribution requires positive data".to_string(),
                        ));
                    }
                }
            }
            ExponentialFamilyType::Bernoulli => {
                // Check for binary values
                for value in X.iter() {
                    if *value != 0.0 && *value != 1.0 {
                        return Err(SklearsError::InvalidInput(
                            "Bernoulli distribution requires binary (0/1) data".to_string(),
                        ));
                    }
                }
            }
            ExponentialFamilyType::Multinomial(n_categories) => {
                // Check dimensions and non-negative integer counts
                if X.dim().1 != *n_categories {
                    return Err(SklearsError::InvalidInput(format!(
                        "Multinomial data must have {} columns for {} categories",
                        n_categories, n_categories
                    )));
                }
                for value in X.iter() {
                    if *value < 0.0 || value.fract() != 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Multinomial distribution requires non-negative integer counts"
                                .to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Initialize natural parameters for the distribution
    fn initialize_natural_parameters(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();

        match &self.family_type {
            ExponentialFamilyType::Poisson => {
                // Natural parameter is log(lambda)
                let mut params = Array2::zeros((self.n_components, 1));
                let mean_rate = X.mean().unwrap_or(1.0).max(0.1);

                for k in 0..self.n_components {
                    // Initialize with slight variations around sample mean
                    let rate =
                        mean_rate * (1.0 + 0.1 * (k as f64 - self.n_components as f64 / 2.0));
                    params[[k, 0]] = rate.max(0.1).ln();
                }
                Ok(params)
            }
            ExponentialFamilyType::Exponential => {
                // Natural parameter is -rate
                let mut params = Array2::zeros((self.n_components, 1));
                let mean_value = X.mean().unwrap_or(1.0);
                let base_rate = 1.0 / mean_value;

                for k in 0..self.n_components {
                    let rate = base_rate * (0.5 + k as f64);
                    params[[k, 0]] = -rate;
                }
                Ok(params)
            }
            ExponentialFamilyType::Gamma => {
                // Natural parameters are [alpha-1, -beta]
                let mut params = Array2::zeros((self.n_components, 2));
                let sample_mean = X.mean().unwrap_or(1.0);
                let sample_var = X.var(0.0);

                // Method of moments initialization
                let beta = sample_var / sample_mean;
                let alpha = sample_mean / beta;

                for k in 0..self.n_components {
                    let scale = 0.5 + k as f64 * 0.5;
                    params[[k, 0]] = (alpha * scale - 1.0).max(0.1);
                    params[[k, 1]] = -beta / scale;
                }
                Ok(params)
            }
            ExponentialFamilyType::Bernoulli => {
                // Natural parameter is logit(p)
                let mut params = Array2::zeros((self.n_components, 1));
                let sample_mean = X.mean().unwrap_or(0.5);

                for k in 0..self.n_components {
                    let p = (sample_mean + 0.1 * (k as f64 - self.n_components as f64 / 2.0))
                        .max(0.01)
                        .min(0.99);
                    params[[k, 0]] = (p / (1.0 - p)).ln();
                }
                Ok(params)
            }
            ExponentialFamilyType::Multinomial(n_categories) => {
                // Natural parameters are log probabilities (except last category)
                let mut params = Array2::zeros((self.n_components, n_categories - 1));

                for k in 0..self.n_components {
                    // Initialize with slight variations
                    let base_prob = 1.0 / *n_categories as f64;
                    for j in 0..(n_categories - 1) {
                        let prob = base_prob
                            * (0.8
                                + 0.4 * (k as f64 + j as f64)
                                    / (self.n_components + n_categories) as f64);
                        params[[k, j]] = prob.max(0.01).ln();
                    }
                }
                Ok(params)
            }
        }
    }

    /// Compute log probability for the given distribution
    fn log_probability(
        &self,
        x: &ArrayView1<f64>,
        natural_params: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        match &self.family_type {
            ExponentialFamilyType::Poisson => {
                let lambda = natural_params[0].exp();
                let x_val = x[0];
                Ok(x_val * natural_params[0] - lambda - self.log_factorial(x_val as u32))
            }
            ExponentialFamilyType::Exponential => {
                let rate = -natural_params[0];
                let x_val = x[0];
                Ok(natural_params[0] * x_val + rate.ln())
            }
            ExponentialFamilyType::Gamma => {
                let alpha = natural_params[0] + 1.0;
                let beta = -natural_params[1];
                let x_val = x[0];
                Ok(
                    natural_params[0] * x_val.ln() + natural_params[1] * x_val + alpha * beta.ln()
                        - self.log_gamma(alpha),
                )
            }
            ExponentialFamilyType::Bernoulli => {
                let x_val = x[0];
                let log_odds = natural_params[0];
                Ok(x_val * log_odds - (1.0 + log_odds.exp()).ln())
            }
            ExponentialFamilyType::Multinomial(n_categories) => {
                let mut log_prob = 0.0;
                let n_trials: f64 = x.iter().sum();

                // Add multinomial coefficient (log factorial terms)
                log_prob += self.log_factorial(n_trials as u32);
                for &count in x.iter() {
                    log_prob -= self.log_factorial(count as u32);
                }

                // Add probability terms
                for j in 0..(n_categories - 1) {
                    log_prob += x[j] * natural_params[j];
                }

                // Normalization term
                let log_sum: f64 = natural_params.iter().map(|&p| p.exp()).sum::<f64>() + 1.0;
                log_prob -= n_trials * log_sum.ln();

                Ok(log_prob)
            }
        }
    }

    /// Simple log factorial implementation
    fn log_factorial(&self, n: u32) -> f64 {
        if n <= 1 {
            0.0
        } else {
            (2..=n).map(|i| (i as f64).ln()).sum()
        }
    }

    /// Simple log gamma implementation using Stirling's approximation
    fn log_gamma(&self, x: f64) -> f64 {
        if x < 12.0 {
            if x < 0.5 {
                (PI / ((PI * x).sin())).ln() - self.log_gamma(1.0 - x)
            } else {
                self.log_gamma(x + 1.0) - x.ln()
            }
        } else {
            0.5 * (2.0 * PI).ln() - 0.5 * x.ln() + x * x.ln() - x
        }
    }

    /// Numerically stable log-sum-exp
    fn log_sum_exp(&self, log_probs: &[f64]) -> f64 {
        let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_log_prob.is_infinite() {
            max_log_prob
        } else {
            max_log_prob
                + log_probs
                    .iter()
                    .map(|&lp| (lp - max_log_prob).exp())
                    .sum::<f64>()
                    .ln()
        }
    }

    /// Compute responsibilities (E-step)
    fn compute_responsibilities(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        natural_params: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_probs = Vec::new();

            for k in 0..self.n_components {
                let log_weight = weights[k].ln();
                let log_pdf = self.log_probability(&sample, &natural_params.row(k))?;
                log_probs.push(log_weight + log_pdf);
            }

            let log_sum_exp = self.log_sum_exp(&log_probs);

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
    ) -> SklResult<(Array1<f64>, Array2<f64>)> {
        let (n_samples, _) = X.dim();

        // Update weights
        let mut new_weights = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            new_weights[k] = responsibilities.column(k).sum() / n_samples as f64;
        }

        // Update natural parameters based on distribution type
        let new_params = match &self.family_type {
            ExponentialFamilyType::Poisson => {
                let mut params = Array2::zeros((self.n_components, 1));
                for k in 0..self.n_components {
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        weighted_sum += responsibilities[[i, k]] * X[[i, 0]];
                        weight_sum += responsibilities[[i, k]];
                    }

                    if weight_sum > 1e-10 {
                        let lambda = (weighted_sum / weight_sum).max(1e-10);
                        params[[k, 0]] = lambda.ln();
                    }
                }
                params
            }
            ExponentialFamilyType::Exponential => {
                let mut params = Array2::zeros((self.n_components, 1));
                for k in 0..self.n_components {
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        weighted_sum += responsibilities[[i, k]] * X[[i, 0]];
                        weight_sum += responsibilities[[i, k]];
                    }

                    if weight_sum > 1e-10 {
                        let mean = weighted_sum / weight_sum;
                        let rate = 1.0 / mean.max(1e-10);
                        params[[k, 0]] = -rate;
                    }
                }
                params
            }
            ExponentialFamilyType::Gamma => {
                let mut params = Array2::zeros((self.n_components, 2));
                for k in 0..self.n_components {
                    let mut weighted_sum = 0.0;
                    let mut weighted_log_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        let weight = responsibilities[[i, k]];
                        let x_val = X[[i, 0]];
                        weighted_sum += weight * x_val;
                        weighted_log_sum += weight * x_val.ln();
                        weight_sum += weight;
                    }

                    if weight_sum > 1e-10 {
                        let mean = weighted_sum / weight_sum;
                        let log_mean = weighted_log_sum / weight_sum;

                        // Method of moments estimation
                        let s = (mean.ln() - log_mean).max(1e-10);
                        let alpha = ((3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt())
                            / (12.0 * s))
                            .max(0.1);
                        let beta = alpha / mean;

                        params[[k, 0]] = alpha - 1.0;
                        params[[k, 1]] = -beta;
                    }
                }
                params
            }
            ExponentialFamilyType::Bernoulli => {
                let mut params = Array2::zeros((self.n_components, 1));
                for k in 0..self.n_components {
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        weighted_sum += responsibilities[[i, k]] * X[[i, 0]];
                        weight_sum += responsibilities[[i, k]];
                    }

                    if weight_sum > 1e-10 {
                        let p = (weighted_sum / weight_sum).max(1e-10).min(1.0 - 1e-10);
                        params[[k, 0]] = (p / (1.0 - p)).ln();
                    }
                }
                params
            }
            ExponentialFamilyType::Multinomial(n_categories) => {
                let mut params = Array2::zeros((self.n_components, n_categories - 1));
                for k in 0..self.n_components {
                    let mut category_sums = Array1::zeros(*n_categories);
                    let mut weight_sum = 0.0;

                    for i in 0..n_samples {
                        let weight = responsibilities[[i, k]];
                        for j in 0..*n_categories {
                            category_sums[j] += weight * X[[i, j]];
                        }
                        weight_sum += weight;
                    }

                    if weight_sum > 1e-10 {
                        let total_counts: f64 = category_sums.sum();
                        if total_counts > 1e-10 {
                            for j in 0..(n_categories - 1) {
                                let prob = (category_sums[j] / total_counts).max(1e-10);
                                params[[k, j]] = prob.ln();
                            }
                        }
                    }
                }
                params
            }
        };

        Ok((new_weights, new_params))
    }

    /// Compute log-likelihood of the model
    fn compute_log_likelihood(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        natural_params: &Array2<f64>,
    ) -> SklResult<f64> {
        let (n_samples, _) = X.dim();
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut sample_likelihood = 0.0;

            for k in 0..self.n_components {
                let log_pdf = self.log_probability(&sample, &natural_params.row(k))?;
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

    /// Get number of parameters for the distribution
    fn n_parameters(&self, n_features: usize) -> usize {
        let component_params = match &self.family_type {
            ExponentialFamilyType::Poisson => 1,
            ExponentialFamilyType::Exponential => 1,
            ExponentialFamilyType::Gamma => 2,
            ExponentialFamilyType::Bernoulli => 1,
            ExponentialFamilyType::Multinomial(n_categories) => n_categories - 1,
        };
        (self.n_components - 1) + self.n_components * component_params
    }
}

impl Default for ExponentialFamilyMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ExponentialFamilyMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ExponentialFamilyMixture<Untrained> {
    type Fitted = ExponentialFamilyMixture<ExponentialFamilyMixtureTrained>;

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

        // Validate data for the specific distribution
        self.validate_data(&X)?;

        let mut best_params = None;
        let mut best_log_likelihood = f64::NEG_INFINITY;
        let mut best_n_iter = 0;
        let mut best_converged = false;

        // Run multiple initializations and keep the best
        for _init_run in 0..self.n_init {
            // Initialize parameters
            let mut weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
            let mut natural_params = self.initialize_natural_parameters(&X)?;

            let mut log_likelihood = f64::NEG_INFINITY;
            let mut converged = false;
            let mut n_iter = 0;

            // EM iterations
            for iteration in 0..self.max_iter {
                n_iter = iteration + 1;

                // E-step: Compute responsibilities
                let responsibilities =
                    self.compute_responsibilities(&X, &weights, &natural_params)?;

                // M-step: Update parameters
                let (new_weights, new_natural_params) =
                    self.update_parameters(&X, &responsibilities)?;

                // Compute log-likelihood
                let new_log_likelihood =
                    self.compute_log_likelihood(&X, &new_weights, &new_natural_params)?;

                // Check convergence
                if iteration > 0 && (new_log_likelihood - log_likelihood).abs() < self.tol {
                    converged = true;
                }

                weights = new_weights;
                natural_params = new_natural_params;
                log_likelihood = new_log_likelihood;

                if converged {
                    break;
                }
            }

            // Keep track of best parameters
            if log_likelihood > best_log_likelihood {
                best_log_likelihood = log_likelihood;
                best_params = Some((weights, natural_params));
                best_n_iter = n_iter;
                best_converged = converged;
            }
        }

        let (weights, natural_params) = best_params.unwrap();

        // Calculate model selection criteria
        let n_params = self.n_parameters(n_features);
        let bic = ModelSelection::bic(best_log_likelihood, n_params, n_samples);
        let aic = ModelSelection::aic(best_log_likelihood, n_params);

        Ok(ExponentialFamilyMixture {
            state: ExponentialFamilyMixtureTrained {
                weights,
                natural_params,
                family_type: self.family_type.clone(),
                log_likelihood: best_log_likelihood,
                n_iter: best_n_iter,
                converged: best_converged,
                bic,
                aic,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            family_type: self.family_type,
            tol: self.tol,
            max_iter: self.max_iter,
            n_init: self.n_init,
            random_state: self.random_state,
        })
    }
}

/// Trained state for ExponentialFamilyMixture
#[derive(Debug, Clone)]
pub struct ExponentialFamilyMixtureTrained {
    /// Mixture component weights
    pub weights: Array1<f64>,
    /// Natural parameters for each component
    pub natural_params: Array2<f64>,
    /// Type of exponential family distribution
    pub family_type: ExponentialFamilyType,
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
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for ExponentialFamilyMixture<ExponentialFamilyMixtureTrained>
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
                let log_weight = self.state.weights[k].ln();
                let log_pdf = self.log_probability(&sample, &self.state.natural_params.row(k))?;
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

impl ExponentialFamilyMixture<ExponentialFamilyMixtureTrained> {
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
                let log_pdf = self.log_probability(&sample, &self.state.natural_params.row(k))?;
                log_probs.push(log_weight + log_pdf);
            }

            let log_sum_exp = self.log_sum_exp(&log_probs);

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
                let log_pdf = self.log_probability(&sample, &self.state.natural_params.row(k))?;
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

    /// Convert natural parameters to mean parameters
    pub fn mean_parameters(&self) -> SklResult<Array2<f64>> {
        let mut mean_params = Array2::zeros(self.state.natural_params.dim());

        for k in 0..self.n_components {
            match &self.state.family_type {
                ExponentialFamilyType::Poisson => {
                    mean_params[[k, 0]] = self.state.natural_params[[k, 0]].exp();
                }
                ExponentialFamilyType::Exponential => {
                    mean_params[[k, 0]] = 1.0 / (-self.state.natural_params[[k, 0]]);
                }
                ExponentialFamilyType::Gamma => {
                    let alpha = self.state.natural_params[[k, 0]] + 1.0;
                    let beta = -self.state.natural_params[[k, 1]];
                    mean_params[[k, 0]] = alpha;
                    mean_params[[k, 1]] = beta;
                }
                ExponentialFamilyType::Bernoulli => {
                    let logit = self.state.natural_params[[k, 0]];
                    mean_params[[k, 0]] = 1.0 / (1.0 + (-logit).exp());
                }
                ExponentialFamilyType::Multinomial(n_categories) => {
                    let mut probs = Array1::zeros(*n_categories);
                    let mut sum_exp = 1.0; // Last category probability (baseline)

                    for j in 0..(n_categories - 1) {
                        probs[j] = self.state.natural_params[[k, j]].exp();
                        sum_exp += probs[j];
                    }
                    probs[n_categories - 1] = 1.0;

                    // Normalize
                    for j in 0..*n_categories {
                        mean_params[[k, j]] = probs[j] / sum_exp;
                    }
                }
            }
        }

        Ok(mean_params)
    }

    /// Shared helper methods (duplicated from untrained implementation)
    fn log_probability(
        &self,
        x: &ArrayView1<f64>,
        natural_params: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        match &self.state.family_type {
            ExponentialFamilyType::Poisson => {
                let lambda = natural_params[0].exp();
                let x_val = x[0];
                Ok(x_val * natural_params[0] - lambda - self.log_factorial(x_val as u32))
            }
            ExponentialFamilyType::Exponential => {
                let rate = -natural_params[0];
                let x_val = x[0];
                Ok(natural_params[0] * x_val + rate.ln())
            }
            ExponentialFamilyType::Gamma => {
                let alpha = natural_params[0] + 1.0;
                let beta = -natural_params[1];
                let x_val = x[0];
                Ok(
                    natural_params[0] * x_val.ln() + natural_params[1] * x_val + alpha * beta.ln()
                        - self.log_gamma(alpha),
                )
            }
            ExponentialFamilyType::Bernoulli => {
                let x_val = x[0];
                let log_odds = natural_params[0];
                Ok(x_val * log_odds - (1.0 + log_odds.exp()).ln())
            }
            ExponentialFamilyType::Multinomial(n_categories) => {
                let mut log_prob = 0.0;
                let n_trials: f64 = x.iter().sum();

                // Add multinomial coefficient (log factorial terms)
                log_prob += self.log_factorial(n_trials as u32);
                for &count in x.iter() {
                    log_prob -= self.log_factorial(count as u32);
                }

                // Add probability terms
                for j in 0..(n_categories - 1) {
                    log_prob += x[j] * natural_params[j];
                }

                // Normalization term
                let log_sum: f64 = natural_params.iter().map(|&p| p.exp()).sum::<f64>() + 1.0;
                log_prob -= n_trials * log_sum.ln();

                Ok(log_prob)
            }
        }
    }

    fn log_factorial(&self, n: u32) -> f64 {
        if n <= 1 {
            0.0
        } else {
            (2..=n).map(|i| (i as f64).ln()).sum()
        }
    }

    fn log_gamma(&self, x: f64) -> f64 {
        if x < 12.0 {
            if x < 0.5 {
                (PI / ((PI * x).sin())).ln() - self.log_gamma(1.0 - x)
            } else {
                self.log_gamma(x + 1.0) - x.ln()
            }
        } else {
            0.5 * (2.0 * PI).ln() - 0.5 * x.ln() + x * x.ln() - x
        }
    }

    fn log_sum_exp(&self, log_probs: &[f64]) -> f64 {
        let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_log_prob.is_infinite() {
            max_log_prob
        } else {
            max_log_prob
                + log_probs
                    .iter()
                    .map(|&lp| (lp - max_log_prob).exp())
                    .sum::<f64>()
                    .ln()
        }
    }
}
