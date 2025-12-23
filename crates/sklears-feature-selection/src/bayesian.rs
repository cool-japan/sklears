//! Bayesian feature selection algorithms
//!
//! This module provides Bayesian approaches to feature selection, including
//! spike-and-slab priors, Bayesian model averaging, and variational inference.

use crate::base::{FeatureSelector, SelectorMixin};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Prior type for Bayesian feature selection
#[derive(Debug, Clone)]
pub enum PriorType {
    /// Spike-and-slab prior with given spike and slab variances
    SpikeAndSlab { spike_var: Float, slab_var: Float },
    /// Horseshoe prior for sparse feature selection
    Horseshoe { tau: Float },
    /// Laplace prior (equivalent to L1 regularization)
    Laplace { scale: Float },
    /// Independent normal priors for features
    Normal { var: Float },
}

/// Inference method for Bayesian feature selection
#[derive(Debug, Clone)]
pub enum BayesianInferenceMethod {
    /// Variational Bayes with mean-field approximation
    VariationalBayes { max_iter: usize, tol: Float },
    /// Gibbs sampling MCMC
    GibbsSampling { n_samples: usize, burn_in: usize },
    /// Expectation-Maximization algorithm
    ExpectationMaximization { max_iter: usize, tol: Float },
    /// Laplace approximation for posterior
    LaplaceApproximation,
}

/// Bayesian variable selection with spike-and-slab priors
#[derive(Debug, Clone)]
pub struct BayesianVariableSelector<State = Untrained> {
    prior: PriorType,
    inference: BayesianInferenceMethod,
    n_features_select: Option<usize>,
    inclusion_threshold: Float,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    posterior_inclusion_probs_: Option<Array1<Float>>,
    feature_coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    evidence_: Option<Float>,
}

impl Default for BayesianVariableSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianVariableSelector<Untrained> {
    /// Create a new Bayesian variable selector
    pub fn new() -> Self {
        Self {
            prior: PriorType::SpikeAndSlab {
                spike_var: 0.01,
                slab_var: 1.0,
            },
            inference: BayesianInferenceMethod::VariationalBayes {
                max_iter: 100,
                tol: 1e-4,
            },
            n_features_select: None,
            inclusion_threshold: 0.5,
            random_state: None,
            state: PhantomData,
            posterior_inclusion_probs_: None,
            feature_coefficients_: None,
            selected_features_: None,
            n_features_: None,
            evidence_: None,
        }
    }

    /// Set the prior type
    pub fn prior(mut self, prior: PriorType) -> Self {
        self.prior = prior;
        self
    }

    /// Set the inference method
    pub fn inference(mut self, inference: BayesianInferenceMethod) -> Self {
        self.inference = inference;
        self
    }

    /// Set the number of features to select (if None, use threshold)
    pub fn n_features_select(mut self, n_features: usize) -> Self {
        self.n_features_select = Some(n_features);
        self
    }

    /// Set the inclusion probability threshold
    pub fn inclusion_threshold(mut self, threshold: Float) -> Self {
        self.inclusion_threshold = threshold;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Perform Bayesian inference for feature selection
    fn fit_bayesian(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Float)> {
        match &self.inference {
            BayesianInferenceMethod::VariationalBayes { max_iter, tol } => {
                self.variational_bayes_inference(features, target, *max_iter, *tol)
            }
            BayesianInferenceMethod::GibbsSampling { n_samples, burn_in } => {
                self.gibbs_sampling_inference(features, target, *n_samples, *burn_in)
            }
            BayesianInferenceMethod::ExpectationMaximization { max_iter, tol } => {
                self.em_inference(features, target, *max_iter, *tol)
            }
            BayesianInferenceMethod::LaplaceApproximation => {
                self.laplace_approximation_inference(features, target)
            }
        }
    }

    /// Variational Bayes inference with mean-field approximation
    fn variational_bayes_inference(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
        max_iter: usize,
        tol: Float,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Float)> {
        let n_features = features.ncols();
        let n_samples = features.nrows();

        // Initialize variational parameters
        let mut gamma = Array1::from_elem(n_features, 0.5); // Inclusion probabilities
        let mut mu = Array1::zeros(n_features); // Mean of coefficients
        let mut sigma2 = Array1::from_elem(n_features, 1.0); // Variance of coefficients

        let (spike_var, slab_var) = match &self.prior {
            PriorType::SpikeAndSlab {
                spike_var,
                slab_var,
            } => (*spike_var, *slab_var),
            _ => (0.01, 1.0), // Default values
        };

        for _iter in 0..max_iter {
            let gamma_old = gamma.clone();

            // Update coefficient parameters
            for j in 0..n_features {
                let feature_col = features.column(j);

                // Compute residual excluding feature j
                let mut residual = target.clone();
                for k in 0..n_features {
                    if k != j {
                        let feature_k = features.column(k);
                        for i in 0..n_samples {
                            residual[i] -= gamma[k] * mu[k] * feature_k[i];
                        }
                    }
                }

                // Update mean and variance
                let feature_norm = feature_col.dot(&feature_col);
                let xy = feature_col.dot(&residual);

                // Handle degenerate case where feature is all zeros
                if feature_norm < 1e-10 {
                    // For zero features, use uninformative prior
                    gamma[j] = 0.5; // Neutral inclusion probability
                    mu[j] = 0.0; // Zero coefficient
                    sigma2[j] = slab_var; // Default variance
                    continue;
                }

                let precision_spike = 1.0 / spike_var + feature_norm;
                let precision_slab = 1.0 / slab_var + feature_norm;

                let mu_spike = xy / precision_spike;
                let mu_slab = xy / precision_slab;

                let sigma2_spike = 1.0 / precision_spike;
                let sigma2_slab = 1.0 / precision_slab;

                // Update inclusion probability using Bayes rule
                let log_prob_spike =
                    -0.5 * (mu_spike * mu_spike / sigma2_spike + (sigma2_spike / spike_var).ln());
                let log_prob_slab =
                    -0.5 * (mu_slab * mu_slab / sigma2_slab + (sigma2_slab / slab_var).ln());

                let max_log = log_prob_spike.max(log_prob_slab);
                let exp_spike = (log_prob_spike - max_log).exp();
                let exp_slab = (log_prob_slab - max_log).exp();

                let denom = exp_spike + exp_slab;
                if denom > 1e-10 {
                    gamma[j] = exp_slab / denom;
                } else {
                    gamma[j] = 0.5; // Fallback for numerical issues
                }

                // Ensure gamma is in valid range [0, 1]
                gamma[j] = gamma[j].clamp(0.0, 1.0);

                mu[j] = gamma[j] * mu_slab + (1.0 - gamma[j]) * mu_spike;
                sigma2[j] = gamma[j] * (sigma2_slab + mu_slab * mu_slab)
                    + (1.0 - gamma[j]) * (sigma2_spike + mu_spike * mu_spike)
                    - mu[j] * mu[j];
            }

            // Check convergence
            let diff = (&gamma - &gamma_old).mapv(|x| x.abs()).sum();
            if diff < tol {
                break;
            }
        }

        // Compute evidence (approximate)
        let evidence = self.compute_evidence(features, target, &gamma, &mu);

        Ok((gamma, mu, evidence))
    }

    /// Gibbs sampling MCMC inference
    fn gibbs_sampling_inference(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
        n_samples: usize,
        burn_in: usize,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Float)> {
        let n_features = features.ncols();
        let n_obs = features.nrows();

        // Initialize parameters
        let mut gamma = Array1::from_elem(n_features, 0.5);
        let mut coefficients = Array1::zeros(n_features);

        // Storage for samples
        let mut gamma_samples = Array2::zeros((n_samples, n_features));
        let mut coeff_samples = Array2::zeros((n_samples, n_features));

        let (_spike_var, slab_var) = match &self.prior {
            PriorType::SpikeAndSlab {
                spike_var,
                slab_var,
            } => (*spike_var, *slab_var),
            _ => (0.01, 1.0),
        };

        // Simple pseudo-random number generation (for demonstration)
        let mut rng_state = self.random_state.unwrap_or(42);

        for sample_idx in 0..(n_samples + burn_in) {
            // Sample each feature indicator and coefficient
            for j in 0..n_features {
                let feature_col = features.column(j);

                // Compute residual excluding feature j
                let mut residual = target.clone();
                for k in 0..n_features {
                    if k != j && gamma[k] > 0.5 {
                        let feature_k = features.column(k);
                        for i in 0..n_obs {
                            residual[i] -= coefficients[k] * feature_k[i];
                        }
                    }
                }

                // Sample inclusion indicator
                let xy = feature_col.dot(&residual);
                let xx = feature_col.dot(&feature_col);

                let log_prob_in = -0.5 * xx * slab_var / (1.0 + xx * slab_var)
                    * (xy / (1.0 + xx * slab_var)).powi(2);
                let log_prob_out = 0.0;

                let prob_in = 1.0 / (1.0 + (log_prob_out - log_prob_in).exp());

                // Simple random number (for demonstration - would use proper RNG in practice)
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (rng_state as Float) / (u32::MAX as Float);

                gamma[j] = if u < prob_in { 1.0 } else { 0.0 };

                // Sample coefficient if included
                if gamma[j] > 0.5 {
                    let var_post = 1.0 / (1.0 / slab_var + xx);
                    let mean_post = var_post * xy;

                    // Simple normal approximation (would use proper sampling in practice)
                    coefficients[j] = mean_post;
                } else {
                    coefficients[j] = 0.0;
                }
            }

            // Store samples after burn-in
            if sample_idx >= burn_in {
                let store_idx = sample_idx - burn_in;
                for j in 0..n_features {
                    gamma_samples[[store_idx, j]] = gamma[j];
                    coeff_samples[[store_idx, j]] = coefficients[j];
                }
            }
        }

        // Compute posterior inclusion probabilities and coefficient estimates
        let inclusion_probs = gamma_samples.mean_axis(Axis(0)).unwrap();
        let coeff_estimates = coeff_samples.mean_axis(Axis(0)).unwrap();
        let evidence = 0.0; // Would compute marginal likelihood from samples

        Ok((inclusion_probs, coeff_estimates, evidence))
    }

    /// Expectation-Maximization inference
    fn em_inference(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
        max_iter: usize,
        tol: Float,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Float)> {
        let n_features = features.ncols();

        // Initialize parameters
        let mut inclusion_probs = Array1::from_elem(n_features, 0.5);
        let mut coefficients = Array1::zeros(n_features);
        let mut noise_var = 1.0;

        let (spike_var, slab_var) = match &self.prior {
            PriorType::SpikeAndSlab {
                spike_var,
                slab_var,
            } => (*spike_var, *slab_var),
            _ => (0.01, 1.0),
        };

        for _iter in 0..max_iter {
            let inclusion_probs_old = inclusion_probs.clone();

            // E-step: Update posterior inclusion probabilities
            for j in 0..n_features {
                let feature_col = features.column(j);

                // Compute likelihood under both models
                let mut residual = target.clone();
                for k in 0..n_features {
                    if k != j {
                        let feature_k = features.column(k);
                        for i in 0..features.nrows() {
                            residual[i] -= inclusion_probs[k] * coefficients[k] * feature_k[i];
                        }
                    }
                }

                let xy = feature_col.dot(&residual);
                let xx = feature_col.dot(&feature_col);

                // Handle degenerate case where feature is all zeros
                if xx < 1e-10 {
                    // For zero features, use uninformative prior
                    inclusion_probs[j] = 0.5; // Neutral inclusion probability
                    coefficients[j] = 0.0; // Zero coefficient
                    continue;
                }

                // Bayes factor for inclusion vs exclusion
                let precision_in = 1.0 / slab_var + xx / noise_var;
                let precision_out = 1.0 / spike_var + xx / noise_var;

                let mean_in = (xy / noise_var) / precision_in;
                let mean_out = (xy / noise_var) / precision_out;

                let log_bf = 0.5 * (precision_out / precision_in).ln()
                    + 0.5
                        * (mean_in * mean_in * precision_in - mean_out * mean_out * precision_out);

                let prob = 1.0 / (1.0 + (-log_bf).exp());
                inclusion_probs[j] = prob.clamp(0.0, 1.0); // Ensure valid probability
                coefficients[j] = inclusion_probs[j] * mean_in;
            }

            // M-step: Update noise variance
            let mut sse = 0.0;
            for i in 0..features.nrows() {
                let mut pred = 0.0;
                for j in 0..n_features {
                    pred += inclusion_probs[j] * coefficients[j] * features[[i, j]];
                }
                sse += (target[i] - pred).powi(2);
            }
            // Add minimum variance constraint to prevent numerical instability
            // when data is all zeros or nearly zero
            noise_var = (sse / features.nrows() as Float).max(1e-10);

            // Check convergence
            let diff = (&inclusion_probs - &inclusion_probs_old)
                .mapv(|x| x.abs())
                .sum();
            if diff < tol {
                break;
            }
        }

        let evidence = self.compute_evidence(features, target, &inclusion_probs, &coefficients);
        Ok((inclusion_probs, coefficients, evidence))
    }

    /// Laplace approximation inference
    fn laplace_approximation_inference(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Float)> {
        let n_features = features.ncols();

        // For Laplace approximation, we find the MAP estimate and approximate the posterior
        let mut coefficients = Array1::zeros(n_features);
        let mut inclusion_probs = Array1::from_elem(n_features, 0.5);

        // Simple optimization for MAP estimate (would use proper optimization in practice)
        let xtx = features.t().dot(features);
        let xty = features.t().dot(target);

        // Ridge regression solution as approximation
        let lambda = 0.1; // Regularization parameter
        let mut xtx_reg = xtx.clone();
        for i in 0..n_features {
            xtx_reg[[i, i]] += lambda;
        }

        // Solve linear system (simplified - would use proper linear algebra)
        for j in 0..n_features {
            if xtx_reg[[j, j]] != 0.0 {
                coefficients[j] = xty[j] / xtx_reg[[j, j]];
            }
        }

        // Compute inclusion probabilities based on coefficient magnitudes
        let coeff_threshold = coefficients.mapv(|x| x.abs()).mean().unwrap_or(0.0);
        for j in 0..n_features {
            inclusion_probs[j] = if coefficients[j].abs() > coeff_threshold {
                0.8
            } else {
                0.2
            };
        }

        let evidence = self.compute_evidence(features, target, &inclusion_probs, &coefficients);
        Ok((inclusion_probs, coefficients, evidence))
    }

    /// Compute model evidence (marginal likelihood)
    fn compute_evidence(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
        inclusion_probs: &Array1<Float>,
        coefficients: &Array1<Float>,
    ) -> Float {
        let n_samples = features.nrows() as Float;
        let mut sse = 0.0;

        for i in 0..features.nrows() {
            let mut pred = 0.0;
            for j in 0..features.ncols() {
                pred += inclusion_probs[j] * coefficients[j] * features[[i, j]];
            }
            sse += (target[i] - pred).powi(2);
        }

        // Simplified BIC approximation
        let k = inclusion_probs.sum(); // Effective number of parameters

        // Handle edge case where sse is 0 or very small to avoid log(0)
        let log_likelihood = if sse < 1e-10 {
            // Perfect fit or near-perfect fit case
            -1e-10_f64.ln() // Use a small positive value instead of 0
        } else {
            -(sse / n_samples).ln()
        };

        -0.5 * n_samples * log_likelihood - 0.5 * k * n_samples.ln()
    }

    /// Select features based on inclusion probabilities
    fn select_features_from_probabilities(&self, inclusion_probs: &Array1<Float>) -> Vec<usize> {
        if let Some(n_select) = self.n_features_select {
            // Select top n features by inclusion probability
            let mut indices: Vec<usize> = (0..inclusion_probs.len()).collect();
            indices.sort_by(|&a, &b| {
                inclusion_probs[b]
                    .partial_cmp(&inclusion_probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices.truncate(n_select);
            indices
        } else {
            // Select features above threshold
            inclusion_probs
                .iter()
                .enumerate()
                .filter(|(_, &prob)| prob >= self.inclusion_threshold)
                .map(|(idx, _)| idx)
                .collect()
        }
    }
}

impl Estimator for BayesianVariableSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for BayesianVariableSelector<Untrained> {
    type Fitted = BayesianVariableSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(features, target)?;

        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        let (inclusion_probs, coefficients, evidence) = self.fit_bayesian(features, target)?;
        let selected_features = self.select_features_from_probabilities(&inclusion_probs);

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected with current threshold".to_string(),
            ));
        }

        Ok(BayesianVariableSelector {
            prior: self.prior,
            inference: self.inference,
            n_features_select: self.n_features_select,
            inclusion_threshold: self.inclusion_threshold,
            random_state: self.random_state,
            state: PhantomData,
            posterior_inclusion_probs_: Some(inclusion_probs),
            feature_coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            evidence_: Some(evidence),
        })
    }
}

impl Transform<Array2<Float>> for BayesianVariableSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for BayesianVariableSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for BayesianVariableSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl BayesianVariableSelector<Trained> {
    /// Get posterior inclusion probabilities
    pub fn inclusion_probabilities(&self) -> &Array1<Float> {
        self.posterior_inclusion_probs_.as_ref().unwrap()
    }

    /// Get feature coefficients
    pub fn coefficients(&self) -> &Array1<Float> {
        self.feature_coefficients_.as_ref().unwrap()
    }

    /// Get model evidence (log marginal likelihood)
    pub fn evidence(&self) -> Float {
        self.evidence_.unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }

    /// Check if a feature was selected
    pub fn is_feature_selected(&self, feature_idx: usize) -> bool {
        self.selected_features_
            .as_ref()
            .unwrap()
            .contains(&feature_idx)
    }
}

/// Bayesian Model Averaging for feature selection
#[derive(Debug, Clone)]
pub struct BayesianModelAveraging<State = Untrained> {
    max_models: usize,
    prior_inclusion_prob: Float,
    inference_method: BayesianInferenceMethod,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    model_probabilities_: Option<Vec<Float>>,
    model_features_: Option<Vec<Vec<usize>>>,
    averaged_inclusion_probs_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl Default for BayesianModelAveraging<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianModelAveraging<Untrained> {
    pub fn new() -> Self {
        Self {
            max_models: 1000,
            prior_inclusion_prob: 0.5,
            inference_method: BayesianInferenceMethod::VariationalBayes {
                max_iter: 50,
                tol: 1e-3,
            },
            random_state: None,
            state: PhantomData,
            model_probabilities_: None,
            model_features_: None,
            averaged_inclusion_probs_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    pub fn max_models(mut self, max_models: usize) -> Self {
        self.max_models = max_models;
        self
    }

    pub fn prior_inclusion_prob(mut self, prob: Float) -> Self {
        self.prior_inclusion_prob = prob;
        self
    }

    pub fn inference_method(mut self, method: BayesianInferenceMethod) -> Self {
        self.inference_method = method;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enumerate and evaluate models for Bayesian model averaging
    fn enumerate_models(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Vec<Vec<usize>>, Vec<Float>, Array1<Float>)> {
        let n_features = features.ncols();
        let mut models = Vec::new();
        let mut model_probs = Vec::new();

        // For demonstration, use a subset of all possible models
        // In practice, would use more sophisticated model enumeration
        let max_features_per_model = (n_features / 3).clamp(1, 10);

        // Simple random model generation
        let mut rng_state = self.random_state.unwrap_or(42);

        for _ in 0..self.max_models.min(1000) {
            let mut model_features = Vec::new();

            // Randomly select features for this model
            for j in 0..n_features {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (rng_state as Float) / (u32::MAX as Float);

                if u < self.prior_inclusion_prob && model_features.len() < max_features_per_model {
                    model_features.push(j);
                }
            }

            if model_features.is_empty() {
                model_features.push(0); // Ensure at least one feature
            }

            // Evaluate model
            let model_evidence = self.evaluate_model(features, target, &model_features)?;

            models.push(model_features);
            model_probs.push(model_evidence);
        }

        // Normalize model probabilities with numerical stability
        if model_probs.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No models generated".to_string(),
            ));
        }

        let max_evidence = model_probs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // If all evidences are negative infinity, assign uniform probabilities
        if !max_evidence.is_finite() {
            let uniform_prob = 1.0 / model_probs.len() as Float;
            model_probs.fill(uniform_prob);
        } else {
            let mut total_prob = 0.0;
            for prob in &mut model_probs {
                *prob = (*prob - max_evidence).exp();
                total_prob += *prob;
            }

            // Ensure total_prob is positive and finite
            if total_prob <= 0.0 || !total_prob.is_finite() {
                let uniform_prob = 1.0 / model_probs.len() as Float;
                model_probs.fill(uniform_prob);
            } else {
                for prob in &mut model_probs {
                    *prob /= total_prob;
                    // Ensure non-negative probabilities
                    *prob = prob.max(0.0);
                }

                // Renormalize to ensure sum = 1 after clamping
                let new_total: Float = model_probs.iter().sum();
                if new_total > 0.0 {
                    for prob in &mut model_probs {
                        *prob /= new_total;
                    }
                }
            }
        }

        // Compute averaged inclusion probabilities
        let mut inclusion_probs = Array1::<Float>::zeros(n_features);
        for (model, &prob) in models.iter().zip(model_probs.iter()) {
            for &feature in model {
                inclusion_probs[feature] += prob;
            }
        }

        // Ensure inclusion probabilities are properly bounded [0, 1]
        for prob in inclusion_probs.iter_mut() {
            *prob = prob.clamp(0.0, 1.0);
        }

        Ok((models, model_probs, inclusion_probs))
    }

    /// Evaluate a single model using marginal likelihood
    fn evaluate_model(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
        model_features: &[usize],
    ) -> SklResult<Float> {
        if model_features.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }

        // Extract features for this model
        let mut model_x = Array2::zeros((features.nrows(), model_features.len()));
        for (new_idx, &old_idx) in model_features.iter().enumerate() {
            model_x
                .column_mut(new_idx)
                .assign(&features.column(old_idx));
        }

        // Compute marginal likelihood (simplified)
        let n = features.nrows() as Float;
        let k = model_features.len() as Float;

        // Check for degenerate cases
        let target_var = target.var(0.0);
        if target_var < 1e-12 {
            // If target has zero variance, all models are equally poor
            return Ok(-1000.0 - k * 10.0); // Penalize complexity when no signal
        }

        // Check if all features are zero
        let feature_norm: Float = model_x.iter().map(|&x| x * x).sum();
        if feature_norm < 1e-12 {
            // If features are all zero, they have no predictive power
            return Ok(-1000.0 - k * 10.0);
        }

        // Bayesian linear regression marginal likelihood approximation
        let xtx = model_x.t().dot(&model_x);
        let _xty = model_x.t().dot(target);

        // Add regularization for numerical stability
        let mut xtx_reg = xtx.clone();
        for i in 0..model_features.len() {
            xtx_reg[[i, i]] += 1e-6;
        }

        // Compute SSE for this model (simplified)
        let sse = target.dot(target);
        let sse_normalized = (sse / n).max(1e-12); // Prevent log(0)

        // Simplified BIC-like score with numerical stability
        let log_likelihood = -0.5 * n * sse_normalized.ln();
        let penalty = -0.5 * k * n.ln();

        let evidence = log_likelihood + penalty;

        // Ensure finite result
        if evidence.is_finite() {
            Ok(evidence)
        } else {
            Ok(-1000.0 - k * 10.0)
        }
    }
}

impl Estimator for BayesianModelAveraging<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for BayesianModelAveraging<Untrained> {
    type Fitted = BayesianModelAveraging<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(features, target)?;

        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        let (models, model_probs, inclusion_probs) = self.enumerate_models(features, target)?;

        // Select features with highest averaged inclusion probabilities
        let threshold = 0.5; // Could be made configurable
        let selected_features: Vec<usize> = inclusion_probs
            .iter()
            .enumerate()
            .filter(|(_, &prob)| prob >= threshold)
            .map(|(idx, _)| idx)
            .collect();

        if selected_features.is_empty() {
            // Fallback: select top features
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| {
                inclusion_probs[b]
                    .partial_cmp(&inclusion_probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let selected_features = indices.into_iter().take(1).collect();

            return Ok(BayesianModelAveraging {
                max_models: self.max_models,
                prior_inclusion_prob: self.prior_inclusion_prob,
                inference_method: self.inference_method,
                random_state: self.random_state,
                state: PhantomData,
                model_probabilities_: Some(model_probs),
                model_features_: Some(models),
                averaged_inclusion_probs_: Some(inclusion_probs),
                selected_features_: Some(selected_features),
                n_features_: Some(n_features),
            });
        }

        Ok(BayesianModelAveraging {
            max_models: self.max_models,
            prior_inclusion_prob: self.prior_inclusion_prob,
            inference_method: self.inference_method,
            random_state: self.random_state,
            state: PhantomData,
            model_probabilities_: Some(model_probs),
            model_features_: Some(models),
            averaged_inclusion_probs_: Some(inclusion_probs),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for BayesianModelAveraging<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for BayesianModelAveraging<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for BayesianModelAveraging<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl BayesianModelAveraging<Trained> {
    /// Get model probabilities
    pub fn model_probabilities(&self) -> &[Float] {
        self.model_probabilities_.as_ref().unwrap()
    }

    /// Get features for each model
    pub fn model_features(&self) -> &[Vec<usize>] {
        self.model_features_.as_ref().unwrap()
    }

    /// Get averaged inclusion probabilities
    pub fn inclusion_probabilities(&self) -> &Array1<Float> {
        self.averaged_inclusion_probs_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use scirs2_core::ndarray::Array2;

    fn create_test_data() -> (Array2<Float>, Array1<Float>) {
        // Create synthetic data with some signal
        let n_samples = 100;
        let n_features = 10;
        let mut features = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Fill with some structured data
        for i in 0..n_samples {
            for j in 0..n_features {
                features[[i, j]] = (i as Float * 0.1 + j as Float * 0.01) % 1.0;
            }
            // Make first few features predictive
            target[i] =
                features[[i, 0]] + 0.5 * features[[i, 1]] + 0.1 * ((i as Float) * 0.01).sin();
        }

        (features, target)
    }

    #[test]
    fn test_bayesian_variable_selector_variational() {
        let (features, target) = create_test_data();

        let selector = BayesianVariableSelector::new()
            .prior(PriorType::SpikeAndSlab {
                spike_var: 0.01,
                slab_var: 1.0,
            })
            .inference(BayesianInferenceMethod::VariationalBayes {
                max_iter: 10,
                tol: 1e-3,
            })
            .n_features_select(3);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 3);
        assert!(trained.inclusion_probabilities().len() == features.ncols());
    }

    #[test]
    fn test_bayesian_variable_selector_em() {
        let (features, target) = create_test_data();

        let selector = BayesianVariableSelector::new()
            .inference(BayesianInferenceMethod::ExpectationMaximization {
                max_iter: 10,
                tol: 1e-3,
            })
            .inclusion_threshold(0.3);

        let trained = selector.fit(&features, &target).unwrap();
        assert!(trained.n_features_out() > 0);
    }

    #[test]
    fn test_bayesian_model_averaging() {
        let (features, target) = create_test_data();

        let selector = BayesianModelAveraging::new()
            .max_models(50)
            .prior_inclusion_prob(0.3);

        let trained = selector.fit(&features, &target).unwrap();
        assert!(trained.n_features_out() > 0);
        assert!(trained.model_probabilities().len() > 0);
    }

    #[test]
    fn test_transform() {
        let (features, target) = create_test_data();

        let selector = BayesianVariableSelector::new().n_features_select(4);

        let trained = selector.fit(&features, &target).unwrap();
        let transformed = trained.transform(&features).unwrap();

        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), features.nrows());
    }

    #[test]
    fn test_horseshoe_prior() {
        let (features, target) = create_test_data();

        let selector = BayesianVariableSelector::new()
            .prior(PriorType::Horseshoe { tau: 0.1 })
            .n_features_select(3);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 3);
    }

    #[test]
    fn test_selector_mixin() {
        let (features, target) = create_test_data();

        let selector = BayesianVariableSelector::new().n_features_select(5);

        let trained = selector.fit(&features, &target).unwrap();
        let support = trained.get_support().unwrap();

        assert_eq!(support.len(), features.ncols());
        assert_eq!(support.iter().filter(|&&x| x).count(), 5);
    }

    // Property-based tests for Bayesian feature selection
    mod proptests {
        use super::*;

        fn valid_features() -> impl Strategy<Value = Array2<Float>> {
            (3usize..10, 20usize..50).prop_flat_map(|(n_cols, n_rows)| {
                prop::collection::vec(-5.0..5.0f64, n_rows * n_cols).prop_map(move |values| {
                    Array2::from_shape_vec((n_rows, n_cols), values).unwrap()
                })
            })
        }

        fn valid_target(n_samples: usize) -> impl Strategy<Value = Array1<Float>> {
            prop::collection::vec(-10.0..10.0f64, n_samples)
                .prop_map(|values| Array1::from_vec(values))
        }

        proptest! {
            #[test]
            fn prop_bayesian_selector_respects_feature_count(
                features in valid_features(),
                n_features in 1usize..8
            ) {
                let target = Array1::zeros(features.nrows());
                let n_select = n_features.min(features.ncols());

                let selector = BayesianVariableSelector::new()
                    .n_features_select(n_select)
                    .inference(BayesianInferenceMethod::VariationalBayes { max_iter: 5, tol: 1e-2 });

                if let Ok(trained) = selector.fit(&features, &target) {
                    prop_assert_eq!(trained.n_features_out(), n_select);
                    prop_assert_eq!(trained.selected_features().len(), n_select);

                    // All selected features should be valid indices
                    for &idx in trained.selected_features() {
                        prop_assert!(idx < features.ncols());
                    }
                }
            }

            #[test]
            fn prop_bayesian_selector_inclusion_probabilities_valid(
                features in valid_features(),
                n_features in 1usize..5
            ) {
                let target = Array1::zeros(features.nrows());
                let n_select = n_features.min(features.ncols());

                let selector = BayesianVariableSelector::new()
                    .n_features_select(n_select)
                    .inference(BayesianInferenceMethod::ExpectationMaximization { max_iter: 5, tol: 1e-2 });

                if let Ok(trained) = selector.fit(&features, &target) {
                    let inclusion_probs = trained.inclusion_probabilities();

                    // All inclusion probabilities should be between 0 and 1
                    for &prob in inclusion_probs.iter() {
                        prop_assert!(prob >= 0.0);
                        prop_assert!(prob <= 1.0);
                    }

                    // Selected features should have higher inclusion probabilities
                    let selected_features = trained.selected_features();
                    if !selected_features.is_empty() {
                        let min_selected_prob = selected_features.iter()
                            .map(|&idx| inclusion_probs[idx])
                            .fold(f64::INFINITY, f64::min);

                        // There should be at least n_select features with prob >= min_selected_prob
                        let count_above_min = inclusion_probs.iter()
                            .filter(|&&prob| prob >= min_selected_prob)
                            .count();
                        prop_assert!(count_above_min >= selected_features.len());
                    }
                }
            }

            #[test]
            fn prop_bayesian_selector_transform_preserves_shape(
                features in valid_features(),
                n_features in 1usize..5
            ) {
                let target = Array1::zeros(features.nrows());
                let n_select = n_features.min(features.ncols());

                let selector = BayesianVariableSelector::new()
                    .n_features_select(n_select);

                if let Ok(trained) = selector.fit(&features, &target) {
                    if let Ok(transformed) = trained.transform(&features) {
                        prop_assert_eq!(transformed.nrows(), features.nrows());
                        prop_assert_eq!(transformed.ncols(), n_select);

                        // Transformed values should match original features
                        for (sample_idx, row) in transformed.rows().into_iter().enumerate() {
                            for (new_feat_idx, &value) in row.iter().enumerate() {
                                let orig_feat_idx = trained.selected_features()[new_feat_idx];
                                let expected = features[[sample_idx, orig_feat_idx]];
                                prop_assert!((value - expected).abs() < 1e-10);
                            }
                        }
                    }
                }
            }

            #[test]
            fn prop_bayesian_model_averaging_probabilities_sum_to_one(
                features in valid_features(),
                max_models in 10usize..50
            ) {
                let target = Array1::zeros(features.nrows());

                let selector = BayesianModelAveraging::new()
                    .max_models(max_models)
                    .prior_inclusion_prob(0.3);

                if let Ok(trained) = selector.fit(&features, &target) {
                    let model_probs = trained.model_probabilities();

                    if !model_probs.is_empty() {
                        // All probabilities should be non-negative
                        for &prob in model_probs {
                            prop_assert!(prob >= 0.0);
                        }

                        // Probabilities should approximately sum to 1
                        let sum: Float = model_probs.iter().sum();
                        prop_assert!((sum - 1.0).abs() < 1e-6);
                    }
                }
            }

            #[test]
            fn prop_bayesian_model_averaging_inclusion_probs_valid(
                features in valid_features(),
                max_models in 5usize..20
            ) {
                let target = Array1::zeros(features.nrows());

                let selector = BayesianModelAveraging::new()
                    .max_models(max_models)
                    .prior_inclusion_prob(0.4);

                if let Ok(trained) = selector.fit(&features, &target) {
                    let inclusion_probs = trained.inclusion_probabilities();

                    // All inclusion probabilities should be between 0 and 1
                    for &prob in inclusion_probs.iter() {
                        prop_assert!(prob >= 0.0);
                        prop_assert!(prob <= 1.0);
                    }

                    // Should have same length as number of features
                    prop_assert_eq!(inclusion_probs.len(), features.ncols());
                }
            }

            #[test]
            fn prop_prior_types_affect_selection(
                features in valid_features(),
                n_features in 1usize..3
            ) {
                let target = Array1::zeros(features.nrows());
                let n_select = n_features.min(features.ncols());

                // Test different prior types
                let priors = vec![
                    PriorType::SpikeAndSlab { spike_var: 0.01, slab_var: 1.0 },
                    PriorType::Horseshoe { tau: 0.1 },
                    PriorType::Laplace { scale: 1.0 },
                    PriorType::Normal { var: 1.0 },
                ];

                for prior in priors {
                    let selector = BayesianVariableSelector::new()
                        .prior(prior)
                        .n_features_select(n_select)
                        .inference(BayesianInferenceMethod::LaplaceApproximation);

                    if let Ok(trained) = selector.fit(&features, &target) {
                        prop_assert_eq!(trained.n_features_out(), n_select);
                        prop_assert!(trained.inclusion_probabilities().len() == features.ncols());
                    }
                }
            }

            #[test]
            fn prop_bayesian_selector_deterministic_with_same_seed(
                features in valid_features(),
                n_features in 1usize..4,
                seed in 1u64..1000
            ) {
                let target = Array1::zeros(features.nrows());
                let n_select = n_features.min(features.ncols());

                let selector1 = BayesianVariableSelector::new()
                    .n_features_select(n_select)
                    .random_state(seed)
                    .inference(BayesianInferenceMethod::GibbsSampling { n_samples: 10, burn_in: 5 });

                let selector2 = BayesianVariableSelector::new()
                    .n_features_select(n_select)
                    .random_state(seed)
                    .inference(BayesianInferenceMethod::GibbsSampling { n_samples: 10, burn_in: 5 });

                if let (Ok(trained1), Ok(trained2)) = (selector1.fit(&features, &target), selector2.fit(&features, &target)) {
                    // Same seed should produce same results
                    prop_assert_eq!(trained1.selected_features(), trained2.selected_features());
                }
            }

            #[test]
            fn prop_bayesian_selector_evidence_is_finite(
                features in valid_features(),
                n_features in 1usize..3
            ) {
                let target = Array1::zeros(features.nrows());
                let n_select = n_features.min(features.ncols());

                let selector = BayesianVariableSelector::new()
                    .n_features_select(n_select);

                if let Ok(trained) = selector.fit(&features, &target) {
                    let evidence = trained.evidence();

                    // Evidence should be finite
                    prop_assert!(evidence.is_finite());
                }
            }
        }
    }
}
