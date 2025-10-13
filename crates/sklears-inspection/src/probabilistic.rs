//! Probabilistic Explanations Module
//!
//! This module provides probabilistic approaches to model interpretability, including
//! Bayesian explanations, probabilistic counterfactuals, uncertainty quantification in explanations,
//! credible intervals, and Bayesian model averaging for explanation generation.

use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sklears_core::{error::SklearsError, types::Float};

/// Configuration for probabilistic explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticConfig {
    /// Number of Monte Carlo samples for Bayesian inference
    pub n_samples: usize,
    /// Confidence level for credible intervals (e.g., 0.95 for 95% intervals)
    pub confidence_level: Float,
    /// Prior distribution parameters
    pub prior_mean: Float,
    /// prior_variance
    pub prior_variance: Float,
    /// Whether to use variational inference (faster) or MCMC (more accurate)
    pub use_variational_inference: bool,
    /// Number of burn-in samples for MCMC
    pub burnin_samples: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Number of chains for MCMC
    pub n_chains: usize,
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            confidence_level: 0.95,
            prior_mean: 0.0,
            prior_variance: 1.0,
            use_variational_inference: false,
            burnin_samples: 200,
            random_seed: None,
            n_chains: 4,
        }
    }
}

/// Bayesian explanation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianExplanationResult {
    /// Posterior means of feature importance
    pub posterior_means: Array1<Float>,
    /// Posterior standard deviations
    pub posterior_stds: Array1<Float>,
    /// Credible intervals for feature importance
    pub credible_intervals: Vec<(Float, Float)>,
    /// Posterior samples for each feature
    pub posterior_samples: Array2<Float>,
    /// Model evidence (marginal likelihood)
    pub model_evidence: Float,
    /// Effective sample size for MCMC diagnostics
    pub effective_sample_size: Array1<Float>,
    /// R-hat convergence diagnostic
    pub r_hat: Array1<Float>,
}

/// Probabilistic counterfactual result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticCounterfactualResult {
    /// Most likely counterfactual example
    pub most_likely_counterfactual: Array1<Float>,
    /// Probability distribution over counterfactuals
    pub counterfactual_distribution: Array2<Float>,
    /// Uncertainty in counterfactual generation
    pub generation_uncertainty: Float,
    /// Credible region for counterfactuals
    pub credible_region: Array2<Float>,
    /// Probability of feasibility for each counterfactual
    pub feasibility_probabilities: Array1<Float>,
}

/// Explanation with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertainExplanationResult {
    /// Feature importance estimates
    pub importance_estimates: Array1<Float>,
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Array1<Float>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: Array1<Float>,
    /// Total uncertainty
    pub total_uncertainty: Array1<Float>,
    /// Confidence intervals for importance
    pub importance_intervals: Vec<(Float, Float)>,
    /// Reliability scores for each explanation
    pub reliability_scores: Array1<Float>,
}

/// Bayesian model averaging result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianModelAveragingResult {
    /// Averaged feature importance across models
    pub averaged_importance: Array1<Float>,
    /// Model weights based on posterior probabilities
    pub model_weights: Array1<Float>,
    /// Individual model importances
    pub individual_importances: Array2<Float>,
    /// Model posterior probabilities
    pub model_posteriors: Array1<Float>,
    /// Uncertainty due to model selection
    pub model_selection_uncertainty: Array1<Float>,
}

/// Generate Bayesian explanations using posterior inference
pub fn generate_bayesian_explanation<F>(
    explain_fn: F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &ProbabilisticConfig,
) -> SklResult<BayesianExplanationResult>
where
    F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(config.random_seed.unwrap_or(42));
    let n_features = X.ncols();

    // Storage for posterior samples
    let mut posterior_samples = Array2::zeros((config.n_samples, n_features));
    let mut log_likelihoods = Array1::zeros(config.n_samples);

    if config.use_variational_inference {
        // Variational inference approach (faster approximation)
        generate_variational_samples(
            &explain_fn,
            X,
            y,
            config,
            &mut posterior_samples,
            &mut log_likelihoods,
            &mut rng,
        )?;
    } else {
        // MCMC sampling approach (more accurate)
        generate_mcmc_samples(
            &explain_fn,
            X,
            y,
            config,
            &mut posterior_samples,
            &mut log_likelihoods,
            &mut rng,
        )?;
    }

    // Compute posterior statistics
    let posterior_means = posterior_samples.mean_axis(Axis(0)).unwrap();
    let posterior_stds = compute_std_axis(&posterior_samples, Axis(0));

    // Compute credible intervals
    let credible_intervals =
        compute_credible_intervals(&posterior_samples, config.confidence_level);

    // Compute model evidence (marginal likelihood)
    let model_evidence = compute_model_evidence(&log_likelihoods);

    // MCMC diagnostics
    let effective_sample_size = compute_effective_sample_size(&posterior_samples);
    let r_hat = compute_r_hat_diagnostic(&posterior_samples, config.n_chains);

    Ok(BayesianExplanationResult {
        posterior_means,
        posterior_stds,
        credible_intervals,
        posterior_samples,
        model_evidence,
        effective_sample_size,
        r_hat,
    })
}

/// Generate probabilistic counterfactuals with uncertainty
pub fn generate_probabilistic_counterfactuals<F>(
    model: F,
    instance: &ArrayView1<Float>,
    target_class: usize,
    config: &ProbabilisticConfig,
) -> SklResult<ProbabilisticCounterfactualResult>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array2<Float>>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(config.random_seed.unwrap_or(42));
    let n_features = instance.len();

    // Generate ensemble of counterfactuals using variational approach
    let mut counterfactuals = Array2::zeros((config.n_samples, n_features));
    let mut probabilities = Array1::zeros(config.n_samples);

    for i in 0..config.n_samples {
        // Sample from posterior distribution of counterfactuals
        let mut candidate = instance.to_owned();

        // Perform optimization with noise injection
        for _ in 0..50 {
            let noise = generate_gaussian_noise(&mut rng, n_features, 0.0, 0.1);
            let perturbed = &candidate + &noise;

            // Evaluate probability of achieving target class
            let input_batch = perturbed.clone().insert_axis(Axis(0));
            let predictions = model(&input_batch.view())?;
            let target_prob = predictions[[0, target_class]];

            if target_prob > probabilities[i] {
                probabilities[i] = target_prob;
                candidate = perturbed;
            }
        }

        counterfactuals.row_mut(i).assign(&candidate);
    }

    // Find most likely counterfactual
    let best_idx = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    let most_likely_counterfactual = counterfactuals.row(best_idx).to_owned();

    // Compute generation uncertainty
    let generation_uncertainty = probabilities.var(0.0);

    // Compute credible region
    let credible_region = compute_credible_region(&counterfactuals, config.confidence_level);

    // Compute feasibility probabilities
    let feasibility_probabilities = compute_feasibility_probabilities(&counterfactuals, instance);

    Ok(ProbabilisticCounterfactualResult {
        most_likely_counterfactual,
        counterfactual_distribution: counterfactuals,
        generation_uncertainty,
        credible_region,
        feasibility_probabilities,
    })
}

/// Quantify uncertainty in explanations
#[allow(non_snake_case)]
pub fn quantify_explanation_uncertainty<F>(
    explain_fn: F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &ProbabilisticConfig,
) -> SklResult<UncertainExplanationResult>
where
    F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(config.random_seed.unwrap_or(42));
    let n_features = X.ncols();
    let n_samples = X.nrows();

    // Generate bootstrap samples to estimate epistemic uncertainty
    let mut epistemic_samples = Array2::zeros((config.n_samples, n_features));

    for i in 0..config.n_samples {
        // Bootstrap sampling
        let indices: Vec<usize> = (0..n_samples)
            .map(|_| rng.gen_range(0..n_samples))
            .collect();

        let X_bootstrap = stack_rows(X, &indices)?;
        let y_bootstrap = select_elements(y, &indices);

        // Compute explanation for bootstrap sample
        let importance = explain_fn(&X_bootstrap.view(), &y_bootstrap.view())?;
        epistemic_samples.row_mut(i).assign(&importance);
    }

    // Compute mean importance and epistemic uncertainty
    let importance_estimates = epistemic_samples.mean_axis(Axis(0)).unwrap();
    let epistemic_uncertainty = compute_std_axis(&epistemic_samples, Axis(0));

    // Estimate aleatoric uncertainty using data perturbation
    let mut aleatoric_samples = Array2::zeros((config.n_samples, n_features));

    for i in 0..config.n_samples {
        // Add noise to data
        let noise_scale = 0.1;
        let X_noisy = add_gaussian_noise(X, noise_scale, &mut rng);

        // Compute explanation for noisy data
        let importance = explain_fn(&X_noisy.view(), y)?;
        aleatoric_samples.row_mut(i).assign(&importance);
    }

    let aleatoric_uncertainty = compute_std_axis(&aleatoric_samples, Axis(0));

    // Total uncertainty (epistemic + aleatoric)
    let total_uncertainty = &epistemic_uncertainty + &aleatoric_uncertainty;

    // Compute confidence intervals
    let importance_intervals =
        compute_credible_intervals(&epistemic_samples, config.confidence_level);

    // Compute reliability scores (inverse of total uncertainty)
    let reliability_scores = total_uncertainty.mapv(|x| 1.0 / (1.0 + x));

    Ok(UncertainExplanationResult {
        importance_estimates,
        epistemic_uncertainty,
        aleatoric_uncertainty,
        total_uncertainty,
        importance_intervals,
        reliability_scores,
    })
}

/// Perform Bayesian model averaging for explanations
pub fn bayesian_model_averaging<F>(
    models: Vec<F>,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &ProbabilisticConfig,
) -> SklResult<BayesianModelAveragingResult>
where
    F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
{
    let n_models = models.len();
    let n_features = X.ncols();

    // Compute individual model importances
    let mut individual_importances = Array2::zeros((n_models, n_features));
    let mut model_likelihoods = Array1::zeros(n_models);

    for (i, model) in models.iter().enumerate() {
        let importance = model(X, y)?;
        individual_importances.row_mut(i).assign(&importance);

        // Compute model likelihood (simplified)
        model_likelihoods[i] = compute_model_likelihood(&importance, X, y)?;
    }

    // Compute model posterior probabilities
    let model_posteriors = softmax(&model_likelihoods);

    // Compute weighted average of importances
    let averaged_importance = weighted_average(&individual_importances, &model_posteriors);

    // Compute model selection uncertainty
    let model_selection_uncertainty = compute_model_selection_uncertainty(
        &individual_importances,
        &model_posteriors,
        &averaged_importance,
    );

    Ok(BayesianModelAveragingResult {
        averaged_importance,
        model_weights: model_posteriors.clone(),
        individual_importances,
        model_posteriors,
        model_selection_uncertainty,
    })
}

// Helper functions

#[allow(non_snake_case)]
fn generate_variational_samples<F>(
    explain_fn: F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &ProbabilisticConfig,
    posterior_samples: &mut Array2<Float>,
    log_likelihoods: &mut Array1<Float>,
    rng: &mut ChaCha8Rng,
) -> SklResult<()>
where
    F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
{
    let n_features = X.ncols();

    // Initialize variational parameters
    let mut mu = Array1::zeros(n_features);
    let mut sigma = Array1::ones(n_features);

    // Variational optimization (simplified)
    for epoch in 0..100 {
        let mut gradient_mu = Array1::<Float>::zeros(n_features);
        let mut gradient_sigma = Array1::<Float>::zeros(n_features);

        // Mini-batch variational inference
        for _ in 0..10 {
            // Sample from current variational distribution
            let sample = sample_gaussian(&mu, &sigma, rng);

            // Compute gradient (simplified)
            let perturbation = 0.01;
            let X_pert = add_noise_to_features(X, &sample, perturbation);
            let importance = explain_fn(&X_pert.view(), y)?;

            // Update gradients (simplified ELBO gradient)
            let learning_rate = 0.01;
            gradient_mu = gradient_mu + &importance * learning_rate;
            gradient_sigma = gradient_sigma + &importance.mapv(|x| x * x) * learning_rate;
        }

        // Update variational parameters
        mu = mu + &gradient_mu * 0.01;
        sigma = sigma + &gradient_sigma * 0.01;
        sigma = sigma.mapv(|x| x.max(0.01)); // Ensure positive
    }

    // Generate final samples
    for i in 0..config.n_samples {
        let sample = sample_gaussian(&mu, &sigma, rng);
        posterior_samples.row_mut(i).assign(&sample);

        // Compute log likelihood (simplified)
        log_likelihoods[i] = compute_log_likelihood(&sample, X, y)?;
    }

    Ok(())
}

fn generate_mcmc_samples<F>(
    explain_fn: F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &ProbabilisticConfig,
    posterior_samples: &mut Array2<Float>,
    log_likelihoods: &mut Array1<Float>,
    rng: &mut ChaCha8Rng,
) -> SklResult<()>
where
    F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
{
    let n_features = X.ncols();
    let total_samples = config.n_samples + config.burnin_samples;

    // Initialize chain
    let mut current_state = Array1::zeros(n_features);
    let mut current_log_likelihood = compute_log_likelihood(&current_state, X, y)?;

    let mut accepted = 0;
    let proposal_scale = 0.1;

    for i in 0..total_samples {
        // Propose new state
        let proposal =
            &current_state + &generate_gaussian_noise(rng, n_features, 0.0, proposal_scale);
        let proposal_log_likelihood = compute_log_likelihood(&proposal, X, y)?;

        // Compute acceptance probability
        let log_alpha = proposal_log_likelihood - current_log_likelihood;
        let alpha = log_alpha.exp().min(1.0);

        // Accept or reject
        if rng.gen::<Float>() < alpha {
            current_state = proposal;
            current_log_likelihood = proposal_log_likelihood;
            accepted += 1;
        }

        // Store sample after burn-in
        if i >= config.burnin_samples {
            let sample_idx = i - config.burnin_samples;
            posterior_samples.row_mut(sample_idx).assign(&current_state);
            log_likelihoods[sample_idx] = current_log_likelihood;
        }
    }

    Ok(())
}

fn compute_credible_intervals(
    samples: &Array2<Float>,
    confidence_level: Float,
) -> Vec<(Float, Float)> {
    let n_features = samples.ncols();
    let mut intervals = Vec::with_capacity(n_features);

    let alpha: Float = 1.0 - confidence_level;
    let lower_percentile: Float = alpha / 2.0;
    let upper_percentile: Float = 1.0 - alpha / 2.0;

    for feature_idx in 0..n_features {
        let mut feature_samples: Vec<Float> = samples.column(feature_idx).to_vec();
        feature_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n_samples = feature_samples.len();
        let lower_idx = (lower_percentile * n_samples as Float).floor() as usize;
        let upper_idx = (upper_percentile * n_samples as Float).ceil() as usize;

        let lower_bound = feature_samples[lower_idx.min(n_samples - 1)];
        let upper_bound = feature_samples[upper_idx.min(n_samples - 1)];

        intervals.push((lower_bound, upper_bound));
    }

    intervals
}

fn compute_std_axis(arr: &Array2<Float>, axis: Axis) -> Array1<Float> {
    let means = arr.mean_axis(axis).unwrap();
    let n = arr.len_of(axis) as Float;

    let variances = if axis == Axis(0) {
        // Computing along rows
        arr.columns()
            .into_iter()
            .zip(means.iter())
            .map(|(col, &mean)| col.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n)
            .collect::<Vec<_>>()
    } else {
        // Computing along columns
        arr.rows()
            .into_iter()
            .zip(means.iter())
            .map(|(row, &mean)| row.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n)
            .collect::<Vec<_>>()
    };

    Array1::from_vec(variances).mapv(|x| x.sqrt())
}

fn compute_model_evidence(log_likelihoods: &Array1<Float>) -> Float {
    // Compute log marginal likelihood using importance sampling
    let max_ll = log_likelihoods
        .iter()
        .cloned()
        .fold(Float::NEG_INFINITY, Float::max);
    let sum_exp = log_likelihoods
        .iter()
        .map(|&ll| (ll - max_ll).exp())
        .sum::<Float>();
    max_ll + (sum_exp / log_likelihoods.len() as Float).ln()
}

fn compute_effective_sample_size(samples: &Array2<Float>) -> Array1<Float> {
    let n_features = samples.ncols();
    let mut ess = Array1::zeros(n_features);

    for feature_idx in 0..n_features {
        let feature_samples = samples.column(feature_idx);

        // Compute autocorrelation function (simplified)
        let mut autocorr_sum = 0.0;
        let n_lags = (feature_samples.len() / 4).min(50);

        for lag in 1..=n_lags {
            let autocorr = compute_autocorrelation(&feature_samples, lag);
            if autocorr <= 0.0 {
                break;
            }
            autocorr_sum += autocorr;
        }

        // Effective sample size
        ess[feature_idx] = feature_samples.len() as Float / (1.0 + 2.0 * autocorr_sum);
    }

    ess
}

fn compute_r_hat_diagnostic(samples: &Array2<Float>, n_chains: usize) -> Array1<Float> {
    let n_features = samples.ncols();
    let mut r_hat = Array1::ones(n_features);

    if n_chains < 2 {
        return r_hat; // Cannot compute R-hat with single chain
    }

    let chain_length = samples.nrows() / n_chains;

    for feature_idx in 0..n_features {
        let feature_samples = samples.column(feature_idx);

        // Split samples into chains
        let mut chain_means = Vec::new();
        let mut chain_vars = Vec::new();

        for chain_idx in 0..n_chains {
            let start = chain_idx * chain_length;
            let end = (start + chain_length).min(feature_samples.len());
            let chain_data: Vec<Float> = feature_samples.slice(s![start..end]).to_vec();

            let mean = chain_data.iter().sum::<Float>() / chain_data.len() as Float;
            let var = chain_data
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<Float>()
                / (chain_data.len() - 1) as Float;

            chain_means.push(mean);
            chain_vars.push(var);
        }

        // Compute R-hat statistic
        let overall_mean = chain_means.iter().sum::<Float>() / n_chains as Float;
        let between_var = chain_length as Float
            * chain_means
                .iter()
                .map(|&m| (m - overall_mean).powi(2))
                .sum::<Float>()
            / (n_chains - 1) as Float;
        let within_var = chain_vars.iter().sum::<Float>() / n_chains as Float;

        let var_estimate =
            ((chain_length - 1) as Float * within_var + between_var) / chain_length as Float;
        r_hat[feature_idx] = (var_estimate / within_var).sqrt();
    }

    r_hat
}

fn compute_credible_region(samples: &Array2<Float>, confidence_level: Float) -> Array2<Float> {
    // Compute minimum volume credible region (simplified to axis-aligned box)
    let n_features = samples.ncols();
    let mut region = Array2::zeros((2, n_features)); // [lower_bounds, upper_bounds]

    let intervals = compute_credible_intervals(samples, confidence_level);

    for (feature_idx, &(lower, upper)) in intervals.iter().enumerate() {
        region[[0, feature_idx]] = lower;
        region[[1, feature_idx]] = upper;
    }

    region
}

fn compute_feasibility_probabilities(
    counterfactuals: &Array2<Float>,
    instance: &ArrayView1<Float>,
) -> Array1<Float> {
    let n_samples = counterfactuals.nrows();
    let mut probabilities = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let counterfactual = counterfactuals.row(i);

        // Simple feasibility metric based on distance from original instance
        let distance = compute_euclidean_distance(&counterfactual, instance);

        // Convert distance to probability (closer = more feasible)
        probabilities[i] = (-distance).exp();
    }

    probabilities
}

// Additional helper functions

fn generate_gaussian_noise(
    rng: &mut ChaCha8Rng,
    size: usize,
    mean: Float,
    std: Float,
) -> Array1<Float> {
    let normal = Normal::new(mean, std).unwrap();
    Array1::from_vec((0..size).map(|_| rng.sample(normal)).collect())
}

fn sample_gaussian(
    mu: &Array1<Float>,
    sigma: &Array1<Float>,
    rng: &mut ChaCha8Rng,
) -> Array1<Float> {
    let n = mu.len();
    let mut sample = Array1::zeros(n);

    for i in 0..n {
        let normal = Normal::new(mu[i], sigma[i]).unwrap();
        sample[i] = rng.sample(normal);
    }

    sample
}

fn add_noise_to_features(
    X: &ArrayView2<Float>,
    noise: &Array1<Float>,
    scale: Float,
) -> Array2<Float> {
    let mut X_noisy = X.to_owned();
    for (i, &noise_val) in noise.iter().enumerate() {
        if i < X_noisy.ncols() {
            let mut column = X_noisy.column_mut(i);
            column.mapv_inplace(|x| x + noise_val * scale);
        }
    }
    X_noisy
}

fn add_gaussian_noise(
    X: &ArrayView2<Float>,
    noise_scale: Float,
    rng: &mut ChaCha8Rng,
) -> Array2<Float> {
    let mut X_noisy = X.to_owned();
    let normal = Normal::new(0.0, noise_scale).unwrap();

    for mut row in X_noisy.axis_iter_mut(Axis(0)) {
        for val in row.iter_mut() {
            *val += rng.sample(normal);
        }
    }

    X_noisy
}

fn compute_log_likelihood(
    params: &Array1<Float>,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
) -> SklResult<Float> {
    // Simplified log likelihood computation
    let n_samples = X.nrows();
    let mut log_likelihood = 0.0;

    for i in 0..n_samples {
        let x_i = X.row(i);
        let y_i = y[i];

        // Simple linear model likelihood
        let prediction = x_i.dot(params);
        let residual = y_i - prediction;
        log_likelihood -= 0.5 * residual * residual;
    }

    Ok(log_likelihood)
}

fn compute_autocorrelation(series: &ArrayView1<Float>, lag: usize) -> Float {
    if lag >= series.len() {
        return 0.0;
    }

    let n = series.len() - lag;
    let mean = series.mean().unwrap_or(0.0);

    let mut numerator: Float = 0.0;
    let mut denominator: Float = 0.0;

    for i in 0..n {
        let x_i = series[i] - mean;
        let x_i_lag = series[i + lag] - mean;
        numerator += x_i * x_i_lag;
        denominator += x_i * x_i;
    }

    if denominator.abs() < Float::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn stack_rows(X: &ArrayView2<Float>, indices: &[usize]) -> SklResult<Array2<Float>> {
    let n_rows = indices.len();
    let n_cols = X.ncols();
    let mut result = Array2::zeros((n_rows, n_cols));

    for (i, &idx) in indices.iter().enumerate() {
        if idx >= X.nrows() {
            return Err(SklearsError::InvalidInput(
                "Index out of bounds".to_string(),
            ));
        }
        result.row_mut(i).assign(&X.row(idx));
    }

    Ok(result)
}

fn select_elements(arr: &ArrayView1<Float>, indices: &[usize]) -> Array1<Float> {
    Array1::from_vec(indices.iter().map(|&i| arr[i]).collect())
}

fn compute_model_likelihood(
    importance: &Array1<Float>,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
) -> SklResult<Float> {
    // Simplified model likelihood based on explanation quality
    let prediction_quality = compute_prediction_quality(importance, X, y)?;
    Ok(prediction_quality.ln())
}

fn compute_prediction_quality(
    importance: &Array1<Float>,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
) -> SklResult<Float> {
    // Simplified quality metric
    let weighted_features = X.dot(importance);
    let correlation = compute_correlation(&weighted_features, y);
    Ok(correlation.abs())
}

fn compute_correlation(x: &Array1<Float>, y: &ArrayView1<Float>) -> Float {
    let n = x.len() as Float;
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut numerator: Float = 0.0;
    let mut sum_sq_x: Float = 0.0;
    let mut sum_sq_y: Float = 0.0;

    for i in 0..x.len() {
        let x_dev = x[i] - mean_x;
        let y_dev = y[i] - mean_y;

        numerator += x_dev * y_dev;
        sum_sq_x += x_dev * x_dev;
        sum_sq_y += y_dev * y_dev;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator.abs() < Float::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn softmax(x: &Array1<Float>) -> Array1<Float> {
    let max_x = x.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
    let exp_x = x.mapv(|val| (val - max_x).exp());
    let sum_exp = exp_x.sum();
    exp_x / sum_exp
}

fn weighted_average(matrix: &Array2<Float>, weights: &Array1<Float>) -> Array1<Float> {
    let n_features = matrix.ncols();
    let mut result = Array1::zeros(n_features);

    for feature_idx in 0..n_features {
        let column = matrix.column(feature_idx);
        result[feature_idx] = column.dot(weights);
    }

    result
}

fn compute_model_selection_uncertainty(
    individual_importances: &Array2<Float>,
    model_posteriors: &Array1<Float>,
    averaged_importance: &Array1<Float>,
) -> Array1<Float> {
    let n_features = individual_importances.ncols();
    let mut uncertainty = Array1::zeros(n_features);

    for feature_idx in 0..n_features {
        let feature_column = individual_importances.column(feature_idx);
        let avg_importance = averaged_importance[feature_idx];

        let weighted_variance = feature_column
            .iter()
            .zip(model_posteriors.iter())
            .map(|(&importance, &weight)| weight * (importance - avg_importance).powi(2))
            .sum::<Float>();

        uncertainty[feature_idx] = weighted_variance.sqrt();
    }

    uncertainty
}

fn compute_euclidean_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<Float>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_explanation() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let explain_fn = |X: &ArrayView2<Float>, y: &ArrayView1<Float>| {
            Ok(array![0.5, 0.3]) // Mock explanation
        };

        let config = ProbabilisticConfig {
            n_samples: 10,
            use_variational_inference: true,
            ..Default::default()
        };

        let result = generate_bayesian_explanation(explain_fn, &X.view(), &y.view(), &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.posterior_means.len(), 2);
        assert_eq!(result.credible_intervals.len(), 2);
    }

    #[test]
    fn test_probabilistic_counterfactuals() {
        let instance = array![1.0, 2.0];
        let target_class = 1;

        let model = |X: &ArrayView2<Float>| {
            let n_samples = X.nrows();
            Ok(Array2::from_shape_fn((n_samples, 2), |(i, j)| {
                if j == 0 {
                    0.6
                } else {
                    0.4
                }
            }))
        };

        let config = ProbabilisticConfig {
            n_samples: 5,
            ..Default::default()
        };

        let result =
            generate_probabilistic_counterfactuals(model, &instance.view(), target_class, &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.most_likely_counterfactual.len(), 2);
        assert_eq!(result.counterfactual_distribution.nrows(), 5);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_uncertainty_quantification() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let explain_fn = |X: &ArrayView2<Float>, y: &ArrayView1<Float>| {
            Ok(array![0.5, 0.3]) // Mock explanation
        };

        let config = ProbabilisticConfig {
            n_samples: 5,
            ..Default::default()
        };

        let result = quantify_explanation_uncertainty(explain_fn, &X.view(), &y.view(), &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.importance_estimates.len(), 2);
        assert_eq!(result.epistemic_uncertainty.len(), 2);
        assert_eq!(result.aleatoric_uncertainty.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_model_averaging() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let model1 = |X: &ArrayView2<Float>, y: &ArrayView1<Float>| Ok(array![0.5, 0.3]);
        let model2 = |X: &ArrayView2<Float>, y: &ArrayView1<Float>| Ok(array![0.4, 0.6]);
        let models = vec![model1, model2];

        let config = ProbabilisticConfig::default();

        let result = bayesian_model_averaging(models, &X.view(), &y.view(), &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.averaged_importance.len(), 2);
        assert_eq!(result.model_weights.len(), 2);
        assert_eq!(result.individual_importances.nrows(), 2);
    }

    #[test]
    fn test_credible_intervals() {
        let samples = array![[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]];

        let intervals = compute_credible_intervals(&samples, 0.8);
        assert_eq!(intervals.len(), 2);

        // Check that intervals make sense
        for (lower, upper) in intervals {
            assert!(lower <= upper);
        }
    }

    #[test]
    fn test_effective_sample_size() {
        let samples = Array2::from_shape_fn((100, 3), |(i, j)| i as Float + j as Float);
        let ess = compute_effective_sample_size(&samples);

        assert_eq!(ess.len(), 3);
        // ESS should be positive
        for &value in ess.iter() {
            assert!(value > 0.0);
        }
    }
}
