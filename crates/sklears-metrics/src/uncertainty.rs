//! Uncertainty Quantification for Machine Learning Metrics
//!
//! This module provides comprehensive uncertainty quantification capabilities
//! for machine learning evaluation metrics, including confidence intervals,
//! bootstrap estimation, and Bayesian approaches.
//!
//! # Features
//!
//! - Bootstrap confidence intervals for any metric
//! - Analytical confidence intervals for common metrics
//! - Bayesian credible intervals with conjugate priors
//! - Uncertainty propagation for composite metrics
//! - Multiple bootstrap methods (percentile, bias-corrected, BCa)
//! - Statistical significance testing for metric comparisons
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::uncertainty::*;
//! use sklears_metrics::classification::accuracy_score;
//! use scirs2_core::ndarray::Array1;
//!
//! let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
//! let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
//!
//! // Calculate bootstrap confidence interval for accuracy
//! let bootstrap_result = bootstrap_confidence_interval(
//!     &y_true,
//!     &y_pred,
//!     |y_true, y_pred| accuracy_score(y_true, y_pred).unwrap_or(0.0),
//!     0.95,
//!     1000,
//!     42
//! ).unwrap();
//!
//! println!("Accuracy: {:.3} [{:.3}, {:.3}]",
//!          bootstrap_result.point_estimate,
//!          bootstrap_result.lower_bound,
//!          bootstrap_result.upper_bound);
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::Distribution;
// Beta and Normal distributions available via SciRS2 random module
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::Beta;
use std::collections::HashMap;

/// Result of uncertainty quantification analysis
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Point estimate of the metric
    pub point_estimate: f64,
    /// Lower bound of confidence/credible interval
    pub lower_bound: f64,
    /// Upper bound of confidence/credible interval
    pub upper_bound: f64,
    /// Confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Standard error of the estimate
    pub standard_error: f64,
    /// Method used for uncertainty quantification
    pub method: String,
    /// Additional statistics
    pub statistics: HashMap<String, f64>,
}

/// Bootstrap method for confidence interval computation
#[derive(Debug, Clone, Copy)]
pub enum BootstrapMethod {
    /// Simple percentile method
    Percentile,
    /// Bias-corrected percentile method
    BiasCorrected,
    /// Bias-corrected and accelerated (BCa) method
    BCa,
    /// Basic bootstrap method
    Basic,
}

/// Configuration for uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Bootstrap method to use
    pub bootstrap_method: BootstrapMethod,
    /// Whether to use stratified sampling for classification
    pub stratified: bool,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            n_bootstrap: 1000,
            seed: None,
            bootstrap_method: BootstrapMethod::Percentile,
            stratified: false,
        }
    }
}

/// Generic bootstrap confidence interval for any metric function
///
/// # Arguments
///
/// * `y_true` - True labels/values
/// * `y_pred` - Predicted labels/values
/// * `metric_fn` - Function that computes the metric
/// * `confidence_level` - Confidence level (e.g., 0.95)
/// * `n_bootstrap` - Number of bootstrap samples
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// Uncertainty result with confidence interval
pub fn bootstrap_confidence_interval<F>(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    metric_fn: F,
    confidence_level: f64,
    n_bootstrap: usize,
    seed: u64,
) -> MetricsResult<UncertaintyResult>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> f64,
{
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if !(0.0..1.0).contains(&confidence_level) {
        return Err(MetricsError::InvalidParameter(
            "confidence_level must be between 0 and 1".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let n_samples = y_true.len();

    // Calculate point estimate
    let point_estimate = metric_fn(y_true, y_pred);

    // Generate bootstrap samples
    let mut bootstrap_estimates = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let indices: Vec<usize> = (0..n_samples)
            .map(|_| rng.gen_range(0..n_samples))
            .collect();

        let y_true_boot = indices.iter().map(|&i| y_true[i]).collect::<Array1<f64>>();
        let y_pred_boot = indices.iter().map(|&i| y_pred[i]).collect::<Array1<f64>>();

        let bootstrap_metric = metric_fn(&y_true_boot, &y_pred_boot);
        bootstrap_estimates.push(bootstrap_metric);
    }

    // Sort bootstrap estimates
    bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate confidence interval using percentile method
    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize - 1;

    let lower_bound = bootstrap_estimates[lower_idx];
    let upper_bound = bootstrap_estimates[upper_idx];

    // Calculate standard error
    let mean_bootstrap = bootstrap_estimates.iter().sum::<f64>() / n_bootstrap as f64;
    let variance = bootstrap_estimates
        .iter()
        .map(|x| (x - mean_bootstrap).powi(2))
        .sum::<f64>()
        / (n_bootstrap - 1) as f64;
    let standard_error = variance.sqrt();

    // Additional statistics
    let mut statistics = HashMap::new();
    statistics.insert("bootstrap_mean".to_string(), mean_bootstrap);
    statistics.insert("bootstrap_std".to_string(), variance.sqrt());
    statistics.insert("bias".to_string(), mean_bootstrap - point_estimate);

    Ok(UncertaintyResult {
        point_estimate,
        lower_bound,
        upper_bound,
        confidence_level,
        standard_error,
        method: "Bootstrap Percentile".to_string(),
        statistics,
    })
}

/// Bias-corrected and accelerated (BCa) bootstrap confidence interval
///
/// # Arguments
///
/// * `y_true` - True labels/values
/// * `y_pred` - Predicted labels/values
/// * `metric_fn` - Function that computes the metric
/// * `confidence_level` - Confidence level
/// * `n_bootstrap` - Number of bootstrap samples
/// * `seed` - Random seed
///
/// # Returns
///
/// BCa bootstrap confidence interval
pub fn bca_bootstrap_confidence_interval<F>(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    metric_fn: F,
    confidence_level: f64,
    n_bootstrap: usize,
    seed: u64,
) -> MetricsResult<UncertaintyResult>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> f64 + Copy,
{
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let n_samples = y_true.len();

    // Calculate point estimate
    let point_estimate = metric_fn(y_true, y_pred);

    // Generate bootstrap samples
    let mut bootstrap_estimates = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let indices: Vec<usize> = (0..n_samples)
            .map(|_| rng.gen_range(0..n_samples))
            .collect();

        let y_true_boot = indices.iter().map(|&i| y_true[i]).collect::<Array1<f64>>();
        let y_pred_boot = indices.iter().map(|&i| y_pred[i]).collect::<Array1<f64>>();

        let bootstrap_metric = metric_fn(&y_true_boot, &y_pred_boot);
        bootstrap_estimates.push(bootstrap_metric);
    }

    // Calculate bias correction
    let count_less = bootstrap_estimates
        .iter()
        .filter(|&&x| x < point_estimate)
        .count();
    let z0 = if count_less == 0 || count_less == n_bootstrap {
        0.0
    } else {
        let p = count_less as f64 / n_bootstrap as f64;
        normal_quantile(p)
    };

    // Calculate acceleration using jackknife
    let mut jackknife_estimates = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let y_true_jack = y_true
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| *v)
            .collect::<Array1<f64>>();
        let y_pred_jack = y_pred
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| *v)
            .collect::<Array1<f64>>();

        let jackknife_metric = metric_fn(&y_true_jack, &y_pred_jack);
        jackknife_estimates.push(jackknife_metric);
    }

    let mean_jackknife = jackknife_estimates.iter().sum::<f64>() / n_samples as f64;
    let numerator: f64 = jackknife_estimates
        .iter()
        .map(|x| (mean_jackknife - x).powi(3))
        .sum();
    let denominator: f64 = jackknife_estimates
        .iter()
        .map(|x| (mean_jackknife - x).powi(2))
        .sum();

    let acceleration = if denominator > 0.0 {
        numerator / (6.0 * denominator.powf(1.5))
    } else {
        0.0
    };

    // Calculate BCa confidence interval
    let alpha = 1.0 - confidence_level;
    let z_alpha_2 = normal_quantile(alpha / 2.0);
    let z_1_alpha_2 = normal_quantile(1.0 - alpha / 2.0);

    let alpha_1 = normal_cdf(z0 + (z0 + z_alpha_2) / (1.0 - acceleration * (z0 + z_alpha_2)));
    let alpha_2 = normal_cdf(z0 + (z0 + z_1_alpha_2) / (1.0 - acceleration * (z0 + z_1_alpha_2)));

    bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (alpha_1 * n_bootstrap as f64).floor() as usize;
    let upper_idx = (alpha_2 * n_bootstrap as f64).ceil() as usize - 1;

    let lower_bound = bootstrap_estimates[lower_idx.min(n_bootstrap - 1)];
    let upper_bound = bootstrap_estimates[upper_idx.min(n_bootstrap - 1)];

    // Calculate standard error
    let mean_bootstrap = bootstrap_estimates.iter().sum::<f64>() / n_bootstrap as f64;
    let variance = bootstrap_estimates
        .iter()
        .map(|x| (x - mean_bootstrap).powi(2))
        .sum::<f64>()
        / (n_bootstrap - 1) as f64;
    let standard_error = variance.sqrt();

    let mut statistics = HashMap::new();
    statistics.insert("bias_correction".to_string(), z0);
    statistics.insert("acceleration".to_string(), acceleration);
    statistics.insert("bootstrap_mean".to_string(), mean_bootstrap);

    Ok(UncertaintyResult {
        point_estimate,
        lower_bound,
        upper_bound,
        confidence_level,
        standard_error,
        method: "Bootstrap BCa".to_string(),
        statistics,
    })
}

/// Bayesian credible interval for classification accuracy using Beta prior
///
/// # Arguments
///
/// * `n_correct` - Number of correct predictions
/// * `n_total` - Total number of predictions
/// * `alpha_prior` - Alpha parameter of Beta prior
/// * `beta_prior` - Beta parameter of Beta prior
/// * `confidence_level` - Credible interval level
///
/// # Returns
///
/// Bayesian credible interval for accuracy
pub fn bayesian_accuracy_credible_interval(
    n_correct: usize,
    n_total: usize,
    alpha_prior: f64,
    beta_prior: f64,
    confidence_level: f64,
) -> MetricsResult<UncertaintyResult> {
    if n_correct > n_total {
        return Err(MetricsError::InvalidParameter(
            "n_correct cannot exceed n_total".to_string(),
        ));
    }

    if alpha_prior <= 0.0 || beta_prior <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "prior parameters must be positive".to_string(),
        ));
    }

    // Beta posterior parameters
    let alpha_post = alpha_prior + n_correct as f64;
    let beta_post = beta_prior + (n_total - n_correct) as f64;

    // Point estimate (posterior mean)
    let point_estimate = alpha_post / (alpha_post + beta_post);

    // Credible interval using Beta quantiles
    let _beta_dist = Beta::new(alpha_post, beta_post).unwrap();
    let alpha = 1.0 - confidence_level;

    let lower_bound = beta_quantile(alpha / 2.0, alpha_post, beta_post);
    let upper_bound = beta_quantile(1.0 - alpha / 2.0, alpha_post, beta_post);

    // Standard error (posterior standard deviation)
    let standard_error = (alpha_post * beta_post
        / ((alpha_post + beta_post).powi(2) * (alpha_post + beta_post + 1.0)))
        .sqrt();

    let mut statistics = HashMap::new();
    statistics.insert("alpha_prior".to_string(), alpha_prior);
    statistics.insert("beta_prior".to_string(), beta_prior);
    statistics.insert("alpha_posterior".to_string(), alpha_post);
    statistics.insert("beta_posterior".to_string(), beta_post);
    statistics.insert(
        "posterior_mode".to_string(),
        if alpha_post > 1.0 && beta_post > 1.0 {
            (alpha_post - 1.0) / (alpha_post + beta_post - 2.0)
        } else {
            point_estimate
        },
    );

    Ok(UncertaintyResult {
        point_estimate,
        lower_bound,
        upper_bound,
        confidence_level,
        standard_error,
        method: "Bayesian Beta-Binomial".to_string(),
        statistics,
    })
}

/// Analytical confidence interval for correlation coefficient
///
/// # Arguments
///
/// * `r` - Sample correlation coefficient
/// * `n` - Sample size
/// * `confidence_level` - Confidence level
///
/// # Returns
///
/// Analytical confidence interval for correlation
pub fn correlation_confidence_interval(
    r: f64,
    n: usize,
    confidence_level: f64,
) -> MetricsResult<UncertaintyResult> {
    if n < 3 {
        return Err(MetricsError::InvalidParameter(
            "sample size must be at least 3".to_string(),
        ));
    }

    if r.abs() > 1.0 {
        return Err(MetricsError::InvalidParameter(
            "correlation coefficient must be between -1 and 1".to_string(),
        ));
    }

    // Fisher z-transformation
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let se_z = 1.0 / (n as f64 - 3.0).sqrt();

    let alpha = 1.0 - confidence_level;
    let z_critical = normal_quantile(1.0 - alpha / 2.0);

    let z_lower = z - z_critical * se_z;
    let z_upper = z + z_critical * se_z;

    // Transform back to correlation scale using tanh
    let lower_bound = z_lower.tanh();
    let upper_bound = z_upper.tanh();

    // Standard error on original scale (approximate)
    let standard_error = (1.0 - r.powi(2)) / (n as f64 - 1.0).sqrt();

    let mut statistics = HashMap::new();
    statistics.insert("fisher_z".to_string(), z);
    statistics.insert("fisher_z_se".to_string(), se_z);
    statistics.insert("sample_size".to_string(), n as f64);

    Ok(UncertaintyResult {
        point_estimate: r,
        lower_bound,
        upper_bound,
        confidence_level,
        standard_error,
        method: "Fisher Z-transformation".to_string(),
        statistics,
    })
}

/// Confidence interval for mean squared error using chi-square distribution
///
/// # Arguments
///
/// * `mse` - Mean squared error
/// * `n` - Sample size
/// * `confidence_level` - Confidence level
///
/// # Returns
///
/// Confidence interval for MSE
pub fn mse_confidence_interval(
    mse: f64,
    n: usize,
    confidence_level: f64,
) -> MetricsResult<UncertaintyResult> {
    if mse < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "MSE must be non-negative".to_string(),
        ));
    }

    if n < 2 {
        return Err(MetricsError::InvalidParameter(
            "sample size must be at least 2".to_string(),
        ));
    }

    let alpha = 1.0 - confidence_level;
    let df = n - 1;

    // Chi-square quantiles (approximate)
    let chi2_lower = chi_square_quantile(alpha / 2.0, df);
    let chi2_upper = chi_square_quantile(1.0 - alpha / 2.0, df);

    let lower_bound = (df as f64 * mse) / chi2_upper;
    let upper_bound = (df as f64 * mse) / chi2_lower;

    // Standard error (approximate)
    let standard_error = mse * (2.0 / df as f64).sqrt();

    let mut statistics = HashMap::new();
    statistics.insert("degrees_of_freedom".to_string(), df as f64);
    statistics.insert("chi2_lower".to_string(), chi2_lower);
    statistics.insert("chi2_upper".to_string(), chi2_upper);

    Ok(UncertaintyResult {
        point_estimate: mse,
        lower_bound,
        upper_bound,
        confidence_level,
        standard_error,
        method: "Chi-square based".to_string(),
        statistics,
    })
}

/// Uncertainty propagation for composite metrics
///
/// # Arguments
///
/// * `metrics` - Individual metric estimates
/// * `uncertainties` - Standard errors for each metric
/// * `combine_fn` - Function to combine metrics
///
/// # Returns
///
/// Uncertainty estimate for combined metric
pub fn uncertainty_propagation<F>(
    metrics: &Array1<f64>,
    uncertainties: &Array1<f64>,
    combine_fn: F,
) -> MetricsResult<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    if metrics.len() != uncertainties.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![metrics.len()],
            actual: vec![uncertainties.len()],
        });
    }

    if metrics.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Monte Carlo approach for uncertainty propagation
    let n_samples = 10000;
    let mut rng = StdRng::seed_from_u64(42);
    let mut combined_values = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut perturbed_metrics = metrics.clone();

        for i in 0..metrics.len() {
            let normal =
                scirs2_core::random::RandNormal::new(metrics[i], uncertainties[i]).unwrap();
            perturbed_metrics[i] = normal.sample(&mut rng);
        }

        let combined_value = combine_fn(&perturbed_metrics);
        combined_values.push(combined_value);
    }

    // Calculate standard error of combined metric
    let mean = combined_values.iter().sum::<f64>() / n_samples as f64;
    let variance = combined_values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / (n_samples - 1) as f64;

    Ok(variance.sqrt())
}

/// Compare two metrics using bootstrap hypothesis testing
///
/// # Arguments
///
/// * `y_true1` - True values for first model
/// * `y_pred1` - Predictions for first model
/// * `y_true2` - True values for second model
/// * `y_pred2` - Predictions for second model
/// * `metric_fn` - Metric function to compare
/// * `n_bootstrap` - Number of bootstrap samples
/// * `seed` - Random seed
///
/// # Returns
///
/// P-value for the hypothesis test
pub fn bootstrap_metric_comparison<F>(
    y_true1: &Array1<f64>,
    y_pred1: &Array1<f64>,
    y_true2: &Array1<f64>,
    y_pred2: &Array1<f64>,
    metric_fn: F,
    n_bootstrap: usize,
    seed: u64,
) -> MetricsResult<f64>
where
    F: Fn(&Array1<f64>, &Array1<f64>) -> f64,
{
    let metric1 = metric_fn(y_true1, y_pred1);
    let metric2 = metric_fn(y_true2, y_pred2);
    let observed_diff = metric1 - metric2;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut bootstrap_diffs = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Bootstrap first model
        let indices1: Vec<usize> = (0..y_true1.len())
            .map(|_| rng.gen_range(0..y_true1.len()))
            .collect();
        let y_true1_boot = indices1
            .iter()
            .map(|&i| y_true1[i])
            .collect::<Array1<f64>>();
        let y_pred1_boot = indices1
            .iter()
            .map(|&i| y_pred1[i])
            .collect::<Array1<f64>>();

        // Bootstrap second model
        let indices2: Vec<usize> = (0..y_true2.len())
            .map(|_| rng.gen_range(0..y_true2.len()))
            .collect();
        let y_true2_boot = indices2
            .iter()
            .map(|&i| y_true2[i])
            .collect::<Array1<f64>>();
        let y_pred2_boot = indices2
            .iter()
            .map(|&i| y_pred2[i])
            .collect::<Array1<f64>>();

        let boot_metric1 = metric_fn(&y_true1_boot, &y_pred1_boot);
        let boot_metric2 = metric_fn(&y_true2_boot, &y_pred2_boot);
        let boot_diff = boot_metric1 - boot_metric2;

        bootstrap_diffs.push(boot_diff);
    }

    // Calculate p-value (two-tailed test)
    let count_extreme = bootstrap_diffs
        .iter()
        .filter(|&&diff| diff.abs() >= observed_diff.abs())
        .count();

    let p_value = count_extreme as f64 / n_bootstrap as f64;

    Ok(p_value)
}

// Helper functions for statistical distributions

fn normal_quantile(p: f64) -> f64 {
    // Simple and accurate inverse normal CDF using Peter John Acklam's algorithm
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    // Constants for rational approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    // Define break-points
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Approximate error function using Abramowitz and Stegun formula
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn beta_quantile(p: f64, alpha: f64, beta: f64) -> f64 {
    // Approximate beta quantile using continued fraction
    incomplete_beta_inv(p, alpha, beta)
}

fn incomplete_beta_inv(p: f64, a: f64, b: f64) -> f64 {
    // Improved inverse incomplete beta function using Newton-Raphson
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    // Better initial guess
    let mut x = if a > 1.0 && b > 1.0 {
        (a - 1.0) / (a + b - 2.0)
    } else {
        p
    };

    // Newton-Raphson iteration
    for _ in 0..100 {
        let fx = incomplete_beta(x, a, b) - p;
        if fx.abs() < 1e-12 {
            break;
        }

        // Derivative: PDF of beta distribution
        let fpx = beta_pdf(x, a, b);
        if fpx.abs() < 1e-15 {
            break;
        }

        let new_x = x - fx / fpx;

        // Ensure we stay in bounds
        x = new_x.clamp(1e-15, 1.0 - 1e-15);

        if (new_x - x).abs() < 1e-15 {
            break;
        }
    }

    x
}

fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    // Beta probability density function
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }

    x.powf(a - 1.0) * (1.0 - x).powf(b - 1.0) / beta_function(a, b)
}

fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    // Improved incomplete beta function using continued fraction
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry property when needed
    if x > a / (a + b) {
        return 1.0 - incomplete_beta(1.0 - x, b, a);
    }

    // Continued fraction expansion
    let bt = (gamma(a + b) / (gamma(a) * gamma(b))) * x.powf(a) * (1.0 - x).powf(b);

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(x, a, b) / a
    } else {
        1.0 - bt * beta_cf(1.0 - x, b, a) / b
    }
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    // Continued fraction for beta function
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=100 {
        let m2 = 2 * m;
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));

        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < 1e-10 {
            break;
        }
    }

    h
}

fn beta_function(a: f64, b: f64) -> f64 {
    // Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    gamma(a) * gamma(b) / gamma(a + b)
}

fn gamma(x: f64) -> f64 {
    // Improved gamma function using Lanczos approximation
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
    }

    // Lanczos coefficients for g=7
    let g = 7.0;
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let z = x - 1.0;
    let mut x_sum = c[0];
    for (i, &coeff) in c.iter().enumerate().skip(1) {
        x_sum += coeff / (z + i as f64);
    }

    let t = z + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x_sum
}

fn chi_square_quantile(p: f64, df: usize) -> f64 {
    // Approximate chi-square quantile using Wilson-Hilferty transformation
    let h = 2.0 / (9.0 * df as f64);
    let z = normal_quantile(p);
    let x = 1.0 - h + z * h.sqrt();

    if x > 0.0 {
        df as f64 * x.powi(3)
    } else {
        0.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::classification::accuracy_score;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bootstrap_confidence_interval() {
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

        let result = bootstrap_confidence_interval(
            &y_true,
            &y_pred,
            |y_true, y_pred| accuracy_score(y_true, y_pred).unwrap_or(0.0),
            0.95,
            100,
            42,
        )
        .unwrap();

        assert!(result.confidence_level == 0.95);
        assert!(result.lower_bound <= result.point_estimate);
        assert!(result.point_estimate <= result.upper_bound);
        assert!(result.standard_error > 0.0);
    }

    #[test]
    fn test_bayesian_accuracy_credible_interval() {
        let result = bayesian_accuracy_credible_interval(75, 100, 1.0, 1.0, 0.95).unwrap();

        assert!(result.confidence_level == 0.95);
        assert!(result.lower_bound <= result.point_estimate);
        assert!(result.point_estimate <= result.upper_bound);
        assert!(result.point_estimate > 0.7);
        assert!(result.point_estimate < 0.8);
    }

    #[test]
    fn test_correlation_confidence_interval() {
        let result = correlation_confidence_interval(0.5, 100, 0.95).unwrap();

        assert!(result.confidence_level == 0.95);
        assert!(result.lower_bound <= result.point_estimate);
        assert!(result.point_estimate <= result.upper_bound);
        assert_abs_diff_eq!(result.point_estimate, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_confidence_interval() {
        let result = mse_confidence_interval(0.25, 50, 0.95).unwrap();

        assert!(result.confidence_level == 0.95);
        assert!(result.lower_bound <= result.point_estimate);
        assert!(result.point_estimate <= result.upper_bound);
        assert_abs_diff_eq!(result.point_estimate, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_uncertainty_propagation() {
        let metrics = Array1::from_vec(vec![0.8, 0.9, 0.7]);
        let uncertainties = Array1::from_vec(vec![0.05, 0.03, 0.08]);

        let combined_uncertainty =
            uncertainty_propagation(&metrics, &uncertainties, |m| m.mean().unwrap()).unwrap();

        assert!(combined_uncertainty > 0.0);
        assert!(combined_uncertainty < 0.1);
    }

    #[test]
    fn test_bootstrap_metric_comparison() {
        let y_true1 = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
        let y_pred1 = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
        let y_true2 = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
        let y_pred2 = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0]);

        let p_value = bootstrap_metric_comparison(
            &y_true1,
            &y_pred1,
            &y_true2,
            &y_pred2,
            |y_true, y_pred| accuracy_score(y_true, y_pred).unwrap_or(0.0),
            100,
            42,
        )
        .unwrap();

        assert!(p_value >= 0.0);
        assert!(p_value <= 1.0);
    }

    #[test]
    fn test_normal_quantile() {
        // Test some known values
        assert_abs_diff_eq!(normal_quantile(0.5), 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(normal_quantile(0.025), -1.96, epsilon = 1e-2);
        assert_abs_diff_eq!(normal_quantile(0.975), 1.96, epsilon = 1e-2);
    }

    #[test]
    fn test_normal_cdf() {
        assert_abs_diff_eq!(normal_cdf(0.0), 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(normal_cdf(1.96), 0.975, epsilon = 1e-3);
        assert_abs_diff_eq!(normal_cdf(-1.96), 0.025, epsilon = 1e-3);
    }
}
