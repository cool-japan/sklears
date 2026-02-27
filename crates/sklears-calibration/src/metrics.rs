//! Calibration evaluation metrics
//!
//! This module provides metrics to evaluate the quality of probability calibration,
//! including Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
//! reliability diagrams, Brier score decomposition, and statistical tests.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

/// Configuration for calibration metrics
#[derive(Debug, Clone)]
pub struct CalibrationMetricsConfig {
    /// Number of bins for reliability diagram and ECE/MCE calculation
    pub n_bins: usize,
    /// Strategy for binning: "uniform" or "quantile"
    pub bin_strategy: BinStrategy,
}

impl Default for CalibrationMetricsConfig {
    fn default() -> Self {
        Self {
            n_bins: 10,
            bin_strategy: BinStrategy::Uniform,
        }
    }
}

/// Binning strategy for calibration metrics
#[derive(Debug, Clone)]
pub enum BinStrategy {
    /// Uniform binning (equal width bins)
    Uniform,
    /// Quantile binning (equal frequency bins)
    Quantile,
}

/// Reliability diagram data for calibration visualization
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Bin boundaries
    pub bin_boundaries: Array1<Float>,
    /// Mean predicted probability for each bin
    pub bin_mean_pred: Array1<Float>,
    /// True frequency for each bin
    pub bin_true_freq: Array1<Float>,
    /// Number of samples in each bin
    pub bin_counts: Array1<usize>,
    /// Confidence intervals for true frequencies
    pub confidence_intervals: Option<Array2<Float>>,
}

/// Brier score decomposition components
#[derive(Debug, Clone)]
pub struct BrierScoreDecomposition {
    /// Total Brier score
    pub brier_score: Float,
    /// Reliability (miscalibration)
    pub reliability: Float,
    /// Resolution (discrimination)
    pub resolution: Float,
    /// Uncertainty (base rate)
    pub uncertainty: Float,
}

/// Expected Calibration Error (ECE)
pub fn expected_calibration_error(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationMetricsConfig,
) -> Result<Float> {
    let reliability = reliability_diagram(y_true, y_prob, config)?;

    let mut ece = 0.0;
    let total_samples = y_true.len() as Float;

    for i in 0..reliability.bin_counts.len() {
        let bin_count = reliability.bin_counts[i] as Float;
        if bin_count > 0.0 {
            let weight = bin_count / total_samples;
            let calibration_error =
                (reliability.bin_mean_pred[i] - reliability.bin_true_freq[i]).abs();
            ece += weight * calibration_error;
        }
    }

    Ok(ece)
}

/// Maximum Calibration Error (MCE)
pub fn maximum_calibration_error(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationMetricsConfig,
) -> Result<Float> {
    let reliability = reliability_diagram(y_true, y_prob, config)?;

    let mut mce: Float = 0.0;

    for i in 0..reliability.bin_counts.len() {
        if reliability.bin_counts[i] > 0 {
            let calibration_error =
                (reliability.bin_mean_pred[i] - reliability.bin_true_freq[i]).abs();
            mce = mce.max(calibration_error);
        }
    }

    Ok(mce)
}

/// Adaptive Calibration Error (ACE) - uses adaptive binning
pub fn adaptive_calibration_error(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    max_bins: usize,
) -> Result<Float> {
    // Implement adaptive binning based on distribution
    let config = CalibrationMetricsConfig {
        n_bins: max_bins.min(y_true.len() / 10).max(2),
        bin_strategy: BinStrategy::Quantile,
    };

    expected_calibration_error(y_true, y_prob, &config)
}

/// Generate reliability diagram data
pub fn reliability_diagram(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationMetricsConfig,
) -> Result<ReliabilityDiagram> {
    if y_true.len() != y_prob.len() {
        return Err(sklears_core::error::SklearsError::InvalidInput(
            "Input arrays must have the same length".to_string(),
        ));
    }

    if y_true.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidInput(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Create bin boundaries
    let bin_boundaries = match config.bin_strategy {
        BinStrategy::Uniform => create_uniform_bins(config.n_bins),
        BinStrategy::Quantile => create_quantile_bins(y_prob, config.n_bins),
    };

    let mut bin_mean_pred = Array1::zeros(config.n_bins);
    let mut bin_true_freq = Array1::zeros(config.n_bins);
    let mut bin_counts = Array1::zeros(config.n_bins);

    // Assign samples to bins and calculate statistics
    for (&prob, &_true_label) in y_prob.iter().zip(y_true.iter()) {
        let bin_idx = find_bin_index(prob, &bin_boundaries);
        if bin_idx < config.n_bins {
            bin_counts[bin_idx] += 1;
        }
    }

    // Calculate bin statistics
    let mut bin_prob_sums = vec![0.0; config.n_bins];
    let mut bin_true_sums = vec![0.0; config.n_bins];

    for (&prob, &true_label) in y_prob.iter().zip(y_true.iter()) {
        let bin_idx = find_bin_index(prob, &bin_boundaries);
        if bin_idx < config.n_bins {
            bin_prob_sums[bin_idx] += prob;
            bin_true_sums[bin_idx] += true_label as Float;
        }
    }

    for i in 0..config.n_bins {
        if bin_counts[i] > 0 {
            bin_mean_pred[i] = bin_prob_sums[i] / bin_counts[i] as Float;
            bin_true_freq[i] = bin_true_sums[i] / bin_counts[i] as Float;
        }
    }

    // Compute confidence intervals for each bin using Wilson method
    let confidence_intervals = compute_confidence_intervals(&bin_true_freq, &bin_counts, 0.95);

    Ok(ReliabilityDiagram {
        bin_boundaries,
        bin_mean_pred,
        bin_true_freq,
        bin_counts,
        confidence_intervals: Some(confidence_intervals),
    })
}

/// Compute confidence intervals for bin true frequencies using Wilson method
fn compute_confidence_intervals(
    bin_true_freq: &Array1<Float>,
    bin_counts: &Array1<usize>,
    confidence_level: Float,
) -> Array2<Float> {
    let n_bins = bin_true_freq.len();
    let mut confidence_intervals = Array2::zeros((n_bins, 2));

    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            let (lower, upper) =
                wilson_confidence_interval(bin_true_freq[i], bin_counts[i], confidence_level);
            confidence_intervals[[i, 0]] = lower;
            confidence_intervals[[i, 1]] = upper;
        } else {
            // For empty bins, use wide confidence interval
            confidence_intervals[[i, 0]] = 0.0;
            confidence_intervals[[i, 1]] = 1.0;
        }
    }

    confidence_intervals
}

/// Wilson confidence interval for proportions
fn wilson_confidence_interval(p: Float, n: usize, confidence_level: Float) -> (Float, Float) {
    if n == 0 {
        return (0.0, 1.0);
    }

    let z = match confidence_level {
        0.90 => 1.645,
        0.95 => 1.96,
        0.99 => 2.576,
        _ => 1.96, // Default to 95%
    };

    let n_f = n as Float;
    let center = (p + z * z / (2.0 * n_f)) / (1.0 + z * z / n_f);
    let margin = z * (p * (1.0 - p) / n_f + z * z / (4.0 * n_f * n_f)).sqrt() / (1.0 + z * z / n_f);

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}

/// Brier score and its decomposition
pub fn brier_score_decomposition(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationMetricsConfig,
) -> Result<BrierScoreDecomposition> {
    if y_true.len() != y_prob.len() {
        return Err(sklears_core::error::SklearsError::InvalidInput(
            "Input arrays must have the same length".to_string(),
        ));
    }

    let n_samples = y_true.len() as Float;

    // Calculate overall base rate
    let base_rate = y_true.iter().map(|&y| y as Float).sum::<Float>() / n_samples;

    // Calculate Brier score
    let brier_score = y_true
        .iter()
        .zip(y_prob.iter())
        .map(|(&y, &p)| (p - y as Float).powi(2))
        .sum::<Float>()
        / n_samples;

    // Get reliability diagram for decomposition
    let reliability = reliability_diagram(y_true, y_prob, config)?;

    // Calculate reliability (miscalibration)
    let mut reliability_score = 0.0;
    for i in 0..reliability.bin_counts.len() {
        let bin_count = reliability.bin_counts[i] as Float;
        if bin_count > 0.0 {
            let weight = bin_count / n_samples;
            let calibration_error =
                (reliability.bin_mean_pred[i] - reliability.bin_true_freq[i]).powi(2);
            reliability_score += weight * calibration_error;
        }
    }

    // Calculate resolution (discrimination)
    let mut resolution = 0.0;
    for i in 0..reliability.bin_counts.len() {
        let bin_count = reliability.bin_counts[i] as Float;
        if bin_count > 0.0 {
            let weight = bin_count / n_samples;
            let discrimination = (reliability.bin_true_freq[i] - base_rate).powi(2);
            resolution += weight * discrimination;
        }
    }

    // Uncertainty (base rate variance)
    let uncertainty = base_rate * (1.0 - base_rate);

    Ok(BrierScoreDecomposition {
        brier_score,
        reliability: reliability_score,
        resolution,
        uncertainty,
    })
}

/// Hosmer-Lemeshow goodness-of-fit test statistic
pub fn hosmer_lemeshow_test(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    n_bins: usize,
) -> Result<Float> {
    let config = CalibrationMetricsConfig {
        n_bins,
        bin_strategy: BinStrategy::Quantile,
    };

    let reliability = reliability_diagram(y_true, y_prob, &config)?;

    let mut chi_square = 0.0;

    for i in 0..reliability.bin_counts.len() {
        let n_i = reliability.bin_counts[i] as Float;
        if n_i > 0.0 {
            let expected_pos = n_i * reliability.bin_mean_pred[i];
            let observed_pos = n_i * reliability.bin_true_freq[i];
            let expected_neg = n_i * (1.0 - reliability.bin_mean_pred[i]);
            let observed_neg = n_i * (1.0 - reliability.bin_true_freq[i]);

            if expected_pos > 0.0 && expected_neg > 0.0 {
                chi_square += (observed_pos - expected_pos).powi(2) / expected_pos;
                chi_square += (observed_neg - expected_neg).powi(2) / expected_neg;
            }
        }
    }

    Ok(chi_square)
}

/// Chi-squared calibration test
///
/// Tests the null hypothesis that the predicted probabilities are well-calibrated
/// by comparing observed vs expected frequencies in bins.
/// Returns the chi-squared test statistic and p-value.
pub fn chi_squared_calibration_test(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    n_bins: usize,
) -> Result<(Float, Float)> {
    let config = CalibrationMetricsConfig {
        n_bins,
        bin_strategy: BinStrategy::Uniform,
    };

    let reliability = reliability_diagram(y_true, y_prob, &config)?;

    let mut chi_square = 0.0;
    let mut degrees_of_freedom = 0;

    for i in 0..reliability.bin_counts.len() {
        let n_i = reliability.bin_counts[i] as Float;
        if n_i >= 5.0 {
            // Standard requirement for chi-squared test
            let expected_pos = n_i * reliability.bin_mean_pred[i];
            let observed_pos = n_i * reliability.bin_true_freq[i];
            let expected_neg = n_i * (1.0 - reliability.bin_mean_pred[i]);
            let observed_neg = n_i * (1.0 - reliability.bin_true_freq[i]);

            if expected_pos > 0.0 && expected_neg > 0.0 {
                chi_square += (observed_pos - expected_pos).powi(2) / expected_pos;
                chi_square += (observed_neg - expected_neg).powi(2) / expected_neg;
                degrees_of_freedom += 1;
            }
        }
    }

    // Calculate p-value using chi-squared distribution
    let p_value = if degrees_of_freedom > 0 {
        chi_squared_p_value(chi_square, degrees_of_freedom)
    } else {
        1.0 // Can't reject null hypothesis
    };

    Ok((chi_square, p_value))
}

/// Approximate p-value for chi-squared distribution
/// Using a simple approximation for small degrees of freedom
fn chi_squared_p_value(chi_square: Float, df: i32) -> Float {
    if df <= 0 {
        return 1.0;
    }

    // For small degrees of freedom, use a simple approximation
    // This is not as accurate as a proper implementation but works for basic testing
    match df {
        1 => {
            if chi_square > 3.84 {
                0.05
            } else if chi_square > 2.71 {
                0.10
            } else if chi_square > 1.64 {
                0.20
            } else {
                0.50
            }
        }
        2 => {
            if chi_square > 5.99 {
                0.05
            } else if chi_square > 4.61 {
                0.10
            } else if chi_square > 3.22 {
                0.20
            } else {
                0.50
            }
        }
        3 => {
            if chi_square > 7.81 {
                0.05
            } else if chi_square > 6.25 {
                0.10
            } else if chi_square > 4.64 {
                0.20
            } else {
                0.50
            }
        }
        _ => {
            // For higher df, use a rough approximation
            let critical_05 = df as Float * 1.5 + 2.0;
            let critical_10 = df as Float * 1.2 + 1.5;
            let critical_20 = df as Float * 1.0 + 1.0;

            if chi_square > critical_05 {
                0.05
            } else if chi_square > critical_10 {
                0.10
            } else if chi_square > critical_20 {
                0.20
            } else {
                0.50
            }
        }
    }
}

/// Kolmogorov-Smirnov test for calibration
///
/// Tests whether the predicted probabilities follow a uniform distribution
/// when conditioned on being correct/incorrect.
/// Returns the KS test statistic and approximate p-value.
pub fn kolmogorov_smirnov_calibration_test(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
) -> Result<(Float, Float)> {
    if y_true.len() != y_prob.len() {
        return Err(sklears_core::error::SklearsError::InvalidInput(
            "Input arrays must have the same length".to_string(),
        ));
    }

    if y_true.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidInput(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Separate probabilities for correct and incorrect predictions
    let mut correct_probs = Vec::new();
    let mut incorrect_probs = Vec::new();

    for (&prob, &true_label) in y_prob.iter().zip(y_true.iter()) {
        if true_label > 0 {
            correct_probs.push(prob);
        } else {
            incorrect_probs.push(1.0 - prob); // For incorrect, we want 1-p
        }
    }

    // For well-calibrated probabilities, both should be uniformly distributed
    let ks_stat_correct = if !correct_probs.is_empty() {
        ks_test_uniform(&correct_probs)
    } else {
        0.0
    };

    let ks_stat_incorrect = if !incorrect_probs.is_empty() {
        ks_test_uniform(&incorrect_probs)
    } else {
        0.0
    };

    // Take the maximum of the two test statistics
    let ks_statistic = ks_stat_correct.max(ks_stat_incorrect);

    // Approximate p-value based on sample size
    let n = y_true.len() as Float;
    let p_value = ks_p_value_approximation(ks_statistic, n);

    Ok((ks_statistic, p_value))
}

/// Kolmogorov-Smirnov test against uniform distribution
fn ks_test_uniform(data: &[Float]) -> Float {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted_data.len() as Float;
    let mut max_diff: Float = 0.0;

    for (i, &value) in sorted_data.iter().enumerate() {
        let empirical_cdf = (i + 1) as Float / n;
        let theoretical_cdf = value; // Uniform[0,1] CDF is just x

        let diff = (empirical_cdf - theoretical_cdf).abs();
        max_diff = max_diff.max(diff);

        // Also check the step before
        if i > 0 {
            let empirical_cdf_prev = i as Float / n;
            let diff_prev = (empirical_cdf_prev - theoretical_cdf).abs();
            max_diff = max_diff.max(diff_prev);
        }
    }

    max_diff
}

/// Approximate p-value for KS test
fn ks_p_value_approximation(ks_stat: Float, n: Float) -> Float {
    if ks_stat <= 0.0 {
        return 1.0;
    }

    // Use asymptotic approximation: P(K > x) ≈ 2 * exp(-2 * x^2)
    // where K is the KS statistic normalized by sqrt(n)
    let normalized_stat = ks_stat * n.sqrt();
    let p_value = 2.0 * (-2.0 * normalized_stat.powi(2)).exp();

    // Clamp to [0, 1]
    p_value.clamp(0.0, 1.0)
}

/// Binomial test for calibration
///
/// Tests whether observed frequencies match expected frequencies for each bin
/// using binomial distribution. Returns test statistics and p-values for each bin.
pub fn binomial_calibration_test(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationMetricsConfig,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let reliability = reliability_diagram(y_true, y_prob, config)?;

    let n_bins = reliability.bin_counts.len();
    let mut test_statistics = Array1::zeros(n_bins);
    let mut p_values = Array1::zeros(n_bins);

    for i in 0..n_bins {
        let n_i = reliability.bin_counts[i];
        if n_i >= 5 {
            // Minimum sample size for reliable test
            let expected_p = reliability.bin_mean_pred[i];
            let observed_successes = (n_i as Float * reliability.bin_true_freq[i]).round() as usize;

            // Binomial test: test if observed_successes ~ Binomial(n_i, expected_p)
            let (test_stat, p_value) = binomial_test(observed_successes, n_i, expected_p);
            test_statistics[i] = test_stat;
            p_values[i] = p_value;
        } else {
            // Not enough samples for reliable test
            test_statistics[i] = 0.0;
            p_values[i] = 1.0;
        }
    }

    Ok((test_statistics, p_values))
}

/// Perform binomial test
/// Returns (z-statistic, p-value) for testing if k successes in n trials
/// matches expected probability p
fn binomial_test(k: usize, n: usize, p: Float) -> (Float, Float) {
    if n == 0 {
        return (0.0, 1.0);
    }

    let n_f = n as Float;
    let k_f = k as Float;

    // Expected value and variance under null hypothesis
    let expected = n_f * p;
    let variance = n_f * p * (1.0 - p);

    if variance <= 0.0 {
        return (0.0, 1.0);
    }

    // Continuity correction for normal approximation
    let correction = if k_f > expected { 0.5 } else { -0.5 };
    let z_stat = (k_f - expected + correction) / variance.sqrt();

    // Two-tailed p-value using normal approximation
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));

    (z_stat, p_value.clamp(0.0, 1.0))
}

/// Approximate standard normal CDF
fn standard_normal_cdf(x: Float) -> Float {
    // Using erf approximation: Φ(x) = 0.5 * (1 + erf(x/√2))
    // Simple polynomial approximation for erf
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * abs_x);
    let erf_approx = 1.0
        - (0.254829592 - 0.284496736 * t + 1.421413741 * t * t - 1.453152027 * t * t * t
            + 1.061405429 * t * t * t * t)
            * (-abs_x * abs_x).exp();

    let result = 0.5 * (1.0 + if x >= 0.0 { erf_approx } else { -erf_approx });
    result.clamp(0.0, 1.0)
}

/// Bootstrap confidence intervals for calibration metrics
///
/// Computes bootstrap confidence intervals for ECE and other calibration metrics
/// Returns (lower_bound, upper_bound) for the specified confidence level
pub fn bootstrap_calibration_confidence_intervals(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationMetricsConfig,
    confidence_level: Float,
    n_bootstrap: usize,
) -> Result<(Float, Float)> {
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(SklearsError::InvalidInput(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    if n_bootstrap < 10 {
        return Err(SklearsError::InvalidInput(
            "Number of bootstrap samples must be at least 10".to_string(),
        ));
    }

    let n_samples = y_true.len();
    if n_samples < 10 {
        return Err(SklearsError::InvalidInput(
            "Need at least 10 samples for bootstrap".to_string(),
        ));
    }

    let mut bootstrap_eces = Vec::with_capacity(n_bootstrap);

    // Simple pseudo-random number generator (LCG)
    let mut rng_state = 12345u64;

    for _ in 0..n_bootstrap {
        // Bootstrap sample with replacement
        let mut bootstrap_y_true = Array1::zeros(n_samples);
        let mut bootstrap_y_prob = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Simple LCG for random index
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let random_idx = (rng_state % n_samples as u64) as usize;

            bootstrap_y_true[i] = y_true[random_idx];
            bootstrap_y_prob[i] = y_prob[random_idx];
        }

        // Compute ECE for bootstrap sample
        if let Ok(ece) = expected_calibration_error(&bootstrap_y_true, &bootstrap_y_prob, config) {
            bootstrap_eces.push(ece);
        }
    }

    if bootstrap_eces.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid bootstrap samples generated".to_string(),
        ));
    }

    // Sort bootstrap ECEs
    bootstrap_eces.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate percentiles for confidence interval
    let alpha = 1.0 - confidence_level;
    let lower_percentile = alpha / 2.0;
    let upper_percentile = 1.0 - alpha / 2.0;

    let lower_idx = (lower_percentile * bootstrap_eces.len() as Float) as usize;
    let upper_idx = (upper_percentile * bootstrap_eces.len() as Float) as usize;

    let lower_bound = bootstrap_eces[lower_idx.min(bootstrap_eces.len() - 1)];
    let upper_bound = bootstrap_eces[upper_idx.min(bootstrap_eces.len() - 1)];

    Ok((lower_bound, upper_bound))
}

/// Advanced Adaptive Calibration Error
///
/// Computes calibration error using adaptive binning that adjusts bin sizes
/// based on data density to provide more reliable estimates
pub fn advanced_adaptive_calibration_error(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    min_bin_size: usize,
) -> Result<Float> {
    if y_true.len() != y_prob.len() {
        return Err(SklearsError::InvalidInput(
            "Input arrays must have same length".to_string(),
        ));
    }

    let n_samples = y_true.len();
    if n_samples < min_bin_size * 2 {
        return Err(SklearsError::InvalidInput(
            "Not enough samples for adaptive binning".to_string(),
        ));
    }

    // Create sorted indices by probability
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.sort_by(|&a, &b| y_prob[a].partial_cmp(&y_prob[b]).unwrap());

    let mut total_error = 0.0;
    let mut processed_samples = 0;

    // Adaptive binning: create bins with at least min_bin_size samples
    while processed_samples < n_samples {
        let remaining_samples = n_samples - processed_samples;
        let bin_size = if remaining_samples < min_bin_size * 2 {
            // Use all remaining samples for the last bin
            remaining_samples
        } else {
            min_bin_size
        };

        // Extract bin data
        let bin_indices = &indices[processed_samples..processed_samples + bin_size];

        // Calculate bin statistics
        let mut bin_positives = 0;
        let mut bin_prob_sum = 0.0;

        for &idx in bin_indices {
            if y_true[idx] > 0 {
                bin_positives += 1;
            }
            bin_prob_sum += y_prob[idx];
        }

        let bin_mean_prob = bin_prob_sum / bin_size as Float;
        let bin_true_freq = bin_positives as Float / bin_size as Float;
        let bin_weight = bin_size as Float / n_samples as Float;

        // Accumulate weighted calibration error
        let calibration_error = (bin_mean_prob - bin_true_freq).abs();
        total_error += bin_weight * calibration_error;

        processed_samples += bin_size;
    }

    Ok(total_error)
}

// Helper functions

fn create_uniform_bins(n_bins: usize) -> Array1<Float> {
    let mut boundaries = Array1::zeros(n_bins + 1);
    for i in 0..=n_bins {
        boundaries[i] = i as Float / n_bins as Float;
    }
    boundaries
}

fn create_quantile_bins(probabilities: &Array1<Float>, n_bins: usize) -> Array1<Float> {
    let mut sorted_probs: Vec<Float> = probabilities.to_vec();
    sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut boundaries = Array1::zeros(n_bins + 1);
    boundaries[0] = 0.0;
    boundaries[n_bins] = 1.0;

    for i in 1..n_bins {
        let quantile_pos = (i * sorted_probs.len()) / n_bins;
        if quantile_pos < sorted_probs.len() {
            boundaries[i] = sorted_probs[quantile_pos];
        } else {
            boundaries[i] = 1.0;
        }
    }

    boundaries
}

fn find_bin_index(prob: Float, bin_boundaries: &Array1<Float>) -> usize {
    for i in 0..bin_boundaries.len() - 1 {
        if prob >= bin_boundaries[i] && prob < bin_boundaries[i + 1] {
            return i;
        }
    }
    // Handle the case where prob == 1.0
    if prob == 1.0 {
        return bin_boundaries.len() - 2;
    }
    0
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_expected_calibration_error() {
        let y_true = array![0, 0, 1, 1];
        let y_prob = array![0.1, 0.3, 0.7, 0.9];
        let config = CalibrationMetricsConfig::default();

        let ece = expected_calibration_error(&y_true, &y_prob, &config).unwrap();

        assert!((0.0..=1.0).contains(&ece));
    }

    #[test]
    fn test_maximum_calibration_error() {
        let y_true = array![0, 0, 1, 1];
        let y_prob = array![0.1, 0.3, 0.7, 0.9];
        let config = CalibrationMetricsConfig::default();

        let mce = maximum_calibration_error(&y_true, &y_prob, &config).unwrap();

        assert!((0.0..=1.0).contains(&mce));
    }

    #[test]
    fn test_reliability_diagram() {
        let y_true = array![0, 0, 1, 1, 0, 1, 1, 0];
        let y_prob = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4];
        let config = CalibrationMetricsConfig {
            n_bins: 4,
            bin_strategy: BinStrategy::Uniform,
        };

        let diagram = reliability_diagram(&y_true, &y_prob, &config).unwrap();

        assert_eq!(diagram.bin_boundaries.len(), 5);
        assert_eq!(diagram.bin_mean_pred.len(), 4);
        assert_eq!(diagram.bin_true_freq.len(), 4);
        assert_eq!(diagram.bin_counts.len(), 4);
    }

    #[test]
    fn test_brier_score_decomposition() {
        let y_true = array![0, 0, 1, 1];
        let y_prob = array![0.1, 0.3, 0.7, 0.9];
        let config = CalibrationMetricsConfig::default();

        let decomp = brier_score_decomposition(&y_true, &y_prob, &config).unwrap();

        assert!(decomp.brier_score >= 0.0);
        assert!(decomp.reliability >= 0.0);
        assert!(decomp.resolution >= 0.0);
        assert!(decomp.uncertainty >= 0.0 && decomp.uncertainty <= 0.25);

        // Brier score decomposition: BS = Reliability - Resolution + Uncertainty
        let expected_bs = decomp.reliability - decomp.resolution + decomp.uncertainty;
        assert!((decomp.brier_score - expected_bs).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_calibration() {
        // Perfect calibration case
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1];
        let y_prob = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let config = CalibrationMetricsConfig::default();

        let ece = expected_calibration_error(&y_true, &y_prob, &config).unwrap();
        let mce = maximum_calibration_error(&y_true, &y_prob, &config).unwrap();

        // Should be very small for perfect calibration
        assert!(ece < 1e-10);
        assert!(mce < 1e-10);
    }

    #[test]
    fn test_hosmer_lemeshow() {
        let y_true = array![0, 0, 1, 1, 0, 1];
        let y_prob = array![0.1, 0.3, 0.7, 0.9, 0.2, 0.8];

        let chi_square = hosmer_lemeshow_test(&y_true, &y_prob, 3).unwrap();

        assert!(chi_square >= 0.0);
    }

    #[test]
    fn test_chi_squared_calibration_test() {
        let y_true = array![0, 0, 1, 1, 0, 1, 1, 0];
        let y_prob = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4];

        let (chi_square, p_value) = chi_squared_calibration_test(&y_true, &y_prob, 4).unwrap();

        assert!(chi_square >= 0.0);
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[test]
    fn test_kolmogorov_smirnov_test() {
        let y_true = array![0, 0, 1, 1, 0, 1, 1, 0];
        let y_prob = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4];

        let (ks_stat, p_value) = kolmogorov_smirnov_calibration_test(&y_true, &y_prob).unwrap();

        assert!((0.0..=1.0).contains(&ks_stat));
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[test]
    fn test_statistical_tests_well_calibrated() {
        // Test with well-calibrated data
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1];
        let y_prob = array![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];

        let (chi_square, chi_p) = chi_squared_calibration_test(&y_true, &y_prob, 4).unwrap();
        let (ks_stat, ks_p) = kolmogorov_smirnov_calibration_test(&y_true, &y_prob).unwrap();

        // For well-calibrated data, we expect higher p-values (less significant)
        assert!(chi_square >= 0.0);
        assert!((0.0..=1.0).contains(&chi_p));
        assert!((0.0..=1.0).contains(&ks_stat));
        assert!((0.0..=1.0).contains(&ks_p));
    }

    #[test]
    fn test_statistical_tests_poorly_calibrated() {
        // Test with poorly calibrated data (overconfident)
        let y_true = array![0, 0, 1, 1];
        let y_prob = array![0.01, 0.05, 0.95, 0.99];

        let (chi_square, chi_p) = chi_squared_calibration_test(&y_true, &y_prob, 2).unwrap();
        let (ks_stat, ks_p) = kolmogorov_smirnov_calibration_test(&y_true, &y_prob).unwrap();

        // For poorly calibrated data, we expect higher test statistics
        assert!(chi_square >= 0.0);
        assert!((0.0..=1.0).contains(&chi_p));
        assert!((0.0..=1.0).contains(&ks_stat));
        assert!((0.0..=1.0).contains(&ks_p));
    }

    #[test]
    fn test_binomial_calibration_test() {
        let y_true = array![0, 0, 1, 1, 0, 1, 1, 0, 1, 0];
        let y_prob = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15];
        let config = CalibrationMetricsConfig {
            n_bins: 3,
            bin_strategy: BinStrategy::Uniform,
        };

        let (test_stats, p_values) = binomial_calibration_test(&y_true, &y_prob, &config).unwrap();

        assert_eq!(test_stats.len(), 3);
        assert_eq!(p_values.len(), 3);

        // All p-values should be between 0 and 1
        for &p in p_values.iter() {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_bootstrap_confidence_intervals() {
        let y_true = array![0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1];
        let y_prob = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15, 0.25, 0.75];
        let config = CalibrationMetricsConfig::default();

        let (lower, upper) =
            bootstrap_calibration_confidence_intervals(&y_true, &y_prob, &config, 0.95, 100)
                .unwrap();

        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
        assert!(lower <= upper);
    }

    #[test]
    fn test_adaptive_calibration_error() {
        let y_true = array![0, 0, 1, 1, 0, 1, 1, 0, 1, 0];
        let y_prob = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15];

        let ace = advanced_adaptive_calibration_error(&y_true, &y_prob, 3).unwrap();

        assert!((0.0..=1.0).contains(&ace));
    }

    #[test]
    fn test_standard_normal_cdf() {
        // Test known values
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(standard_normal_cdf(-2.0) < 0.1);
        assert!(standard_normal_cdf(2.0) > 0.9);
        assert!(standard_normal_cdf(-5.0) < 0.01);
        assert!(standard_normal_cdf(5.0) > 0.99);
    }

    #[test]
    fn test_binomial_test() {
        // Test exact cases
        let (_z_stat, p_value) = binomial_test(5, 10, 0.5);
        assert!((0.0..=1.0).contains(&p_value));

        // Test extreme cases
        let (_z_stat2, p_value2) = binomial_test(0, 10, 0.8);
        assert!(p_value2 < 0.1); // Should be significant

        let (_z_stat3, p_value3) = binomial_test(5, 10, 0.5);
        assert!(p_value3 > 0.1); // Should not be significant
    }

    #[test]
    fn test_bootstrap_edge_cases() {
        let y_true = array![0, 1];
        let y_prob = array![0.3, 0.7];
        let config = CalibrationMetricsConfig::default();

        // Test with very small sample
        let result =
            bootstrap_calibration_confidence_intervals(&y_true, &y_prob, &config, 0.95, 10);
        assert!(result.is_err()); // Should fail with too few samples

        // Test with invalid confidence level
        let y_true_large = array![0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1];
        let y_prob_large = array![0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15, 0.25, 0.75];

        let result2 = bootstrap_calibration_confidence_intervals(
            &y_true_large,
            &y_prob_large,
            &config,
            1.5,
            100,
        );
        assert!(result2.is_err()); // Should fail with invalid confidence level
    }
}
