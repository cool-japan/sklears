//! Statistical tests for model comparison and evaluation
//!
//! This module provides various statistical tests used in machine learning
//! for comparing models, testing hypotheses, and validating results.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive};
use std::collections::HashMap;

/// McNemar's test for comparing two binary classifiers
///
/// Tests the null hypothesis that two classifiers have the same error rate
/// on a test set. Uses the McNemar test statistic based on disagreements.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_pred1` - Predictions from first classifier
/// * `y_pred2` - Predictions from second classifier
/// * `exact` - Whether to use exact binomial test (for small samples)
///
/// # Returns
/// Tuple of (test statistic, p-value)
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::statistical_tests::mcnemar_test;
/// use scirs2_core::ndarray::array;
///
/// let y_true = array![1, 0, 1, 1, 0, 1, 0, 0];
/// let y_pred1 = array![1, 0, 1, 0, 0, 1, 1, 0];
/// let y_pred2 = array![1, 1, 1, 1, 0, 0, 0, 0];
/// let (statistic, p_value): (f64, f64) = mcnemar_test(&y_true, &y_pred1, &y_pred2, false).unwrap();
/// ```
pub fn mcnemar_test<F: FloatTrait + FromPrimitive>(
    y_true: &Array1<i32>,
    y_pred1: &Array1<i32>,
    y_pred2: &Array1<i32>,
    exact: bool,
) -> MetricsResult<(F, F)> {
    if y_true.len() != y_pred1.len() || y_true.len() != y_pred2.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred1.len(), y_pred2.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check that all labels are binary (0 or 1)
    for &label in y_true.iter().chain(y_pred1.iter()).chain(y_pred2.iter()) {
        if label != 0 && label != 1 {
            return Err(MetricsError::InvalidParameter(
                "All labels must be binary (0 or 1)".to_string(),
            ));
        }
    }

    // Build confusion matrix for McNemar test
    // McNemar focuses on disagreements between classifiers
    let mut n_01 = 0; // classifier 1 wrong, classifier 2 right
    let mut n_10 = 0; // classifier 1 right, classifier 2 wrong

    for i in 0..y_true.len() {
        let correct1 = y_pred1[i] == y_true[i];
        let correct2 = y_pred2[i] == y_true[i];

        match (correct1, correct2) {
            (false, true) => n_01 += 1,
            (true, false) => n_10 += 1,
            _ => {} // Both right or both wrong - not used in McNemar test
        }
    }

    let n_discordant = n_01 + n_10;

    if n_discordant == 0 {
        // No disagreements, classifiers perform identically
        return Ok((F::zero(), F::one()));
    }

    if exact || n_discordant < 25 {
        // Use exact binomial test
        let p_value = binomial_test(n_01, n_discordant, 0.5);
        Ok((F::from(n_01).unwrap(), F::from(p_value).unwrap()))
    } else {
        // Use chi-square approximation with continuity correction
        let numerator = (F::from(n_01).unwrap() - F::from(n_10).unwrap()).abs() - F::one();
        let denominator = F::from(n_01 + n_10).unwrap();
        let chi_square = numerator * numerator / denominator;

        // Approximate p-value using chi-square distribution with 1 df
        let p_value = chi_square_p_value(chi_square);

        Ok((chi_square, p_value))
    }
}

/// Friedman test for comparing multiple algorithms across multiple datasets
///
/// Non-parametric test that ranks algorithms on each dataset and tests
/// if all algorithms perform equally well.
///
/// # Arguments
/// * `scores` - Matrix where rows are datasets and columns are algorithms
///
/// # Returns
/// Tuple of (Friedman statistic, p-value, critical difference for post-hoc analysis)
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::statistical_tests::friedman_test;
/// use scirs2_core::ndarray::array;
///
/// // 3 datasets, 4 algorithms
/// let scores = array![
///     [0.85, 0.82, 0.88, 0.84],
///     [0.92, 0.89, 0.91, 0.87],
///     [0.78, 0.75, 0.80, 0.76]
/// ];
/// let (statistic, p_value, cd) = friedman_test(&scores).unwrap();
/// ```
pub fn friedman_test<F: FloatTrait + FromPrimitive>(
    scores: &Array2<F>,
) -> MetricsResult<(F, F, F)> {
    let (n_datasets, n_algorithms) = scores.dim();

    if n_datasets < 2 || n_algorithms < 2 {
        return Err(MetricsError::InvalidParameter(
            "Need at least 2 datasets and 2 algorithms".to_string(),
        ));
    }

    // Rank algorithms within each dataset (higher score = better rank)
    let mut rank_sums = vec![F::zero(); n_algorithms];

    for i in 0..n_datasets {
        let mut row_with_indices: Vec<(F, usize)> = scores
            .row(i)
            .iter()
            .enumerate()
            .map(|(idx, &score)| (score, idx))
            .collect();

        // Sort by score (descending for better = higher rank)
        row_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks (1 = best, n_algorithms = worst)
        for (rank, (_, alg_idx)) in row_with_indices.iter().enumerate() {
            rank_sums[*alg_idx] = rank_sums[*alg_idx] + F::from(rank + 1).unwrap();
        }
    }

    // Friedman statistic
    let n_datasets_f = F::from(n_datasets).unwrap();
    let n_algorithms_f = F::from(n_algorithms).unwrap();

    let mean_rank = (n_algorithms_f + F::one()) / F::from(2).unwrap();
    let sum_squared_deviations = rank_sums
        .iter()
        .map(|&rank_sum| {
            let mean_rank_for_alg = rank_sum / n_datasets_f;
            let deviation = mean_rank_for_alg - mean_rank;
            deviation * deviation
        })
        .fold(F::zero(), |acc, x| acc + x);

    let friedman_stat = F::from(12).unwrap() * n_datasets_f * sum_squared_deviations
        / (n_algorithms_f * (n_algorithms_f + F::one()));

    // For large samples, Friedman statistic follows chi-square with (k-1) df
    let p_value = chi_square_p_value_df(friedman_stat, n_algorithms as i32 - 1);

    // Critical difference for Nemenyi post-hoc test
    let q_alpha = F::from(2.576).unwrap(); // Critical value for alpha=0.01 (approximation)
    let critical_difference = q_alpha
        * (n_algorithms_f * (n_algorithms_f + F::one()) / (F::from(6).unwrap() * n_datasets_f))
            .sqrt();

    Ok((friedman_stat, p_value, critical_difference))
}

/// Wilcoxon signed-rank test for paired samples
///
/// Non-parametric test for comparing two related samples to assess
/// whether their population mean ranks differ.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample (paired with x)
/// * `alternative` - Test alternative ("two-sided", "greater", "less")
///
/// # Returns
/// Tuple of (W statistic, p-value)
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::statistical_tests::wilcoxon_signed_rank_test;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.2, 2.4, 1.8, 2.1, 1.7, 2.0, 1.9, 2.2];
/// let y = array![1.0, 2.0, 1.5, 1.8, 1.4, 1.8, 1.7, 2.0];
/// let (statistic, p_value) = wilcoxon_signed_rank_test(&x, &y, "two-sided").unwrap();
/// ```
pub fn wilcoxon_signed_rank_test<F: FloatTrait + FromPrimitive>(
    x: &Array1<F>,
    y: &Array1<F>,
    alternative: &str,
) -> MetricsResult<(F, F)> {
    if x.len() != y.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        });
    }

    if x.len() < 6 {
        return Err(MetricsError::InvalidParameter(
            "Need at least 6 paired observations for Wilcoxon test".to_string(),
        ));
    }

    // Compute differences
    let mut differences: Vec<F> = Vec::new();
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        if diff != F::zero() {
            // Skip zero differences
            differences.push(diff);
        }
    }

    if differences.is_empty() {
        return Ok((F::zero(), F::one()));
    }

    let n = differences.len();

    // Rank absolute differences
    let mut abs_diffs_with_indices: Vec<(F, usize, bool)> = differences
        .iter()
        .enumerate()
        .map(|(idx, &diff)| (diff.abs(), idx, diff > F::zero()))
        .collect();

    abs_diffs_with_indices
        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks and compute W+ (sum of ranks for positive differences)
    let mut w_plus = F::zero();
    let mut current_rank = F::one();

    let mut i = 0;
    while i < abs_diffs_with_indices.len() {
        let current_value = abs_diffs_with_indices[i].0;
        let start_idx = i;

        // Find all tied values
        while i < abs_diffs_with_indices.len() && abs_diffs_with_indices[i].0 == current_value {
            i += 1;
        }

        // Average rank for tied values
        let end_idx = i;
        let avg_rank = (current_rank + F::from(end_idx - start_idx - 1).unwrap())
            / F::from(2).unwrap()
            + current_rank / F::from(2).unwrap();

        // Assign average rank to all tied values
        for item in abs_diffs_with_indices.iter().take(end_idx).skip(start_idx) {
            if item.2 {
                // Positive difference
                w_plus = w_plus + avg_rank;
            }
        }

        current_rank = current_rank + F::from(end_idx - start_idx).unwrap();
    }

    // For large n, use normal approximation
    let n_f = F::from(n).unwrap();
    let mean_w = n_f * (n_f + F::one()) / F::from(4).unwrap();
    let var_w =
        n_f * (n_f + F::one()) * (F::from(2).unwrap() * n_f + F::one()) / F::from(24).unwrap();
    let std_w = var_w.sqrt();

    // Continuity correction
    let z = if w_plus > mean_w {
        (w_plus - mean_w - F::from(0.5).unwrap()) / std_w
    } else {
        (w_plus - mean_w + F::from(0.5).unwrap()) / std_w
    };

    let p_value = match alternative {
        "two-sided" => F::from(2).unwrap() * (F::one() - standard_normal_cdf(z.abs())),
        "greater" => F::one() - standard_normal_cdf(z),
        "less" => standard_normal_cdf(z),
        _ => {
            return Err(MetricsError::InvalidParameter(format!(
                "Unknown alternative: {}. Use 'two-sided', 'greater', or 'less'",
                alternative
            )));
        }
    };

    Ok((w_plus, p_value))
}

/// Permutation test for comparing two samples
///
/// Non-parametric test that uses random permutations to generate
/// the null distribution of a test statistic.
///
/// # Arguments
/// * `sample1` - First sample
/// * `sample2` - Second sample
/// * `test_statistic` - Function that computes test statistic from two samples
/// * `n_permutations` - Number of permutations to use (default 10000)
///
/// # Returns
/// Tuple of (observed statistic, p-value)
pub fn permutation_test<F: FloatTrait + FromPrimitive>(
    sample1: &Array1<F>,
    sample2: &Array1<F>,
    test_statistic: fn(&Array1<F>, &Array1<F>) -> F,
    n_permutations: usize,
) -> MetricsResult<(F, F)> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let observed_stat = test_statistic(sample1, sample2);

    // Combine samples
    let mut combined = Vec::new();
    combined.extend(sample1.iter());
    combined.extend(sample2.iter());

    let n1 = sample1.len();
    let n2 = sample2.len();
    let n_total = n1 + n2;

    let mut extreme_count = 0;

    // Simple linear congruential generator for reproducible randomness
    let mut rng_state = 12345u64;

    for _ in 0..n_permutations {
        // Shuffle combined array using Fisher-Yates algorithm
        let mut permuted = combined.clone();
        for i in (1..n_total).rev() {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng_state % (i + 1) as u64) as usize;
            permuted.swap(i, j);
        }

        // Split into two samples
        let perm_sample1 = Array1::from(permuted[..n1].to_vec());
        let perm_sample2 = Array1::from(permuted[n1..].to_vec());

        let perm_stat = test_statistic(&perm_sample1, &perm_sample2);

        if perm_stat.abs() >= observed_stat.abs() {
            extreme_count += 1;
        }
    }

    let p_value = F::from(extreme_count).unwrap() / F::from(n_permutations).unwrap();

    Ok((observed_stat, p_value))
}

// Helper functions for statistical distributions

/// Binomial test for exact McNemar test
fn binomial_test(k: i32, n: i32, p: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }

    // Two-tailed test: P(X >= k) + P(X <= n-k) for p = 0.5
    let prob_k_or_more = (k..=n).map(|i| binomial_pmf(i, n, p)).sum::<f64>();

    let prob_n_minus_k_or_less = (0..=(n - k)).map(|i| binomial_pmf(i, n, p)).sum::<f64>();

    2.0 * prob_k_or_more.min(prob_n_minus_k_or_less)
}

/// Binomial probability mass function
fn binomial_pmf(k: i32, n: i32, p: f64) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }

    let binom_coeff = (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
    binom_coeff * p.powi(k) * (1.0 - p).powi(n - k)
}

/// Chi-square p-value approximation for 1 degree of freedom
fn chi_square_p_value<F: FloatTrait + FromPrimitive>(x: F) -> F {
    if x <= F::zero() {
        return F::one();
    }

    // Approximate using complementary error function
    // For 1 df: P(X > x) ≈ 2 * (1 - Φ(√x))
    let sqrt_x = x.sqrt();
    F::from(2).unwrap() * (F::one() - standard_normal_cdf(sqrt_x))
}

/// Chi-square p-value approximation for arbitrary degrees of freedom
fn chi_square_p_value_df<F: FloatTrait + FromPrimitive>(x: F, df: i32) -> F {
    if x <= F::zero() {
        return F::one();
    }

    // Simplified approximation for common cases
    match df {
        1 => chi_square_p_value(x),
        2 => (-x / F::from(2).unwrap()).exp(),
        _ => {
            // Wilson-Hilferty transformation for large df
            let df_f = F::from(df).unwrap();
            let h = F::from(2).unwrap() / (F::from(9).unwrap() * df_f);
            let z = (F::one() - h + (x / df_f).powf(F::one() / F::from(3).unwrap())) / h.sqrt();
            F::one() - standard_normal_cdf(z)
        }
    }
}

/// Standard normal CDF approximation
fn standard_normal_cdf<F: FloatTrait + FromPrimitive>(x: F) -> F {
    let half = F::from(0.5).unwrap();
    let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();

    // Use error function approximation: Φ(x) = 0.5 * (1 + erf(x/√2))
    half * (F::one() + erf_approximation(x / sqrt2))
}

/// Error function approximation using Abramowitz and Stegun
fn erf_approximation<F: FloatTrait + FromPrimitive>(x: F) -> F {
    let a1 = F::from(0.254829592).unwrap();
    let a2 = F::from(-0.284496736).unwrap();
    let a3 = F::from(1.421413741).unwrap();
    let a4 = F::from(-1.453152027).unwrap();
    let a5 = F::from(1.061405429).unwrap();
    let p = F::from(0.3275911).unwrap();

    let sign = if x >= F::zero() { F::one() } else { -F::one() };
    let x_abs = x.abs();

    let t = F::one() / (F::one() + p * x_abs);
    let y =
        F::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

    sign * y
}

/// Transfer Entropy
///
/// Measures the amount of directed (time-asymmetric) information transfer between two time series.
/// Transfer entropy quantifies the reduction in uncertainty about future values of Y given
/// knowledge of past values of X and Y, compared to knowledge of past values of Y alone.
///
/// Formula: TE_{X→Y} = H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t, X_t)
///
/// # Arguments
/// * `x` - Source time series
/// * `y` - Target time series  
/// * `lag` - Time lag to consider (default: 1)
/// * `bins` - Number of bins for discretization (default: 10)
///
/// # Returns
/// Transfer entropy value (in bits)
///
/// # Examples
///
/// ```rust
/// use sklears_metrics::statistical_tests::transfer_entropy;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let y = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
/// let te = transfer_entropy(&x, &y, 1, 5).unwrap();
/// ```
pub fn transfer_entropy<F: FloatTrait + FromPrimitive + PartialOrd>(
    x: &Array1<F>,
    y: &Array1<F>,
    lag: usize,
    bins: usize,
) -> MetricsResult<F> {
    if x.len() != y.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        });
    }

    if x.len() <= lag + 1 {
        return Err(MetricsError::InvalidInput(
            "Time series must be longer than lag + 1".to_string(),
        ));
    }

    if bins == 0 {
        return Err(MetricsError::InvalidParameter(
            "Number of bins must be greater than 0".to_string(),
        ));
    }

    // Discretize the time series
    let x_discrete = discretize_time_series(x, bins)?;
    let y_discrete = discretize_time_series(y, bins)?;

    // Extract past and future values
    let n = x.len() - lag - 1;
    let mut y_future = Vec::with_capacity(n);
    let mut y_past = Vec::with_capacity(n);
    let mut x_past = Vec::with_capacity(n);

    for i in 0..n {
        y_future.push(y_discrete[i + lag + 1]);
        y_past.push(y_discrete[i + lag]);
        x_past.push(x_discrete[i]);
    }

    // Calculate conditional entropies
    // H(Y_{t+1}|Y_t)
    let h_y_future_given_y_past = conditional_entropy_discrete::<F>(&y_future, &y_past)?;

    // H(Y_{t+1}|Y_t, X_t)
    let xy_past: Vec<(usize, usize)> = y_past
        .iter()
        .zip(x_past.iter())
        .map(|(&y, &x)| (y, x))
        .collect();
    let h_y_future_given_xy_past = conditional_entropy_discrete_joint::<F>(&y_future, &xy_past)?;

    Ok(h_y_future_given_y_past - h_y_future_given_xy_past)
}

/// Bidirectional Transfer Entropy
///
/// Computes transfer entropy in both directions: X→Y and Y→X.
/// Useful for understanding bidirectional causal relationships.
///
/// # Arguments
/// * `x` - First time series
/// * `y` - Second time series
/// * `lag` - Time lag to consider
/// * `bins` - Number of bins for discretization
///
/// # Returns
/// Tuple of (TE_{X→Y}, TE_{Y→X})
pub fn bidirectional_transfer_entropy<F: FloatTrait + FromPrimitive + PartialOrd>(
    x: &Array1<F>,
    y: &Array1<F>,
    lag: usize,
    bins: usize,
) -> MetricsResult<(F, F)> {
    let te_x_to_y = transfer_entropy(x, y, lag, bins)?;
    let te_y_to_x = transfer_entropy(y, x, lag, bins)?;
    Ok((te_x_to_y, te_y_to_x))
}

/// Net Transfer Entropy
///
/// Computes the net information flow between two time series.
/// Positive values indicate X→Y dominance, negative values indicate Y→X dominance.
///
/// Formula: Net TE = TE_{X→Y} - TE_{Y→X}
pub fn net_transfer_entropy<F: FloatTrait + FromPrimitive + PartialOrd>(
    x: &Array1<F>,
    y: &Array1<F>,
    lag: usize,
    bins: usize,
) -> MetricsResult<F> {
    let (te_x_to_y, te_y_to_x) = bidirectional_transfer_entropy(x, y, lag, bins)?;
    Ok(te_x_to_y - te_y_to_x)
}

/// Multi-lag Transfer Entropy
///
/// Computes transfer entropy across multiple time lags to capture
/// delayed causal effects. Returns the maximum transfer entropy across all lags.
///
/// # Arguments
/// * `x` - Source time series
/// * `y` - Target time series
/// * `max_lag` - Maximum lag to consider
/// * `bins` - Number of bins for discretization
///
/// # Returns
/// Tuple of (max_transfer_entropy, optimal_lag)
pub fn multi_lag_transfer_entropy<F: FloatTrait + FromPrimitive + PartialOrd>(
    x: &Array1<F>,
    y: &Array1<F>,
    max_lag: usize,
    bins: usize,
) -> MetricsResult<(F, usize)> {
    if max_lag == 0 {
        return Err(MetricsError::InvalidParameter(
            "Maximum lag must be greater than 0".to_string(),
        ));
    }

    let mut max_te = F::from(-f64::INFINITY).unwrap();
    let mut optimal_lag = 1;

    for lag in 1..=max_lag {
        if x.len() > lag + 1 {
            match transfer_entropy(x, y, lag, bins) {
                Ok(te) => {
                    if te > max_te {
                        max_te = te;
                        optimal_lag = lag;
                    }
                }
                Err(_) => continue, // Skip lags that cause errors
            }
        }
    }

    if max_te == F::from(-f64::INFINITY).unwrap() {
        return Err(MetricsError::InvalidInput(
            "Could not compute transfer entropy for any lag".to_string(),
        ));
    }

    Ok((max_te, optimal_lag))
}

/// Partial Transfer Entropy
///
/// Computes transfer entropy while conditioning on a third variable Z.
/// This helps to identify direct vs. indirect causal relationships.
///
/// Formula: PTE_{X→Y|Z} = H(Y_{t+1}|Y_t, Z_t) - H(Y_{t+1}|Y_t, X_t, Z_t)
pub fn partial_transfer_entropy<F: FloatTrait + FromPrimitive + PartialOrd>(
    x: &Array1<F>,
    y: &Array1<F>,
    z: &Array1<F>,
    lag: usize,
    bins: usize,
) -> MetricsResult<F> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len(), z.len()],
        });
    }

    if x.len() <= lag + 1 {
        return Err(MetricsError::InvalidInput(
            "Time series must be longer than lag + 1".to_string(),
        ));
    }

    // Discretize the time series
    let x_discrete = discretize_time_series(x, bins)?;
    let y_discrete = discretize_time_series(y, bins)?;
    let z_discrete = discretize_time_series(z, bins)?;

    // Extract past and future values
    let n = x.len() - lag - 1;
    let mut y_future = Vec::with_capacity(n);
    let mut y_past = Vec::with_capacity(n);
    let mut x_past = Vec::with_capacity(n);
    let mut z_past = Vec::with_capacity(n);

    for i in 0..n {
        y_future.push(y_discrete[i + lag + 1]);
        y_past.push(y_discrete[i + lag]);
        x_past.push(x_discrete[i]);
        z_past.push(z_discrete[i]);
    }

    // H(Y_{t+1}|Y_t, Z_t)
    let yz_past: Vec<(usize, usize)> = y_past
        .iter()
        .zip(z_past.iter())
        .map(|(&y, &z)| (y, z))
        .collect();
    let h_y_future_given_yz_past = conditional_entropy_discrete_joint::<F>(&y_future, &yz_past)?;

    // H(Y_{t+1}|Y_t, X_t, Z_t)
    let xyz_past: Vec<(usize, usize, usize)> = y_past
        .iter()
        .zip(x_past.iter())
        .zip(z_past.iter())
        .map(|((&y, &x), &z)| (y, x, z))
        .collect();
    let h_y_future_given_xyz_past = conditional_entropy_discrete_triple::<F>(&y_future, &xyz_past)?;

    Ok(h_y_future_given_yz_past - h_y_future_given_xyz_past)
}

/// Helper function to discretize time series using equal-width binning
fn discretize_time_series<F: FloatTrait + FromPrimitive + PartialOrd>(
    series: &Array1<F>,
    bins: usize,
) -> MetricsResult<Vec<usize>> {
    if series.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let min_val = series
        .iter()
        .fold(series[0], |min, &x| if x < min { x } else { min });
    let max_val = series
        .iter()
        .fold(series[0], |max, &x| if x > max { x } else { max });

    if min_val == max_val {
        // All values are the same
        return Ok(vec![0; series.len()]);
    }

    let range = max_val - min_val;
    let bin_width = range / F::from(bins).unwrap();

    let discretized: Vec<usize> = series
        .iter()
        .map(|&val| {
            let bin_idx = ((val - min_val) / bin_width).floor();
            let idx = bin_idx.to_usize().unwrap_or(0);
            if idx >= bins {
                bins - 1
            } else {
                idx
            }
        })
        .collect();

    Ok(discretized)
}

/// Helper function to compute conditional entropy for discrete variables
fn conditional_entropy_discrete<F: FloatTrait + FromPrimitive>(
    y: &[usize],
    x: &[usize],
) -> MetricsResult<F> {
    let n = y.len();
    if n != x.len() || n == 0 {
        return Err(MetricsError::InvalidInput(
            "Invalid input sizes".to_string(),
        ));
    }

    let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut x_counts: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        *joint_counts.entry((x[i], y[i])).or_insert(0) += 1;
        *x_counts.entry(x[i]).or_insert(0) += 1;
    }

    let mut entropy = F::zero();
    let n_f = F::from(n).unwrap();

    for ((x_val, _y_val), joint_count) in joint_counts.iter() {
        let x_count = x_counts[x_val];
        let p_xy = F::from(*joint_count).unwrap() / n_f;
        let _p_x = F::from(x_count).unwrap() / n_f;
        let p_y_given_x = F::from(*joint_count).unwrap() / F::from(x_count).unwrap();

        if p_y_given_x > F::zero() {
            entropy = entropy - p_xy * p_y_given_x.ln();
        }
    }

    Ok(entropy)
}

/// Helper function for joint conditional entropy H(Y|X1,X2)
fn conditional_entropy_discrete_joint<F: FloatTrait + FromPrimitive>(
    y: &[usize],
    x_joint: &[(usize, usize)],
) -> MetricsResult<F> {
    let n = y.len();
    if n != x_joint.len() || n == 0 {
        return Err(MetricsError::InvalidInput(
            "Invalid input sizes".to_string(),
        ));
    }

    let mut joint_counts: HashMap<((usize, usize), usize), usize> = HashMap::new();
    let mut x_counts: HashMap<(usize, usize), usize> = HashMap::new();

    for i in 0..n {
        *joint_counts.entry((x_joint[i], y[i])).or_insert(0) += 1;
        *x_counts.entry(x_joint[i]).or_insert(0) += 1;
    }

    let mut entropy = F::zero();
    let n_f = F::from(n).unwrap();

    for ((x_val, _y_val), joint_count) in joint_counts.iter() {
        let x_count = x_counts[x_val];
        let p_xy = F::from(*joint_count).unwrap() / n_f;
        let p_y_given_x = F::from(*joint_count).unwrap() / F::from(x_count).unwrap();

        if p_y_given_x > F::zero() {
            entropy = entropy - p_xy * p_y_given_x.ln();
        }
    }

    Ok(entropy)
}

/// Helper function for triple conditional entropy H(Y|X1,X2,X3)
fn conditional_entropy_discrete_triple<F: FloatTrait + FromPrimitive>(
    y: &[usize],
    x_triple: &[(usize, usize, usize)],
) -> MetricsResult<F> {
    let n = y.len();
    if n != x_triple.len() || n == 0 {
        return Err(MetricsError::InvalidInput(
            "Invalid input sizes".to_string(),
        ));
    }

    let mut joint_counts: HashMap<((usize, usize, usize), usize), usize> = HashMap::new();
    let mut x_counts: HashMap<(usize, usize, usize), usize> = HashMap::new();

    for i in 0..n {
        *joint_counts.entry((x_triple[i], y[i])).or_insert(0) += 1;
        *x_counts.entry(x_triple[i]).or_insert(0) += 1;
    }

    let mut entropy = F::zero();
    let n_f = F::from(n).unwrap();

    for ((x_val, _y_val), joint_count) in joint_counts.iter() {
        let x_count = x_counts[x_val];
        let p_xy = F::from(*joint_count).unwrap() / n_f;
        let p_y_given_x = F::from(*joint_count).unwrap() / F::from(x_count).unwrap();

        if p_y_given_x > F::zero() {
            entropy = entropy - p_xy * p_y_given_x.ln();
        }
    }

    Ok(entropy)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mcnemar_test() {
        let y_true = Array1::from(vec![1, 0, 1, 1, 0, 1, 0, 0, 1, 1]);
        let y_pred1 = Array1::from(vec![1, 0, 1, 0, 0, 1, 1, 0, 1, 0]);
        let y_pred2 = Array1::from(vec![1, 1, 1, 1, 0, 0, 0, 0, 1, 1]);

        let (statistic, p_value) = mcnemar_test::<f64>(&y_true, &y_pred1, &y_pred2, false).unwrap();

        assert!(statistic >= 0.0);
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[test]
    fn test_friedman_test() {
        // Test with 3 datasets and 4 algorithms
        let scores = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.85, 0.82, 0.88, 0.84, 0.92, 0.89, 0.91, 0.87, 0.78, 0.75, 0.80, 0.76,
            ],
        )
        .unwrap();

        let (statistic, p_value, cd) = friedman_test(&scores).unwrap();

        assert!(statistic >= 0.0);
        assert!((0.0..=1.0).contains(&p_value));
        assert!(cd > 0.0);
    }

    #[test]
    fn test_wilcoxon_signed_rank_test() {
        let x = Array1::from(vec![1.2, 2.4, 1.8, 2.1, 1.7, 2.0]);
        let y = Array1::from(vec![1.0, 2.0, 1.5, 1.8, 1.4, 1.8]);

        let (statistic, p_value) = wilcoxon_signed_rank_test(&x, &y, "two-sided").unwrap();

        assert!(statistic >= 0.0);
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[test]
    fn test_permutation_test() {
        let sample1 = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let sample2 = Array1::from(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test using difference of means as test statistic
        fn mean_difference(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
            let mean_x = x.sum() / x.len() as f64;
            let mean_y = y.sum() / y.len() as f64;
            mean_x - mean_y
        }

        let (statistic, p_value) =
            permutation_test(&sample1, &sample2, mean_difference, 1000).unwrap();

        assert!((0.0..=1.0).contains(&p_value));
        assert_abs_diff_eq!(statistic, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mcnemar_identical_classifiers() {
        let y_true = Array1::from(vec![1, 0, 1, 1, 0]);
        let y_pred = Array1::from(vec![1, 0, 1, 0, 0]);

        let (statistic, p_value) = mcnemar_test::<f64>(&y_true, &y_pred, &y_pred, false).unwrap();

        assert_abs_diff_eq!(statistic, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p_value, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_error_cases() {
        let y_true = Array1::from(vec![1, 0, 1]);
        let y_pred1 = Array1::from(vec![1, 0]);
        let y_pred2 = Array1::from(vec![1, 0, 1]);

        // Test shape mismatch
        assert!(mcnemar_test::<f64>(&y_true, &y_pred1, &y_pred2, false).is_err());

        // Test empty input
        let empty = Array1::from(vec![]);
        assert!(mcnemar_test::<f64>(&empty, &empty, &empty, false).is_err());

        // Test insufficient data for Wilcoxon
        let small_x = Array1::from(vec![1.0, 2.0]);
        let small_y = Array1::from(vec![1.5, 2.5]);
        assert!(wilcoxon_signed_rank_test(&small_x, &small_y, "two-sided").is_err());
    }
}
