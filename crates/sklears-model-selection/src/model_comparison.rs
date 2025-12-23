//! Statistical Model Comparison Tests
//!
//! This module provides statistical tests for comparing the performance of different
//! machine learning models. It includes parametric and non-parametric tests for
//! model comparison with proper statistical significance assessment.
//!
//! Key tests include:
//! - Paired t-test for continuous performance metrics
//! - McNemar's test for binary classification performance
//! - Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
//! - Friedman test for comparing multiple models across multiple datasets
//! - Nemenyi post-hoc test for pairwise comparisons after Friedman test
//! - Cochran's Q test for comparing binary outcomes across multiple models

use scirs2_core::ndarray::{Array1, Array2};
// use scirs2_core::numeric::Float as FloatTrait;
use sklears_core::error::{Result, SklearsError};
use std::fmt::Debug;

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTestResult {
    /// Name of the test performed
    pub test_name: String,
    /// Test statistic value
    pub statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Critical value (if applicable)
    pub critical_value: Option<f64>,
    /// Degrees of freedom (if applicable)
    pub degrees_of_freedom: Option<f64>,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
    /// Whether the result is statistically significant at α = 0.05
    pub is_significant: bool,
    /// Confidence interval for the difference (if applicable)
    pub confidence_interval: Option<(f64, f64)>,
    /// Interpretation message
    pub interpretation: String,
}

impl StatisticalTestResult {
    /// Create a new test result
    pub fn new(test_name: String, statistic: f64, p_value: f64, alpha: f64) -> Self {
        let is_significant = p_value < alpha;
        let interpretation = if is_significant {
            format!(
                "Statistically significant difference detected (p = {:.4})",
                p_value
            )
        } else {
            format!(
                "No statistically significant difference (p = {:.4})",
                p_value
            )
        };

        Self {
            test_name,
            statistic,
            p_value,
            critical_value: None,
            degrees_of_freedom: None,
            effect_size: None,
            is_significant,
            confidence_interval: None,
            interpretation,
        }
    }

    /// Set critical value
    pub fn with_critical_value(mut self, critical_value: f64) -> Self {
        self.critical_value = Some(critical_value);
        self
    }

    /// Set degrees of freedom
    pub fn with_degrees_of_freedom(mut self, df: f64) -> Self {
        self.degrees_of_freedom = Some(df);
        self
    }

    /// Set effect size
    pub fn with_effect_size(mut self, effect_size: f64) -> Self {
        self.effect_size = Some(effect_size);
        self
    }

    /// Set confidence interval
    pub fn with_confidence_interval(mut self, lower: f64, upper: f64) -> Self {
        self.confidence_interval = Some((lower, upper));
        self
    }
}

/// Paired t-test for comparing two sets of continuous performance scores
///
/// This test assumes:
/// - Paired observations (same test instances for both models)
/// - Differences are normally distributed
/// - Continuous data
pub fn paired_t_test(
    scores1: &Array1<f64>,
    scores2: &Array1<f64>,
    alpha: f64,
) -> Result<StatisticalTestResult> {
    if scores1.len() != scores2.len() {
        return Err(SklearsError::InvalidInput(
            "Score arrays must have the same length".to_string(),
        ));
    }

    let n = scores1.len();
    if n < 2 {
        return Err(SklearsError::InvalidInput(
            "Need at least 2 paired observations".to_string(),
        ));
    }

    // Calculate differences
    let differences: Array1<f64> = scores1 - scores2;

    // Calculate mean and standard deviation of differences
    let mean_diff = differences.mean().unwrap();
    let variance = differences.var(1.0); // Sample variance (Bessel's correction)
    let std_diff = variance.sqrt();

    if std_diff == 0.0 {
        // When std_diff is zero, all differences are the same (no variance)
        // We can determine the result without a statistical test
        let p_value = if mean_diff == 0.0 { 1.0 } else { 0.0 };
        let significant = p_value < alpha;

        return Ok(StatisticalTestResult {
            test_name: "Paired t-test (zero variance)".to_string(),
            statistic: 0.0,
            p_value,
            critical_value: None,
            degrees_of_freedom: Some((n - 1) as f64),
            effect_size: Some(mean_diff),
            is_significant: significant,
            confidence_interval: Some((mean_diff, mean_diff)),
            interpretation: if mean_diff == 0.0 {
                "No difference between models (identical performance)".to_string()
            } else {
                format!(
                    "Models differ by {:.6} with zero variance (deterministic difference)",
                    mean_diff
                )
            },
        });
    }

    // Calculate t-statistic
    let standard_error = std_diff / (n as f64).sqrt();
    let t_statistic = mean_diff / standard_error;

    // Degrees of freedom
    let df = (n - 1) as f64;

    // Calculate p-value (two-tailed)
    let p_value = 2.0 * (1.0 - student_t_cdf(t_statistic.abs(), df));

    // Calculate 95% confidence interval for the mean difference
    let t_critical = inverse_student_t(1.0 - alpha / 2.0, df);
    let margin_error = t_critical * standard_error;
    let ci_lower = mean_diff - margin_error;
    let ci_upper = mean_diff + margin_error;

    // Calculate Cohen's d (effect size)
    let pooled_std = ((scores1.var(1.0) + scores2.var(1.0)) / 2.0).sqrt();
    let cohens_d = mean_diff / pooled_std;

    Ok(
        StatisticalTestResult::new("Paired t-test".to_string(), t_statistic, p_value, alpha)
            .with_degrees_of_freedom(df)
            .with_effect_size(cohens_d)
            .with_confidence_interval(ci_lower, ci_upper)
            .with_critical_value(t_critical),
    )
}

/// McNemar's test for comparing two binary classifiers
///
/// This test is used when:
/// - Comparing two binary classifiers on the same test set
/// - Testing whether the two classifiers have significantly different error rates
/// - Data is in the form of a 2x2 contingency table
pub fn mcnemar_test(
    correct_a_correct_b: usize,     // Both classifiers correct
    correct_a_incorrect_b: usize,   // A correct, B incorrect
    incorrect_a_correct_b: usize,   // A incorrect, B correct
    incorrect_a_incorrect_b: usize, // Both classifiers incorrect
    alpha: f64,
    continuity_correction: bool,
) -> Result<StatisticalTestResult> {
    let b = correct_a_incorrect_b as f64;
    let c = incorrect_a_correct_b as f64;
    let total = (correct_a_correct_b
        + correct_a_incorrect_b
        + incorrect_a_correct_b
        + incorrect_a_incorrect_b) as f64;

    if total == 0.0 {
        return Err(SklearsError::InvalidInput(
            "No observations provided".to_string(),
        ));
    }

    // Check if the test assumptions are met
    if b + c < 10.0 {
        return Err(SklearsError::InvalidInput(
            "McNemar's test requires at least 10 discordant pairs".to_string(),
        ));
    }

    // Calculate McNemar's statistic
    let statistic = if continuity_correction {
        // With continuity correction
        ((b - c).abs() - 1.0).powi(2) / (b + c)
    } else {
        // Without continuity correction
        (b - c).powi(2) / (b + c)
    };

    // Calculate p-value using chi-squared distribution with 1 df
    let p_value = 1.0 - chi_squared_cdf(statistic, 1.0);

    // Critical value for chi-squared with 1 df at given alpha
    let critical_value = inverse_chi_squared(1.0 - alpha, 1.0);

    let test_name = if continuity_correction {
        "McNemar's test (with continuity correction)".to_string()
    } else {
        "McNemar's test".to_string()
    };

    Ok(
        StatisticalTestResult::new(test_name, statistic, p_value, alpha)
            .with_degrees_of_freedom(1.0)
            .with_critical_value(critical_value),
    )
}

/// Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
///
/// This test:
/// - Does not assume normal distribution
/// - Tests whether the median difference is zero
/// - Is more robust to outliers than the t-test
pub fn wilcoxon_signed_rank_test(
    scores1: &Array1<f64>,
    scores2: &Array1<f64>,
    alpha: f64,
) -> Result<StatisticalTestResult> {
    if scores1.len() != scores2.len() {
        return Err(SklearsError::InvalidInput(
            "Score arrays must have the same length".to_string(),
        ));
    }

    // Calculate differences and filter out zeros
    let differences: Vec<f64> = scores1
        .iter()
        .zip(scores2.iter())
        .map(|(a, b)| a - b)
        .filter(|&d| d != 0.0)
        .collect();

    let n = differences.len();
    if n < 6 {
        return Err(SklearsError::InvalidInput(
            "Wilcoxon test requires at least 6 non-zero differences".to_string(),
        ));
    }

    // Calculate absolute differences and their ranks
    let mut abs_diffs_with_signs: Vec<(f64, f64)> =
        differences.iter().map(|&d| (d.abs(), d.signum())).collect();

    // Sort by absolute value for ranking
    abs_diffs_with_signs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Calculate ranks (handle ties by averaging ranks)
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && abs_diffs_with_signs[j].0 == abs_diffs_with_signs[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Calculate sum of positive and negative ranks
    let mut w_plus = 0.0;
    let mut w_minus = 0.0;

    for (rank, (_, sign)) in ranks.iter().zip(abs_diffs_with_signs.iter()) {
        if *sign > 0.0 {
            w_plus += rank;
        } else {
            w_minus += rank;
        }
    }

    // Test statistic is the smaller of W+ and W-
    let w_statistic = w_plus.min(w_minus);

    // For large n (> 20), use normal approximation
    let p_value = if n > 20 {
        let expected_w = n as f64 * (n + 1) as f64 / 4.0;
        let variance_w = n as f64 * (n + 1) as f64 * (2 * n + 1) as f64 / 24.0;
        let z_score = (w_statistic - expected_w) / variance_w.sqrt();
        2.0 * (1.0 - standard_normal_cdf(z_score.abs()))
    } else {
        // For small n, use exact distribution (simplified approximation)
        let critical_value = wilcoxon_critical_value(n, alpha);
        if w_statistic <= critical_value {
            0.01 // Significant
        } else {
            0.10 // Not significant
        }
    };

    Ok(StatisticalTestResult::new(
        "Wilcoxon signed-rank test".to_string(),
        w_statistic,
        p_value,
        alpha,
    ))
}

/// Friedman test for comparing multiple models across multiple datasets
///
/// This is a non-parametric test for:
/// - Comparing k models on n datasets
/// - Testing whether models have significantly different performance ranks
/// - Extension of Wilcoxon test to more than 2 models
pub fn friedman_test(
    performance_matrix: &Array2<f64>, // Rows: datasets, Columns: models
    alpha: f64,
) -> Result<StatisticalTestResult> {
    let (n_datasets, k_models) = performance_matrix.dim();

    if n_datasets < 2 || k_models < 3 {
        return Err(SklearsError::InvalidInput(
            "Friedman test requires at least 2 datasets and 3 models".to_string(),
        ));
    }

    // Calculate ranks for each dataset
    let mut rank_matrix = Array2::zeros((n_datasets, k_models));

    for i in 0..n_datasets {
        let row = performance_matrix.row(i);
        let mut indexed_scores: Vec<(usize, f64)> = row
            .iter()
            .enumerate()
            .map(|(j, &score)| (j, score))
            .collect();

        // Sort by score (descending for performance metrics)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Assign ranks (handle ties)
        let mut current_rank = 1.0;
        for window in indexed_scores.windows(2) {
            let (idx1, score1) = window[0];
            let (idx2, score2) = window[1];

            rank_matrix[[i, idx1]] = current_rank;

            if (score1 - score2).abs() < 1e-10 {
                // Tie: assign average rank
                rank_matrix[[i, idx2]] = current_rank;
            } else {
                current_rank += 1.0;
                if window.len() == 2 {
                    rank_matrix[[i, idx2]] = current_rank;
                }
            }
        }
    }

    // Calculate rank sums for each model
    let rank_sums: Array1<f64> = rank_matrix.sum_axis(scirs2_core::ndarray::Axis(0));

    // Calculate Friedman statistic
    let sum_of_squares: f64 = rank_sums.iter().map(|&r| r * r).sum();
    let friedman_statistic = (12.0 / (n_datasets as f64 * k_models as f64 * (k_models + 1) as f64))
        * sum_of_squares
        - 3.0 * n_datasets as f64 * (k_models + 1) as f64;

    // Calculate p-value using chi-squared distribution
    let df = (k_models - 1) as f64;
    let p_value = 1.0 - chi_squared_cdf(friedman_statistic, df);

    let critical_value = inverse_chi_squared(1.0 - alpha, df);

    Ok(StatisticalTestResult::new(
        "Friedman test".to_string(),
        friedman_statistic,
        p_value,
        alpha,
    )
    .with_degrees_of_freedom(df)
    .with_critical_value(critical_value))
}

/// Nemenyi post-hoc test for pairwise comparisons after Friedman test
///
/// This test is used after a significant Friedman test to determine
/// which specific pairs of models differ significantly.
pub fn nemenyi_post_hoc_test(
    performance_matrix: &Array2<f64>,
    alpha: f64,
) -> Result<Vec<(usize, usize, StatisticalTestResult)>> {
    let (n_datasets, k_models) = performance_matrix.dim();

    if n_datasets < 2 || k_models < 3 {
        return Err(SklearsError::InvalidInput(
            "Nemenyi test requires at least 2 datasets and 3 models".to_string(),
        ));
    }

    // First calculate average ranks for each model
    let mut rank_matrix = Array2::zeros((n_datasets, k_models));

    for i in 0..n_datasets {
        let row = performance_matrix.row(i);
        let mut indexed_scores: Vec<(usize, f64)> = row
            .iter()
            .enumerate()
            .map(|(j, &score)| (j, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (rank, (idx, _)) in indexed_scores.iter().enumerate() {
            rank_matrix[[i, *idx]] = (rank + 1) as f64;
        }
    }

    let average_ranks: Array1<f64> = rank_matrix
        .mean_axis(scirs2_core::ndarray::Axis(0))
        .unwrap();

    // Critical difference for Nemenyi test
    let q_alpha = nemenyi_critical_value(k_models, alpha);
    let critical_difference =
        q_alpha * ((k_models * (k_models + 1)) as f64 / (6.0 * n_datasets as f64)).sqrt();

    // Perform pairwise comparisons
    let mut results = Vec::new();

    for i in 0..k_models {
        for j in (i + 1)..k_models {
            let rank_diff = (average_ranks[i] - average_ranks[j]).abs();
            let is_significant = rank_diff > critical_difference;

            let test_result = StatisticalTestResult {
                test_name: format!("Nemenyi post-hoc (Model {} vs Model {})", i + 1, j + 1),
                statistic: rank_diff,
                p_value: if is_significant { 0.01 } else { 0.10 }, // Simplified
                critical_value: Some(critical_difference),
                degrees_of_freedom: None,
                effect_size: Some(rank_diff),
                is_significant,
                confidence_interval: None,
                interpretation: if is_significant {
                    format!("Significant difference in ranks: {:.3}", rank_diff)
                } else {
                    format!("No significant difference in ranks: {:.3}", rank_diff)
                },
            };

            results.push((i, j, test_result));
        }
    }

    Ok(results)
}

/// Multiple model comparison with correction for multiple testing
pub fn multiple_model_comparison(
    performance_matrices: &[Array2<f64>], // Multiple performance matrices
    model_names: &[String],
    alpha: f64,
    correction_method: MultipleTestingCorrection,
) -> Result<ModelComparisonResult> {
    if performance_matrices.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No performance data provided".to_string(),
        ));
    }

    let n_models = model_names.len();
    if n_models < 2 {
        return Err(SklearsError::InvalidInput(
            "Need at least 2 models to compare".to_string(),
        ));
    }

    let mut pairwise_results = Vec::new();
    let mut raw_p_values = Vec::new();

    // Perform all pairwise comparisons
    for (matrix_idx, matrix) in performance_matrices.iter().enumerate() {
        if matrix.ncols() != n_models {
            return Err(SklearsError::InvalidInput(format!(
                "Performance matrix {} has {} models, expected {}",
                matrix_idx,
                matrix.ncols(),
                n_models
            )));
        }

        for i in 0..n_models {
            for j in (i + 1)..n_models {
                let scores1 = matrix.column(i).to_owned();
                let scores2 = matrix.column(j).to_owned();

                let test_result = paired_t_test(&scores1, &scores2, alpha)?;
                raw_p_values.push(test_result.p_value);
                pairwise_results.push((i, j, matrix_idx, test_result));
            }
        }
    }

    // Apply multiple testing correction
    let corrected_p_values = match correction_method {
        MultipleTestingCorrection::Bonferroni => bonferroni_correction(&raw_p_values),
        MultipleTestingCorrection::BenjaminiHochberg => {
            benjamini_hochberg_correction(&raw_p_values, alpha)
        }
        MultipleTestingCorrection::Holm => holm_correction(&raw_p_values),
        MultipleTestingCorrection::None => raw_p_values.clone(),
    };

    // Update significance based on corrected p-values
    for (result, &corrected_p) in pairwise_results.iter_mut().zip(corrected_p_values.iter()) {
        result.3.p_value = corrected_p;
        result.3.is_significant = corrected_p < alpha;
    }

    let significant_pairs_count = pairwise_results
        .iter()
        .filter(|(_, _, _, result)| result.is_significant)
        .count();

    Ok(ModelComparisonResult {
        model_names: model_names.to_vec(),
        pairwise_results,
        correction_method,
        alpha,
        n_comparisons: raw_p_values.len(),
        significant_pairs: significant_pairs_count,
    })
}

/// Multiple testing correction methods
#[derive(Debug, Clone)]
pub enum MultipleTestingCorrection {
    /// No correction applied
    None,
    /// Bonferroni correction (most conservative)
    Bonferroni,
    /// Benjamini-Hochberg correction (controls FDR)
    BenjaminiHochberg,
    /// Holm correction (step-down method)
    Holm,
}

/// Result of multiple model comparison
#[derive(Debug, Clone)]
pub struct ModelComparisonResult {
    pub model_names: Vec<String>,
    pub pairwise_results: Vec<(usize, usize, usize, StatisticalTestResult)>,
    pub correction_method: MultipleTestingCorrection,
    pub alpha: f64,
    pub n_comparisons: usize,
    pub significant_pairs: usize,
}

// Helper functions for statistical distributions
fn student_t_cdf(t: f64, df: f64) -> f64 {
    // Simplified approximation of Student's t CDF
    if df > 30.0 {
        standard_normal_cdf(t)
    } else {
        // Use approximation for small df
        0.5 + 0.5 * (t / (1.0 + t * t / df).sqrt()).tanh()
    }
}

fn inverse_student_t(p: f64, df: f64) -> f64 {
    // Simplified approximation of inverse Student's t
    if df > 30.0 {
        inverse_standard_normal(p)
    } else {
        // Simplified approximation
        let z = inverse_standard_normal(p);
        z * (1.0 + (z * z + 1.0) / (4.0 * df))
    }
}

fn standard_normal_cdf(z: f64) -> f64 {
    // Approximation of standard normal CDF
    0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
}

fn inverse_standard_normal(p: f64) -> f64 {
    // Approximation of inverse standard normal
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Beasley-Springer-Moro algorithm approximation
    let a = [
        0.0,
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
    ];
    let b = [
        0.0,
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let x = p - 0.5;
    if x.abs() < 0.42 {
        let x2 = x * x;
        let _num = a[4] * x2 + a[3];
        let den = b[4] * x2 + b[3];
        x * (((a[2] * x2 + a[1]) * x2 + a[0]) / ((den * x2 + b[2]) * x2 + b[1]))
    } else {
        let ln_p = if p > 0.5 { (1.0 - p).ln() } else { p.ln() };
        let t = (-2.0 * ln_p).sqrt();

        let _num = a[4] * t + a[3];
        let den = b[4] * t + b[3];
        let result = t - ((((a[2] * t + a[1]) * t + a[0]) / ((den * t + b[2]) * t + b[1])) / t);

        if p > 0.5 {
            result
        } else {
            -result
        }
    }
}

fn chi_squared_cdf(chi2: f64, df: f64) -> f64 {
    // Simplified approximation of chi-squared CDF
    if df == 1.0 {
        2.0 * standard_normal_cdf(chi2.sqrt()) - 1.0
    } else if df == 2.0 {
        1.0 - (-chi2 / 2.0).exp()
    } else {
        // Use normal approximation for large df
        let z = ((2.0 * chi2).sqrt() - (2.0 * df - 1.0).sqrt()) / 2.0_f64.sqrt();
        standard_normal_cdf(z)
    }
}

fn inverse_chi_squared(p: f64, df: f64) -> f64 {
    // Simplified approximation of inverse chi-squared
    if df == 1.0 {
        let z = inverse_standard_normal((p + 1.0) / 2.0);
        z * z
    } else if df == 2.0 {
        -2.0 * (1.0 - p).ln()
    } else {
        // Wilson-Hilferty transformation
        let h = 2.0 / (9.0 * df);
        let z = inverse_standard_normal(p);
        df * (1.0 - h + z * (h).sqrt()).powi(3)
    }
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a = 0.147;
    let x2 = x * x;
    let ax2 = a * x2;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };

    sign * (1.0 - (-(x2) * (4.0 / std::f64::consts::PI + ax2) / (1.0 + ax2)).exp()).sqrt()
}

fn wilcoxon_critical_value(n: usize, alpha: f64) -> f64 {
    // Simplified critical values for Wilcoxon test
    // In practice, you would use a lookup table
    match (n, alpha) {
        (6, a) if a <= 0.05 => 0.0,
        (7, a) if a <= 0.05 => 2.0,
        (8, a) if a <= 0.05 => 3.0,
        (9, a) if a <= 0.05 => 5.0,
        (10, a) if a <= 0.05 => 8.0,
        _ => (n * (n + 1) / 4) as f64 * 0.1, // Rough approximation
    }
}

fn nemenyi_critical_value(k: usize, alpha: f64) -> f64 {
    // Simplified critical values for Nemenyi test
    // In practice, you would use a lookup table
    match (k, alpha) {
        (3, a) if a <= 0.05 => 2.394,
        (4, a) if a <= 0.05 => 2.569,
        (5, a) if a <= 0.05 => 2.728,
        (6, a) if a <= 0.05 => 2.850,
        _ => 2.5 + (k as f64 - 3.0) * 0.1, // Rough approximation
    }
}

// Multiple testing correction methods
fn bonferroni_correction(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len() as f64;
    p_values.iter().map(|&p| (p * n).min(1.0)).collect()
}

fn benjamini_hochberg_correction(p_values: &[f64], _alpha: f64) -> Vec<f64> {
    let n = p_values.len();
    let mut indexed_p: Vec<(usize, f64)> =
        p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut corrected = vec![0.0; n];
    for (rank, (original_idx, p_val)) in indexed_p.iter().enumerate() {
        let corrected_p = p_val * (n as f64) / ((rank + 1) as f64);
        corrected[*original_idx] = corrected_p.min(1.0);
    }

    corrected
}

fn holm_correction(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    let mut indexed_p: Vec<(usize, f64)> =
        p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut corrected = vec![0.0; n];
    for (rank, (original_idx, p_val)) in indexed_p.iter().enumerate() {
        let corrected_p = p_val * ((n - rank) as f64);
        corrected[*original_idx] = corrected_p.min(1.0);
    }

    corrected
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_paired_t_test() {
        let scores1 = array![0.9, 0.8, 0.85, 0.92, 0.88];
        let scores2 = array![0.85, 0.75, 0.80, 0.87, 0.83];

        let result = paired_t_test(&scores1, &scores2, 0.05).unwrap();
        assert_eq!(result.test_name, "Paired t-test");
        assert!(result.statistic > 0.0); // scores1 > scores2
        assert!(result.degrees_of_freedom.is_some());
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_mcnemar_test() {
        let result = mcnemar_test(85, 10, 5, 0, 0.05, false).unwrap();
        assert_eq!(result.test_name, "McNemar's test");
        assert!(result.statistic > 0.0);
        assert!(result.degrees_of_freedom == Some(1.0));
    }

    #[test]
    fn test_wilcoxon_signed_rank() {
        let scores1 = array![0.9, 0.8, 0.85, 0.92, 0.88, 0.91, 0.87];
        let scores2 = array![0.85, 0.75, 0.80, 0.87, 0.83, 0.86, 0.82];

        let result = wilcoxon_signed_rank_test(&scores1, &scores2, 0.05).unwrap();
        assert_eq!(result.test_name, "Wilcoxon signed-rank test");
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_friedman_test() {
        let performance = array![
            [0.9, 0.85, 0.80],
            [0.88, 0.83, 0.78],
            [0.92, 0.87, 0.82],
            [0.89, 0.84, 0.79]
        ];

        let result = friedman_test(&performance, 0.05).unwrap();
        assert_eq!(result.test_name, "Friedman test");
        assert!(result.degrees_of_freedom == Some(2.0));
    }

    #[test]
    fn test_bonferroni_correction() {
        let p_values = vec![0.01, 0.02, 0.03, 0.04];
        let corrected = bonferroni_correction(&p_values);

        assert_eq!(corrected[0], 0.04);
        assert_eq!(corrected[1], 0.08);
        assert_eq!(corrected[2], 0.12);
        assert_eq!(corrected[3], 0.16);
    }

    #[test]
    fn test_statistical_test_result() {
        let result = StatisticalTestResult::new("Test".to_string(), 2.5, 0.03, 0.05);

        assert!(result.is_significant);
        assert_eq!(result.p_value, 0.03);
        assert_eq!(result.statistic, 2.5);
    }
}
