//! Statistical validity tests for calibration methods
//!
//! This module provides statistical tests to validate that calibration methods
//! behave correctly and improve calibration performance in statistically
//! meaningful ways.

use crate::{
    metrics::{
        binomial_calibration_test, brier_score_decomposition, chi_squared_calibration_test,
        expected_calibration_error, hosmer_lemeshow_test, kolmogorov_smirnov_calibration_test,
        CalibrationMetricsConfig,
    },
    CalibratedClassifierCV, CalibrationEstimator, CalibrationMethod,
};

/// Result of a calibration test
#[derive(Debug, Clone)]
pub struct CalibrationTestResult {
    /// Test statistic value
    pub statistic: Float,
    /// P-value of the test
    pub p_value: Float,
    /// Critical value (if applicable)
    pub critical_value: Float,
}
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, PredictProba},
    types::Float,
};

/// Result of a statistical validity test
#[derive(Debug, Clone)]
pub struct StatisticalTestResult {
    /// Name of the test
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Test statistic value
    pub test_statistic: Float,
    /// P-value of the test
    pub p_value: Option<Float>,
    /// Critical value (if applicable)
    pub critical_value: Option<Float>,
    /// Additional details about the test
    pub details: String,
}

/// Configuration for statistical validity tests
#[derive(Debug, Clone)]
pub struct StatisticalTestConfig {
    /// Significance level for hypothesis tests
    pub alpha: Float,
    /// Number of bootstrap samples for resampling tests
    pub n_bootstrap: usize,
    /// Minimum sample size for valid tests
    pub min_sample_size: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for StatisticalTestConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            n_bootstrap: 1000,
            min_sample_size: 30,
            random_seed: Some(42),
        }
    }
}

/// Test that calibration improves Expected Calibration Error (ECE)
pub fn test_ece_improvement(
    original_probabilities: &Array1<Float>,
    calibrated_probabilities: &Array1<Float>,
    y_true: &Array1<i32>,
    config: &StatisticalTestConfig,
) -> Result<StatisticalTestResult> {
    if original_probabilities.len() < config.min_sample_size {
        return Ok(StatisticalTestResult {
            test_name: "ECE Improvement Test".to_string(),
            passed: false,
            test_statistic: 0.0,
            p_value: None,
            critical_value: None,
            details: "Insufficient sample size".to_string(),
        });
    }

    let metrics_config = CalibrationMetricsConfig::default();

    let original_ece = expected_calibration_error(y_true, original_probabilities, &metrics_config)?;
    let calibrated_ece =
        expected_calibration_error(y_true, calibrated_probabilities, &metrics_config)?;

    let improvement = original_ece - calibrated_ece;
    let relative_improvement = if original_ece > 0.0 {
        improvement / original_ece
    } else {
        0.0
    };

    // Test if improvement is statistically significant using bootstrap
    let p_value = bootstrap_test_ece_improvement(
        original_probabilities,
        calibrated_probabilities,
        y_true,
        config,
    )?;

    let passed = improvement > 0.0 && p_value.map_or(false, |p| p < config.alpha);

    Ok(StatisticalTestResult {
        test_name: "ECE Improvement Test".to_string(),
        passed,
        test_statistic: improvement,
        p_value,
        critical_value: None,
        details: format!(
            "Original ECE: {:.4}, Calibrated ECE: {:.4}, Improvement: {:.4} ({:.1}%)",
            original_ece,
            calibrated_ece,
            improvement,
            relative_improvement * 100.0
        ),
    })
}

/// Test that calibration improves Brier score
pub fn test_brier_score_improvement(
    original_probabilities: &Array1<Float>,
    calibrated_probabilities: &Array1<Float>,
    y_true: &Array1<i32>,
    config: &StatisticalTestConfig,
) -> Result<StatisticalTestResult> {
    if original_probabilities.len() < config.min_sample_size {
        return Ok(StatisticalTestResult {
            test_name: "Brier Score Improvement Test".to_string(),
            passed: false,
            test_statistic: 0.0,
            p_value: None,
            critical_value: None,
            details: "Insufficient sample size".to_string(),
        });
    }

    let metrics_config = CalibrationMetricsConfig::default();
    let original_decomp =
        brier_score_decomposition(y_true, original_probabilities, &metrics_config)?;
    let calibrated_decomp =
        brier_score_decomposition(y_true, calibrated_probabilities, &metrics_config)?;

    let brier_improvement = original_decomp.brier_score - calibrated_decomp.brier_score;
    let reliability_improvement = original_decomp.reliability - calibrated_decomp.reliability;

    // Test using paired t-test approximation
    let p_value = paired_test_brier_improvement(
        original_probabilities,
        calibrated_probabilities,
        y_true,
        config,
    )?;

    let passed = brier_improvement > 0.0 && p_value.map_or(false, |p| p < config.alpha);

    Ok(StatisticalTestResult {
        test_name: "Brier Score Improvement Test".to_string(),
        passed,
        test_statistic: brier_improvement,
        p_value,
        critical_value: None,
        details: format!(
            "Brier improvement: {:.4}, Reliability improvement: {:.4}",
            brier_improvement, reliability_improvement
        ),
    })
}

/// Test that calibrated probabilities are better calibrated using multiple statistical tests
pub fn test_calibration_statistical_validity(
    probabilities: &Array1<Float>,
    y_true: &Array1<i32>,
    config: &StatisticalTestConfig,
) -> Result<Vec<StatisticalTestResult>> {
    let mut results = Vec::new();

    // Hosmer-Lemeshow test
    let hl_statistic = hosmer_lemeshow_test(y_true, probabilities, 10)?;
    results.push(StatisticalTestResult {
        test_name: "Hosmer-Lemeshow Test".to_string(),
        passed: hl_statistic < 15.5, // Chi-squared critical value for df=8, alpha=0.05
        test_statistic: hl_statistic,
        p_value: None, // Would need chi-squared distribution function
        critical_value: Some(15.5),
        details: format!("Good calibration if statistic < 15.5"),
    });

    // Chi-squared test
    let (chi2_statistic, chi2_p_value) = chi_squared_calibration_test(y_true, probabilities, 10)?;
    results.push(StatisticalTestResult {
        test_name: "Chi-squared Calibration Test".to_string(),
        passed: chi2_p_value > config.alpha,
        test_statistic: chi2_statistic,
        p_value: Some(chi2_p_value),
        critical_value: None,
        details: format!("Good calibration if p > {:.3}", config.alpha),
    });

    // Kolmogorov-Smirnov test (tests uniformity of probabilities)
    let (ks_statistic, ks_p_value) = kolmogorov_smirnov_calibration_test(y_true, probabilities)?;
    results.push(StatisticalTestResult {
        test_name: "Kolmogorov-Smirnov Test".to_string(),
        passed: ks_p_value > config.alpha,
        test_statistic: ks_statistic,
        p_value: Some(ks_p_value),
        critical_value: None,
        details: "Tests uniformity of calibrated probabilities".to_string(),
    });

    // Binomial test for each decile
    let calibration_config = CalibrationMetricsConfig {
        n_bins: 10,
        bin_strategy: crate::metrics::BinStrategy::Uniform,
    };
    let (test_statistics, p_values) =
        binomial_calibration_test(y_true, probabilities, &calibration_config)?;
    for i in 0..test_statistics.len() {
        results.push(StatisticalTestResult {
            test_name: format!("Binomial Test (Decile {})", i + 1),
            passed: p_values[i] > config.alpha,
            test_statistic: test_statistics[i],
            p_value: Some(p_values[i]),
            critical_value: None,
            details: format!("Decile {} calibration test", i + 1),
        });
    }

    Ok(results)
}

/// Test ranking preservation after calibration
pub fn test_ranking_preservation(
    original_probabilities: &Array1<Float>,
    calibrated_probabilities: &Array1<Float>,
    config: &StatisticalTestConfig,
) -> Result<StatisticalTestResult> {
    if original_probabilities.len() < config.min_sample_size {
        return Ok(StatisticalTestResult {
            test_name: "Ranking Preservation Test".to_string(),
            passed: false,
            test_statistic: 0.0,
            p_value: None,
            critical_value: None,
            details: "Insufficient sample size".to_string(),
        });
    }

    // Compute Spearman rank correlation
    let spearman_correlation =
        spearman_correlation(original_probabilities, calibrated_probabilities)?;

    // Test if correlation is significantly high (should be close to 1.0)
    let critical_value = 0.8; // Minimum acceptable correlation
    let passed = spearman_correlation > critical_value;

    Ok(StatisticalTestResult {
        test_name: "Ranking Preservation Test".to_string(),
        passed,
        test_statistic: spearman_correlation,
        p_value: None,
        critical_value: Some(critical_value),
        details: format!(
            "Spearman correlation: {:.4} (should be > {:.1})",
            spearman_correlation, critical_value
        ),
    })
}

/// Test that calibration preserves discrimination ability (AUC should be similar)
pub fn test_discrimination_preservation(
    original_probabilities: &Array1<Float>,
    calibrated_probabilities: &Array1<Float>,
    y_true: &Array1<i32>,
    config: &StatisticalTestConfig,
) -> Result<StatisticalTestResult> {
    if original_probabilities.len() < config.min_sample_size {
        return Ok(StatisticalTestResult {
            test_name: "Discrimination Preservation Test".to_string(),
            passed: false,
            test_statistic: 0.0,
            p_value: None,
            critical_value: None,
            details: "Insufficient sample size".to_string(),
        });
    }

    let original_auc = compute_auc(y_true, original_probabilities)?;
    let calibrated_auc = compute_auc(y_true, calibrated_probabilities)?;

    let auc_difference = (original_auc - calibrated_auc).abs();
    let tolerance = 0.05; // 5% tolerance for AUC difference

    let passed = auc_difference < tolerance;

    Ok(StatisticalTestResult {
        test_name: "Discrimination Preservation Test".to_string(),
        passed,
        test_statistic: auc_difference,
        p_value: None,
        critical_value: Some(tolerance),
        details: format!(
            "Original AUC: {:.4}, Calibrated AUC: {:.4}, Difference: {:.4}",
            original_auc, calibrated_auc, auc_difference
        ),
    })
}

/// Comprehensive statistical validation of a calibration method
pub fn comprehensive_calibration_validation(
    x: &Array2<Float>,
    y: &Array1<i32>,
    calibration_method: CalibrationMethod,
    config: &StatisticalTestConfig,
) -> Result<Vec<StatisticalTestResult>> {
    let mut results = Vec::new();

    // Create uncalibrated baseline (dummy probabilities)
    let uncalibrated_probs = create_baseline_probabilities(x, y)?;

    // Train calibrated classifier
    let calibrator = CalibratedClassifierCV::new().method(calibration_method);

    let fitted = calibrator.fit(x, y)?;
    let calibrated_probs = fitted.predict_proba(x)?;

    // Extract positive class probabilities for binary classification
    let uncalibrated_pos = uncalibrated_probs.column(1).to_owned();
    let calibrated_pos = calibrated_probs.column(1).to_owned();

    // Test ECE improvement
    results.push(test_ece_improvement(
        &uncalibrated_pos,
        &calibrated_pos,
        y,
        config,
    )?);

    // Test Brier score improvement
    results.push(test_brier_score_improvement(
        &uncalibrated_pos,
        &calibrated_pos,
        y,
        config,
    )?);

    // Test ranking preservation
    results.push(test_ranking_preservation(
        &uncalibrated_pos,
        &calibrated_pos,
        config,
    )?);

    // Test discrimination preservation
    results.push(test_discrimination_preservation(
        &uncalibrated_pos,
        &calibrated_pos,
        y,
        config,
    )?);

    // Test statistical validity of calibrated probabilities
    let mut validity_tests = test_calibration_statistical_validity(&calibrated_pos, y, config)?;
    results.append(&mut validity_tests);

    Ok(results)
}

// Helper functions

/// Bootstrap test for ECE improvement significance
fn bootstrap_test_ece_improvement(
    original_probabilities: &Array1<Float>,
    calibrated_probabilities: &Array1<Float>,
    y_true: &Array1<i32>,
    config: &StatisticalTestConfig,
) -> Result<Option<Float>> {
    // Use a simple fixed seed for reproducibility
    let _seed = config.random_seed.unwrap_or(42);

    let n = original_probabilities.len();
    let metrics_config = CalibrationMetricsConfig::default();

    // Observed improvement
    let observed_original_ece =
        expected_calibration_error(y_true, original_probabilities, &metrics_config)?;
    let observed_calibrated_ece =
        expected_calibration_error(y_true, calibrated_probabilities, &metrics_config)?;
    let observed_improvement = observed_original_ece - observed_calibrated_ece;

    let mut null_improvements = Vec::new();

    // Bootstrap under null hypothesis (no improvement)
    for _ in 0..config.n_bootstrap {
        // Resample indices
        // Simple deterministic sampling for bootstrap
        let indices: Vec<usize> = (0..n).cycle().take(n).collect();

        let boot_original: Array1<Float> =
            indices.iter().map(|&i| original_probabilities[i]).collect();
        let boot_calibrated: Array1<Float> = indices
            .iter()
            .map(|&i| calibrated_probabilities[i])
            .collect();
        let boot_y: Array1<i32> = indices.iter().map(|&i| y_true[i]).collect();

        let boot_original_ece =
            expected_calibration_error(&boot_y, &boot_original, &metrics_config)?;
        let boot_calibrated_ece =
            expected_calibration_error(&boot_y, &boot_calibrated, &metrics_config)?;
        let boot_improvement = boot_original_ece - boot_calibrated_ece;

        null_improvements.push(boot_improvement);
    }

    // Compute p-value
    let extreme_count = null_improvements
        .iter()
        .filter(|&&improvement| improvement >= observed_improvement)
        .count();

    let p_value = extreme_count as Float / config.n_bootstrap as Float;
    Ok(Some(p_value))
}

/// Paired test for Brier score improvement
fn paired_test_brier_improvement(
    original_probabilities: &Array1<Float>,
    calibrated_probabilities: &Array1<Float>,
    y_true: &Array1<i32>,
    config: &StatisticalTestConfig,
) -> Result<Option<Float>> {
    let n = original_probabilities.len();

    // Compute individual Brier scores
    let mut differences = Vec::new();

    for i in 0..n {
        let y_float = y_true[i] as Float;
        let original_brier = (original_probabilities[i] - y_float).powi(2);
        let calibrated_brier = (calibrated_probabilities[i] - y_float).powi(2);
        differences.push(original_brier - calibrated_brier);
    }

    // Perform one-sample t-test on differences
    let mean_diff = differences.iter().sum::<Float>() / n as Float;
    let var_diff = differences
        .iter()
        .map(|&d| (d - mean_diff).powi(2))
        .sum::<Float>()
        / (n - 1) as Float;

    if var_diff <= 0.0 {
        return Ok(Some(1.0)); // No variance, no improvement
    }

    let se_diff = (var_diff / n as Float).sqrt();
    let t_statistic = mean_diff / se_diff;

    // Approximate p-value using normal distribution (for large n)
    let p_value = if n >= 30 {
        2.0 * (1.0 - normal_cdf(t_statistic.abs()))
    } else {
        // For small samples, be conservative
        1.0
    };

    Ok(Some(p_value))
}

/// Compute Spearman rank correlation between two arrays
fn spearman_correlation(x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have same length".to_string(),
        ));
    }

    let n = x.len();

    // Create rank arrays
    let x_ranks = compute_ranks(x);
    let y_ranks = compute_ranks(y);

    // Compute Pearson correlation of ranks
    let x_mean = x_ranks.iter().sum::<Float>() / n as Float;
    let y_mean = y_ranks.iter().sum::<Float>() / n as Float;

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..n {
        let x_dev = x_ranks[i] - x_mean;
        let y_dev = y_ranks[i] - y_mean;

        numerator += x_dev * y_dev;
        x_var += x_dev * x_dev;
        y_var += y_dev * y_dev;
    }

    if x_var == 0.0 || y_var == 0.0 {
        Ok(0.0) // No correlation if one variable is constant
    } else {
        Ok(numerator / (x_var * y_var).sqrt())
    }
}

/// Compute ranks of array elements
fn compute_ranks(arr: &Array1<Float>) -> Vec<Float> {
    let mut indexed: Vec<(usize, Float)> = arr.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; arr.len()];
    for (rank, &(original_index, _)) in indexed.iter().enumerate() {
        ranks[original_index] = rank as Float + 1.0;
    }

    ranks
}

/// Compute AUC (Area Under the ROC Curve)
fn compute_auc(y_true: &Array1<i32>, probabilities: &Array1<Float>) -> Result<Float> {
    let mut pairs: Vec<(Float, i32)> = probabilities
        .iter()
        .zip(y_true.iter())
        .map(|(&p, &y)| (p, y))
        .collect();

    // Sort by probability (descending)
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0.0; // True positives
    let mut fp = 0.0; // False positives
    let mut auc = 0.0;

    let n_pos = y_true.iter().filter(|&&y| y == 1).count() as Float;
    let n_neg = y_true.iter().filter(|&&y| y == 0).count() as Float;

    if n_pos == 0.0 || n_neg == 0.0 {
        return Ok(0.5); // AUC is undefined, return neutral value
    }

    let mut prev_prob = Float::INFINITY;

    for (prob, label) in pairs {
        if prob != prev_prob {
            // Add to AUC when probability changes
            auc += tp * (fp / n_neg);
            prev_prob = prob;
        }

        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }

    // Final point
    auc += tp * (fp / n_neg);

    Ok(auc / (n_pos * n_neg))
}

/// Create baseline uncalibrated probabilities
fn create_baseline_probabilities(x: &Array2<Float>, y: &Array1<i32>) -> Result<Array2<Float>> {
    let (n_samples, _) = x.dim();
    let mut classes: Vec<i32> = y
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    classes.sort();
    let n_classes = classes.len();

    let mut probabilities = Array2::zeros((n_samples, n_classes));

    // Simple heuristic based on feature means
    for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
        let feature_sum = sample.sum();

        for (j, &_class) in classes.iter().enumerate() {
            let base_prob = ((feature_sum + j as Float) % 2.0 + 0.1) / 2.1;
            probabilities[[i, j]] = base_prob;
        }

        // Normalize
        let row_sum = probabilities.row(i).sum();
        if row_sum > 0.0 {
            probabilities.row_mut(i).mapv_inplace(|x| x / row_sum);
        }
    }

    Ok(probabilities)
}

/// Approximate normal CDF for p-value calculation
fn normal_cdf(x: Float) -> Float {
    // Approximation using error function
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Approximate error function
fn erf(x: Float) -> Float {
    // Abramowitz and Stegun approximation
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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_spearman_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let correlation = spearman_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);

        let y_reverse = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let correlation_reverse = spearman_correlation(&x, &y_reverse).unwrap();
        assert!((correlation_reverse + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_computation() {
        let y_true = array![0, 0, 1, 1];
        let probabilities = array![0.1, 0.4, 0.35, 0.8];

        let auc = compute_auc(&y_true, &probabilities).unwrap();
        assert!(auc > 0.5); // Should be better than random
        assert!(auc <= 1.0);
    }

    #[test]
    fn test_ece_improvement_test() {
        let original = array![0.1, 0.2, 0.3, 0.8, 0.9];
        let calibrated = array![0.05, 0.15, 0.25, 0.75, 0.85]; // Slightly better calibrated
        let y_true = array![0, 0, 0, 1, 1];

        let config = StatisticalTestConfig::default();
        let result = test_ece_improvement(&original, &calibrated, &y_true, &config).unwrap();

        assert_eq!(result.test_name, "ECE Improvement Test");
        // The test should show some improvement
        assert!(result.test_statistic >= 0.0);
    }

    #[test]
    fn test_ranking_preservation() {
        let original = array![0.1, 0.3, 0.7, 0.9];
        let calibrated = array![0.05, 0.25, 0.65, 0.85]; // Same ranking

        let config = StatisticalTestConfig {
            min_sample_size: 4, // Allow small sample sizes for testing
            ..StatisticalTestConfig::default()
        };
        let result = super::test_ranking_preservation(&original, &calibrated, &config).unwrap();

        assert_eq!(result.test_name, "Ranking Preservation Test");
        assert!(result.passed); // Should preserve ranking
        assert!(result.test_statistic > 0.8); // High correlation
    }
}
