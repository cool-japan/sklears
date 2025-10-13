//! Distribution testing and validation functions
//!
//! This module provides functions for testing whether data follows
//! specific statistical distributions and performing goodness-of-fit tests.

use super::types::{ValidationConfig, ValidationReport, ValidationResult};

/// Distribution types for testing
#[derive(Debug, Clone)]
pub enum DistributionType {
    /// Normal

    Normal(f64, f64),  // mean, std
    /// Uniform

    Uniform(f64, f64), // min, max
    /// Exponential

    Exponential(f64),  // rate
    /// Custom

    Custom(Vec<f64>),  // reference samples
}

/// Kolmogorov-Smirnov test for distribution comparison
pub fn kolmogorov_smirnov_test(

    data: &[f64],

    expected_dist: &DistributionType,
    config: &ValidationConfig,
) -> ValidationResult {
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_data.len() as f64;
    let mut max_diff: f64 = 0.0;

    for (i, &value) in sorted_data.iter().enumerate() {
        let empirical_cdf = (i + 1) as f64 / n;
        let theoretical_cdf = match expected_dist {
            DistributionType::Normal(mean, std) => {
                // Approximate normal CDF using error function approximation
                let z = (value - mean) / std;
                0.5 * (1.0
                    + z / (1.0
                        + 0.278393 * z.abs()
                        + 0.230389 * z.abs().powi(2)
                        + 0.000972 * z.abs().powi(3)
                        + 0.078108 * z.abs().powi(4))
                    .powi(4))
            }
            DistributionType::Uniform(min, max) => {
                if value < *min {
                    0.0
                } else if value > *max {
                    1.0
                } else {
                    (value - min) / (max - min)
                }
            }
            DistributionType::Exponential(rate) => {
                if value < 0.0 {
                    0.0
                } else {
                    1.0 - (-rate * value).exp()
                }
            }
            DistributionType::Custom(ref samples) => {
                let pos = samples.iter().filter(|&&x| x <= value).count();
                pos as f64 / samples.len() as f64
            }
        };

        let diff = (empirical_cdf - theoretical_cdf).abs();
        max_diff = max_diff.max(diff);
    }

    // Critical value for α = 0.05
    let critical_value = 1.36 / (n.sqrt());
    let passed = max_diff < critical_value;

    ValidationResult {
        property: "Kolmogorov-Smirnov Test".to_string(),
        passed,
        expected: critical_value,
        actual: max_diff,
        tolerance: config.tolerance,
        message: if passed {
            "Distribution matches expected distribution".to_string()
        } else {
            "Distribution does not match expected distribution".to_string()
        },
    }
}

/// Chi-square goodness of fit test
pub fn chi_square_goodness_of_fit_test(
    data: &[f64],
    expected_dist: &DistributionType,
    num_bins: usize,
    config: &ValidationConfig,
) -> ValidationResult {
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (max_val - min_val) / num_bins as f64;

    let mut observed_counts = vec![0; num_bins];
    let mut expected_counts = vec![0.0; num_bins];

    // Count observed frequencies
    for &value in data {
        let bin_idx = if value == max_val {
            num_bins - 1
        } else {
            ((value - min_val) / bin_width).floor() as usize
        };
        observed_counts[bin_idx] += 1;
    }

    // Calculate expected frequencies
    for i in 0..num_bins {
        let bin_start = min_val + i as f64 * bin_width;
        let bin_end = min_val + (i + 1) as f64 * bin_width;

        let prob = match expected_dist {
            DistributionType::Normal(mean, std) => {
                // Simple approximation for normal distribution probability in bin
                let z1 = (bin_start - mean) / std;
                let z2 = (bin_end - mean) / std;
                let cdf1 =
                    0.5 * (1.0 + z1 / (1.0 + 0.278393 * z1.abs() + 0.230389 * z1.abs().powi(2)));
                let cdf2 =
                    0.5 * (1.0 + z2 / (1.0 + 0.278393 * z2.abs() + 0.230389 * z2.abs().powi(2)));
                cdf2 - cdf1
            }
            DistributionType::Uniform(min, max) => {
                let overlap_start = bin_start.max(*min);
                let overlap_end = bin_end.min(*max);
                if overlap_start < overlap_end {
                    (overlap_end - overlap_start) / (max - min)
                } else {
                    0.0
                }
            }
            DistributionType::Exponential(rate) => {
                let cdf1 = if bin_start < 0.0 {
                    0.0
                } else {
                    1.0 - (-rate * bin_start).exp()
                };
                let cdf2 = if bin_end < 0.0 {
                    0.0
                } else {
                    1.0 - (-rate * bin_end).exp()
                };
                cdf2 - cdf1
            }
            DistributionType::Custom(_) => {
                // For custom distributions, use uniform as fallback
                1.0 / num_bins as f64
            }
        };

        expected_counts[i] = prob * data.len() as f64;
    }

    // Calculate chi-square statistic
    let mut chi_square = 0.0;
    for i in 0..num_bins {
        if expected_counts[i] > 0.0 {
            let diff = observed_counts[i] as f64 - expected_counts[i];
            chi_square += diff * diff / expected_counts[i];
        }
    }

    // Critical value for α = 0.05, df = num_bins - 1
    let degrees_of_freedom = num_bins - 1;
    let critical_value = match degrees_of_freedom {
        1 => 3.841,
        2 => 5.991,
        3 => 7.815,
        4 => 9.488,
        5 => 11.070,
        _ => 3.841 + degrees_of_freedom as f64 * 2.0, // Rough approximation
    };

    let passed = chi_square < critical_value;

    ValidationResult {
        property: "Chi-Square Goodness of Fit".to_string(),
        passed,
        expected: critical_value,
        actual: chi_square,
        tolerance: config.tolerance,
        message: if passed {
            "Distribution passes chi-square goodness of fit test".to_string()
        } else {
            "Distribution fails chi-square goodness of fit test".to_string()
        },
    }
}

/// Test for uniform distribution
pub fn validate_uniform_distribution(
    data: &[f64],
    expected_min: f64,
    expected_max: f64,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Check if data range is within expected bounds
    let min_passed = (min_val - expected_min).abs() <= config.tolerance;
    let max_passed = (max_val - expected_max).abs() <= config.tolerance;

    report.add_result(ValidationResult {
        property: "Uniform Distribution Min".to_string(),
        passed: min_passed,
        expected: expected_min,
        actual: min_val,
        tolerance: config.tolerance,
        message: if min_passed {
            "Minimum value matches expected".to_string()
        } else {
            "Minimum value does not match expected".to_string()
        },
    });

    report.add_result(ValidationResult {
        property: "Uniform Distribution Max".to_string(),
        passed: max_passed,
        expected: expected_max,
        actual: max_val,
        tolerance: config.tolerance,
        message: if max_passed {
            "Maximum value matches expected".to_string()
        } else {
            "Maximum value does not match expected".to_string()
        },
    });

    // Use Kolmogorov-Smirnov test
    let ks_result = kolmogorov_smirnov_test(
        data,
        &DistributionType::Uniform(expected_min, expected_max),
        config,
    );
    report.add_result(ks_result);

    report
}

/// Test for normal distribution with enhanced checks
pub fn validate_normal_distribution(
    data: &[f64],
    expected_mean: f64,
    expected_std: f64,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Check mean and standard deviation
    let mean_passed = (mean - expected_mean).abs() <= config.tolerance;
    let std_passed = (std_dev - expected_std).abs() <= config.tolerance;

    report.add_result(ValidationResult {
        property: "Normal Distribution Mean".to_string(),
        passed: mean_passed,
        expected: expected_mean,
        actual: mean,
        tolerance: config.tolerance,
        message: if mean_passed {
            "Mean matches expected value".to_string()
        } else {
            "Mean does not match expected value".to_string()
        },
    });

    report.add_result(ValidationResult {
        property: "Normal Distribution Std Dev".to_string(),
        passed: std_passed,
        expected: expected_std,
        actual: std_dev,
        tolerance: config.tolerance,
        message: if std_passed {
            "Standard deviation matches expected value".to_string()
        } else {
            "Standard deviation does not match expected value".to_string()
        },
    });

    // Use Kolmogorov-Smirnov test
    let ks_result = kolmogorov_smirnov_test(
        data,
        &DistributionType::Normal(expected_mean, expected_std),
        config,
    );
    report.add_result(ks_result);

    // Use Chi-square test
    let chi_square_result = chi_square_goodness_of_fit_test(
        data,
        &DistributionType::Normal(expected_mean, expected_std),
        10,
        config,
    );
    report.add_result(chi_square_result);

    report
}

/// Test for exponential distribution
pub fn validate_exponential_distribution(
    data: &[f64],
    expected_rate: f64,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let expected_mean = 1.0 / expected_rate;

    // Check mean (should be 1/rate for exponential distribution)
    let mean_passed = (mean - expected_mean).abs() <= config.tolerance;

    report.add_result(ValidationResult {
        property: "Exponential Distribution Mean".to_string(),
        passed: mean_passed,
        expected: expected_mean,
        actual: mean,
        tolerance: config.tolerance,
        message: if mean_passed {
            "Mean matches expected value for exponential distribution".to_string()
        } else {
            "Mean does not match expected value for exponential distribution".to_string()
        },
    });

    // Use Kolmogorov-Smirnov test
    let ks_result =
        kolmogorov_smirnov_test(data, &DistributionType::Exponential(expected_rate), config);
    report.add_result(ks_result);

    report
}
