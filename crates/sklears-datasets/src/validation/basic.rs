//! Basic validation functions for dataset statistical properties
//!
//! This module provides fundamental validation functions that check
//! basic statistical properties of datasets including size, consistency,
//! distribution properties, correlations, and outliers.

use super::types::{ValidationConfig, ValidationReport, ValidationResult};
use std::collections::HashMap;

/// Validate basic statistical properties of a dataset
pub fn validate_basic_statistics(data: &[Vec<f64>], config: &ValidationConfig) -> ValidationReport {
    let mut report = ValidationReport::new();

    if data.is_empty() {
        report.add_result(ValidationResult {
            property: "Dataset Size".to_string(),
            passed: false,
            expected: config.min_samples as f64,
            actual: 0.0,
            tolerance: 0.0,
            message: "Dataset is empty".to_string(),
        });
        return report;
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Check minimum sample size
    let min_samples_passed = n_samples >= config.min_samples;
    report.add_result(ValidationResult {
        property: "Minimum Samples".to_string(),
        passed: min_samples_passed,
        expected: config.min_samples as f64,
        actual: n_samples as f64,
        tolerance: 0.0,
        message: if min_samples_passed {
            format!(
                "Dataset has {} samples (≥ {})",
                n_samples, config.min_samples
            )
        } else {
            format!(
                "Dataset has {} samples (< {})",
                n_samples, config.min_samples
            )
        },
    });

    // Validate feature consistency
    let consistent_features = data.iter().all(|row| row.len() == n_features);
    report.add_result(ValidationResult {
        property: "Feature Consistency".to_string(),
        passed: consistent_features,
        expected: n_features as f64,
        actual: if consistent_features {
            n_features as f64
        } else {
            -1.0
        },
        tolerance: 0.0,
        message: if consistent_features {
            format!("All samples have {} features", n_features)
        } else {
            "Inconsistent number of features across samples".to_string()
        },
    });

    // Check for NaN and infinite values
    let mut has_nan = false;
    let mut has_inf = false;

    for row in data {
        for &value in row {
            if value.is_nan() {
                has_nan = true;
            }
            if value.is_infinite() {
                has_inf = true;
            }
        }
    }

    report.add_result(ValidationResult {
        property: "No NaN Values".to_string(),
        passed: !has_nan,
        expected: 0.0,
        actual: if has_nan { 1.0 } else { 0.0 },
        tolerance: 0.0,
        message: if has_nan {
            "Dataset contains NaN values".to_string()
        } else {
            "No NaN values found".to_string()
        },
    });

    report.add_result(ValidationResult {
        property: "No Infinite Values".to_string(),
        passed: !has_inf,
        expected: 0.0,
        actual: if has_inf { 1.0 } else { 0.0 },
        tolerance: 0.0,
        message: if has_inf {
            "Dataset contains infinite values".to_string()
        } else {
            "No infinite values found".to_string()
        },
    });

    report
}

/// Validate distribution properties (mean, variance, etc.)
pub fn validate_distribution_properties(
    data: &[Vec<f64>],
    expected_mean: Option<f64>,
    expected_std: Option<f64>,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    if data.is_empty() {
        return report;
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Calculate statistics for each feature
    for feature_idx in 0..n_features {
        let values: Vec<f64> = data.iter().map(|row| row[feature_idx]).collect();

        // Calculate mean
        let mean = values.iter().sum::<f64>() / n_samples as f64;

        // Calculate standard deviation
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
        let std_dev = variance.sqrt();

        // Validate mean if expected
        if let Some(expected_mean) = expected_mean {
            let mean_diff = (mean - expected_mean).abs();
            let mean_passed = mean_diff <= config.tolerance;

            report.add_result(ValidationResult {
                property: format!("Feature {} Mean", feature_idx),
                passed: mean_passed,
                expected: expected_mean,
                actual: mean,
                tolerance: config.tolerance,
                message: if mean_passed {
                    format!(
                        "Mean {:.4} is within tolerance of {:.4}",
                        mean, expected_mean
                    )
                } else {
                    format!(
                        "Mean {:.4} differs from expected {:.4} by {:.4}",
                        mean, expected_mean, mean_diff
                    )
                },
            });
        }

        // Validate standard deviation if expected
        if let Some(expected_std) = expected_std {
            let std_diff = (std_dev - expected_std).abs();
            let std_passed = std_diff <= config.tolerance;

            report.add_result(ValidationResult {
                property: format!("Feature {} Std Dev", feature_idx),
                passed: std_passed,
                expected: expected_std,
                actual: std_dev,
                tolerance: config.tolerance,
                message: if std_passed {
                    format!(
                        "Std dev {:.4} is within tolerance of {:.4}",
                        std_dev, expected_std
                    )
                } else {
                    format!(
                        "Std dev {:.4} differs from expected {:.4} by {:.4}",
                        std_dev, expected_std, std_diff
                    )
                },
            });
        }
    }

    report
}

/// Validate correlation structure
pub fn validate_correlation_structure(
    data: &[Vec<f64>],
    expected_correlations: Option<&HashMap<(usize, usize), f64>>,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    if data.is_empty() {
        return report;
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    if n_features < 2 {
        return report;
    }

    // Calculate correlation matrix
    let mut correlations = HashMap::new();

    for i in 0..n_features {
        for j in (i + 1)..n_features {
            let values_i: Vec<f64> = data.iter().map(|row| row[i]).collect();
            let values_j: Vec<f64> = data.iter().map(|row| row[j]).collect();

            let mean_i = values_i.iter().sum::<f64>() / n_samples as f64;
            let mean_j = values_j.iter().sum::<f64>() / n_samples as f64;

            let numerator: f64 = values_i
                .iter()
                .zip(values_j.iter())
                .map(|(&x, &y)| (x - mean_i) * (y - mean_j))
                .sum();

            let sum_sq_i: f64 = values_i.iter().map(|&x| (x - mean_i).powi(2)).sum();
            let sum_sq_j: f64 = values_j.iter().map(|&x| (x - mean_j).powi(2)).sum();

            let correlation = if sum_sq_i > 0.0 && sum_sq_j > 0.0 {
                numerator / (sum_sq_i.sqrt() * sum_sq_j.sqrt())
            } else {
                0.0
            };

            correlations.insert((i, j), correlation);
        }
    }

    // Validate expected correlations
    if let Some(expected_correlations) = expected_correlations {
        for (&(i, j), &expected_corr) in expected_correlations {
            if let Some(&actual_corr) = correlations.get(&(i, j)) {
                let corr_diff = (actual_corr - expected_corr).abs();
                let corr_passed = corr_diff <= config.tolerance;

                report.add_result(ValidationResult {
                    property: format!("Correlation [{}, {}]", i, j),
                    passed: corr_passed,
                    expected: expected_corr,
                    actual: actual_corr,
                    tolerance: config.tolerance,
                    message: if corr_passed {
                        format!(
                            "Correlation {:.4} is within tolerance of {:.4}",
                            actual_corr, expected_corr
                        )
                    } else {
                        format!(
                            "Correlation {:.4} differs from expected {:.4} by {:.4}",
                            actual_corr, expected_corr, corr_diff
                        )
                    },
                });
            }
        }
    }

    report
}

/// Validate normality using Shapiro-Wilk approximation
pub fn validate_normality(data: &[Vec<f64>], _config: &ValidationConfig) -> ValidationReport {
    let mut report = ValidationReport::new();

    if data.is_empty() {
        return report;
    }

    let n_features = data[0].len();

    for feature_idx in 0..n_features {
        let mut values: Vec<f64> = data.iter().map(|row| row[feature_idx]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        if n < 3 {
            continue;
        }

        // Simple normality test: check if data is approximately normal
        // by comparing percentiles with expected normal distribution
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            continue;
        }

        // Check if 68% of data is within 1 standard deviation
        let within_1_std = values
            .iter()
            .filter(|&&x| (x - mean).abs() <= std_dev)
            .count();
        let expected_within_1_std = (n as f64 * 0.68) as usize;
        let tolerance_1_std = (n as f64 * 0.1) as usize; // 10% tolerance

        let normality_passed =
            (within_1_std as i32 - expected_within_1_std as i32).abs() <= tolerance_1_std as i32;

        report.add_result(ValidationResult {
            property: format!("Feature {} Normality", feature_idx),
            passed: normality_passed,
            expected: expected_within_1_std as f64,
            actual: within_1_std as f64,
            tolerance: tolerance_1_std as f64,
            message: if normality_passed {
                format!("Feature {} appears normally distributed", feature_idx)
            } else {
                format!("Feature {} may not be normally distributed", feature_idx)
            },
        });
    }

    report
}

/// Validate outlier detection
pub fn validate_outliers(
    data: &[Vec<f64>],
    expected_outlier_ratio: Option<f64>,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    if data.is_empty() {
        return report;
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    for feature_idx in 0..n_features {
        let values: Vec<f64> = data.iter().map(|row| row[feature_idx]).collect();

        // Calculate Q1, Q3, and IQR
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = n_samples / 4;
        let q3_idx = 3 * n_samples / 4;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;

        // Count outliers (values beyond 1.5 * IQR from quartiles)
        let outlier_threshold = 1.5 * iqr;
        let outliers = values
            .iter()
            .filter(|&&x| x < q1 - outlier_threshold || x > q3 + outlier_threshold)
            .count();

        let outlier_ratio = outliers as f64 / n_samples as f64;

        if let Some(expected_ratio) = expected_outlier_ratio {
            let ratio_diff = (outlier_ratio - expected_ratio).abs();
            let outlier_passed = ratio_diff <= config.tolerance;

            report.add_result(ValidationResult {
                property: format!("Feature {} Outlier Ratio", feature_idx),
                passed: outlier_passed,
                expected: expected_ratio,
                actual: outlier_ratio,
                tolerance: config.tolerance,
                message: if outlier_passed {
                    format!(
                        "Outlier ratio {:.4} is within tolerance of {:.4}",
                        outlier_ratio, expected_ratio
                    )
                } else {
                    format!(
                        "Outlier ratio {:.4} differs from expected {:.4} by {:.4}",
                        outlier_ratio, expected_ratio, ratio_diff
                    )
                },
            });
        }
    }

    report
}
