//! Advanced validation functions for comprehensive dataset analysis
//!
//! This module provides advanced validation capabilities including
//! comprehensive dataset validation, quality metrics calculation,
//! data drift detection, and anomaly detection.

use super::basic::{
    validate_basic_statistics, validate_correlation_structure, validate_distribution_properties,
    validate_normality, validate_outliers,
};
use super::types::{
    AnomalyDetectionResult, DataDriftReport, DatasetQualityMetrics, ValidationConfig,
    ValidationReport,
};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Comprehensive dataset validation
pub fn validate_dataset(
    data: &[Vec<f64>],
    expected_mean: Option<f64>,
    expected_std: Option<f64>,
    expected_correlations: Option<&HashMap<(usize, usize), f64>>,
    expected_outlier_ratio: Option<f64>,
    config: &ValidationConfig,
) -> ValidationReport {
    let mut combined_report = ValidationReport::new();

    // Run all validation checks
    let basic_report = validate_basic_statistics(data, config);
    let distribution_report =
        validate_distribution_properties(data, expected_mean, expected_std, config);
    let correlation_report = validate_correlation_structure(data, expected_correlations, config);
    let outlier_report = validate_outliers(data, expected_outlier_ratio, config);

    // Combine results
    for result in basic_report.results {
        combined_report.add_result(result);
    }
    for result in distribution_report.results {
        combined_report.add_result(result);
    }
    for result in correlation_report.results {
        combined_report.add_result(result);
    }
    for result in outlier_report.results {
        combined_report.add_result(result);
    }

    // Add normality check if requested
    if config.check_normality {
        let normality_report = validate_normality(data, config);
        for result in normality_report.results {
            combined_report.add_result(result);
        }
    }

    combined_report
}

/// Calculate comprehensive dataset quality metrics
pub fn calculate_dataset_quality_metrics(
    data: &[Vec<f64>],
    targets: Option<&[f64]>,
    _config: &ValidationConfig,
) -> DatasetQualityMetrics {
    let mut metrics = DatasetQualityMetrics {
        overall_quality_score: 0.0,
        completeness_score: 100.0,
        consistency_score: 100.0,
        validity_score: 100.0,
        accuracy_score: 100.0,
        uniqueness_score: 100.0,
        timeliness_score: 100.0,
        missing_data_ratio: 0.0,
        outlier_ratio: 0.0,
        duplicate_ratio: 0.0,
        data_type_violations: 0,
        range_violations: 0,
        pattern_violations: 0,
        fingerprint: String::new(),
        quality_issues: Vec::new(),
        recommendations: Vec::new(),
    };

    if data.is_empty() {
        metrics.add_issue(
            "Dataset is empty".to_string(),
            "Provide a non-empty dataset".to_string(),
        );
        metrics.completeness_score = 0.0;
        metrics.calculate_overall_score();
        return metrics;
    }

    let n_samples = data.len();
    let n_features = data[0].len();
    let total_values = n_samples * n_features;

    // Calculate completeness score (missing data assessment)
    let mut missing_count = 0;
    for row in data {
        for &value in row {
            if value.is_nan() {
                missing_count += 1;
            }
        }
    }
    metrics.missing_data_ratio = missing_count as f64 / total_values as f64;
    metrics.completeness_score = (1.0 - metrics.missing_data_ratio) * 100.0;

    if metrics.missing_data_ratio > 0.1 {
        metrics.add_issue(
            format!(
                "High missing data ratio: {:.2}%",
                metrics.missing_data_ratio * 100.0
            ),
            "Consider imputation strategies or data collection improvements".to_string(),
        );
    }

    // Calculate consistency score (data type and format consistency)
    let mut inconsistent_features = 0;
    for row in data {
        if row.len() != n_features {
            inconsistent_features += 1;
        }
        for &value in row {
            if value.is_infinite() {
                metrics.data_type_violations += 1;
            }
        }
    }

    if inconsistent_features > 0 {
        metrics.consistency_score = (1.0 - inconsistent_features as f64 / n_samples as f64) * 100.0;
        metrics.add_issue(
            "Inconsistent number of features across samples".to_string(),
            "Ensure all samples have the same number of features".to_string(),
        );
    }

    // Calculate validity score (outlier detection)
    let mut total_outliers = 0;
    for feature_idx in 0..n_features {
        let values: Vec<f64> = data
            .iter()
            .map(|row| row[feature_idx])
            .filter(|&x| !x.is_nan())
            .collect();

        if values.len() > 4 {
            let outliers = detect_outliers_iqr(&values, 1.5);
            total_outliers += outliers.len();
        }
    }

    metrics.outlier_ratio = total_outliers as f64 / (n_samples * n_features) as f64;
    metrics.validity_score = (1.0 - metrics.outlier_ratio.min(0.2) / 0.2) * 100.0;

    if metrics.outlier_ratio > 0.05 {
        metrics.add_issue(
            format!("High outlier ratio: {:.2}%", metrics.outlier_ratio * 100.0),
            "Review data collection process or apply outlier treatment".to_string(),
        );
    }

    // Calculate uniqueness score (duplicate detection)
    let mut unique_rows = HashSet::new();
    for row in data {
        let row_str = row
            .iter()
            .map(|&x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",");
        unique_rows.insert(row_str);
    }

    let duplicate_count = n_samples - unique_rows.len();
    metrics.duplicate_ratio = duplicate_count as f64 / n_samples as f64;
    metrics.uniqueness_score = (1.0 - metrics.duplicate_ratio) * 100.0;

    if metrics.duplicate_ratio > 0.01 {
        metrics.add_issue(
            format!(
                "Duplicate samples detected: {:.2}%",
                metrics.duplicate_ratio * 100.0
            ),
            "Remove duplicate samples or investigate data collection process".to_string(),
        );
    }

    // Calculate accuracy score (target-related if available)
    if let Some(targets) = targets {
        let target_missing = targets.iter().filter(|&&x| x.is_nan()).count();
        let target_missing_ratio = target_missing as f64 / targets.len() as f64;
        metrics.accuracy_score = (1.0 - target_missing_ratio) * 100.0;

        if target_missing_ratio > 0.05 {
            metrics.add_issue(
                format!("Missing targets: {:.2}%", target_missing_ratio * 100.0),
                "Improve target data collection or consider semi-supervised approaches".to_string(),
            );
        }
    }

    // Generate dataset fingerprint
    let mut hasher = DefaultHasher::new();
    for row in data {
        for &value in row {
            if !value.is_nan() {
                ((value * 1000000.0) as i64).hash(&mut hasher);
            }
        }
    }
    metrics.fingerprint = format!("{:x}", hasher.finish());

    // Calculate overall quality score
    metrics.calculate_overall_score();

    metrics
}

/// Detect data drift between two datasets
pub fn detect_data_drift(
    reference_data: &[Vec<f64>],
    current_data: &[Vec<f64>],
    _config: &ValidationConfig,
) -> DataDriftReport {
    let mut report = DataDriftReport::new();

    if reference_data.is_empty() || current_data.is_empty() {
        return report;
    }

    let n_features = reference_data[0].len().min(current_data[0].len());

    for feature_idx in 0..n_features {
        let ref_values: Vec<f64> = reference_data
            .iter()
            .map(|row| row[feature_idx])
            .filter(|&x| !x.is_nan())
            .collect();

        let curr_values: Vec<f64> = current_data
            .iter()
            .map(|row| row[feature_idx])
            .filter(|&x| !x.is_nan())
            .collect();

        if ref_values.len() < 10 || curr_values.len() < 10 {
            continue;
        }

        // Perform Kolmogorov-Smirnov test
        let ks_statistic = kolmogorov_smirnov_statistic(&ref_values, &curr_values);
        let feature_name = format!("feature_{}", feature_idx);

        report.add_feature_drift(feature_name, ks_statistic);
    }

    report.calculate_overall_drift();

    // Determine drift type based on feature patterns
    if report.drift_detected {
        if report.affected_features.len() > n_features / 2 {
            report.drift_type = "covariate".to_string();
        } else {
            report.drift_type = "feature_specific".to_string();
        }
    }

    report
}

/// Detect anomalies in generated data
pub fn detect_anomalies(data: &[Vec<f64>], _config: &ValidationConfig) -> AnomalyDetectionResult {
    let mut result = AnomalyDetectionResult::new();

    if data.is_empty() {
        return result;
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Use Isolation Forest-like approach: detect outliers using distance-based method
    let mut anomaly_scores = vec![0.0; n_samples];

    for sample_idx in 0..n_samples {
        let mut distances = Vec::new();

        // Calculate distances to other samples
        for other_idx in 0..n_samples {
            if sample_idx != other_idx {
                let distance = euclidean_distance(&data[sample_idx], &data[other_idx]);
                distances.push(distance);
            }
        }

        // Use average distance as anomaly score
        if !distances.is_empty() {
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k = (distances.len() / 10).max(1).min(10); // Use top 10% or at least 1
            let avg_distance = distances.iter().take(k).sum::<f64>() / k as f64;
            anomaly_scores[sample_idx] = avg_distance;
        }
    }

    // Determine threshold using IQR method on scores
    let mut sorted_scores = anomaly_scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q3_idx = (sorted_scores.len() * 3) / 4;
    let q1_idx = sorted_scores.len() / 4;
    let q1 = sorted_scores[q1_idx];
    let q3 = sorted_scores[q3_idx];
    let iqr = q3 - q1;
    let threshold = q3 + 1.5 * iqr;

    // Identify anomalies
    for (idx, &score) in anomaly_scores.iter().enumerate() {
        if score > threshold {
            result.add_anomaly(idx, score);
        }
    }

    result.calculate_statistics(n_samples);

    // Feature-level anomaly detection
    for feature_idx in 0..n_features {
        let feature_values: Vec<f64> = data
            .iter()
            .map(|row| row[feature_idx])
            .filter(|&x| !x.is_nan())
            .collect();

        let feature_outliers = detect_outliers_iqr(&feature_values, 1.5);
        if !feature_outliers.is_empty() {
            let feature_name = format!("feature_{}", feature_idx);
            result
                .feature_anomalies
                .insert(feature_name, feature_outliers);
        }
    }

    result
}

// Helper functions

/// Detect outliers using IQR method
fn detect_outliers_iqr(values: &[f64], threshold: f64) -> Vec<usize> {
    if values.len() < 4 {
        return Vec::new();
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1_idx = sorted_values.len() / 4;
    let q3_idx = (sorted_values.len() * 3) / 4;
    let q1 = sorted_values[q1_idx];
    let q3 = sorted_values[q3_idx];
    let iqr = q3 - q1;

    let lower_bound = q1 - threshold * iqr;
    let upper_bound = q3 + threshold * iqr;

    let mut outliers = Vec::new();
    for (idx, &value) in values.iter().enumerate() {
        if value < lower_bound || value > upper_bound {
            outliers.push(idx);
        }
    }

    outliers
}

/// Calculate Kolmogorov-Smirnov statistic between two samples
fn kolmogorov_smirnov_statistic(sample1: &[f64], sample2: &[f64]) -> f64 {
    let mut all_values = Vec::new();
    all_values.extend_from_slice(sample1);
    all_values.extend_from_slice(sample2);
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    let mut max_diff: f64 = 0.0;

    for &value in &all_values {
        let cdf1 = sample1.iter().filter(|&&x| x <= value).count() as f64 / n1;
        let cdf2 = sample2.iter().filter(|&&x| x <= value).count() as f64 / n2;
        let diff = (cdf1 - cdf2).abs();
        max_diff = max_diff.max(diff);
    }

    max_diff
}

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(vec1: &[f64], vec2: &[f64]) -> f64 {
    if vec1.len() != vec2.len() {
        return f64::INFINITY;
    }

    let sum_sq: f64 = vec1
        .iter()
        .zip(vec2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();

    sum_sq.sqrt()
}
