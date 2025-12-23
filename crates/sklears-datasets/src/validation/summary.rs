//! Statistical summary generation functions
//!
//! This module provides functions for generating comprehensive
//! statistical summaries of datasets including feature statistics,
//! target analysis, and correlation matrices.

use super::types::{FeatureStatistics, StatisticalSummary, SummaryConfig, TargetStatistics};
use std::collections::{HashMap, HashSet};

/// Generate comprehensive statistical summary for a dataset
pub fn generate_statistical_summary(
    data: &[Vec<f64>],
    targets: Option<&[f64]>,
    config: &SummaryConfig,
) -> StatisticalSummary {
    let dataset_name = config
        .target_name
        .clone()
        .unwrap_or_else(|| "Dataset".to_string());
    let n_samples = data.len();
    let n_features = if data.is_empty() { 0 } else { data[0].len() };

    let mut summary = StatisticalSummary::new(dataset_name, n_samples, n_features);

    if data.is_empty() {
        return summary;
    }

    // Generate feature statistics
    for feature_idx in 0..n_features {
        let feature_name = config
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_idx))
            .cloned()
            .unwrap_or_else(|| format!("feature_{}", feature_idx));

        let feature_stats = calculate_feature_statistics(data, feature_idx, &feature_name, config);
        summary.add_feature_stats(feature_stats);
    }

    // Generate target statistics if provided
    if let Some(targets) = targets {
        let target_name = config
            .target_name
            .clone()
            .unwrap_or_else(|| "target".to_string());
        let target_stats = calculate_target_statistics(targets, &target_name, config);
        summary.set_target_stats(target_stats);
    }

    // Calculate correlation matrix if requested
    if config.include_correlation_matrix && n_features > 1 {
        summary.correlation_matrix = calculate_correlation_matrix(data);
    }

    // Calculate data quality score
    summary.calculate_quality_score();

    // Add metadata
    summary
        .metadata
        .insert("n_samples".to_string(), n_samples.to_string());
    summary
        .metadata
        .insert("n_features".to_string(), n_features.to_string());
    summary.metadata.insert(
        "outlier_threshold".to_string(),
        config.outlier_threshold.to_string(),
    );

    summary
}

/// Calculate comprehensive statistics for a single feature
fn calculate_feature_statistics(
    data: &[Vec<f64>],
    feature_idx: usize,
    feature_name: &str,
    config: &SummaryConfig,
) -> FeatureStatistics {
    let values: Vec<f64> = data.iter().map(|row| row[feature_idx]).collect();
    let count = values.len();

    // Handle missing values (NaN)
    let valid_values: Vec<f64> = values.iter().filter(|&&x| !x.is_nan()).cloned().collect();
    let missing_count = count - valid_values.len();
    let missing_ratio = missing_count as f64 / count as f64;

    if valid_values.is_empty() {
        return FeatureStatistics {
            name: feature_name.to_string(),
            count,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            variance: 0.0,
            min: 0.0,
            max: 0.0,
            range: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            q1: 0.0,
            q3: 0.0,
            iqr: 0.0,
            outlier_count: 0,
            outlier_ratio: 0.0,
            missing_count,
            missing_ratio,
            unique_count: 0,
            mode: None,
            percentiles: HashMap::new(),
        };
    }

    // Sort values for percentile calculations
    let mut sorted_values = valid_values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Basic statistics
    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
    let variance = valid_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / valid_values.len() as f64;
    let std_dev = variance.sqrt();
    let min = sorted_values[0];
    let max = sorted_values[sorted_values.len() - 1];
    let range = max - min;

    // Percentiles
    let median = calculate_percentile(&sorted_values, 50.0);
    let q1 = calculate_percentile(&sorted_values, 25.0);
    let q3 = calculate_percentile(&sorted_values, 75.0);
    let iqr = q3 - q1;

    // Skewness and kurtosis
    let skewness = if std_dev > 0.0 {
        let n = valid_values.len() as f64;
        let skew_sum = valid_values
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>();
        skew_sum / n
    } else {
        0.0
    };

    let kurtosis = if std_dev > 0.0 {
        let n = valid_values.len() as f64;
        let kurt_sum = valid_values
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>();
        kurt_sum / n - 3.0 // Excess kurtosis
    } else {
        0.0
    };

    // Outlier detection using IQR method
    let outlier_threshold = config.outlier_threshold * iqr;
    let outlier_count = valid_values
        .iter()
        .filter(|&&x| x < q1 - outlier_threshold || x > q3 + outlier_threshold)
        .count();
    let outlier_ratio = outlier_count as f64 / valid_values.len() as f64;

    // Unique values and mode
    let mut value_counts: HashMap<String, usize> = HashMap::new();
    for &value in &valid_values {
        let key = format!("{:.6}", value); // Round to 6 decimal places for grouping
        *value_counts.entry(key).or_insert(0) += 1;
    }
    let unique_count = value_counts.len();
    let mode = value_counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .and_then(|(value_str, _)| value_str.parse::<f64>().ok());

    // Calculate percentiles if requested
    let mut percentiles = HashMap::new();
    if config.include_percentiles {
        for &p in &config.percentile_values {
            let percentile_value = calculate_percentile(&sorted_values, p as f64);
            percentiles.insert(p, percentile_value);
        }
    }

    FeatureStatistics {
        name: feature_name.to_string(),
        count,
        mean,
        median,
        std_dev,
        variance,
        min,
        max,
        range,
        skewness,
        kurtosis,
        q1,
        q3,
        iqr,
        outlier_count,
        outlier_ratio,
        missing_count,
        missing_ratio,
        unique_count,
        mode,
        percentiles,
    }
}

/// Calculate statistics for target/label data
fn calculate_target_statistics(
    targets: &[f64],
    target_name: &str,
    config: &SummaryConfig,
) -> TargetStatistics {
    let count = targets.len();
    let valid_targets: Vec<f64> = targets.iter().filter(|&&x| !x.is_nan()).cloned().collect();
    let missing_count = count - valid_targets.len();
    let missing_ratio = missing_count as f64 / count as f64;

    // Determine if targets are continuous or categorical
    let unique_values: HashSet<String> =
        valid_targets.iter().map(|&x| format!("{:.6}", x)).collect();
    let unique_count = unique_values.len();

    // Check if values are all integers (categorical indicator)
    let all_integers = valid_targets.iter().all(|&x| x.fract() == 0.0);

    // Heuristic: if unique values are less than 20% of total or less than 10 AND all integers, treat as categorical
    let is_categorical =
        all_integers && (unique_count <= 10 || (unique_count as f64 / count as f64) < 0.2);

    let data_type = if unique_count == 2 && all_integers {
        "binary".to_string()
    } else if is_categorical {
        "categorical".to_string()
    } else {
        "continuous".to_string()
    };

    // Calculate class distribution for categorical targets
    let mut class_distribution = HashMap::new();
    let mut class_balance_ratio = 1.0;
    let mut entropy = 0.0;

    if is_categorical {
        for &target in &valid_targets {
            let class_str = if target.fract() == 0.0 {
                format!("{:.0}", target)
            } else {
                format!("{:.6}", target)
            };
            *class_distribution.entry(class_str).or_insert(0) += 1;
        }

        // Calculate class balance ratio (min class count / max class count)
        if let (Some(min_count), Some(max_count)) = (
            class_distribution.values().min(),
            class_distribution.values().max(),
        ) {
            class_balance_ratio = *min_count as f64 / *max_count as f64;
        }

        // Calculate entropy
        let total = valid_targets.len() as f64;
        entropy = class_distribution
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();
    }

    // Calculate continuous statistics if needed
    let continuous_stats = if data_type == "continuous" {
        Some(calculate_feature_statistics(
            &[valid_targets.clone()],
            0,
            target_name,
            config,
        ))
    } else {
        None
    };

    TargetStatistics {
        name: target_name.to_string(),
        data_type,
        count,
        unique_count,
        missing_count,
        missing_ratio,
        class_distribution,
        class_balance_ratio,
        entropy,
        continuous_stats,
    }
}

/// Calculate correlation matrix for features
fn calculate_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_features = data[0].len();
    let n_samples = data.len();
    let mut correlation_matrix = vec![vec![0.0; n_features]; n_features];

    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                correlation_matrix[i][j] = 1.0;
            } else {
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

                correlation_matrix[i][j] = correlation;
            }
        }
    }

    correlation_matrix
}

/// Calculate percentile value from sorted data
fn calculate_percentile(sorted_data: &[f64], percentile: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let index = (percentile / 100.0) * (sorted_data.len() - 1) as f64;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        sorted_data[lower_index]
    } else {
        let lower_value = sorted_data[lower_index];
        let upper_value = sorted_data[upper_index];
        let weight = index - lower_index as f64;
        lower_value + weight * (upper_value - lower_value)
    }
}
