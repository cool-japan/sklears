//! Missing data and outlier dataset generators
//!
//! This module provides generators for introducing missing data patterns and outliers
//! into existing datasets, which is useful for testing imputation methods and robust algorithms.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::{Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};

/// Introduce missing values completely at random (MCAR)
///
/// Creates missing values that are independent of both observed and unobserved data.
/// This is the simplest missing data pattern and easiest to handle statistically.
///
/// # Parameters
/// - `data`: Input data matrix
/// - `missing_rate`: Proportion of values to make missing (0.0 to 1.0)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Data matrix with missing values (NaN) introduced randomly
pub fn make_missing_completely_at_random(
    data: &Array2<f64>,
    missing_rate: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if missing_rate < 0.0 || missing_rate > 1.0 {
        return Err(SklearsError::InvalidInput(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut result = data.clone();
    let (n_rows, n_cols) = data.dim();

    for i in 0..n_rows {
        for j in 0..n_cols {
            if rng.gen() < missing_rate {
                result[[i, j]] = f64::NAN;
            }
        }
    }

    Ok(result)
}

/// Introduce missing values at random (MAR)
///
/// Creates missing values that depend on observed data but not on the missing values themselves.
/// The missingness pattern depends on values in a predictor column.
///
/// # Parameters
/// - `data`: Input data matrix
/// - `missing_rate`: Base proportion of values to make missing
/// - `predictor_column`: Column index that determines missingness pattern
/// - `threshold_percentile`: Percentile threshold for the predictor (0.0 to 100.0)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Data matrix with MAR missing pattern
pub fn make_missing_at_random(
    data: &Array2<f64>,
    missing_rate: f64,
    predictor_column: usize,
    threshold_percentile: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if missing_rate < 0.0 || missing_rate > 1.0 {
        return Err(SklearsError::InvalidInput(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if threshold_percentile < 0.0 || threshold_percentile > 100.0 {
        return Err(SklearsError::InvalidInput(
            "threshold_percentile must be between 0.0 and 100.0".to_string(),
        ));
    }

    let (n_rows, n_cols) = data.dim();
    if predictor_column >= n_cols {
        return Err(SklearsError::InvalidInput(
            "predictor_column must be less than the number of columns".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut result = data.clone();

    // Calculate threshold based on percentile of predictor column
    let mut predictor_values: Vec<f64> = data.column(predictor_column).to_vec();
    predictor_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold_index = (threshold_percentile / 100.0 * n_rows as f64) as usize;
    let threshold = predictor_values[threshold_index.min(n_rows - 1)];

    for i in 0..n_rows {
        let predictor_value = data[[i, predictor_column]];
        let missing_prob = if predictor_value > threshold {
            missing_rate * 2.0
        } else {
            missing_rate * 0.5
        };

        for j in 0..n_cols {
            if j != predictor_column && rng.gen() < missing_prob {
                result[[i, j]] = f64::NAN;
            }
        }
    }

    Ok(result)
}

/// Introduce missing values not at random (MNAR)
///
/// Creates missing values that depend on the unobserved values themselves.
/// Higher values in the target column are more likely to be missing.
///
/// # Parameters
/// - `data`: Input data matrix
/// - `missing_rate`: Base proportion of values to make missing
/// - `target_column`: Column index where missingness depends on its own values
/// - `threshold_percentile`: Percentile threshold for missingness (0.0 to 100.0)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Data matrix with MNAR missing pattern
pub fn make_missing_not_at_random(
    data: &Array2<f64>,
    missing_rate: f64,
    target_column: usize,
    threshold_percentile: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if missing_rate < 0.0 || missing_rate > 1.0 {
        return Err(SklearsError::InvalidInput(
            "missing_rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if threshold_percentile < 0.0 || threshold_percentile > 100.0 {
        return Err(SklearsError::InvalidInput(
            "threshold_percentile must be between 0.0 and 100.0".to_string(),
        ));
    }

    let (n_rows, n_cols) = data.dim();
    if target_column >= n_cols {
        return Err(SklearsError::InvalidInput(
            "target_column must be less than the number of columns".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut result = data.clone();

    // Calculate threshold based on percentile of target column
    let mut target_values: Vec<f64> = data.column(target_column).to_vec();
    target_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold_index = (threshold_percentile / 100.0 * n_rows as f64) as usize;
    let threshold = target_values[threshold_index.min(n_rows - 1)];

    for i in 0..n_rows {
        let target_value = data[[i, target_column]];

        // Higher values are more likely to be missing (MNAR pattern)
        let missing_prob = if target_value > threshold {
            missing_rate * 3.0
        } else {
            missing_rate * 0.2
        };

        if rng.gen() < missing_prob {
            result[[i, target_column]] = f64::NAN;
        }
    }

    Ok(result)
}

/// Introduce outliers into a dataset
///
/// Adds outliers by modifying existing data points to have extreme values.
/// Useful for testing robust algorithms and outlier detection methods.
///
/// # Parameters
/// - `data`: Input data matrix
/// - `outlier_fraction`: Proportion of samples to make outliers (0.0 to 1.0)
/// - `outlier_magnitude`: Magnitude of outlier deviation (in standard deviations)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Data matrix with outliers introduced
pub fn make_outliers(
    data: &Array2<f64>,
    outlier_fraction: f64,
    outlier_magnitude: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if outlier_fraction < 0.0 || outlier_fraction > 1.0 {
        return Err(SklearsError::InvalidInput(
            "outlier_fraction must be between 0.0 and 1.0".to_string(),
        ));
    }

    if outlier_magnitude <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "outlier_magnitude must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut result = data.clone();
    let (n_rows, n_cols) = data.dim();
    let n_outliers = (outlier_fraction * n_rows as f64) as usize;

    // Calculate statistics for each feature
    let means = data.mean_axis(Axis(0)).unwrap();
    let stds = data.std_axis(Axis(0), 0.0);

    // Randomly select samples to become outliers
    let mut outlier_indices: Vec<usize> = (0..n_rows).collect();
    for i in (1..outlier_indices.len()).rev() {
        let j = rng.gen_range(0..i + 1);
        outlier_indices.swap(i, j);
    }
    outlier_indices.truncate(n_outliers);

    // Modify selected samples
    for &sample_idx in &outlier_indices {
        for feature_idx in 0..n_cols {
            let mean = means[feature_idx];
            let std = stds[feature_idx];

            // Add extreme deviation
            let sign = if rng.gen() < 0.5 { -1.0 } else { 1.0 };
            let deviation = sign * outlier_magnitude * std;
            result[[sample_idx, feature_idx]] = mean + deviation;
        }
    }

    Ok(result)
}

/// Generate imbalanced classification dataset
///
/// Creates a classification dataset with specified class imbalance ratios.
/// Useful for testing algorithms designed to handle imbalanced data.
///
/// # Parameters
/// - `n_samples`: Total number of samples to generate
/// - `n_features`: Number of features
/// - `n_classes`: Number of classes
/// - `class_weights`: Weight for each class (must sum to 1.0)
/// - `cluster_std`: Standard deviation of clusters
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (features, imbalanced_labels)
pub fn make_imbalanced_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    class_weights: &Array1<f64>,
    cluster_std: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 || n_classes == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_features, and n_classes must be positive".to_string(),
        ));
    }

    if class_weights.len() != n_classes {
        return Err(SklearsError::InvalidInput(
            "class_weights must have same length as n_classes".to_string(),
        ));
    }

    // Check if weights sum to approximately 1.0
    let weight_sum = class_weights.sum();
    if (weight_sum - 1.0).abs() > 1e-10 {
        return Err(SklearsError::InvalidInput(
            "class_weights must sum to 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Generate cluster centers for each class
    let mut centers = Array2::zeros((n_classes, n_features));
    for i in 0..n_classes {
        for j in 0..n_features {
            centers[[i, j]] = rng.random_range(-5.0, 5.0);
        }
    }

    // Calculate number of samples per class
    let mut class_sizes = Array1::zeros(n_classes);
    let mut remaining_samples = n_samples;

    for i in 0..(n_classes - 1) {
        let class_size = (class_weights[i] * n_samples as f64) as usize;
        class_sizes[i] = class_size as f64;
        remaining_samples -= class_size;
    }
    class_sizes[n_classes - 1] = remaining_samples as f64;

    // Generate samples
    let mut data = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    let mut sample_idx = 0;

    for class_idx in 0..n_classes {
        let n_class_samples = class_sizes[class_idx] as usize;

        for _ in 0..n_class_samples {
            labels[sample_idx] = class_idx as i32;

            for feature_idx in 0..n_features {
                let center = centers[[class_idx, feature_idx]];
                let normal = Normal::new(center, cluster_std).unwrap();
                data[[sample_idx, feature_idx]] = rng.sample(normal);
            }

            sample_idx += 1;
        }
    }

    Ok((data, labels))
}

/// Generate anomaly patterns in time series or structured data
///
/// Creates various types of anomalies including point anomalies, contextual anomalies,
/// and collective anomalies for testing anomaly detection algorithms.
///
/// # Parameters
/// - `data`: Input data matrix
/// - `anomaly_fraction`: Proportion of samples to make anomalous
/// - `anomaly_type`: Type of anomaly ("point", "contextual", "collective")
/// - `severity`: Severity of anomalies (1.0 = mild, 5.0 = severe)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (data_with_anomalies, anomaly_labels)
pub fn make_anomalies(
    data: &Array2<f64>,
    anomaly_fraction: f64,
    anomaly_type: &str,
    severity: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if anomaly_fraction < 0.0 || anomaly_fraction > 1.0 {
        return Err(SklearsError::InvalidInput(
            "anomaly_fraction must be between 0.0 and 1.0".to_string(),
        ));
    }

    if severity <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "severity must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut result = data.clone();
    let (n_rows, n_cols) = data.dim();
    let n_anomalies = (anomaly_fraction * n_rows as f64) as usize;

    let mut labels = Array1::zeros(n_rows);

    // Calculate data statistics
    let means = data.mean_axis(Axis(0)).unwrap();
    let stds = data.std_axis(Axis(0), 0.0);

    // Select anomaly indices
    let mut anomaly_indices: Vec<usize> = (0..n_rows).collect();
    for i in (1..anomaly_indices.len()).rev() {
        let j = rng.gen_range(0..i + 1);
        anomaly_indices.swap(i, j);
    }
    anomaly_indices.truncate(n_anomalies);

    match anomaly_type {
        "point" => {
            // Point anomalies: individual data points that are anomalous
            for &idx in &anomaly_indices {
                labels[idx] = 1;
                for j in 0..n_cols {
                    let sign = if rng.gen() < 0.5 { -1.0 } else { 1.0 };
                    result[[idx, j]] = means[j] + sign * severity * stds[j];
                }
            }
        }

        "contextual" => {
            // Contextual anomalies: anomalous in specific context
            for &idx in &anomaly_indices {
                labels[idx] = 1;
                // Modify only a subset of features to create context-dependent anomalies
                let n_features_to_modify = (n_cols / 2).max(1);
                for j in 0..n_features_to_modify {
                    let sign = if rng.gen() < 0.5 { -1.0 } else { 1.0 };
                    result[[idx, j]] = means[j] + sign * severity * stds[j];
                }
            }
        }

        "collective" => {
            // Collective anomalies: groups of data points that are collectively anomalous
            let group_size = (n_anomalies / 3).max(1);
            for chunk in anomaly_indices.chunks(group_size) {
                // Add same pattern to all points in the group
                let pattern_feature = rng.gen_range(0..n_cols);
                let pattern_magnitude = severity * stds[pattern_feature];

                for &idx in chunk {
                    labels[idx] = 1;
                    result[[idx, pattern_feature]] += pattern_magnitude;
                }
            }
        }

        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown anomaly_type: {}. Use 'point', 'contextual', or 'collective'",
                anomaly_type
            )));
        }
    }

    Ok((result, labels))
}