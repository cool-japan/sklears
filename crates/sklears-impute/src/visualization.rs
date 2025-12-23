//! Missing data visualization utilities
//!
//! This module provides visualization tools for understanding missing data patterns,
//! including heatmaps, pattern plots, and correlation visualizations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Missing data pattern visualization data
#[derive(Debug, Clone)]
pub struct MissingPatternPlot {
    /// Matrix indicating missingness (1 = missing, 0 = observed)
    pub missing_matrix: Array2<u8>,
    /// Pattern counts for each unique pattern
    pub pattern_counts: HashMap<Vec<u8>, usize>,
    /// Feature names or indices
    pub feature_names: Vec<String>,
    /// Sample indices
    pub sample_indices: Vec<usize>,
}

/// Missing data correlation heatmap data
#[derive(Debug, Clone)]
pub struct MissingCorrelationHeatmap {
    /// Correlation matrix between missingness indicators
    pub correlation_matrix: Array2<f64>,
    /// Feature names or indices
    pub feature_names: Vec<String>,
    /// P-values for correlations (if computed)
    pub p_values: Option<Array2<f64>>,
}

/// Missing data completeness matrix visualization data
#[derive(Debug, Clone)]
pub struct CompletenessMatrix {
    /// Matrix showing joint observation rates between features
    pub completeness_matrix: Array2<f64>,
    /// Feature names or indices
    pub feature_names: Vec<String>,
}

/// Missing data distribution plot data
#[derive(Debug, Clone)]
pub struct MissingDistributionPlot {
    /// Count of missing values per feature
    pub missing_counts: Array1<usize>,
    /// Percentage of missing values per feature
    pub missing_percentages: Array1<f64>,
    /// Feature names or indices
    pub feature_names: Vec<String>,
}

/// Generate missing data pattern visualization data
///
/// Creates a visualization matrix showing the pattern of missing data across samples and features.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `missing_values` - Value representing missing data (NaN by default)
/// * `feature_names` - Optional feature names for labeling
///
/// # Returns
///
/// A `MissingPatternPlot` struct containing the visualization data
///
/// # Examples
///
/// ```
/// use sklears_impute::visualization::create_missing_pattern_plot;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
/// let plot_data = create_missing_pattern_plot(&X.view(), f64::NAN, None).unwrap();
/// ```
#[allow(non_snake_case)]
pub fn create_missing_pattern_plot(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
    feature_names: Option<Vec<String>>,
) -> SklResult<MissingPatternPlot> {
    let X = X.mapv(|x| x);
    let (n_samples, n_features) = X.dim();

    // Generate feature names if not provided
    let feature_names = feature_names
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    if feature_names.len() != n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Number of feature names {} does not match number of features {}",
            feature_names.len(),
            n_features
        )));
    }

    // Create missing matrix
    let mut missing_matrix = Array2::zeros((n_samples, n_features));
    let mut pattern_counts: HashMap<Vec<u8>, usize> = HashMap::new();

    for i in 0..n_samples {
        let mut pattern = Vec::new();
        for j in 0..n_features {
            let is_missing = if missing_values.is_nan() {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };

            let missing_indicator = if is_missing { 1u8 } else { 0u8 };
            missing_matrix[[i, j]] = missing_indicator;
            pattern.push(missing_indicator);
        }

        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    let sample_indices: Vec<usize> = (0..n_samples).collect();

    Ok(MissingPatternPlot {
        missing_matrix,
        pattern_counts,
        feature_names,
        sample_indices,
    })
}

/// Generate missing data correlation heatmap data
///
/// Creates correlation matrix between missingness indicators of different features.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `missing_values` - Value representing missing data (NaN by default)
/// * `feature_names` - Optional feature names for labeling
/// * `compute_p_values` - Whether to compute p-values for correlations
///
/// # Returns
///
/// A `MissingCorrelationHeatmap` struct containing the correlation data
#[allow(non_snake_case)]
pub fn create_missing_correlation_heatmap(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
    feature_names: Option<Vec<String>>,
    compute_p_values: bool,
) -> SklResult<MissingCorrelationHeatmap> {
    let X = X.mapv(|x| x);
    let (n_samples, n_features) = X.dim();

    // Generate feature names if not provided
    let feature_names = feature_names
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    if feature_names.len() != n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Number of feature names {} does not match number of features {}",
            feature_names.len(),
            n_features
        )));
    }

    // Create missingness indicators
    let mut missing_indicators = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let is_missing = if missing_values.is_nan() {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };
            missing_indicators[[i, j]] = if is_missing { 1.0 } else { 0.0 };
        }
    }

    // Compute correlation matrix
    let mut correlation_matrix = Array2::zeros((n_features, n_features));
    let mut p_values = if compute_p_values {
        Some(Array2::zeros((n_features, n_features)))
    } else {
        None
    };

    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                correlation_matrix[[i, j]] = 1.0;
                if let Some(ref mut p_vals) = p_values {
                    p_vals[[i, j]] = 0.0;
                }
            } else {
                let col_i = missing_indicators.column(i);
                let col_j = missing_indicators.column(j);

                let correlation = compute_correlation(&col_i.to_owned(), &col_j.to_owned());
                correlation_matrix[[i, j]] = correlation;

                if let Some(ref mut p_vals) = p_values {
                    let p_value = compute_correlation_p_value(
                        &col_i.to_owned(),
                        &col_j.to_owned(),
                        correlation,
                    );
                    p_vals[[i, j]] = p_value;
                }
            }
        }
    }

    Ok(MissingCorrelationHeatmap {
        correlation_matrix,
        feature_names,
        p_values,
    })
}

/// Generate completeness matrix visualization data
///
/// Creates a matrix showing the joint observation rates between all pairs of features.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `missing_values` - Value representing missing data (NaN by default)
/// * `feature_names` - Optional feature names for labeling
///
/// # Returns
///
/// A `CompletenessMatrix` struct containing the completeness data
#[allow(non_snake_case)]
pub fn create_completeness_matrix(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
    feature_names: Option<Vec<String>>,
) -> SklResult<CompletenessMatrix> {
    let X = X.mapv(|x| x);
    let (n_samples, n_features) = X.dim();

    // Generate feature names if not provided
    let feature_names = feature_names
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    if feature_names.len() != n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Number of feature names {} does not match number of features {}",
            feature_names.len(),
            n_features
        )));
    }

    let mut completeness_matrix = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            let mut joint_observed = 0;

            for sample_idx in 0..n_samples {
                let i_observed = if missing_values.is_nan() {
                    !X[[sample_idx, i]].is_nan()
                } else {
                    (X[[sample_idx, i]] - missing_values).abs() >= f64::EPSILON
                };

                let j_observed = if missing_values.is_nan() {
                    !X[[sample_idx, j]].is_nan()
                } else {
                    (X[[sample_idx, j]] - missing_values).abs() >= f64::EPSILON
                };

                if i_observed && j_observed {
                    joint_observed += 1;
                }
            }

            completeness_matrix[[i, j]] = joint_observed as f64 / n_samples as f64;
        }
    }

    Ok(CompletenessMatrix {
        completeness_matrix,
        feature_names,
    })
}

/// Generate missing data distribution plot data
///
/// Creates data for plotting the distribution of missing values across features.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `missing_values` - Value representing missing data (NaN by default)
/// * `feature_names` - Optional feature names for labeling
///
/// # Returns
///
/// A `MissingDistributionPlot` struct containing the distribution data
#[allow(non_snake_case)]
pub fn create_missing_distribution_plot(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
    feature_names: Option<Vec<String>>,
) -> SklResult<MissingDistributionPlot> {
    let X = X.mapv(|x| x);
    let (n_samples, n_features) = X.dim();

    // Generate feature names if not provided
    let feature_names = feature_names
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    if feature_names.len() != n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Number of feature names {} does not match number of features {}",
            feature_names.len(),
            n_features
        )));
    }

    let mut missing_counts = Array1::zeros(n_features);
    let mut missing_percentages = Array1::zeros(n_features);

    for j in 0..n_features {
        let mut count = 0;
        for i in 0..n_samples {
            let is_missing = if missing_values.is_nan() {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };

            if is_missing {
                count += 1;
            }
        }

        missing_counts[j] = count;
        missing_percentages[j] = (count as f64 / n_samples as f64) * 100.0;
    }

    Ok(MissingDistributionPlot {
        missing_counts,
        missing_percentages,
        feature_names,
    })
}

/// Export missing pattern data to CSV format
///
/// Exports the missing pattern matrix to a CSV-like string format for external visualization.
///
/// # Parameters
///
/// * `plot_data` - Missing pattern plot data
/// * `include_headers` - Whether to include feature names as headers
///
/// # Returns
///
/// A string in CSV format representing the missing pattern matrix
pub fn export_missing_pattern_csv(plot_data: &MissingPatternPlot, include_headers: bool) -> String {
    let mut csv_lines = Vec::new();

    if include_headers {
        let header = format!("Sample,{}", plot_data.feature_names.join(","));
        csv_lines.push(header);
    }

    for (i, sample_idx) in plot_data.sample_indices.iter().enumerate() {
        let mut row = vec![sample_idx.to_string()];
        for j in 0..plot_data.missing_matrix.ncols() {
            row.push(plot_data.missing_matrix[[i, j]].to_string());
        }
        csv_lines.push(row.join(","));
    }

    csv_lines.join("\n")
}

/// Export correlation heatmap data to CSV format
///
/// Exports the correlation matrix to a CSV-like string format for external visualization.
///
/// # Parameters
///
/// * `heatmap_data` - Missing correlation heatmap data
/// * `include_headers` - Whether to include feature names as headers
///
/// # Returns
///
/// A string in CSV format representing the correlation matrix
pub fn export_correlation_csv(
    heatmap_data: &MissingCorrelationHeatmap,
    include_headers: bool,
) -> String {
    let mut csv_lines = Vec::new();

    if include_headers {
        let header = format!("Feature,{}", heatmap_data.feature_names.join(","));
        csv_lines.push(header);
    }

    for (i, feature_name) in heatmap_data.feature_names.iter().enumerate() {
        let mut row = vec![feature_name.clone()];
        for j in 0..heatmap_data.correlation_matrix.ncols() {
            row.push(format!("{:.4}", heatmap_data.correlation_matrix[[i, j]]));
        }
        csv_lines.push(row.join(","));
    }

    csv_lines.join("\n")
}

/// Generate summary statistics for missing data patterns
///
/// Provides comprehensive summary statistics about missing data patterns in the dataset.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `missing_values` - Value representing missing data (NaN by default)
///
/// # Returns
///
/// A string containing formatted summary statistics
#[allow(non_snake_case)]
pub fn generate_missing_summary_stats(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
) -> SklResult<String> {
    let X = X.mapv(|x| x);
    let (n_samples, n_features) = X.dim();

    let mut total_missing = 0;
    let mut feature_missing_counts = vec![0; n_features];
    let mut sample_missing_counts = vec![0; n_samples];

    for i in 0..n_samples {
        for j in 0..n_features {
            let is_missing = if missing_values.is_nan() {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };

            if is_missing {
                total_missing += 1;
                feature_missing_counts[j] += 1;
                sample_missing_counts[i] += 1;
            }
        }
    }

    let total_cells = n_samples * n_features;
    let overall_missing_rate = (total_missing as f64 / total_cells as f64) * 100.0;

    let features_with_missing = feature_missing_counts
        .iter()
        .filter(|&&count| count > 0)
        .count();
    let samples_with_missing = sample_missing_counts
        .iter()
        .filter(|&&count| count > 0)
        .count();

    let max_feature_missing = *feature_missing_counts.iter().max().unwrap_or(&0);
    let max_sample_missing = *sample_missing_counts.iter().max().unwrap_or(&0);

    let completely_missing_features = feature_missing_counts
        .iter()
        .filter(|&&count| count == n_samples)
        .count();
    let completely_observed_samples = sample_missing_counts
        .iter()
        .filter(|&&count| count == 0)
        .count();

    let summary = format!(
        "Missing Data Summary Statistics\n\
         ===============================\n\
         Dataset dimensions: {} samples × {} features\n\
         Total cells: {}\n\
         Missing cells: {} ({:.2}%)\n\
         Observed cells: {} ({:.2}%)\n\
         \n\
         Feature-wise statistics:\n\
         - Features with missing values: {} / {} ({:.1}%)\n\
         - Completely missing features: {}\n\
         - Maximum missing values in a feature: {} / {} ({:.1}%)\n\
         \n\
         Sample-wise statistics:\n\
         - Samples with missing values: {} / {} ({:.1}%)\n\
         - Completely observed samples: {} ({:.1}%)\n\
         - Maximum missing values in a sample: {} / {} ({:.1}%)\n",
        n_samples,
        n_features,
        total_cells,
        total_missing,
        overall_missing_rate,
        total_cells - total_missing,
        100.0 - overall_missing_rate,
        features_with_missing,
        n_features,
        (features_with_missing as f64 / n_features as f64) * 100.0,
        completely_missing_features,
        max_feature_missing,
        n_samples,
        (max_feature_missing as f64 / n_samples as f64) * 100.0,
        samples_with_missing,
        n_samples,
        (samples_with_missing as f64 / n_samples as f64) * 100.0,
        completely_observed_samples,
        (completely_observed_samples as f64 / n_samples as f64) * 100.0,
        max_sample_missing,
        n_features,
        (max_sample_missing as f64 / n_features as f64) * 100.0
    );

    Ok(summary)
}

// Helper functions

fn compute_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut numerator = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denominator = (var_x * var_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn compute_correlation_p_value(x: &Array1<f64>, _y: &Array1<f64>, correlation: f64) -> f64 {
    let n = x.len() as f64;
    if n <= 2.0 {
        return 1.0;
    }

    // Approximate p-value using t-distribution
    // t = r * sqrt((n-2)/(1-r^2))
    let r_squared = correlation * correlation;
    if r_squared >= 1.0 {
        return 0.0;
    }

    let t_stat = correlation * ((n - 2.0) / (1.0 - r_squared)).sqrt();

    // Simplified p-value approximation (two-tailed test)
    // For a more accurate calculation, you would use the t-distribution CDF
    let _df = n - 2.0;

    // Very crude approximation - in practice, you'd want to use a proper statistical library
    if t_stat.abs() > 2.0 {
        0.05
    } else if t_stat.abs() > 1.0 {
        0.1
    } else {
        0.5
    }
}
