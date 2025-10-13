//! Label combination frequency analysis utilities

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Information about a label combination
#[derive(Debug, Clone)]
pub struct CombinationInfo {
    /// The label combination
    pub combination: Vec<i32>,
    /// Number of times this combination appears
    pub frequency: usize,
    /// Relative frequency (proportion of total samples)
    pub relative_frequency: f64,
    /// Number of active labels in this combination
    pub cardinality: usize,
}

/// Results of label combination frequency analysis
#[derive(Debug, Clone)]
pub struct LabelAnalysisResults {
    /// All unique label combinations found
    pub combinations: Vec<CombinationInfo>,
    /// Total number of samples analyzed
    pub total_samples: usize,
    /// Total number of unique combinations
    pub unique_combinations: usize,
    /// Most frequent combination
    pub most_frequent: Option<CombinationInfo>,
    /// Least frequent combination (with frequency > 0)
    pub least_frequent: Option<CombinationInfo>,
    /// Average label cardinality (number of active labels per sample)
    pub average_cardinality: f64,
    /// Label cardinality distribution
    pub cardinality_distribution: HashMap<usize, usize>,
}

/// Analyze label combinations in multi-label data
///
/// This function analyzes the frequency of different label combinations
/// in a multi-label dataset, providing insights into the data distribution.
///
/// # Arguments
///
/// * `y` - Multi-label binary matrix where each row is a sample and each column is a label
///
/// # Returns
///
/// A `LabelAnalysisResults` struct containing various statistics about label combinations
///
/// # Examples
///
/// ```
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use sklears_multioutput::label_analysis::analyze_combinations;
///
/// let y = array![[1, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1]];
/// let results = analyze_combinations(&y.view()).unwrap();
/// println!("Total unique combinations: {}", results.unique_combinations);
/// ```
pub fn analyze_combinations(y: &ArrayView2<'_, i32>) -> SklResult<LabelAnalysisResults> {
    let (n_samples, n_labels) = y.dim();

    if n_samples == 0 || n_labels == 0 {
        return Err(SklearsError::InvalidInput(
            "Input array must have at least one sample and one label".to_string(),
        ));
    }

    // Validate that all labels are binary (0 or 1)
    for sample_idx in 0..n_samples {
        for label_idx in 0..n_labels {
            let value = y[[sample_idx, label_idx]];
            if value != 0 && value != 1 {
                return Err(SklearsError::InvalidInput(format!(
                    "All label values must be 0 or 1, found: {}",
                    value
                )));
            }
        }
    }

    let mut combination_counts: HashMap<Vec<i32>, usize> = HashMap::new();
    let mut cardinality_distribution: HashMap<usize, usize> = HashMap::new();
    let mut total_cardinality = 0;

    // Analyze each sample
    for sample_idx in 0..n_samples {
        let mut combination = Vec::new();
        let mut cardinality = 0;

        for label_idx in 0..n_labels {
            let label_value = y[[sample_idx, label_idx]];
            combination.push(label_value);
            if label_value == 1 {
                cardinality += 1;
            }
        }

        // Update combination counts
        *combination_counts.entry(combination).or_insert(0) += 1;

        // Update cardinality distribution
        *cardinality_distribution.entry(cardinality).or_insert(0) += 1;
        total_cardinality += cardinality;
    }

    // Convert to CombinationInfo structs
    let mut combinations: Vec<CombinationInfo> = combination_counts
        .into_iter()
        .map(|(combination, frequency)| {
            let cardinality = combination.iter().sum::<i32>() as usize;
            CombinationInfo {
                combination,
                frequency,
                relative_frequency: frequency as f64 / n_samples as f64,
                cardinality,
            }
        })
        .collect();

    // Sort by frequency (descending)
    combinations.sort_by(|a, b| b.frequency.cmp(&a.frequency));

    let most_frequent = combinations.first().cloned();
    let least_frequent = combinations.last().cloned();
    let unique_combinations = combinations.len();
    let average_cardinality = total_cardinality as f64 / n_samples as f64;

    Ok(LabelAnalysisResults {
        combinations,
        total_samples: n_samples,
        unique_combinations,
        most_frequent,
        least_frequent,
        average_cardinality,
        cardinality_distribution,
    })
}

/// Compute label co-occurrence matrix
///
/// This function computes a matrix showing how often each pair of labels
/// occurs together in the dataset.
///
/// # Arguments
///
/// * `y` - Multi-label binary matrix
///
/// # Returns
///
/// A symmetric matrix where element (i,j) contains the number of samples
/// where both label i and label j are active
pub fn label_cooccurrence_matrix(y: &ArrayView2<'_, i32>) -> SklResult<Array2<usize>> {
    let (n_samples, n_labels) = y.dim();

    if n_samples == 0 || n_labels == 0 {
        return Err(SklearsError::InvalidInput(
            "Input array must have at least one sample and one label".to_string(),
        ));
    }

    let mut cooccurrence = Array2::<usize>::zeros((n_labels, n_labels));

    for sample_idx in 0..n_samples {
        for i in 0..n_labels {
            for j in 0..n_labels {
                if y[[sample_idx, i]] == 1 && y[[sample_idx, j]] == 1 {
                    cooccurrence[[i, j]] += 1;
                }
            }
        }
    }

    Ok(cooccurrence)
}

/// Compute label correlation matrix
///
/// This function computes the Pearson correlation coefficient between
/// each pair of labels.
///
/// # Arguments
///
/// * `y` - Multi-label binary matrix
///
/// # Returns
///
/// A symmetric correlation matrix where element (i,j) contains the
/// correlation coefficient between label i and label j
pub fn label_correlation_matrix(y: &ArrayView2<'_, i32>) -> SklResult<Array2<f64>> {
    let (n_samples, n_labels) = y.dim();

    if n_samples == 0 || n_labels == 0 {
        return Err(SklearsError::InvalidInput(
            "Input array must have at least one sample and one label".to_string(),
        ));
    }

    let mut correlation = Array2::<Float>::zeros((n_labels, n_labels));

    // Compute means for each label
    let mut means = vec![0.0; n_labels];
    for j in 0..n_labels {
        let mut sum = 0.0;
        for i in 0..n_samples {
            sum += y[[i, j]] as f64;
        }
        means[j] = sum / n_samples as f64;
    }

    // Compute correlations
    for i in 0..n_labels {
        for j in 0..n_labels {
            if i == j {
                correlation[[i, j]] = 1.0;
            } else {
                let mut numerator = 0.0;
                let mut sum_sq_i = 0.0;
                let mut sum_sq_j = 0.0;

                for sample_idx in 0..n_samples {
                    let val_i = y[[sample_idx, i]] as f64 - means[i];
                    let val_j = y[[sample_idx, j]] as f64 - means[j];

                    numerator += val_i * val_j;
                    sum_sq_i += val_i * val_i;
                    sum_sq_j += val_j * val_j;
                }

                let denominator = (sum_sq_i * sum_sq_j).sqrt();
                if denominator > 1e-10 {
                    correlation[[i, j]] = numerator / denominator;
                } else {
                    correlation[[i, j]] = 0.0;
                }
            }
        }
    }

    Ok(correlation)
}

/// Get rare combinations with frequency <= threshold
pub fn get_rare_combinations(
    results: &LabelAnalysisResults,
    threshold: usize,
) -> Vec<CombinationInfo> {
    results
        .combinations
        .iter()
        .filter(|combo| combo.frequency <= threshold)
        .cloned()
        .collect()
}

/// Get combinations by cardinality (number of active labels)
pub fn get_combinations_by_cardinality(
    results: &LabelAnalysisResults,
    cardinality: usize,
) -> Vec<CombinationInfo> {
    results
        .combinations
        .iter()
        .filter(|combo| combo.cardinality == cardinality)
        .cloned()
        .collect()
}

/// Find singleton labels (labels that appear alone)
///
/// This function identifies labels that frequently appear as the only
/// active label in a sample.
///
/// # Arguments
///
/// * `y` - Multi-label binary matrix
///
/// # Returns
///
/// A vector of tuples containing (label_index, singleton_count, singleton_percentage)
pub fn find_singleton_labels(y: &ArrayView2<'_, i32>) -> SklResult<Vec<(usize, usize, f64)>> {
    let (n_samples, n_labels) = y.dim();

    if n_samples == 0 || n_labels == 0 {
        return Err(SklearsError::InvalidInput(
            "Input array must have at least one sample and one label".to_string(),
        ));
    }

    let mut singleton_counts = vec![0; n_labels];

    for sample_idx in 0..n_samples {
        let active_labels: Vec<usize> = (0..n_labels)
            .filter(|&label_idx| y[[sample_idx, label_idx]] == 1)
            .collect();

        // If exactly one label is active, it's a singleton
        if active_labels.len() == 1 {
            singleton_counts[active_labels[0]] += 1;
        }
    }

    let results = singleton_counts
        .into_iter()
        .enumerate()
        .map(|(label_idx, count)| {
            let percentage = count as f64 / n_samples as f64 * 100.0;
            (label_idx, count, percentage)
        })
        .collect();

    Ok(results)
}

/// Compute label frequency distribution
///
/// This function computes how often each label appears in the dataset.
///
/// # Arguments
///
/// * `y` - Multi-label binary matrix
///
/// # Returns
///
/// A vector of tuples containing (label_index, frequency, percentage)
pub fn label_frequency_distribution(
    y: &ArrayView2<'_, i32>,
) -> SklResult<Vec<(usize, usize, f64)>> {
    let (n_samples, n_labels) = y.dim();

    if n_samples == 0 || n_labels == 0 {
        return Err(SklearsError::InvalidInput(
            "Input array must have at least one sample and one label".to_string(),
        ));
    }

    let mut frequencies = vec![0; n_labels];

    for sample_idx in 0..n_samples {
        for label_idx in 0..n_labels {
            if y[[sample_idx, label_idx]] == 1 {
                frequencies[label_idx] += 1;
            }
        }
    }

    let results = frequencies
        .into_iter()
        .enumerate()
        .map(|(label_idx, frequency)| {
            let percentage = frequency as f64 / n_samples as f64 * 100.0;
            (label_idx, frequency, percentage)
        })
        .collect();

    Ok(results)
}
