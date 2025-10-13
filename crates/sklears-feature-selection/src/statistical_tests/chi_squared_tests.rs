//! Chi-squared tests for feature selection
//!
//! This module implements chi-squared statistical tests for categorical feature selection,
//! following SciRS2 Policy for numerical operations.

use crate::statistical_tests::distributions::chi2_cdf;
use scirs2_core::error::CoreError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
type CoreResult<T> = std::result::Result<T, CoreError>;

/// Chi-squared test for independence between categorical features and target
///
/// # Arguments
/// * `X` - Feature matrix where each column is a categorical feature
/// * `y` - Target vector with categorical labels
///
/// # Returns
/// * Array of chi-squared statistics for each feature
/// * Array of p-values for each feature
pub fn chi2(X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> CoreResult<(Array1<f64>, Array1<f64>)> {
    let n_features = X.ncols();
    let mut chi2_stats = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = X.column(i);
        let (chi2_stat, p_val) = chi2_single_feature(&feature, y)?;
        chi2_stats[i] = chi2_stat;
        p_values[i] = p_val;
    }

    Ok((chi2_stats, p_values))
}

/// Chi-squared test for a single feature
fn chi2_single_feature(feature: &ArrayView1<f64>, y: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    // Create contingency table
    let contingency = create_contingency_table(feature, y)?;
    chi2_contingency(&contingency.view())
}

/// Chi-squared test on contingency table
pub fn chi2_contingency(contingency: &ArrayView2<f64>) -> CoreResult<(f64, f64)> {
    let (n_rows, n_cols) = contingency.dim();

    // Calculate row and column totals
    let row_totals = contingency.sum_axis(scirs2_core::ndarray::Axis(1));
    let col_totals = contingency.sum_axis(scirs2_core::ndarray::Axis(0));
    let grand_total = row_totals.sum();

    if grand_total == 0.0 {
        return Ok((0.0, 1.0));
    }

    // Calculate expected frequencies and chi-squared statistic
    let mut chi2_stat = 0.0;

    for i in 0..n_rows {
        for j in 0..n_cols {
            let expected = row_totals[i] * col_totals[j] / grand_total;

            if expected > 0.0 {
                let observed = contingency[[i, j]];
                chi2_stat += (observed - expected).powi(2) / expected;
            }
        }
    }

    // Degrees of freedom
    let df = ((n_rows - 1) * (n_cols - 1)) as f64;

    // Calculate p-value
    let p_value = 1.0 - chi2_cdf(chi2_stat, df);

    Ok((chi2_stat, p_value))
}

/// Create contingency table from feature and target vectors
fn create_contingency_table(
    feature: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> CoreResult<Array2<f64>> {
    // Find unique values in feature and target
    let mut feature_values: Vec<f64> = feature.to_vec();
    feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    feature_values.dedup();

    let mut target_values: Vec<f64> = y.to_vec();
    target_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    target_values.dedup();

    let n_feature_vals = feature_values.len();
    let n_target_vals = target_values.len();

    let mut contingency = Array2::zeros((n_feature_vals, n_target_vals));

    // Fill contingency table
    for (feat_val, target_val) in feature.iter().zip(y.iter()) {
        if let (Some(i), Some(j)) = (
            feature_values
                .iter()
                .position(|&x| (x - feat_val).abs() < 1e-10),
            target_values
                .iter()
                .position(|&x| (x - target_val).abs() < 1e-10),
        ) {
            contingency[[i, j]] += 1.0;
        }
    }

    Ok(contingency)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_chi2_contingency() {
        // Simple 2x2 contingency table
        let contingency = array![[10.0, 5.0], [3.0, 12.0]];
        let (chi2_stat, p_value) = chi2_contingency(&contingency.view()).unwrap();

        assert!(chi2_stat > 0.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_chi2_feature_selection() {
        // Create sample data
        let X = array![[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]];
        let y = array![1.0, 1.0, 0.0, 0.0];

        let (chi2_stats, p_values) = chi2(&X.view(), &y.view()).unwrap();

        assert_eq!(chi2_stats.len(), 2);
        assert_eq!(p_values.len(), 2);
        assert!(chi2_stats.iter().all(|&x| x >= 0.0));
        assert!(p_values.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}
