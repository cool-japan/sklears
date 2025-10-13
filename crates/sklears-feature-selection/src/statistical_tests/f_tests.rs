//! F-tests for feature selection
//!
//! This module implements F-statistic based tests for feature selection,
//! following SciRS2 Policy for numerical operations.

use crate::statistical_tests::distributions::f_cdf;
use scirs2_core::error::CoreError;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
type CoreResult<T> = std::result::Result<T, CoreError>;

/// F-test for classification (ANOVA F-test)
pub fn f_classif(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
) -> CoreResult<(Array1<f64>, Array1<f64>)> {
    let n_features = X.ncols();
    let mut f_stats = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = X.column(i);
        let (f_stat, p_val) = f_classif_single_feature(&feature, y)?;
        f_stats[i] = f_stat;
        p_values[i] = p_val;
    }

    Ok((f_stats, p_values))
}

/// F-test for regression
pub fn f_regression(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
) -> CoreResult<(Array1<f64>, Array1<f64>)> {
    let n_features = X.ncols();
    let mut f_stats = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = X.column(i);
        let (f_stat, p_val) = f_regression_single_feature(&feature, y)?;
        f_stats[i] = f_stat;
        p_values[i] = p_val;
    }

    Ok((f_stats, p_values))
}

/// One-way ANOVA F-test
pub fn f_oneway(samples: &[ArrayView1<f64>]) -> CoreResult<(f64, f64)> {
    if samples.is_empty() {
        return Ok((0.0, 1.0));
    }

    let mut all_values = Vec::new();
    let mut group_means = Vec::new();
    let mut group_sizes = Vec::new();

    // Collect data and compute group statistics
    for sample in samples {
        let values: Vec<f64> = sample.to_vec();
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        all_values.extend(values);
        group_means.push(mean);
        group_sizes.push(sample.len());
    }

    let n = all_values.len();
    let k = samples.len();

    if n == 0 || k <= 1 {
        return Ok((0.0, 1.0));
    }

    let grand_mean = all_values.iter().sum::<f64>() / n as f64;

    // Between-group sum of squares
    let mut ss_between = 0.0;
    for (i, &mean) in group_means.iter().enumerate() {
        ss_between += group_sizes[i] as f64 * (mean - grand_mean).powi(2);
    }

    // Within-group sum of squares
    let mut ss_within = 0.0;
    let mut idx = 0;
    for (i, sample) in samples.iter().enumerate() {
        let group_mean = group_means[i];
        for &value in sample {
            ss_within += (value - group_mean).powi(2);
        }
        idx += sample.len();
    }

    // Degrees of freedom
    let df_between = (k - 1) as f64;
    let df_within = (n - k) as f64;

    if df_within <= 0.0 || ss_within == 0.0 {
        return Ok((f64::INFINITY, 0.0));
    }

    // F-statistic
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f_stat = ms_between / ms_within;

    // P-value
    let p_value = 1.0 - f_cdf(f_stat, df_between, df_within);

    Ok((f_stat, p_value))
}

/// F-test for single feature classification
fn f_classif_single_feature(
    feature: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> CoreResult<(f64, f64)> {
    // Group feature values by class
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();

    let mut groups = Vec::new();
    for &class in &classes {
        let mut group = Vec::new();
        for (i, &label) in y.iter().enumerate() {
            if (label - class).abs() < 1e-10 {
                group.push(feature[i]);
            }
        }
        if !group.is_empty() {
            groups.push(Array1::from(group));
        }
    }

    let group_views: Vec<ArrayView1<f64>> = groups.iter().map(|g| g.view()).collect();
    f_oneway(&group_views)
}

/// F-test for single feature regression
fn f_regression_single_feature(
    feature: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> CoreResult<(f64, f64)> {
    let n = feature.len() as f64;

    if n < 3.0 {
        return Ok((0.0, 1.0));
    }

    // Calculate correlation coefficient
    let x_mean = feature.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..feature.len() {
        let x_dev = feature[i] - x_mean;
        let y_dev = y[i] - y_mean;
        numerator += x_dev * y_dev;
        x_var += x_dev * x_dev;
        y_var += y_dev * y_dev;
    }

    if x_var == 0.0 || y_var == 0.0 {
        return Ok((0.0, 1.0));
    }

    let r = numerator / (x_var * y_var).sqrt();
    let r_squared = r * r;

    // F-statistic for regression
    let df1 = 1.0;
    let df2 = n - 2.0;

    if df2 <= 0.0 {
        return Ok((0.0, 1.0));
    }

    let f_stat = (r_squared / (1.0 - r_squared)) * df2 / df1;

    // P-value
    let p_value = 1.0 - f_cdf(f_stat, df1, df2);

    Ok((f_stat, p_value))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_f_oneway() {
        let group1 = array![1.0, 2.0, 3.0];
        let group2 = array![4.0, 5.0, 6.0];
        let group3 = array![7.0, 8.0, 9.0];

        let (f_stat, p_value) = f_oneway(&[group1.view(), group2.view(), group3.view()]).unwrap();

        assert!(f_stat > 0.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_f_classif() {
        let X = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [10.0, 40.0]];
        let y = array![0.0, 0.0, 0.0, 1.0];

        let (f_stats, p_values) = f_classif(&X.view(), &y.view()).unwrap();

        assert_eq!(f_stats.len(), 2);
        assert_eq!(p_values.len(), 2);
        assert!(f_stats.iter().all(|&x| x >= 0.0));
        assert!(p_values.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}
