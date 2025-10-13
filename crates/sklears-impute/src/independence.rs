//! Independence tests for missing data mechanisms
//!
//! This module provides statistical tests to assess independence between
//! missing data patterns and observed data values, helping to determine
//! the missing data mechanism (MCAR, MAR, MNAR).

use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{error::Result as SklResult, error::SklearsError, types::Float};
use std::collections::HashMap;

/// Result of a chi-square independence test
#[derive(Debug, Clone)]
pub struct ChiSquareTestResult {
    /// Chi-square test statistic
    pub chi_square_statistic: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    /// P-value of the test
    pub p_value: f64,
    /// Critical value at alpha = 0.05
    pub critical_value: f64,
    /// Whether to reject the null hypothesis (independence)
    pub reject_independence: bool,
    /// Expected frequencies under independence assumption
    pub expected_frequencies: Array2<f64>,
    /// Observed frequencies (contingency table)
    pub observed_frequencies: Array2<f64>,
}

/// Result of Fisher's exact test
#[derive(Debug, Clone)]
pub struct FisherExactTestResult {
    /// P-value of the test (two-tailed)
    pub p_value: f64,
    /// P-value of the test (one-tailed, less)
    pub p_value_less: f64,
    /// P-value of the test (one-tailed, greater)
    pub p_value_greater: f64,
    /// Odds ratio
    pub odds_ratio: f64,
    /// 95% confidence interval for odds ratio
    pub confidence_interval: (f64, f64),
    /// Whether to reject the null hypothesis (independence) at alpha = 0.05
    pub reject_independence: bool,
}

/// Result of Cramér's V association test
#[derive(Debug, Clone)]
pub struct CramersVTestResult {
    /// Cramér's V statistic (0 = no association, 1 = perfect association)
    pub cramers_v: f64,
    /// Chi-square statistic used in calculation
    pub chi_square_statistic: f64,
    /// Sample size
    pub n: usize,
    /// Minimum dimension of the contingency table
    pub min_dimension: usize,
    /// Strength of association interpretation
    pub association_strength: String,
}

/// Result of the Kolmogorov-Smirnov test for distribution differences
#[derive(Debug, Clone)]
pub struct KolmogorovSmirnovTestResult {
    /// KS test statistic (maximum difference between CDFs)
    pub ks_statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Critical value at alpha = 0.05
    pub critical_value: f64,
    /// Whether to reject the null hypothesis (same distribution)
    pub reject_same_distribution: bool,
    /// Sample sizes for both groups
    pub sample_sizes: (usize, usize),
}

/// Comprehensive independence test results
#[derive(Debug, Clone)]
pub struct IndependenceTestSuite {
    /// Results for each feature tested
    pub feature_results: Vec<FeatureIndependenceResult>,
    /// Overall summary statistics
    pub summary: IndependenceTestSummary,
}

/// Independence test results for a single feature
#[derive(Debug, Clone)]
pub struct FeatureIndependenceResult {
    /// Feature index
    pub feature_index: usize,
    /// Feature name (if provided)
    pub feature_name: Option<String>,
    /// Chi-square test result (if applicable)
    pub chi_square_test: Option<ChiSquareTestResult>,
    /// Fisher's exact test result (if applicable)
    pub fisher_exact_test: Option<FisherExactTestResult>,
    /// Cramér's V test result
    pub cramers_v_test: Option<CramersVTestResult>,
    /// KS test result (for continuous variables)
    pub ks_test: Option<KolmogorovSmirnovTestResult>,
    /// Test recommendation based on data characteristics
    pub test_recommendation: String,
}

/// Summary of independence test suite
#[derive(Debug, Clone)]
pub struct IndependenceTestSummary {
    /// Number of features tested
    pub features_tested: usize,
    /// Number of features showing significant dependence
    pub features_with_dependence: usize,
    /// Proportion of features showing dependence
    pub dependence_rate: f64,
    /// Overall missing data mechanism assessment
    pub mechanism_assessment: String,
    /// Recommended follow-up actions
    pub recommendations: Vec<String>,
}

/// Test independence between missingness patterns and observed values using chi-square test
///
/// This test is appropriate for categorical variables or discretized continuous variables.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `feature_idx` - Index of the feature to test
/// * `other_feature_idx` - Index of the other feature to test against
/// * `missing_values` - Value representing missing data
/// * `bins` - Number of bins for discretizing continuous variables (if needed)
///
/// # Returns
///
/// Chi-square test result
#[allow(non_snake_case)]
pub fn chi_square_independence_test(
    X: &ArrayView2<'_, Float>,
    feature_idx: usize,
    other_feature_idx: usize,
    missing_values: f64,
    bins: Option<usize>,
) -> SklResult<ChiSquareTestResult> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    if feature_idx >= n_features || other_feature_idx >= n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Feature indices {} and {} must be less than number of features {}",
            feature_idx, other_feature_idx, n_features
        )));
    }

    if feature_idx == other_feature_idx {
        return Err(SklearsError::InvalidInput(
            "Feature indices must be different".to_string(),
        ));
    }

    // Create missingness indicator for the target feature
    let mut missing_indicator = Vec::new();
    let mut other_values = Vec::new();

    for i in 0..n_samples {
        let is_missing = if missing_values.is_nan() {
            X[[i, feature_idx]].is_nan()
        } else {
            (X[[i, feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        // Only include samples where the other feature is observed
        let other_is_missing = if missing_values.is_nan() {
            X[[i, other_feature_idx]].is_nan()
        } else {
            (X[[i, other_feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        if !other_is_missing {
            missing_indicator.push(if is_missing { 1 } else { 0 });
            other_values.push(X[[i, other_feature_idx]]);
        }
    }

    if missing_indicator.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid observations for comparison".to_string(),
        ));
    }

    // Discretize the other feature if needed
    let n_bins = bins.unwrap_or(5);
    let discretized_values = discretize_values(&other_values, n_bins)?;

    // Create contingency table
    let contingency_table = create_contingency_table(&missing_indicator, &discretized_values)?;

    // Perform chi-square test
    let chi_square_result = compute_chi_square_test(&contingency_table)?;

    Ok(chi_square_result)
}

/// Test independence using Fisher's exact test
///
/// This test is appropriate for 2x2 contingency tables with small sample sizes.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `feature_idx` - Index of the feature to test
/// * `other_feature_idx` - Index of the other feature to test against
/// * `missing_values` - Value representing missing data
/// * `threshold` - Threshold for binarizing the other feature
///
/// # Returns
///
/// Fisher's exact test result
#[allow(non_snake_case)]
pub fn fisher_exact_independence_test(
    X: &ArrayView2<'_, Float>,
    feature_idx: usize,
    other_feature_idx: usize,
    missing_values: f64,
    threshold: Option<f64>,
) -> SklResult<FisherExactTestResult> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    if feature_idx >= n_features || other_feature_idx >= n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Feature indices {} and {} must be less than number of features {}",
            feature_idx, other_feature_idx, n_features
        )));
    }

    // Collect valid observations
    let mut missing_indicator = Vec::new();
    let mut other_values = Vec::new();

    for i in 0..n_samples {
        let is_missing = if missing_values.is_nan() {
            X[[i, feature_idx]].is_nan()
        } else {
            (X[[i, feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        let other_is_missing = if missing_values.is_nan() {
            X[[i, other_feature_idx]].is_nan()
        } else {
            (X[[i, other_feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        if !other_is_missing {
            missing_indicator.push(if is_missing { 1 } else { 0 });
            other_values.push(X[[i, other_feature_idx]]);
        }
    }

    if missing_indicator.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid observations for comparison".to_string(),
        ));
    }

    // Binarize the other feature
    let threshold =
        threshold.unwrap_or_else(|| other_values.iter().sum::<f64>() / other_values.len() as f64);

    let binary_values: Vec<usize> = other_values
        .iter()
        .map(|&x| if x > threshold { 1 } else { 0 })
        .collect();

    // Create 2x2 contingency table
    let mut table = [[0; 2]; 2];
    for (missing, binary) in missing_indicator.iter().zip(binary_values.iter()) {
        table[*missing][*binary] += 1;
    }

    // Perform Fisher's exact test
    let fisher_result = compute_fisher_exact_test(&table)?;

    Ok(fisher_result)
}

/// Compute Cramér's V association measure
///
/// Cramér's V measures the association between two categorical variables,
/// ranging from 0 (no association) to 1 (perfect association).
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `feature_idx` - Index of the feature to test
/// * `other_feature_idx` - Index of the other feature to test against
/// * `missing_values` - Value representing missing data
/// * `bins` - Number of bins for discretizing continuous variables
///
/// # Returns
///
/// Cramér's V test result
#[allow(non_snake_case)]
pub fn cramers_v_association_test(
    X: &ArrayView2<'_, Float>,
    feature_idx: usize,
    other_feature_idx: usize,
    missing_values: f64,
    bins: Option<usize>,
) -> SklResult<CramersVTestResult> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    if feature_idx >= n_features || other_feature_idx >= n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Feature indices {} and {} must be less than number of features {}",
            feature_idx, other_feature_idx, n_features
        )));
    }

    // Collect valid observations
    let mut missing_indicator = Vec::new();
    let mut other_values = Vec::new();

    for i in 0..n_samples {
        let is_missing = if missing_values.is_nan() {
            X[[i, feature_idx]].is_nan()
        } else {
            (X[[i, feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        let other_is_missing = if missing_values.is_nan() {
            X[[i, other_feature_idx]].is_nan()
        } else {
            (X[[i, other_feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        if !other_is_missing {
            missing_indicator.push(if is_missing { 1 } else { 0 });
            other_values.push(X[[i, other_feature_idx]]);
        }
    }

    if missing_indicator.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid observations for comparison".to_string(),
        ));
    }

    let n = missing_indicator.len();

    // Discretize the other feature
    let n_bins = bins.unwrap_or(5);
    let discretized_values = discretize_values(&other_values, n_bins)?;

    // Create contingency table
    let contingency_table = create_contingency_table(&missing_indicator, &discretized_values)?;

    // Compute chi-square statistic
    let chi_square_statistic = compute_chi_square_statistic(&contingency_table)?;

    // Compute Cramér's V
    let min_dimension = (contingency_table.nrows() - 1).min(contingency_table.ncols() - 1);
    let cramers_v = if min_dimension > 0 {
        (chi_square_statistic / (n as f64 * min_dimension as f64)).sqrt()
    } else {
        0.0
    };

    // Interpret association strength
    let association_strength = match cramers_v {
        v if v < 0.1 => "Negligible association".to_string(),
        v if v < 0.3 => "Weak association".to_string(),
        v if v < 0.5 => "Moderate association".to_string(),
        v if v < 0.7 => "Strong association".to_string(),
        _ => "Very strong association".to_string(),
    };

    Ok(CramersVTestResult {
        cramers_v,
        chi_square_statistic,
        n,
        min_dimension,
        association_strength,
    })
}

/// Perform Kolmogorov-Smirnov test for distribution differences
///
/// Tests whether the distribution of a continuous variable differs between
/// missing and non-missing groups of another variable.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `feature_idx` - Index of the feature to test (target with missing values)
/// * `other_feature_idx` - Index of the continuous feature to compare distributions
/// * `missing_values` - Value representing missing data
///
/// # Returns
///
/// Kolmogorov-Smirnov test result
#[allow(non_snake_case)]
pub fn kolmogorov_smirnov_independence_test(
    X: &ArrayView2<'_, Float>,
    feature_idx: usize,
    other_feature_idx: usize,
    missing_values: f64,
) -> SklResult<KolmogorovSmirnovTestResult> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    if feature_idx >= n_features || other_feature_idx >= n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Feature indices {} and {} must be less than number of features {}",
            feature_idx, other_feature_idx, n_features
        )));
    }

    // Separate samples based on missingness in target feature
    let mut missing_group = Vec::new();
    let mut observed_group = Vec::new();

    for i in 0..n_samples {
        let is_missing = if missing_values.is_nan() {
            X[[i, feature_idx]].is_nan()
        } else {
            (X[[i, feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        let other_is_missing = if missing_values.is_nan() {
            X[[i, other_feature_idx]].is_nan()
        } else {
            (X[[i, other_feature_idx]] - missing_values).abs() < f64::EPSILON
        };

        if !other_is_missing {
            if is_missing {
                missing_group.push(X[[i, other_feature_idx]]);
            } else {
                observed_group.push(X[[i, other_feature_idx]]);
            }
        }
    }

    if missing_group.is_empty() || observed_group.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Need observations in both missing and observed groups".to_string(),
        ));
    }

    // Sort both groups
    missing_group.sort_by(|a, b| a.partial_cmp(b).unwrap());
    observed_group.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute KS statistic
    let ks_statistic = compute_ks_statistic(&missing_group, &observed_group);

    // Compute critical value and p-value (approximate)
    let n1 = missing_group.len();
    let n2 = observed_group.len();
    let critical_value = 1.36 * ((n1 + n2) as f64 / (n1 * n2) as f64).sqrt();

    // Approximate p-value using Kolmogorov distribution
    let p_value = compute_ks_p_value(ks_statistic, n1, n2);

    let reject_same_distribution = ks_statistic > critical_value;

    Ok(KolmogorovSmirnovTestResult {
        ks_statistic,
        p_value,
        critical_value,
        reject_same_distribution,
        sample_sizes: (n1, n2),
    })
}

/// Run comprehensive independence test suite
///
/// Performs multiple independence tests for each feature with missing values
/// against all other features to assess missing data mechanisms.
///
/// # Parameters
///
/// * `X` - Input data matrix
/// * `missing_values` - Value representing missing data
/// * `feature_names` - Optional feature names
/// * `alpha` - Significance level for tests (default: 0.05)
///
/// # Returns
///
/// Comprehensive independence test results
#[allow(non_snake_case)]
pub fn run_independence_test_suite(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
    feature_names: Option<Vec<String>>,
    alpha: Option<f64>,
) -> SklResult<IndependenceTestSuite> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();
    let alpha = alpha.unwrap_or(0.05);

    let feature_names = feature_names
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    if feature_names.len() != n_features {
        return Err(SklearsError::InvalidInput(format!(
            "Number of feature names {} does not match number of features {}",
            feature_names.len(),
            n_features
        )));
    }

    let mut feature_results = Vec::new();
    let mut features_with_dependence = 0;

    // Find features with missing values
    let mut features_with_missing = Vec::new();
    for j in 0..n_features {
        let column = X.column(j);
        let has_missing = column.iter().any(|&x| {
            if missing_values.is_nan() {
                x.is_nan()
            } else {
                (x - missing_values).abs() < f64::EPSILON
            }
        });

        if has_missing {
            features_with_missing.push(j);
        }
    }

    // Test each feature with missing values against all others
    for &feature_idx in &features_with_missing {
        let mut has_significant_dependence = false;
        let mut chi_square_test = None;
        let mut fisher_exact_test = None;
        let mut cramers_v_test = None;
        let mut ks_test = None;

        // Test against other features
        for other_feature_idx in 0..n_features {
            if feature_idx == other_feature_idx {
                continue;
            }

            // Determine appropriate test based on data characteristics
            let other_column = X.column(other_feature_idx);
            let other_unique_values: std::collections::HashSet<_> = other_column
                .iter()
                .filter(|&&x| {
                    if missing_values.is_nan() {
                        !x.is_nan()
                    } else {
                        (x - missing_values).abs() >= f64::EPSILON
                    }
                })
                .map(|&x| x.to_bits())
                .collect();

            let is_likely_categorical = other_unique_values.len() <= 10;

            // Count valid observations for this pair
            let mut valid_pairs = 0;
            for i in 0..n_samples {
                let other_is_missing = if missing_values.is_nan() {
                    X[[i, other_feature_idx]].is_nan()
                } else {
                    (X[[i, other_feature_idx]] - missing_values).abs() < f64::EPSILON
                };

                if !other_is_missing {
                    valid_pairs += 1;
                }
            }

            if valid_pairs < 10 {
                continue; // Skip if too few observations
            }

            // Choose and perform appropriate test
            if is_likely_categorical && other_unique_values.len() == 2 && valid_pairs < 50 {
                // Use Fisher's exact test for small 2x2 tables
                if let Ok(result) = fisher_exact_independence_test(
                    &X.view(),
                    feature_idx,
                    other_feature_idx,
                    missing_values,
                    None,
                ) {
                    if result.p_value < alpha {
                        has_significant_dependence = true;
                    }
                    fisher_exact_test = Some(result);
                    break; // Found a significant result
                }
            } else if is_likely_categorical {
                // Use chi-square test for categorical data
                if let Ok(result) = chi_square_independence_test(
                    &X.view(),
                    feature_idx,
                    other_feature_idx,
                    missing_values,
                    None,
                ) {
                    if result.p_value < alpha {
                        has_significant_dependence = true;
                    }
                    chi_square_test = Some(result);

                    // Also compute Cramér's V
                    if let Ok(cv_result) = cramers_v_association_test(
                        &X.view(),
                        feature_idx,
                        other_feature_idx,
                        missing_values,
                        None,
                    ) {
                        cramers_v_test = Some(cv_result);
                    }
                    break; // Found a test result
                }
            } else {
                // Use KS test for continuous data
                if let Ok(result) = kolmogorov_smirnov_independence_test(
                    &X.view(),
                    feature_idx,
                    other_feature_idx,
                    missing_values,
                ) {
                    if result.p_value < alpha {
                        has_significant_dependence = true;
                    }
                    ks_test = Some(result);
                    break; // Found a test result
                }
            }
        }

        if has_significant_dependence {
            features_with_dependence += 1;
        }

        let test_recommendation = if chi_square_test.is_some() {
            "Chi-square test used (categorical data)".to_string()
        } else if fisher_exact_test.is_some() {
            "Fisher's exact test used (small 2x2 table)".to_string()
        } else if ks_test.is_some() {
            "Kolmogorov-Smirnov test used (continuous data)".to_string()
        } else {
            "No suitable test could be performed".to_string()
        };

        feature_results.push(FeatureIndependenceResult {
            feature_index: feature_idx,
            feature_name: Some(feature_names[feature_idx].clone()),
            chi_square_test,
            fisher_exact_test,
            cramers_v_test,
            ks_test,
            test_recommendation,
        });
    }

    let features_tested = features_with_missing.len();
    let dependence_rate = if features_tested > 0 {
        features_with_dependence as f64 / features_tested as f64
    } else {
        0.0
    };

    // Assess overall missing data mechanism
    let mechanism_assessment = if dependence_rate == 0.0 {
        "Evidence supports MCAR (Missing Completely At Random)".to_string()
    } else if dependence_rate < 0.3 {
        "Evidence suggests mostly MCAR with some MAR (Missing At Random)".to_string()
    } else if dependence_rate < 0.7 {
        "Evidence suggests MAR (Missing At Random)".to_string()
    } else {
        "Evidence suggests MNAR (Missing Not At Random) - consider domain knowledge".to_string()
    };

    let mut recommendations = Vec::new();
    if dependence_rate > 0.5 {
        recommendations.push(
            "Consider using advanced imputation methods that account for dependencies".to_string(),
        );
        recommendations.push("Review domain knowledge to assess MNAR mechanisms".to_string());
    }
    if features_tested > 0 && dependence_rate < 0.2 {
        recommendations.push("Simple imputation methods may be adequate".to_string());
    }
    if features_tested == 0 {
        recommendations.push("No missing data found - no imputation needed".to_string());
    }

    let summary = IndependenceTestSummary {
        features_tested,
        features_with_dependence,
        dependence_rate,
        mechanism_assessment,
        recommendations,
    };

    Ok(IndependenceTestSuite {
        feature_results,
        summary,
    })
}

// Helper functions

fn discretize_values(values: &[f64], n_bins: usize) -> SklResult<Vec<usize>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() < f64::EPSILON {
        // All values are the same
        return Ok(vec![0; values.len()]);
    }

    let bin_width = (max_val - min_val) / n_bins as f64;

    let discretized: Vec<usize> = values
        .iter()
        .map(|&x| {
            let bin = ((x - min_val) / bin_width).floor() as usize;
            bin.min(n_bins - 1)
        })
        .collect();

    Ok(discretized)
}

fn create_contingency_table(group1: &[usize], group2: &[usize]) -> SklResult<Array2<f64>> {
    if group1.len() != group2.len() {
        return Err(SklearsError::InvalidInput(
            "Groups must have the same length".to_string(),
        ));
    }

    let max1 = group1.iter().max().copied().unwrap_or(0);
    let max2 = group2.iter().max().copied().unwrap_or(0);

    let mut table = Array2::zeros((max1 + 1, max2 + 1));

    for (&val1, &val2) in group1.iter().zip(group2.iter()) {
        table[[val1, val2]] += 1.0;
    }

    Ok(table)
}

fn compute_chi_square_test(contingency_table: &Array2<f64>) -> SklResult<ChiSquareTestResult> {
    let chi_square_statistic = compute_chi_square_statistic(contingency_table)?;

    let (rows, cols) = contingency_table.dim();
    let degrees_of_freedom = (rows - 1) * (cols - 1);

    // Compute expected frequencies
    let n_total: f64 = contingency_table.sum();
    let mut expected_frequencies = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let row_sum: f64 = contingency_table.row(i).sum();
            let col_sum: f64 = contingency_table.column(j).sum();
            expected_frequencies[[i, j]] = (row_sum * col_sum) / n_total;
        }
    }

    // Approximate p-value and critical value (simplified)
    let critical_value = match degrees_of_freedom {
        1 => 3.841,
        2 => 5.991,
        3 => 7.815,
        4 => 9.488,
        _ => 9.488 + (degrees_of_freedom as f64 - 4.0) * 2.0, // Very rough approximation
    };

    let p_value = if chi_square_statistic > critical_value {
        0.01 // Rough approximation
    } else {
        0.1
    };

    let reject_independence = chi_square_statistic > critical_value;

    Ok(ChiSquareTestResult {
        chi_square_statistic,
        degrees_of_freedom,
        p_value,
        critical_value,
        reject_independence,
        expected_frequencies,
        observed_frequencies: contingency_table.clone(),
    })
}

fn compute_chi_square_statistic(contingency_table: &Array2<f64>) -> SklResult<f64> {
    let (rows, cols) = contingency_table.dim();
    let n_total: f64 = contingency_table.sum();

    if n_total == 0.0 {
        return Ok(0.0);
    }

    let mut chi_square = 0.0;

    for i in 0..rows {
        for j in 0..cols {
            let observed = contingency_table[[i, j]];
            let row_sum: f64 = contingency_table.row(i).sum();
            let col_sum: f64 = contingency_table.column(j).sum();
            let expected = (row_sum * col_sum) / n_total;

            if expected > 0.0 {
                chi_square += (observed - expected).powi(2) / expected;
            }
        }
    }

    Ok(chi_square)
}

fn compute_fisher_exact_test(table: &[[usize; 2]; 2]) -> SklResult<FisherExactTestResult> {
    let a = table[0][0] as f64;
    let b = table[0][1] as f64;
    let c = table[1][0] as f64;
    let d = table[1][1] as f64;

    // Compute odds ratio
    let odds_ratio = if b * c > 0.0 {
        (a * d) / (b * c)
    } else {
        f64::INFINITY
    };

    // Simplified p-value calculation (exact calculation requires hypergeometric distribution)
    let n = a + b + c + d;
    let expected_a = (a + b) * (a + c) / n;
    let chi_square = if expected_a > 0.0 {
        (a - expected_a).powi(2) / expected_a
    } else {
        0.0
    };

    // Very rough approximation of Fisher's exact test p-value
    let p_value = if chi_square > 3.841 { 0.02 } else { 0.5 };
    let p_value_less = p_value / 2.0;
    let p_value_greater = p_value / 2.0;

    // Rough confidence interval for odds ratio
    let log_or = odds_ratio.ln();
    let se_log_or = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();
    let margin = 1.96 * se_log_or;
    let confidence_interval = ((log_or - margin).exp(), (log_or + margin).exp());

    let reject_independence = p_value < 0.05;

    Ok(FisherExactTestResult {
        p_value,
        p_value_less,
        p_value_greater,
        odds_ratio,
        confidence_interval,
        reject_independence,
    })
}

fn compute_ks_statistic(sample1: &[f64], sample2: &[f64]) -> f64 {
    if sample1.is_empty() || sample2.is_empty() {
        return 0.0;
    }

    // Combine and sort all unique values
    let mut all_values: Vec<f64> = sample1.iter().chain(sample2.iter()).cloned().collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_values.dedup();

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    let mut max_diff: f64 = 0.0;

    for &value in &all_values {
        // Compute empirical CDFs at this value
        let cdf1 = sample1.iter().filter(|&&x| x <= value).count() as f64 / n1;
        let cdf2 = sample2.iter().filter(|&&x| x <= value).count() as f64 / n2;

        let diff = (cdf1 - cdf2).abs();
        max_diff = max_diff.max(diff);
    }

    max_diff
}

fn compute_ks_p_value(ks_statistic: f64, n1: usize, n2: usize) -> f64 {
    // Approximate p-value for KS test (very simplified)
    let effective_n = ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    let lambda = effective_n * ks_statistic;

    // Very rough approximation
    if lambda > 1.36 {
        0.02
    } else if lambda > 1.0 {
        0.1
    } else {
        0.5
    }
}

/// Sensitivity Analysis for Missing Data Mechanisms
///
/// Performs sensitivity analysis to assess the robustness of missing data mechanism assumptions.
/// This helps understand how sensitive conclusions about missing data patterns are to different assumptions.

/// Result of sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisResult {
    /// Base case results under MCAR assumption
    pub mcar_results: MissingDataAssessment,
    /// Results under MAR assumption with different correlation strengths
    pub mar_sensitivity: Vec<MARSensitivityCase>,
    /// Results under MNAR assumption with different selection mechanisms
    pub mnar_sensitivity: Vec<MNARSensitivityCase>,
    /// Summary of robustness across scenarios
    pub robustness_summary: RobustnessSummary,
}

/// Assessment of missing data under specific assumptions
#[derive(Debug, Clone)]
pub struct MissingDataAssessment {
    /// Proportion of missing data
    pub missing_proportion: f64,
    /// Pattern entropy (measure of pattern complexity)
    pub pattern_entropy: f64,
    /// Predictability of missingness
    pub missingness_predictability: f64,
    /// Independence test results
    pub independence_results: IndependenceTestSuite,
}

/// Sensitivity case under MAR assumption
#[derive(Debug, Clone)]
pub struct MARSensitivityCase {
    /// Correlation strength with observed variables
    pub correlation_strength: f64,
    /// Affected features
    pub affected_features: Vec<usize>,
    /// Assessment results under this scenario
    pub assessment: MissingDataAssessment,
    /// Change in conclusions compared to base case
    pub conclusion_change: f64,
}

/// Sensitivity case under MNAR assumption
#[derive(Debug, Clone)]
pub struct MNARSensitivityCase {
    /// Selection mechanism description
    pub selection_mechanism: String,
    /// Selection strength parameter
    pub selection_strength: f64,
    /// Affected features
    pub affected_features: Vec<usize>,
    /// Assessment results under this scenario
    pub assessment: MissingDataAssessment,
    /// Change in conclusions compared to base case
    pub conclusion_change: f64,
}

/// Summary of robustness across scenarios
#[derive(Debug, Clone)]
pub struct RobustnessSummary {
    /// Overall robustness score (0-1, higher = more robust)
    pub robustness_score: f64,
    /// Most sensitive aspects
    pub sensitive_aspects: Vec<String>,
    /// Recommended approach based on sensitivity
    pub recommended_approach: String,
    /// Confidence in missing data mechanism classification
    pub mechanism_confidence: f64,
}

/// Perform comprehensive sensitivity analysis for missing data mechanisms
///
/// # Arguments
///
/// * `X` - Input data matrix with missing values
/// * `missing_values` - Value representing missing data (typically NaN)
/// * `correlation_strengths` - Different correlation strengths to test for MAR
/// * `selection_strengths` - Different selection strengths to test for MNAR
///
/// # Returns
///
/// Comprehensive sensitivity analysis results
///
/// # Examples
///
/// ```
/// use sklears_impute::sensitivity_analysis;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
/// let correlation_strengths = vec![0.1, 0.3, 0.5, 0.7, 0.9];
/// let selection_strengths = vec![0.1, 0.3, 0.5, 0.7, 0.9];
///
/// let results = sensitivity_analysis(&X.view(), f64::NAN, &correlation_strengths, &selection_strengths).unwrap();
/// println!("Robustness score: {}", results.robustness_summary.robustness_score);
/// ```
#[allow(non_snake_case)]
pub fn sensitivity_analysis(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
    correlation_strengths: &[f64],
    selection_strengths: &[f64],
) -> SklResult<SensitivityAnalysisResult> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    if n_samples == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
    }

    // Base case assessment under MCAR assumption
    let mcar_results = assess_missing_data(&X, missing_values)?;

    // MAR sensitivity analysis
    let mut mar_sensitivity = Vec::new();
    for &strength in correlation_strengths {
        for feature_idx in 0..n_features {
            let case = assess_mar_sensitivity(&X, missing_values, feature_idx, strength)?;
            mar_sensitivity.push(case);
        }
    }

    // MNAR sensitivity analysis
    let mut mnar_sensitivity = Vec::new();
    for &strength in selection_strengths {
        for feature_idx in 0..n_features {
            let case = assess_mnar_sensitivity(&X, missing_values, feature_idx, strength)?;
            mnar_sensitivity.push(case);
        }
    }

    // Compute robustness summary
    let robustness_summary =
        compute_robustness_summary(&mcar_results, &mar_sensitivity, &mnar_sensitivity)?;

    Ok(SensitivityAnalysisResult {
        mcar_results,
        mar_sensitivity,
        mnar_sensitivity,
        robustness_summary,
    })
}

/// Perform pattern-based sensitivity analysis
///
/// Analyzes how different missing data patterns affect conclusions about the missing data mechanism.
///
/// # Arguments
///
/// * `X` - Input data matrix with missing values
/// * `missing_values` - Value representing missing data
/// * `pattern_perturbations` - Different ways to perturb the missing patterns
///
/// # Returns
///
/// Pattern-based sensitivity analysis results
#[allow(non_snake_case)]
pub fn pattern_sensitivity_analysis(
    X: &ArrayView2<'_, Float>,
    missing_values: f64,
    pattern_perturbations: &[f64],
) -> SklResult<Vec<PatternSensitivityResult>> {
    let X = X.mapv(|x| x as f64);
    let (n_samples, n_features) = X.dim();

    let mut results = Vec::new();

    for &perturbation in pattern_perturbations {
        // Create perturbed missing pattern
        let X_perturbed = perturb_missing_pattern(&X, missing_values, perturbation)?;

        // Assess missing data under perturbed pattern
        let assessment = assess_missing_data(&X_perturbed, missing_values)?;

        // Compare with original assessment
        let original_assessment = assess_missing_data(&X, missing_values)?;
        let sensitivity_score =
            compute_pattern_sensitivity_score(&original_assessment, &assessment);

        results.push(PatternSensitivityResult {
            perturbation_strength: perturbation,
            assessment,
            sensitivity_score,
            pattern_changes: count_pattern_changes(&X, &X_perturbed, missing_values),
        });
    }

    Ok(results)
}

/// Result of pattern-based sensitivity analysis
#[derive(Debug, Clone)]
pub struct PatternSensitivityResult {
    /// Strength of pattern perturbation applied
    pub perturbation_strength: f64,
    /// Assessment results under perturbed pattern
    pub assessment: MissingDataAssessment,
    /// Sensitivity score (how much conclusions changed)
    pub sensitivity_score: f64,
    /// Number of pattern changes made
    pub pattern_changes: usize,
}

// Helper functions for sensitivity analysis

#[allow(non_snake_case)]
fn assess_missing_data(X: &Array2<f64>, missing_values: f64) -> SklResult<MissingDataAssessment> {
    let (n_samples, n_features) = X.dim();

    // Compute missing proportion
    let mut total_missing = 0;
    let total_values = n_samples * n_features;

    let is_missing_nan = missing_values.is_nan();

    for i in 0..n_samples {
        for j in 0..n_features {
            let is_missing = if is_missing_nan {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };

            if is_missing {
                total_missing += 1;
            }
        }
    }

    let missing_proportion = total_missing as f64 / total_values as f64;

    // Compute pattern entropy
    let pattern_entropy = compute_pattern_entropy(X, missing_values)?;

    // Compute missingness predictability
    let missingness_predictability = compute_missingness_predictability(X, missing_values)?;

    // Run independence tests
    let X_view = X.view().mapv(|x| x as Float);
    let independence_results =
        run_independence_test_suite(&X_view.view(), missing_values as Float, None, None)?;

    Ok(MissingDataAssessment {
        missing_proportion,
        pattern_entropy,
        missingness_predictability,
        independence_results,
    })
}

#[allow(non_snake_case)]
fn assess_mar_sensitivity(
    X: &Array2<f64>,
    missing_values: f64,
    target_feature: usize,
    correlation_strength: f64,
) -> SklResult<MARSensitivityCase> {
    // Simulate MAR mechanism with specified correlation strength
    let X_mar = simulate_mar_mechanism(X, missing_values, target_feature, correlation_strength)?;

    let assessment = assess_missing_data(&X_mar, missing_values)?;
    let base_assessment = assess_missing_data(X, missing_values)?;

    let conclusion_change = compute_assessment_difference(&base_assessment, &assessment);

    Ok(MARSensitivityCase {
        correlation_strength,
        affected_features: vec![target_feature],
        assessment,
        conclusion_change,
    })
}

#[allow(non_snake_case)]
fn assess_mnar_sensitivity(
    X: &Array2<f64>,
    missing_values: f64,
    target_feature: usize,
    selection_strength: f64,
) -> SklResult<MNARSensitivityCase> {
    // Simulate MNAR mechanism with specified selection strength
    let X_mnar = simulate_mnar_mechanism(X, missing_values, target_feature, selection_strength)?;

    let assessment = assess_missing_data(&X_mnar, missing_values)?;
    let base_assessment = assess_missing_data(X, missing_values)?;

    let conclusion_change = compute_assessment_difference(&base_assessment, &assessment);

    Ok(MNARSensitivityCase {
        selection_mechanism: "threshold_based".to_string(),
        selection_strength,
        affected_features: vec![target_feature],
        assessment,
        conclusion_change,
    })
}

fn compute_robustness_summary(
    mcar_results: &MissingDataAssessment,
    mar_sensitivity: &[MARSensitivityCase],
    mnar_sensitivity: &[MNARSensitivityCase],
) -> SklResult<RobustnessSummary> {
    // Compute overall robustness based on variation in conclusions
    let mar_variations: Vec<f64> = mar_sensitivity
        .iter()
        .map(|case| case.conclusion_change)
        .collect();
    let mnar_variations: Vec<f64> = mnar_sensitivity
        .iter()
        .map(|case| case.conclusion_change)
        .collect();

    let mar_avg_variation = if mar_variations.is_empty() {
        0.0
    } else {
        mar_variations.iter().sum::<f64>() / mar_variations.len() as f64
    };

    let mnar_avg_variation = if mnar_variations.is_empty() {
        0.0
    } else {
        mnar_variations.iter().sum::<f64>() / mnar_variations.len() as f64
    };

    let robustness_score = 1.0 - (mar_avg_variation + mnar_avg_variation) / 2.0;
    let robustness_score = robustness_score.max(0.0).min(1.0);

    // Identify sensitive aspects
    let mut sensitive_aspects = Vec::new();
    if mar_avg_variation > 0.3 {
        sensitive_aspects.push("MAR assumptions".to_string());
    }
    if mnar_avg_variation > 0.3 {
        sensitive_aspects.push("MNAR assumptions".to_string());
    }
    if mcar_results.missing_proportion > 0.5 {
        sensitive_aspects.push("High missing proportion".to_string());
    }

    // Recommend approach
    let recommended_approach = if robustness_score > 0.8 {
        "MCAR assumption appears robust".to_string()
    } else if mar_avg_variation < mnar_avg_variation {
        "Consider MAR-based imputation".to_string()
    } else {
        "Consider MNAR-aware methods".to_string()
    };

    // Mechanism confidence based on independence tests and robustness
    let independence_confidence = 1.0 - mcar_results.independence_results.summary.dependence_rate;
    let mechanism_confidence = (robustness_score + independence_confidence) / 2.0;

    Ok(RobustnessSummary {
        robustness_score,
        sensitive_aspects,
        recommended_approach,
        mechanism_confidence,
    })
}

fn compute_pattern_entropy(X: &Array2<f64>, missing_values: f64) -> SklResult<f64> {
    let (n_samples, n_features) = X.dim();
    let mut pattern_counts = HashMap::new();

    let is_missing_nan = missing_values.is_nan();

    for i in 0..n_samples {
        let mut pattern = Vec::new();
        for j in 0..n_features {
            let is_missing = if is_missing_nan {
                X[[i, j]].is_nan()
            } else {
                (X[[i, j]] - missing_values).abs() < f64::EPSILON
            };
            pattern.push(if is_missing { 1 } else { 0 });
        }

        let pattern_key = format!("{:?}", pattern);
        *pattern_counts.entry(pattern_key).or_insert(0) += 1;
    }

    // Compute entropy
    let mut entropy = 0.0;
    for &count in pattern_counts.values() {
        let probability = count as f64 / n_samples as f64;
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }

    Ok(entropy)
}

fn compute_missingness_predictability(X: &Array2<f64>, missing_values: f64) -> SklResult<f64> {
    let (n_samples, n_features) = X.dim();

    let is_missing_nan = missing_values.is_nan();

    // Simple measure: correlation between missingness indicators
    let mut total_correlation = 0.0;
    let mut correlation_count = 0;

    for j1 in 0..n_features {
        for j2 in (j1 + 1)..n_features {
            let mut missing1 = Vec::new();
            let mut missing2 = Vec::new();

            for i in 0..n_samples {
                let is_missing1 = if is_missing_nan {
                    X[[i, j1]].is_nan()
                } else {
                    (X[[i, j1]] - missing_values).abs() < f64::EPSILON
                };

                let is_missing2 = if is_missing_nan {
                    X[[i, j2]].is_nan()
                } else {
                    (X[[i, j2]] - missing_values).abs() < f64::EPSILON
                };

                missing1.push(if is_missing1 { 1.0 } else { 0.0 });
                missing2.push(if is_missing2 { 1.0 } else { 0.0 });
            }

            let correlation = compute_correlation_coefficient(&missing1, &missing2);
            total_correlation += correlation.abs();
            correlation_count += 1;
        }
    }

    Ok(if correlation_count > 0 {
        total_correlation / correlation_count as f64
    } else {
        0.0
    })
}

fn compute_assessment_difference(
    base: &MissingDataAssessment,
    other: &MissingDataAssessment,
) -> f64 {
    let prop_diff = (base.missing_proportion - other.missing_proportion).abs();
    let entropy_diff =
        (base.pattern_entropy - other.pattern_entropy).abs() / base.pattern_entropy.max(1e-8);
    let pred_diff = (base.missingness_predictability - other.missingness_predictability).abs();
    let p_value_diff = (base.independence_results.summary.dependence_rate
        - other.independence_results.summary.dependence_rate)
        .abs();

    (prop_diff + entropy_diff + pred_diff + p_value_diff) / 4.0
}

fn simulate_mar_mechanism(
    X: &Array2<f64>,
    missing_values: f64,
    target_feature: usize,
    correlation_strength: f64,
) -> SklResult<Array2<f64>> {
    let mut X_mar = X.clone();
    let (n_samples, n_features) = X.dim();

    if target_feature >= n_features {
        return Err(SklearsError::InvalidInput(
            "Invalid target feature index".to_string(),
        ));
    }

    // Simple MAR simulation: missingness depends on another observed feature
    let predictor_feature = (target_feature + 1) % n_features;

    // Find threshold for creating MAR missingness
    let mut predictor_values: Vec<f64> = Vec::new();
    for i in 0..n_samples {
        if !is_value_missing(X[[i, predictor_feature]], missing_values) {
            predictor_values.push(X[[i, predictor_feature]]);
        }
    }

    if predictor_values.is_empty() {
        return Ok(X_mar);
    }

    predictor_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold_idx = ((1.0 - correlation_strength) * predictor_values.len() as f64) as usize;
    let threshold = predictor_values
        .get(threshold_idx)
        .cloned()
        .unwrap_or(predictor_values[0]);

    // Introduce MAR missingness
    for i in 0..n_samples {
        if !is_value_missing(X[[i, predictor_feature]], missing_values)
            && X[[i, predictor_feature]] > threshold
            && !is_value_missing(X[[i, target_feature]], missing_values)
        {
            X_mar[[i, target_feature]] = missing_values;
        }
    }

    Ok(X_mar)
}

fn simulate_mnar_mechanism(
    X: &Array2<f64>,
    missing_values: f64,
    target_feature: usize,
    selection_strength: f64,
) -> SklResult<Array2<f64>> {
    let mut X_mnar = X.clone();
    let (n_samples, n_features) = X.dim();

    if target_feature >= n_features {
        return Err(SklearsError::InvalidInput(
            "Invalid target feature index".to_string(),
        ));
    }

    // MNAR simulation: missingness depends on the value itself
    let mut target_values: Vec<f64> = Vec::new();
    for i in 0..n_samples {
        if !is_value_missing(X[[i, target_feature]], missing_values) {
            target_values.push(X[[i, target_feature]]);
        }
    }

    if target_values.is_empty() {
        return Ok(X_mnar);
    }

    target_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold_idx = ((1.0 - selection_strength) * target_values.len() as f64) as usize;
    let threshold = target_values
        .get(threshold_idx)
        .cloned()
        .unwrap_or(target_values[0]);

    // Introduce MNAR missingness (high values more likely to be missing)
    for i in 0..n_samples {
        if !is_value_missing(X[[i, target_feature]], missing_values)
            && X[[i, target_feature]] > threshold
        {
            X_mnar[[i, target_feature]] = missing_values;
        }
    }

    Ok(X_mnar)
}

fn perturb_missing_pattern(
    X: &Array2<f64>,
    missing_values: f64,
    perturbation_strength: f64,
) -> SklResult<Array2<f64>> {
    let mut X_perturbed = X.clone();
    let (n_samples, n_features) = X.dim();

    let perturbation_rate = perturbation_strength.min(0.5); // Limit perturbation
    let n_perturbations = ((n_samples * n_features) as f64 * perturbation_rate) as usize;

    use scirs2_core::random::{Random, Rng};
    let mut rng = Random::default();

    for _ in 0..n_perturbations {
        let i = rng.gen_range(0..n_samples);
        let j = rng.gen_range(0..n_features);

        let is_currently_missing = is_value_missing(X_perturbed[[i, j]], missing_values);

        if is_currently_missing {
            // Replace missing with mean of observed values in this feature
            let mut observed_values = Vec::new();
            for row in 0..n_samples {
                if !is_value_missing(X[[row, j]], missing_values) {
                    observed_values.push(X[[row, j]]);
                }
            }

            if !observed_values.is_empty() {
                let mean = observed_values.iter().sum::<f64>() / observed_values.len() as f64;
                X_perturbed[[i, j]] = mean;
            }
        } else {
            // Make observed value missing
            X_perturbed[[i, j]] = missing_values;
        }
    }

    Ok(X_perturbed)
}

fn compute_pattern_sensitivity_score(
    original: &MissingDataAssessment,
    perturbed: &MissingDataAssessment,
) -> f64 {
    compute_assessment_difference(original, perturbed)
}

fn count_pattern_changes(
    X_original: &Array2<f64>,
    X_perturbed: &Array2<f64>,
    missing_values: f64,
) -> usize {
    let (n_samples, n_features) = X_original.dim();
    let mut changes = 0;

    for i in 0..n_samples {
        for j in 0..n_features {
            let orig_missing = is_value_missing(X_original[[i, j]], missing_values);
            let pert_missing = is_value_missing(X_perturbed[[i, j]], missing_values);

            if orig_missing != pert_missing {
                changes += 1;
            }
        }
    }

    changes
}

fn is_value_missing(value: f64, missing_values: f64) -> bool {
    if missing_values.is_nan() {
        value.is_nan()
    } else {
        (value - missing_values).abs() < f64::EPSILON
    }
}

fn compute_correlation_coefficient(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;

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
