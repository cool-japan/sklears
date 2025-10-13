//! Fairness metrics for algorithmic bias evaluation
//!
//! This module contains metrics for evaluating fairness and detecting bias
//! in machine learning models across different demographic groups.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Demographic Parity Difference
///
/// Computes the difference in positive prediction rates between protected and unprotected groups.
/// Demographic parity (also called statistical parity) is achieved when the positive prediction
/// rates are equal across groups. A value close to 0 indicates demographic parity (fairness).
///
/// # Arguments
/// * `y_pred` - Binary predictions (0 or 1)
/// * `sensitive_features` - Group membership indicators (0 for unprotected, 1 for protected)
///
/// # Returns
/// Difference in positive rates (protected_rate - unprotected_rate)
/// - Positive values indicate higher positive rate for protected group
/// - Negative values indicate higher positive rate for unprotected group
/// - Value of 0 indicates perfect demographic parity
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::fairness_metrics::demographic_parity_difference;
///
/// let y_pred = array![1, 0, 1, 1, 0, 1];
/// let sensitive_features = array![0, 0, 0, 1, 1, 1];
/// let dpd = demographic_parity_difference(&y_pred, &sensitive_features).unwrap();
/// println!("Demographic Parity Difference: {:.3}", dpd);
/// ```
pub fn demographic_parity_difference(
    y_pred: &Array1<i32>,
    sensitive_features: &Array1<i32>,
) -> MetricsResult<f64> {
    if y_pred.len() != sensitive_features.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_pred.len()],
            actual: vec![sensitive_features.len()],
        });
    }

    if y_pred.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Validate binary predictions
    for &pred in y_pred.iter() {
        if pred != 0 && pred != 1 {
            return Err(MetricsError::InvalidParameter(
                "y_pred must contain only 0 and 1".to_string(),
            ));
        }
    }

    // Validate binary sensitive features
    for &feature in sensitive_features.iter() {
        if feature != 0 && feature != 1 {
            return Err(MetricsError::InvalidParameter(
                "sensitive_features must contain only 0 and 1".to_string(),
            ));
        }
    }

    // Calculate positive rates for each group
    let mut unprotected_total = 0;
    let mut unprotected_positive = 0;
    let mut protected_total = 0;
    let mut protected_positive = 0;

    for (pred, feature) in y_pred.iter().zip(sensitive_features.iter()) {
        if *feature == 0 {
            // Unprotected group
            unprotected_total += 1;
            if *pred == 1 {
                unprotected_positive += 1;
            }
        } else {
            // Protected group
            protected_total += 1;
            if *pred == 1 {
                protected_positive += 1;
            }
        }
    }

    if unprotected_total == 0 || protected_total == 0 {
        return Err(MetricsError::InvalidParameter(
            "Both groups must have at least one sample".to_string(),
        ));
    }

    let unprotected_rate = unprotected_positive as f64 / unprotected_total as f64;
    let protected_rate = protected_positive as f64 / protected_total as f64;

    Ok(protected_rate - unprotected_rate)
}

/// Equalized Odds Difference
///
/// Computes the difference in true positive rates between protected and unprotected groups.
/// Equalized odds is achieved when the true positive rates (sensitivity) and false positive
/// rates (1 - specificity) are equal across groups. This function focuses on the true positive
/// rate difference. A value close to 0 indicates equalized odds (fairness).
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_pred` - Binary predictions (0 or 1)
/// * `sensitive_features` - Group membership indicators (0 for unprotected, 1 for protected)
///
/// # Returns
/// Difference in true positive rates (protected_tpr - unprotected_tpr)
/// - Positive values indicate higher TPR for protected group
/// - Negative values indicate higher TPR for unprotected group
/// - Value of 0 indicates equal true positive rates
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::fairness_metrics::equalized_odds_difference;
///
/// let y_true = array![1, 0, 1, 1, 0, 1];
/// let y_pred = array![1, 0, 0, 1, 0, 1];
/// let sensitive_features = array![0, 0, 0, 1, 1, 1];
/// let eod = equalized_odds_difference(&y_true, &y_pred, &sensitive_features).unwrap();
/// println!("Equalized Odds Difference: {:.3}", eod);
/// ```
pub fn equalized_odds_difference(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    sensitive_features: &Array1<i32>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() || y_true.len() != sensitive_features.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len(), sensitive_features.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Validate binary inputs
    for &label in y_true.iter() {
        if label != 0 && label != 1 {
            return Err(MetricsError::InvalidParameter(
                "y_true must contain only 0 and 1".to_string(),
            ));
        }
    }

    for &pred in y_pred.iter() {
        if pred != 0 && pred != 1 {
            return Err(MetricsError::InvalidParameter(
                "y_pred must contain only 0 and 1".to_string(),
            ));
        }
    }

    for &feature in sensitive_features.iter() {
        if feature != 0 && feature != 1 {
            return Err(MetricsError::InvalidParameter(
                "sensitive_features must contain only 0 and 1".to_string(),
            ));
        }
    }

    // Calculate TPR for each group
    let mut unprotected_tp = 0;
    let mut unprotected_p = 0;
    let mut protected_tp = 0;
    let mut protected_p = 0;

    for ((true_val, pred_val), feature) in y_true
        .iter()
        .zip(y_pred.iter())
        .zip(sensitive_features.iter())
    {
        if *true_val == 1 {
            // Positive case
            if *feature == 0 {
                // Unprotected group
                unprotected_p += 1;
                if *pred_val == 1 {
                    unprotected_tp += 1;
                }
            } else {
                // Protected group
                protected_p += 1;
                if *pred_val == 1 {
                    protected_tp += 1;
                }
            }
        }
    }

    if unprotected_p == 0 || protected_p == 0 {
        return Err(MetricsError::InvalidParameter(
            "Both groups must have at least one positive sample".to_string(),
        ));
    }

    let unprotected_tpr = unprotected_tp as f64 / unprotected_p as f64;
    let protected_tpr = protected_tp as f64 / protected_p as f64;

    Ok(protected_tpr - unprotected_tpr)
}

/// Equal Opportunity Difference
///
/// This is an alias for equalized_odds_difference for the positive class only.
/// Equal opportunity requires equal true positive rates across groups, which is
/// a relaxation of equalized odds that only considers the positive class.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_pred` - Binary predictions (0 or 1)
/// * `sensitive_features` - Group membership indicators (0 for unprotected, 1 for protected)
///
/// # Returns
/// Difference in true positive rates (protected_tpr - unprotected_tpr)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::fairness_metrics::equal_opportunity_difference;
///
/// let y_true = array![1, 0, 1, 1, 0, 1];
/// let y_pred = array![1, 0, 0, 1, 0, 1];
/// let sensitive_features = array![0, 0, 0, 1, 1, 1];
/// let eod = equal_opportunity_difference(&y_true, &y_pred, &sensitive_features).unwrap();
/// println!("Equal Opportunity Difference: {:.3}", eod);
/// ```
pub fn equal_opportunity_difference(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    sensitive_features: &Array1<i32>,
) -> MetricsResult<f64> {
    equalized_odds_difference(y_true, y_pred, sensitive_features)
}

/// Demographic Parity Ratio
///
/// Computes the ratio of positive prediction rates between protected and unprotected groups.
/// This is the multiplicative version of demographic parity difference. A value close to 1
/// indicates demographic parity (fairness).
///
/// # Arguments
/// * `y_pred` - Binary predictions (0 or 1)
/// * `sensitive_features` - Group membership indicators (0 for unprotected, 1 for protected)
///
/// # Returns
/// Ratio of positive rates (protected_rate / unprotected_rate)
/// - Values > 1 indicate higher positive rate for protected group
/// - Values < 1 indicate higher positive rate for unprotected group
/// - Value of 1 indicates perfect demographic parity
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::fairness_metrics::demographic_parity_ratio;
///
/// let y_pred = array![1, 0, 1, 1, 0, 1];
/// let sensitive_features = array![0, 0, 0, 1, 1, 1];
/// let dpr = demographic_parity_ratio(&y_pred, &sensitive_features).unwrap();
/// println!("Demographic Parity Ratio: {:.3}", dpr);
/// ```
pub fn demographic_parity_ratio(
    y_pred: &Array1<i32>,
    sensitive_features: &Array1<i32>,
) -> MetricsResult<f64> {
    if y_pred.len() != sensitive_features.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_pred.len()],
            actual: vec![sensitive_features.len()],
        });
    }

    if y_pred.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Validate binary predictions
    for &pred in y_pred.iter() {
        if pred != 0 && pred != 1 {
            return Err(MetricsError::InvalidParameter(
                "y_pred must contain only 0 and 1".to_string(),
            ));
        }
    }

    // Validate binary sensitive features
    for &feature in sensitive_features.iter() {
        if feature != 0 && feature != 1 {
            return Err(MetricsError::InvalidParameter(
                "sensitive_features must contain only 0 and 1".to_string(),
            ));
        }
    }

    // Calculate positive rates for each group
    let mut unprotected_total = 0;
    let mut unprotected_positive = 0;
    let mut protected_total = 0;
    let mut protected_positive = 0;

    for (pred, feature) in y_pred.iter().zip(sensitive_features.iter()) {
        if *feature == 0 {
            // Unprotected group
            unprotected_total += 1;
            if *pred == 1 {
                unprotected_positive += 1;
            }
        } else {
            // Protected group
            protected_total += 1;
            if *pred == 1 {
                protected_positive += 1;
            }
        }
    }

    if unprotected_total == 0 || protected_total == 0 {
        return Err(MetricsError::InvalidParameter(
            "Both groups must have at least one sample".to_string(),
        ));
    }

    let unprotected_rate = unprotected_positive as f64 / unprotected_total as f64;
    let protected_rate = protected_positive as f64 / protected_total as f64;

    if unprotected_rate == 0.0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(protected_rate / unprotected_rate)
}
