//! Basic classification metrics
//!
//! This module contains fundamental classification evaluation metrics including
//! accuracy, precision, recall, F1-score, confusion matrix, and related functions.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{BTreeSet, HashSet};

/// Calculate accuracy score for classification
///
/// Accuracy is the ratio of correctly predicted observations to the total observations.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
///
/// # Returns
/// Accuracy score between 0 and 1
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::basic_metrics::accuracy_score;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 1, 0, 0, 1];
/// let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
/// println!("Accuracy: {:.3}", accuracy);
/// ```
pub fn accuracy_score<T: PartialEq + Copy>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();

    Ok(correct as f64 / y_true.len() as f64)
}

/// Calculate precision score for binary/multiclass classification
///
/// Precision is the ratio of correctly predicted positive observations
/// to the total predicted positive observations.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `pos_label` - Positive class label (if None, uses the largest label)
///
/// # Returns
/// Precision score between 0 and 1
pub fn precision_score<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    pos_label: Option<T>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let label = pos_label.unwrap_or_else(|| *y_true.iter().max().unwrap());

    let tp = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| **t == label && **p == label)
        .count();

    let fp = y_pred.iter().filter(|&&p| p == label).count() - tp;

    if tp + fp == 0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(tp as f64 / (tp + fp) as f64)
}

/// Calculate recall score for binary/multiclass classification
///
/// Recall is the ratio of correctly predicted positive observations
/// to all observations in actual class.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `pos_label` - Positive class label (if None, uses the largest label)
///
/// # Returns
/// Recall score between 0 and 1
pub fn recall_score<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    pos_label: Option<T>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let label = pos_label.unwrap_or_else(|| *y_true.iter().max().unwrap());

    let tp = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| **t == label && **p == label)
        .count();

    let fn_count = y_true.iter().filter(|&&t| t == label).count() - tp;

    if tp + fn_count == 0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(tp as f64 / (tp + fn_count) as f64)
}

/// Calculate F1 score for binary/multiclass classification
///
/// F1 score is the harmonic mean of precision and recall.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `pos_label` - Positive class label (if None, uses the largest label)
///
/// # Returns
/// F1 score between 0 and 1
pub fn f1_score<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    pos_label: Option<T>,
) -> MetricsResult<f64> {
    let precision = precision_score(y_true, y_pred, pos_label)?;
    let recall = recall_score(y_true, y_pred, pos_label)?;

    if precision + recall == 0.0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(2.0 * precision * recall / (precision + recall))
}

/// Calculate confusion matrix
///
/// A confusion matrix is a table that describes the performance of a classification model.
/// Each row represents actual class, each column represents predicted class.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
///
/// # Returns
/// Confusion matrix as a 2D array
pub fn confusion_matrix<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
) -> MetricsResult<Array2<usize>> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut labels = BTreeSet::new();
    for &label in y_true.iter().chain(y_pred.iter()) {
        labels.insert(label);
    }

    let labels: Vec<T> = labels.into_iter().collect();
    let n_labels = labels.len();
    let mut matrix = Array2::zeros((n_labels, n_labels));

    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        let true_idx = labels.iter().position(|&x| x == *true_label).unwrap();
        let pred_idx = labels.iter().position(|&x| x == *pred_label).unwrap();
        matrix[[true_idx, pred_idx]] += 1;
    }

    Ok(matrix)
}

/// Calculate F-beta score
///
/// F-beta score is the weighted harmonic mean of precision and recall.
/// When beta=1, it becomes the F1 score.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `beta` - Weight of recall in harmonic mean
/// * `pos_label` - Positive class label (if None, uses the largest label)
///
/// # Returns
/// F-beta score between 0 and 1
pub fn fbeta_score<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
    beta: f64,
    pos_label: Option<T>,
) -> MetricsResult<f64> {
    if beta < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "beta should be >= 0".to_string(),
        ));
    }

    let precision = precision_score(y_true, y_pred, pos_label)?;
    let recall = recall_score(y_true, y_pred, pos_label)?;

    if precision + recall == 0.0 {
        return Ok(0.0);
    }

    let beta_squared = beta * beta;
    Ok((1.0 + beta_squared) * precision * recall / (beta_squared * precision + recall))
}

/// Calculate balanced accuracy score
///
/// Balanced accuracy adjusts accuracy for imbalanced datasets by averaging
/// recall obtained on each class.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
///
/// # Returns
/// Balanced accuracy score between 0 and 1
pub fn balanced_accuracy_score<T: PartialEq + Copy + Ord + std::hash::Hash>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Get unique classes
    let mut classes = HashSet::new();
    for &label in y_true.iter() {
        classes.insert(label);
    }

    let mut sum_recall = 0.0;
    for &class in &classes {
        let tp = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| **t == class && **p == class)
            .count() as f64;

        let condition_positive = y_true.iter().filter(|&&t| t == class).count() as f64;

        if condition_positive > 0.0 {
            sum_recall += tp / condition_positive;
        }
    }

    Ok(sum_recall / classes.len() as f64)
}

/// Calculate Cohen's kappa coefficient
///
/// Cohen's kappa measures inter-annotator agreement for categorical items.
/// It considers the possibility of agreement occurring by chance.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
///
/// # Returns
/// Cohen's kappa coefficient
pub fn cohen_kappa_score<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let cm = confusion_matrix(y_true, y_pred)?;
    let n = cm.sum() as f64;

    // Observed accuracy
    let po = cm.diag().sum() as f64 / n;

    // Expected accuracy (by chance)
    let mut pe = 0.0;
    for i in 0..cm.nrows() {
        let row_sum = cm.row(i).sum() as f64;
        let col_sum = cm.column(i).sum() as f64;
        pe += (row_sum * col_sum) / (n * n);
    }

    if (1.0 - pe).abs() < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok((po - pe) / (1.0 - pe))
}

/// Calculate Matthews correlation coefficient (MCC)
///
/// MCC is a correlation coefficient between observed and predicted binary classifications.
/// It returns a value between -1 and +1, where +1 represents perfect prediction,
/// 0 no better than random, and -1 indicates total disagreement.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
///
/// # Returns
/// Matthews correlation coefficient
pub fn matthews_corrcoef<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_pred: &Array1<T>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let cm = confusion_matrix(y_true, y_pred)?;

    // For multiclass, use generalized MCC formula
    let n = cm.sum() as f64;
    let sum_row: Vec<f64> = (0..cm.nrows()).map(|i| cm.row(i).sum() as f64).collect();
    let sum_col: Vec<f64> = (0..cm.ncols()).map(|j| cm.column(j).sum() as f64).collect();

    let mut numerator = 0.0;
    for i in 0..cm.nrows() {
        for j in 0..cm.ncols() {
            numerator += cm[[i, j]] as f64 * cm[[j, i]] as f64;
        }
    }
    numerator *= n;

    for i in 0..cm.nrows() {
        numerator -= sum_row[i] * sum_col[i];
    }

    let sum_row_sq: f64 = sum_row.iter().map(|x| x * x).sum();
    let sum_col_sq: f64 = sum_col.iter().map(|x| x * x).sum();

    let denominator = ((n * n - sum_row_sq) * (n * n - sum_col_sq)).sqrt();

    if denominator == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / denominator)
}
