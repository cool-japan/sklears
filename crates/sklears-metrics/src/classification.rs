//! Classification metrics
//!
//! This module provides comprehensive metrics for evaluating classification models,
//! including basic metrics, multi-label metrics, probabilistic metrics, fairness metrics,
//! and specialized metrics for advanced use cases.

// Re-export all classification metrics from specialized modules
pub use crate::advanced_metrics::*;
pub use crate::basic_metrics::*;
pub use crate::display_utils::*;
pub use crate::fairness_metrics::*;
pub use crate::multilabel_metrics::*;
pub use crate::probabilistic_metrics::*;

// Additional utility functions that don't fit into other modules

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Top-k accuracy score for multi-class classification
///
/// Computes the fraction of samples whose true labels are among the top k predictions.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_score` - Prediction scores (n_samples x n_classes)
/// * `k` - Number of top predictions to consider
///
/// # Returns
/// Top-k accuracy score (0.0 to 1.0)
pub fn top_k_accuracy_score<T: PartialEq + Copy + Ord + std::hash::Hash>(
    y_true: &Array1<T>,
    y_score: &Array2<f64>,
    k: usize,
) -> MetricsResult<f64> {
    if y_true.len() != y_score.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_score.nrows()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if k == 0 || k > y_score.ncols() {
        return Err(MetricsError::InvalidParameter(format!(
            "k must be between 1 and {} (number of classes)",
            y_score.ncols()
        )));
    }

    // Get unique classes to create mapping
    let mut unique_classes = std::collections::BTreeSet::new();
    for &label in y_true.iter() {
        unique_classes.insert(label);
    }
    let classes: Vec<T> = unique_classes.into_iter().collect();

    if classes.len() != y_score.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len(), classes.len()],
            actual: vec![y_score.nrows(), y_score.ncols()],
        });
    }

    let class_to_index: std::collections::HashMap<T, usize> = classes
        .iter()
        .enumerate()
        .map(|(i, &class)| (class, i))
        .collect();

    let mut correct = 0;

    for (i, &true_label) in y_true.iter().enumerate() {
        let scores = y_score.row(i);

        // Get indices sorted by score in descending order
        let mut score_indices: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        score_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check if true label is in top k
        let true_class_idx = class_to_index[&true_label];
        if score_indices[..k]
            .iter()
            .any(|(idx, _)| *idx == true_class_idx)
        {
            correct += 1;
        }
    }

    Ok(correct as f64 / y_true.len() as f64)
}

/// Top-2 accuracy score (convenience function)
pub fn top_2_accuracy_score<T: PartialEq + Copy + Ord + std::hash::Hash>(
    y_true: &Array1<T>,
    y_score: &Array2<f64>,
) -> MetricsResult<f64> {
    top_k_accuracy_score(y_true, y_score, 2)
}

/// Top-3 accuracy score (convenience function)
pub fn top_3_accuracy_score<T: PartialEq + Copy + Ord + std::hash::Hash>(
    y_true: &Array1<T>,
    y_score: &Array2<f64>,
) -> MetricsResult<f64> {
    top_k_accuracy_score(y_true, y_score, 3)
}

/// Top-5 accuracy score (convenience function)
pub fn top_5_accuracy_score<T: PartialEq + Copy + Ord + std::hash::Hash>(
    y_true: &Array1<T>,
    y_score: &Array2<f64>,
) -> MetricsResult<f64> {
    top_k_accuracy_score(y_true, y_score, 5)
}

/// Hamming loss - fraction of labels that are incorrectly predicted
///
/// This is a general hamming loss implementation that works for both
/// single-label and multi-label classification.
pub fn hamming_loss<T: PartialEq + Copy>(
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

    let incorrect = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a != b)
        .count();

    Ok(incorrect as f64 / y_true.len() as f64)
}

/// Zero-one loss - fraction of misclassifications
///
/// This is equivalent to 1 - accuracy for single-label classification.
pub fn zero_one_loss<T: PartialEq + Copy>(
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

    let incorrect = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a != b)
        .count();

    Ok(incorrect as f64 / y_true.len() as f64)
}

/// Hinge loss for binary classification (commonly used with SVM)
///
/// # Arguments
/// * `y_true` - True binary labels (should be -1 or 1)
/// * `decision` - Decision function values (raw scores before applying threshold)
///
/// # Returns
/// Hinge loss (lower is better)
pub fn hinge_loss(y_true: &Array1<i32>, decision: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != decision.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![decision.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Validate that labels are binary (-1 or 1)
    for &label in y_true.iter() {
        if label != -1 && label != 1 {
            return Err(MetricsError::InvalidParameter(
                "Labels must be -1 or 1 for hinge loss".to_string(),
            ));
        }
    }

    let losses: f64 = y_true
        .iter()
        .zip(decision.iter())
        .map(|(&y, &d)| (1.0 - y as f64 * d).max(0.0))
        .sum();

    Ok(losses / y_true.len() as f64)
}
