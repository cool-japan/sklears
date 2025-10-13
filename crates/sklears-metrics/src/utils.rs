//! Utility functions for metrics calculations
#![allow(dead_code)]

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

/// Check if two arrays have the same length
pub fn check_consistent_length<T>(a: &Array1<T>, b: &Array1<T>) -> MetricsResult<()> {
    if a.len() != b.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }
    Ok(())
}

/// Get unique labels from an array
pub fn unique_labels(y: &Array1<Int>) -> Vec<Int> {
    let mut labels: Vec<Int> = y.iter().copied().collect();
    labels.sort_unstable();
    labels.dedup();
    labels
}

/// Create a confusion matrix
pub fn confusion_matrix(
    y_true: &Array1<Int>,
    y_pred: &Array1<Int>,
    labels: Option<&[Int]>,
) -> MetricsResult<Array2<Int>> {
    check_consistent_length(y_true, y_pred)?;

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let unique_labels = if let Some(labels) = labels {
        labels.to_vec()
    } else {
        let mut all_labels = unique_labels(y_true);
        let pred_labels = unique_labels(y_pred);
        all_labels.extend_from_slice(&pred_labels);
        all_labels.sort_unstable();
        all_labels.dedup();
        all_labels
    };

    let n_labels = unique_labels.len();
    let mut cm = Array2::zeros((n_labels, n_labels));

    // Create label to index mapping
    let label_to_idx: HashMap<Int, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        if let (Some(&true_idx), Some(&pred_idx)) =
            (label_to_idx.get(&true_label), label_to_idx.get(&pred_label))
        {
            cm[[true_idx, pred_idx]] += 1;
        }
    }

    Ok(cm)
}

/// Calculate true positives, false positives, true negatives, false negatives for binary classification
pub fn binary_confusion_matrix_stats(
    y_true: &Array1<Int>,
    y_pred: &Array1<Int>,
    pos_label: Int,
) -> MetricsResult<(Int, Int, Int, Int)> {
    check_consistent_length(y_true, y_pred)?;

    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_count = 0;

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        match (true_label == pos_label, pred_label == pos_label) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_count += 1,
            (false, false) => tn += 1,
        }
    }

    Ok((tp, fp, tn, fn_count))
}

/// Average strategies for multi-class metrics
#[derive(Debug, Clone, Copy)]
pub enum Average {
    /// Micro
    Micro,
    /// Macro
    Macro,
    /// Weighted
    Weighted,

    None,
}

impl Average {
    pub fn from_str(s: &str) -> MetricsResult<Self> {
        match s {
            "micro" => Ok(Average::Micro),
            "macro" => Ok(Average::Macro),
            "weighted" => Ok(Average::Weighted),
            "none" => Ok(Average::None),
            _ => Err(MetricsError::InvalidParameter(format!(
                "Unknown average strategy: {s}"
            ))),
        }
    }
}

/// Calculate class weights for weighted averaging
pub fn class_support(y_true: &Array1<Int>, labels: &[Int]) -> Array1<Float> {
    let mut support = Array1::zeros(labels.len());

    let label_to_idx: HashMap<Int, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    for &true_label in y_true.iter() {
        if let Some(&idx) = label_to_idx.get(&true_label) {
            support[idx] += 1.0;
        }
    }

    support
}

/// Safe division that handles division by zero
pub fn safe_divide(numerator: Float, denominator: Float) -> Float {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute per-class metrics and aggregate according to averaging strategy
pub fn aggregate_scores(
    per_class_scores: &Array1<Float>,
    support: &Array1<Float>,
    average: Average,
) -> MetricsResult<Float> {
    match average {
        Average::Micro => {
            // For micro-averaging, we need to compute globally
            // This should be handled at the caller level
            Err(MetricsError::InvalidParameter(
                "Micro averaging should be handled at caller level".to_string(),
            ))
        }
        Average::Macro => Ok(per_class_scores.mean().unwrap_or(0.0)),
        Average::Weighted => {
            let total_support = support.sum();
            if total_support == 0.0 {
                return Err(MetricsError::DivisionByZero);
            }
            let weighted_sum = per_class_scores
                .iter()
                .zip(support.iter())
                .map(|(&score, &weight)| score * weight)
                .sum::<Float>();
            Ok(weighted_sum / total_support)
        }
        Average::None => {
            // Return NaN to indicate that averaging is not applied
            Ok(Float::NAN)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_check_consistent_length() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];
        let c = array![7, 8];

        assert!(check_consistent_length(&a, &b).is_ok());
        assert!(check_consistent_length(&a, &c).is_err());
    }

    #[test]
    fn test_unique_labels() {
        let y = array![0, 1, 2, 1, 0, 2, 1];
        let labels = unique_labels(&y);
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 1];

        let cm = confusion_matrix(&y_true, &y_pred, None).unwrap();

        // Expected confusion matrix:
        // [2, 0, 0]  (class 0: 2 correct, 0 misclassified as 1, 0 as 2)
        // [1, 0, 1]  (class 1: 1 misclassified as 0, 0 correct, 1 misclassified as 2)
        // [0, 2, 0]  (class 2: 0 misclassified as 0, 2 misclassified as 1, 0 correct)

        assert_eq!(cm[[0, 0]], 2);
        assert_eq!(cm[[1, 0]], 1);
        assert_eq!(cm[[2, 1]], 2);
    }

    #[test]
    fn test_binary_confusion_matrix_stats() {
        let y_true = array![1, 1, 0, 0, 1, 0];
        let y_pred = array![1, 0, 0, 1, 1, 0];

        let (tp, fp, tn, fn_count) = binary_confusion_matrix_stats(&y_true, &y_pred, 1).unwrap();

        assert_eq!(tp, 2); // positions 0, 4
        assert_eq!(fp, 1); // position 3
        assert_eq!(tn, 2); // positions 2, 5
        assert_eq!(fn_count, 1); // position 1
    }
}
