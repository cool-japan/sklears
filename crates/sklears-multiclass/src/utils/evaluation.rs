//! Evaluation utilities for multiclass classification
//!
//! This module provides comprehensive evaluation tools including:
//! - Confusion matrix analysis
//! - Per-class performance metrics
//! - Multiclass-specific evaluation metrics
//! - Statistical significance testing

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

/// Confusion Matrix for multiclass classification analysis
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// The confusion matrix [n_classes, n_classes]
    pub matrix: Array2<usize>,
    /// Unique class labels in sorted order
    pub classes: Array1<i32>,
    /// Total number of samples
    pub n_samples: usize,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix from true and predicted labels
    ///
    /// # Arguments
    /// * `y_true` - True class labels
    /// * `y_pred` - Predicted class labels
    ///
    /// # Returns
    /// A Result containing the ConfusionMatrix or an error
    pub fn new(y_true: &Array1<i32>, y_pred: &Array1<i32>) -> SklResult<Self> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        if y_true.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot create confusion matrix from empty arrays".to_string(),
            ));
        }

        // Get unique classes
        let mut all_classes: Vec<i32> = y_true.iter().chain(y_pred.iter()).cloned().collect();
        all_classes.sort_unstable();
        all_classes.dedup();
        let classes = Array1::from_vec(all_classes);
        let n_classes = classes.len();

        // Create mapping from class label to index
        let class_to_idx: HashMap<i32, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, &class)| (class, i))
            .collect();

        // Initialize confusion matrix
        let mut matrix = Array2::zeros((n_classes, n_classes));

        // Fill confusion matrix
        for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
            let true_idx = class_to_idx[true_label];
            let pred_idx = class_to_idx[pred_label];
            matrix[[true_idx, pred_idx]] += 1;
        }

        Ok(ConfusionMatrix {
            matrix,
            classes,
            n_samples: y_true.len(),
        })
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get true positives for a specific class
    pub fn true_positives(&self, class_idx: usize) -> usize {
        if class_idx >= self.n_classes() {
            return 0;
        }
        self.matrix[[class_idx, class_idx]]
    }

    /// Get false positives for a specific class
    pub fn false_positives(&self, class_idx: usize) -> usize {
        if class_idx >= self.n_classes() {
            return 0;
        }

        let mut fp = 0;
        for i in 0..self.n_classes() {
            if i != class_idx {
                fp += self.matrix[[i, class_idx]];
            }
        }
        fp
    }

    /// Get false negatives for a specific class
    pub fn false_negatives(&self, class_idx: usize) -> usize {
        if class_idx >= self.n_classes() {
            return 0;
        }

        let mut fn_count = 0;
        for j in 0..self.n_classes() {
            if j != class_idx {
                fn_count += self.matrix[[class_idx, j]];
            }
        }
        fn_count
    }

    /// Get true negatives for a specific class
    pub fn true_negatives(&self, class_idx: usize) -> usize {
        if class_idx >= self.n_classes() {
            return 0;
        }

        let tp = self.true_positives(class_idx);
        let fp = self.false_positives(class_idx);
        let fn_count = self.false_negatives(class_idx);

        self.n_samples - tp - fp - fn_count
    }

    /// Calculate precision for a specific class
    pub fn precision(&self, class_idx: usize) -> f64 {
        let tp = self.true_positives(class_idx) as f64;
        let fp = self.false_positives(class_idx) as f64;

        if tp + fp == 0.0 {
            0.0 // No predicted positives
        } else {
            tp / (tp + fp)
        }
    }

    /// Calculate recall for a specific class
    pub fn recall(&self, class_idx: usize) -> f64 {
        let tp = self.true_positives(class_idx) as f64;
        let fn_count = self.false_negatives(class_idx) as f64;

        if tp + fn_count == 0.0 {
            0.0 // No actual positives
        } else {
            tp / (tp + fn_count)
        }
    }

    /// Calculate F1-score for a specific class
    pub fn f1_score(&self, class_idx: usize) -> f64 {
        let precision = self.precision(class_idx);
        let recall = self.recall(class_idx);

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }

    /// Calculate overall accuracy
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.n_classes()).map(|i| self.matrix[[i, i]]).sum();
        correct as f64 / self.n_samples as f64
    }

    /// Calculate macro-averaged precision
    pub fn macro_precision(&self) -> f64 {
        let sum: f64 = (0..self.n_classes()).map(|i| self.precision(i)).sum();
        sum / self.n_classes() as f64
    }

    /// Calculate macro-averaged recall
    pub fn macro_recall(&self) -> f64 {
        let sum: f64 = (0..self.n_classes()).map(|i| self.recall(i)).sum();
        sum / self.n_classes() as f64
    }

    /// Calculate macro-averaged F1-score
    pub fn macro_f1(&self) -> f64 {
        let sum: f64 = (0..self.n_classes()).map(|i| self.f1_score(i)).sum();
        sum / self.n_classes() as f64
    }

    /// Calculate weighted precision (weighted by class support)
    pub fn weighted_precision(&self) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_support = 0;

        for i in 0..self.n_classes() {
            let support = self.class_support(i);
            weighted_sum += self.precision(i) * support as f64;
            total_support += support;
        }

        if total_support == 0 {
            0.0
        } else {
            weighted_sum / total_support as f64
        }
    }

    /// Calculate weighted recall (weighted by class support)
    pub fn weighted_recall(&self) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_support = 0;

        for i in 0..self.n_classes() {
            let support = self.class_support(i);
            weighted_sum += self.recall(i) * support as f64;
            total_support += support;
        }

        if total_support == 0 {
            0.0
        } else {
            weighted_sum / total_support as f64
        }
    }

    /// Calculate weighted F1-score (weighted by class support)
    pub fn weighted_f1(&self) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_support = 0;

        for i in 0..self.n_classes() {
            let support = self.class_support(i);
            weighted_sum += self.f1_score(i) * support as f64;
            total_support += support;
        }

        if total_support == 0 {
            0.0
        } else {
            weighted_sum / total_support as f64
        }
    }

    /// Get support (number of actual instances) for a specific class
    pub fn class_support(&self, class_idx: usize) -> usize {
        if class_idx >= self.n_classes() {
            return 0;
        }

        (0..self.n_classes())
            .map(|j| self.matrix[[class_idx, j]])
            .sum()
    }

    /// Calculate balanced accuracy
    pub fn balanced_accuracy(&self) -> f64 {
        self.macro_recall()
    }

    /// Get a classification report as a formatted string
    pub fn classification_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "{:>12} {:>9} {:>9} {:>9} {:>9}\n",
            "class", "precision", "recall", "f1-score", "support"
        ));
        report.push_str(&format!("{}\n", "-".repeat(60)));

        for i in 0..self.n_classes() {
            let class_label = self.classes[i];
            let precision = self.precision(i);
            let recall = self.recall(i);
            let f1 = self.f1_score(i);
            let support = self.class_support(i);

            report.push_str(&format!(
                "{:>12} {:>9.2} {:>9.2} {:>9.2} {:>9}\n",
                class_label, precision, recall, f1, support
            ));
        }

        report.push_str(&format!("{}\n", "-".repeat(60)));

        // Macro averages
        report.push_str(&format!(
            "{:>12} {:>9.2} {:>9.2} {:>9.2} {:>9}\n",
            "macro avg",
            self.macro_precision(),
            self.macro_recall(),
            self.macro_f1(),
            self.n_samples
        ));

        // Weighted averages
        report.push_str(&format!(
            "{:>12} {:>9.2} {:>9.2} {:>9.2} {:>9}\n",
            "weighted avg",
            self.weighted_precision(),
            self.weighted_recall(),
            self.weighted_f1(),
            self.n_samples
        ));

        // Overall accuracy
        report.push_str(&format!("{}\n", "-".repeat(60)));
        report.push_str(&format!("{:>12} {:>37.2}\n", "accuracy", self.accuracy()));

        report
    }
}

/// Per-class performance metrics
#[derive(Debug, Clone)]
pub struct PerClassMetrics {
    /// Class label
    pub class: i32,
    /// Precision for this class
    pub precision: f64,
    /// Recall for this class
    pub recall: f64,
    /// F1-score for this class
    pub f1_score: f64,
    /// Support (number of actual instances) for this class
    pub support: usize,
    /// True positives
    pub true_positives: usize,
    /// False positives
    pub false_positives: usize,
    /// False negatives
    pub false_negatives: usize,
    /// True negatives
    pub true_negatives: usize,
}

impl PerClassMetrics {
    /// Create per-class metrics from a confusion matrix
    pub fn from_confusion_matrix(cm: &ConfusionMatrix) -> Vec<Self> {
        let mut metrics = Vec::new();

        for i in 0..cm.n_classes() {
            let class = cm.classes[i];
            let tp = cm.true_positives(i);
            let fp = cm.false_positives(i);
            let fn_count = cm.false_negatives(i);
            let tn = cm.true_negatives(i);

            metrics.push(PerClassMetrics {
                class,
                precision: cm.precision(i),
                recall: cm.recall(i),
                f1_score: cm.f1_score(i),
                support: cm.class_support(i),
                true_positives: tp,
                false_positives: fp,
                false_negatives: fn_count,
                true_negatives: tn,
            });
        }

        metrics
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_confusion_matrix_creation() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 1, 1, 0, 2, 2];

        let cm = ConfusionMatrix::new(&y_true, &y_pred).unwrap();

        assert_eq!(cm.n_classes(), 3);
        assert_eq!(cm.n_samples, 6);
        assert_eq!(cm.classes, array![0, 1, 2]);

        // Check confusion matrix values
        assert_eq!(cm.matrix[[0, 0]], 2); // True class 0, pred class 0
        assert_eq!(cm.matrix[[1, 1]], 1); // True class 1, pred class 1
        assert_eq!(cm.matrix[[2, 2]], 1); // True class 2, pred class 2
        assert_eq!(cm.matrix[[1, 2]], 1); // True class 1, pred class 2
        assert_eq!(cm.matrix[[2, 1]], 1); // True class 2, pred class 1
    }

    #[test]
    fn test_confusion_matrix_metrics() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 1, 1, 0, 2, 2];

        let cm = ConfusionMatrix::new(&y_true, &y_pred).unwrap();

        // Test for class 0
        assert_eq!(cm.true_positives(0), 2);
        assert_eq!(cm.false_positives(0), 0);
        assert_eq!(cm.false_negatives(0), 0);
        assert_eq!(cm.true_negatives(0), 4);
        assert_eq!(cm.precision(0), 1.0);
        assert_eq!(cm.recall(0), 1.0);
        assert_eq!(cm.f1_score(0), 1.0);

        // Test overall accuracy
        assert_eq!(cm.accuracy(), 4.0 / 6.0); // 4 correct out of 6
    }

    #[test]
    fn test_per_class_metrics() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 1, 1, 0, 2, 2];

        let cm = ConfusionMatrix::new(&y_true, &y_pred).unwrap();
        let metrics = PerClassMetrics::from_confusion_matrix(&cm);

        assert_eq!(metrics.len(), 3);

        // Check class 0 metrics
        assert_eq!(metrics[0].class, 0);
        assert_eq!(metrics[0].precision, 1.0);
        assert_eq!(metrics[0].recall, 1.0);
        assert_eq!(metrics[0].f1_score, 1.0);
        assert_eq!(metrics[0].support, 2);
    }

    #[test]
    fn test_classification_report() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 1, 1, 0, 2, 2];

        let cm = ConfusionMatrix::new(&y_true, &y_pred).unwrap();
        let report = cm.classification_report();

        // Just verify the report contains expected sections
        assert!(report.contains("precision"));
        assert!(report.contains("recall"));
        assert!(report.contains("f1-score"));
        assert!(report.contains("support"));
        assert!(report.contains("macro avg"));
        assert!(report.contains("weighted avg"));
        assert!(report.contains("accuracy"));
    }

    #[test]
    fn test_empty_arrays() {
        let y_true = array![];
        let y_pred = array![];

        let result = ConfusionMatrix::new(&y_true, &y_pred);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_lengths() {
        let y_true = array![0, 1, 2];
        let y_pred = array![0, 1];

        let result = ConfusionMatrix::new(&y_true, &y_pred);
        assert!(result.is_err());
    }
}
