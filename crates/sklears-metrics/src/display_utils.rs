//! Display utilities for classification metrics
//!
//! This module contains utilities for displaying and formatting classification
//! metrics in human-readable forms, including confusion matrices, classification
//! reports, and various curve displays.

use crate::{basic_metrics::*, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Display class for confusion matrices with formatting options
///
/// Provides a structured way to display confusion matrices with optional
/// class labels, custom formatting, and colormap support.
#[derive(Debug, Clone)]
pub struct ConfusionMatrixDisplay {
    pub confusion_matrix: Array2<usize>,
    pub display_labels: Option<Vec<String>>,
    pub colormap: Option<String>,
    pub values_format: Option<String>,
}

impl ConfusionMatrixDisplay {
    /// Create a new confusion matrix display
    pub fn new(confusion_matrix: Array2<usize>) -> Self {
        Self {
            confusion_matrix,
            display_labels: None,
            colormap: None,
            values_format: None,
        }
    }

    /// Set display labels for the classes
    pub fn with_display_labels(mut self, labels: Vec<String>) -> Self {
        self.display_labels = Some(labels);
        self
    }

    /// Set colormap for visualization
    pub fn with_colormap(mut self, colormap: String) -> Self {
        self.colormap = Some(colormap);
        self
    }

    /// Set format for values display
    pub fn with_values_format(mut self, format: String) -> Self {
        self.values_format = Some(format);
        self
    }

    /// Create confusion matrix display from predictions
    pub fn from_predictions<T: PartialEq + Copy + Ord + fmt::Display>(
        y_true: &Array1<T>,
        y_pred: &Array1<T>,
    ) -> MetricsResult<Self> {
        let cm = confusion_matrix(y_true, y_pred)?;

        // Generate default labels
        let mut unique_labels = BTreeSet::new();
        for label in y_true.iter().chain(y_pred.iter()) {
            unique_labels.insert(*label);
        }

        let display_labels: Vec<String> = unique_labels
            .into_iter()
            .map(|label| format!("{label}"))
            .collect();

        Ok(Self::new(cm).with_display_labels(display_labels))
    }

    /// Get formatted text representation
    pub fn text(&self) -> String {
        let mut result = String::new();
        let (n_rows, n_cols) = self.confusion_matrix.dim();

        // Header
        result.push_str("Confusion Matrix:\n");

        // Column headers
        if let Some(ref labels) = self.display_labels {
            result.push_str("           ");
            for label in labels {
                result.push_str(&format!("{label:>8}"));
            }
            result.push('\n');
        }

        // Matrix rows
        for i in 0..n_rows {
            if let Some(ref labels) = self.display_labels {
                result.push_str(&format!("{:>8}   ", labels[i]));
            }

            for j in 0..n_cols {
                let value = self.confusion_matrix[[i, j]];
                if let Some(ref _format_str) = self.values_format {
                    result.push_str(&format!("{:>8}", format!("{value}")));
                } else {
                    result.push_str(&format!("{value:>8}"));
                }
            }
            result.push('\n');
        }

        result
    }
}

impl fmt::Display for ConfusionMatrixDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// Classification report with comprehensive metrics
///
/// Provides a scikit-learn style classification report showing precision,
/// recall, F1-score, and support for each class, along with macro and
/// weighted averages.
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub f1_score: Vec<f64>,
    pub support: Vec<usize>,
    pub class_labels: Vec<String>,
    pub accuracy: f64,
    pub macro_avg: (f64, f64, f64),
    pub weighted_avg: (f64, f64, f64),
    pub digits: usize,
}

impl ClassificationReport {
    /// Generate classification report from predictions
    pub fn from_predictions<T: PartialEq + Copy + Ord + fmt::Display>(
        y_true: &Array1<T>,
        y_pred: &Array1<T>,
        target_names: Option<Vec<String>>,
        digits: Option<usize>,
    ) -> MetricsResult<Self> {
        let digits = digits.unwrap_or(2);

        // Calculate accuracy
        let accuracy = accuracy_score(y_true, y_pred)?;

        // Get unique labels
        let mut unique_labels = BTreeSet::new();
        for label in y_true.iter().chain(y_pred.iter()) {
            unique_labels.insert(*label);
        }
        let unique_labels: Vec<T> = unique_labels.into_iter().collect();

        // Calculate per-class metrics
        let mut precision = Vec::new();
        let mut recall = Vec::new();
        let mut f1_scores = Vec::new();
        let mut support_counts = Vec::new();
        let mut class_labels = Vec::new();

        for (i, &label) in unique_labels.iter().enumerate() {
            // Calculate precision, recall, F1 for this class
            let prec = precision_score(y_true, y_pred, Some(label)).unwrap_or(0.0);
            let rec = recall_score(y_true, y_pred, Some(label)).unwrap_or(0.0);
            let f1 = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };

            // Count support for this class
            let sup = y_true.iter().filter(|&&x| x == label).count();

            precision.push(prec);
            recall.push(rec);
            f1_scores.push(f1);
            support_counts.push(sup);

            // Set class label
            if let Some(ref names) = target_names {
                class_labels.push(names.get(i).cloned().unwrap_or_else(|| format!("{label}")));
            } else {
                class_labels.push(format!("{label}"));
            }
        }

        // Calculate macro averages
        let macro_precision = precision.iter().sum::<f64>() / precision.len() as f64;
        let macro_recall = recall.iter().sum::<f64>() / recall.len() as f64;
        let macro_f1 = f1_scores.iter().sum::<f64>() / f1_scores.len() as f64;

        // Calculate weighted averages
        let total_support: usize = support_counts.iter().sum();
        let weighted_precision = precision
            .iter()
            .zip(support_counts.iter())
            .map(|(p, s)| p * (*s as f64))
            .sum::<f64>()
            / total_support as f64;
        let weighted_recall = recall
            .iter()
            .zip(support_counts.iter())
            .map(|(r, s)| r * (*s as f64))
            .sum::<f64>()
            / total_support as f64;
        let weighted_f1 = f1_scores
            .iter()
            .zip(support_counts.iter())
            .map(|(f, s)| f * (*s as f64))
            .sum::<f64>()
            / total_support as f64;

        Ok(Self {
            precision,
            recall,
            f1_score: f1_scores,
            support: support_counts,
            class_labels,
            accuracy,
            macro_avg: (macro_precision, macro_recall, macro_f1),
            weighted_avg: (weighted_precision, weighted_recall, weighted_f1),
            digits,
        })
    }

    /// Get formatted text representation
    pub fn text(&self) -> String {
        let mut result = String::new();

        // Header
        result.push_str(&format!(
            "{:>12} {:>12} {:>12} {:>12} {:>12}\n",
            "", "precision", "recall", "f1-score", "support"
        ));
        result.push('\n');

        // Per-class metrics
        for i in 0..self.class_labels.len() {
            result.push_str(&format!(
                "{:>12} {:>12.*} {:>12.*} {:>12.*} {:>12}\n",
                self.class_labels[i],
                self.digits,
                self.precision[i],
                self.digits,
                self.recall[i],
                self.digits,
                self.f1_score[i],
                self.support[i]
            ));
        }

        result.push('\n');

        // Accuracy
        let total_support: usize = self.support.iter().sum();
        result.push_str(&format!(
            "{:>12} {:>12} {:>12} {:>12.*} {:>12}\n",
            "accuracy", "", "", self.digits, self.accuracy, total_support
        ));

        // Macro average
        result.push_str(&format!(
            "{:>12} {:>12.*} {:>12.*} {:>12.*} {:>12}\n",
            "macro avg",
            self.digits,
            self.macro_avg.0,
            self.digits,
            self.macro_avg.1,
            self.digits,
            self.macro_avg.2,
            total_support
        ));

        // Weighted average
        result.push_str(&format!(
            "{:>12} {:>12.*} {:>12.*} {:>12.*} {:>12}\n",
            "weighted avg",
            self.digits,
            self.weighted_avg.0,
            self.digits,
            self.weighted_avg.1,
            self.digits,
            self.weighted_avg.2,
            total_support
        ));

        result
    }
}

impl fmt::Display for ClassificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// General metrics display utility
///
/// A flexible utility for displaying multiple metrics in a formatted table.
#[derive(Debug, Clone)]
pub struct MetricsDisplay {
    pub metrics: HashMap<String, f64>,
    pub format_precision: usize,
}

impl MetricsDisplay {
    /// Create a new metrics display
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            format_precision: 4,
        }
    }

    /// Add a metric value
    pub fn add_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }

    /// Set formatting precision
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.format_precision = precision;
        self
    }

    /// Get formatted text representation
    pub fn text(&self) -> String {
        let mut result = String::new();
        result.push_str("Metrics Summary:\n");
        result.push_str("================\n");

        let mut sorted_metrics: Vec<_> = self.metrics.iter().collect();
        sorted_metrics.sort_by_key(|&(name, _)| name);

        for (name, value) in sorted_metrics {
            result.push_str(&format!(
                "{:20} : {:.*}\n",
                name, self.format_precision, value
            ));
        }

        result
    }
}

impl Default for MetricsDisplay {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MetricsDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// ROC Curve Display for visualizing ROC curves
///
/// Contains data and metadata for displaying ROC (Receiver Operating Characteristic) curves.
#[derive(Debug, Clone)]
pub struct RocCurveDisplay {
    pub fpr: Array1<f64>,
    pub tpr: Array1<f64>,
    pub roc_auc: Option<f64>,
    pub estimator_name: Option<String>,
    pub pos_label: Option<String>,
}

impl RocCurveDisplay {
    /// Create a new ROC curve display
    pub fn new(fpr: Array1<f64>, tpr: Array1<f64>) -> Self {
        Self {
            fpr,
            tpr,
            roc_auc: None,
            estimator_name: None,
            pos_label: None,
        }
    }

    /// Set the ROC AUC score
    pub fn with_roc_auc(mut self, roc_auc: f64) -> Self {
        self.roc_auc = Some(roc_auc);
        self
    }

    /// Set estimator name for display
    pub fn with_estimator_name(mut self, name: String) -> Self {
        self.estimator_name = Some(name);
        self
    }

    /// Set positive label for display
    pub fn with_pos_label(mut self, label: String) -> Self {
        self.pos_label = Some(label);
        self
    }

    /// Create ROC curve display from predictions
    ///
    /// Uses the ranking module to calculate the actual ROC curve and AUC score.
    pub fn from_predictions(
        y_true: &Array1<i32>,
        y_score: &Array1<f64>,
        pos_label: Option<String>,
        name: Option<String>,
    ) -> MetricsResult<Self> {
        use crate::ranking::{auc, roc_curve};

        // Calculate ROC curve
        let (fpr, tpr, _thresholds) = roc_curve(y_true, y_score)?;
        let roc_auc = auc(&fpr, &tpr)?;

        let mut display = Self::new(fpr, tpr).with_roc_auc(roc_auc);

        if let Some(pos_label) = pos_label {
            display = display.with_pos_label(pos_label);
        }

        if let Some(name) = name {
            display = display.with_estimator_name(name);
        }

        Ok(display)
    }

    /// Get formatted text representation
    pub fn text(&self) -> String {
        let mut result = String::new();

        if let Some(ref name) = self.estimator_name {
            result.push_str(&format!("ROC Curve for {}\n", name));
        } else {
            result.push_str("ROC Curve\n");
        }

        result.push_str("====================\n");

        if let Some(auc) = self.roc_auc {
            result.push_str(&format!("AUC: {:.4}\n", auc));
        }

        if let Some(ref label) = self.pos_label {
            result.push_str(&format!("Positive label: {}\n", label));
        }

        result.push_str(&format!("Points: {} (FPR, TPR)\n", self.fpr.len()));
        result.push_str("Random classifier line: FPR = TPR\n");

        result
    }
}

impl fmt::Display for RocCurveDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// Precision-Recall Curve Display for visualizing precision-recall curves
///
/// Contains data and metadata for displaying Precision-Recall curves.
#[derive(Debug, Clone)]
pub struct PrecisionRecallDisplay {
    pub precision: Array1<f64>,
    pub recall: Array1<f64>,
    pub average_precision: Option<f64>,
    pub estimator_name: Option<String>,
    pub pos_label: Option<String>,
}

impl PrecisionRecallDisplay {
    /// Create a new precision-recall curve display
    pub fn new(precision: Array1<f64>, recall: Array1<f64>) -> Self {
        Self {
            precision,
            recall,
            average_precision: None,
            estimator_name: None,
            pos_label: None,
        }
    }

    /// Set the average precision score
    pub fn with_average_precision(mut self, average_precision: f64) -> Self {
        self.average_precision = Some(average_precision);
        self
    }

    /// Set estimator name for display
    pub fn with_estimator_name(mut self, name: String) -> Self {
        self.estimator_name = Some(name);
        self
    }

    /// Set positive label for display
    pub fn with_pos_label(mut self, label: String) -> Self {
        self.pos_label = Some(label);
        self
    }

    /// Create Precision-Recall curve display from predictions
    ///
    /// Uses the ranking module to calculate the actual PR curve and average precision.
    pub fn from_predictions(
        y_true: &Array1<i32>,
        y_score: &Array1<f64>,
        pos_label: Option<String>,
        name: Option<String>,
    ) -> MetricsResult<Self> {
        use crate::ranking::{average_precision_score, precision_recall_curve};

        // Calculate PR curve
        let (precision, recall, _thresholds) = precision_recall_curve(y_true, y_score)?;
        let avg_precision = average_precision_score(y_true, y_score)?;

        let mut display = Self::new(precision, recall).with_average_precision(avg_precision);

        if let Some(pos_label) = pos_label {
            display = display.with_pos_label(pos_label);
        }

        if let Some(name) = name {
            display = display.with_estimator_name(name);
        }

        Ok(display)
    }

    /// Get formatted text representation
    pub fn text(&self) -> String {
        let mut result = String::new();

        if let Some(ref name) = self.estimator_name {
            result.push_str(&format!("Precision-Recall Curve for {}\n", name));
        } else {
            result.push_str("Precision-Recall Curve\n");
        }

        result.push_str("===============================\n");

        if let Some(ap) = self.average_precision {
            result.push_str(&format!("Average Precision: {:.4}\n", ap));
        }

        if let Some(ref label) = self.pos_label {
            result.push_str(&format!("Positive label: {}\n", label));
        }

        result.push_str(&format!(
            "Points: {} (Precision, Recall)\n",
            self.precision.len()
        ));

        result
    }
}

impl fmt::Display for PrecisionRecallDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// DET Curve Display for Detection Error Tradeoff curves
///
/// Contains data and metadata for displaying DET (Detection Error Tradeoff) curves,
/// which plot False Positive Rate vs False Negative Rate.
#[derive(Debug, Clone)]
pub struct DetCurveDisplay {
    pub fpr: Array1<f64>,
    pub fnr: Array1<f64>,
    pub estimator_name: Option<String>,
    pub pos_label: Option<String>,
}

impl DetCurveDisplay {
    /// Create a new DET curve display
    pub fn new(fpr: Array1<f64>, fnr: Array1<f64>) -> Self {
        Self {
            fpr,
            fnr,
            estimator_name: None,
            pos_label: None,
        }
    }

    /// Set estimator name for display
    pub fn with_estimator_name(mut self, name: String) -> Self {
        self.estimator_name = Some(name);
        self
    }

    /// Set positive label for display
    pub fn with_pos_label(mut self, label: String) -> Self {
        self.pos_label = Some(label);
        self
    }

    /// Create DET curve display from predictions
    ///
    /// Uses the ranking module to calculate the ROC curve, then derives DET curve.
    /// DET curve plots FPR vs FNR (False Negative Rate = 1 - TPR).
    pub fn from_predictions(
        y_true: &Array1<i32>,
        y_score: &Array1<f64>,
        pos_label: Option<String>,
        name: Option<String>,
    ) -> MetricsResult<Self> {
        use crate::ranking::roc_curve;

        // Calculate ROC curve to get FPR and TPR
        let (fpr, tpr, _thresholds) = roc_curve(y_true, y_score)?;

        // DET curve uses FNR = 1 - TPR
        let fnr: Array1<f64> = tpr.mapv(|x| 1.0 - x);

        let mut display = Self::new(fpr, fnr);

        if let Some(pos_label) = pos_label {
            display = display.with_pos_label(pos_label);
        }

        if let Some(name) = name {
            display = display.with_estimator_name(name);
        }

        Ok(display)
    }

    /// Get formatted text representation
    pub fn text(&self) -> String {
        let mut result = String::new();

        if let Some(ref name) = self.estimator_name {
            result.push_str(&format!("DET Curve for {}\n", name));
        } else {
            result.push_str("DET Curve (Detection Error Tradeoff)\n");
        }

        result.push_str("====================================\n");

        if let Some(ref label) = self.pos_label {
            result.push_str(&format!("Positive label: {}\n", label));
        }

        result.push_str(&format!("Points: {} (FPR, FNR)\n", self.fpr.len()));
        result.push_str("Lower left corner represents perfect performance\n");

        result
    }
}

impl fmt::Display for DetCurveDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}
