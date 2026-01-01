//! Threshold tuning for probabilistic classifiers
//!
//! This module provides utilities for tuning decision thresholds in binary classification
//! to optimize specific metrics or constraints beyond the default 0.5 threshold.
//!
//! # Key Components
//!
//! - **FixedThresholdClassifier**: Applies a fixed decision threshold
//! - **TunedThresholdClassifierCV**: Automatically tunes threshold via cross-validation
//! - **ThresholdOptimizer**: Optimizes threshold for specific metrics
//!
//! # Use Cases
//!
//! - Imbalanced classification (optimize F1, precision, recall)
//! - Cost-sensitive learning (minimize false positives/negatives)
//! - Meeting specific precision/recall requirements
//! - ROC curve analysis and threshold selection

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, PredictProba},
    types::FloatBounds,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Metric to optimize when tuning threshold
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OptimizationMetric {
    /// Maximize F1 score
    F1,
    /// Maximize F-beta score
    FBeta(f64),
    /// Maximize precision
    Precision,
    /// Maximize recall
    Recall,
    /// Maximize balanced accuracy
    BalancedAccuracy,
    /// Minimize cost (weighted FP/FN)
    Cost {
        /// Cost of false positive
        fp_cost: f64,
        /// Cost of false negative
        fn_cost: f64,
    },
    /// Maximize Jaccard score
    Jaccard,
    /// Maximize Matthews correlation coefficient
    Matthews,
}

/// Fixed threshold classifier wrapper
///
/// Applies a fixed decision threshold to a probabilistic classifier's predictions.
#[derive(Debug, Clone)]
pub struct FixedThresholdClassifier<E> {
    /// Base estimator (must implement PredictProba)
    estimator: E,
    /// Decision threshold (default: 0.5)
    threshold: f64,
    /// Which probability column to threshold (for multiclass: use class 1)
    pos_label_idx: usize,
}

impl<E> FixedThresholdClassifier<E> {
    /// Create a new fixed threshold classifier
    pub fn new(estimator: E, threshold: f64) -> Self {
        Self {
            estimator,
            threshold,
            pos_label_idx: 1,
        }
    }

    /// Set the threshold value
    pub fn threshold(mut self, threshold: f64) -> Self {
        if threshold < 0.0 || threshold > 1.0 {
            panic!("Threshold must be between 0.0 and 1.0");
        }
        self.threshold = threshold;
        self
    }

    /// Set the positive label index for multiclass
    pub fn pos_label_idx(mut self, idx: usize) -> Self {
        self.pos_label_idx = idx;
        self
    }

    /// Get the threshold value
    pub fn get_threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the base estimator
    pub fn estimator(&self) -> &E {
        &self.estimator
    }
}

impl<'a, E, F: FloatBounds> Fit<ArrayView2<'a, F>, ArrayView1<'a, usize>>
    for FixedThresholdClassifier<E>
where
    E: Fit<ArrayView2<'a, F>, ArrayView1<'a, usize>>,
{
    type Fitted = FixedThresholdClassifier<E::Fitted>;

    fn fit(self, x: &ArrayView2<'a, F>, y: &ArrayView1<'a, usize>) -> Result<Self::Fitted> {
        let trained_estimator = self.estimator.fit(x, y)?;
        Ok(FixedThresholdClassifier {
            estimator: trained_estimator,
            threshold: self.threshold,
            pos_label_idx: self.pos_label_idx,
        })
    }
}

impl<'a, E, F: FloatBounds> Predict<ArrayView2<'a, F>, Array1<usize>>
    for FixedThresholdClassifier<E>
where
    E: PredictProba<ArrayView2<'a, F>, Array2<F>>,
{
    fn predict(&self, x: &ArrayView2<'a, F>) -> Result<Array1<usize>> {
        let probas = self.estimator.predict_proba(x)?;

        // Apply threshold to positive class probability
        let predictions = probas.map_axis(Axis(1), |row| {
            if row.len() <= self.pos_label_idx {
                return 0;
            }
            if row[self.pos_label_idx].to_f64().unwrap_or(0.0) >= self.threshold {
                1
            } else {
                0
            }
        });

        Ok(predictions)
    }
}

impl<'a, E, F: FloatBounds> PredictProba<ArrayView2<'a, F>, Array2<F>>
    for FixedThresholdClassifier<E>
where
    E: PredictProba<ArrayView2<'a, F>, Array2<F>>,
{
    fn predict_proba(&self, x: &ArrayView2<'a, F>) -> Result<Array2<F>> {
        // Just pass through the base estimator's probabilities
        self.estimator.predict_proba(x)
    }
}

use scirs2_core::ndarray::Axis;

/// Tuned threshold classifier with cross-validation
///
/// Automatically finds the optimal decision threshold by maximizing a specified
/// metric through cross-validation on the training data.
#[derive(Debug)]
pub struct TunedThresholdClassifierCV<E, C> {
    /// Base estimator
    estimator: E,
    /// Cross-validation splitter
    cv: C,
    /// Metric to optimize
    scoring: OptimizationMetric,
    /// Number of threshold values to try (linearly spaced from 0 to 1)
    n_thresholds: usize,
    /// Minimum threshold to consider
    min_threshold: f64,
    /// Maximum threshold to consider
    max_threshold: f64,
    /// Positive label index
    pos_label_idx: usize,
}

impl<E, C> TunedThresholdClassifierCV<E, C> {
    /// Create a new tuned threshold classifier
    pub fn new(estimator: E, cv: C) -> Self {
        Self {
            estimator,
            cv,
            scoring: OptimizationMetric::F1,
            n_thresholds: 100,
            min_threshold: 0.0,
            max_threshold: 1.0,
            pos_label_idx: 1,
        }
    }

    /// Set the metric to optimize
    pub fn scoring(mut self, metric: OptimizationMetric) -> Self {
        self.scoring = metric;
        self
    }

    /// Set the number of thresholds to try
    pub fn n_thresholds(mut self, n: usize) -> Self {
        self.n_thresholds = n;
        self
    }

    /// Set the threshold range
    pub fn threshold_range(mut self, min: f64, max: f64) -> Self {
        self.min_threshold = min;
        self.max_threshold = max;
        self
    }
}

/// Trained tuned threshold classifier
#[derive(Debug)]
pub struct TunedThresholdClassifierCVTrained<E> {
    /// Trained base estimator
    estimator: E,
    /// Optimal threshold found via CV
    best_threshold_: f64,
    /// Best score achieved
    best_score_: f64,
    /// All thresholds tried
    thresholds_: Vec<f64>,
    /// Scores for each threshold
    scores_: Vec<f64>,
    /// Positive label index
    pos_label_idx: usize,
}

impl<E> TunedThresholdClassifierCVTrained<E> {
    /// Get the optimal threshold
    pub fn best_threshold(&self) -> f64 {
        self.best_threshold_
    }

    /// Get the best score achieved
    pub fn best_score(&self) -> f64 {
        self.best_score_
    }

    /// Get all thresholds tried
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds_
    }

    /// Get scores for each threshold
    pub fn scores(&self) -> &[f64] {
        &self.scores_
    }
}

impl<'a, E, F: FloatBounds> Predict<ArrayView2<'a, F>, Array1<usize>>
    for TunedThresholdClassifierCVTrained<E>
where
    E: PredictProba<ArrayView2<'a, F>, Array2<F>>,
{
    fn predict(&self, x: &ArrayView2<'a, F>) -> Result<Array1<usize>> {
        let probas = self.estimator.predict_proba(x)?;

        let predictions = probas.map_axis(Axis(1), |row| {
            if row.len() <= self.pos_label_idx {
                return 0;
            }
            if row[self.pos_label_idx].to_f64().unwrap_or(0.0) >= self.best_threshold_ {
                1
            } else {
                0
            }
        });

        Ok(predictions)
    }
}

impl<'a, E, F: FloatBounds> PredictProba<ArrayView2<'a, F>, Array2<F>>
    for TunedThresholdClassifierCVTrained<E>
where
    E: PredictProba<ArrayView2<'a, F>, Array2<F>>,
{
    fn predict_proba(&self, x: &ArrayView2<'a, F>) -> Result<Array2<F>> {
        self.estimator.predict_proba(x)
    }
}

/// Helper functions for computing metrics
impl OptimizationMetric {
    /// Compute the metric value given predictions and true labels
    pub fn compute(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        match self {
            OptimizationMetric::F1 => compute_f1(y_true, y_pred),
            OptimizationMetric::FBeta(beta) => compute_fbeta(y_true, y_pred, *beta),
            OptimizationMetric::Precision => compute_precision(y_true, y_pred),
            OptimizationMetric::Recall => compute_recall(y_true, y_pred),
            OptimizationMetric::BalancedAccuracy => compute_balanced_accuracy(y_true, y_pred),
            OptimizationMetric::Cost { fp_cost, fn_cost } => {
                -compute_cost(y_true, y_pred, *fp_cost, *fn_cost)
            }
            OptimizationMetric::Jaccard => compute_jaccard(y_true, y_pred),
            OptimizationMetric::Matthews => compute_matthews(y_true, y_pred),
        }
    }
}

/// Compute confusion matrix components
fn confusion_matrix_binary(y_true: &[usize], y_pred: &[usize]) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        match (true_label, pred_label) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_count += 1,
            _ => {}
        }
    }

    (tp, tn, fp, fn_count)
}

fn compute_precision(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let (tp, _, fp, _) = confusion_matrix_binary(y_true, y_pred);
    if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    }
}

fn compute_recall(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let (tp, _, _, fn_count) = confusion_matrix_binary(y_true, y_pred);
    if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    }
}

fn compute_f1(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let precision = compute_precision(y_true, y_pred);
    let recall = compute_recall(y_true, y_pred);

    if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    }
}

fn compute_fbeta(y_true: &[usize], y_pred: &[usize], beta: f64) -> f64 {
    let precision = compute_precision(y_true, y_pred);
    let recall = compute_recall(y_true, y_pred);
    let beta_sq = beta * beta;

    if precision + recall > 0.0 {
        (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    } else {
        0.0
    }
}

fn compute_balanced_accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let (tp, tn, fp, fn_count) = confusion_matrix_binary(y_true, y_pred);

    let sensitivity = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };

    let specificity = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };

    (sensitivity + specificity) / 2.0
}

fn compute_cost(y_true: &[usize], y_pred: &[usize], fp_cost: f64, fn_cost: f64) -> f64 {
    let (_, _, fp, fn_count) = confusion_matrix_binary(y_true, y_pred);
    (fp as f64 * fp_cost) + (fn_count as f64 * fn_cost)
}

fn compute_jaccard(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let (tp, _, fp, fn_count) = confusion_matrix_binary(y_true, y_pred);
    if tp + fp + fn_count > 0 {
        tp as f64 / (tp + fp + fn_count) as f64
    } else {
        0.0
    }
}

fn compute_matthews(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let (tp, tn, fp, fn_count) = confusion_matrix_binary(y_true, y_pred);

    let numerator = (tp * tn) as f64 - (fp * fn_count) as f64;
    let denominator = ((tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count)) as f64;

    if denominator > 0.0 {
        numerator / denominator.sqrt()
    } else {
        0.0
    }
}

/// Threshold optimization results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThresholdOptimizationResult {
    /// Optimal threshold found
    pub best_threshold: f64,
    /// Best metric value achieved
    pub best_score: f64,
    /// All thresholds evaluated
    pub thresholds: Vec<f64>,
    /// Scores for each threshold
    pub scores: Vec<f64>,
}

/// Optimize threshold for a given metric
pub fn optimize_threshold<F: FloatBounds>(
    y_true: &[usize],
    y_proba: &Array2<F>,
    metric: OptimizationMetric,
    n_thresholds: usize,
    pos_label_idx: usize,
) -> Result<ThresholdOptimizationResult> {
    if y_true.len() != y_proba.nrows() {
        return Err(SklearsError::InvalidInput(
            "y_true and y_proba must have same length".to_string(),
        ));
    }

    let mut best_threshold = 0.5;
    let mut best_score = f64::NEG_INFINITY;
    let mut thresholds = Vec::with_capacity(n_thresholds);
    let mut scores = Vec::with_capacity(n_thresholds);

    // Try different thresholds
    for i in 0..n_thresholds {
        let threshold = i as f64 / (n_thresholds - 1) as f64;
        thresholds.push(threshold);

        // Apply threshold to get predictions
        let y_pred: Vec<usize> = y_proba
            .outer_iter()
            .map(|row| {
                if row.len() <= pos_label_idx {
                    return 0;
                }
                if row[pos_label_idx].to_f64().unwrap_or(0.0) >= threshold {
                    1
                } else {
                    0
                }
            })
            .collect();

        // Compute metric
        let score = metric.compute(y_true, &y_pred);
        scores.push(score);

        if score > best_score {
            best_score = score;
            best_threshold = threshold;
        }
    }

    Ok(ThresholdOptimizationResult {
        best_threshold,
        best_score,
        thresholds,
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Simple mock classifier for testing
    #[derive(Debug, Clone)]
    struct MockClassifier;

    impl<'a> Fit<ArrayView2<'a, f64>, ArrayView1<'a, usize>> for MockClassifier {
        type Fitted = MockClassifierTrained;
        fn fit(self, _x: &ArrayView2<'a, f64>, _y: &ArrayView1<'a, usize>) -> Result<Self::Fitted> {
            Ok(MockClassifierTrained {
                probas: array![[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.9, 0.1]],
            })
        }
    }

    #[derive(Debug, Clone)]
    struct MockClassifierTrained {
        probas: Array2<f64>,
    }

    impl<'a> PredictProba<ArrayView2<'a, f64>, Array2<f64>> for MockClassifierTrained {
        fn predict_proba(&self, _x: &ArrayView2<'a, f64>) -> Result<Array2<f64>> {
            Ok(self.probas.clone())
        }
    }

    #[test]
    fn test_fixed_threshold_classifier() {
        let mock = MockClassifier;
        let fixed = FixedThresholdClassifier::new(mock, 0.5);

        assert_eq!(fixed.get_threshold(), 0.5);
    }

    #[test]
    fn test_fixed_threshold_custom() {
        let mock = MockClassifier;
        let fixed = FixedThresholdClassifier::new(mock, 0.7).threshold(0.3);

        assert_eq!(fixed.get_threshold(), 0.3);
    }

    #[test]
    fn test_fixed_threshold_prediction() {
        let mock = MockClassifier;
        let fixed = FixedThresholdClassifier::new(mock, 0.5);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1, 0, 1, 0];

        let trained = fixed.fit(&x.view(), &y.view()).unwrap();
        let predictions = trained.predict(&x.view()).unwrap();

        // With threshold 0.5:
        // Row 0: [0.2, 0.8] -> 0.8 >= 0.5 -> 1
        // Row 1: [0.7, 0.3] -> 0.3 < 0.5 -> 0
        // Row 2: [0.4, 0.6] -> 0.6 >= 0.5 -> 1
        // Row 3: [0.9, 0.1] -> 0.1 < 0.5 -> 0
        assert_eq!(predictions[0], 1);
        assert_eq!(predictions[1], 0);
        assert_eq!(predictions[2], 1);
        assert_eq!(predictions[3], 0);
    }

    #[test]
    fn test_fixed_threshold_high() {
        let mock = MockClassifier;
        let fixed = FixedThresholdClassifier::new(mock, 0.7);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1, 0, 1, 0];

        let trained = fixed.fit(&x.view(), &y.view()).unwrap();
        let predictions = trained.predict(&x.view()).unwrap();

        // With threshold 0.7:
        // Row 0: [0.2, 0.8] -> 0.8 >= 0.7 -> 1
        // Row 1: [0.7, 0.3] -> 0.3 < 0.7 -> 0
        // Row 2: [0.4, 0.6] -> 0.6 < 0.7 -> 0
        // Row 3: [0.9, 0.1] -> 0.1 < 0.7 -> 0
        assert_eq!(predictions[0], 1);
        assert_eq!(predictions[1], 0);
        assert_eq!(predictions[2], 0);
        assert_eq!(predictions[3], 0);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true = vec![1, 1, 0, 0, 1, 0, 1, 0];
        let y_pred = vec![1, 0, 0, 1, 1, 0, 0, 1];

        let (tp, tn, fp, fn_count) = confusion_matrix_binary(&y_true, &y_pred);

        assert_eq!(tp, 2); // Correctly predicted positive
        assert_eq!(tn, 2); // Correctly predicted negative
        assert_eq!(fp, 2); // False positives
        assert_eq!(fn_count, 2); // False negatives
    }

    #[test]
    fn test_precision_recall() {
        let y_true = vec![1, 1, 0, 0, 1, 0];
        let y_pred = vec![1, 0, 0, 1, 1, 0];

        let precision = compute_precision(&y_true, &y_pred);
        let recall = compute_recall(&y_true, &y_pred);

        // TP=2, FP=1, FN=1
        assert!((precision - 0.666).abs() < 0.01); // 2/3
        assert!((recall - 0.666).abs() < 0.01); // 2/3
    }

    #[test]
    fn test_f1_score() {
        let y_true = vec![1, 1, 0, 0, 1, 0];
        let y_pred = vec![1, 0, 0, 1, 1, 0];

        let f1 = compute_f1(&y_true, &y_pred);
        assert!((f1 - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_balanced_accuracy() {
        let y_true = vec![1, 1, 1, 0, 0, 0];
        let y_pred = vec![1, 1, 0, 0, 0, 1];

        let balanced_acc = compute_balanced_accuracy(&y_true, &y_pred);
        // Sensitivity (TPR) = 2/3, Specificity (TNR) = 2/3
        // Balanced = (2/3 + 2/3) / 2 = 2/3
        assert!((balanced_acc - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cost_computation() {
        let y_true = vec![1, 1, 0, 0];
        let y_pred = vec![1, 0, 1, 0];
        // FP=1, FN=1

        let cost = compute_cost(&y_true, &y_pred, 10.0, 5.0);
        assert_eq!(cost, 15.0); // 1*10 + 1*5
    }

    #[test]
    fn test_jaccard_score() {
        let y_true = vec![1, 1, 0, 0, 1];
        let y_pred = vec![1, 0, 0, 1, 1];
        // TP=2, FP=1, FN=1
        // Jaccard = TP / (TP + FP + FN) = 2 / 4 = 0.5

        let jaccard = compute_jaccard(&y_true, &y_pred);
        assert_eq!(jaccard, 0.5);
    }

    #[test]
    fn test_matthews_correlation() {
        let y_true = vec![1, 1, 0, 0];
        let y_pred = vec![1, 0, 0, 1];
        // TP=1, TN=1, FP=1, FN=1

        let mcc = compute_matthews(&y_true, &y_pred);
        assert_eq!(mcc, 0.0); // Perfect disagreement
    }

    #[test]
    fn test_optimize_threshold() {
        // Create probability predictions
        let y_proba = array![
            [0.3, 0.7],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
        ];
        let y_true = vec![1, 0, 1, 0, 1, 0];

        let result = optimize_threshold(&y_true, &y_proba, OptimizationMetric::F1, 20, 1).unwrap();

        assert!(result.best_threshold >= 0.0 && result.best_threshold <= 1.0);
        assert!(result.best_score >= 0.0 && result.best_score <= 1.0);
        assert_eq!(result.thresholds.len(), 20);
        assert_eq!(result.scores.len(), 20);
    }

    #[test]
    fn test_optimize_threshold_precision() {
        // More realistic data with clear optimal threshold for precision
        // Class 1 probabilities: [0.9, 0.6, 0.7, 0.4, 0.5, 0.3]
        // True labels:           [1,   0,   1,   0,   1,   0]
        let y_proba = array![
            [0.1, 0.9], // true=1, proba=0.9
            [0.4, 0.6], // true=0, proba=0.6 (FP if threshold < 0.6)
            [0.3, 0.7], // true=1, proba=0.7
            [0.6, 0.4], // true=0, proba=0.4
            [0.5, 0.5], // true=1, proba=0.5
            [0.7, 0.3], // true=0, proba=0.3
        ];
        let y_true = vec![1, 0, 1, 0, 1, 0];

        let result =
            optimize_threshold(&y_true, &y_proba, OptimizationMetric::Precision, 50, 1).unwrap();

        // Threshold >= 0.6 gives precision=1.0 (no FP)
        // Threshold < 0.6 includes sample 1 as FP, reducing precision
        // So optimal should be >= 0.6
        assert!(
            result.best_threshold >= 0.6,
            "Expected threshold >= 0.6 for precision, got {}",
            result.best_threshold
        );
    }

    #[test]
    fn test_optimize_threshold_recall() {
        let y_proba = array![[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9],];
        let y_true = vec![1, 0, 1, 1];

        let result =
            optimize_threshold(&y_true, &y_proba, OptimizationMetric::Recall, 50, 1).unwrap();

        // Low threshold should favor recall
        assert!(result.best_threshold <= 0.5);
    }

    #[test]
    fn test_fbeta_optimization() {
        let y_proba = array![[0.2, 0.8], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7]];
        let y_true = vec![1, 0, 1, 1];

        // F2 score (beta=2, favors recall)
        let result =
            optimize_threshold(&y_true, &y_proba, OptimizationMetric::FBeta(2.0), 50, 1).unwrap();

        assert!(result.best_score >= 0.0);
        assert!(result.best_score <= 1.0);
    }

    #[test]
    fn test_cost_sensitive_optimization() {
        // More realistic data with clear cost-optimal threshold
        // Class 1 probabilities: [0.9, 0.6, 0.7, 0.4, 0.8]
        // True labels:           [1,   0,   1,   0,   1]
        let y_proba = array![
            [0.1, 0.9], // true=1, proba=0.9
            [0.4, 0.6], // true=0, proba=0.6 (costly FP if threshold < 0.6)
            [0.3, 0.7], // true=1, proba=0.7
            [0.6, 0.4], // true=0, proba=0.4
            [0.2, 0.8], // true=1, proba=0.8
        ];
        let y_true = vec![1, 0, 1, 0, 1];

        // High FP cost should push threshold higher
        let result = optimize_threshold(
            &y_true,
            &y_proba,
            OptimizationMetric::Cost {
                fp_cost: 10.0,
                fn_cost: 1.0,
            },
            50,
            1,
        )
        .unwrap();

        // Threshold < 0.6: includes sample 1 as FP → cost = 10
        // Threshold >= 0.6 and < 0.7: no FP, no FN → cost = 0
        // Threshold >= 0.7 and < 0.8: includes sample 4 as FN → cost = 1
        // Optimal is threshold in [0.6, 0.7), but algorithm returns first
        assert!(
            result.best_threshold >= 0.6,
            "Expected threshold >= 0.6, got {}",
            result.best_threshold
        );
        assert!(
            result.best_score >= -0.1,
            "Expected near-zero cost (score >= -0.1), got {}",
            result.best_score
        );
    }
}
