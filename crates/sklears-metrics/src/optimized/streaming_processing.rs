//! Streaming Metric Computation for Large-Scale Data Processing
//!
//! This module provides streaming and incremental implementations for metric computation
//! that can handle very large datasets that don't fit in memory. Key features include:
//!
//! ## Key Features
//!
//! - **Streaming Metrics**: For datasets that don't fit in memory
//! - **Incremental Updates**: Efficient single-sample and batch updates
//! - **Caching Optimization**: Avoids recomputation of expensive metrics
//! - **Classification Support**: Streaming confusion matrix with incremental updates
//! - **Memory Efficiency**: Constant memory usage regardless of dataset size

use super::OptimizedConfig;
use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive, Zero};
use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;

/// Streaming metric computation for very large datasets that don't fit in memory
pub struct StreamingMetrics<F> {
    _config: OptimizedConfig,
    n_samples: usize,
    sum_absolute_error: F,
    sum_squared_error: F,
    sum_true: F,
    sum_true_squared: F,
}

impl<F: FloatTrait + FromPrimitive + Zero> StreamingMetrics<F> {
    pub fn new(config: OptimizedConfig) -> Self {
        Self {
            _config: config,
            n_samples: 0,
            sum_absolute_error: F::zero(),
            sum_squared_error: F::zero(),
            sum_true: F::zero(),
            sum_true_squared: F::zero(),
        }
    }

    /// Update metrics with a batch of predictions
    pub fn update_batch(&mut self, y_true: &Array1<F>, y_pred: &Array1<F>) -> MetricsResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let diff = *t - *p;
            self.sum_absolute_error = self.sum_absolute_error + diff.abs();
            self.sum_squared_error = self.sum_squared_error + diff * diff;
            self.sum_true = self.sum_true + *t;
            self.sum_true_squared = self.sum_true_squared + *t * *t;
        }

        self.n_samples += y_true.len();
        Ok(())
    }

    /// Update metrics with a single prediction
    pub fn update_single(&mut self, y_true: F, y_pred: F) {
        let diff = y_true - y_pred;
        self.sum_absolute_error = self.sum_absolute_error + diff.abs();
        self.sum_squared_error = self.sum_squared_error + diff * diff;
        self.sum_true = self.sum_true + y_true;
        self.sum_true_squared = self.sum_true_squared + y_true * y_true;
        self.n_samples += 1;
    }

    /// Get current mean absolute error
    pub fn mean_absolute_error(&self) -> MetricsResult<F> {
        if self.n_samples == 0 {
            return Err(MetricsError::EmptyInput);
        }
        Ok(self.sum_absolute_error / F::from(self.n_samples).unwrap())
    }

    /// Get current mean squared error
    pub fn mean_squared_error(&self) -> MetricsResult<F> {
        if self.n_samples == 0 {
            return Err(MetricsError::EmptyInput);
        }
        Ok(self.sum_squared_error / F::from(self.n_samples).unwrap())
    }

    /// Get current RMSE
    pub fn root_mean_squared_error(&self) -> MetricsResult<F> {
        Ok(self.mean_squared_error()?.sqrt())
    }

    /// Get current R² score
    pub fn r2_score(&self) -> MetricsResult<F> {
        if self.n_samples == 0 {
            return Err(MetricsError::EmptyInput);
        }

        let n = F::from(self.n_samples).unwrap();
        let _mean_true = self.sum_true / n;
        let ss_tot = self.sum_true_squared - (self.sum_true * self.sum_true) / n;

        if ss_tot == F::zero() {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(F::one() - (self.sum_squared_error / ss_tot))
    }

    /// Reset all accumulated metrics
    pub fn reset(&mut self) {
        self.n_samples = 0;
        self.sum_absolute_error = F::zero();
        self.sum_squared_error = F::zero();
        self.sum_true = F::zero();
        self.sum_true_squared = F::zero();
    }

    /// Get number of samples processed
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

/// Incremental metric updates with caching for efficiency
pub struct IncrementalMetrics<F> {
    _config: OptimizedConfig,
    n_samples: usize,
    sum_absolute_error: F,
    sum_squared_error: F,
    sum_true: F,
    sum_true_squared: F,
    sum_pred: F,
    sum_pred_squared: F,
    last_mae: Option<F>,
    last_mse: Option<F>,
    last_r2: Option<F>,
}

impl<F: FloatTrait + FromPrimitive + Zero + Copy> IncrementalMetrics<F> {
    pub fn new(config: OptimizedConfig) -> Self {
        Self {
            _config: config,
            n_samples: 0,
            sum_absolute_error: F::zero(),
            sum_squared_error: F::zero(),
            sum_true: F::zero(),
            sum_true_squared: F::zero(),
            sum_pred: F::zero(),
            sum_pred_squared: F::zero(),
            last_mae: None,
            last_mse: None,
            last_r2: None,
        }
    }

    /// Incrementally update metrics with new sample
    pub fn update(&mut self, y_true: F, y_pred: F) {
        let diff = y_true - y_pred;
        self.sum_absolute_error = self.sum_absolute_error + diff.abs();
        self.sum_squared_error = self.sum_squared_error + diff * diff;
        self.sum_true = self.sum_true + y_true;
        self.sum_true_squared = self.sum_true_squared + y_true * y_true;
        self.sum_pred = self.sum_pred + y_pred;
        self.sum_pred_squared = self.sum_pred_squared + y_pred * y_pred;
        self.n_samples += 1;

        // Invalidate cached values
        self.last_mae = None;
        self.last_mse = None;
        self.last_r2 = None;
    }

    /// Update with a batch of samples
    pub fn update_batch(&mut self, y_true: &Array1<F>, y_pred: &Array1<F>) -> MetricsResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            self.update(true_val, pred_val);
        }

        Ok(())
    }

    /// Get current MAE (cached)
    pub fn mean_absolute_error(&mut self) -> MetricsResult<F> {
        if let Some(mae) = self.last_mae {
            return Ok(mae);
        }

        if self.n_samples == 0 {
            return Err(MetricsError::EmptyInput);
        }

        let mae = self.sum_absolute_error / F::from(self.n_samples).unwrap();
        self.last_mae = Some(mae);
        Ok(mae)
    }

    /// Get current MSE (cached)
    pub fn mean_squared_error(&mut self) -> MetricsResult<F> {
        if let Some(mse) = self.last_mse {
            return Ok(mse);
        }

        if self.n_samples == 0 {
            return Err(MetricsError::EmptyInput);
        }

        let mse = self.sum_squared_error / F::from(self.n_samples).unwrap();
        self.last_mse = Some(mse);
        Ok(mse)
    }

    /// Get current R² score (cached)
    pub fn r2_score(&mut self) -> MetricsResult<F> {
        if let Some(r2) = self.last_r2 {
            return Ok(r2);
        }

        if self.n_samples == 0 {
            return Err(MetricsError::EmptyInput);
        }

        let n = F::from(self.n_samples).unwrap();
        let mean_true = self.sum_true / n;
        let total_sum_squares = self.sum_true_squared - n * mean_true * mean_true;

        if total_sum_squares == F::zero() {
            self.last_r2 = Some(F::zero());
            return Ok(F::zero());
        }

        let r2 = F::one() - (self.sum_squared_error / total_sum_squares);
        self.last_r2 = Some(r2);
        Ok(r2)
    }

    /// Get current RMSE
    pub fn root_mean_squared_error(&mut self) -> MetricsResult<F> {
        Ok(self.mean_squared_error()?.sqrt())
    }

    /// Get all regression metrics at once
    pub fn all_metrics(&mut self) -> MetricsResult<RegressionMetrics<F>> {
        Ok(RegressionMetrics {
            mae: self.mean_absolute_error()?,
            mse: self.mean_squared_error()?,
            rmse: self.root_mean_squared_error()?,
            r2: self.r2_score()?,
            n_samples: self.n_samples,
        })
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.n_samples = 0;
        self.sum_absolute_error = F::zero();
        self.sum_squared_error = F::zero();
        self.sum_true = F::zero();
        self.sum_true_squared = F::zero();
        self.sum_pred = F::zero();
        self.sum_pred_squared = F::zero();
        self.last_mae = None;
        self.last_mse = None;
        self.last_r2 = None;
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

/// Struct to hold all regression metrics
#[derive(Debug, Clone)]
pub struct RegressionMetrics<F> {
    pub mae: F,
    pub mse: F,
    pub rmse: F,
    pub r2: F,
    pub n_samples: usize,
}

/// Streaming confusion matrix that can be updated incrementally
pub struct StreamingConfusionMatrix<T> {
    /// Sparse storage of matrix entries (true_label, pred_label) -> count
    entries: HashMap<(T, T), usize>,
    /// Set of all labels seen
    labels: BTreeSet<T>,
    /// Total number of samples processed
    n_samples: usize,
    _config: OptimizedConfig,
}

impl<T: PartialEq + Copy + Ord + Hash> StreamingConfusionMatrix<T> {
    pub fn new(config: OptimizedConfig) -> Self {
        Self {
            entries: HashMap::new(),
            labels: BTreeSet::new(),
            n_samples: 0,
            _config: config,
        }
    }

    /// Update with a batch of predictions
    pub fn update_batch(&mut self, y_true: &Array1<T>, y_pred: &Array1<T>) -> MetricsResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
            self.labels.insert(true_label);
            self.labels.insert(pred_label);
            *self.entries.entry((true_label, pred_label)).or_insert(0) += 1;
            self.n_samples += 1;
        }

        Ok(())
    }

    /// Update with a single prediction
    pub fn update_single(&mut self, y_true: T, y_pred: T) {
        self.labels.insert(y_true);
        self.labels.insert(y_pred);
        *self.entries.entry((y_true, y_pred)).or_insert(0) += 1;
        self.n_samples += 1;
    }

    /// Get confusion matrix value for specific labels
    pub fn get(&self, true_label: T, pred_label: T) -> usize {
        self.entries
            .get(&(true_label, pred_label))
            .copied()
            .unwrap_or(0)
    }

    /// Get all unique labels
    pub fn labels(&self) -> Vec<T> {
        self.labels.iter().copied().collect()
    }

    /// Get total number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get precision for a specific label
    pub fn precision(&self, label: T) -> MetricsResult<f64> {
        let tp = self.get(label, label);
        let predicted_positive: usize = self
            .labels
            .iter()
            .map(|&pred_label| self.get(label, pred_label))
            .sum();

        if predicted_positive == 0 {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(tp as f64 / predicted_positive as f64)
    }

    /// Get recall for a specific label
    pub fn recall(&self, label: T) -> MetricsResult<f64> {
        let tp = self.get(label, label);
        let actual_positive: usize = self
            .labels
            .iter()
            .map(|&true_label| self.get(true_label, label))
            .sum();

        if actual_positive == 0 {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(tp as f64 / actual_positive as f64)
    }

    /// Get F1 score for a specific label
    pub fn f1_score(&self, label: T) -> MetricsResult<f64> {
        let precision = self.precision(label)?;
        let recall = self.recall(label)?;

        if precision + recall == 0.0 {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(2.0 * precision * recall / (precision + recall))
    }

    /// Get overall accuracy
    pub fn accuracy(&self) -> f64 {
        if self.n_samples == 0 {
            return 0.0;
        }

        let correct: usize = self
            .labels
            .iter()
            .map(|&label| self.get(label, label))
            .sum();

        correct as f64 / self.n_samples as f64
    }

    /// Get all metrics for all labels
    pub fn classification_report(&self) -> HashMap<T, ClassificationMetrics> {
        let mut report = HashMap::new();

        for label in self.labels() {
            let precision = self.precision(label).unwrap_or(0.0);
            let recall = self.recall(label).unwrap_or(0.0);
            let f1 = self.f1_score(label).unwrap_or(0.0);
            let support = self
                .labels
                .iter()
                .map(|&true_label| self.get(true_label, label))
                .sum();

            report.insert(
                label,
                ClassificationMetrics {
                    precision,
                    recall,
                    f1_score: f1,
                    support,
                },
            );
        }

        report
    }

    /// Reset the matrix
    pub fn reset(&mut self) {
        self.entries.clear();
        self.labels.clear();
        self.n_samples = 0;
    }
}

impl<T: PartialEq + Copy + Ord + Hash> Default for StreamingConfusionMatrix<T> {
    fn default() -> Self {
        Self::new(OptimizedConfig::default())
    }
}

/// Classification metrics for a single class
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_streaming_metrics_basic() {
        let mut metrics = StreamingMetrics::new(OptimizedConfig::default());

        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];

        metrics.update_batch(&y_true, &y_pred).unwrap();

        assert_eq!(metrics.n_samples(), 3);
        assert!(metrics.mean_absolute_error().unwrap() < 0.2);
    }

    #[test]
    fn test_incremental_metrics_caching() {
        let mut metrics = IncrementalMetrics::new(OptimizedConfig::default());

        metrics.update(1.0, 1.1);
        metrics.update(2.0, 2.1);

        let mae1 = metrics.mean_absolute_error().unwrap();
        let mae2 = metrics.mean_absolute_error().unwrap(); // Should use cached value

        assert_eq!(mae1, mae2);
        assert_eq!(metrics.n_samples(), 2);
    }

    #[test]
    fn test_streaming_confusion_matrix() {
        let mut matrix = StreamingConfusionMatrix::new(OptimizedConfig::default());

        let y_true = array![0, 1, 0, 1];
        let y_pred = array![0, 1, 1, 1];

        matrix.update_batch(&y_true, &y_pred).unwrap();

        assert_eq!(matrix.n_samples(), 4);
        assert_eq!(matrix.get(0, 0), 1); // True negative
        assert_eq!(matrix.get(1, 1), 2); // True positive
        assert_eq!(matrix.accuracy(), 0.75);
    }
}
