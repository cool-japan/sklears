//! Sparse Matrix Operations for Memory-Efficient Metric Computation
//!
//! This module provides specialized implementations for sparse data structures,
//! optimized for scenarios where most values are zero. These implementations
//! can significantly reduce memory usage and computation time for sparse datasets.
//!
//! ## Key Features
//!
//! - **Sparse Confusion Matrix**: Memory-efficient confusion matrix using HashMap storage
//! - **Sparse Vector Metrics**: MAE, MSE, and cosine similarity for sparse vectors
//! - **Memory Optimization**: Only stores non-zero entries to minimize memory usage
//! - **Classification Support**: Complete precision, recall, F1, and accuracy metrics
//! - **Efficient Iteration**: Specialized algorithms that skip zero entries

use crate::{MetricsError, MetricsResult};
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive, Zero};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

#[cfg(feature = "sparse")]
use sprs::{CsMat, CsVec};

/// Memory-efficient confusion matrix using sparse storage
pub struct SparseConfusionMatrix<T> {
    /// Sparse storage of matrix entries (true_label, pred_label) -> count
    entries: HashMap<(T, T), usize>,
    /// Set of all labels seen
    labels: BTreeSet<T>,
    /// Total number of samples processed
    n_samples: usize,
}

impl<T: PartialEq + Copy + Ord + Hash> SparseConfusionMatrix<T> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            labels: BTreeSet::new(),
            n_samples: 0,
        }
    }

    /// Update matrix with new predictions
    pub fn update(&mut self, y_true: &Array1<T>, y_pred: &Array1<T>) -> MetricsResult<()> {
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

    /// Update matrix with a single prediction
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

    /// Convert to dense matrix representation
    pub fn to_dense(&self) -> Array2<usize> {
        let labels: Vec<T> = self.labels.iter().copied().collect();
        let n_labels = labels.len();
        let mut matrix = Array2::zeros((n_labels, n_labels));

        for (i, &true_label) in labels.iter().enumerate() {
            for (j, &pred_label) in labels.iter().enumerate() {
                matrix[[i, j]] = self.get(true_label, pred_label);
            }
        }

        matrix
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

    /// Get support (number of true instances) for a specific label
    pub fn support(&self, label: T) -> usize {
        self.labels
            .iter()
            .map(|&true_label| self.get(true_label, label))
            .sum()
    }

    /// Get macro-averaged precision across all labels
    pub fn macro_precision(&self) -> f64 {
        let precisions: Vec<f64> = self
            .labels
            .iter()
            .filter_map(|&label| self.precision(label).ok())
            .collect();

        if precisions.is_empty() {
            0.0
        } else {
            precisions.iter().sum::<f64>() / precisions.len() as f64
        }
    }

    /// Get macro-averaged recall across all labels
    pub fn macro_recall(&self) -> f64 {
        let recalls: Vec<f64> = self
            .labels
            .iter()
            .filter_map(|&label| self.recall(label).ok())
            .collect();

        if recalls.is_empty() {
            0.0
        } else {
            recalls.iter().sum::<f64>() / recalls.len() as f64
        }
    }

    /// Get macro-averaged F1 score across all labels
    pub fn macro_f1(&self) -> f64 {
        let f1_scores: Vec<f64> = self
            .labels
            .iter()
            .filter_map(|&label| self.f1_score(label).ok())
            .collect();

        if f1_scores.is_empty() {
            0.0
        } else {
            f1_scores.iter().sum::<f64>() / f1_scores.len() as f64
        }
    }

    /// Reset the matrix
    pub fn reset(&mut self) {
        self.entries.clear();
        self.labels.clear();
        self.n_samples = 0;
    }

    /// Get number of non-zero entries (sparsity measure)
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Get sparsity ratio (1.0 - density)
    pub fn sparsity(&self) -> f64 {
        let total_possible = self.labels.len() * self.labels.len();
        if total_possible == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_possible as f64)
        }
    }
}

impl<T: PartialEq + Copy + Ord + Hash> Default for SparseConfusionMatrix<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Sparse vector metrics for compressed sparse vectors
#[cfg(feature = "sparse")]
pub struct SparseMetrics;

#[cfg(feature = "sparse")]
impl SparseMetrics {
    /// Compute mean absolute error for sparse matrices
    pub fn mean_absolute_error_sparse<F: FloatTrait + FromPrimitive + Copy>(
        y_true: &CsVec<F>,
        y_pred: &CsVec<F>,
    ) -> MetricsResult<F> {
        if y_true.dim() != y_pred.dim() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.dim()],
                actual: vec![y_pred.dim()],
            });
        }

        if y_true.dim() == 0 {
            return Err(MetricsError::EmptyInput);
        }

        let mut sum = F::zero();
        let mut processed = HashSet::new();

        // Process non-zero elements in y_true
        for (idx, &true_val) in y_true.iter() {
            let pred_val = *y_pred.get(idx).unwrap_or(&F::zero());
            sum = sum + (true_val - pred_val).abs();
            processed.insert(idx);
        }

        // Process remaining non-zero elements in y_pred
        for (idx, &pred_val) in y_pred.iter() {
            if !processed.contains(&idx) {
                let true_val = F::zero(); // y_true[idx] is implicitly zero
                sum = sum + (true_val - pred_val).abs();
            }
        }

        Ok(sum / F::from(y_true.dim()).unwrap())
    }

    /// Compute mean squared error for sparse matrices
    pub fn mean_squared_error_sparse<F: FloatTrait + FromPrimitive + Copy>(
        y_true: &CsVec<F>,
        y_pred: &CsVec<F>,
    ) -> MetricsResult<F> {
        if y_true.dim() != y_pred.dim() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.dim()],
                actual: vec![y_pred.dim()],
            });
        }

        if y_true.dim() == 0 {
            return Err(MetricsError::EmptyInput);
        }

        let mut sum = F::zero();
        let mut processed = HashSet::new();

        // Process non-zero elements in y_true
        for (idx, &true_val) in y_true.iter() {
            let pred_val = *y_pred.get(idx).unwrap_or(&F::zero());
            let diff = true_val - pred_val;
            sum = sum + diff * diff;
            processed.insert(idx);
        }

        // Process remaining non-zero elements in y_pred
        for (idx, &pred_val) in y_pred.iter() {
            if !processed.contains(&idx) {
                let true_val = F::zero(); // y_true[idx] is implicitly zero
                let diff = true_val - pred_val;
                sum = sum + diff * diff;
            }
        }

        Ok(sum / F::from(y_true.dim()).unwrap())
    }

    /// Compute cosine similarity for sparse vectors
    pub fn cosine_similarity_sparse<F: FloatTrait + FromPrimitive + Copy>(
        a: &CsVec<F>,
        b: &CsVec<F>,
    ) -> MetricsResult<F> {
        if a.dim() != b.dim() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![a.dim()],
                actual: vec![b.dim()],
            });
        }

        let mut dot_product = F::zero();
        let mut norm_a_sq = F::zero();
        let mut norm_b_sq = F::zero();
        let mut processed = HashSet::new();

        // Process non-zero elements in vector a
        for (idx, &val_a) in a.iter() {
            let val_b = *b.get(idx).unwrap_or(&F::zero());
            dot_product = dot_product + val_a * val_b;
            norm_a_sq = norm_a_sq + val_a * val_a;
            if val_b != F::zero() {
                norm_b_sq = norm_b_sq + val_b * val_b;
            }
            processed.insert(idx);
        }

        // Process remaining non-zero elements in vector b
        for (idx, &val_b) in b.iter() {
            if !processed.contains(&idx) {
                norm_b_sq = norm_b_sq + val_b * val_b;
            }
        }

        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Err(MetricsError::DivisionByZero);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Compute Jaccard similarity for sparse binary vectors
    pub fn jaccard_similarity_sparse<F: FloatTrait + FromPrimitive + Copy>(
        a: &CsVec<F>,
        b: &CsVec<F>,
    ) -> MetricsResult<F> {
        if a.dim() != b.dim() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![a.dim()],
                actual: vec![b.dim()],
            });
        }

        let mut intersection = 0;
        let mut processed = HashSet::new();

        // Count intersection (both vectors have non-zero values)
        for (idx, &val_a) in a.iter() {
            if val_a != F::zero() && b.get(idx).unwrap_or(&F::zero()) != &F::zero() {
                intersection += 1;
            }
            processed.insert(idx);
        }

        // Union is the total number of positions where either vector has non-zero values
        let mut union = a.nnz();
        for (idx, &val_b) in b.iter() {
            if !processed.contains(&idx) && val_b != F::zero() {
                union += 1;
            }
        }

        if union == 0 {
            return Ok(F::zero());
        }

        Ok(F::from(intersection).unwrap() / F::from(union).unwrap())
    }
}

/// Classification metrics for sparse matrices
#[derive(Debug, Clone)]
pub struct SparseClassificationMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

impl SparseClassificationMetrics {
    pub fn new(precision: f64, recall: f64, f1_score: f64, support: usize) -> Self {
        Self {
            precision,
            recall,
            f1_score,
            support,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_confusion_matrix() {
        let mut matrix = SparseConfusionMatrix::new();

        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 1];

        matrix.update(&y_true, &y_pred).unwrap();

        assert_eq!(matrix.get(0, 0), 2); // True 0, Pred 0
        assert_eq!(matrix.get(1, 0), 1); // True 1, Pred 0
        assert_eq!(matrix.get(2, 1), 2); // True 2, Pred 1
        assert_eq!(matrix.n_samples(), 6);

        let accuracy = matrix.accuracy();
        assert_relative_eq!(accuracy, 2.0 / 6.0, epsilon = 1e-10);

        // Test sparsity
        assert_eq!(matrix.nnz(), 4); // 4 non-zero entries
        assert!(matrix.sparsity() > 0.0);
    }

    #[test]
    fn test_sparse_confusion_matrix_metrics() {
        let mut matrix = SparseConfusionMatrix::new();

        // Perfect predictions for label 0
        let y_true = array![0, 0, 1, 1, 2, 2];
        let y_pred = array![0, 0, 1, 2, 2, 1];

        matrix.update(&y_true, &y_pred).unwrap();

        // Label 0: perfect precision and recall
        assert_relative_eq!(matrix.precision(0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(matrix.recall(0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(matrix.f1_score(0).unwrap(), 1.0, epsilon = 1e-10);

        assert_eq!(matrix.support(0), 2);
        assert!(matrix.macro_precision() > 0.0);
        assert!(matrix.macro_recall() > 0.0);
        assert!(matrix.macro_f1() > 0.0);
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_mae() {
        use sprs::CsVec;

        // Create sparse vectors: y_true = [1, 0, 3, 0], y_pred = [0, 0, 2, 1]
        let y_true = CsVec::new(4, vec![0, 2], vec![1.0, 3.0]);
        let y_pred = CsVec::new(4, vec![2, 3], vec![2.0, 1.0]);

        let mae = SparseMetrics::mean_absolute_error_sparse(&y_true, &y_pred).unwrap();

        // MAE = (|1-0| + |0-0| + |3-2| + |0-1|) / 4 = (1 + 0 + 1 + 1) / 4 = 0.75
        assert_relative_eq!(mae, 0.75, epsilon = 1e-10);
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_cosine_similarity() {
        use sprs::CsVec;

        // Create sparse vectors: a = [1, 0, 2, 0], b = [0, 0, 4, 0]
        let a = CsVec::new(4, vec![0, 2], vec![1.0, 2.0]);
        let b = CsVec::new(4, vec![2], vec![4.0]);

        let similarity = SparseMetrics::cosine_similarity_sparse(&a, &b).unwrap();

        // cos = (1*0 + 0*0 + 2*4 + 0*0) / (sqrt(1²+2²) * sqrt(4²))
        //     = 8 / (sqrt(5) * 4) = 8 / (4*sqrt(5)) = 2/sqrt(5)
        let expected = 2.0 / 5.0_f64.sqrt();
        assert_relative_eq!(similarity, expected, epsilon = 1e-10);
    }
}
