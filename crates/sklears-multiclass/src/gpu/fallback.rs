//! CPU fallback implementations for GPU operations
//!
//! This module provides CPU-based implementations that are used when GPU is not available.

use super::{GpuContext, GpuMatrixOps, GpuVotingOps};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// CPU-based voting aggregation (fallback)
pub struct CpuVotingOps;

impl GpuVotingOps for CpuVotingOps {
    fn aggregate_votes_gpu(
        &self,
        votes: &Array2<f64>,
        weights: Option<&Array1<f64>>,
        _ctx: &GpuContext,
    ) -> SklResult<Array1<i32>> {
        let (n_samples, _n_classes) = votes.dim();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = votes.row(i);

            // Apply weights if provided
            let weighted_votes = if let Some(w) = weights {
                row.iter()
                    .zip(w.iter())
                    .map(|(&vote, &weight)| vote * weight)
                    .collect::<Vec<_>>()
            } else {
                row.to_vec()
            };

            // Find class with maximum votes
            let max_idx = weighted_votes
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .ok_or_else(|| SklearsError::InvalidInput("Empty votes".to_string()))?;

            predictions[i] = max_idx as i32;
        }

        Ok(predictions)
    }

    fn aggregate_probabilities_gpu(
        &self,
        probabilities: &[Array2<f64>],
        weights: Option<&Array1<f64>>,
        _ctx: &GpuContext,
    ) -> SklResult<Array2<f64>> {
        if probabilities.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty probabilities array".to_string(),
            ));
        }

        let (n_samples, n_classes) = probabilities[0].dim();
        let n_estimators = probabilities.len();
        let mut aggregated = Array2::zeros((n_samples, n_classes));

        // Default uniform weights if not provided
        let default_weights = Array1::from_elem(n_estimators, 1.0 / n_estimators as f64);
        let weights = weights.unwrap_or(&default_weights);

        if weights.len() != n_estimators {
            return Err(SklearsError::InvalidInput(
                "Weights length must match number of estimators".to_string(),
            ));
        }

        // Aggregate probabilities with weights
        for (estimator_idx, probs) in probabilities.iter().enumerate() {
            let weight = weights[estimator_idx];
            for i in 0..n_samples {
                for j in 0..n_classes {
                    aggregated[[i, j]] += probs[[i, j]] * weight;
                }
            }
        }

        // Normalize
        for i in 0..n_samples {
            let row_sum: f64 = aggregated.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_classes {
                    aggregated[[i, j]] /= row_sum;
                }
            }
        }

        Ok(aggregated)
    }
}

/// CPU-based matrix operations (fallback)
pub struct CpuMatrixOps;

impl GpuMatrixOps for CpuMatrixOps {
    fn matmul_gpu(
        &self,
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
        _ctx: &GpuContext,
    ) -> SklResult<Array1<f64>> {
        if matrix.ncols() != vector.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix columns ({}) must match vector length ({})",
                matrix.ncols(),
                vector.len()
            )));
        }

        Ok(matrix.dot(vector))
    }

    fn batch_matmul_gpu(
        &self,
        matrices: &[Array2<f64>],
        vectors: &Array2<f64>,
        _ctx: &GpuContext,
    ) -> SklResult<Array2<f64>> {
        if matrices.is_empty() {
            return Err(SklearsError::InvalidInput("Empty matrices".to_string()));
        }

        let n_matrices = matrices.len();
        let n_vectors = vectors.nrows();

        if n_matrices != n_vectors {
            return Err(SklearsError::InvalidInput(
                "Number of matrices must match number of vectors".to_string(),
            ));
        }

        let result_dim = matrices[0].nrows();
        let mut results = Array2::zeros((n_vectors, result_dim));

        for i in 0..n_vectors {
            let vector = vectors.row(i);
            let result = matrices[i].dot(&vector.to_owned());
            for (j, &val) in result.iter().enumerate() {
                results[[i, j]] = val;
            }
        }

        Ok(results)
    }

    fn compute_distances_gpu(
        &self,
        predictions: &Array2<f64>,
        code_matrix: &Array2<i8>,
        _ctx: &GpuContext,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_estimators) = predictions.dim();
        let n_classes = code_matrix.nrows();

        if code_matrix.ncols() != n_estimators {
            return Err(SklearsError::InvalidInput(
                "Code matrix columns must match number of estimators".to_string(),
            ));
        }

        let mut distances = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for j in 0..n_classes {
                let mut distance = 0.0;
                for k in 0..n_estimators {
                    let pred = predictions[[i, k]];
                    let code = code_matrix[[j, k]] as f64;
                    distance += (pred - code).powi(2);
                }
                distances[[i, j]] = distance.sqrt();
            }
        }

        Ok(distances)
    }
}

/// CPU-based batch prediction (fallback)
pub struct CpuBatchPredict<P> {
    predictor: P,
}

impl<P> CpuBatchPredict<P> {
    pub fn new(predictor: P) -> Self {
        Self { predictor }
    }
}

// Note: Actual implementation would require a Predictor trait
// This is a placeholder structure

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_cpu_voting_ops_basic() {
        let ops = CpuVotingOps;
        let votes = array![[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 1.0, 1.0]];
        let config = super::super::GpuConfig::default();
        let ctx = super::super::GpuContext::new(config).unwrap();

        let predictions = ops.aggregate_votes_gpu(&votes, None, &ctx).unwrap();
        assert_eq!(predictions[0], 1); // Max at index 1
        assert_eq!(predictions[1], 2); // Max at index 2
        assert_eq!(predictions[2], 0); // Max at index 0
    }

    #[test]
    fn test_cpu_voting_ops_with_weights() {
        let ops = CpuVotingOps;
        let votes = array![[1.0, 2.0, 0.0], [1.0, 1.0, 1.0]];
        let weights = array![2.0, 0.5, 1.0];
        let config = super::super::GpuConfig::default();
        let ctx = super::super::GpuContext::new(config).unwrap();

        let predictions = ops
            .aggregate_votes_gpu(&votes, Some(&weights), &ctx)
            .unwrap();
        // First sample: class0=1.0*2.0=2.0, class1=2.0*0.5=1.0, class2=0.0*1.0=0.0
        // Max is class 0
        assert_eq!(predictions[0], 0);
    }

    #[test]
    fn test_cpu_matrix_ops_matmul() {
        let ops = CpuMatrixOps;
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let vector = array![1.0, 2.0];
        let config = super::super::GpuConfig::default();
        let ctx = super::super::GpuContext::new(config).unwrap();

        let result = ops.matmul_gpu(&matrix, &vector, &ctx).unwrap();
        assert_eq!(result[0], 5.0); // 1*1 + 2*2
        assert_eq!(result[1], 11.0); // 3*1 + 4*2
    }

    #[test]
    fn test_cpu_matrix_ops_distance() {
        let ops = CpuMatrixOps;
        let predictions = array![[1.0, 1.0], [0.0, 0.0]];
        let code_matrix = array![[1i8, 1i8], [0i8, 0i8]];
        let config = super::super::GpuConfig::default();
        let ctx = super::super::GpuContext::new(config).unwrap();

        let distances = ops
            .compute_distances_gpu(&predictions, &code_matrix, &ctx)
            .unwrap();
        assert_eq!(distances.dim(), (2, 2));
        // First sample matches first class perfectly
        assert!((distances[[0, 0]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_aggregate_probabilities() {
        let ops = CpuVotingOps;
        let probs1 = array![[0.7, 0.3], [0.4, 0.6]];
        let probs2 = array![[0.6, 0.4], [0.5, 0.5]];
        let probabilities = vec![probs1, probs2];
        let config = super::super::GpuConfig::default();
        let ctx = super::super::GpuContext::new(config).unwrap();

        let aggregated = ops
            .aggregate_probabilities_gpu(&probabilities, None, &ctx)
            .unwrap();
        assert_eq!(aggregated.dim(), (2, 2));

        // Check normalization
        let row0_sum: f64 = aggregated.row(0).sum();
        let row1_sum: f64 = aggregated.row(1).sum();
        assert!((row0_sum - 1.0).abs() < 1e-10);
        assert!((row1_sum - 1.0).abs() < 1e-10);
    }
}
