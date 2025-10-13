//! Compressed sensing methods for multi-label classification
//!
//! This module provides compressed sensing approaches for multi-label classification,
//! which can handle high-dimensional label spaces efficiently by exploiting sparsity
//! and low-rank structure in the label matrix.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use scirs2_core::random::RandNormal;
use scirs2_core::random::{Random, rng, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Compressed Sensing Label Powerset
///
/// A compressed sensing approach for multi-label classification that reduces
/// the dimensionality of the label space while preserving the essential
/// label correlation structure.
#[derive(Debug, Clone)]
pub struct CompressedSensingLabelPowerset<S = Untrained> {
    state: S,
    compressed_dimension: usize,
    reconstruction_method: ReconstructionMethod,
    learning_rate: Float,
    max_iter: usize,
    random_state: Option<u64>,
}

/// Reconstruction methods for compressed sensing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructionMethod {
    /// L1 minimization (LASSO)
    L1,
    /// L2 minimization (Ridge)
    L2,
    /// Orthogonal Matching Pursuit
    OMP,
}

/// Trained state for CompressedSensingLabelPowerset
#[derive(Debug, Clone)]
pub struct CompressedSensingLabelPowersetTrained {
    projection_matrix: Array2<Float>,
    reconstruction_weights: Array2<Float>,
    compressed_dimension: usize,
    n_features: usize,
    n_labels: usize,
    reconstruction_method: ReconstructionMethod,
}

impl Default for CompressedSensingLabelPowerset<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CompressedSensingLabelPowerset<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for CompressedSensingLabelPowerset<Untrained> {
    type Fitted = CompressedSensingLabelPowerset<CompressedSensingLabelPowersetTrained>;

    fn fit(
        self,
        X: &ArrayView2<'_, Float>,
        y: &Array2<i32>,
    ) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.compressed_dimension >= n_labels {
            return Err(SklearsError::InvalidInput(
                "Compressed dimension must be less than number of labels".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            thread_rng()
        };

        // Create random projection matrix
        let projection_matrix = Array2::random_using(
            (self.compressed_dimension, n_labels),
            RandNormal::new(0.0, 1.0 / (n_labels as Float).sqrt()).unwrap(),
            &mut rng,
        );

        // Convert labels to float
        let y_float = y.mapv(|x| x as Float);

        // Project labels to compressed space
        let compressed_labels = projection_matrix.dot(&y_float.t()).t();

        // Learn mapping from features to compressed labels
        let mut reconstruction_weights = Array2::<Float>::zeros((n_features, self.compressed_dimension));

        match self.reconstruction_method {
            ReconstructionMethod::L2 => {
                // Simple linear regression
                for _iter in 0..self.max_iter {
                    for sample_idx in 0..n_samples {
                        let x = X.row(sample_idx);
                        let y_compressed = compressed_labels.row(sample_idx);

                        // Compute prediction
                        let prediction = x.dot(&reconstruction_weights);

                        // Compute error and update weights
                        for comp_idx in 0..self.compressed_dimension {
                            let error = prediction[comp_idx] - y_compressed[comp_idx];
                            for feat_idx in 0..n_features {
                                reconstruction_weights[[feat_idx, comp_idx]] -=
                                    self.learning_rate * error * x[feat_idx];
                            }
                        }
                    }
                }
            }
            ReconstructionMethod::L1 => {
                // L1 regularized regression (simplified)
                for _iter in 0..self.max_iter {
                    for sample_idx in 0..n_samples {
                        let x = X.row(sample_idx);
                        let y_compressed = compressed_labels.row(sample_idx);

                        let prediction = x.dot(&reconstruction_weights);

                        for comp_idx in 0..self.compressed_dimension {
                            let error = prediction[comp_idx] - y_compressed[comp_idx];
                            for feat_idx in 0..n_features {
                                let weight = reconstruction_weights[[feat_idx, comp_idx]];
                                let l1_grad = if weight > 0.0 { 1.0 } else { -1.0 };
                                reconstruction_weights[[feat_idx, comp_idx]] -=
                                    self.learning_rate * (error * x[feat_idx] + 0.01 * l1_grad);
                            }
                        }
                    }
                }
            }
            ReconstructionMethod::OMP => {
                // Simplified OMP (just use L2 for now)
                for _iter in 0..self.max_iter {
                    for sample_idx in 0..n_samples {
                        let x = X.row(sample_idx);
                        let y_compressed = compressed_labels.row(sample_idx);

                        let prediction = x.dot(&reconstruction_weights);

                        for comp_idx in 0..self.compressed_dimension {
                            let error = prediction[comp_idx] - y_compressed[comp_idx];
                            for feat_idx in 0..n_features {
                                reconstruction_weights[[feat_idx, comp_idx]] -=
                                    self.learning_rate * error * x[feat_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok(CompressedSensingLabelPowerset {
            state: CompressedSensingLabelPowersetTrained {
                projection_matrix,
                reconstruction_weights,
                compressed_dimension: self.compressed_dimension,
                n_features,
                n_labels,
                reconstruction_method: self.reconstruction_method,
            },
            compressed_dimension: self.compressed_dimension,
            reconstruction_method: self.reconstruction_method,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            random_state: self.random_state,
        })
    }
}

impl CompressedSensingLabelPowerset<Untrained> {
    /// Create a new CompressedSensingLabelPowerset
    pub fn new() -> Self {
        Self {
            state: Untrained,
            compressed_dimension: 10,
            reconstruction_method: ReconstructionMethod::L2,
            learning_rate: 0.01,
            max_iter: 100,
            random_state: None,
        }
    }

    /// Set the compressed dimension
    pub fn compressed_dimension(mut self, compressed_dimension: usize) -> Self {
        self.compressed_dimension = compressed_dimension;
        self
    }

    /// Set the reconstruction method
    pub fn reconstruction_method(mut self, method: ReconstructionMethod) -> Self {
        self.reconstruction_method = method;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for CompressedSensingLabelPowerset<CompressedSensingLabelPowersetTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let x = X.row(sample_idx);

            // Predict compressed representation
            let compressed_pred = x.dot(&self.state.reconstruction_weights);

            // Reconstruct full label vector using pseudoinverse of projection matrix
            // For simplicity, use transpose (in practice, would use pseudoinverse)
            let label_pred = self.state.projection_matrix.t().dot(&compressed_pred);

            // Threshold to get binary predictions
            for label_idx in 0..self.state.n_labels {
                predictions[[sample_idx, label_idx]] = if label_pred[label_idx] > 0.5 { 1 } else { 0 };
            }
        }

        Ok(predictions)
    }
}

impl CompressedSensingLabelPowerset<CompressedSensingLabelPowersetTrained> {
    /// Get the compressed dimension
    pub fn compressed_dimension(&self) -> usize {
        self.state.compressed_dimension
    }

    /// Get the compression ratio
    pub fn compression_ratio(&self) -> Float {
        self.state.compressed_dimension as Float / self.state.n_labels as Float
    }

    /// Get the projection matrix
    pub fn projection_matrix(&self) -> &Array2<Float> {
        &self.state.projection_matrix
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)

    #[test]
    fn test_compressed_sensing_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]];

        let cs = CompressedSensingLabelPowerset::new()
            .compressed_dimension(2)
            .reconstruction_method(ReconstructionMethod::L2);

        let trained_cs = cs.fit(&X.view(), &y).unwrap();
        let predictions = trained_cs.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 4));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));
        assert_eq!(trained_cs.compressed_dimension(), 2);
        assert!((trained_cs.compression_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compressed_sensing_methods() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1, 0, 1], [0, 1, 0]];

        let cs_l1 = CompressedSensingLabelPowerset::new()
            .compressed_dimension(2)
            .reconstruction_method(ReconstructionMethod::L1);

        let cs_l2 = CompressedSensingLabelPowerset::new()
            .compressed_dimension(2)
            .reconstruction_method(ReconstructionMethod::L2);

        let cs_omp = CompressedSensingLabelPowerset::new()
            .compressed_dimension(2)
            .reconstruction_method(ReconstructionMethod::OMP);

        let trained_l1 = cs_l1.fit(&X.view(), &y).unwrap();
        let trained_l2 = cs_l2.fit(&X.view(), &y).unwrap();
        let trained_omp = cs_omp.fit(&X.view(), &y).unwrap();

        let pred_l1 = trained_l1.predict(&X.view()).unwrap();
        let pred_l2 = trained_l2.predict(&X.view()).unwrap();
        let pred_omp = trained_omp.predict(&X.view()).unwrap();

        assert_eq!(pred_l1.dim(), (2, 3));
        assert_eq!(pred_l2.dim(), (2, 3));
        assert_eq!(pred_omp.dim(), (2, 3));
    }

    #[test]
    fn test_compressed_sensing_reproducibility() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]];

        let cs1 = CompressedSensingLabelPowerset::new()
            .compressed_dimension(2)
            .random_state(42);
        let trained_cs1 = cs1.fit(&X.view(), &y).unwrap();

        let cs2 = CompressedSensingLabelPowerset::new()
            .compressed_dimension(2)
            .random_state(42);
        let trained_cs2 = cs2.fit(&X.view(), &y).unwrap();

        // Should produce same projection matrices
        let proj1 = trained_cs1.projection_matrix();
        let proj2 = trained_cs2.projection_matrix();

        for i in 0..proj1.nrows() {
            for j in 0..proj1.ncols() {
                assert!((proj1[[i, j]] - proj2[[i, j]]).abs() < 1e-10);
            }
        }
    }
}