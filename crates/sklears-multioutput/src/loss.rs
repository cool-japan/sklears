//! Loss functions for neural network training
//!
//! This module provides various loss functions commonly used in neural network training,
//! including Mean Squared Error for regression and Cross-Entropy for classification.

// Use SciRS2-Core for arrays (SciRS2 Policy)
use scirs2_core::ndarray::Array2;
use sklears_core::types::Float;

/// Loss functions for neural network training
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFunction {
    /// Mean squared error for regression
    MeanSquaredError,
    /// Cross-entropy for classification
    CrossEntropy,
    /// Binary cross-entropy for multi-label classification
    BinaryCrossEntropy,
}

impl LossFunction {
    /// Compute loss between predictions and targets
    pub fn compute_loss(&self, y_pred: &Array2<Float>, y_true: &Array2<Float>) -> Float {
        match self {
            LossFunction::MeanSquaredError => {
                let diff = y_pred - y_true;
                diff.map(|x| x * x).mean().unwrap()
            }
            LossFunction::CrossEntropy => {
                let mut total_loss = 0.0;
                for i in 0..y_pred.nrows() {
                    for j in 0..y_pred.ncols() {
                        let pred = y_pred[[i, j]].clamp(1e-15, 1.0 - 1e-15); // Clip for numerical stability
                        total_loss -= y_true[[i, j]] * pred.ln();
                    }
                }
                total_loss / (y_pred.nrows() as Float)
            }
            LossFunction::BinaryCrossEntropy => {
                let mut total_loss = 0.0;
                for i in 0..y_pred.nrows() {
                    for j in 0..y_pred.ncols() {
                        let pred = y_pred[[i, j]].clamp(1e-15, 1.0 - 1e-15); // Clip for numerical stability
                        total_loss -=
                            y_true[[i, j]] * pred.ln() + (1.0 - y_true[[i, j]]) * (1.0 - pred).ln();
                    }
                }
                total_loss / (y_pred.nrows() as Float * y_pred.ncols() as Float)
            }
        }
    }
}
