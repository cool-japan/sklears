//! Contrastive learning methods for semi-supervised learning
//!
//! This module implements various contrastive learning approaches that can be used
//! for semi-supervised learning by learning representations that cluster similar
//! samples together while pushing dissimilar samples apart.

mod contrastive_predictive_coding;
mod momentum_contrast;
mod simclr;
mod supervised_contrastive;

pub use contrastive_predictive_coding::*;
pub use momentum_contrast::*;
pub use simclr::*;
pub use supervised_contrastive::*;

use scirs2_core::ndarray_ext::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Random, Rng};
// use scirs2_core::random::rand::seq::SliceRandom;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, PredictProba};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ContrastiveLearningError {
    #[error("Invalid temperature parameter: {0}")]
    InvalidTemperature(f64),
    #[error("Invalid augmentation strength: {0}")]
    InvalidAugmentationStrength(f64),
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),
    #[error("Insufficient labeled samples for contrastive learning")]
    InsufficientLabeledSamples,
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    EmbeddingDimensionMismatch { expected: usize, actual: usize },
    #[error("Matrix operation failed: {0}")]
    MatrixOperationFailed(String),
}

impl From<ContrastiveLearningError> for SklearsError {
    fn from(err: ContrastiveLearningError) -> Self {
        SklearsError::FitError(err.to_string())
    }
}
