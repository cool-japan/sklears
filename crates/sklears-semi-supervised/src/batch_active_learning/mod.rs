//! Batch active learning methods for semi-supervised learning
//!
//! This module implements various batch active learning strategies that select
//! multiple samples simultaneously for labeling, considering both uncertainty
//! and diversity to create informative batches.

mod batch_mode;
mod core_set;
mod diverse_minibatch;
mod diversity_based;
mod gradient_embedding;

pub use batch_mode::*;
pub use core_set::*;
pub use diverse_minibatch::*;
pub use diversity_based::*;
pub use gradient_embedding::*;

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
// use scirs2_core::random::rand::seq::SliceRandom as _; // Bring trait methods into scope
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BatchActiveLearningError {
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),
    #[error("Invalid diversity weight: {0}")]
    InvalidDiversityWeight(f64),
    #[error("Invalid cluster count: {0}")]
    InvalidClusterCount(usize),
    #[error("Insufficient unlabeled samples")]
    InsufficientUnlabeledSamples,
    #[error("Invalid distance metric: {0}")]
    InvalidDistanceMetric(String),
    #[error("Matrix operation failed: {0}")]
    MatrixOperationFailed(String),
    #[error("Core-set computation failed: {0}")]
    CoreSetComputationFailed(String),
}

impl From<BatchActiveLearningError> for SklearsError {
    fn from(err: BatchActiveLearningError) -> Self {
        SklearsError::FitError(err.to_string())
    }
}
