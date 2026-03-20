//! Common types and utilities for tensor methods

use scirs2_core::ndarray::Array2;
use sklears_core::types::Float;
use std::marker::PhantomData;

/// Initialization methods for tensor decomposition
#[derive(Debug, Clone)]
pub enum TensorInitMethod {
    /// Random initialization
    Random,
    /// SVD-based initialization
    SVD,
    /// User-provided initialization
    Custom(Vec<Array2<Float>>),
}

/// Marker type for untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Marker type for trained state
#[derive(Debug, Clone)]
pub struct Trained;

impl Default for TensorInitMethod {
    fn default() -> Self {
        TensorInitMethod::Random
    }
}
