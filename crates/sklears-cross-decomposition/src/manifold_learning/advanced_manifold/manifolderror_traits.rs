//! # ManifoldError - Trait Implementations
//!
//! This module contains trait implementations for `ManifoldError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ManifoldError;

impl std::fmt::Display for ManifoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifoldError::DimensionMismatch(msg) => {
                write!(f, "Dimension mismatch: {}", msg)
            }
            ManifoldError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
            ManifoldError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            ManifoldError::ConvergenceFailure(msg) => {
                write!(f, "Convergence failure: {}", msg)
            }
            ManifoldError::InsufficientData(msg) => {
                write!(f, "Insufficient data: {}", msg)
            }
        }
    }
}

impl std::error::Error for ManifoldError {}
