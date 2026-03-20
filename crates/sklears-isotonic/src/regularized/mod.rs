//! Regularized isotonic regression algorithms
//!
//! This module contains advanced regularized isotonic regression variants including
//! sparse methods, additive models, tensor algorithms, L1/L2 regularization,
//! feature selection, and diagnostic tools.

pub mod additive_isotonic;
pub mod analysis_diagnostics;
pub mod feature_selection_isotonic;
pub mod regularized_isotonic;
pub mod simd_operations;
pub mod sparse_isotonic;
pub mod tensor_isotonic;

// Re-export main types and functions for convenience
pub use additive_isotonic::{additive_isotonic_regression, AdditiveIsotonicRegression};
pub use analysis_diagnostics::{
    breakdown_point_analysis, influence_diagnostics, BreakdownPointAnalysis, InfluenceDiagnostics,
};
pub use feature_selection_isotonic::{FeatureSelectionIsotonicRegression, FeatureSelectionMethod};
pub use regularized_isotonic::{regularized_isotonic_regression, RegularizedIsotonicRegression};
pub use simd_operations::*;
pub use sparse_isotonic::{sparse_isotonic_regression, SparseIsotonicRegression};
pub use tensor_isotonic::{tensor_isotonic_regression, TensorIsotonicRegression};
