//! Feature engineering utilities
//!
//! This module provides comprehensive feature engineering capabilities including
//! polynomial features, spline transformations, power transformations, and more.
//! All modules have been refactored for better maintainability and comply with
//! the 2000-line refactoring policy.

pub mod function_transformer;
pub mod polynomial_features;
pub mod power_transformer;
pub mod simd_features;
pub mod sparse_polynomial_features;
pub mod spline_transformer;

// Re-export main types and functions
pub use polynomial_features::{FeatureOrder, PolynomialFeatures, PolynomialFeaturesConfig};

pub use spline_transformer::{
    ExtrapolationStrategy, KnotStrategy, SplineTransformer, SplineTransformerConfig,
};

pub use power_transformer::{PowerMethod, PowerTransformer, PowerTransformerConfig};

pub use function_transformer::{transforms, FunctionTransformer, FunctionTransformerConfig};

pub use sparse_polynomial_features::{
    SparseCoefficient, SparsePolynomialFeatures, SparsePolynomialFeaturesConfig,
};

// Re-export SIMD functions for internal use
pub use simd_features::*;
