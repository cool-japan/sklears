//! Outlier detection utilities
//!
//! This module provides comprehensive outlier detection capabilities including
//! univariate and multivariate methods, SIMD-optimized operations, and detailed
//! statistical analysis. All modules have been refactored for better maintainability
//! and comply with the 2000-line refactoring policy.

pub mod core;
pub mod detector;
pub mod simd_operations;

// Re-export main types and functions
pub use core::*;
pub use detector::OutlierDetector;

// Re-export SIMD functions for internal use
pub use simd_operations::*;
