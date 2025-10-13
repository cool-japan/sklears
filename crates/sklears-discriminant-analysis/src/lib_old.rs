//! Discriminant Analysis Algorithms
//!
//! This module provides comprehensive discriminant analysis algorithms including Linear,
//! Quadratic, and Mixture Discriminant Analysis through a modular architecture supporting
//! multiple statistical learning paradigms.
//!
//! ## Architecture
//!
//! The discriminant analysis system is organized into focused modules:
//! - **Linear Discriminant Analysis**: Fisher's linear discriminant with dimensionality reduction
//! - **Quadratic Discriminant Analysis**: Quadratic decision boundaries with class-specific covariances
//! - **Mixture Discriminant Analysis**: Gaussian mixture models for complex decision boundaries
//! - **Discriminant Types**: Common type definitions and configurations
//! - **Discriminant Utils**: Shared utilities and mathematical operations
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use sklears_discriminant_analysis::*;
//! use scirs2_core::ndarray::Array2;
//!
//! // Linear Discriminant Analysis
//! let lda = LinearDiscriminantAnalysis::new()
//!     .n_components(2)
//!     .solver("svd");
//!
//! // Quadratic Discriminant Analysis
//! let qda = QuadraticDiscriminantAnalysis::new()
//!     .reg_param(0.01);
//!
//! // Mixture Discriminant Analysis
//! let mda = MixtureDiscriminantAnalysis::new()
//!     .n_components(3)
//!     .covariance_type("full");
//!
//! let X = Array2::from_shape_vec((100, 4), data)?;
//! let y = labels;
//!
//! let trained_lda = lda.fit(&X, &y)?;
//! let predictions = trained_lda.predict(&X_test)?;
//! ```

#![warn(missing_docs)]

mod linear_discriminant_analysis;
mod quadratic_discriminant_analysis;
mod mixture_discriminant_analysis;
mod discriminant_types;
mod discriminant_utils;

pub use linear_discriminant_analysis::*;
pub use quadratic_discriminant_analysis::*;
pub use mixture_discriminant_analysis::*;
pub use discriminant_types::*;
pub use discriminant_utils::*;
