//! Dimensionality reduction-based imputation methods
//!
//! This module provides imputation strategies using dimensionality reduction techniques
//! including PCA, ICA, sparse methods, and manifold learning approaches.

pub mod ica;
pub mod manifold;
pub mod pca;
pub mod sparse;

// Re-export all the main types for convenience
pub use ica::{ICAImputer, ICAImputerTrained};
pub use manifold::{ManifoldLearningImputer, ManifoldLearningImputerTrained};
pub use pca::{PCAImputer, PCAImputerTrained};
pub use sparse::{CompressedSensingImputer, SparseImputer, SparseImputerTrained};
