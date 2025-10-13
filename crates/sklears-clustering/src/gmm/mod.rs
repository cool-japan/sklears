//! Gaussian Mixture Models (GMM)
//!
//! GMM is a probabilistic model that assumes data points are generated from
//! a mixture of several Gaussian distributions with unknown parameters.
//! This implementation includes model selection criteria (AIC, BIC, ICL)
//! for automatic determination of optimal number of components.
//!
//! # Architecture
//!
//! The GMM implementation is organized into specialized modules:
//!
//! - `simd_operations` - High-performance SIMD-accelerated mathematical operations
//! - `types_config` - Core types, enums, and configuration structures
//! - `classical_gmm` - Traditional EM algorithm-based Gaussian Mixture Models
//! - `bayesian_gmm` - Variational Bayesian Gaussian Mixture Models
//! - `model_selection` - AIC, BIC, ICL model selection criteria
//! - `em_algorithm` - Core Expectation-Maximization algorithm implementations
//! - `tests` - Comprehensive test suite

pub mod bayesian_gmm;
pub mod classical_gmm;
pub mod em_algorithm;
pub mod model_selection;
pub mod simd_operations;
pub mod types_config;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod tests;

// Re-export core types and functionality for API compatibility
pub use types_config::{
    BayesianGaussianMixtureConfig, CovarianceType, GaussianMixtureConfig, ModelSelectionCriterion,
    ModelSelectionResult, WeightInit,
};

pub use bayesian_gmm::BayesianGaussianMixture;
pub use classical_gmm::{GaussianMixture, PredictProba};

pub use model_selection::{select_model, ModelSelector};

// Re-export SIMD operations for advanced users
pub use simd_operations::*;
