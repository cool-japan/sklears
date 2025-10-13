//! Bayesian Discriminant Analysis
//!
//! This module implements Bayesian Discriminant Analysis (BDA), which incorporates
//! prior information and uncertainty quantification into discriminant analysis.
//! It uses Bayesian inference to estimate parameters and provides posterior
//! distributions over predictions.

pub mod core;
pub mod posterior;
pub mod trained;
pub mod types;

// Re-export main types for convenience
pub use core::BayesianDiscriminantAnalysis;
pub use posterior::{HierarchicalPosterior, MCMCSamples, PosteriorParameters};
pub use trained::TrainedBayesianDiscriminantAnalysis;
pub use types::{BayesianDiscriminantAnalysisConfig, InferenceMethod, PriorType};
