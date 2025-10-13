//! Modular voting ensemble methods with high-performance SIMD implementations
//!
//! This module provides a comprehensive voting ensemble framework with:
//! - SIMD-accelerated operations (6x-10x speedup)
//! - Multiple voting strategies (hard, soft, weighted, Bayesian)
//! - Type-safe state management (Untrained/Trained)
//! - Uncertainty estimation and dynamic weight adjustment

pub mod config;
pub mod core;
pub mod ensemble;
pub mod simd_ops;
pub mod strategies;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod tests;

// Re-export main types and functions
pub use config::{
    EnsembleSizeAnalysis, EnsembleSizeRecommendations, UncertaintyMethod, VotingClassifierConfig,
    VotingStrategy,
};
pub use core::{TrainedVotingClassifier, VotingClassifier};
pub use ensemble::{EnsembleMember, MockEstimator};
pub use simd_ops::{
    simd_adaptive_weights, simd_add_f32, simd_aggregate_probabilities, simd_argmax_f32,
    simd_bayesian_averaging, simd_bootstrap_aggregate, simd_calculate_ranks, simd_confidence_f32,
    simd_confidence_weighted_voting, simd_ensemble_disagreement, simd_entropy_f32,
    simd_entropy_weighted_voting, simd_hard_voting_weighted, simd_matrix_vector_multiply,
    simd_mean_f32, simd_normalize_f32, simd_scale_f32, simd_soft_voting_weighted, simd_sum_f32,
    simd_variance_f32, simd_variance_weighted_voting, simd_weighted_sum_f32,
};
pub use strategies::{
    consensus_voting, entropy_f32, mean_f32, meta_voting, variance_f32, weighted_average_f32,
};
