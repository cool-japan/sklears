//! Information-theoretic manifold learning methods
//!
//! This module implements manifold learning algorithms based on information theory,
//! including mutual information maximization, entropy-based methods, and Fisher
//! information geometry.

// Import all the sub-modules
pub mod bregman_divergence;
pub mod fisher_information;
pub mod information_bottleneck;
pub mod max_mutual_information;
pub mod natural_gradient;
pub mod utils;

// Re-export the main types for convenience
pub use bregman_divergence::{BregmanDivergenceEmbedding, BregmanDivergenceType, BregmanTrained};
pub use fisher_information::{FIETrained, FisherInformationEmbedding};
pub use information_bottleneck::{IBTrained, InformationBottleneck};
pub use max_mutual_information::{MMITrained, MaxMutualInformation};
pub use natural_gradient::{NaturalGradientEmbedding, NaturalGradientTrained};

// Include tests module
#[allow(non_snake_case)]
#[cfg(test)]
mod tests;
