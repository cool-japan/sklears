//! Deep Learning Integration for Cross-Decomposition
//!
//! This module provides deep learning methods for advanced cross-modal analysis,
//! including variational autoencoders, attention mechanisms, and neural approaches
//! to canonical correlation analysis.
//!
//! ## Methods Included
//! - Variational Autoencoders for cross-modal learning
//! - Deep Canonical Correlation Analysis
//! - Attention-based cross-modal fusion
//! - Neural tensor decomposition
//! - Cross-modal representation learning

pub mod variational_autoencoder;

pub use variational_autoencoder::{
    ActivationFunction, CrossModalSimilarity, CrossModalVAE, VAEConfig, VAETrainingResults,
};
