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
//! - Multi-head attention and transformer architectures
//! - Neural tensor decomposition with deep learning
//! - Cross-modal representation learning

pub mod attention_mechanisms;
pub mod neural_tensor_decomposition;
pub mod variational_autoencoder;

pub use attention_mechanisms::{
    AttentionActivation, AttentionConfig, AttentionLayer, AttentionOutput, AttentionType,
    CrossModalAttention, CrossModalAttentionOutput, MultiHeadAttention, TransformerDecoderBlock,
    TransformerEncoderBlock,
};
pub use neural_tensor_decomposition::{
    AttentionTensorDecomposition, NeuralActivation, NeuralParafacDecomposition, NeuralTensorConfig,
    NeuralTensorResults, NeuralTuckerDecomposition, VariationalTensorNetwork,
};
pub use variational_autoencoder::{
    ActivationFunction, CrossModalSimilarity, CrossModalVAE, VAEConfig, VAETrainingResults,
};
