//! Neural feature extraction
//!
//! This module provides comprehensive neural network-based feature extraction methods
//! through a modular architecture supporting multiple neural paradigms.

mod autoencoder_extractor;
mod neural_embedding_extractor;
mod cnn_feature_extractor;
mod attention_feature_extractor;
mod transformer_feature_extractor;
mod neural_activations;
mod neural_layers;
mod neural_utilities;
mod attention_mechanisms;
mod transformer_layers;
mod positional_encoding;
mod neural_optimizers;
mod neural_types;

pub use autoencoder_extractor::*;
pub use neural_embedding_extractor::*;
pub use cnn_feature_extractor::*;
pub use attention_feature_extractor::*;
pub use transformer_feature_extractor::*;
pub use neural_activations::*;
pub use neural_layers::*;
pub use neural_utilities::*;
pub use attention_mechanisms::*;
pub use transformer_layers::*;
pub use positional_encoding::*;
pub use neural_optimizers::*;
pub use neural_types::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests;