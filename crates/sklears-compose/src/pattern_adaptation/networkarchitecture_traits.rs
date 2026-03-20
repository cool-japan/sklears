//! # NetworkArchitecture - Trait Implementations
//!
//! This module contains trait implementations for `NetworkArchitecture`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ActivationFunction, LayerConfig, LayerType, NetworkArchitecture, NormalizationConfig, RegularizationConfig, WeightInitialization};

impl Default for NetworkArchitecture {
    fn default() -> Self {
        Self {
            layers: vec![
                LayerConfig { layer_type : LayerType::Dense, size : 128, dropout_rate :
                0.2, weight_initialization : WeightInitialization::Xavier, }, LayerConfig
                { layer_type : LayerType::Dense, size : 64, dropout_rate : 0.2,
                weight_initialization : WeightInitialization::Xavier, },
            ],
            activation_functions: vec![
                ActivationFunction::ReLU, ActivationFunction::ReLU
            ],
            regularization: RegularizationConfig::default(),
            normalization: NormalizationConfig::default(),
        }
    }
}

