//! # BiasVarianceConfig - Trait Implementations
//!
//! This module contains trait implementations for `BiasVarianceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BiasVarianceConfig, ModelSelectionLossFunction};

impl Default for BiasVarianceConfig {
    fn default() -> Self {
        Self {
            n_bootstrap_samples: 100,
            random_state: None,
            bootstrap_size: 1.0,
            compute_sample_level: false,
            loss_function: ModelSelectionLossFunction::SquaredLoss,
        }
    }
}

