//! # OptimizerConfig - Trait Implementations
//!
//! This module contains trait implementations for `OptimizerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OptimizerConfig, OptimizerType};

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            gradient_clipping: Some(1.0),
            adaptive_lr: true,
        }
    }
}

