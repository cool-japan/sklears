//! # NormalizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `NormalizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NormalizationConfig;

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            batch_norm: true,
            layer_norm: false,
            instance_norm: false,
            group_norm: None,
        }
    }
}

