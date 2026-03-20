//! # RegularizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `RegularizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RegularizationConfig;

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_regularization: 0.0,
            l2_regularization: 0.001,
            dropout_rate: 0.2,
            batch_normalization: true,
        }
    }
}

