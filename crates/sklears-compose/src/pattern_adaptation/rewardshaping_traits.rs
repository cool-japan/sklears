//! # RewardShaping - Trait Implementations
//!
//! This module contains trait implementations for `RewardShaping`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RewardShaping;

impl Default for RewardShaping {
    fn default() -> Self {
        Self {
            potential_function: None,
            intrinsic_motivation: false,
            curiosity_bonus: 0.0,
            novelty_bonus: 0.0,
        }
    }
}

