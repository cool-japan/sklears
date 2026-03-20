//! # RewardFunction - Trait Implementations
//!
//! This module contains trait implementations for `RewardFunction`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{RewardFunction, RewardNormalization, RewardShaping};

impl Default for RewardFunction {
    fn default() -> Self {
        Self {
            function_id: format!(
                "reward_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default().as_millis()
            ),
            reward_components: vec![],
            normalization_strategy: RewardNormalization::None,
            reward_shaping: RewardShaping::default(),
        }
    }
}

