//! # RLHyperparameters - Trait Implementations
//!
//! This module contains trait implementations for `RLHyperparameters`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RLHyperparameters;

impl Default for RLHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            discount_factor: 0.99,
            batch_size: 32,
            target_update_frequency: 100,
            exploration_decay: 0.995,
            replay_buffer_size: 100000,
            max_episodes: 1000,
            max_steps_per_episode: 1000,
        }
    }
}

