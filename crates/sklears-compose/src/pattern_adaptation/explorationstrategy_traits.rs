//! # ExplorationStrategy - Trait Implementations
//!
//! This module contains trait implementations for `ExplorationStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ExplorationHistory, ExplorationSchedule, ExplorationStrategy, ExplorationStrategyType};

impl Default for ExplorationStrategy {
    fn default() -> Self {
        Self {
            strategy_type: ExplorationStrategyType::EpsilonGreedy,
            exploration_rate: 0.1,
            exploration_schedule: ExplorationSchedule::default(),
            exploration_history: ExplorationHistory::default(),
        }
    }
}

