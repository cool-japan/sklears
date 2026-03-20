//! # DeliveryScheduling - Trait Implementations
//!
//! This module contains trait implementations for `DeliveryScheduling`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DeliveryScheduling {
    fn default() -> Self {
        Self {
            default_strategy: DeliveryStrategy::Immediate,
            priority_scheduling: true,
            load_balancing: true,
            time_windows: vec![],
        }
    }
}

