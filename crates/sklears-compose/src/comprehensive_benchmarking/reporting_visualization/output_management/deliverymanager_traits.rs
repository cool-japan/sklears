//! # DeliveryManager - Trait Implementations
//!
//! This module contains trait implementations for `DeliveryManager`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DeliveryManager {
    fn default() -> Self {
        Self {
            delivery_strategies: vec![
                DeliveryStrategy::Immediate,
                DeliveryStrategy::Batched(BatchDeliveryConfig::default()),
            ],
            scheduling: DeliveryScheduling::default(),
            tracking: DeliveryTracking::default(),
            failure_handling: DeliveryFailureHandling::default(),
        }
    }
}

