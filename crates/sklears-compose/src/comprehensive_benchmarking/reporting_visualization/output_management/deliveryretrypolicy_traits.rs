//! # DeliveryRetryPolicy - Trait Implementations
//!
//! This module contains trait implementations for `DeliveryRetryPolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DeliveryRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::seconds(30),
            backoff_multiplier: 2.0,
            max_delay: Duration::hours(1),
            jitter_factor: 0.1,
        }
    }
}

