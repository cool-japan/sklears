//! # DeliveryFailureHandling - Trait Implementations
//!
//! This module contains trait implementations for `DeliveryFailureHandling`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DeliveryFailureHandling {
    fn default() -> Self {
        Self {
            retry_policy: DeliveryRetryPolicy::default(),
            notifications: FailureNotifications::default(),
            dead_letter_queue: DeadLetterQueue::default(),
        }
    }
}

