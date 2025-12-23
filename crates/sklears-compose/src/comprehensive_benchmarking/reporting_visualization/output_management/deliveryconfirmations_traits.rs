//! # DeliveryConfirmations - Trait Implementations
//!
//! This module contains trait implementations for `DeliveryConfirmations`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DeliveryConfirmations {
    fn default() -> Self {
        Self {
            required: false,
            timeout: Duration::hours(1),
            retry_on_missing: true,
        }
    }
}

