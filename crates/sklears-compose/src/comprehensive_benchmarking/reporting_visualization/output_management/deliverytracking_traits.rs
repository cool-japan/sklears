//! # DeliveryTracking - Trait Implementations
//!
//! This module contains trait implementations for `DeliveryTracking`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DeliveryTracking {
    fn default() -> Self {
        Self {
            enabled: true,
            track_attempts: true,
            track_status: true,
            track_timing: true,
            confirmations: DeliveryConfirmations::default(),
        }
    }
}

