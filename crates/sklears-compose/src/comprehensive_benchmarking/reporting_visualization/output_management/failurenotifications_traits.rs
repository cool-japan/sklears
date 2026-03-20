//! # FailureNotifications - Trait Implementations
//!
//! This module contains trait implementations for `FailureNotifications`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for FailureNotifications {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec![NotificationChannel::Log],
            frequency_limits: NotificationFrequencyLimits::default(),
        }
    }
}

