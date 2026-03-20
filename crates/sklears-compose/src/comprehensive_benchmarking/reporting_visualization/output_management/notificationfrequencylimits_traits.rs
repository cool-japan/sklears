//! # NotificationFrequencyLimits - Trait Implementations
//!
//! This module contains trait implementations for `NotificationFrequencyLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for NotificationFrequencyLimits {
    fn default() -> Self {
        Self {
            max_per_window: 10,
            window_duration: Duration::hours(1),
            cooldown_period: Duration::minutes(15),
        }
    }
}

