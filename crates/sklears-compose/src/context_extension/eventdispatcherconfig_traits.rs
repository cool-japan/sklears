//! # EventDispatcherConfig - Trait Implementations
//!
//! This module contains trait implementations for `EventDispatcherConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::EventDispatcherConfig;

impl Default for EventDispatcherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            queue_size: 1000,
            dispatch_interval: Duration::from_millis(100),
            async_dispatching: true,
        }
    }
}

