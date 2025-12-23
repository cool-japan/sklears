//! # QosConfig - Trait Implementations
//!
//! This module contains trait implementations for `QosConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for QosConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            traffic_class: TrafficClass::BestEffort,
            dscp: None,
            priority_queuing: false,
        }
    }
}

