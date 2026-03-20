//! # TrafficShapingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TrafficShapingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for TrafficShapingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ingress_limit: None,
            egress_limit: None,
            burst_size: None,
            algorithm: TrafficShapingAlgorithm::TokenBucket,
        }
    }
}

