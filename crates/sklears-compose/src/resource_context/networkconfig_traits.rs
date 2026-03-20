//! # NetworkConfig - Trait Implementations
//!
//! This module contains trait implementations for `NetworkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            max_bandwidth: Some(1024 * 1024 * 1024),
            traffic_shaping: TrafficShapingConfig::default(),
            isolation: NetworkIsolationConfig::default(),
            qos: QosConfig::default(),
            connection_limits: ConnectionLimits::default(),
        }
    }
}

