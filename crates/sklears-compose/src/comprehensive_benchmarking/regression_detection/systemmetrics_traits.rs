//! # `SystemMetrics` - Trait Implementations
//!
//! This module contains trait implementations for `SystemMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_20::SystemMetrics;

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            disk_utilization: 0.0,
            network_utilization: 0.0,
            load_average: 0.0,
        }
    }
}
