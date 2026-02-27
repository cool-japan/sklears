//! # PerformanceContext - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::PerformanceContext;

impl Default for PerformanceContext {
    fn default() -> Self {
        Self {
            memory_usage: 0,
            cpu_utilization: 0.0,
            execution_time: Duration::from_secs(0),
            bottlenecks: Vec::new(),
        }
    }
}
