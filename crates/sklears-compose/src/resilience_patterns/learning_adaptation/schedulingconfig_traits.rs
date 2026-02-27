//! # SchedulingConfig - Trait Implementations
//!
//! This module contains trait implementations for `SchedulingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime, Instant};
use crate::fault_core::*;

use super::types::SchedulingConfig;

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scheduling_algorithm: "priority_queue".to_string(),
            max_concurrent_tasks: 10,
            task_timeout: Duration::from_secs(3600),
        }
    }
}

