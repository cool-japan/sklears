//! # SchedulingConfig - Trait Implementations
//!
//! This module contains trait implementations for `SchedulingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fair,
            priority: 0,
            cpu_affinity: None,
            memory_affinity: None,
            preemption: PreemptionPolicy::NonPreemptive,
        }
    }
}

