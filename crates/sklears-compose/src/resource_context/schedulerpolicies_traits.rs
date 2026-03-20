//! # SchedulerPolicies - Trait Implementations
//!
//! This module contains trait implementations for `SchedulerPolicies`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for SchedulerPolicies {
    fn default() -> Self {
        Self {
            primary_policy: SchedulingPolicy::Fair,
            preemption_enabled: false,
            fair_share_weights: HashMap::new(),
            priority_levels: vec![
                PriorityLevel { level : 0, weight : 1.0 }, PriorityLevel { level : 1,
                weight : 2.0 }, PriorityLevel { level : 2, weight : 4.0 },
            ],
            backfill_enabled: true,
        }
    }
}

