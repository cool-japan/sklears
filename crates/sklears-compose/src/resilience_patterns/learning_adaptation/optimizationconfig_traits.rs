//! # OptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime, Instant};
use crate::fault_core::*;

use super::types::OptimizationConfig;

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_optimizations: 5,
            optimization_frequency: Duration::from_secs(3600),
            risk_tolerance: 0.3,
            max_concurrent_adaptations: 3,
        }
    }
}

