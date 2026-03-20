//! # AdaptationImpact - Trait Implementations
//!
//! This module contains trait implementations for `AdaptationImpact`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AdaptationImpact;

impl Default for AdaptationImpact {
    fn default() -> Self {
        Self {
            latency_change: 0.0,
            throughput_change: 0.0,
            error_rate_change: 0.0,
            cost_change: 0.0,
            overall_score: 0.0,
        }
    }
}

