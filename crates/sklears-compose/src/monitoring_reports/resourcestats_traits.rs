//! # ResourceStats - Trait Implementations
//!
//! This module contains trait implementations for `ResourceStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};
use crate::monitoring_config::*;
use crate::monitoring_metrics::*;
use crate::monitoring_events::*;
use crate::monitoring_core::*;

use super::types::ResourceStats;

impl Default for ResourceStats {
    fn default() -> Self {
        Self {
            average: 0.0,
            peak: 0.0,
            minimum: 0.0,
            p95: 0.0,
            time_above_threshold: Duration::ZERO,
            trend: TrendDirection::Unknown,
        }
    }
}

