//! # PerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use crate::error::{Result, SklearsComposeError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::PerformanceMetrics;

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_execution_time: Duration::from_millis(0),
            peak_memory_usage: 0,
            throughput: 0.0,
            latency_percentiles: HashMap::new(),
            error_rate: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}
