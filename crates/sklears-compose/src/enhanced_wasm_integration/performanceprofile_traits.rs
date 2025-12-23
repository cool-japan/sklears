//! # PerformanceProfile - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceProfile`.
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

use super::types::PerformanceProfile;

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self {
            execution_profile: ExecutionProfile {
                avg_cpu_time: Duration::from_millis(0),
                peak_cpu_usage: 0.0,
                call_frequency: HashMap::new(),
                hot_paths: Vec::new(),
            },
            memory_profile: MemoryProfile {
                peak_memory: 0,
                avg_memory: 0,
                allocation_patterns: Vec::new(),
                gc_frequency: 0.0,
            },
            io_profile: IoProfile {
                transfer_rates: HashMap::new(),
                io_latency: Duration::from_millis(0),
                bandwidth_utilization: 0.0,
            },
            scaling_profile: ScalingProfile {
                parallel_efficiency: 0.0,
                optimal_threads: 1,
                scaling_overhead: 0.0,
            },
        }
    }
}
