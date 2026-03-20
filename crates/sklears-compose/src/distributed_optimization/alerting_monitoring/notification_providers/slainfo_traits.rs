//! # SlaInfo - Trait Implementations
//!
//! This module contains trait implementations for `SlaInfo`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use super::types::*;
use super::functions::*;

use super::types::SlaInfo;

impl Default for SlaInfo {
    fn default() -> Self {
        Self {
            sla_level: SlaLevel::Standard,
            availability_target: 0.99,
            response_time_target: Duration::from_secs(1),
            throughput_target: 100.0,
            error_rate_target: 0.01,
            monitoring: SlaMonitoring::default(),
        }
    }
}

