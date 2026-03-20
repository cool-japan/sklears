//! # HealthThresholds - Trait Implementations
//!
//! This module contains trait implementations for `HealthThresholds`.
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

use super::types::HealthThresholds;

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            response_time: Duration::from_secs(5),
            error_rate: 0.05,
            availability: 0.95,
        }
    }
}

