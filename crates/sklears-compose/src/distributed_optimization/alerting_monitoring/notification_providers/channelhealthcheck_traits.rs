//! # ChannelHealthCheck - Trait Implementations
//!
//! This module contains trait implementations for `ChannelHealthCheck`.
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

use super::types::ChannelHealthCheck;

impl Default for ChannelHealthCheck {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(10),
            check_type: HealthCheckType::Basic,
            parameters: HashMap::new(),
            thresholds: HealthThresholds::default(),
        }
    }
}

