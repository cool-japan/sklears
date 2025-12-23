//! # MonitorConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `MonitorConfiguration`.
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

use super::types::MonitorConfiguration;

impl Default for MonitorConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(60),
            performance_interval: Duration::from_secs(30),
            retention_period: Duration::from_secs(86400),
        }
    }
}

