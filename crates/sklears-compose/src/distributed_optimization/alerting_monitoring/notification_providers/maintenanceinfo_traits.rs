//! # MaintenanceInfo - Trait Implementations
//!
//! This module contains trait implementations for `MaintenanceInfo`.
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

use super::types::MaintenanceInfo;

impl Default for MaintenanceInfo {
    fn default() -> Self {
        Self {
            maintenance_scheduled: false,
            maintenance_start: None,
            maintenance_end: None,
            maintenance_reason: None,
            alternative_channels: Vec::new(),
            maintenance_type: MaintenanceType::Scheduled,
            impact_level: MaintenanceImpact::None,
        }
    }
}

