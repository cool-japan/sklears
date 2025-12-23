//! # SlaMonitoring - Trait Implementations
//!
//! This module contains trait implementations for `SlaMonitoring`.
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

use super::types::SlaMonitoring;

impl Default for SlaMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(60),
            violation_thresholds: SlaViolationThresholds::default(),
            remediation_actions: Vec::new(),
        }
    }
}

