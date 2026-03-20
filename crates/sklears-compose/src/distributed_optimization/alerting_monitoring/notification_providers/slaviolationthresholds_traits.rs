//! # SlaViolationThresholds - Trait Implementations
//!
//! This module contains trait implementations for `SlaViolationThresholds`.
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

use super::types::SlaViolationThresholds;

impl Default for SlaViolationThresholds {
    fn default() -> Self {
        Self {
            minor_threshold: 0.95,
            major_threshold: 0.90,
            critical_threshold: 0.85,
        }
    }
}

