//! # CostLimits - Trait Implementations
//!
//! This module contains trait implementations for `CostLimits`.
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

use super::types::CostLimits;

impl Default for CostLimits {
    fn default() -> Self {
        Self {
            daily_limit: None,
            monthly_limit: None,
            per_message_limit: None,
            alert_thresholds: Vec::new(),
        }
    }
}

