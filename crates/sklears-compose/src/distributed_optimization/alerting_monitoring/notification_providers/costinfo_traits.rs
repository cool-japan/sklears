//! # CostInfo - Trait Implementations
//!
//! This module contains trait implementations for `CostInfo`.
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

use super::types::CostInfo;

impl Default for CostInfo {
    fn default() -> Self {
        Self {
            cost_model: CostModel::PerMessage,
            current_cost: 0.0,
            projected_cost: 0.0,
            cost_limits: CostLimits::default(),
            cost_tracking: CostTracking::default(),
        }
    }
}

