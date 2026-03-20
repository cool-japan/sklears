//! # CostTracking - Trait Implementations
//!
//! This module contains trait implementations for `CostTracking`.
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

use super::types::CostTracking;

impl Default for CostTracking {
    fn default() -> Self {
        Self {
            enabled: false,
            calculation_frequency: Duration::from_secs(3600),
            history_retention: Duration::from_secs(86400 * 30),
            reporting_config: CostReportingConfig::default(),
        }
    }
}

