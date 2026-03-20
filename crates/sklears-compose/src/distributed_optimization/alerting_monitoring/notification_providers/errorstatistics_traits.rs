//! # ErrorStatistics - Trait Implementations
//!
//! This module contains trait implementations for `ErrorStatistics`.
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

use super::types::ErrorStatistics;

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_type: HashMap::new(),
            recent_error_rate: 0.0,
            last_error: None,
            error_trend: ErrorTrend::Stable,
        }
    }
}

