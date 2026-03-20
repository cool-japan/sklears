//! # TrendConfig - Trait Implementations
//!
//! This module contains trait implementations for `TrendConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;
use serde::{Serialize, Deserialize};
use crate::fault_core::*;
use super::types::*;
use super::functions::*;

use super::types::TrendConfig;

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            trend_analysis: true,
            trend_window: Duration::from_secs(86400 * 3),
            prediction_horizon: Duration::from_secs(86400),
            algorithms: vec!["linear_regression".to_string(), "time_series".to_string()],
        }
    }
}

