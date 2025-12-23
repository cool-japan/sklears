//! # ReportingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ReportingConfig`.
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

use super::types::ReportingConfig;

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            auto_reporting: true,
            report_formats: vec!["json".to_string(), "html".to_string()],
            report_frequency: Duration::from_secs(3600 * 24),
            recipients: vec!["ops-team@company.com".to_string()],
        }
    }
}

