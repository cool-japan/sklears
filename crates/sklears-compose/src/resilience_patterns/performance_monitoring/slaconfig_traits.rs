//! # SlaConfig - Trait Implementations
//!
//! This module contains trait implementations for `SlaConfig`.
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

use super::types::SlaConfig;

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            sla_monitoring: true,
            default_slas: vec![
                "availability".to_string(), "response_time".to_string(), "throughput"
                .to_string(),
            ],
            reporting_frequency: Duration::from_secs(3600),
            violation_tracking: true,
        }
    }
}

