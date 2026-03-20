//! # AlertingConfig - Trait Implementations
//!
//! This module contains trait implementations for `AlertingConfig`.
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

use super::types::AlertingConfig;

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_rules: vec![
                "high_latency".to_string(), "high_error_rate".to_string(),
                "low_availability".to_string(),
            ],
            notification_channels: vec!["email".to_string(), "slack".to_string()],
            escalation_enabled: true,
        }
    }
}

