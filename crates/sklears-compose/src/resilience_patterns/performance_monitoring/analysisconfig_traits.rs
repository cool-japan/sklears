//! # AnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisConfig`.
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

use super::types::AnalysisConfig;

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_interval: Duration::from_secs(60),
            algorithms_enabled: vec![
                "statistical".to_string(), "machine_learning".to_string(), "rule_based"
                .to_string(),
            ],
            confidence_threshold: 0.8,
            bottleneck_detection: true,
        }
    }
}

