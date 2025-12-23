//! # CollectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `CollectionConfig`.
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

use super::types::CollectionConfig;

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(30),
            retention_period: Duration::from_secs(86400 * 7),
            metrics_enabled: vec![
                "latency".to_string(), "throughput".to_string(), "errors".to_string(),
                "availability".to_string(), "resources".to_string(),
            ],
            custom_collectors: vec![],
        }
    }
}

