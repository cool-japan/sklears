//! # RealtimeConfig - Trait Implementations
//!
//! This module contains trait implementations for `RealtimeConfig`.
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

use super::types::RealtimeConfig;

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            realtime_monitoring: true,
            stream_processing: true,
            buffer_size: 10000,
            processing_interval: Duration::from_secs(5),
        }
    }
}

