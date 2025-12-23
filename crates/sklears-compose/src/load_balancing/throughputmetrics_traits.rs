//! # ThroughputMetrics - Trait Implementations
//!
//! This module contains trait implementations for `ThroughputMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::execution_core::*;
use crate::resource_management::*;
use crate::performance_optimization::*;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use super::types::*;
use super::functions::*;

use super::types::ThroughputMetrics;

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            requests_per_second: 0.0,
            peak_rps: 0.0,
            average_rps: 0.0,
            backend_throughput: HashMap::new(),
        }
    }
}

