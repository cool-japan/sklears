//! # BackendPerformance - Trait Implementations
//!
//! This module contains trait implementations for `BackendPerformance`.
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

use super::types::BackendPerformance;

impl Default for BackendPerformance {
    fn default() -> Self {
        Self {
            avg_response_time: Duration::from_millis(0),
            p95_response_time: Duration::from_millis(0),
            p99_response_time: Duration::from_millis(0),
            success_rate: 1.0,
            error_rate: 0.0,
            throughput: 0.0,
            quality_score: 1.0,
        }
    }
}

