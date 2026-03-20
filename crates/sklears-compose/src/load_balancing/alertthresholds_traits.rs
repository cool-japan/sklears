//! # AlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `AlertThresholds`.
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

use super::types::AlertThresholds;

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            high_error_rate: 0.05,
            high_response_time: Duration::from_secs(5),
            low_availability: 0.99,
            high_resource_utilization: 0.90,
        }
    }
}

