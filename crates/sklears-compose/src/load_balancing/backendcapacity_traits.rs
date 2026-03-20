//! # BackendCapacity - Trait Implementations
//!
//! This module contains trait implementations for `BackendCapacity`.
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

use super::types::BackendCapacity;

impl Default for BackendCapacity {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            max_cpu_cores: 4,
            max_memory: 8 * 1024 * 1024 * 1024,
            max_bandwidth: 1_000_000_000,
            max_iops: 10000,
        }
    }
}

