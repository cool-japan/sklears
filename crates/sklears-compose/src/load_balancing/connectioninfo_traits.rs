//! # ConnectionInfo - Trait Implementations
//!
//! This module contains trait implementations for `ConnectionInfo`.
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

use super::types::ConnectionInfo;

impl Default for ConnectionInfo {
    fn default() -> Self {
        Self {
            active_connections: 0,
            total_connections: 0,
            connection_rate: 0.0,
            timeout_rate: 0.0,
            avg_connection_duration: Duration::from_secs(0),
        }
    }
}

