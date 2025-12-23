//! # BackendConfig - Trait Implementations
//!
//! This module contains trait implementations for `BackendConfig`.
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

use super::types::BackendConfig;

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            keep_alive: true,
            pool_size: 10,
            tls_config: None,
        }
    }
}

