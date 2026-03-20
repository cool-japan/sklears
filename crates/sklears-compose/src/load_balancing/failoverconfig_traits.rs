//! # FailoverConfig - Trait Implementations
//!
//! This module contains trait implementations for `FailoverConfig`.
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

use super::types::FailoverConfig;

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            enable_auto_failover: true,
            failover_delay: Duration::from_secs(1),
            max_failover_attempts: 3,
            failback_config: FailbackConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

