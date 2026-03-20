//! # SessionAffinityConfig - Trait Implementations
//!
//! This module contains trait implementations for `SessionAffinityConfig`.
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

use super::types::SessionAffinityConfig;

impl Default for SessionAffinityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: AffinityMethod::Cookie {
                name: "SKLEARS_LB".to_string(),
                secure: true,
            },
            session_timeout: Duration::from_secs(3600),
            sticky_duration: Duration::from_secs(1800),
            failover_behavior: AffinityFailoverBehavior::FailoverToHealthy,
        }
    }
}

