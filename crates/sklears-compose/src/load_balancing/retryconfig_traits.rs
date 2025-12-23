//! # RetryConfig - Trait Implementations
//!
//! This module contains trait implementations for `RetryConfig`.
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

use super::types::RetryConfig;

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            backoff_strategy: BackoffStrategy::Exponential {
                base: 2.0,
                max_delay: Duration::from_secs(30),
            },
            jitter: true,
        }
    }
}

