//! # ExpectedResponse - Trait Implementations
//!
//! This module contains trait implementations for `ExpectedResponse`.
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

use super::types::ExpectedResponse;

impl Default for ExpectedResponse {
    fn default() -> Self {
        Self {
            status_code: Some(200),
            body: None,
            headers: HashMap::new(),
            max_response_time: Duration::from_secs(5),
        }
    }
}

