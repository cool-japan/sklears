//! # ValidationCriteria - Trait Implementations
//!
//! This module contains trait implementations for `ValidationCriteria`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use super::types::*;
use super::functions::*;

use super::types::ValidationCriteria;

impl Default for ValidationCriteria {
    fn default() -> Self {
        Self {
            min_success_rate: 0.9,
            max_response_time: Duration::from_secs(5),
            required_tests: Vec::new(),
        }
    }
}

