//! # RecoveryValidation - Trait Implementations
//!
//! This module contains trait implementations for `RecoveryValidation`.
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

use super::types::RecoveryValidation;

impl Default for RecoveryValidation {
    fn default() -> Self {
        Self {
            tests: Vec::new(),
            timeout: Duration::from_secs(30),
            success_criteria: ValidationCriteria::default(),
        }
    }
}

