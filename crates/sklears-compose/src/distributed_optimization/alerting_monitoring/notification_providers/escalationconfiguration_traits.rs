//! # EscalationConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `EscalationConfiguration`.
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

use super::types::EscalationConfiguration;

impl Default for EscalationConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            delay: Duration::from_secs(300),
            levels: Vec::new(),
        }
    }
}

