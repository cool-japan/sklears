//! # ThrottlingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ThrottlingConfig`.
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

use super::types::ThrottlingConfig;

impl Default for ThrottlingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 0.8,
            response: ThrottlingResponse::Delay,
            recovery_time: Duration::from_secs(60),
        }
    }
}

