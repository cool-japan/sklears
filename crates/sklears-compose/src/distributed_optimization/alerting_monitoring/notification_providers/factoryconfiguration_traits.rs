//! # FactoryConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `FactoryConfiguration`.
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

use super::types::FactoryConfiguration;

impl Default for FactoryConfiguration {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            enable_caching: true,
            max_concurrent_creations: 10,
            validation_enabled: true,
        }
    }
}

