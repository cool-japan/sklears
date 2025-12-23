//! # ProtocolConfig - Trait Implementations
//!
//! This module contains trait implementations for `ProtocolConfig`.
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

use super::types::ProtocolConfig;

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            options: HashMap::new(),
            custom_handlers: Vec::new(),
        }
    }
}

