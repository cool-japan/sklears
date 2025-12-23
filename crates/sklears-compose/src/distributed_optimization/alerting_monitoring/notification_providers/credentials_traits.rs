//! # Credentials - Trait Implementations
//!
//! This module contains trait implementations for `Credentials`.
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

use super::types::Credentials;

impl Default for Credentials {
    fn default() -> Self {
        Self {
            username: None,
            password: None,
            api_key: None,
            token: None,
            client_id: None,
            client_secret: None,
            certificate: None,
            additional_fields: HashMap::new(),
        }
    }
}

