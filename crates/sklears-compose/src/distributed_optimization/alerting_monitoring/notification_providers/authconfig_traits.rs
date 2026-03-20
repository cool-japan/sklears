//! # AuthConfig - Trait Implementations
//!
//! This module contains trait implementations for `AuthConfig`.
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

use super::types::AuthConfig;

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            auth_type: AuthType::None,
            credentials: Credentials::default(),
            token_refresh: None,
            auth_headers: HashMap::new(),
            oauth_config: None,
            mfa_config: None,
        }
    }
}

