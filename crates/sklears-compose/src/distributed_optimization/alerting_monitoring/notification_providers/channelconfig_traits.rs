//! # ChannelConfig - Trait Implementations
//!
//! This module contains trait implementations for `ChannelConfig`.
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

use super::types::ChannelConfig;

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            auth_config: AuthConfig::default(),
            timeout: Duration::from_secs(30),
            connection_config: ConnectionConfig::default(),
            protocol_config: ProtocolConfig::default(),
            custom_headers: HashMap::new(),
            custom_parameters: HashMap::new(),
            environment_variables: HashMap::new(),
        }
    }
}

