//! # ConnectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConnectionConfig`.
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

use super::types::ConnectionConfig;

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            pool_size: 10,
            keep_alive: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            max_redirects: 5,
            enable_compression: true,
            http_version: HttpVersion::Http2,
            proxy_config: None,
        }
    }
}

