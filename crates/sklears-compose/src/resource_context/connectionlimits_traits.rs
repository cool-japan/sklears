//! # ConnectionLimits - Trait Implementations
//!
//! This module contains trait implementations for `ConnectionLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ConnectionLimits {
    fn default() -> Self {
        Self {
            max_connections: Some(10000),
            max_connections_per_ip: Some(100),
            connection_timeout: Some(Duration::from_secs(30)),
            keepalive_timeout: Some(Duration::from_secs(60)),
        }
    }
}

