//! # ConnectionSettings - Trait Implementations
//!
//! This module contains trait implementations for `ConnectionSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ConnectionSettings {
    fn default() -> Self {
        Self {
            timeout_seconds: 30,
            max_retries: 3,
            retry_interval_seconds: 5,
            use_tls: true,
            verify_ssl: true,
            pool_size: 5,
        }
    }
}

