//! # ClusterConfig - Trait Implementations
//!
//! This module contains trait implementations for `ClusterConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ClusterConfig;
use std::time::Duration;

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(30),
            node_timeout: Duration::from_secs(120),
            job_timeout: Duration::from_secs(3600),
            max_retries: 3,
            load_threshold: 0.8,
            replication_factor: 2,
        }
    }
}
