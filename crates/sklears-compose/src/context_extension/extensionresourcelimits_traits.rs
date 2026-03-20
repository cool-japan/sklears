//! # ExtensionResourceLimits - Trait Implementations
//!
//! This module contains trait implementations for `ExtensionResourceLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExtensionResourceLimits;

impl Default for ExtensionResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 256 * 1024 * 1024,
            max_cpu_time: Duration::from_secs(60),
            max_file_handles: 100,
            max_network_connections: 10,
            max_threads: 4,
            max_disk_usage: 100 * 1024 * 1024,
        }
    }
}

