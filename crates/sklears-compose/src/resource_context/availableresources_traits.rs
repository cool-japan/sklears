//! # AvailableResources - Trait Implementations
//!
//! This module contains trait implementations for `AvailableResources`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for AvailableResources {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get() as f32,
            memory: 8 * 1024 * 1024 * 1024,
            storage: 100 * 1024 * 1024 * 1024,
            network_bandwidth: 1024 * 1024 * 1024,
            gpu_devices: 0,
            custom: HashMap::new(),
            last_updated: SystemTime::now(),
        }
    }
}

