//! # ResourceConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            cpu_config: CpuConfig::default(),
            memory_config: MemoryConfig::default(),
            storage_config: StorageConfig::default(),
            network_config: NetworkConfig::default(),
            gpu_config: None,
            custom_resources: HashMap::new(),
            limits: ResourceLimits::default(),
            quotas: ResourceQuotas::default(),
            monitoring: MonitoringConfig::default(),
            scheduling: SchedulingConfig::default(),
        }
    }
}

