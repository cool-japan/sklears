//! # StorageManager - Trait Implementations
//!
//! This module contains trait implementations for `StorageManager`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageManager {
    fn default() -> Self {
        Self {
            storage_pools: vec![
                StoragePool { pool_id : "default".to_string(), pool_type :
                StoragePoolType::Local, capacity : StorageCapacity::Unlimited,
                performance_tier : PerformanceTier::Standard, redundancy :
                RedundancyLevel::None, },
            ],
            optimization: StorageOptimization::default(),
            monitoring: StorageMonitoring::default(),
            quotas: StorageQuotas::default(),
        }
    }
}

