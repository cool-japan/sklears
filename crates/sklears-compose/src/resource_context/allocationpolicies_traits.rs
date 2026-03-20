//! # AllocationPolicies - Trait Implementations
//!
//! This module contains trait implementations for `AllocationPolicies`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for AllocationPolicies {
    fn default() -> Self {
        Self {
            default_policy: AllocationPolicy::FirstFit,
            cpu_policy: CpuAllocationPolicy::Proportional,
            memory_policy: MemoryAllocationPolicy::Strict,
            storage_policy: StorageAllocationPolicy::OnDemand,
            network_policy: NetworkAllocationPolicy::Shared,
            overcommit: OvercommitPolicies::default(),
        }
    }
}

