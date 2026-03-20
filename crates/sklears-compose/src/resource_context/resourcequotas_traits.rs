//! # ResourceQuotas - Trait Implementations
//!
//! This module contains trait implementations for `ResourceQuotas`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self {
            cpu_quota: None,
            memory_quota: None,
            storage_quota: None,
            network_quota: None,
            gpu_quota: None,
            custom_quotas: HashMap::new(),
            quota_period: Duration::from_secs(24 * 60 * 60),
            reset_policy: QuotaResetPolicy::Daily,
        }
    }
}

