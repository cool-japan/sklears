//! # BatchDeliveryConfig - Trait Implementations
//!
//! This module contains trait implementations for `BatchDeliveryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BatchDeliveryConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            batch_timeout: Duration::minutes(30),
            batching_criteria: BatchingCriteria::default(),
        }
    }
}

