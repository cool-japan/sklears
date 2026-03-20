//! # BatchingCriteria - Trait Implementations
//!
//! This module contains trait implementations for `BatchingCriteria`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BatchingCriteria {
    fn default() -> Self {
        Self {
            group_by_destination: true,
            group_by_format: false,
            group_by_user: true,
            group_by_time_window: Some(Duration::minutes(15)),
        }
    }
}

