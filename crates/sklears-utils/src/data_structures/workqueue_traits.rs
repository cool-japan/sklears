//! # WorkQueue - Trait Implementations
//!
//! This module contains trait implementations for `WorkQueue`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::WorkQueue;

impl<T: Clone> Default for WorkQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}
