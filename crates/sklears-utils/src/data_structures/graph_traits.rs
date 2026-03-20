//! # Graph - Trait Implementations
//!
//! This module contains trait implementations for `Graph`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Graph;
use std::hash::Hash;

impl<T: Clone + Eq + Hash> Default for Graph<T> {
    fn default() -> Self {
        Self::new()
    }
}
