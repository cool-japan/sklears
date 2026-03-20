//! # WeightedGraph - Trait Implementations
//!
//! This module contains trait implementations for `WeightedGraph`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::WeightedGraph;
use std::hash::Hash;

impl<T: Clone + Eq + Hash, W: Clone + PartialOrd> Default for WeightedGraph<T, W> {
    fn default() -> Self {
        Self::new()
    }
}
