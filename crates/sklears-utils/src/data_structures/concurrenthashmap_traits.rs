//! # ConcurrentHashMap - Trait Implementations
//!
//! This module contains trait implementations for `ConcurrentHashMap`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConcurrentHashMap;
use std::hash::Hash;

impl<K: Clone + Eq + Hash, V: Clone> Default for ConcurrentHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
