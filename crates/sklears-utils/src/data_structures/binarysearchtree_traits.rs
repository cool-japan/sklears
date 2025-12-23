//! # BinarySearchTree - Trait Implementations
//!
//! This module contains trait implementations for `BinarySearchTree`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BinarySearchTree;

impl<T: Clone + PartialOrd> Default for BinarySearchTree<T> {
    fn default() -> Self {
        Self::new()
    }
}
