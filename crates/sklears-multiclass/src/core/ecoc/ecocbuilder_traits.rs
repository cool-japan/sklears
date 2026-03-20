//! # ECOCBuilder - Trait Implementations
//!
//! This module contains trait implementations for `ECOCBuilder`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ECOCBuilder;

impl<C: Clone> Clone for ECOCBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}
