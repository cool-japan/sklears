//! # TraitPerformanceAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `TraitPerformanceAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{PerformanceConfig, TraitPerformanceAnalyzer};

impl Default for TraitPerformanceAnalyzer {
    fn default() -> Self {
        Self::new(PerformanceConfig::default())
    }
}
