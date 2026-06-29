//! # `RegressionDetector` - Trait Implementations
//!
//! This module contains trait implementations for `RegressionDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_19::RegressionDetector;
use super::types_20::RegressionDetectorConfig;

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new(RegressionDetectorConfig::default())
    }
}
