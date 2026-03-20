//! # FeatureExtractionConfig - Trait Implementations
//!
//! This module contains trait implementations for `FeatureExtractionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::fault_core::*;

use super::types::FeatureExtractionConfig;

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            feature_selection: true,
            max_features: 100,
            feature_importance_threshold: 0.1,
        }
    }
}

