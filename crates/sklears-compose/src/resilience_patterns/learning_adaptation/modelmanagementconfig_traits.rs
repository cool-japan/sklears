//! # ModelManagementConfig - Trait Implementations
//!
//! This module contains trait implementations for `ModelManagementConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::fault_core::*;

use super::types::ModelManagementConfig;

impl Default for ModelManagementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_versioning: true,
            auto_model_selection: true,
            model_performance_threshold: 0.7,
        }
    }
}

