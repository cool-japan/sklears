//! # LearningProgress - Trait Implementations
//!
//! This module contains trait implementations for `LearningProgress`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::fault_core::*;

use super::types::LearningProgress;

impl Default for LearningProgress {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            models_trained: 0,
            average_accuracy: 0.0,
            improvement_rate: 0.0,
            last_learning_time: None,
        }
    }
}

