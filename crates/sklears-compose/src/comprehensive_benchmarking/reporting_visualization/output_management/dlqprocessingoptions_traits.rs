//! # DlqProcessingOptions - Trait Implementations
//!
//! This module contains trait implementations for `DlqProcessingOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for DlqProcessingOptions {
    fn default() -> Self {
        Self {
            manual_reprocessing: true,
            automatic_reprocessing: false,
            reprocessing_schedule: None,
        }
    }
}

