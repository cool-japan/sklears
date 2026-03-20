//! # FeedbackProcessingConfig - Trait Implementations
//!
//! This module contains trait implementations for `FeedbackProcessingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime, Instant};
use crate::fault_core::*;

use super::types::FeedbackProcessingConfig;

impl Default for FeedbackProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            processing_frequency: Duration::from_secs(300),
            feedback_retention_period: Duration::from_secs(86400 * 30),
        }
    }
}

