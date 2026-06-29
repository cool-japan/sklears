//! # `RegressionMetadata` - Trait Implementations
//!
//! This module contains trait implementations for `RegressionMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::RegressionMetadata;

impl Default for RegressionMetadata {
    fn default() -> Self {
        Self {
            detector_version: "1.0.0".to_string(),
            detection_parameters: HashMap::new(),
            analysis_duration: Duration::from_millis(100),
            data_quality_score: 0.85,
        }
    }
}
