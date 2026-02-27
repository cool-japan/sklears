//! # AnalysisMetadata - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{AnalysisMetadata, PerformanceConfig};

impl Default for AnalysisMetadata {
    fn default() -> Self {
        Self {
            analyzer_version: option_env!("CARGO_PKG_VERSION")
                .unwrap_or("unknown")
                .to_string(),
            analysis_timestamp: chrono::Utc::now(),
            analysis_duration: Duration::from_millis(0),
            config_used: PerformanceConfig::default(),
        }
    }
}
