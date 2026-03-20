//! # PerformanceConfig - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::PerformanceConfig;

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            advanced_analysis: true,
            optimization_hints: true,
            benchmarking: false,
            cross_platform: false,
            regression_detection: false,
            benchmark_samples: 100,
            analysis_timeout: Duration::from_secs(30),
        }
    }
}
