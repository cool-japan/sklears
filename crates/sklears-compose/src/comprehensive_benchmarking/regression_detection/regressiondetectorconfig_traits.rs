//! # `RegressionDetectorConfig` - Trait Implementations
//!
//! This module contains trait implementations for `RegressionDetectorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::config_types::*;
use std::time::Duration;

use super::types::DetectionSensitivity;
use super::types_20::RegressionDetectorConfig;

impl Default for RegressionDetectorConfig {
    fn default() -> Self {
        Self {
            sensitivity: DetectionSensitivity::Medium,
            regression_thresholds: RegressionThresholds::default(),
            anomaly_threshold: 0.8,
            monitoring_period: Duration::from_secs(300),
            enable_caching: true,
            cache_size: 1000,
            continuous_monitoring: true,
        }
    }
}
