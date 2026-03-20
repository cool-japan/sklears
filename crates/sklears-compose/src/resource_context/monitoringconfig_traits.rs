//! # MonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(7 * 24 * 60 * 60),
            enable_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            granularity: MonitoringGranularity::Standard,
        }
    }
}

