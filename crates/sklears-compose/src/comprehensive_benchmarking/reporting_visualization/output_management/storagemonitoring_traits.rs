//! # StorageMonitoring - Trait Implementations
//!
//! This module contains trait implementations for `StorageMonitoring`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: vec![
                StorageMetric::Capacity, StorageMetric::Usage,
                StorageMetric::Performance, StorageMetric::Health,
            ],
            alert_thresholds: StorageAlertThresholds::default(),
            monitoring_frequency: Duration::minutes(5),
        }
    }
}

