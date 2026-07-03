//! # `SecurityMetricsConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SecurityMetricsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::SecurityMetricsConfig;
use super::types_9::MetricType;

impl Default for SecurityMetricsConfig {
    fn default() -> Self {
        let mut collection_intervals = HashMap::new();
        collection_intervals.insert(MetricType::Vulnerability, Duration::from_secs(3600));
        collection_intervals.insert(MetricType::Threat, Duration::from_secs(300));
        collection_intervals.insert(MetricType::Risk, Duration::from_secs(86400));
        collection_intervals.insert(MetricType::Compliance, Duration::from_secs(21600));
        collection_intervals.insert(MetricType::Performance, Duration::from_secs(60));
        collection_intervals.insert(MetricType::Operational, Duration::from_secs(1800));
        let mut retention_policies = HashMap::new();
        retention_policies.insert(MetricType::Vulnerability, Duration::from_secs(86400 * 365));
        retention_policies.insert(MetricType::Threat, Duration::from_secs(86400 * 180));
        retention_policies.insert(MetricType::Risk, Duration::from_secs(86400 * 1095));
        retention_policies.insert(MetricType::Compliance, Duration::from_secs(86400 * 2555));
        retention_policies.insert(MetricType::Performance, Duration::from_secs(86400 * 90));
        retention_policies.insert(MetricType::Operational, Duration::from_secs(86400 * 365));
        let mut quality_thresholds = HashMap::new();
        quality_thresholds.insert("data_completeness".to_string(), 0.95);
        quality_thresholds.insert("data_accuracy".to_string(), 0.98);
        quality_thresholds.insert("data_timeliness".to_string(), 0.90);
        Self {
            collection_intervals,
            retention_policies,
            quality_thresholds,
            alerting_enabled: true,
            real_time_processing: true,
            anomaly_detection_sensitivity: 0.85,
            trend_analysis_window: Duration::from_secs(86400 * 30),
            benchmarking_enabled: true,
            dashboard_refresh_rate: Duration::from_secs(60),
        }
    }
}
