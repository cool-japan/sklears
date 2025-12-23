//! # ResilienceMetricsCollector - Trait Implementations
//!
//! This module contains trait implementations for `ResilienceMetricsCollector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ResilienceMetricsCollector {
    fn default() -> Self {
        Self {
            collector_id: format!(
                "collector_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap().as_millis()
            ),
            collection_interval: Duration::from_secs(60),
            metrics_storage: Arc::new(RwLock::new(MetricsStorage::default())),
            real_time_metrics: Arc::new(Mutex::new(RealTimeMetrics::default())),
            aggregation_engine: Arc::new(
                Mutex::new(MetricsAggregationEngine::default()),
            ),
            alert_manager: Arc::new(Mutex::new(MetricsAlertManager::default())),
            export_manager: Arc::new(Mutex::new(MetricsExportManager::default())),
            pattern_metrics_registry: Arc::new(
                RwLock::new(PatternMetricsRegistry::default()),
            ),
            business_metrics_tracker: Arc::new(
                Mutex::new(BusinessMetricsTracker::default()),
            ),
            performance_analyzer: Arc::new(Mutex::new(PerformanceAnalyzer::default())),
            historical_analyzer: Arc::new(
                Mutex::new(HistoricalMetricsAnalyzer::default()),
            ),
            anomaly_detector: Arc::new(Mutex::new(MetricsAnomalyDetector::default())),
            metric_validators: vec![],
            collection_statistics: Arc::new(Mutex::new(CollectionStatistics::default())),
            is_collecting: Arc::new(AtomicU64::new(0)),
            total_metrics_collected: Arc::new(AtomicU64::new(0)),
        }
    }
}

