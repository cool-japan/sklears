//! # MetricsStorage - Trait Implementations
//!
//! This module contains trait implementations for `MetricsStorage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for MetricsStorage {
    fn default() -> Self {
        Self {
            storage_id: format!(
                "storage_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap().as_millis()
            ),
            raw_metrics: VecDeque::new(),
            aggregated_metrics: BTreeMap::new(),
            pattern_metrics: HashMap::new(),
            business_metrics: BusinessMetricsHistory::default(),
            system_metrics: SystemMetricsHistory::default(),
            performance_metrics: PerformanceMetricsHistory::default(),
            max_storage_size: 1_000_000,
            retention_period: Duration::from_secs(7 * 24 * 3600),
            compression_enabled: true,
            backup_settings: BackupSettings::default(),
        }
    }
}

