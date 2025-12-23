//! # MetricsExportManager - Trait Implementations
//!
//! This module contains trait implementations for `MetricsExportManager`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for MetricsExportManager {
    fn default() -> Self {
        Self {
            manager_id: format!(
                "export_mgr_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap().as_millis()
            ),
            export_configurations: HashMap::new(),
            export_schedules: HashMap::new(),
            export_history: VecDeque::new(),
            data_formatters: HashMap::new(),
            compression_engines: HashMap::new(),
            encryption_engines: HashMap::new(),
        }
    }
}

