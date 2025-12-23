//! # CollectionStatistics - Trait Implementations
//!
//! This module contains trait implementations for `CollectionStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for CollectionStatistics {
    fn default() -> Self {
        Self {
            total_metrics_collected: 0,
            collection_rate: 0.0,
            average_processing_time: Duration::from_millis(0),
            error_count: 0,
            last_collection_time: None,
            buffer_utilization: 0.0,
            memory_usage: 0,
            cpu_usage: 0.0,
        }
    }
}

