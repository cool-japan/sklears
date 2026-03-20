//! # DataContext - Trait Implementations
//!
//! This module contains trait implementations for `DataContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DataContext, DataQualityMetrics, DataStatistics, MissingValueInfo};

impl Default for DataContext {
    fn default() -> Self {
        Self {
            input_shape: Vec::new(),
            input_dtype: "unknown".to_string(),
            expected_shape: None,
            expected_dtype: None,
            data_statistics: DataStatistics {
                n_samples: 0,
                n_features: 0,
                means: Vec::new(),
                stds: Vec::new(),
                mins: Vec::new(),
                maxs: Vec::new(),
            },
            missing_values: MissingValueInfo {
                total_missing: 0,
                missing_per_feature: Vec::new(),
                missing_patterns: Vec::new(),
            },
            quality_metrics: DataQualityMetrics {
                quality_score: 0.0,
                completeness: 0.0,
                consistency: 0.0,
                validity: 0.0,
                quality_issues: Vec::new(),
            },
        }
    }
}
