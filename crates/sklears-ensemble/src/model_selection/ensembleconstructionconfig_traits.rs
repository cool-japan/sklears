//! # EnsembleConstructionConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnsembleConstructionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EnsembleCVStrategy, EnsembleConstructionConfig, ScoringMetric};

impl Default for EnsembleConstructionConfig {
    fn default() -> Self {
        Self {
            cv_strategy: EnsembleCVStrategy::KFold {
                n_splits: 5,
                shuffle: true,
            },
            scoring: ScoringMetric::Accuracy,
            random_state: None,
            early_stopping: false,
            patience: 3,
            min_improvement: 1e-4,
            max_ensemble_size: None,
            diversity_weight: 0.1,
        }
    }
}

