//! # EnsembleBayesianConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnsembleBayesianConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for EnsembleBayesianConfig {
    fn default() -> Self {
        Self {
            n_models: 5,
            strategy: EnsembleStrategy::BayesianAveraging,
            bootstrap_ratio: 1.0,
            base_config: BayesianMultiOutputConfig::default(),
            random_state: None,
            model_prior: Vec::new(),
        }
    }
}
