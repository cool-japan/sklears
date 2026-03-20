//! # LimeExplainer - Trait Implementations
//!
//! This module contains trait implementations for `LimeExplainer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DistanceMetric, LimeExplainer, PerturbationStrategy};

impl Default for LimeExplainer {
    fn default() -> Self {
        Self {
            n_samples: 5000,
            sigma: 0.25,
            distance_metric: DistanceMetric::Euclidean,
            perturbation_strategy: PerturbationStrategy::Gaussian,
            random_seed: None,
        }
    }
}

