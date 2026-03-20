//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::{MaxFeatures, SplitCriterion};

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            bootstrap: true,
            oob_score: false,
            random_state: None,
            n_jobs: None,
            min_weight_fraction_leaf: 0.0,
            max_leaf_nodes: None,
            min_impurity_decrease: 0.0,
            warm_start: false,
            class_weight: ClassWeight::None,
            sampling_strategy: SamplingStrategy::Bootstrap,
        }
    }
}
