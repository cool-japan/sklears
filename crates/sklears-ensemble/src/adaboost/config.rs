//! Configuration implementations for AdaBoost

use super::types::*;
use sklears_core::types::Float;

impl Default for AdaBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            algorithm: AdaBoostAlgorithm::SAMME,
            random_state: None,
        }
    }
}

impl Default for LogitBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            random_state: None,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            tolerance: 1e-4,
            max_iter: 100,
        }
    }
}
