//! # HyperparameterOptimizer - Trait Implementations
//!
//! This module contains trait implementations for `HyperparameterOptimizer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{HyperparameterOptimizer, ObjectiveFunction, OptimizationAlgorithm, OptimizationEarlyStopping, ParallelEvaluation, SearchHistory, SearchSpace};

impl Default for HyperparameterOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: format!(
                "hp_opt_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default().as_millis()
            ),
            search_space: SearchSpace::default(),
            optimization_algorithm: OptimizationAlgorithm::BayesianOptimization,
            objective_function: ObjectiveFunction::default(),
            search_history: SearchHistory::default(),
            early_stopping: OptimizationEarlyStopping::default(),
            parallel_evaluation: ParallelEvaluation::default(),
        }
    }
}

