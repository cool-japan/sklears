//! # MultiObjectiveOptimizer - Trait Implementations
//!
//! This module contains trait implementations for `MultiObjectiveOptimizer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{ConvergenceCriteria, MultiObjectiveAlgorithm, MultiObjectiveOptimizer, ParetoFront, PreferenceModel, SolutionArchive};

impl Default for MultiObjectiveOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: format!(
                "mo_opt_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default().as_millis()
            ),
            objectives: vec![],
            pareto_front: ParetoFront::default(),
            optimization_algorithm: MultiObjectiveAlgorithm::NSGA_II,
            preference_model: PreferenceModel::default(),
            solution_archive: SolutionArchive::default(),
            convergence_criteria: ConvergenceCriteria::default(),
        }
    }
}

