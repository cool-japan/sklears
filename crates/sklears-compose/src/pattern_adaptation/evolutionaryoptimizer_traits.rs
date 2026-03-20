//! # EvolutionaryOptimizer - Trait Implementations
//!
//! This module contains trait implementations for `EvolutionaryOptimizer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{EvolutionHistory, EvolutionParameters, EvolutionaryOptimizer, FitnessEvaluator, GeneticOperators, SelectionStrategy};

impl Default for EvolutionaryOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: format!(
                "evo_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default().as_millis()
            ),
            population: vec![],
            genetic_operators: GeneticOperators::default(),
            selection_strategy: SelectionStrategy::default(),
            fitness_evaluator: FitnessEvaluator::default(),
            evolution_parameters: EvolutionParameters::default(),
            evolution_history: EvolutionHistory::default(),
        }
    }
}

