//! # GeneticOperators - Trait Implementations
//!
//! This module contains trait implementations for `GeneticOperators`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GeneticOperators;

impl Default for GeneticOperators {
    fn default() -> Self {
        Self {
            crossover_operators: vec![],
            mutation_operators: vec![],
            selection_operators: vec![],
        }
    }
}

