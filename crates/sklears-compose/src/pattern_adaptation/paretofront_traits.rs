//! # ParetoFront - Trait Implementations
//!
//! This module contains trait implementations for `ParetoFront`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ParetoFront;

impl Default for ParetoFront {
    fn default() -> Self {
        Self {
            solutions: vec![],
            dominated_solutions: vec![],
            hypervolume: 0.0,
            spread_metric: 0.0,
            convergence_metric: 0.0,
        }
    }
}

