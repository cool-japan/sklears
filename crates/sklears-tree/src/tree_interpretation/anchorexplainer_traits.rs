//! # AnchorExplainer - Trait Implementations
//!
//! This module contains trait implementations for `AnchorExplainer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AnchorExplainer;

impl Default for AnchorExplainer {
    fn default() -> Self {
        Self {
            precision_threshold: 0.95,
            max_anchor_size: 5,
            coverage_samples: 10000,
            beam_width: 10,
            min_coverage: 0.05,
            random_seed: None,
        }
    }
}

