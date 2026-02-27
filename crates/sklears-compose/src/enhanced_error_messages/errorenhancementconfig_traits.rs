//! # ErrorEnhancementConfig - Trait Implementations
//!
//! This module contains trait implementations for `ErrorEnhancementConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ErrorEnhancementConfig;

impl Default for ErrorEnhancementConfig {
    fn default() -> Self {
        Self {
            enable_pattern_analysis: true,
            enable_context_collection: true,
            enable_auto_suggestions: true,
            enable_recovery_strategies: true,
            max_suggestions_per_error: 5,
            suggestion_confidence_threshold: 0.7,
            enable_learning: true,
            error_history_size: 1000,
            enable_detailed_diagnostics: true,
        }
    }
}
