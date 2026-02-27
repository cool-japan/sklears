//! # ErrorTemplate - Trait Implementations
//!
//! This module contains trait implementations for `ErrorTemplate`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ErrorTemplate, OutputFormat, SuggestionFormat};

impl Default for ErrorTemplate {
    fn default() -> Self {
        Self {
            template_id: "default".to_string(),
            message_template: "{error}\n\nContext: {context}\n\nSuggestions: {suggestions}"
                .to_string(),
            context_sections: Vec::new(),
            suggestion_format: SuggestionFormat {
                max_suggestions: 5,
                include_code_examples: true,
                include_confidence_scores: true,
                include_time_estimates: false,
            },
            output_format: OutputFormat::PlainText,
        }
    }
}
