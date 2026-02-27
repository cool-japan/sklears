//! # DataErrorSuggestionGenerator - Trait Implementations
//!
//! This module contains trait implementations for `DataErrorSuggestionGenerator`.
//!
//! ## Implemented Traits
//!
//! - `SuggestionGenerator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SklearsComposeError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::functions::{DurationExt, SuggestionGenerator};
use super::types::{
    ActionableSuggestion, CodeExample, DataErrorSuggestionGenerator, EnhancedErrorContext,
    ExpertiseLevel, ImplementationStep,
};

impl SuggestionGenerator for DataErrorSuggestionGenerator {
    fn generate_suggestions(
        &self,
        error: &SklearsComposeError,
        context: &EnhancedErrorContext,
    ) -> Result<Vec<ActionableSuggestion>> {
        let mut suggestions = Vec::new();
        let error_str = error.to_string().to_lowercase();
        if error_str.contains("shape") {
            suggestions
                .push(ActionableSuggestion {
                    suggestion_id: "fix_data_shape".to_string(),
                    title: "Fix Data Shape Mismatch".to_string(),
                    description: "The input data shape doesn't match the expected shape. Check your data preprocessing steps."
                        .to_string(),
                    implementation_steps: vec![
                        #[doc = " ImplementationStep"] ImplementationStep { step_number :
                        1, description : "Check the shape of your input data"
                        .to_string(), code_snippet :
                        Some("println!(\"Data shape: {:?}\", data.shape());"
                        .to_string()), command : None, expected_outcome :
                        "Display the actual data shape".to_string(), validation_check :
                        None, }
                    ],
                    code_examples: vec![
                        #[doc = " CodeExample"] CodeExample { language : "rust"
                        .to_string(), code :
                        "let data = data.into_shape((n_samples, n_features))?;"
                        .to_string(), description : "Reshape data to correct dimensions"
                        .to_string(), file_path : None, }
                    ],
                    confidence: 0.9,
                    estimated_time: <Duration as DurationExt>::from_mins(5),
                    success_probability: 0.85,
                    expertise_level: ExpertiseLevel::Beginner,
                    prerequisites: Vec::new(),
                    validation_method: Some(
                        "Check that data.shape() matches expected shape".to_string(),
                    ),
                    follow_up_suggestions: Vec::new(),
                });
        }
        Ok(suggestions)
    }
    fn error_types(&self) -> Vec<String> {
        vec!["DataError".to_string()]
    }
}
