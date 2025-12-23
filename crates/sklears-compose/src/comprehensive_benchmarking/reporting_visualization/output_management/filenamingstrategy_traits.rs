//! # FileNamingStrategy - Trait Implementations
//!
//! This module contains trait implementations for `FileNamingStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for FileNamingStrategy {
    fn default() -> Self {
        let mut variables = HashMap::new();
        variables.insert("project".to_string(), "default".to_string());
        variables.insert("version".to_string(), "1.0".to_string());
        Self {
            naming_pattern: "{project}_{timestamp}_{counter}.{extension}".to_string(),
            timestamp_format: "%Y%m%d_%H%M%S".to_string(),
            counter_format: "{:04}".to_string(),
            custom_variables: variables,
            case_conversion: CaseConversion::None,
            invalid_char_handling: InvalidCharHandling::Replace,
            max_filename_length: 255,
        }
    }
}

