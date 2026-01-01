//! # CsvConfig - Trait Implementations
//!
//! This module contains trait implementations for `CsvConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CsvConfig;

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            has_header: true,
            quote_char: '"',
            escape_char: Some('\\'),
        }
    }
}

