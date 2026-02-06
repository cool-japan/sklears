use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

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

