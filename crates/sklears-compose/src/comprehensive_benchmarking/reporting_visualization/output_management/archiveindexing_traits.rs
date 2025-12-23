//! # ArchiveIndexing - Trait Implementations
//!
//! This module contains trait implementations for `ArchiveIndexing`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ArchiveIndexing {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_type: IndexStorageType::SQLite,
            searchable_fields: vec![
                "filename".to_string(), "format".to_string(), "created_at".to_string(),
                "size".to_string(), "checksum".to_string(),
            ],
            full_text_search: false,
        }
    }
}

