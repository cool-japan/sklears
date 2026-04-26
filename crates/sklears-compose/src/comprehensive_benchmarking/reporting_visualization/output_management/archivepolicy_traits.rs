//! # ArchivePolicy - Trait Implementations
//!
//! This module contains trait implementations for `ArchivePolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ArchivePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            archive_after: Duration::days(7),
            archive_destination: ArchiveDestination::LocalDirectory(
                std::env::temp_dir().join("sklears_archive").display().to_string(),
            ),
            compression: ArchiveCompression::default(),
            indexing: ArchiveIndexing::default(),
        }
    }
}

