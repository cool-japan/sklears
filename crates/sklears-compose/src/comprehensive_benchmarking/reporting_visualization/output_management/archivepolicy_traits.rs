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
                "/tmp/sklears_archive".to_string(),
            ),
            compression: ArchiveCompression::default(),
            indexing: ArchiveIndexing::default(),
        }
    }
}

