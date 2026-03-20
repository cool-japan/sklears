//! # OutputOrganization - Trait Implementations
//!
//! This module contains trait implementations for `OutputOrganization`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for OutputOrganization {
    fn default() -> Self {
        Self {
            directory_structure: DirectoryStructure::ByDate,
            file_grouping: FileGrouping::ByType,
            cleanup_policy: CleanupPolicy::default(),
            archive_policy: ArchivePolicy::default(),
            version_management: VersionManagement::default(),
        }
    }
}

