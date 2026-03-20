//! # VersionManagement - Trait Implementations
//!
//! This module contains trait implementations for `VersionManagement`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for VersionManagement {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: VersionStrategy::Timestamp,
            max_versions: Some(10),
            metadata_tracking: VersionMetadataTracking::default(),
        }
    }
}

