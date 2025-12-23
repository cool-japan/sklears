//! # VersionMetadataTracking - Trait Implementations
//!
//! This module contains trait implementations for `VersionMetadataTracking`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for VersionMetadataTracking {
    fn default() -> Self {
        Self {
            track_created_at: true,
            track_creator: true,
            track_checksums: true,
            track_parameters: true,
            custom_fields: vec![],
        }
    }
}

