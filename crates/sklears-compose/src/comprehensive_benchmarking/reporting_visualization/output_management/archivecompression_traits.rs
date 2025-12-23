//! # ArchiveCompression - Trait Implementations
//!
//! This module contains trait implementations for `ArchiveCompression`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for ArchiveCompression {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 6,
            per_file: false,
        }
    }
}

