//! # FilePermissions - Trait Implementations
//!
//! This module contains trait implementations for `FilePermissions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for FilePermissions {
    fn default() -> Self {
        Self {
            file_mode: 0o644,
            directory_mode: 0o755,
            owner: None,
            group: None,
        }
    }
}

