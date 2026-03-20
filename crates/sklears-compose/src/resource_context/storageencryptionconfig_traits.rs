//! # StorageEncryptionConfig - Trait Implementations
//!
//! This module contains trait implementations for `StorageEncryptionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for StorageEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: "AES".to_string(),
            key_size: 256,
            mode: "GCM".to_string(),
        }
    }
}

