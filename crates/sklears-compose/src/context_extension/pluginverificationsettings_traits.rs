//! # PluginVerificationSettings - Trait Implementations
//!
//! This module contains trait implementations for `PluginVerificationSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PluginVerificationSettings;

impl Default for PluginVerificationSettings {
    fn default() -> Self {
        Self {
            verify_signature: false,
            verify_checksum: true,
            trusted_cas: Vec::new(),
            allow_self_signed: false,
        }
    }
}

