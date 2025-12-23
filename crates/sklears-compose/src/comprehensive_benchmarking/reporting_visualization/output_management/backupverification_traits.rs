//! # BackupVerification - Trait Implementations
//!
//! This module contains trait implementations for `BackupVerification`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BackupVerification {
    fn default() -> Self {
        Self {
            enabled: true,
            methods: vec![
                VerificationMethod::Checksum, VerificationMethod::FileCount,
                VerificationMethod::SizeCheck,
            ],
            frequency: VerificationFrequency::AfterBackup,
        }
    }
}

