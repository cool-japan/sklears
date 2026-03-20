//! # SandboxConfig - Trait Implementations
//!
//! This module contains trait implementations for `SandboxConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{SandboxConfig, SandboxType};

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            default_sandbox_type: SandboxType::Process,
            strict_mode: true,
            monitoring_interval: Duration::from_secs(5),
            violation_threshold: 3,
            auto_terminate: true,
        }
    }
}

