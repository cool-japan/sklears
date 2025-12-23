//! # FailoverRecoveryConfig - Trait Implementations
//!
//! This module contains trait implementations for `FailoverRecoveryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use super::types::*;
use super::functions::*;

use super::types::FailoverRecoveryConfig;

impl Default for FailoverRecoveryConfig {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            recovery_delay: Duration::from_secs(60),
            recovery_validation: RecoveryValidation::default(),
        }
    }
}

