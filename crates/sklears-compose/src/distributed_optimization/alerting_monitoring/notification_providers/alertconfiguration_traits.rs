//! # AlertConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `AlertConfiguration`.
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

use super::types::AlertConfiguration;

impl Default for AlertConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            evaluation_interval: Duration::from_secs(60),
            max_active_alerts: 100,
            suppression_enabled: false,
            escalation_config: EscalationConfiguration::default(),
        }
    }
}

