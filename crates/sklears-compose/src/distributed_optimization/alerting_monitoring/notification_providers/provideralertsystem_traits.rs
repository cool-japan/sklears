//! # ProviderAlertSystem - Trait Implementations
//!
//! This module contains trait implementations for `ProviderAlertSystem`.
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

use super::types::ProviderAlertSystem;

impl Default for ProviderAlertSystem {
    fn default() -> Self {
        Self {
            alert_rules: Vec::new(),
            active_alerts: HashMap::new(),
            alert_history: Vec::new(),
            config: AlertConfiguration::default(),
        }
    }
}

