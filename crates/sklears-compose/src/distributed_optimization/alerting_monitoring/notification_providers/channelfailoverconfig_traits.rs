//! # ChannelFailoverConfig - Trait Implementations
//!
//! This module contains trait implementations for `ChannelFailoverConfig`.
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

use super::types::ChannelFailoverConfig;

impl Default for ChannelFailoverConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            failover_targets: Vec::new(),
            strategy: FailoverStrategy::Automatic,
            failover_delay: Duration::from_secs(30),
            health_check: FailoverHealthCheck::default(),
            recovery_config: FailoverRecoveryConfig::default(),
        }
    }
}

