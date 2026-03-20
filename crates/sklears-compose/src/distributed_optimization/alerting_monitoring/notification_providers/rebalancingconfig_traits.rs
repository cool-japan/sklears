//! # RebalancingConfig - Trait Implementations
//!
//! This module contains trait implementations for `RebalancingConfig`.
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

use super::types::RebalancingConfig;

impl Default for RebalancingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 0.1,
            interval: Duration::from_secs(60),
            strategy: RebalancingStrategy::Gradual,
        }
    }
}

