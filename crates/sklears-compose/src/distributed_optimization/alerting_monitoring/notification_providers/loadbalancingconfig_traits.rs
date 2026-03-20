//! # LoadBalancingConfig - Trait Implementations
//!
//! This module contains trait implementations for `LoadBalancingConfig`.
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

use super::types::LoadBalancingConfig;

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: LoadBalancingAlgorithm::RoundRobin,
            weights: HashMap::new(),
            health_check_integration: true,
            sticky_session: None,
            load_distribution: LoadDistributionConfig::default(),
        }
    }
}

