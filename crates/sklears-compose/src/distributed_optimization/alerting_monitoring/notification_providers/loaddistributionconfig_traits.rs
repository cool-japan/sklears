//! # LoadDistributionConfig - Trait Implementations
//!
//! This module contains trait implementations for `LoadDistributionConfig`.
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

use super::types::LoadDistributionConfig;

impl Default for LoadDistributionConfig {
    fn default() -> Self {
        Self {
            weights: HashMap::new(),
            max_load_per_endpoint: None,
            load_calculation: LoadCalculationMethod::RequestCount,
            rebalancing: RebalancingConfig::default(),
        }
    }
}

