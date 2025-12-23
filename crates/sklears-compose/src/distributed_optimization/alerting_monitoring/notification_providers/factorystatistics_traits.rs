//! # FactoryStatistics - Trait Implementations
//!
//! This module contains trait implementations for `FactoryStatistics`.
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

use super::types::FactoryStatistics;

impl Default for FactoryStatistics {
    fn default() -> Self {
        Self {
            providers_created: 0,
            creation_failures: 0,
            avg_creation_time: Duration::from_millis(0),
            usage_statistics: HashMap::new(),
        }
    }
}

