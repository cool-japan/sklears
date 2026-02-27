//! # PredictionConfig - Trait Implementations
//!
//! This module contains trait implementations for `PredictionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime, Instant};
use crate::fault_core::*;

use super::types::PredictionConfig;

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prediction_horizon: Duration::from_secs(3600),
            confidence_threshold: 0.8,
            model_retrain_frequency: Duration::from_secs(86400),
        }
    }
}

