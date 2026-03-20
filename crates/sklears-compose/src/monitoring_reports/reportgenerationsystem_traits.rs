//! # ReportGenerationSystem - Trait Implementations
//!
//! This module contains trait implementations for `ReportGenerationSystem`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::monitoring_config::*;
use crate::monitoring_metrics::*;
use crate::monitoring_events::*;
use crate::monitoring_core::*;

use super::types::ReportGenerationSystem;

impl Default for ReportGenerationSystem {
    fn default() -> Self {
        Self::new()
    }
}

