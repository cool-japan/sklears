//! # CostReportingConfig - Trait Implementations
//!
//! This module contains trait implementations for `CostReportingConfig`.
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

use super::types::CostReportingConfig;

impl Default for CostReportingConfig {
    fn default() -> Self {
        Self {
            enable_reports: false,
            report_frequency: Duration::from_secs(86400),
            recipients: Vec::new(),
            format: ReportFormat::JSON,
        }
    }
}

