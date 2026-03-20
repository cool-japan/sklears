//! # ColorScheme - Trait Implementations
//!
//! This module contains trait implementations for `ColorScheme`.
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

use super::types::ColorScheme;

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            primary: "#007bff".to_string(),
            secondary: "#6c757d".to_string(),
            success: "#28a745".to_string(),
            warning: "#ffc107".to_string(),
            error: "#dc3545".to_string(),
            background: "#ffffff".to_string(),
            text: "#212529".to_string(),
        }
    }
}

