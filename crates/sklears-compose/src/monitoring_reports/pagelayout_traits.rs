//! # PageLayout - Trait Implementations
//!
//! This module contains trait implementations for `PageLayout`.
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

use super::types::{Margins, PageLayout};

impl Default for PageLayout {
    fn default() -> Self {
        Self {
            margins: Margins {
                top: 20,
                bottom: 20,
                left: 20,
                right: 20,
            },
            header_height: 60,
            footer_height: 40,
            content_width: 800,
        }
    }
}

