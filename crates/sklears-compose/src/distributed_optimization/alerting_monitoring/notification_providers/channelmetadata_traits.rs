//! # ChannelMetadata - Trait Implementations
//!
//! This module contains trait implementations for `ChannelMetadata`.
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

use super::types::ChannelMetadata;

impl Default for ChannelMetadata {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            modified_at: now,
            created_by: "system".to_string(),
            description: String::new(),
            tags: Vec::new(),
            version: 1,
            usage_stats: ChannelUsageStats::default(),
            maintenance_info: MaintenanceInfo::default(),
            cost_info: CostInfo::default(),
            sla_info: SlaInfo::default(),
        }
    }
}

