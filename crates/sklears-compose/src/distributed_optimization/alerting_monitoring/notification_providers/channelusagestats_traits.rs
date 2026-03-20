//! # ChannelUsageStats - Trait Implementations
//!
//! This module contains trait implementations for `ChannelUsageStats`.
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

use super::types::ChannelUsageStats;

impl Default for ChannelUsageStats {
    fn default() -> Self {
        Self {
            total_messages: 0,
            messages_today: 0,
            success_rate: 1.0,
            avg_delivery_time: Duration::from_millis(0),
            last_used: None,
            peak_usage_time: None,
            bandwidth_usage: BandwidthUsage::default(),
            error_stats: ErrorStatistics::default(),
        }
    }
}

