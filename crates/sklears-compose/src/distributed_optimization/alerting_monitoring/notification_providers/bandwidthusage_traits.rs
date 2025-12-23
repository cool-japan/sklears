//! # BandwidthUsage - Trait Implementations
//!
//! This module contains trait implementations for `BandwidthUsage`.
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

use super::types::BandwidthUsage;

impl Default for BandwidthUsage {
    fn default() -> Self {
        Self {
            total_bytes_sent: 0,
            total_bytes_received: 0,
            avg_message_size: 0,
            peak_bandwidth: 0.0,
            current_bandwidth: 0.0,
        }
    }
}

