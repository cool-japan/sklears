//! # ChannelRateLimit - Trait Implementations
//!
//! This module contains trait implementations for `ChannelRateLimit`.
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

use super::types::ChannelRateLimit;

impl Default for ChannelRateLimit {
    fn default() -> Self {
        Self {
            enabled: false,
            rate_per_second: 10,
            rate_per_minute: 600,
            rate_per_hour: 36000,
            burst_allowance: 20,
            strategy: RateLimitStrategy::TokenBucket,
            throttling: ThrottlingConfig::default(),
        }
    }
}

