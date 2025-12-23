//! # WasmIntegrationConfig - Trait Implementations
//!
//! This module contains trait implementations for `WasmIntegrationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use crate::error::{Result, SklearsComposeError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::WasmIntegrationConfig;

impl Default for WasmIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_multithreading: true,
            enable_memory_optimization: true,
            enable_streaming: true,
            max_memory_size: 1024 * 1024 * 1024,
            module_cache_size: 100,
            enable_compatibility_checks: true,
            enable_performance_monitoring: true,
            optimization_level: 2,
            enable_debug_info: false,
            target_features: vec![
                BrowserFeature::Simd128,
                BrowserFeature::BulkMemory,
                BrowserFeature::ReferenceTypes,
            ],
        }
    }
}
