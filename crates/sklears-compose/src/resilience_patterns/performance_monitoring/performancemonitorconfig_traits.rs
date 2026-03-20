//! # PerformanceMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;
use serde::{Serialize, Deserialize};
use crate::fault_core::*;
use super::types::*;
use super::functions::*;

use super::types::PerformanceMonitorConfig;

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            collection: CollectionConfig::default(),
            analysis: AnalysisConfig::default(),
            alerting: AlertingConfig::default(),
            baseline: BaselineConfig::default(),
            trends: TrendConfig::default(),
            sla: SlaConfig::default(),
            realtime: RealtimeConfig::default(),
            reporting: ReportingConfig::default(),
        }
    }
}

