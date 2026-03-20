//! # MonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::execution_core::*;
use crate::resource_management::*;
use crate::performance_optimization::*;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use super::types::*;
use super::functions::*;

use super::types::MonitoringConfig;

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_interval: Duration::from_secs(10),
            metrics_retention: Duration::from_hours(24),
            alert_thresholds: AlertThresholds::default(),
            export_config: MetricsExportConfig::default(),
        }
    }
}

