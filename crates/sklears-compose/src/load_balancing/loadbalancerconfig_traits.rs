//! # LoadBalancerConfig - Trait Implementations
//!
//! This module contains trait implementations for `LoadBalancerConfig`.
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

use super::types::LoadBalancerConfig;

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            algorithm: BalancingAlgorithm::RoundRobin,
            health_check_config: HealthCheckConfig::default(),
            failover_config: FailoverConfig::default(),
            scaling_config: None,
            traffic_config: TrafficConfig::default(),
            session_config: SessionAffinityConfig::default(),
            behavior: LoadBalancerBehavior::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

