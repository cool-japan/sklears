//! # RoundRobinBalancer - Trait Implementations
//!
//! This module contains trait implementations for `RoundRobinBalancer`.
//!
//! ## Implemented Traits
//!
//! - `LoadBalancingAlgorithm`
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

use super::types::RoundRobinBalancer;

impl LoadBalancingAlgorithm for RoundRobinBalancer {
    fn name(&self) -> &str {
        "RoundRobin"
    }
    fn select_backend(
        &mut self,
        _request: &LoadBalancingRequest,
        backends: &[Backend],
    ) -> SklResult<Option<String>> {
        if backends.is_empty() {
            return Ok(None);
        }
        let backend = &backends[self.current_index % backends.len()];
        self.current_index += 1;
        Ok(Some(backend.id.clone()))
    }
    fn update_state(
        &mut self,
        _backend_id: &str,
        _result: &RequestResult,
    ) -> SklResult<()> {
        Ok(())
    }
    fn get_config(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    fn update_config(&mut self, _config: HashMap<String, String>) -> SklResult<()> {
        Ok(())
    }
    fn reset(&mut self) -> SklResult<()> {
        self.current_index = 0;
        Ok(())
    }
}

