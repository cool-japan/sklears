//! # WeightedRoundRobinBalancer - Trait Implementations
//!
//! This module contains trait implementations for `WeightedRoundRobinBalancer`.
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

use super::types::WeightedRoundRobinBalancer;

impl LoadBalancingAlgorithm for WeightedRoundRobinBalancer {
    fn name(&self) -> &str {
        "WeightedRoundRobin"
    }
    fn select_backend(
        &mut self,
        _request: &LoadBalancingRequest,
        backends: &[Backend],
    ) -> SklResult<Option<String>> {
        if backends.is_empty() {
            return Ok(None);
        }
        let mut best_backend = None;
        let mut best_weight = -1.0;
        for backend in backends {
            let current_weight = self
                .current_weights
                .entry(backend.id.clone())
                .or_insert(backend.weight);
            if *current_weight > best_weight {
                best_weight = *current_weight;
                best_backend = Some(backend.id.clone());
            }
        }
        if let Some(ref selected) = best_backend {
            for backend in backends {
                let weight = self.current_weights.get_mut(&backend.id).unwrap();
                if backend.id == *selected {
                    *weight -= backends.iter().map(|b| b.weight).sum::<f64>();
                } else {
                    *weight += backend.weight;
                }
            }
        }
        Ok(best_backend)
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
        self.current_weights.clear();
        Ok(())
    }
}

