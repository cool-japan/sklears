//! # ResourceBaselines - Trait Implementations
//!
//! This module contains trait implementations for `ResourceBaselines`.
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

use super::types::ResourceBaselines;

impl Default for ResourceBaselines {
    fn default() -> Self {
        Self {
            cpu_baseline: 0.5,
            memory_baseline: 0.6,
            network_baseline: 0.3,
            storage_baseline: 0.4,
        }
    }
}

