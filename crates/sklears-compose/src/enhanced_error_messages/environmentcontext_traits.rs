//! # EnvironmentContext - Trait Implementations
//!
//! This module contains trait implementations for `EnvironmentContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::EnvironmentContext;

impl Default for EnvironmentContext {
    fn default() -> Self {
        Self {
            os_info: "unknown".to_string(),
            available_memory: 0,
            cpu_info: "unknown".to_string(),
            runtime_version: "unknown".to_string(),
            package_versions: HashMap::new(),
            environment_variables: HashMap::new(),
        }
    }
}
