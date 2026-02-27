//! # PipelineContext - Trait Implementations
//!
//! This module contains trait implementations for `PipelineContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::PipelineContext;

impl Default for PipelineContext {
    fn default() -> Self {
        Self {
            pipeline_name: "unknown".to_string(),
            current_step: "unknown".to_string(),
            step_index: 0,
            total_steps: 0,
            completed_steps: Vec::new(),
            pipeline_config: HashMap::new(),
        }
    }
}
