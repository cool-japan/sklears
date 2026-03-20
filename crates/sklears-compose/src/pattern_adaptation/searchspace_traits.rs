//! # SearchSpace - Trait Implementations
//!
//! This module contains trait implementations for `SearchSpace`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{Duration, Instant, SystemTime};

use super::types::SearchSpace;

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            space_id: format!(
                "search_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default().as_millis()
            ),
            hyperparameters: HashMap::new(),
            constraints: vec![],
            conditional_spaces: HashMap::new(),
        }
    }
}

