//! # ActionSpace - Trait Implementations
//!
//! This module contains trait implementations for `ActionSpace`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ActionSpace, ActionSpaceType};

impl Default for ActionSpace {
    fn default() -> Self {
        Self {
            space_type: ActionSpaceType::Discrete,
            dimensions: 10,
            bounds: None,
            discrete_actions: Some(vec!["action_0".to_string(), "action_1".to_string()]),
        }
    }
}

