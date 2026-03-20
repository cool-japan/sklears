//! # EnvironmentContextProvider - Trait Implementations
//!
//! This module contains trait implementations for `EnvironmentContextProvider`.
//!
//! ## Implemented Traits
//!
//! - `ContextProvider`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SklearsComposeError};
use std::collections::{HashMap, VecDeque};

use super::functions::ContextProvider;
use super::types::{ContextType, EnvironmentContextProvider};

impl ContextProvider for EnvironmentContextProvider {
    fn collect_context(&self, _error: &SklearsComposeError) -> Result<HashMap<String, String>> {
        let mut context = HashMap::new();
        context.insert("os".to_string(), std::env::consts::OS.to_string());
        context.insert("arch".to_string(), std::env::consts::ARCH.to_string());
        Ok(context)
    }
    fn context_type(&self) -> ContextType {
        ContextType::Environment
    }
}
