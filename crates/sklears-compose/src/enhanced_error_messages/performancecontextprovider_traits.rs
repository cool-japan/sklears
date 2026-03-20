//! # PerformanceContextProvider - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceContextProvider`.
//!
//! ## Implemented Traits
//!
//! - `ContextProvider`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SklearsComposeError};
use std::collections::{HashMap, VecDeque};

use super::functions::ContextProvider;
use super::types::{ContextType, PerformanceContextProvider};

impl ContextProvider for PerformanceContextProvider {
    fn collect_context(&self, _error: &SklearsComposeError) -> Result<HashMap<String, String>> {
        let mut context = HashMap::new();
        context.insert("memory_usage".to_string(), "unknown".to_string());
        Ok(context)
    }
    fn context_type(&self) -> ContextType {
        ContextType::Performance
    }
}
