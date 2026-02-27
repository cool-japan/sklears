//! # EnhancedErrorContext - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedErrorContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::{
    ConfigurationContext, DataContext, EnhancedErrorContext, EnvironmentContext,
    PerformanceContext, PipelineContext,
};

impl Default for EnhancedErrorContext {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            pipeline_context: PipelineContext::default(),
            data_context: DataContext::default(),
            environment_context: EnvironmentContext::default(),
            performance_context: PerformanceContext::default(),
            configuration_context: ConfigurationContext::default(),
            call_stack: Vec::new(),
            related_issues: Vec::new(),
        }
    }
}
