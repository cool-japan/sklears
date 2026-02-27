//! # PipelineConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `PipelineConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    CachingStrategy, ErrorHandling, LoggingLevel, MemoryOptimization, PipelineConfiguration,
    ValidationStrategy,
};

impl Default for PipelineConfiguration {
    fn default() -> Self {
        Self {
            parallel_execution: true,
            memory_optimization: MemoryOptimization::Conservative,
            caching_strategy: CachingStrategy::LRU { size: 1000 },
            validation_strategy: ValidationStrategy::Basic,
            error_handling: ErrorHandling::Graceful,
            logging_level: LoggingLevel::Info,
        }
    }
}
