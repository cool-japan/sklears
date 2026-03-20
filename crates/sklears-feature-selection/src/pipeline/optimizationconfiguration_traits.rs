//! # OptimizationConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OptimizationConfiguration, PrefetchStrategy};

impl Default for OptimizationConfiguration {
    fn default() -> Self {
        Self {
            use_simd: true,
            chunk_size: 1000,
            thread_pool_size: None,
            memory_pool_size: 1024 * 1024,
            cache_size: 100,
            prefetch_strategy: PrefetchStrategy::Sequential,
            vectorization_threshold: 1000,
        }
    }
}
