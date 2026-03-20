//! # BenchmarkConfig - Trait Implementations
//!
//! This module contains trait implementations for `BenchmarkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            detailed_metrics: false,
            gpu_benchmarking: false,
            memory_profiling: false,
            iterations: 1000,
            confidence_level: 0.95,
            timeout: Duration::from_secs(300),
        }
    }
}

