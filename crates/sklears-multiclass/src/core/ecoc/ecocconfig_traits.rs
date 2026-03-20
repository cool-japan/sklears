//! # ECOCConfig - Trait Implementations
//!
//! This module contains trait implementations for `ECOCConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::ECOCConfig;

impl Default for ECOCConfig {
    fn default() -> Self {
        Self {
            strategy: ECOCStrategy::default(),
            code_size: 1.5,
            random_state: None,
            n_jobs: None,
            use_sparse: false,
            sparse_threshold: 0.3,
            gpu_mode: GPUMode::default(),
            gpu_batch_size: 1000,
            memory_mode: MemoryMode::default(),
            compression_level: 6,
            quantize_models: false,
        }
    }
}
