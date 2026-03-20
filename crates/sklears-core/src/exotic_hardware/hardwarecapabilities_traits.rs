//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::collections::HashMap;

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            compute_units: 1,
            memory_gb: 1.0,
            peak_performance_ops: 1e9,
            supported_precisions: vec![Precision::Float32],
            supports_sparsity: false,
            supports_quantization: false,
            supports_dynamic_shapes: false,
            custom_features: HashMap::new(),
        }
    }
}
