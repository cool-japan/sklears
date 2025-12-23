//! # PlatformAnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `PlatformAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for PlatformAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_advanced_analysis: true,
            enable_performance_benchmarking: false,
            enable_cloud_platform_analysis: true,
            enable_gpu_analysis: false,
            enable_container_analysis: true,
            enable_embedded_analysis: false,
            enable_security_analysis: true,
            enable_compliance_analysis: false,
            benchmark_config: BenchmarkConfig::default(),
        }
    }
}

