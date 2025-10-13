#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(unexpected_cfgs)]
//! SIMD-optimized operations for sklears
//!
//! This crate provides SIMD-accelerated implementations of common machine learning operations
//! using Rust's portable SIMD API and platform-specific intrinsics.

#![allow(incomplete_features)]
// Note: no-std feature is temporarily disabled until implementation is complete
#![cfg_attr(feature = "no-std", no_std)]
// Portable SIMD is unstable and requires nightly Rust
// #![cfg_attr(feature = "nightly", feature(portable_simd))]

#[cfg(feature = "no-std")]
extern crate alloc;

#[cfg(feature = "no-std")]
use alloc::{boxed::Box, string::String, vec::Vec};

pub mod activation;
pub mod adaptive_optimization;
pub mod advanced_optimizations;
pub mod allocator;
pub mod approximate;
pub mod audio_processing;
pub mod batch_operations;
pub mod benchmark_framework;
pub mod bit_operations;
pub mod clustering;
pub mod comprehensive_benchmarks;
pub mod compression;
pub mod custom_accelerator;
pub mod distance;
pub mod distributions;
pub mod energy_benchmarks;
pub mod error_correction;
pub mod external_integration;
pub mod fluent;
pub mod fpga;
pub mod gpu;
pub mod gpu_memory;
pub mod half_precision;
pub mod image_processing;
pub mod intrinsics;
pub mod kernels;
pub mod loss;
pub mod matrix;
pub mod memory;
pub mod middleware;
pub mod multi_gpu;
pub mod neuromorphic;
#[cfg(feature = "no-std")]
pub mod no_std;
pub mod optimization;
pub mod optimization_hints;
pub mod performance_hooks;
pub mod performance_monitor;
pub mod plugin_architecture;
pub mod profiling;
pub mod quantum;
pub mod reduction;
pub mod regression;
#[cfg(target_arch = "riscv64")]
pub mod riscv_vector;
pub mod safe_simd;
pub mod safety;
pub mod search;
pub mod signal_processing;
pub mod sorting;
pub mod target;
pub mod tpu;
pub mod traits;
pub mod validation;
pub mod vector;

// Re-export key types and functions
pub use clustering::LinkageType;

/// Platform-specific SIMD capabilities
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub neon: bool,
    pub riscv_vector: bool,
    pub riscv_vlen: usize,
}

impl SimdCapabilities {
    /// Detect available SIMD instructions on the current platform
    pub fn detect() -> Self {
        Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse: is_x86_feature_detected!("sse"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse2: is_x86_feature_detected!("sse2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse3: is_x86_feature_detected!("sse3"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ssse3: is_x86_feature_detected!("ssse3"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse41: is_x86_feature_detected!("sse4.1"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse42: is_x86_feature_detected!("sse4.2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx: is_x86_feature_detected!("avx"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse2: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse3: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            ssse3: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse41: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse42: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx2: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx512: false,
            #[cfg(target_arch = "aarch64")]
            neon: true,
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,

            #[cfg(target_arch = "riscv64")]
            riscv_vector: {
                #[cfg(target_arch = "riscv64")]
                {
                    crate::riscv_vector::RiscVVectorCaps::detect().available
                }
                #[cfg(not(target_arch = "riscv64"))]
                {
                    false
                }
            },
            #[cfg(not(target_arch = "riscv64"))]
            riscv_vector: false,

            #[cfg(target_arch = "riscv64")]
            riscv_vlen: {
                #[cfg(target_arch = "riscv64")]
                {
                    crate::riscv_vector::RiscVVectorCaps::detect().vlen
                }
                #[cfg(not(target_arch = "riscv64"))]
                {
                    0
                }
            },
            #[cfg(not(target_arch = "riscv64"))]
            riscv_vlen: 0,
        }
    }

    /// Get the best available SIMD width for f32 operations
    pub fn best_f32_width(&self) -> usize {
        if self.avx512 {
            16 // 512 bits / 32 bits
        } else if self.avx2 || self.avx {
            8 // 256 bits / 32 bits
        } else if self.sse || self.neon {
            4 // 128 bits / 32 bits
        } else if self.riscv_vector && self.riscv_vlen > 0 {
            self.riscv_vlen / 32 // VLEN bits / 32 bits per f32
        } else {
            1 // Scalar fallback
        }
    }

    /// Get the best available SIMD width for f64 operations
    pub fn best_f64_width(&self) -> usize {
        if self.avx512 {
            8 // 512 bits / 64 bits
        } else if self.avx2 || self.avx {
            4 // 256 bits / 64 bits
        } else if self.sse2 || self.neon {
            2 // 128 bits / 64 bits
        } else if self.riscv_vector && self.riscv_vlen > 0 {
            self.riscv_vlen / 64 // VLEN bits / 64 bits per f64
        } else {
            1 // Scalar fallback
        }
    }

    /// Get the platform name for current SIMD capabilities
    pub fn platform_name(&self) -> &'static str {
        if self.avx512 {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX"
        } else if self.sse42 {
            "SSE4.2"
        } else if self.sse41 {
            "SSE4.1"
        } else if self.ssse3 {
            "SSSE3"
        } else if self.sse3 {
            "SSE3"
        } else if self.sse2 {
            "SSE2"
        } else if self.sse {
            "SSE"
        } else if self.neon {
            "NEON"
        } else if self.riscv_vector {
            "RISC-V Vector"
        } else {
            "Scalar"
        }
    }
}

/// Global SIMD capabilities detection
pub static SIMD_CAPS: once_cell::sync::Lazy<SimdCapabilities> =
    once_cell::sync::Lazy::new(SimdCapabilities::detect);

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        println!("SIMD Capabilities: {:?}", caps);

        // At least one width should be available
        assert!(caps.best_f32_width() >= 1);
        assert!(caps.best_f64_width() >= 1);
    }
}
