//! SIMD-Optimized Vector Operations
//!
//! This module provides comprehensive SIMD-accelerated vector operations organized
//! into focused sub-modules for optimal performance across different platforms and
//! operation categories.
//!
//! ## Architecture
//!
//! The SIMD vector system is organized into specialized modules:
//! - **Basic Operations**: Fundamental vector operations (dot product, norm, scaling)
//! - **Statistical Functions**: Statistical computations (mean, variance, quantile)
//! - **Mathematical Functions**: Advanced math operations (trigonometric, exponential)
//! - **Advanced Operations**: Complex algorithms (cross product, histogram, outer product)
//! - **SIMD Types**: Platform detection and common type definitions
//! - **SIMD Utils**: Low-level intrinsics and utility functions
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use sklears_simd::vector::*;
//!
//! // Basic vector operations
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![5.0, 6.0, 7.0, 8.0];
//!
//! // SIMD-optimized dot product
//! let dot = dot_product(&a, &b);
//!
//! // SIMD-optimized norm computation
//! let norm = norm_l2(&a);
//!
//! // Statistical operations
//! let mean_val = mean(&a);
//! let var_val = variance(&a);
//!
//! // Mathematical functions
//! let mut result = vec![0.0; a.len()];
//! exp_vec(&a, &mut result);
//! sin_vec(&a, &mut result);
//! ```
//!
//! ## Platform Optimization
//!
//! The vector operations automatically detect and utilize the best available SIMD
//! instruction set on the target platform:
//!
//! - **x86/x86_64**: SSE2, AVX2, AVX-512
//! - **ARM**: NEON
//! - **RISC-V**: Vector Extensions
//! - **Fallback**: Optimized scalar implementations
//!
//! ## Performance Features
//!
//! - **Auto-Vectorization**: Automatic selection of optimal SIMD instructions
//! - **Memory Alignment**: Efficient memory access patterns
//! - **Branch-Free Operations**: Reduced branch misprediction overhead
//! - **Cache-Friendly**: Optimized for modern CPU cache hierarchies
//!
//! ## Mathematical Background
//!
//! ### Dot Product
//! ```
//! dot(a, b) = Σᵢ aᵢ × bᵢ
//! ```
//!
//! ### L2 Norm
//! ```
//! ||x||₂ = √(Σᵢ xᵢ²)
//! ```
//!
//! ### Fused Multiply-Add (FMA)
//! ```
//! FMA(a, b, c) = a × b + c
//! ```
//!
//! ### Statistical Variance
//! ```
//! σ² = (1/n) Σᵢ (xᵢ - μ)²
//! ```

#[cfg(feature = "no-std")]
use alloc::vec;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

mod simd_basic;
mod simd_statistical;
mod simd_mathematical;
mod simd_advanced;
mod simd_types;
mod simd_utils;

// Re-export all public functions and types
pub use simd_basic::*;
pub use simd_statistical::*;
pub use simd_mathematical::*;
pub use simd_advanced::*;
pub use simd_types::*;
pub use simd_utils::*;

// Legacy compatibility aliases for existing APIs
pub use simd_basic::{
    dot_product as dot_product_simd,
    norm_l2 as norm_simd,
    scale as scale_simd,
};

pub use simd_statistical::{
    mean,
    variance,
    sum as sum_simd,
    min_max,
};

pub use simd_mathematical::{
    sqrt_vec,
    exp_vec,
    ln_vec,
    sin_vec,
    cos_vec,
    tan_vec,
    reciprocal_vec,
};

pub use simd_advanced::{
    cross_product,
    outer_product,
    histogram_simd,
    quantile_simd,
};

// Convenience functions that maintain API compatibility
pub fn add_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    simd_basic::add_vectors(a, b)
}

pub fn fma_simd(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    simd_basic::fused_multiply_add(a, b, c)
}

pub fn min_simd(vector: &[f32]) -> f32 {
    simd_statistical::minimum(vector)
}

pub fn max_simd(vector: &[f32]) -> f32 {
    simd_statistical::maximum(vector)
}

/// Vector operation configuration
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Force specific SIMD instruction set (None = auto-detect)
    pub force_simd: Option<SimdInstructionSet>,
    /// Enable unsafe optimizations
    pub unsafe_optimizations: bool,
    /// Memory alignment preference
    pub alignment: usize,
    /// Enable detailed performance metrics
    pub enable_metrics: bool,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            force_simd: None,
            unsafe_optimizations: false,
            alignment: 32, // 256-bit alignment for AVX2
            enable_metrics: false,
        }
    }
}

/// Global vector operation configuration
pub static mut VECTOR_CONFIG: VectorConfig = VectorConfig {
    force_simd: None,
    unsafe_optimizations: false,
    alignment: 32,
    enable_metrics: false,
};

/// Set global vector operation configuration
pub fn set_vector_config(config: VectorConfig) {
    unsafe {
        VECTOR_CONFIG = config;
    }
}

/// Get current vector operation configuration
pub fn get_vector_config() -> &'static VectorConfig {
    unsafe { &VECTOR_CONFIG }
}

/// Benchmark results for vector operations
#[derive(Debug, Clone)]
pub struct VectorBenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Vector size tested
    pub size: usize,
    /// Execution time in nanoseconds
    pub time_ns: u64,
    /// Throughput in operations per second
    pub ops_per_sec: f64,
    /// SIMD instruction set used
    pub simd_used: SimdInstructionSet,
}

/// Performance benchmarking for vector operations
pub fn benchmark_vector_operations(sizes: &[usize]) -> Vec<VectorBenchmarkResult> {
    let mut results = Vec::new();

    for &size in sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        // Benchmark dot product
        let start = std::time::Instant::now();
        let _ = dot_product(&a, &b);
        let time_ns = start.elapsed().as_nanos() as u64;

        results.push(VectorBenchmarkResult {
            operation: "dot_product".to_string(),
            size,
            time_ns,
            ops_per_sec: (size as f64 * 1_000_000_000.0) / time_ns as f64,
            simd_used: detect_best_simd(),
        });
    }

    results
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_basic_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // Test dot product
        let dot = dot_product(&a, &b);
        assert!((dot - 70.0).abs() < 1e-6);

        // Test norm
        let norm = norm_l2(&a);
        assert!((norm - (1.0 + 4.0 + 9.0 + 16.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_vector_statistical_operations() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test mean
        let mean_val = mean(&data);
        assert!((mean_val - 3.0).abs() < 1e-6);

        // Test sum
        let sum_val = sum(&data);
        assert!((sum_val - 15.0).abs() < 1e-6);

        // Test min/max
        let (min_val, max_val) = min_max(&data);
        assert!((min_val - 1.0).abs() < 1e-6);
        assert!((max_val - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_mathematical_operations() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; input.len()];

        // Test sqrt
        sqrt_vec(&input, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!((val - (input[i] as f32).sqrt()).abs() < 1e-6);
        }

        // Test exp
        exp_vec(&input, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!((val - (input[i] as f32).exp()).abs() < 1e-3);
        }
    }

    #[test]
    fn test_vector_advanced_operations() {
        // Test cross product (3D)
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let cross = cross_product(&a, &b).unwrap();
        assert_eq!(cross, vec![0.0, 0.0, 1.0]);

        // Test outer product
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let outer = outer_product(&a, &b);
        assert_eq!(outer[0], vec![3.0, 4.0]);
        assert_eq!(outer[1], vec![6.0, 8.0]);
    }

    #[test]
    fn test_simd_detection() {
        let simd = detect_best_simd();
        println!("Best SIMD: {:?}", simd);

        // Should always have at least scalar capability
        assert!(simd != SimdInstructionSet::None);
    }

    #[test]
    fn test_vector_config() {
        let config = VectorConfig {
            force_simd: Some(SimdInstructionSet::Scalar),
            unsafe_optimizations: true,
            alignment: 64,
            enable_metrics: true,
        };

        set_vector_config(config.clone());
        let retrieved = get_vector_config();

        assert_eq!(retrieved.force_simd, config.force_simd);
        assert_eq!(retrieved.unsafe_optimizations, config.unsafe_optimizations);
        assert_eq!(retrieved.alignment, config.alignment);
        assert_eq!(retrieved.enable_metrics, config.enable_metrics);
    }

    #[test]
    fn test_performance_benchmark() {
        let sizes = vec![100, 1000];
        let results = benchmark_vector_operations(&sizes);

        assert_eq!(results.len(), 2);
        for result in results {
            assert!(result.time_ns > 0);
            assert!(result.ops_per_sec > 0.0);
            assert_eq!(result.operation, "dot_product");
        }
    }
}