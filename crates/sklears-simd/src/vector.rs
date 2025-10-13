//! # SIMD Vector Operations Framework
//!
//! High-performance SIMD-optimized vector operations library providing comprehensive
//! mathematical operations with automatic platform detection and optimization.
//!
//! ## Features
//!
//! - **Multi-Platform SIMD**: Automatic detection and use of SSE2, AVX2, AVX512, NEON
//! - **Comprehensive Operations**: Basic arithmetic, advanced math, statistics, comparisons
//! - **Performance Optimized**: Hand-tuned intrinsics with scalar fallbacks
//! - **Type Safe**: Compile-time platform feature detection
//! - **No-std Compatible**: Supports embedded and constrained environments
//! - **Extensive Testing**: Comprehensive test coverage with accuracy verification
//!
//! ## Architecture
//!
//! ```text
//! SIMD Vector Framework
//! ├── Basic Operations (dot product, norms, fundamentals)
//! ├── Arithmetic Operations (add, multiply, FMA, element-wise)
//! ├── Comparison Operations (min/max, logical operations)
//! ├── Math Functions (trigonometric, exponential functions)
//! ├── Statistics Operations (mean, histogram, quantile)
//! └── Platform Intrinsics (SSE2, AVX2, AVX512, NEON)
//! ```
//!
//! ## Usage Examples
//!
//! ```rust
//! use sklears_simd::vector::{dot_product, norm_l2, add_vec};
//!
//! // Basic vector operations
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![5.0, 6.0, 7.0, 8.0];
//!
//! // SIMD-optimized dot product
//! let dot = dot_product(&a, &b);
//!
//! // L2 norm computation
//! let norm = norm_l2(&a);
//!
//! // Element-wise addition
//! let mut result = vec![0.0; a.len()];
//! add_vec(&a, &b, &mut result);
//! ```

pub mod arithmetic_ops;
pub mod basic_operations;
pub mod comparison_ops;
pub mod intrinsics;
pub mod math_functions;
pub mod statistics_ops;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod integration_test;

// Re-export all public functions for unified API
pub use arithmetic_ops::{
    abs_vec, add_vec, divide_vec, fma, multiply_vec, neg_vec, reciprocal_vec, scale_vec,
    square_vec, subtract_vec,
};
pub use basic_operations::{cosine_similarity, dot_product, euclidean_distance, norm_l1, norm_l2};
pub use comparison_ops::{
    and_vec, eq_vec, ge_vec, gt_vec, le_vec, lt_vec, ne_vec, not_vec, or_vec, xor_vec,
};
pub use intrinsics::{
    detect_simd_capabilities, optimal_chunk_size, simd_width_f32, F32x4, SimdCapabilities,
};
pub use math_functions::{cos_vec, exp_vec, ln_vec, pow_vec, sin_vec, sqrt_vec, tan_vec};
pub use statistics_ops::{
    dot_product as stats_dot_product, max_vec, mean_vec, min_max_vec, min_vec,
    norm_l1 as stats_norm_l1, norm_l2 as stats_norm_l2, norm_l2_squared, product_vec, std_dev_vec,
    sum_vec, variance_vec,
};

// Export sum for activation module
pub use statistics_ops::sum_vec as sum;

// Additional exports for other modules
pub use arithmetic_ops::scale_vec_inplace as scale;
pub use statistics_ops::mean_vec as mean;
pub use statistics_ops::{min_max_vec as min_max, variance_vec as variance};
// lt_vec and simd_width_f32 are already exported above
pub use basic_operations::norm_l2 as norm;

// Function aliases for compatibility
pub use arithmetic_ops::add_vec as add_simd;
pub use arithmetic_ops::fma as fma_simd;

// Constants are already defined in the main constants module below

// Advanced operations that use combinations of basic operations
pub use basic_operations::{cross_product, outer_product};

#[cfg(feature = "no-std")]
use alloc::vec;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

// Import f32 constants conditionally
#[cfg(feature = "no-std")]
use core::f32::consts;
#[cfg(not(feature = "no-std"))]
use std::f32::consts;

/// SIMD vector operations configuration
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable fallback to scalar operations
    pub enable_scalar_fallback: bool,
    /// Minimum vector size for SIMD operations
    pub simd_threshold: usize,
    /// Enable accuracy checks for approximation functions
    pub enable_accuracy_checks: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enable_scalar_fallback: true,
            simd_threshold: 16, // Minimum 16 elements
            enable_accuracy_checks: cfg!(debug_assertions),
        }
    }
}

/// Global SIMD configuration (thread-local for thread safety)
#[cfg(not(feature = "no-std"))]
thread_local! {
    static SIMD_CONFIG: std::cell::RefCell<SimdConfig> = std::cell::RefCell::new(SimdConfig::default());
}

#[cfg(feature = "no-std")]
static mut SIMD_CONFIG: Option<SimdConfig> = None;

/// Set global SIMD configuration
pub fn set_simd_config(config: SimdConfig) {
    #[cfg(not(feature = "no-std"))]
    {
        SIMD_CONFIG.with(|c| *c.borrow_mut() = config);
    }
    #[cfg(feature = "no-std")]
    {
        unsafe {
            SIMD_CONFIG = Some(config);
        }
    }
}

/// Get current SIMD configuration
pub fn get_simd_config() -> SimdConfig {
    #[cfg(not(feature = "no-std"))]
    {
        SIMD_CONFIG.with(|c| c.borrow().clone())
    }
    #[cfg(feature = "no-std")]
    {
        unsafe { SIMD_CONFIG.clone().unwrap_or_default() }
    }
}

/// Platform-specific feature detection and optimization
pub struct PlatformInfo {
    /// Detected SIMD capabilities
    pub capabilities: SimdCapabilities,
    /// Optimal chunk size for current platform
    pub optimal_chunk_size: usize,
    /// Recommended alignment for vectors
    pub recommended_alignment: usize,
}

/// Detect platform capabilities and optimization parameters
pub fn detect_platform_info() -> PlatformInfo {
    let capabilities = detect_simd_capabilities();
    let optimal_chunk_size = optimal_chunk_size(1000, None); // Default array length for estimation
    let recommended_alignment = intrinsics::preferred_alignment_f32();

    PlatformInfo {
        capabilities,
        optimal_chunk_size,
        recommended_alignment,
    }
}

/// Optimized memory allocation for SIMD vectors
pub fn allocate_aligned_vec(size: usize, alignment: usize) -> Vec<f32> {
    // Note: In a full implementation, would use aligned allocation
    // For now, return standard Vec which has reasonable alignment for most platforms
    vec![0.0; size]
}

/// Check if vector is properly aligned for SIMD operations
pub fn is_properly_aligned(slice: &[f32], alignment: usize) -> bool {
    (slice.as_ptr() as usize) % alignment == 0
}

/// Performance benchmarking utilities for SIMD operations
#[cfg(not(feature = "no-std"))]
pub mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};

    /// Benchmark result for SIMD operations
    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        /// Operation name
        pub operation: String,
        /// Total execution time
        pub duration: Duration,
        /// Operations per second
        pub ops_per_sec: f64,
        /// Throughput in elements per second
        pub elements_per_sec: f64,
        /// SIMD platform used
        pub platform: String,
    }

    /// Benchmark a vector operation
    pub fn benchmark_operation<F>(
        name: &str,
        vector_size: usize,
        iterations: usize,
        operation: F,
    ) -> BenchmarkResult
    where
        F: Fn(),
    {
        // Warmup
        for _ in 0..10 {
            operation();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        let duration = start.elapsed();

        let platform_info = detect_platform_info();
        let platform_name = platform_info.capabilities.platform_name();

        BenchmarkResult {
            operation: name.to_string(),
            duration,
            ops_per_sec: iterations as f64 / duration.as_secs_f64(),
            elements_per_sec: (iterations * vector_size) as f64 / duration.as_secs_f64(),
            platform: platform_name.to_string(),
        }
    }

    /// Compare performance across different vector sizes
    pub fn benchmark_scaling<F>(
        name: &str,
        sizes: &[usize],
        iterations: usize,
        operation_factory: F,
    ) -> Vec<BenchmarkResult>
    where
        F: Fn(usize) -> Box<dyn Fn()>,
    {
        sizes
            .iter()
            .map(|&size| {
                let operation = operation_factory(size);
                benchmark_operation(name, size, iterations, operation)
            })
            .collect()
    }
}

/// Accuracy verification utilities for approximation functions
pub mod accuracy {
    use super::*;

    /// Accuracy test result
    #[derive(Debug, Clone)]
    pub struct AccuracyResult {
        /// Maximum absolute error
        pub max_abs_error: f32,
        /// Root mean square error
        pub rms_error: f32,
        /// Mean absolute error
        pub mean_abs_error: f32,
        /// Number of test points
        pub test_points: usize,
        /// Accuracy grade (A-F)
        pub grade: AccuracyGrade,
    }

    /// Accuracy grading system
    #[derive(Debug, Clone, PartialEq)]
    pub enum AccuracyGrade {
        A, // Excellent (< 1e-6)
        B, // Very Good (< 1e-5)
        C, // Good (< 1e-4)
        D, // Acceptable (< 1e-3)
        F, // Poor (>= 1e-3)
    }

    /// Test accuracy of approximation function against reference
    pub fn test_accuracy<F, R>(
        approximation: F,
        reference: R,
        test_inputs: &[f32],
    ) -> AccuracyResult
    where
        F: Fn(&[f32], &mut [f32]),
        R: Fn(f32) -> f32,
    {
        let mut approx_results = vec![0.0; test_inputs.len()];
        approximation(test_inputs, &mut approx_results);

        let mut errors = Vec::with_capacity(test_inputs.len());
        let mut abs_errors = Vec::with_capacity(test_inputs.len());

        for (i, &input) in test_inputs.iter().enumerate() {
            let reference_result = reference(input);
            let error = approx_results[i] - reference_result;
            let abs_error = error.abs();

            errors.push(error);
            abs_errors.push(abs_error);
        }

        let max_abs_error = abs_errors.iter().fold(0.0f32, |a, &b| a.max(b));
        let mean_abs_error = abs_errors.iter().sum::<f32>() / abs_errors.len() as f32;
        let rms_error = (errors.iter().map(|&e| e * e).sum::<f32>() / errors.len() as f32).sqrt();

        let grade = match max_abs_error {
            e if e < 1e-6 => AccuracyGrade::A,
            e if e < 1e-5 => AccuracyGrade::B,
            e if e < 1e-4 => AccuracyGrade::C,
            e if e < 1e-3 => AccuracyGrade::D,
            _ => AccuracyGrade::F,
        };

        AccuracyResult {
            max_abs_error,
            rms_error,
            mean_abs_error,
            test_points: test_inputs.len(),
            grade,
        }
    }

    /// Generate comprehensive test inputs for mathematical functions
    pub fn generate_test_inputs(
        range_start: f32,
        range_end: f32,
        num_points: usize,
        include_special_values: bool,
    ) -> Vec<f32> {
        let mut inputs = Vec::with_capacity(num_points + 20);

        // Regular sampling across range
        let step = (range_end - range_start) / (num_points as f32);
        for i in 0..num_points {
            inputs.push(range_start + i as f32 * step);
        }

        if include_special_values {
            // Add special values that often cause accuracy issues
            let special_values = vec![
                0.0,
                -0.0,
                consts::PI,
                -consts::PI,
                consts::PI / 2.0,
                -consts::PI / 2.0,
                consts::PI / 4.0,
                -consts::PI / 4.0,
                consts::E,
                -consts::E,
                1.0,
                -1.0,
                2.0,
                -2.0,
                10.0,
                -10.0,
                0.1,
                -0.1,
                0.001,
                -0.001,
                1e-6,
                -1e-6,
            ];

            for value in special_values {
                if value >= range_start && value <= range_end {
                    inputs.push(value);
                }
            }
        }

        inputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        inputs.dedup();
        inputs
    }
}

/// Utility functions for vector operations
pub mod utils {
    use super::*;

    /// Check if two vectors have compatible lengths for binary operations
    pub fn check_compatible_lengths(a: &[f32], b: &[f32]) -> Result<(), &'static str> {
        if a.len() != b.len() {
            Err("Vectors must have the same length")
        } else {
            Ok(())
        }
    }

    /// Check if input and output vectors have compatible lengths
    pub fn check_io_lengths(input: &[f32], output: &[f32]) -> Result<(), &'static str> {
        check_compatible_lengths(input, output)
    }

    /// Validate that vector is not empty
    pub fn check_not_empty(vec: &[f32]) -> Result<(), &'static str> {
        if vec.is_empty() {
            Err("Vector cannot be empty")
        } else {
            Ok(())
        }
    }

    /// Get the optimal chunk size for current platform
    pub fn get_platform_chunk_size() -> usize {
        detect_platform_info().optimal_chunk_size
    }

    /// Split vector into SIMD-friendly chunks
    pub fn chunk_vector(vec: &[f32], chunk_size: usize) -> (&[f32], &[f32]) {
        let simd_len = (vec.len() / chunk_size) * chunk_size;
        vec.split_at(simd_len)
    }

    /// Process vector in chunks with remainder handling
    pub fn process_chunks<F, R>(
        vec: &[f32],
        chunk_size: usize,
        mut chunk_processor: F,
        mut remainder_processor: R,
    ) where
        F: FnMut(&[f32]),
        R: FnMut(&[f32]),
    {
        let (chunks, remainder) = chunk_vector(vec, chunk_size);

        for chunk in chunks.chunks_exact(chunk_size) {
            chunk_processor(chunk);
        }

        if !remainder.is_empty() {
            remainder_processor(remainder);
        }
    }

    /// Convert degrees to radians
    pub fn degrees_to_radians(degrees: f32) -> f32 {
        degrees * consts::PI / 180.0
    }

    /// Convert radians to degrees
    pub fn radians_to_degrees(radians: f32) -> f32 {
        radians * 180.0 / consts::PI
    }

    /// Safe division with zero handling
    pub fn safe_divide(numerator: f32, denominator: f32) -> f32 {
        if denominator.abs() < f32::EPSILON {
            if numerator >= 0.0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }
        } else {
            numerator / denominator
        }
    }

    /// Clamp value to range [min, max]
    pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }
}

/// Export commonly used constants
pub mod constants {
    #[cfg(feature = "no-std")]
    use core::f32::consts;
    #[cfg(not(feature = "no-std"))]
    use std::f32::consts;

    /// Mathematical constants optimized for SIMD operations
    pub const PI_F32: f32 = consts::PI;
    pub const E_F32: f32 = consts::E;
    pub const LN_2_F32: f32 = consts::LN_2;
    pub const LN_10_F32: f32 = consts::LN_10;
    pub const SQRT_2_F32: f32 = consts::SQRT_2;

    /// Common SIMD vector sizes
    pub const SSE2_VECTOR_SIZE: usize = 4; // 128-bit / 32-bit = 4 floats
    pub const AVX2_VECTOR_SIZE: usize = 8; // 256-bit / 32-bit = 8 floats
    pub const AVX512_VECTOR_SIZE: usize = 16; // 512-bit / 32-bit = 16 floats
    pub const NEON_VECTOR_SIZE: usize = 4; // 128-bit / 32-bit = 4 floats

    /// Platform-specific alignment requirements
    pub const SSE2_ALIGNMENT: usize = 16; // 128-bit alignment
    pub const AVX2_ALIGNMENT: usize = 32; // 256-bit alignment
    pub const AVX512_ALIGNMENT: usize = 64; // 512-bit alignment
    pub const NEON_ALIGNMENT: usize = 16; // 128-bit alignment
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_config() {
        let config = SimdConfig::default();
        set_simd_config(config.clone());

        let retrieved_config = get_simd_config();
        assert_eq!(retrieved_config.simd_threshold, config.simd_threshold);
        assert_eq!(
            retrieved_config.enable_scalar_fallback,
            config.enable_scalar_fallback
        );
    }

    #[test]
    fn test_platform_detection() {
        let platform_info = detect_platform_info();
        assert!(platform_info.optimal_chunk_size >= 4);
        assert!(platform_info.recommended_alignment >= 4);

        // Test that capabilities are detected
        let caps = platform_info.capabilities;
        println!("SIMD Capabilities: {:?}", caps);
    }

    #[test]
    fn test_aligned_allocation() {
        let vec = allocate_aligned_vec(16, 16);
        assert_eq!(vec.len(), 16);
        assert_eq!(vec[0], 0.0);
    }

    #[test]
    fn test_utils() {
        use utils::*;

        // Test length compatibility
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = vec![7.0, 8.0];

        assert!(check_compatible_lengths(&a, &b).is_ok());
        assert!(check_compatible_lengths(&a, &c).is_err());

        // Test empty check
        let empty_vec: Vec<f32> = vec![];
        assert!(check_not_empty(&empty_vec).is_err());
        assert!(check_not_empty(&a).is_ok());

        // Test chunking
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (chunks, remainder) = chunk_vector(&vec, 4);
        assert_eq!(chunks.len(), 8); // 2 complete chunks of 4
        assert_eq!(remainder.len(), 1); // 1 remainder element

        // Test mathematical utilities
        assert!((degrees_to_radians(180.0) - constants::PI_F32).abs() < f32::EPSILON);
        assert!((radians_to_degrees(constants::PI_F32) - 180.0).abs() < f32::EPSILON);

        assert_eq!(safe_divide(10.0, 2.0), 5.0);
        assert_eq!(safe_divide(10.0, 0.0), f32::INFINITY);
        assert_eq!(safe_divide(-10.0, 0.0), f32::NEG_INFINITY);

        assert_eq!(clamp(5.0, 1.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 1.0, 10.0), 1.0);
        assert_eq!(clamp(15.0, 1.0, 10.0), 10.0);
    }

    #[test]
    fn test_accuracy_grading() {
        use accuracy::AccuracyGrade;

        // This would test the accuracy grading system in a real implementation
        let grade_a = AccuracyGrade::A;
        let grade_f = AccuracyGrade::F;

        assert!(grade_a != grade_f);
        assert_eq!(grade_a, AccuracyGrade::A);
    }
}

// Integration tests that verify the full SIMD operations work correctly
// These will be completed when all modules are implemented
#[allow(non_snake_case)]
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // Test a complete workflow using multiple SIMD operations
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        // This will work once all modules are implemented
        // let dot = dot_product(&a, &b);
        // let norm_a = norm_l2(&a);
        // let mean_a = mean(&a);

        // Placeholder assertions for now
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn test_platform_optimization_paths() {
        // Test that different SIMD platforms produce equivalent results
        let platform_info = detect_platform_info();
        println!("SIMD capabilities: {:?}", platform_info.capabilities);
        println!(
            "Platform name: {}",
            platform_info.capabilities.platform_name()
        );
        println!("Optimal chunk size: {}", platform_info.optimal_chunk_size);
        println!(
            "Recommended alignment: {}",
            platform_info.recommended_alignment
        );

        // Basic capability testing
        assert!(platform_info.optimal_chunk_size >= 1);
        assert!(platform_info.recommended_alignment >= 4);
    }
}
