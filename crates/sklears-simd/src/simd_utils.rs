//! SIMD Utility Functions and Helper Routines
//!
//! This module provides utility functions, memory management helpers, benchmarking tools,
//! and low-level intrinsics support for the SIMD vector operations framework.
//!
//! # Key Features
//!
//! - **Memory Management**: Aligned allocation, prefetching, cache optimization
//! - **Benchmarking**: Performance measurement and profiling utilities
//! - **Validation**: Input validation and bounds checking
//! - **Debug Support**: Debugging aids and diagnostic tools
//! - **Conversion Utilities**: Data format conversion and transformation helpers
//! - **Platform Abstraction**: Cross-platform compatibility utilities

#[cfg(feature = "no-std")]
use alloc::{string::String, vec::Vec, boxed::Box};
#[cfg(not(feature = "no-std"))]
use std::{string::String, vec::Vec, boxed::Box};

use crate::simd_types::{
    SimdInstructionSet, VectorError, VectorPerformanceMetrics, VectorConfig,
    ArchitectureInfo, VectorErrorCategory, VectorErrorSeverity
};

/// Memory alignment utilities for SIMD operations
pub mod memory {
    use super::*;

    /// Check if a pointer is aligned to the specified boundary
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// Get the next aligned address for the given alignment
    pub fn align_up(addr: usize, alignment: usize) -> usize {
        (addr + alignment - 1) & !(alignment - 1)
    }

    /// Get the previous aligned address for the given alignment
    pub fn align_down(addr: usize, alignment: usize) -> usize {
        addr & !(alignment - 1)
    }

    /// Calculate padding needed to reach alignment
    pub fn alignment_padding(addr: usize, alignment: usize) -> usize {
        let aligned = align_up(addr, alignment);
        aligned - addr
    }

    /// Check if vector data is properly aligned for SIMD operations
    pub fn check_vector_alignment(data: &[f32], simd_set: SimdInstructionSet) -> bool {
        let required_alignment = simd_set.required_alignment();
        is_aligned(data.as_ptr(), required_alignment)
    }

    /// Prefetch memory for better cache performance
    #[inline]
    pub fn prefetch_read<T>(ptr: *const T, locality: i32) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "no-std")]
            use core::arch::x86_64::*;
            #[cfg(not(feature = "no-std"))]
            use core::arch::x86_64::*;

            unsafe {
                match locality {
                    0 => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
                    1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
                    2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
                    _ => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
                }
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // No-op on other architectures
            let _ = (ptr, locality);
        }
    }

    /// Prefetch memory for write operations
    #[inline]
    pub fn prefetch_write<T>(ptr: *const T) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "no-std")]
            use core::arch::x86_64::*;
            #[cfg(not(feature = "no-std"))]
            use core::arch::x86_64::*;

            unsafe {
                _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // No-op on other architectures
            let _ = ptr;
        }
    }

    /// Create aligned vector with specified alignment
    pub fn create_aligned_vector(size: usize, alignment: usize) -> Vec<f32> {
        let mut vec = Vec::with_capacity(size + alignment / 4);

        // Add padding elements to ensure we can get proper alignment
        let ptr = vec.as_ptr() as usize;
        let padding = alignment_padding(ptr, alignment) / 4; // f32 size

        vec.resize(size + padding, 0.0);

        // Find aligned start within the vector
        let aligned_start = align_up(vec.as_ptr() as usize, alignment) as usize / 4;
        let start_offset = aligned_start - (vec.as_ptr() as usize / 4);

        // Return properly sized vector starting from aligned position
        if start_offset + size <= vec.len() {
            vec.drain(0..start_offset);
            vec.truncate(size);
        }

        vec
    }

    /// Memory bandwidth estimation utility
    pub fn estimate_memory_bandwidth(vector_size: usize, operations: usize, time_ns: u64) -> f64 {
        let bytes_transferred = vector_size * operations * 4; // f32 = 4 bytes
        let time_seconds = time_ns as f64 / 1_000_000_000.0;
        let bandwidth_bps = bytes_transferred as f64 / time_seconds;
        bandwidth_bps / (1024.0 * 1024.0 * 1024.0) // Convert to GB/s
    }
}

/// Benchmarking and performance measurement utilities
pub mod benchmark {
    use super::*;
    #[cfg(not(feature = "no-std"))]
    use std::time::{Instant, Duration};

    /// Simple benchmark runner for vector operations
    #[cfg(not(feature = "no-std"))]
    pub fn benchmark_operation<F, T>(
        name: &str,
        operation: F,
        iterations: usize,
    ) -> BenchmarkResult
    where
        F: Fn() -> T,
    {
        // Warm-up
        for _ in 0..10 {
            let _ = operation();
        }

        let start = Instant::now();

        for _ in 0..iterations {
            let _ = operation();
        }

        let elapsed = start.elapsed();

        BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_time: elapsed,
            avg_time_ns: elapsed.as_nanos() as u64 / iterations as u64,
            operations_per_sec: iterations as f64 / elapsed.as_secs_f64(),
        }
    }

    /// Benchmark result summary
    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        pub name: String,
        pub iterations: usize,
        pub total_time: Duration,
        pub avg_time_ns: u64,
        pub operations_per_sec: f64,
    }

    impl BenchmarkResult {
        /// Compare with another benchmark result
        pub fn speedup_vs(&self, other: &BenchmarkResult) -> f64 {
            other.avg_time_ns as f64 / self.avg_time_ns as f64
        }

        /// Get throughput in operations per second
        pub fn throughput(&self) -> f64 {
            self.operations_per_sec
        }

        /// Get average latency in nanoseconds
        pub fn avg_latency_ns(&self) -> u64 {
            self.avg_time_ns
        }
    }

    /// Micro-benchmark utilities for specific SIMD operations
    pub struct SimdBenchmarker {
        pub config: VectorConfig,
        pub warmup_iterations: usize,
        pub measurement_iterations: usize,
    }

    impl Default for SimdBenchmarker {
        fn default() -> Self {
            Self {
                config: VectorConfig::default(),
                warmup_iterations: 100,
                measurement_iterations: 1000,
            }
        }
    }

    impl SimdBenchmarker {
        /// Create new benchmarker with specific configuration
        pub fn new(config: VectorConfig) -> Self {
            Self {
                config,
                warmup_iterations: 100,
                measurement_iterations: 1000,
            }
        }

        /// Benchmark vector dot product operation
        #[cfg(not(feature = "no-std"))]
        pub fn benchmark_dot_product(&self, size: usize) -> BenchmarkResult {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

            benchmark_operation(
                &format!("dot_product_{}", size),
                || {
                    crate::simd_basic::dot_product(&a, &b)
                },
                self.measurement_iterations,
            )
        }

        /// Benchmark vector addition operation
        #[cfg(not(feature = "no-std"))]
        pub fn benchmark_vector_add(&self, size: usize) -> BenchmarkResult {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

            benchmark_operation(
                &format!("vector_add_{}", size),
                || {
                    crate::simd_basic::add_vectors(&a, &b)
                },
                self.measurement_iterations,
            )
        }
    }
}

/// Input validation utilities
pub mod validation {
    use super::*;

    /// Validate that two vectors have the same length
    pub fn validate_same_length<T>(a: &[T], b: &[T], operation: &str) -> Result<(), VectorError> {
        if a.len() != b.len() {
            Err(VectorError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
                operation: operation.to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Validate that vector is not empty
    pub fn validate_non_empty<T>(vec: &[T], operation: &str) -> Result<(), VectorError> {
        if vec.is_empty() {
            Err(VectorError::EmptyVector {
                operation: operation.to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Validate vector size constraints
    pub fn validate_size_constraints<T>(
        vec: &[T],
        min_size: Option<usize>,
        max_size: Option<usize>,
        operation: &str,
    ) -> Result<(), VectorError> {
        let size = vec.len();

        if let Some(min) = min_size {
            if size < min {
                return Err(VectorError::InvalidSize {
                    size,
                    min_required: min,
                    max_allowed: max_size,
                    operation: operation.to_string(),
                });
            }
        }

        if let Some(max) = max_size {
            if size > max {
                return Err(VectorError::InvalidSize {
                    size,
                    min_required: min_size.unwrap_or(0),
                    max_allowed: Some(max),
                    operation: operation.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Validate that all values are finite (not NaN or infinite)
    pub fn validate_finite_values(vec: &[f32], operation: &str) -> Result<(), VectorError> {
        for (i, &value) in vec.iter().enumerate() {
            if !value.is_finite() {
                return Err(VectorError::NumericalError {
                    message: format!("Non-finite value at index {}: {}", i, value),
                    value: Some(value),
                });
            }
        }
        Ok(())
    }

    /// Validate vector for specific SIMD operation requirements
    pub fn validate_for_simd<T>(
        vec: &[T],
        simd_set: SimdInstructionSet,
        operation: &str,
    ) -> Result<(), VectorError> {
        let required_alignment = simd_set.required_alignment();

        if !memory::is_aligned(vec.as_ptr(), required_alignment) {
            return Err(VectorError::AlignmentError {
                address: vec.as_ptr() as usize,
                required_alignment,
                simd_set,
            });
        }

        // Check minimum size for SIMD efficiency
        let min_size = simd_set.vector_width();
        if vec.len() < min_size && simd_set != SimdInstructionSet::Scalar {
            return Err(VectorError::InvalidSize {
                size: vec.len(),
                min_required: min_size,
                max_allowed: None,
                operation: operation.to_string(),
            });
        }

        Ok(())
    }
}

/// Debug and diagnostic utilities
pub mod debug {
    use super::*;

    /// Print vector operation diagnostics
    pub fn print_vector_info<T>(vec: &[T], name: &str) {
        println!("Vector '{}': length={}, ptr={:p}", name, vec.len(), vec.as_ptr());
        println!("  Alignment: {} bytes", vec.as_ptr() as usize % 64);
    }

    /// Print SIMD capability information
    pub fn print_simd_capabilities() {
        let arch = ArchitectureInfo::detect();
        println!("SIMD Capabilities:");
        println!("  Best SIMD: {:?}", arch.best_simd);
        println!("  Available: {:?}", arch.available_simd);
        println!("  CPU cores: {}", arch.cpu_cores);
        println!("  Cache line size: {} bytes", arch.cache_line_size);
    }

    /// Debug-friendly vector content display (limited elements)
    pub fn debug_vector_content(vec: &[f32], name: &str, max_elements: usize) {
        print!("Vector '{}' [{}]: [", name, vec.len());

        let display_count = vec.len().min(max_elements);
        for i in 0..display_count {
            if i > 0 { print!(", "); }
            print!("{:.3}", vec[i]);
        }

        if vec.len() > max_elements {
            print!(", ... ({} more)", vec.len() - max_elements);
        }

        println!("]");
    }

    /// Performance diagnostics for vector operations
    pub fn diagnose_performance_issues(metrics: &VectorPerformanceMetrics) -> Vec<String> {
        let mut issues = Vec::new();

        if metrics.fallback_used {
            issues.push("SIMD fallback to scalar operations detected".to_string());
        }

        if metrics.cache_hit_rate < 0.8 {
            issues.push(format!("Low cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0));
        }

        if metrics.efficiency_percent() < 50.0 {
            issues.push(format!("Low SIMD efficiency: {:.1}%", metrics.efficiency_percent()));
        }

        if metrics.simd_lanes_used < metrics.simd_used.vector_width() {
            issues.push("Not all SIMD lanes utilized".to_string());
        }

        issues
    }
}

/// Data conversion and transformation utilities
pub mod conversion {
    use super::*;

    /// Convert vector of one type to another
    pub fn convert_vector<T, U>(input: &[T]) -> Vec<U>
    where
        T: Clone + Into<U>,
    {
        input.iter().cloned().map(|x| x.into()).collect()
    }

    /// Convert between different floating-point precisions
    pub fn f64_to_f32(input: &[f64]) -> Vec<f32> {
        input.iter().map(|&x| x as f32).collect()
    }

    /// Convert between different floating-point precisions
    pub fn f32_to_f64(input: &[f32]) -> Vec<f64> {
        input.iter().map(|&x| x as f64).collect()
    }

    /// Convert integer vector to floating-point
    pub fn i32_to_f32(input: &[i32]) -> Vec<f32> {
        input.iter().map(|&x| x as f32).collect()
    }

    /// Convert floating-point vector to integer (with rounding)
    pub fn f32_to_i32_rounded(input: &[f32]) -> Vec<i32> {
        input.iter().map(|&x| x.round() as i32).collect()
    }

    /// Interleave two vectors (useful for complex numbers, stereo audio, etc.)
    pub fn interleave<T: Clone>(a: &[T], b: &[T]) -> Vec<T> {
        let mut result = Vec::with_capacity(a.len() + b.len());
        let min_len = a.len().min(b.len());

        for i in 0..min_len {
            result.push(a[i].clone());
            result.push(b[i].clone());
        }

        // Add remaining elements from longer vector
        if a.len() > min_len {
            result.extend_from_slice(&a[min_len..]);
        } else if b.len() > min_len {
            result.extend_from_slice(&b[min_len..]);
        }

        result
    }

    /// Deinterleave vector into two separate vectors
    pub fn deinterleave<T: Clone>(input: &[T]) -> (Vec<T>, Vec<T>) {
        let mut a = Vec::with_capacity(input.len() / 2 + 1);
        let mut b = Vec::with_capacity(input.len() / 2 + 1);

        for (i, item) in input.iter().enumerate() {
            if i % 2 == 0 {
                a.push(item.clone());
            } else {
                b.push(item.clone());
            }
        }

        (a, b)
    }

    /// Pack boolean mask into bit vector
    pub fn pack_bool_mask(mask: &[bool]) -> Vec<u8> {
        let byte_count = (mask.len() + 7) / 8;
        let mut result = vec![0u8; byte_count];

        for (i, &bit) in mask.iter().enumerate() {
            if bit {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                result[byte_idx] |= 1 << bit_idx;
            }
        }

        result
    }

    /// Unpack bit vector into boolean mask
    pub fn unpack_bool_mask(packed: &[u8], length: usize) -> Vec<bool> {
        let mut result = Vec::with_capacity(length);

        for i in 0..length {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            if byte_idx < packed.len() {
                let bit_set = (packed[byte_idx] & (1 << bit_idx)) != 0;
                result.push(bit_set);
            } else {
                result.push(false);
            }
        }

        result
    }
}

/// Platform-specific utilities and abstractions
pub mod platform {
    use super::*;

    /// Get cache line size for current platform
    pub fn get_cache_line_size() -> usize {
        // Most modern CPUs use 64-byte cache lines
        64
    }

    /// Get page size for current platform
    pub fn get_page_size() -> usize {
        #[cfg(unix)]
        {
            unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
        }
        #[cfg(not(unix))]
        {
            4096 // Common default
        }
    }

    /// Check if running on little-endian architecture
    pub fn is_little_endian() -> bool {
        cfg!(target_endian = "little")
    }

    /// Get optimal number of worker threads for parallel operations
    pub fn get_optimal_thread_count() -> usize {
        num_cpus::get().max(1)
    }

    /// Platform-specific optimization hints
    pub struct PlatformOptimizations {
        pub use_prefetch: bool,
        pub prefer_aligned_access: bool,
        pub cache_line_size: usize,
        pub page_size: usize,
        pub numa_aware: bool,
    }

    impl PlatformOptimizations {
        /// Detect platform-specific optimizations
        pub fn detect() -> Self {
            Self {
                use_prefetch: cfg!(any(target_arch = "x86", target_arch = "x86_64")),
                prefer_aligned_access: true,
                cache_line_size: get_cache_line_size(),
                page_size: get_page_size(),
                numa_aware: false, // Would need runtime detection
            }
        }
    }
}

/// Error handling utilities
pub fn create_dimension_mismatch_error(
    expected: usize,
    actual: usize,
    operation: &str,
) -> VectorError {
    VectorError::DimensionMismatch {
        expected,
        actual,
        operation: operation.to_string(),
    }
}

pub fn create_empty_vector_error(operation: &str) -> VectorError {
    VectorError::EmptyVector {
        operation: operation.to_string(),
    }
}

pub fn create_alignment_error(
    address: usize,
    required_alignment: usize,
    simd_set: SimdInstructionSet,
) -> VectorError {
    VectorError::AlignmentError {
        address,
        required_alignment,
        simd_set,
    }
}

/// Utility macros for common SIMD operations (would be defined if needed)
#[macro_export]
macro_rules! simd_fallback {
    ($simd_fn:expr, $scalar_fn:expr, $enable_simd:expr) => {
        if $enable_simd {
            $simd_fn
        } else {
            $scalar_fn
        }
    };
}

/// Utility functions for working with SIMD vectors
pub fn create_test_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    (a, b)
}

pub fn random_test_vector(size: usize, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducible random numbers
    let mut rng = seed;
    let mut result = Vec::with_capacity(size);

    for _ in 0..size {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = (rng as f32) / (u64::MAX as f32);
        result.push(normalized * 2.0 - 1.0); // Range [-1, 1]
    }

    result
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_alignment() {
        let ptr = 0x1000usize as *const f32;
        assert!(memory::is_aligned(ptr, 16));

        let ptr2 = 0x1001usize as *const f32;
        assert!(!memory::is_aligned(ptr2, 16));

        assert_eq!(memory::align_up(0x1001, 16), 0x1010);
        assert_eq!(memory::align_down(0x1001, 16), 0x1000);
    }

    #[test]
    fn test_validation_functions() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = vec![7.0, 8.0];

        assert!(validation::validate_same_length(&a, &b, "test").is_ok());
        assert!(validation::validate_same_length(&a, &c, "test").is_err());

        assert!(validation::validate_non_empty(&a, "test").is_ok());
        assert!(validation::validate_non_empty(&Vec::<f32>::new(), "test").is_err());

        let finite_vec = vec![1.0, 2.0, 3.0];
        assert!(validation::validate_finite_values(&finite_vec, "test").is_ok());

        let infinite_vec = vec![1.0, f32::INFINITY, 3.0];
        assert!(validation::validate_finite_values(&infinite_vec, "test").is_err());
    }

    #[test]
    fn test_conversion_functions() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        let interleaved = conversion::interleave(&a, &b);
        assert_eq!(interleaved, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let (deint_a, deint_b) = conversion::deinterleave(&interleaved);
        assert_eq!(deint_a, a);
        assert_eq!(deint_b, b);

        let bool_mask = vec![true, false, true, true, false, false, true, false];
        let packed = conversion::pack_bool_mask(&bool_mask);
        let unpacked = conversion::unpack_bool_mask(&packed, bool_mask.len());
        assert_eq!(unpacked, bool_mask);
    }

    #[test]
    fn test_error_creation() {
        let error = create_dimension_mismatch_error(4, 3, "test_op");
        assert_eq!(error.category(), VectorErrorCategory::Input);
        assert_eq!(error.severity(), VectorErrorSeverity::High);

        let empty_error = create_empty_vector_error("test_op");
        assert_eq!(empty_error.category(), VectorErrorCategory::Input);
    }

    #[test]
    fn test_test_vector_creation() {
        let (a, b) = create_test_vectors(5);
        assert_eq!(a, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b, vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let random_vec = random_test_vector(10, 12345);
        assert_eq!(random_vec.len(), 10);
        assert!(random_vec.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_platform_utilities() {
        let cache_size = platform::get_cache_line_size();
        assert!(cache_size > 0);

        let page_size = platform::get_page_size();
        assert!(page_size > 0);

        let thread_count = platform::get_optimal_thread_count();
        assert!(thread_count > 0);

        let optimizations = platform::PlatformOptimizations::detect();
        assert!(optimizations.cache_line_size > 0);
    }
}