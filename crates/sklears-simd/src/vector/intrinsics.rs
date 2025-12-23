//! # Low-Level SIMD Intrinsics Wrapper
//!
//! Provides a unified abstraction layer over platform-specific SIMD intrinsics.
//! This module offers consistent interfaces for SIMD operations across different
//! architectures while maintaining maximum performance.
//!
//! ## Features
//!
//! - **Unified Vector Types**: Abstract SIMD vector types (f32x4, f32x8, f32x16)
//! - **Architecture Detection**: Runtime and compile-time SIMD feature detection
//! - **Load/Store Operations**: Memory operations with alignment handling
//! - **Core Intrinsics**: Arithmetic, comparison, and bitwise operations
//! - **Cross-Platform Support**: SSE2, AVX2, AVX512, NEON abstractions
//! - **Fallback Implementation**: Scalar fallbacks when SIMD unavailable
//!
//! ## Usage
//!
//! This module is primarily used internally by higher-level SIMD operations.
//! It provides the building blocks for vectorized computations while hiding
//! platform-specific implementation details.

// Import ARM64 feature detection macro
#[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
use std::arch::is_aarch64_feature_detected;

// Import SIMD arch modules conditionally
#[cfg(all(target_arch = "aarch64", feature = "no-std"))]
use core::arch::aarch64;
#[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
use std::arch::aarch64;

/// SIMD capabilities detected at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    pub sse2: bool,
    pub sse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,
}

impl SimdCapabilities {
    /// Get the platform name for current SIMD capabilities
    pub fn platform_name(&self) -> &'static str {
        if self.avx512f {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX"
        } else if self.sse42 {
            "SSE4.2"
        } else if self.sse41 {
            "SSE4.1"
        } else if self.sse3 {
            "SSE3"
        } else if self.sse2 {
            "SSE2"
        } else if self.neon {
            "NEON"
        } else {
            "Scalar"
        }
    }
}

/// Detect available SIMD capabilities on the current CPU
///
/// This function performs runtime detection of SIMD instruction sets
/// available on the current processor.
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::intrinsics::detect_simd_capabilities;
///
/// let caps = detect_simd_capabilities();
/// println!("AVX2 available: {}", caps.avx2);
/// ```
pub fn detect_simd_capabilities() -> SimdCapabilities {
    SimdCapabilities {
        sse2: detect_sse2(),
        sse3: detect_sse3(),
        sse41: detect_sse41(),
        sse42: detect_sse42(),
        avx: detect_avx(),
        avx2: detect_avx2(),
        avx512f: detect_avx512f(),
        fma: detect_fma(),
        neon: detect_neon(),
    }
}

/// Get the optimal SIMD width for f32 operations on the current CPU
///
/// Returns the number of f32 elements that can be processed in parallel
/// using the best available SIMD instruction set.
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::intrinsics::simd_width_f32;
///
/// let width = simd_width_f32();
/// println!("Can process {} f32 elements in parallel", width);
/// ```
pub fn simd_width_f32() -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return 16; // 512 bits / 32 bits per f32 = 16 elements
        } else if crate::simd_feature_detected!("avx2") {
            return 8; // 256 bits / 32 bits per f32 = 8 elements
        } else if crate::simd_feature_detected!("sse2") {
            return 4; // 128 bits / 32 bits per f32 = 4 elements
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return 4; // 128 bits / 32 bits per f32 = 4 elements
        }
    }

    1 // Scalar fallback
}

/// Calculate the optimal chunk size for processing arrays with SIMD
///
/// This function considers SIMD width, cache line size, and array length
/// to determine the best chunk size for vectorized processing.
///
/// # Arguments
/// * `array_len` - Length of the array to process
/// * `min_chunk` - Minimum chunk size (default: SIMD width)
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::intrinsics::optimal_chunk_size;
///
/// let array_len = 1000;
/// let chunk_size = optimal_chunk_size(array_len, None);
/// println!("Process in chunks of {} elements", chunk_size);
/// ```
pub fn optimal_chunk_size(array_len: usize, min_chunk: Option<usize>) -> usize {
    let simd_width = simd_width_f32().max(1);
    let min_chunk = min_chunk.unwrap_or(simd_width);

    // Prefer multiples of SIMD width while respecting caller preference
    let preferred_chunk = simd_width.max(min_chunk);

    // Treat small workloads specially to avoid over-partitioning
    let small_threshold = preferred_chunk.max(16);
    if array_len <= small_threshold {
        return array_len;
    }

    // For larger arrays, use a cache-friendly multiple of the SIMD width
    let cache_line_f32 = 64 / 4; // Assume 64-byte cache lines, 16 f32 elements
    let optimal = cache_line_f32.max(preferred_chunk);

    // Round down to the nearest multiple of the SIMD width and cap by the array length
    let aligned = ((optimal / simd_width).max(1)) * simd_width;
    aligned.min(array_len)
}

// ============================================================================
// Vector Type Abstractions
// ============================================================================

/// 4-element f32 SIMD vector abstraction
#[derive(Debug, Clone, Copy)]
pub struct F32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    inner: core::arch::x86_64::__m128,
    #[cfg(target_arch = "aarch64")]
    inner: aarch64::float32x4_t,
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    inner: [f32; 4],
}

/// 8-element f32 SIMD vector abstraction
#[derive(Debug, Clone, Copy)]
pub struct F32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    inner: core::arch::x86_64::__m256,
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    inner: [f32; 8],
}

/// 16-element f32 SIMD vector abstraction
#[derive(Debug, Clone, Copy)]
pub struct F32x16 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    inner: core::arch::x86_64::__m512,
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    inner: [f32; 16],
}

impl F32x4 {
    /// Create a new F32x4 with all elements set to the same value
    #[inline]
    pub fn splat(value: f32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                Self {
                    inner: core::arch::x86_64::_mm_set1_ps(value),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                Self {
                    inner: aarch64::vdupq_n_f32(value),
                }
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { inner: [value; 4] }
        }
    }

    /// Create a new F32x4 from four individual values
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                Self {
                    inner: core::arch::x86_64::_mm_setr_ps(a, b, c, d),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                let arr = [a, b, c, d];
                Self {
                    inner: aarch64::vld1q_f32(arr.as_ptr()),
                }
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                inner: [a, b, c, d],
            }
        }
    }

    /// Load four f32 values from memory (aligned)
    #[inline]
    pub unsafe fn load_aligned(ptr: *const f32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                inner: core::arch::x86_64::_mm_load_ps(ptr),
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                inner: aarch64::vld1q_f32(ptr),
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                inner: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
            }
        }
    }

    /// Load four f32 values from memory (unaligned)
    #[inline]
    pub unsafe fn load_unaligned(ptr: *const f32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                inner: core::arch::x86_64::_mm_loadu_ps(ptr),
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                inner: aarch64::vld1q_f32(ptr),
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                inner: [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)],
            }
        }
    }

    /// Store four f32 values to memory (aligned)
    #[inline]
    pub unsafe fn store_aligned(self, ptr: *mut f32) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            core::arch::x86_64::_mm_store_ps(ptr, self.inner);
        }

        #[cfg(target_arch = "aarch64")]
        {
            aarch64::vst1q_f32(ptr, self.inner);
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            *ptr = self.inner[0];
            *ptr.add(1) = self.inner[1];
            *ptr.add(2) = self.inner[2];
            *ptr.add(3) = self.inner[3];
        }
    }

    /// Store four f32 values to memory (unaligned)
    #[inline]
    pub unsafe fn store_unaligned(self, ptr: *mut f32) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            core::arch::x86_64::_mm_storeu_ps(ptr, self.inner);
        }

        #[cfg(target_arch = "aarch64")]
        {
            aarch64::vst1q_f32(ptr, self.inner);
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            *ptr = self.inner[0];
            *ptr.add(1) = self.inner[1];
            *ptr.add(2) = self.inner[2];
            *ptr.add(3) = self.inner[3];
        }
    }

    /// Add two F32x4 vectors
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                Self {
                    inner: core::arch::x86_64::_mm_add_ps(self.inner, other.inner),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                Self {
                    inner: aarch64::vaddq_f32(self.inner, other.inner),
                }
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                inner: [
                    self.inner[0] + other.inner[0],
                    self.inner[1] + other.inner[1],
                    self.inner[2] + other.inner[2],
                    self.inner[3] + other.inner[3],
                ],
            }
        }
    }

    /// Multiply two F32x4 vectors
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                Self {
                    inner: core::arch::x86_64::_mm_mul_ps(self.inner, other.inner),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                Self {
                    inner: aarch64::vmulq_f32(self.inner, other.inner),
                }
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                inner: [
                    self.inner[0] * other.inner[0],
                    self.inner[1] * other.inner[1],
                    self.inner[2] * other.inner[2],
                    self.inner[3] * other.inner[3],
                ],
            }
        }
    }

    /// Horizontal sum of all elements
    #[inline]
    pub fn horizontal_sum(self) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                let temp = core::arch::x86_64::_mm_add_ps(
                    self.inner,
                    core::arch::x86_64::_mm_movehl_ps(self.inner, self.inner),
                );
                let result = core::arch::x86_64::_mm_add_ps(
                    temp,
                    core::arch::x86_64::_mm_shuffle_ps(temp, temp, 0x01),
                );
                core::arch::x86_64::_mm_cvtss_f32(result)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                let sum2 = aarch64::vpadd_f32(
                    aarch64::vget_low_f32(self.inner),
                    aarch64::vget_high_f32(self.inner),
                );
                let sum1 = aarch64::vpadd_f32(sum2, sum2);
                aarch64::vget_lane_f32(sum1, 0)
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.inner[0] + self.inner[1] + self.inner[2] + self.inner[3]
        }
    }

    /// Extract a single element by index
    #[inline]
    pub fn extract(self, index: usize) -> f32 {
        assert!(index < 4, "Index out of bounds");

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                match index {
                    0 => core::arch::x86_64::_mm_cvtss_f32(self.inner),
                    1 => core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_shuffle_ps(
                        self.inner, self.inner, 0x01,
                    )),
                    2 => core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_shuffle_ps(
                        self.inner, self.inner, 0x02,
                    )),
                    3 => core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_shuffle_ps(
                        self.inner, self.inner, 0x03,
                    )),
                    _ => unreachable!(),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                match index {
                    0 => aarch64::vgetq_lane_f32(self.inner, 0),
                    1 => aarch64::vgetq_lane_f32(self.inner, 1),
                    2 => aarch64::vgetq_lane_f32(self.inner, 2),
                    3 => aarch64::vgetq_lane_f32(self.inner, 3),
                    _ => unreachable!(),
                }
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.inner[index]
        }
    }
}

// ============================================================================
// Architecture-specific feature detection
// ============================================================================

#[inline]
fn detect_sse2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("sse2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_sse3() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("sse3")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_sse41() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("sse4.1")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_sse42() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("sse4.2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_avx() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("avx")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_avx512f() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("avx512f")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_fma() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        crate::simd_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn detect_neon() -> bool {
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        is_aarch64_feature_detected!("neon")
    }
    #[cfg(all(target_arch = "aarch64", feature = "no-std"))]
    {
        false
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Check if a pointer is aligned to a specific boundary
#[inline]
pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
    if alignment == 0 || !alignment.is_power_of_two() {
        return false;
    }

    (ptr as usize) & (alignment - 1) == 0
}

/// Align a value up to the nearest multiple of alignment
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }

    let mask = alignment - 1;
    if !alignment.is_power_of_two() {
        // Fallback to modulo arithmetic for non power-of-two alignments.
        return if value % alignment == 0 {
            value
        } else {
            value + (alignment - (value % alignment))
        };
    }

    (value + mask) & !mask
}

/// Align a value down to the nearest multiple of alignment
#[inline]
pub fn align_down(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }

    let mask = alignment - 1;
    if !alignment.is_power_of_two() {
        return value - (value % alignment);
    }

    value & !mask
}

/// Get the preferred alignment for f32 SIMD operations
#[inline]
pub fn preferred_alignment_f32() -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if detect_avx512f() {
            64 // AVX512: 64-byte alignment
        } else if detect_avx2() {
            32 // AVX2: 32-byte alignment
        } else if detect_sse2() {
            16 // SSE2: 16-byte alignment
        } else {
            4 // Scalar: 4-byte alignment
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if detect_neon() {
            16 // NEON: 16-byte alignment
        } else {
            4 // Scalar: 4-byte alignment
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        4 // Scalar: 4-byte alignment
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_simd_capabilities() {
        let caps = detect_simd_capabilities();

        // At least one of these should be available on most modern systems
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(caps.sse2 || caps.avx2 || caps.avx512f);

        #[cfg(target_arch = "aarch64")]
        assert!(caps.neon);

        println!("SIMD capabilities: {:?}", caps);
    }

    #[test]
    fn test_simd_width() {
        let width = simd_width_f32();
        assert!(width >= 1);
        assert!(width <= 16);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if detect_avx512f() {
                assert_eq!(width, 16);
            } else if detect_avx2() {
                assert_eq!(width, 8);
            } else if detect_sse2() {
                assert_eq!(width, 4);
            } else {
                assert_eq!(width, 1);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if detect_neon() {
                assert_eq!(width, 4);
            } else {
                assert_eq!(width, 1);
            }
        }
    }

    #[test]
    fn test_optimal_chunk_size() {
        // Small array
        let small_chunk = optimal_chunk_size(10, None);
        assert_eq!(small_chunk, 10);

        // Large array
        let large_chunk = optimal_chunk_size(1000, None);
        let simd_width = simd_width_f32();
        assert!(large_chunk >= simd_width);
        assert_eq!(large_chunk % simd_width, 0);

        // With minimum chunk size
        let min_chunk = optimal_chunk_size(1000, Some(32));
        assert!(min_chunk >= 32);
    }

    #[test]
    fn test_f32x4_basic_operations() {
        let a = F32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = F32x4::new(5.0, 6.0, 7.0, 8.0);

        // Test extraction
        assert_eq!(a.extract(0), 1.0);
        assert_eq!(a.extract(1), 2.0);
        assert_eq!(a.extract(2), 3.0);
        assert_eq!(a.extract(3), 4.0);

        // Test addition
        let sum = a.add(b);
        assert_eq!(sum.extract(0), 6.0);
        assert_eq!(sum.extract(1), 8.0);
        assert_eq!(sum.extract(2), 10.0);
        assert_eq!(sum.extract(3), 12.0);

        // Test multiplication
        let product = a.mul(b);
        assert_eq!(product.extract(0), 5.0);
        assert_eq!(product.extract(1), 12.0);
        assert_eq!(product.extract(2), 21.0);
        assert_eq!(product.extract(3), 32.0);

        // Test horizontal sum
        assert_eq!(a.horizontal_sum(), 10.0);
    }

    #[test]
    fn test_f32x4_splat() {
        let splat = F32x4::splat(42.0);

        assert_eq!(splat.extract(0), 42.0);
        assert_eq!(splat.extract(1), 42.0);
        assert_eq!(splat.extract(2), 42.0);
        assert_eq!(splat.extract(3), 42.0);

        assert_eq!(splat.horizontal_sum(), 168.0); // 42 * 4 = 168
    }

    #[test]
    fn test_f32x4_load_store() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 8];

        unsafe {
            // Load unaligned
            let vec1 = F32x4::load_unaligned(data.as_ptr());
            let vec2 = F32x4::load_unaligned(data.as_ptr().add(4));

            // Verify loaded values
            assert_eq!(vec1.extract(0), 1.0);
            assert_eq!(vec1.extract(1), 2.0);
            assert_eq!(vec1.extract(2), 3.0);
            assert_eq!(vec1.extract(3), 4.0);

            // Store unaligned
            vec1.store_unaligned(result.as_mut_ptr());
            vec2.store_unaligned(result.as_mut_ptr().add(4));
        }

        assert_eq!(result, data);
    }

    #[test]
    fn test_alignment_functions() {
        #[repr(align(32))]
        struct AlignedBytes([u8; 32]);

        let aligned_storage = AlignedBytes([0u8; 32]);
        // Test alignment detection
        let aligned_ptr = aligned_storage.0.as_ptr();
        let unaligned_ptr = unsafe { aligned_ptr.add(1) };

        assert!(is_aligned(aligned_ptr, 16));
        assert!(!is_aligned(unaligned_ptr, 16));

        // Test alignment utilities
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);

        assert_eq!(align_down(15, 16), 0);
        assert_eq!(align_down(16, 16), 16);
        assert_eq!(align_down(31, 16), 16);
    }

    #[test]
    fn test_preferred_alignment() {
        let alignment = preferred_alignment_f32();

        // Should be a power of 2 and at least 4 bytes
        assert!(alignment >= 4);
        assert!(alignment.is_power_of_two());

        // Should match the SIMD capabilities
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if detect_avx512f() {
                assert_eq!(alignment, 64);
            } else if detect_avx2() {
                assert_eq!(alignment, 32);
            } else if detect_sse2() {
                assert_eq!(alignment, 16);
            } else {
                assert_eq!(alignment, 4);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if detect_neon() {
                assert_eq!(alignment, 16);
            } else {
                assert_eq!(alignment, 4);
            }
        }
    }

    #[test]
    fn test_large_vector_operations() {
        // Test that our abstractions work with realistic data sizes
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut result = vec![0.0; size];

        let simd_width = simd_width_f32();
        let chunks = size / simd_width;

        for i in 0..chunks {
            let offset = i * simd_width;
            unsafe {
                let vec = F32x4::load_unaligned(data.as_ptr().add(offset));
                let doubled = vec.add(vec); // Double each element
                doubled.store_unaligned(result.as_mut_ptr().add(offset));
            }
        }

        // Verify first few elements
        for i in 0..(chunks * simd_width) {
            assert_eq!(result[i], 2.0 * (i as f32));
        }
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_f32x4_extract_out_of_bounds() {
        let vec = F32x4::splat(1.0);
        vec.extract(4); // Should panic
    }
}
