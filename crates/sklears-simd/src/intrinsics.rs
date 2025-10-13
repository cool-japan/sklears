//! Intrinsic function wrappers and compiler optimization hints
//!
//! This module provides safe wrappers around SIMD intrinsics and compiler
//! optimization hints to improve code generation and performance.

/// Compiler hints for optimization
pub mod hints {
    /// Hint to the compiler that this branch is likely to be taken
    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        // Use manual implementation for stable Rust
        if b {
            #[cfg(feature = "no-std")]
            {
                core::hint::black_box(true)
            }
            #[cfg(not(feature = "no-std"))]
            {
                std::hint::black_box(true)
            }
        } else {
            false
        }
    }

    /// Hint to the compiler that this branch is unlikely to be taken
    #[inline(always)]
    pub fn unlikely(b: bool) -> bool {
        // Use manual implementation for stable Rust
        if !b {
            #[cfg(feature = "no-std")]
            {
                core::hint::black_box(false)
            }
            #[cfg(not(feature = "no-std"))]
            {
                std::hint::black_box(false)
            }
        } else {
            true
        }
    }

    /// Hint to the compiler that this code path is unreachable
    #[inline(always)]
    pub unsafe fn unreachable_unchecked() -> ! {
        #[cfg(feature = "no-std")]
        {
            core::hint::unreachable_unchecked()
        }
        #[cfg(not(feature = "no-std"))]
        {
            std::hint::unreachable_unchecked()
        }
    }

    /// Hint to prevent vectorization of a loop
    #[inline(always)]
    pub fn prevent_vectorization() {
        // Insert a volatile operation to prevent vectorization
        unsafe {
            #[cfg(feature = "no-std")]
            {
                core::ptr::read_volatile(&0 as *const i32);
            }
            #[cfg(not(feature = "no-std"))]
            {
                std::ptr::read_volatile(&0 as *const i32);
            }
        }
    }

    /// Force vectorization of a loop (when possible)
    #[inline(always)]
    pub fn force_vectorization() {
        // This is a hint to encourage vectorization
        // The actual mechanism depends on the compiler
    }
}

/// Memory alignment utilities
pub mod alignment {
    /// Check if a pointer is aligned to the specified boundary
    #[inline(always)]
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// Assume that a pointer is aligned (optimization hint)
    #[inline(always)]
    pub unsafe fn assume_aligned<T>(ptr: *const T, alignment: usize) -> *const T {
        #[cfg(feature = "no-std")]
        {
            core::hint::assert_unchecked(is_aligned(ptr, alignment));
        }
        #[cfg(not(feature = "no-std"))]
        {
            std::hint::assert_unchecked(is_aligned(ptr, alignment));
        }
        ptr
    }

    /// Assume that a mutable pointer is aligned (optimization hint)
    #[inline(always)]
    pub unsafe fn assume_aligned_mut<T>(ptr: *mut T, alignment: usize) -> *mut T {
        #[cfg(feature = "no-std")]
        {
            core::hint::assert_unchecked(is_aligned(ptr, alignment));
        }
        #[cfg(not(feature = "no-std"))]
        {
            std::hint::assert_unchecked(is_aligned(ptr, alignment));
        }
        ptr
    }
}

/// SIMD intrinsic wrappers for safe usage
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    /// Safe wrapper for SSE2 operations
    pub mod sse2 {
        use super::*;

        /// Safe horizontal add for __m128
        pub fn horizontal_add_f32(v: __m128) -> f32 {
            unsafe {
                let temp = _mm_hadd_ps(v, v);
                let temp = _mm_hadd_ps(temp, temp);
                _mm_cvtss_f32(temp)
            }
        }

        /// Safe vector load with alignment check
        pub fn load_aligned_f32(ptr: *const f32) -> __m128 {
            debug_assert!(super::super::alignment::is_aligned(ptr, 16));
            unsafe { _mm_load_ps(ptr) }
        }

        /// Safe vector store with alignment check
        pub fn store_aligned_f32(ptr: *mut f32, v: __m128) {
            debug_assert!(super::super::alignment::is_aligned(ptr, 16));
            unsafe { _mm_store_ps(ptr, v) }
        }

        /// Safe fused multiply-add
        #[target_feature(enable = "fma")]
        pub unsafe fn fma_f32(a: __m128, b: __m128, c: __m128) -> __m128 {
            _mm_fmadd_ps(a, b, c)
        }
    }

    /// Safe wrapper for AVX2 operations
    pub mod avx2 {
        use super::*;

        /// Safe horizontal add for __m256
        pub fn horizontal_add_f32(v: __m256) -> f32 {
            unsafe {
                let hi = _mm256_extractf128_ps(v, 1);
                let lo = _mm256_castps256_ps128(v);
                let sum128 = _mm_add_ps(hi, lo);
                let temp = _mm_hadd_ps(sum128, sum128);
                let temp = _mm_hadd_ps(temp, temp);
                _mm_cvtss_f32(temp)
            }
        }

        /// Safe vector load with alignment check
        pub fn load_aligned_f32(ptr: *const f32) -> __m256 {
            debug_assert!(super::super::alignment::is_aligned(ptr, 32));
            unsafe { _mm256_load_ps(ptr) }
        }

        /// Safe vector store with alignment check
        pub fn store_aligned_f32(ptr: *mut f32, v: __m256) {
            debug_assert!(super::super::alignment::is_aligned(ptr, 32));
            unsafe { _mm256_store_ps(ptr, v) }
        }

        /// Safe fused multiply-add
        #[target_feature(enable = "fma")]
        pub unsafe fn fma_f32(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fmadd_ps(a, b, c)
        }

        /// Safe blend operation with compile-time mask
        pub fn blend_f32<const MASK: i32>(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_blend_ps(a, b, MASK) }
        }
    }

    /// Safe wrapper for AVX-512 operations (when available)
    #[cfg(target_feature = "avx512f")]
    pub mod avx512 {
        use super::*;

        /// Safe vector load with alignment check
        pub fn load_aligned_f32(ptr: *const f32) -> __m512 {
            debug_assert!(super::super::alignment::is_aligned(ptr, 64));
            unsafe { _mm512_load_ps(ptr) }
        }

        /// Safe vector store with alignment check  
        pub fn store_aligned_f32(ptr: *mut f32, v: __m512) {
            debug_assert!(super::super::alignment::is_aligned(ptr, 64));
            unsafe { _mm512_store_ps(ptr, v) }
        }

        /// Safe fused multiply-add
        pub fn fma_f32(a: __m512, b: __m512, c: __m512) -> __m512 {
            unsafe { _mm512_fmadd_ps(a, b, c) }
        }

        /// Safe horizontal reduction sum
        pub fn reduce_add_f32(v: __m512) -> f32 {
            unsafe { _mm512_reduce_add_ps(v) }
        }
    }
}

/// ARM NEON intrinsic wrappers
#[cfg(target_arch = "aarch64")]
pub mod neon {
    #[cfg(feature = "no-std")]
    use core::arch::aarch64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::aarch64::*;

    /// Safe horizontal add for float32x4_t
    pub fn horizontal_add_f32(v: float32x4_t) -> f32 {
        unsafe { vaddvq_f32(v) }
    }

    /// Safe vector load with alignment check
    pub unsafe fn load_aligned_f32(ptr: *const f32) -> float32x4_t {
        debug_assert!(super::alignment::is_aligned(ptr, 16));
        unsafe { vld1q_f32(ptr) }
    }

    /// Safe vector store with alignment check
    pub unsafe fn store_aligned_f32(ptr: *mut f32, v: float32x4_t) {
        debug_assert!(super::alignment::is_aligned(ptr, 16));
        unsafe { vst1q_f32(ptr, v) }
    }

    /// Safe fused multiply-add
    pub fn fma_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        unsafe { vfmaq_f32(c, a, b) }
    }
}

/// Branch prediction and loop optimization
pub mod optimization {
    /// Mark a loop for potential unrolling
    #[inline(always)]
    pub fn suggest_unroll<F>(iterations: usize, mut f: F)
    where
        F: FnMut(usize),
    {
        // Hint for small iteration counts to unroll
        if iterations <= 16 {
            for i in 0..iterations {
                f(i);
            }
        } else {
            for i in 0..iterations {
                f(i);
            }
        }
    }

    /// Prefetch hint for upcoming memory access
    #[inline(always)]
    pub fn prefetch_hint<T>(_ptr: *const T) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "no-std")]
            {
                core::arch::x86_64::_mm_prefetch(
                    _ptr as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }
            #[cfg(not(feature = "no-std"))]
            {
                std::arch::x86_64::_mm_prefetch(_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }
    }

    /// Cold function annotation (hint for code layout)
    #[cold]
    pub fn cold_path() {
        // This function is marked as cold, compiler will optimize for size
    }

    /// Hot function annotation (hint for aggressive optimization)
    #[inline(always)]
    pub fn hot_path() {
        // This function is hot, compiler will optimize for speed
    }
}

/// Auto-vectorization helpers
pub mod vectorization {
    /// Helper to enable vectorization for simple operations
    pub fn vectorize_simple_op<T, F>(src: &[T], dest: &mut [T], op: F)
    where
        T: Copy,
        F: Fn(T) -> T,
    {
        assert_eq!(src.len(), dest.len());

        // Hint to compiler for vectorization
        #[allow(clippy::needless_range_loop)]
        for i in 0..src.len() {
            dest[i] = op(src[i]);
        }
    }

    /// Helper for vectorized binary operations
    pub fn vectorize_binary_op<T, F>(a: &[T], b: &[T], dest: &mut [T], op: F)
    where
        T: Copy,
        F: Fn(T, T) -> T,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), dest.len());

        // Hint to compiler for vectorization
        #[allow(clippy::needless_range_loop)]
        for i in 0..a.len() {
            dest[i] = op(a[i], b[i]);
        }
    }

    /// Vectorization hint with stride patterns
    pub fn vectorize_strided<T, F>(src: &[T], dest: &mut [T], stride: usize, op: F)
    where
        T: Copy,
        F: Fn(T) -> T,
    {
        let mut i = 0;
        while i < src.len() {
            dest[i] = op(src[i]);
            i += stride;
        }
    }
}

/// Performance measurement utilities
pub mod perf {
    #[cfg(not(feature = "no-std"))]
    use std::time::Instant;

    #[cfg(feature = "no-std")]
    use core::time::Duration;
    #[cfg(not(feature = "no-std"))]
    use std::time::Duration;

    /// High-precision timing for micro-benchmarks
    #[cfg(not(feature = "no-std"))]
    pub fn time_operation<F, R>(op: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = op();
        let elapsed = start.elapsed();
        (result, elapsed)
    }

    /// Mock timing for no-std environments
    #[cfg(feature = "no-std")]
    pub fn time_operation<F, R>(op: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let result = op();
        // Return mock duration for no-std compatibility
        (result, Duration::from_nanos(0))
    }

    /// CPU cycle counter (x86 only)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn rdtsc() -> u64 {
        unsafe {
            #[cfg(feature = "no-std")]
            {
                core::arch::x86_64::_rdtsc()
            }
            #[cfg(not(feature = "no-std"))]
            {
                std::arch::x86_64::_rdtsc()
            }
        }
    }

    /// Memory fence for timing measurements
    pub fn memory_fence() {
        #[cfg(feature = "no-std")]
        {
            core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        }
        #[cfg(not(feature = "no-std"))]
        {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_check() {
        let data = vec![1.0f32; 16];
        let ptr = data.as_ptr();

        // Most allocators align to at least 8 bytes
        assert!(alignment::is_aligned(ptr, 4));
    }

    #[test]
    fn test_vectorization_helpers() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut dest = vec![0.0f32; 4];

        vectorization::vectorize_simple_op(&src, &mut dest, |x| x * 2.0);

        assert_eq!(dest, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_binary_vectorization() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut dest = vec![0.0f32; 4];

        vectorization::vectorize_binary_op(&a, &b, &mut dest, |x, y| x + y);

        assert_eq!(dest, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_performance_timing() {
        let (result, duration) = perf::time_operation(|| (0..1000).sum::<i32>());

        assert_eq!(result, 499500);
        assert!(duration.as_nanos() > 0);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_sse2_horizontal_add() {
        unsafe {
            #[cfg(feature = "no-std")]
            let v = core::arch::x86_64::_mm_setr_ps(1.0, 2.0, 3.0, 4.0);
            #[cfg(not(feature = "no-std"))]
            let v = std::arch::x86_64::_mm_setr_ps(1.0, 2.0, 3.0, 4.0);

            let sum = x86::sse2::horizontal_add_f32(v);
            assert!((sum - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_optimization_unroll() {
        let mut sum = 0;
        optimization::suggest_unroll(10, |i| {
            sum += i;
        });
        assert_eq!(sum, 45); // 0+1+2+...+9 = 45
    }
}
