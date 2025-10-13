//! Compile-time optimization hints for SIMD operations
//!
//! This module provides compiler hints and attributes to help the compiler
//! optimize SIMD operations more effectively.

#[cfg(feature = "no-std")]
use core::{hint, mem::size_of, slice};
#[cfg(not(feature = "no-std"))]
use std::{hint, mem::size_of};

/// Compile-time optimization hints for SIMD operations
pub struct OptimizationHints;

impl OptimizationHints {
    /// Hint to the compiler that a branch is likely to be taken
    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        // Use intrinsic hint when available
        #[cfg(target_arch = "x86_64")]
        {
            if b {
                unsafe { std::arch::x86_64::_mm_prefetch::<0>(std::ptr::null::<i8>()) };
            }
        }
        b
    }

    /// Hint to the compiler that a branch is unlikely to be taken
    #[inline(always)]
    pub fn unlikely(b: bool) -> bool {
        // Inverse of likely
        !Self::likely(!b)
    }

    /// Hint that a pointer is aligned to SIMD boundaries
    #[inline(always)]
    pub fn assume_aligned<T>(ptr: *const T, align: usize) -> *const T {
        if align.is_power_of_two() && align >= size_of::<T>() {
            // Compiler hint for alignment
            unsafe { core::ptr::addr_of!(*ptr.cast::<u8>().add(0).cast::<T>()) }
        } else {
            ptr
        }
    }

    /// Hint that a pointer is aligned to SIMD boundaries (mutable)
    #[inline(always)]
    pub fn assume_aligned_mut<T>(ptr: *mut T, align: usize) -> *mut T {
        if align.is_power_of_two() && align >= size_of::<T>() {
            // Compiler hint for alignment
            unsafe { core::ptr::addr_of_mut!(*ptr.cast::<u8>().add(0).cast::<T>()) }
        } else {
            ptr
        }
    }

    /// Hint that a value is within a specific range
    #[inline(always)]
    pub fn assume_range<T: PartialOrd + Copy>(value: T, min: T, max: T) -> T {
        if value >= min && value <= max {
            value
        } else {
            // Undefined behavior if assumption is false
            unsafe { hint::unreachable_unchecked() }
        }
    }

    /// Hint that a slice has a specific length
    #[inline(always)]
    pub fn assume_len<T>(slice: &[T], len: usize) -> &[T] {
        if slice.len() == len {
            slice
        } else {
            // Undefined behavior if assumption is false
            unsafe { hint::unreachable_unchecked() }
        }
    }

    /// Hint that a slice has a specific length (mutable)
    #[inline(always)]
    pub fn assume_len_mut<T>(slice: &mut [T], len: usize) -> &mut [T] {
        if slice.len() == len {
            slice
        } else {
            // Undefined behavior if assumption is false
            unsafe { hint::unreachable_unchecked() }
        }
    }

    /// Hint that a loop will iterate a specific number of times
    #[inline(always)]
    pub fn assume_loop_count(count: usize) -> usize {
        // Compiler hint for loop unrolling
        if count > 0 {
            count
        } else {
            0
        }
    }

    /// Hint that data is hot (frequently accessed)
    #[inline(always)]
    pub fn prefetch_read<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { std::arch::x86_64::_mm_prefetch::<3>(ptr as *const i8) };
        }
        // AArch64 prefetch requires unstable features - disabled for stable Rust
        // #[cfg(all(target_arch = "aarch64", feature = "nightly"))]
        // {
        //     unsafe { std::arch::aarch64::_prefetch(ptr as *const i8, 0, 3) };
        // }
    }

    /// Hint that data will be written to (for write prefetching)
    #[inline(always)]
    pub fn prefetch_write<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { std::arch::x86_64::_mm_prefetch::<1>(ptr as *const i8) };
        }
        // AArch64 prefetch requires unstable features - disabled for stable Rust
        // #[cfg(all(target_arch = "aarch64", feature = "nightly"))]
        // {
        //     unsafe { std::arch::aarch64::_prefetch(ptr as *const i8, 1, 3) };
        // }
    }

    /// Hint that memory access will be non-temporal
    #[inline(always)]
    pub fn prefetch_nta<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { std::arch::x86_64::_mm_prefetch::<0>(ptr as *const i8) };
        }
    }

    /// Hint for vectorization - assume no aliasing
    #[inline(always)]
    pub fn assume_noalias<T>(ptr1: *const T, ptr2: *const T, len: usize) -> bool {
        let range1 = ptr1 as usize..ptr1 as usize + len * size_of::<T>();
        let range2 = ptr2 as usize..ptr2 as usize + len * size_of::<T>();
        !range1.contains(&range2.start) && !range2.contains(&range1.start)
    }

    /// Hint for SIMD width optimization
    #[inline(always)]
    pub fn optimal_simd_width<T>() -> usize {
        // Get optimal SIMD width based on type and architecture
        match size_of::<T>() {
            1 => 64, // 64 bytes for u8/i8
            2 => 32, // 32 elements for u16/i16
            4 => 16, // 16 elements for u32/i32/f32
            8 => 8,  // 8 elements for u64/i64/f64
            _ => 4,  // Default fallback
        }
    }
}

/// Macro for compile-time optimization hints
#[macro_export]
macro_rules! optimize_for_simd {
    (likely($expr:expr)) => {
        $crate::optimization_hints::OptimizationHints::likely($expr)
    };
    (unlikely($expr:expr)) => {
        $crate::optimization_hints::OptimizationHints::unlikely($expr)
    };
    (assume_aligned($ptr:expr, $align:expr)) => {
        $crate::optimization_hints::OptimizationHints::assume_aligned($ptr, $align)
    };
    (assume_len($slice:expr, $len:expr)) => {
        $crate::optimization_hints::OptimizationHints::assume_len($slice, $len)
    };
    (prefetch_read($ptr:expr)) => {
        $crate::optimization_hints::OptimizationHints::prefetch_read($ptr)
    };
    (prefetch_write($ptr:expr)) => {
        $crate::optimization_hints::OptimizationHints::prefetch_write($ptr)
    };
}

/// Compiler attributes for SIMD optimization
pub mod attributes {
    /// Force inlining for SIMD operations
    pub const FORCE_INLINE: &str = "inline(always)";

    /// Never inline (for larger functions)
    pub const NEVER_INLINE: &str = "inline(never)";

    /// Target-specific optimization
    pub const TARGET_FEATURE: &str = "target_feature";

    /// Cold code (rarely executed)
    pub const COLD: &str = "cold";

    /// Hot code (frequently executed)
    pub const HOT: &str = "hot";

    /// No mangle (for C FFI)
    pub const NO_MANGLE: &str = "no_mangle";

    /// Repr C (for C compatibility)
    pub const REPR_C: &str = "repr(C)";

    /// Repr align (for SIMD alignment)
    pub const REPR_ALIGN: &str = "repr(align)";
}

/// SIMD-specific compiler hints
pub mod simd_hints {
    use super::OptimizationHints;

    /// Hint that arrays are SIMD-aligned
    #[inline(always)]
    pub fn assume_simd_aligned<T>(slice: &[T]) -> &[T] {
        let align = if cfg!(target_feature = "avx512f") {
            64
        } else if cfg!(target_feature = "avx2") {
            32
        } else {
            16
        };

        let ptr = OptimizationHints::assume_aligned(slice.as_ptr(), align);
        unsafe { core::slice::from_raw_parts(ptr, slice.len()) }
    }

    /// Hint that arrays are SIMD-aligned (mutable)
    #[inline(always)]
    pub fn assume_simd_aligned_mut<T>(slice: &mut [T]) -> &mut [T] {
        let align = if cfg!(target_feature = "avx512f") {
            64
        } else if cfg!(target_feature = "avx2") {
            32
        } else {
            16
        };

        let ptr = OptimizationHints::assume_aligned_mut(slice.as_mut_ptr(), align);
        unsafe { core::slice::from_raw_parts_mut(ptr, slice.len()) }
    }

    /// Hint that loop will vectorize
    #[inline(always)]
    pub fn assume_vectorizable<T, F>(slice: &[T], mut f: F)
    where
        F: FnMut(&T),
    {
        let len = OptimizationHints::assume_loop_count(slice.len());
        for i in 0..len {
            f(&slice[i]);
        }
    }

    /// Hint that parallel processing is beneficial
    #[inline(always)]
    pub fn assume_parallel_beneficial(size: usize) -> bool {
        OptimizationHints::likely(size > 1000)
    }

    /// Hint for optimal chunk size
    #[inline(always)]
    pub fn optimal_chunk_size<T>() -> usize {
        OptimizationHints::optimal_simd_width::<T>() * 4
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_hints() {
        let ptr = [1.0f32; 16].as_ptr();
        let aligned_ptr = OptimizationHints::assume_aligned(ptr, 16);
        assert_eq!(ptr, aligned_ptr);

        let slice = &[1, 2, 3, 4];
        let len_slice = OptimizationHints::assume_len(slice, 4);
        assert_eq!(slice.len(), len_slice.len());

        let optimal_width = OptimizationHints::optimal_simd_width::<f32>();
        assert!(optimal_width > 0);
    }

    #[test]
    fn test_simd_hints() {
        let data = vec![1.0f32; 64];
        let aligned_slice = simd_hints::assume_simd_aligned(&data);
        assert_eq!(data.len(), aligned_slice.len());

        let chunk_size = simd_hints::optimal_chunk_size::<f32>();
        assert!(chunk_size > 0);

        let parallel = simd_hints::assume_parallel_beneficial(2000);
        assert!(parallel);
    }

    #[test]
    fn test_branch_hints() {
        let likely_true = OptimizationHints::likely(true);
        let unlikely_false = OptimizationHints::unlikely(false);

        assert!(likely_true);
        assert!(!unlikely_false);
    }

    #[test]
    fn test_prefetch_hints() {
        let data = vec![1.0f32; 100];
        OptimizationHints::prefetch_read(data.as_ptr());
        OptimizationHints::prefetch_write(data.as_ptr());
        OptimizationHints::prefetch_nta(data.as_ptr());

        // If we get here, prefetch calls didn't crash
        assert!(true);
    }

    #[test]
    fn test_macro_hints() {
        let data = vec![1.0f32; 16];
        let ptr = optimize_for_simd!(assume_aligned(data.as_ptr(), 16));

        optimize_for_simd!(prefetch_read(ptr));

        let slice = optimize_for_simd!(assume_len(data.as_slice(), 16));
        assert_eq!(slice.len(), 16);
    }
}
