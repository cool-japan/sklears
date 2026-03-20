//! Unsafe optimizations for maximum performance (use with caution)
//!
//! This module contains performance optimizations that sacrifice safety for speed.
//! All functions are marked unsafe and require careful validation by the caller.

/// Unsafe fast memory operations
pub mod unsafe_memory {
    pub unsafe fn fast_copy<T: Copy>(src: *const T, dst: *mut T, len: usize) {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }

    /// Fast memory set using unsafe operations
    /// # Safety
    /// Caller must ensure dst is valid for len elements
    pub unsafe fn fast_set<T: Copy>(dst: *mut T, value: T, len: usize) {
        for i in 0..len {
            *dst.add(i) = value;
        }
    }

    /// Unsafe memory comparison
    /// # Safety
    /// Caller must ensure both pointers are valid for len bytes
    pub unsafe fn fast_compare(a: *const u8, b: *const u8, len: usize) -> bool {
        for i in 0..len {
            if *a.add(i) != *b.add(i) {
                return false;
            }
        }
        true
    }
}

/// Unsafe vectorized operations (scalar fallbacks for stable Rust)
pub mod unsafe_vectorized {
    /// Unsafe sum without bounds checking (scalar fallback)
    /// # Safety
    /// Caller must ensure data is properly aligned and has sufficient length
    pub unsafe fn unsafe_sum_f64(data: *const f64, len: usize) -> f64 {
        if len == 0 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..len {
            sum += *data.add(i);
        }
        sum
    }

    /// Unsafe mean without bounds checking (scalar fallback)
    /// # Safety
    /// Caller must ensure data is properly aligned and has sufficient length
    pub unsafe fn unsafe_mean_f64(data: *const f64, len: usize) -> f64 {
        if len == 0 {
            return 0.0;
        }
        unsafe_sum_f64(data, len) / len as f64
    }

    /// Unsafe min/max search without bounds checking
    /// # Safety
    /// Caller must ensure data is valid for len elements
    pub unsafe fn unsafe_min_max_f64(data: *const f64, len: usize) -> (f64, f64) {
        if len == 0 {
            return (f64::NAN, f64::NAN);
        }

        let mut min_val = *data;
        let mut max_val = *data;

        for i in 1..len {
            let val = *data.add(i);
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        (min_val, max_val)
    }
}

/// Branch prediction hints
pub mod branch_hints {
    /// Hint that the condition is likely to be true
    #[inline(always)]
    pub fn likely(condition: bool) -> bool {
        std::hint::black_box(condition)
    }

    /// Hint that the condition is unlikely to be true
    #[inline(always)]
    pub fn unlikely(condition: bool) -> bool {
        std::hint::black_box(condition)
    }

    /// Create a cold path for error handling
    #[cold]
    #[inline(never)]
    pub fn cold_path() {}
}

/// Memory prefetching (no-op on stable Rust)
pub mod prefetch {
    /// Prefetch data into cache (no-op fallback)
    /// # Safety
    /// Caller must ensure ptr is valid
    pub unsafe fn prefetch_read<T>(_ptr: *const T) {
        // No-op on stable Rust - would use intrinsics on nightly
    }

    /// Prefetch data for writing (no-op fallback)
    /// # Safety
    /// Caller must ensure ptr is valid
    pub unsafe fn prefetch_write<T>(_ptr: *const T) {
        // No-op on stable Rust - would use intrinsics on nightly
    }

    /// Manual prefetching using simple memory access
    /// # Safety
    /// Caller must ensure ptr is valid
    pub unsafe fn manual_prefetch<T>(ptr: *const T) {
        std::ptr::read_volatile(ptr);
    }
}

/// Manual memory alignment utilities
pub mod alignment {
    /// Check if pointer is aligned to specified boundary
    /// # Safety
    /// Caller must ensure ptr is valid
    pub unsafe fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        ptr as usize % alignment == 0
    }

    /// Align pointer to next boundary
    /// # Safety
    /// Caller must ensure the alignment is valid and the pointer arithmetic is safe
    pub unsafe fn align_up<T>(ptr: *const T, alignment: usize) -> *const T {
        let addr = ptr as usize;
        let aligned = (addr + alignment - 1) & !(alignment - 1);
        aligned as *const T
    }

    /// Get optimal alignment for a type
    pub fn optimal_alignment<T>() -> usize {
        std::mem::align_of::<T>().max(64) // Cache line alignment
    }
}

/// Warning: These optimizations sacrifice safety for performance
/// Only use in performance-critical sections with careful validation
pub fn performance_warning() {
    // This module contains unsafe code that can cause undefined behavior
    // if used incorrectly. Thoroughly test and validate before production use.
}
