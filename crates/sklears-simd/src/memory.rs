//! Memory optimization utilities for SIMD operations
//!
//! This module provides cache-aware algorithms, memory prefetching,
//! and aligned memory operations to improve SIMD performance.

#[cfg(feature = "no-std")]
use alloc::alloc::{alloc, dealloc, Layout};
#[cfg(not(feature = "no-std"))]
use std::alloc::{alloc, dealloc, Layout};

#[cfg(feature = "no-std")]
use core::ptr::NonNull;
#[cfg(not(feature = "no-std"))]
use std::ptr::NonNull;

#[cfg(feature = "no-std")]
use core::{mem, slice};
#[cfg(not(feature = "no-std"))]
use std::{mem, slice};

/// Simple allocation error type
#[derive(Debug)]
pub struct AllocError;

#[cfg(feature = "no-std")]
impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Memory allocation failed")
    }
}

#[cfg(not(feature = "no-std"))]
impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Memory allocation failed")
    }
}

#[cfg(not(feature = "no-std"))]
#[cfg(not(feature = "no-std"))]
impl std::error::Error for AllocError {}

#[cfg(feature = "no-std")]
impl core::error::Error for AllocError {}

/// Cache line size constants for different architectures
pub const CACHE_LINE_SIZE: usize = 64;
pub const L1_CACHE_SIZE: usize = 32 * 1024;
pub const L2_CACHE_SIZE: usize = 256 * 1024;
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;

/// Alignment requirements for SIMD operations
pub const SIMD_ALIGNMENT: usize = 32; // AVX2 alignment

/// Memory prefetch hint types
#[derive(Debug, Clone, Copy)]
pub enum PrefetchHint {
    /// Prefetch for read with temporal locality
    T0,
    /// Prefetch for read with low temporal locality
    T1,
    /// Prefetch for read with minimal temporal locality
    T2,
    /// Prefetch for read with no temporal locality
    Nta,
}

/// SIMD-aligned memory allocator
pub struct AlignedAlloc<T> {
    ptr: NonNull<T>,
    layout: Layout,
    len: usize,
}

impl<T> AlignedAlloc<T> {
    /// Allocate aligned memory for SIMD operations
    pub fn new(len: usize) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(len * mem::size_of::<T>(), SIMD_ALIGNMENT)
            .map_err(|_| AllocError)?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(AllocError);
        }

        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
            layout,
            len,
        })
    }

    /// Get a mutable slice to the aligned memory
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get a slice to the aligned memory
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get the raw mutable pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for AlignedAlloc<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

/// Memory prefetch operations
pub mod prefetch {
    use super::PrefetchHint;

    #[cfg(feature = "no-std")]
    use core::mem;

    /// Prefetch memory to cache with specified hint
    #[inline(always)]
    pub fn prefetch_read_data(_address: *const u8, _hint: PrefetchHint) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "no-std")]
            use core::arch::x86_64::*;
            #[cfg(not(feature = "no-std"))]
            use core::arch::x86_64::*;
            match _hint {
                PrefetchHint::T0 => _mm_prefetch(_address as *const i8, _MM_HINT_T0),
                PrefetchHint::T1 => _mm_prefetch(_address as *const i8, _MM_HINT_T1),
                PrefetchHint::T2 => _mm_prefetch(_address as *const i8, _MM_HINT_T2),
                PrefetchHint::Nta => _mm_prefetch(_address as *const i8, _MM_HINT_NTA),
            }
        }
    }

    /// Prefetch multiple cache lines for a memory range
    #[inline]
    pub fn prefetch_range<T>(slice: &[T], hint: PrefetchHint) {
        let start = slice.as_ptr() as *const u8;
        let size = std::mem::size_of_val(slice);
        let end = unsafe { start.add(size) };

        let mut current = start;
        while current < end {
            prefetch_read_data(current, hint);
            current = unsafe { current.add(super::CACHE_LINE_SIZE) };
        }
    }
}

/// Cache-aware matrix operations
pub mod cache_aware {

    /// Calculate optimal block size for cache-friendly matrix operations
    pub fn optimal_block_size(cache_size: usize, element_size: usize) -> usize {
        // Use square root of available cache space
        let elements_in_cache = cache_size / element_size;
        (elements_in_cache as f64).sqrt() as usize
    }

    /// Cache-friendly matrix transpose
    pub fn transpose_blocked(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(output.len(), rows * cols);

        for block_row in (0..rows).step_by(block_size) {
            for block_col in (0..cols).step_by(block_size) {
                let end_row = (block_row + block_size).min(rows);
                let end_col = (block_col + block_size).min(cols);

                for i in block_row..end_row {
                    for j in block_col..end_col {
                        output[j * rows + i] = input[i * cols + j];
                    }
                }
            }
        }
    }

    /// Cache-friendly matrix multiplication with blocking
    pub fn matrix_multiply_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        block_size: usize,
    ) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        // Initialize output
        c.fill(0.0);

        for kk in (0..k).step_by(block_size) {
            for ii in (0..m).step_by(block_size) {
                for jj in (0..n).step_by(block_size) {
                    let end_k = (kk + block_size).min(k);
                    let end_i = (ii + block_size).min(m);
                    let end_j = (jj + block_size).min(n);

                    for i in ii..end_i {
                        for j in jj..end_j {
                            let mut sum = 0.0;
                            for l in kk..end_k {
                                sum += a[i * k + l] * b[l * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

/// Non-temporal store operations for streaming data
pub mod streaming {
    /// Non-temporal store for f32 arrays (bypasses cache)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn stream_store_f32(dest: &mut [f32], src: &[f32]) {
        assert_eq!(dest.len(), src.len());

        if !is_x86_feature_detected!("sse2") {
            dest.copy_from_slice(src);
            return;
        }

        unsafe {
            stream_store_sse2(dest, src);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn stream_store_sse2(dest: &mut [f32], src: &[f32]) {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;

        let mut i = 0;
        let len = dest.len();

        // Process 4 elements at a time with non-temporal stores
        while i + 4 <= len {
            let data = _mm_loadu_ps(src.as_ptr().add(i));
            _mm_stream_ps(dest.as_mut_ptr().add(i), data);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            dest[i] = src[i];
            i += 1;
        }

        // Ensure all stores are complete
        _mm_sfence();
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    pub fn stream_store_f32(dest: &mut [f32], src: &[f32]) {
        dest.copy_from_slice(src);
    }
}

/// Memory bandwidth optimization utilities
pub mod bandwidth {
    use super::{prefetch::prefetch_range, PrefetchHint};

    #[cfg(feature = "no-std")]
    use core::{mem, slice};
    #[cfg(not(feature = "no-std"))]
    use std::{mem, slice, time::Instant};

    /// Bandwidth-optimized vector copy with prefetching
    pub fn copy_with_prefetch<T: Copy>(dest: &mut [T], src: &[T]) {
        assert_eq!(dest.len(), src.len());

        // Prefetch the source data
        prefetch_range(src, PrefetchHint::Nta);

        // Use streaming store for large arrays
        if std::mem::size_of_val(dest) > super::L1_CACHE_SIZE {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if mem::size_of::<T>() == mem::size_of::<f32>() {
                unsafe {
                    super::streaming::stream_store_f32(
                        slice::from_raw_parts_mut(dest.as_mut_ptr() as *mut f32, dest.len()),
                        slice::from_raw_parts(src.as_ptr() as *const f32, src.len()),
                    );
                }
                return;
            }
        }

        dest.copy_from_slice(src);
    }

    /// Memory bandwidth test for performance tuning
    #[cfg(not(feature = "no-std"))]
    pub fn measure_bandwidth() -> f64 {
        const SIZE: usize = 1024 * 1024; // 1MB
        let src = vec![1.0f32; SIZE];
        let mut dest = vec![0.0f32; SIZE];

        let start = Instant::now();
        for _ in 0..100 {
            copy_with_prefetch(&mut dest, &src);
        }
        let elapsed = start.elapsed();

        let bytes_transferred = SIZE * mem::size_of::<f32>() * 100 * 2; // read + write
        bytes_transferred as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
        // GB/s
    }

    /// Memory bandwidth test for performance tuning (no-std version)
    #[cfg(feature = "no-std")]
    pub fn measure_bandwidth() -> f64 {
        // Return a mock value for no-std environments where timing is not available
        1.0 // GB/s
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_aligned_alloc() {
        let mut alloc = AlignedAlloc::<f32>::new(1024).unwrap();
        let slice = alloc.as_mut_slice();

        // Check alignment
        assert_eq!(slice.as_ptr() as usize % SIMD_ALIGNMENT, 0);

        // Test basic operations
        slice[0] = 1.0;
        slice[1023] = 2.0;
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1023], 2.0);
    }

    #[test]
    fn test_cache_aware_transpose() {
        let rows = 64;
        let cols = 64;
        let mut input = vec![0.0f32; rows * cols];
        let mut output = vec![0.0f32; rows * cols];

        // Initialize input matrix
        for i in 0..rows {
            for j in 0..cols {
                input[i * cols + j] = (i * cols + j) as f32;
            }
        }

        cache_aware::transpose_blocked(&input, &mut output, rows, cols, 16);

        // Verify transpose
        for i in 0..rows {
            for j in 0..cols {
                assert_relative_eq!(output[j * rows + i], input[i * cols + j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_cache_aware_matrix_multiply() {
        let m = 32;
        let n = 32;
        let k = 32;

        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        cache_aware::matrix_multiply_blocked(&a, &b, &mut c, m, n, k, 16);

        // Verify result (each element should be k)
        for &val in &c {
            assert_relative_eq!(val, k as f32, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_stream_store() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dest = vec![0.0f32; 8];

        streaming::stream_store_f32(&mut dest, &src);

        for (i, &val) in dest.iter().enumerate() {
            assert_relative_eq!(val, src[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_bandwidth_measurement() {
        let bandwidth = bandwidth::measure_bandwidth();
        // Just check that bandwidth measurement runs and returns positive value
        assert!(bandwidth > 0.0);
        println!("Measured bandwidth: {:.2} GB/s", bandwidth);
    }

    #[test]
    fn test_optimal_block_size() {
        let block_size = cache_aware::optimal_block_size(L1_CACHE_SIZE, 4);
        assert!(block_size > 0);
        assert!(block_size < 1000); // Reasonable upper bound
    }

    #[test]
    fn test_copy_with_prefetch() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut dest = vec![0.0f32; 5];

        bandwidth::copy_with_prefetch(&mut dest, &src);

        for (i, &val) in dest.iter().enumerate() {
            assert_relative_eq!(val, src[i], epsilon = 1e-6);
        }
    }
}
