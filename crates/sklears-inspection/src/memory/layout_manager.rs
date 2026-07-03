//! Memory layout optimization and SIMD operations for explanation algorithms
//!
//! This module provides advanced memory management capabilities including aligned allocation,
//! memory layout optimization, and high-performance SIMD vectorized operations.

use crate::types::*;
use std::alloc::{alloc, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Memory-efficient data structure for explanation results
#[derive(Clone, Debug)]
pub struct ExplanationDataLayout {
    /// Feature-major layout for better cache locality
    pub feature_major: bool,
    /// Block size for tiled access patterns
    pub block_size: usize,
    /// Memory alignment
    pub alignment: usize,
}

impl Default for ExplanationDataLayout {
    fn default() -> Self {
        Self {
            feature_major: true,
            block_size: 64,
            alignment: 64,
        }
    }
}

/// Owned buffer of `Float` values allocated with an explicit (possibly
/// over-aligned) [`Layout`].
///
/// `Vec<Float>` cannot safely hold memory allocated with an alignment
/// greater than `align_of::<Float>()`: its `Drop` implementation always
/// deallocates using `Layout::array::<Float>(capacity)`, i.e. `Float`'s
/// *natural* alignment, with no way to override it. Wrapping a custom,
/// over-aligned allocation in a `Vec` via `Vec::from_raw_parts` therefore
/// deallocates with the wrong layout -- Undefined Behavior (confirmed by
/// Miri: "incorrect layout on deallocation"). `AlignedVec` instead
/// remembers the exact layout used at allocation time and frees the memory
/// with that same layout, so it can safely provide cache-line-aligned
/// storage.
pub struct AlignedVec {
    ptr: NonNull<Float>,
    len: usize,
    layout: Layout,
}

impl AlignedVec {
    /// Allocate a new zero-initialized buffer of `len` elements, aligned to
    /// `alignment` bytes (rounded up to a power of two, and to at least
    /// `align_of::<Float>()`).
    fn new_zeroed(len: usize, alignment: usize) -> Self {
        let alignment = alignment
            .max(std::mem::align_of::<Float>())
            .next_power_of_two();
        let total_size = len * std::mem::size_of::<Float>();

        if total_size == 0 {
            // `GlobalAlloc::alloc` forbids zero-size layouts; use a
            // dangling, well-aligned pointer instead, mirroring how `Vec`
            // itself represents zero-capacity buffers.
            let layout = Layout::from_size_align(0, alignment).expect(
                "alignment is a validated power of two, so a zero-size layout is always valid",
            );
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout,
            };
        }

        let layout = Layout::from_size_align(total_size, alignment)
            .expect("size/alignment combination must fit within isize::MAX for any realistic buffer length");

        // SAFETY: `layout` has non-zero size (checked above).
        let raw_ptr = unsafe { alloc(layout) };
        let ptr = match NonNull::new(raw_ptr.cast::<Float>()) {
            Some(ptr) => ptr,
            None => std::alloc::handle_alloc_error(layout),
        };

        // SAFETY: `ptr` was just allocated and is valid for `total_size`
        // bytes (`len` elements of `Float`).
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, len) };

        Self { ptr, len, layout }
    }
}

impl Deref for AlignedVec {
    type Target = [Float];

    fn deref(&self) -> &[Float] {
        // SAFETY: `ptr` is valid and initialized for `len` elements for as
        // long as `self` is alive.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedVec {
    fn deref_mut(&mut self) -> &mut [Float] {
        // SAFETY: `ptr` is valid and initialized for `len` elements, and
        // `self` has exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: `self.ptr` was allocated by `alloc` with exactly
            // `self.layout`, and this is the only place that frees it.
            unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout) };
        }
    }
}

// SAFETY: `AlignedVec` exclusively owns its heap allocation; sharing it
// across threads is as safe as sharing a `Vec<Float>` (which is Send+Sync).
unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

/// Memory-efficient data layout manager
pub struct MemoryLayoutManager {
    /// Current layout configuration
    layout: ExplanationDataLayout,
    /// Memory pool for reuse
    memory_pool: Arc<Mutex<Vec<AlignedVec>>>,
}

impl MemoryLayoutManager {
    pub fn new(layout: ExplanationDataLayout) -> Self {
        Self {
            layout,
            memory_pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get optimized memory layout for explanation data
    pub fn get_optimized_layout(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> ExplanationDataLayout {
        // Choose layout based on access patterns
        let feature_major = if n_features < n_samples {
            // More samples than features - use feature-major layout
            true
        } else {
            // More features than samples - use sample-major layout
            false
        };

        ExplanationDataLayout {
            feature_major,
            block_size: self.layout.block_size,
            alignment: self.layout.alignment,
        }
    }

    /// Allocate aligned memory for explanation computation
    pub fn allocate_aligned(&self, size: usize) -> AlignedVec {
        // Try to reuse memory from pool
        {
            // Handle poisoned mutex by recovering or creating empty pool
            let mut pool = self.memory_pool.lock().unwrap_or_else(|poisoned| {
                // Clear the poisoned state and return the guard
                poisoned.into_inner()
            });
            if let Some(memory) = pool.pop() {
                if memory.len() >= size {
                    return memory;
                }
            }
        }

        // Allocate new aligned memory. `AlignedVec` encapsulates the unsafe
        // allocation and remembers its own layout so it can be freed
        // correctly, regardless of the requested alignment.
        AlignedVec::new_zeroed(size, self.layout.alignment)
    }

    /// Return memory to pool for reuse
    pub fn deallocate(&self, memory: AlignedVec) {
        // Handle poisoned mutex by recovering or creating empty pool
        let mut pool = self.memory_pool.lock().unwrap_or_else(|poisoned| {
            // Clear the poisoned state and return the guard
            poisoned.into_inner()
        });
        pool.push(memory);

        // Limit pool size to prevent memory bloat
        if pool.len() > 10 {
            pool.truncate(5);
        }
    }

    /// Unsafe memory copy with prefetching for better cache performance
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `src` and `dst` are valid pointers
    /// - `src` and `dst` do not overlap
    /// - `len` is within bounds for both arrays
    pub unsafe fn copy_with_prefetch(&self, src: *const Float, dst: *mut Float, len: usize) {
        let prefetch_distance = self.layout.alignment / std::mem::size_of::<Float>();

        for i in 0..len {
            // Prefetch next cache line
            if i + prefetch_distance < len {
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::*;
                    _mm_prefetch(src.add(i + prefetch_distance) as *const i8, _MM_HINT_T0);
                }
            }

            // Copy data
            *dst.add(i) = *src.add(i);
        }
    }

    /// Unsafe vectorized addition with SIMD for maximum performance
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `a`, `b`, and `result` are valid pointers
    /// - All arrays have at least `len` elements
    /// - The arrays are properly aligned for SIMD operations
    pub unsafe fn vectorized_add(
        &self,
        a: *const Float,
        b: *const Float,
        result: *mut Float,
        len: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;

            // Check if we can use AVX2 for f64 or SSE for f32
            if std::mem::size_of::<Float>() == 8 && is_x86_feature_detected!("avx2") {
                // Process 4 f64 values at a time with AVX2
                let chunks = len / 4;
                let a_ptr = a;
                let b_ptr = b;
                let result_ptr = result;

                for i in 0..chunks {
                    let a_vec = _mm256_loadu_pd(a_ptr.add(i * 4));
                    let b_vec = _mm256_loadu_pd(b_ptr.add(i * 4));
                    let sum = _mm256_add_pd(a_vec, b_vec);
                    _mm256_storeu_pd(result_ptr.add(i * 4), sum);
                }

                // Handle remaining elements
                for i in (chunks * 4)..len {
                    *result.add(i) = *a.add(i) + *b.add(i);
                }
            } else if std::mem::size_of::<Float>() == 4 && is_x86_feature_detected!("sse") {
                // Process 4 f32 values at a time with SSE
                let chunks = len / 4;
                let a_ptr = a as *const f32;
                let b_ptr = b as *const f32;
                let result_ptr = result as *mut f32;

                for i in 0..chunks {
                    let a_vec = _mm_loadu_ps(a_ptr.add(i * 4));
                    let b_vec = _mm_loadu_ps(b_ptr.add(i * 4));
                    let sum = _mm_add_ps(a_vec, b_vec);
                    _mm_storeu_ps(result_ptr.add(i * 4), sum);
                }

                // Handle remaining elements
                for i in (chunks * 4)..len {
                    *result.add(i) = *a.add(i) + *b.add(i);
                }
            } else {
                // Fallback to scalar addition
                for i in 0..len {
                    *result.add(i) = *a.add(i) + *b.add(i);
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback to scalar addition for non-x86 architectures
            for i in 0..len {
                *result.add(i) = *a.add(i) + *b.add(i);
            }
        }
    }

    /// Unsafe fast dot product computation with SIMD
    ///
    /// # Safety
    ///
    /// This function is safe when:
    /// - `a` and `b` are valid pointers
    /// - Both arrays have at least `len` elements
    /// - The arrays are properly aligned for SIMD operations
    pub unsafe fn fast_dot_product(&self, a: *const Float, b: *const Float, len: usize) -> Float {
        let mut result = 0.0;

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;

            if std::mem::size_of::<Float>() == 8 && is_x86_feature_detected!("avx2") {
                // Process 4 f64 values at a time with AVX2
                let chunks = len / 4;
                let a_ptr = a;
                let b_ptr = b;

                let mut sum_vec = _mm256_setzero_pd();

                for i in 0..chunks {
                    let a_vec = _mm256_loadu_pd(a_ptr.add(i * 4));
                    let b_vec = _mm256_loadu_pd(b_ptr.add(i * 4));
                    let prod = _mm256_mul_pd(a_vec, b_vec);
                    sum_vec = _mm256_add_pd(sum_vec, prod);
                }

                // Extract and sum the 4 values
                let mut sum_arr = [0.0; 4];
                _mm256_storeu_pd(sum_arr.as_mut_ptr(), sum_vec);
                result = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

                // Handle remaining elements
                for i in (chunks * 4)..len {
                    result += (*a.add(i)) * (*b.add(i));
                }
            } else if std::mem::size_of::<Float>() == 4 && is_x86_feature_detected!("sse") {
                // Process 4 f32 values at a time with SSE
                let chunks = len / 4;
                let a_ptr = a as *const f32;
                let b_ptr = b as *const f32;

                let mut sum_vec = _mm_setzero_ps();

                for i in 0..chunks {
                    let a_vec = _mm_loadu_ps(a_ptr.add(i * 4));
                    let b_vec = _mm_loadu_ps(b_ptr.add(i * 4));
                    let prod = _mm_mul_ps(a_vec, b_vec);
                    sum_vec = _mm_add_ps(sum_vec, prod);
                }

                // Extract and sum the 4 values
                let mut sum_arr = [0.0; 4];
                _mm_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
                result = (sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3]) as Float;

                // Handle remaining elements
                for i in (chunks * 4)..len {
                    result += (*a.add(i)) * (*b.add(i));
                }
            } else {
                // Fallback to scalar multiplication
                for i in 0..len {
                    result += (*a.add(i)) * (*b.add(i));
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback to scalar multiplication for non-x86 architectures
            for i in 0..len {
                result += (*a.add(i)) * (*b.add(i));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_layout_manager() {
        let layout = ExplanationDataLayout {
            feature_major: true,
            block_size: 64,
            alignment: 32,
        };

        let manager = MemoryLayoutManager::new(layout);
        let optimized = manager.get_optimized_layout(100, 10);

        // Should prefer feature-major for more samples than features
        assert!(optimized.feature_major);
    }

    #[test]
    fn test_aligned_memory_allocation() {
        let layout = ExplanationDataLayout {
            feature_major: true,
            block_size: 64,
            alignment: 32,
        };

        let manager = MemoryLayoutManager::new(layout);
        let memory = manager.allocate_aligned(100);

        assert_eq!(memory.len(), 100);

        // Return to pool
        manager.deallocate(memory);
    }

    #[test]
    fn test_layout_optimization() {
        let layout = ExplanationDataLayout::default();
        let manager = MemoryLayoutManager::new(layout);

        // More samples than features
        let opt1 = manager.get_optimized_layout(1000, 10);
        assert!(opt1.feature_major);

        // More features than samples
        let opt2 = manager.get_optimized_layout(10, 1000);
        assert!(!opt2.feature_major);
    }

    #[test]
    fn test_memory_pool() {
        let layout = ExplanationDataLayout::default();
        let manager = MemoryLayoutManager::new(layout);

        // Allocate and deallocate memory
        let mem1 = manager.allocate_aligned(50);
        let mem2 = manager.allocate_aligned(100);

        manager.deallocate(mem1);
        manager.deallocate(mem2);

        // Next allocation should reuse from pool
        let mem3 = manager.allocate_aligned(75);
        assert_eq!(mem3.len(), 100); // Should reuse the larger buffer
    }

    #[test]
    fn test_explanation_data_layout_default() {
        let layout = ExplanationDataLayout::default();
        assert!(layout.feature_major);
        assert_eq!(layout.block_size, 64);
        assert_eq!(layout.alignment, 64);
    }

    #[test]
    fn test_unsafe_operations_safety() {
        let layout = ExplanationDataLayout::default();
        let manager = MemoryLayoutManager::new(layout);

        // Test with properly aligned memory
        let vec_a = [1.0, 2.0, 3.0, 4.0];
        let vec_b = [5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        unsafe {
            // Test vectorized addition
            manager.vectorized_add(vec_a.as_ptr(), vec_b.as_ptr(), result.as_mut_ptr(), 4);

            // Test dot product
            let dot = manager.fast_dot_product(vec_a.as_ptr(), vec_b.as_ptr(), 4);
            assert!(dot > 0.0); // Should be positive for these positive vectors
        }

        // Check addition results
        for i in 0..4 {
            assert!((result[i] - (vec_a[i] + vec_b[i])).abs() < 1e-6);
        }
    }
}
