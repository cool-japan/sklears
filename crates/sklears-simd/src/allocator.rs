//! Custom allocators optimized for SIMD operations
//!
//! This module provides specialized memory allocators that ensure proper alignment
//! and memory layout for optimal SIMD performance.

#[cfg(not(feature = "no-std"))]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(not(feature = "no-std"))]
use std::ptr::{self, NonNull};
#[cfg(not(feature = "no-std"))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(feature = "no-std"))]
use std::{mem, slice};

#[cfg(feature = "no-std")]
use core::alloc::{GlobalAlloc, Layout};
#[cfg(feature = "no-std")]
use core::ptr::{self, NonNull};
#[cfg(feature = "no-std")]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "no-std")]
use core::{mem, slice};
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
use alloc::alloc as global_alloc;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;

/// Statistics for SIMD allocator performance monitoring
#[derive(Debug, Default)]
pub struct AllocatorStats {
    pub total_allocations: AtomicUsize,
    pub total_deallocations: AtomicUsize,
    pub bytes_allocated: AtomicUsize,
    pub bytes_deallocated: AtomicUsize,
    pub aligned_allocations: AtomicUsize,
    pub peak_memory_usage: AtomicUsize,
}

impl AllocatorStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_allocation(&self, size: usize, aligned: bool) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_add(size, Ordering::Relaxed);

        if aligned {
            self.aligned_allocations.fetch_add(1, Ordering::Relaxed);
        }

        // Update peak memory usage
        let current_usage = self.current_memory_usage();
        let mut peak = self.peak_memory_usage.load(Ordering::Relaxed);
        while current_usage > peak {
            match self.peak_memory_usage.compare_exchange_weak(
                peak,
                current_usage,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
    }

    pub fn record_deallocation(&self, size: usize) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_deallocated.fetch_add(size, Ordering::Relaxed);
    }

    pub fn current_memory_usage(&self) -> usize {
        let allocated = self.bytes_allocated.load(Ordering::Relaxed);
        let deallocated = self.bytes_deallocated.load(Ordering::Relaxed);
        allocated.saturating_sub(deallocated)
    }

    pub fn allocation_efficiency(&self) -> f64 {
        let total_allocs = self.total_allocations.load(Ordering::Relaxed);
        let aligned_allocs = self.aligned_allocations.load(Ordering::Relaxed);

        if total_allocs == 0 {
            1.0
        } else {
            aligned_allocs as f64 / total_allocs as f64
        }
    }
}

/// SIMD-optimized allocator with alignment guarantees
pub struct SimdAllocator {
    stats: AllocatorStats,
    default_alignment: usize,
}

impl SimdAllocator {
    /// Create a new SIMD allocator with default 32-byte alignment (AVX2)
    pub const fn new() -> Self {
        Self::with_alignment(32)
    }

    /// Create a new SIMD allocator with custom alignment
    pub const fn with_alignment(alignment: usize) -> Self {
        Self {
            stats: AllocatorStats {
                total_allocations: AtomicUsize::new(0),
                total_deallocations: AtomicUsize::new(0),
                bytes_allocated: AtomicUsize::new(0),
                bytes_deallocated: AtomicUsize::new(0),
                aligned_allocations: AtomicUsize::new(0),
                peak_memory_usage: AtomicUsize::new(0),
            },
            default_alignment: alignment,
        }
    }

    /// Get allocator statistics
    pub fn stats(&self) -> &AllocatorStats {
        &self.stats
    }

    /// Allocate aligned memory for SIMD operations
    pub fn allocate_simd<T>(&self, count: usize) -> Option<NonNull<T>> {
        let size = count * mem::size_of::<T>();
        let align = self.default_alignment.max(mem::align_of::<T>());

        let layout = Layout::from_size_align(size, align).ok()?;

        // Use system allocator for the actual allocation
        #[cfg(not(feature = "no-std"))]
        let ptr = unsafe { System.alloc(layout) };
        #[cfg(feature = "no-std")]
        let ptr = unsafe { global_alloc::alloc(layout) };

        if ptr.is_null() {
            None
        } else {
            self.stats.record_allocation(size, true);
            NonNull::new(ptr.cast())
        }
    }

    /// Deallocate SIMD memory
    pub unsafe fn deallocate_simd<T>(&self, ptr: NonNull<T>, count: usize) {
        let size = count * mem::size_of::<T>();
        let align = self.default_alignment.max(mem::align_of::<T>());

        if let Ok(layout) = Layout::from_size_align(size, align) {
            #[cfg(not(feature = "no-std"))]
            System.dealloc(ptr.cast().as_ptr(), layout);
            #[cfg(feature = "no-std")]
            global_alloc::dealloc(ptr.cast().as_ptr(), layout);
            self.stats.record_deallocation(size);
        }
    }

    /// Allocate zero-initialized SIMD memory
    pub fn allocate_zeroed_simd<T>(&self, count: usize) -> Option<NonNull<T>>
    where
        T: Copy,
    {
        let size = count * mem::size_of::<T>();
        let align = self.default_alignment.max(mem::align_of::<T>());

        let layout = Layout::from_size_align(size, align).ok()?;

        #[cfg(not(feature = "no-std"))]
        let ptr = unsafe { System.alloc_zeroed(layout) };
        #[cfg(feature = "no-std")]
        let ptr = unsafe { global_alloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            None
        } else {
            self.stats.record_allocation(size, true);
            NonNull::new(ptr.cast())
        }
    }

    /// Reallocate SIMD memory with preserved alignment
    pub unsafe fn reallocate_simd<T>(
        &self,
        ptr: NonNull<T>,
        old_count: usize,
        new_count: usize,
    ) -> Option<NonNull<T>> {
        let old_size = old_count * mem::size_of::<T>();
        let new_size = new_count * mem::size_of::<T>();
        let align = self.default_alignment.max(mem::align_of::<T>());

        let old_layout = Layout::from_size_align(old_size, align).ok()?;

        #[cfg(not(feature = "no-std"))]
        let new_ptr = System.realloc(ptr.cast().as_ptr(), old_layout, new_size);
        #[cfg(feature = "no-std")]
        let new_ptr = global_alloc::realloc(ptr.cast().as_ptr(), old_layout, new_size);

        if new_ptr.is_null() {
            None
        } else {
            self.stats.record_deallocation(old_size);
            self.stats.record_allocation(new_size, true);
            NonNull::new(new_ptr.cast())
        }
    }
}

impl Default for SimdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for SimdAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        #[cfg(not(feature = "no-std"))]
        let ptr = System.alloc(layout);
        #[cfg(feature = "no-std")]
        let ptr = global_alloc::alloc(layout);
        if !ptr.is_null() {
            let is_aligned = layout.align() >= self.default_alignment;
            self.stats.record_allocation(layout.size(), is_aligned);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        #[cfg(not(feature = "no-std"))]
        System.dealloc(ptr, layout);
        #[cfg(feature = "no-std")]
        global_alloc::dealloc(ptr, layout);
        self.stats.record_deallocation(layout.size());
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        #[cfg(not(feature = "no-std"))]
        let ptr = System.alloc_zeroed(layout);
        #[cfg(feature = "no-std")]
        let ptr = global_alloc::alloc_zeroed(layout);
        if !ptr.is_null() {
            let is_aligned = layout.align() >= self.default_alignment;
            self.stats.record_allocation(layout.size(), is_aligned);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        #[cfg(not(feature = "no-std"))]
        let new_ptr = System.realloc(ptr, layout, new_size);
        #[cfg(feature = "no-std")]
        let new_ptr = global_alloc::realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            self.stats.record_deallocation(layout.size());
            self.stats
                .record_allocation(new_size, layout.align() >= self.default_alignment);
        }
        new_ptr
    }
}

/// SIMD-aligned vector type with custom allocator
pub struct SimdVec<T> {
    ptr: Option<NonNull<T>>,
    len: usize,
    capacity: usize,
    allocator: SimdAllocator,
}

impl<T> SimdVec<T> {
    /// Create a new SIMD vector with default alignment
    pub fn new() -> Self {
        Self::with_allocator(SimdAllocator::new())
    }

    /// Create a new SIMD vector with custom allocator
    pub fn with_allocator(allocator: SimdAllocator) -> Self {
        Self {
            ptr: None,
            len: 0,
            capacity: 0,
            allocator,
        }
    }

    /// Create a SIMD vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        vec.reserve(capacity);
        vec
    }

    /// Reserve capacity for additional elements
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.len.checked_add(additional).expect("Capacity overflow");

        if new_capacity <= self.capacity {
            return;
        }

        let new_capacity = new_capacity.next_power_of_two().max(4);

        if let Some(old_ptr) = self.ptr {
            // Reallocate existing memory
            let new_ptr = unsafe {
                self.allocator
                    .reallocate_simd(old_ptr, self.capacity, new_capacity)
            };

            if let Some(new_ptr) = new_ptr {
                self.ptr = Some(new_ptr);
                self.capacity = new_capacity;
            } else {
                panic!("Failed to reallocate SIMD memory");
            }
        } else {
            // First allocation
            let new_ptr = self.allocator.allocate_simd::<T>(new_capacity);

            if let Some(new_ptr) = new_ptr {
                self.ptr = Some(new_ptr);
                self.capacity = new_capacity;
            } else {
                panic!("Failed to allocate SIMD memory");
            }
        }
    }

    /// Push an element to the vector
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.reserve(1);
        }

        unsafe {
            let ptr = self
                .ptr
                .expect("Vector should have allocated memory")
                .as_ptr();
            ptr::write(ptr.add(self.len), value);
        }

        self.len += 1;
    }

    /// Pop an element from the vector
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe {
                let ptr = self
                    .ptr
                    .expect("Vector should have allocated memory")
                    .as_ptr();
                Some(ptr::read(ptr.add(self.len)))
            }
        }
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a slice view of the vector
    pub fn as_slice(&self) -> &[T] {
        if let Some(ptr) = self.ptr {
            unsafe { slice::from_raw_parts(ptr.as_ptr(), self.len) }
        } else {
            &[]
        }
    }

    /// Get a mutable slice view of the vector
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if let Some(ptr) = self.ptr {
            unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), self.len) }
        } else {
            &mut []
        }
    }

    /// Clear the vector
    pub fn clear(&mut self) {
        if mem::needs_drop::<T>() {
            for i in 0..self.len {
                unsafe {
                    let ptr = self
                        .ptr
                        .expect("Vector should have allocated memory")
                        .as_ptr();
                    ptr::drop_in_place(ptr.add(i));
                }
            }
        }
        self.len = 0;
    }

    /// Get allocator statistics
    pub fn allocator_stats(&self) -> &AllocatorStats {
        self.allocator.stats()
    }

    /// Check if the underlying memory is properly aligned for SIMD
    pub fn is_simd_aligned(&self) -> bool {
        if let Some(ptr) = self.ptr {
            let addr = ptr.as_ptr() as usize;
            addr % self.allocator.default_alignment == 0
        } else {
            true // Empty vector is considered aligned
        }
    }
}

impl<T> Default for SimdVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for SimdVec<T> {
    fn drop(&mut self) {
        self.clear();

        if let Some(ptr) = self.ptr.take() {
            unsafe {
                self.allocator.deallocate_simd(ptr, self.capacity);
            }
        }
    }
}

impl<T: Clone> Clone for SimdVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_allocator(SimdAllocator::with_alignment(
            self.allocator.default_alignment,
        ));

        new_vec.reserve(self.len);

        for item in self.as_slice() {
            new_vec.push(item.clone());
        }

        new_vec
    }
}

/// Memory pool for frequent SIMD allocations
pub struct SimdMemoryPool<T> {
    free_blocks: Vec<(NonNull<T>, usize)>, // (ptr, capacity)
    allocator: SimdAllocator,
    block_size: usize,
}

impl<T> SimdMemoryPool<T> {
    pub fn new(block_size: usize) -> Self {
        Self {
            free_blocks: Vec::new(),
            allocator: SimdAllocator::new(),
            block_size,
        }
    }

    pub fn acquire(&mut self, min_capacity: usize) -> Option<(NonNull<T>, usize)> {
        // Try to find a suitable free block
        for (i, (_ptr, capacity)) in self.free_blocks.iter().enumerate() {
            if *capacity >= min_capacity {
                return Some(self.free_blocks.swap_remove(i));
            }
        }

        // Allocate a new block if no suitable free block found
        let capacity = min_capacity.max(self.block_size);
        let ptr = self.allocator.allocate_simd(capacity)?;
        Some((ptr, capacity))
    }

    pub fn release(&mut self, ptr: NonNull<T>, capacity: usize) {
        self.free_blocks.push((ptr, capacity));
    }

    pub fn clear(&mut self) {
        for (ptr, capacity) in self.free_blocks.drain(..) {
            unsafe {
                self.allocator.deallocate_simd(ptr, capacity);
            }
        }
    }

    pub fn stats(&self) -> &AllocatorStats {
        self.allocator.stats()
    }
}

impl<T> Drop for SimdMemoryPool<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_simd_allocator_basic() {
        let allocator = SimdAllocator::new();

        let ptr = allocator.allocate_simd::<f32>(16);
        assert!(ptr.is_some());

        if let Some(ptr) = ptr {
            // Check alignment
            let addr = ptr.as_ptr() as usize;
            assert_eq!(addr % 32, 0, "Memory should be 32-byte aligned");

            unsafe {
                allocator.deallocate_simd(ptr, 16);
            }
        }

        let stats = allocator.stats();
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_deallocations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_simd_vec_basic_operations() {
        let mut vec = SimdVec::<i32>::new();

        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
        assert!(vec.is_simd_aligned());

        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert!(!vec.is_empty());
        assert_eq!(vec.as_slice(), &[1, 2, 3]);

        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);

        vec.clear();
        assert!(vec.is_empty());
    }

    #[test]
    fn test_simd_vec_capacity_growth() {
        let mut vec = SimdVec::<u64>::new();

        for i in 0..100 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 100);
        assert!(vec.capacity() >= 100);
        assert!(vec.is_simd_aligned());

        // Check that values are correct
        for (i, &value) in vec.as_slice().iter().enumerate() {
            assert_eq!(value, i as u64);
        }
    }

    #[test]
    fn test_simd_vec_with_capacity() {
        let vec = SimdVec::<f64>::with_capacity(50);

        assert_eq!(vec.len(), 0);
        assert!(vec.capacity() >= 50);
        assert!(vec.is_simd_aligned());
    }

    #[test]
    fn test_allocator_stats() {
        let allocator = SimdAllocator::new();

        let ptr1 = allocator.allocate_simd::<f32>(16).unwrap();
        let ptr2 = allocator.allocate_simd::<f64>(8).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 2);
        assert_eq!(stats.aligned_allocations.load(Ordering::Relaxed), 2);
        assert!(stats.current_memory_usage() > 0);
        assert_eq!(stats.allocation_efficiency(), 1.0);

        unsafe {
            allocator.deallocate_simd(ptr1, 16);
            allocator.deallocate_simd(ptr2, 8);
        }

        assert_eq!(stats.total_deallocations.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = SimdMemoryPool::<i32>::new(64);

        let (ptr1, cap1) = pool.acquire(32).unwrap();
        assert!(cap1 >= 32);

        let (ptr2, cap2) = pool.acquire(16).unwrap();
        assert!(cap2 >= 16);

        pool.release(ptr1, cap1);

        // Should reuse the released block
        let (ptr3, cap3) = pool.acquire(30).unwrap();
        assert_eq!(ptr3, ptr1);
        assert_eq!(cap3, cap1);

        pool.release(ptr2, cap2);
        pool.release(ptr3, cap3);
    }

    #[test]
    fn test_zeroed_allocation() {
        let allocator = SimdAllocator::new();

        let ptr = allocator.allocate_zeroed_simd::<u32>(16).unwrap();

        unsafe {
            let slice = slice::from_raw_parts(ptr.as_ptr(), 16);
            for &value in slice {
                assert_eq!(value, 0);
            }

            allocator.deallocate_simd(ptr, 16);
        }
    }

    #[test]
    fn test_custom_alignment() {
        let allocator = SimdAllocator::with_alignment(64); // AVX-512 alignment

        let ptr = allocator.allocate_simd::<f32>(16);
        assert!(ptr.is_some());

        if let Some(ptr) = ptr {
            let addr = ptr.as_ptr() as usize;
            assert_eq!(addr % 64, 0, "Memory should be 64-byte aligned");

            unsafe {
                allocator.deallocate_simd(ptr, 16);
            }
        }
    }

    #[test]
    fn test_simd_vec_clone() {
        let mut vec1 = SimdVec::<i32>::new();
        vec1.push(1);
        vec1.push(2);
        vec1.push(3);

        let vec2 = vec1.clone();

        assert_eq!(vec1.as_slice(), vec2.as_slice());
        assert!(vec2.is_simd_aligned());
    }
}
