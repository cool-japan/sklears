//! Memory management utilities for high-performance ML workloads
//!
//! This module provides memory management utilities including custom allocators,
//! memory pools, leak detection, memory-mapped file utilities, bounds checking,
//! and safe memory management helpers.

use crate::{UtilsError, UtilsResult};
use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Custom allocator that tracks memory usage
pub struct TrackingAllocator<A: GlobalAlloc> {
    inner: A,
    stats: Arc<RwLock<AllocationStats>>,
}

/// Allocation statistics
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    pub total_allocated: u64,
    pub total_deallocated: u64,
    pub current_allocated: u64,
    pub peak_allocated: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub leak_count: u64,
}

impl<A: GlobalAlloc> TrackingAllocator<A> {
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            stats: Arc::new(RwLock::new(AllocationStats::default())),
        }
    }

    pub fn stats(&self) -> AllocationStats {
        self.stats.read().unwrap().clone()
    }

    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = AllocationStats::default();
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            let mut stats = self.stats.write().unwrap();
            stats.total_allocated += layout.size() as u64;
            stats.current_allocated += layout.size() as u64;
            stats.allocation_count += 1;
            if stats.current_allocated > stats.peak_allocated {
                stats.peak_allocated = stats.current_allocated;
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout);
        let mut stats = self.stats.write().unwrap();
        stats.total_deallocated += layout.size() as u64;
        stats.current_allocated = stats.current_allocated.saturating_sub(layout.size() as u64);
        stats.deallocation_count += 1;
    }
}

/// Memory pool for efficient allocation of fixed-size objects
pub struct MemoryPool<T> {
    blocks: Vec<Box<[T]>>,
    free_list: Vec<*mut T>,
    block_size: usize,
    stats: AllocationStats,
}

impl<T: Default + Clone> MemoryPool<T> {
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            free_list: Vec::new(),
            block_size,
            stats: AllocationStats::default(),
        }
    }

    pub fn allocate(&mut self) -> Option<&mut T> {
        if self.free_list.is_empty() {
            self.add_block();
        }

        if let Some(ptr) = self.free_list.pop() {
            self.stats.allocation_count += 1;
            self.stats.current_allocated += std::mem::size_of::<T>() as u64;
            unsafe { Some(&mut *ptr) }
        } else {
            None
        }
    }

    pub fn deallocate(&mut self, item: &mut T) {
        let ptr = item as *mut T;
        self.free_list.push(ptr);
        self.stats.deallocation_count += 1;
        self.stats.current_allocated = self
            .stats
            .current_allocated
            .saturating_sub(std::mem::size_of::<T>() as u64);
    }

    fn add_block(&mut self) {
        let mut block = vec![T::default(); self.block_size].into_boxed_slice();
        for item in block.iter_mut() {
            self.free_list.push(item as *mut T);
        }
        self.blocks.push(block);
        self.stats.total_allocated += (self.block_size * std::mem::size_of::<T>()) as u64;
    }

    pub fn stats(&self) -> &AllocationStats {
        &self.stats
    }

    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    pub fn used(&self) -> usize {
        self.capacity() - self.free_list.len()
    }
}

/// Memory leak detector
pub struct LeakDetector {
    allocations: Arc<Mutex<HashMap<usize, AllocationInfo>>>,
    enabled: bool,
}

#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub size: usize,
    pub timestamp: Instant,
    pub backtrace: String,
}

impl LeakDetector {
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            enabled: true,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn track_allocation(&self, ptr: *mut u8, size: usize) {
        if !self.enabled {
            return;
        }

        let mut allocations = self.allocations.lock().unwrap();
        allocations.insert(
            ptr as usize,
            AllocationInfo {
                size,
                timestamp: Instant::now(),
                backtrace: format!("Allocation at {ptr:p}"), // In real implementation, use backtrace crate
            },
        );
    }

    pub fn track_deallocation(&self, ptr: *mut u8) {
        if !self.enabled {
            return;
        }

        let mut allocations = self.allocations.lock().unwrap();
        allocations.remove(&(ptr as usize));
    }

    pub fn check_leaks(&self) -> Vec<AllocationInfo> {
        let allocations = self.allocations.lock().unwrap();
        allocations.values().cloned().collect()
    }

    pub fn check_leaks_older_than(&self, duration: Duration) -> Vec<AllocationInfo> {
        let allocations = self.allocations.lock().unwrap();
        let now = Instant::now();
        allocations
            .values()
            .filter(|info| now.duration_since(info.timestamp) > duration)
            .cloned()
            .collect()
    }

    pub fn total_leaked_bytes(&self) -> usize {
        let allocations = self.allocations.lock().unwrap();
        allocations.values().map(|info| info.size).sum()
    }

    pub fn clear(&self) {
        let mut allocations = self.allocations.lock().unwrap();
        allocations.clear();
    }
}

impl Default for LeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-mapped file utilities
pub struct MemoryMappedFile {
    #[allow(dead_code)]
    file: File,
    ptr: *mut u8,
    size: usize,
}

impl MemoryMappedFile {
    #[cfg(unix)]
    pub fn new(file: File, writable: bool) -> Result<Self, std::io::Error> {
        use std::os::unix::io::AsRawFd;

        let size = file.metadata()?.len() as usize;
        let prot = if writable {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                prot,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self {
            file,
            ptr: ptr as *mut u8,
            size,
        })
    }

    #[cfg(windows)]
    pub fn new(file: File, writable: bool) -> Result<Self, std::io::Error> {
        use std::os::windows::io::AsRawHandle;
        use winapi::um::handleapi::CloseHandle;
        use winapi::um::memoryapi::{
            CreateFileMappingW, MapViewOfFile, FILE_MAP_READ, FILE_MAP_WRITE,
        };
        use winapi::um::winnt::{PAGE_READONLY, PAGE_READWRITE};

        let size = file.metadata()?.len() as usize;
        let protect = if writable {
            PAGE_READWRITE
        } else {
            PAGE_READONLY
        };
        let access = if writable {
            FILE_MAP_WRITE
        } else {
            FILE_MAP_READ
        };

        let mapping = unsafe {
            CreateFileMappingW(
                file.as_raw_handle() as _,
                std::ptr::null_mut(),
                protect,
                0,
                0,
                std::ptr::null(),
            )
        };

        if mapping.is_null() {
            return Err(std::io::Error::last_os_error());
        }

        let ptr = unsafe { MapViewOfFile(mapping, access, 0, 0, 0) };
        unsafe { CloseHandle(mapping) };

        if ptr.is_null() {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self {
            file,
            ptr: ptr as *mut u8,
            size,
        })
    }

    #[cfg(not(any(unix, windows)))]
    pub fn new(_file: File, _writable: bool) -> Result<Self, std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Memory mapping not supported on this platform",
        ))
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for MemoryMappedFile {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(unix)]
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.size);
            }

            #[cfg(windows)]
            unsafe {
                winapi::um::memoryapi::UnmapViewOfFile(self.ptr as *mut winapi::ctypes::c_void);
            }
        }
    }
}

unsafe impl Send for MemoryMappedFile {}
unsafe impl Sync for MemoryMappedFile {}

/// Garbage collection helper for reference counting
pub struct GcHelper<T> {
    data: Arc<T>,
    weak_refs: Arc<Mutex<Vec<std::sync::Weak<T>>>>,
}

impl<T> GcHelper<T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(data),
            weak_refs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_ref(&self) -> Arc<T> {
        self.data.clone()
    }

    pub fn get_weak_ref(&self) -> std::sync::Weak<T> {
        let weak = Arc::downgrade(&self.data);
        let mut refs = self.weak_refs.lock().unwrap();
        refs.push(weak.clone());
        weak
    }

    pub fn collect_garbage(&self) {
        let mut refs = self.weak_refs.lock().unwrap();
        refs.retain(|weak_ref| weak_ref.upgrade().is_some());
    }

    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    pub fn weak_ref_count(&self) -> usize {
        let refs = self.weak_refs.lock().unwrap();
        refs.len()
    }
}

impl<T> Clone for GcHelper<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            weak_refs: self.weak_refs.clone(),
        }
    }
}

/// Memory usage monitor
pub struct MemoryMonitor {
    start_time: Instant,
    peak_memory: u64,
    current_memory: u64,
    samples: Vec<(Instant, u64)>,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            peak_memory: 0,
            current_memory: 0,
            samples: Vec::new(),
        }
    }

    pub fn update(&mut self, memory_usage: u64) {
        self.current_memory = memory_usage;
        if memory_usage > self.peak_memory {
            self.peak_memory = memory_usage;
        }
        self.samples.push((Instant::now(), memory_usage));
    }

    pub fn peak_memory(&self) -> u64 {
        self.peak_memory
    }

    pub fn current_memory(&self) -> u64 {
        self.current_memory
    }

    pub fn average_memory(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.samples.iter().map(|(_, mem)| *mem).sum();
        sum as f64 / self.samples.len() as f64
    }

    pub fn memory_over_time(&self) -> &[(Instant, u64)] {
        &self.samples
    }

    pub fn duration(&self) -> Duration {
        Instant::now().duration_since(self.start_time)
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ===== BOUNDS CHECKING HELPERS =====

/// Safe wrapper around vectors with bounds checking
#[derive(Debug, Clone)]
pub struct SafeVec<T> {
    data: Vec<T>,
    bounds_check: bool,
}

impl<T> SafeVec<T> {
    /// Create a new safe vector with bounds checking enabled
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bounds_check: true,
        }
    }

    /// Create a safe vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bounds_check: true,
        }
    }

    /// Create from existing vector
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self {
            data: vec,
            bounds_check: true,
        }
    }

    /// Disable bounds checking for performance (unsafe)
    pub fn disable_bounds_check(mut self) -> Self {
        self.bounds_check = false;
        self
    }

    /// Safe get with bounds checking
    pub fn get(&self, index: usize) -> UtilsResult<&T> {
        if self.bounds_check && index >= self.data.len() {
            return Err(UtilsError::InvalidParameter(format!(
                "Index {} out of bounds for vector of length {}",
                index,
                self.data.len()
            )));
        }
        self.data
            .get(index)
            .ok_or_else(|| UtilsError::InvalidParameter(format!("Index {index} out of bounds")))
    }

    /// Safe mutable get with bounds checking
    pub fn get_mut(&mut self, index: usize) -> UtilsResult<&mut T> {
        if self.bounds_check && index >= self.data.len() {
            return Err(UtilsError::InvalidParameter(format!(
                "Index {} out of bounds for vector of length {}",
                index,
                self.data.len()
            )));
        }
        let len = self.data.len();
        self.data.get_mut(index).ok_or_else(|| {
            UtilsError::InvalidParameter(format!(
                "Index {index} out of bounds for vector of length {len}"
            ))
        })
    }

    /// Safe slice access
    pub fn safe_slice(&self, start: usize, end: usize) -> UtilsResult<&[T]> {
        if self.bounds_check {
            if start > end {
                return Err(UtilsError::InvalidParameter(
                    "Start index cannot be greater than end index".to_string(),
                ));
            }
            if end > self.data.len() {
                return Err(UtilsError::InvalidParameter(format!(
                    "End index {end} out of bounds for vector of length {}",
                    self.data.len()
                )));
            }
        }
        Ok(&self.data[start..end])
    }

    /// Push element
    pub fn push(&mut self, item: T) {
        self.data.push(item);
    }

    /// Pop element
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Access underlying vector (unsafe)
    ///
    /// # Safety
    ///
    /// This function exposes the underlying Vec<T> directly, bypassing all
    /// bounds checking and overflow protection mechanisms. The caller must
    /// ensure that any modifications to the returned vector do not violate
    /// the buffer's safety guarantees and internal invariants.
    pub unsafe fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// Convert to vector
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> Default for SafeVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Safe buffer with automatic bounds checking and buffer overflow protection
#[derive(Debug, Clone)]
pub struct SafeBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    size: usize,
    overflow_protection: bool,
}

impl<T: Clone> SafeBuffer<T> {
    /// Create a new safe buffer with fixed capacity
    pub fn new(capacity: usize, default_value: T) -> Self {
        Self {
            data: vec![default_value; capacity],
            capacity,
            size: 0,
            overflow_protection: true,
        }
    }

    /// Write to buffer with bounds checking
    pub fn write(&mut self, index: usize, value: T) -> UtilsResult<()> {
        if self.overflow_protection && index >= self.capacity {
            return Err(UtilsError::InvalidParameter(format!(
                "Buffer overflow: index {} exceeds capacity {}",
                index, self.capacity
            )));
        }

        if index < self.data.len() {
            self.data[index] = value;
            self.size = self.size.max(index + 1);
            Ok(())
        } else {
            Err(UtilsError::InvalidParameter(format!(
                "Index {} out of bounds for buffer of capacity {}",
                index, self.capacity
            )))
        }
    }

    /// Read from buffer with bounds checking
    pub fn read(&self, index: usize) -> UtilsResult<&T> {
        if index >= self.size {
            return Err(UtilsError::InvalidParameter(format!(
                "Index {} out of bounds for buffer of size {}",
                index, self.size
            )));
        }

        self.data
            .get(index)
            .ok_or_else(|| UtilsError::InvalidParameter(format!("Index {index} out of bounds")))
    }

    /// Append to buffer
    pub fn append(&mut self, value: T) -> UtilsResult<()> {
        if self.size >= self.capacity {
            return Err(UtilsError::InvalidParameter(
                "Buffer overflow: cannot append to full buffer".to_string(),
            ));
        }

        self.data[self.size] = value;
        self.size += 1;
        Ok(())
    }

    /// Get current size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.size >= self.capacity
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.size = 0;
    }

    /// Disable overflow protection (unsafe)
    ///
    /// # Safety
    ///
    /// This function disables the buffer's overflow protection mechanism,
    /// allowing operations that could potentially lead to buffer overflows
    /// or memory corruption. The caller must ensure that all subsequent
    /// operations on the buffer are within proper bounds.
    pub unsafe fn disable_overflow_protection(&mut self) {
        self.overflow_protection = false;
    }
}

/// Memory-safe smart pointer with reference counting and automatic cleanup
pub struct SafePtr<T> {
    data: Arc<RwLock<Option<T>>>,
    cleanup_fn: Option<Box<dyn Fn() + Send + Sync>>,
}

impl<T> std::fmt::Debug for SafePtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafePtr")
            .field("data", &"Arc<RwLock<Option<T>>>")
            .field(
                "cleanup_fn",
                &self.cleanup_fn.as_ref().map(|_| "Some(cleanup_fn)"),
            )
            .finish()
    }
}

impl<T> SafePtr<T> {
    /// Create a new safe pointer
    pub fn new(value: T) -> Self {
        Self {
            data: Arc::new(RwLock::new(Some(value))),
            cleanup_fn: None,
        }
    }

    /// Create with cleanup function
    pub fn with_cleanup<F>(value: T, cleanup: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        Self {
            data: Arc::new(RwLock::new(Some(value))),
            cleanup_fn: Some(Box::new(cleanup)),
        }
    }

    /// Try to read the value
    pub fn try_read(&self) -> UtilsResult<std::sync::RwLockReadGuard<'_, Option<T>>> {
        self.data
            .read()
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to acquire read lock: {e}")))
    }

    /// Try to write the value
    pub fn try_write(&self) -> UtilsResult<std::sync::RwLockWriteGuard<'_, Option<T>>> {
        self.data
            .write()
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to acquire write lock: {e}")))
    }

    /// Check if the pointer is still valid
    pub fn is_valid(&self) -> bool {
        if let Ok(guard) = self.data.read() {
            guard.is_some()
        } else {
            false
        }
    }

    /// Take the value, leaving None
    pub fn take(&self) -> UtilsResult<Option<T>> {
        let mut guard = self.try_write()?;
        Ok(guard.take())
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
}

impl<T> Clone for SafePtr<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            cleanup_fn: None, // Cleanup functions are not cloned
        }
    }
}

impl<T> Drop for SafePtr<T> {
    fn drop(&mut self) {
        // Run cleanup function if this is the last reference
        if Arc::strong_count(&self.data) == 1 {
            if let Some(cleanup) = &self.cleanup_fn {
                cleanup();
            }
        }
    }
}

/// Memory alignment utilities
pub struct MemoryAlignment;

impl MemoryAlignment {
    /// Check if a pointer is aligned to the specified boundary
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// Get the alignment of a type
    pub fn alignment_of<T>() -> usize {
        std::mem::align_of::<T>()
    }

    /// Calculate aligned size
    pub fn aligned_size(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Create aligned memory layout
    pub fn aligned_layout(
        size: usize,
        alignment: usize,
    ) -> Result<Layout, std::alloc::LayoutError> {
        Layout::from_size_align(size, alignment)
    }
}

/// Stack-based memory guard for automatic cleanup
pub struct StackGuard<F: FnOnce()> {
    cleanup: Option<F>,
}

impl<F: FnOnce()> StackGuard<F> {
    /// Create a new stack guard with cleanup function
    pub fn new(cleanup: F) -> Self {
        Self {
            cleanup: Some(cleanup),
        }
    }

    /// Manually trigger cleanup (consumes the guard)
    pub fn cleanup(mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

impl<F: FnOnce()> Drop for StackGuard<F> {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

/// Macro for creating stack guards
#[macro_export]
macro_rules! defer {
    ($cleanup:expr) => {
        let _guard = $crate::memory::StackGuard::new(|| $cleanup);
    };
}

/// Memory validation utilities
pub struct MemoryValidator;

impl MemoryValidator {
    /// Validate that a memory range is accessible
    ///
    /// # Safety
    ///
    /// This function performs raw pointer arithmetic and validation. The caller
    /// must ensure that the provided pointer was obtained through safe means and
    /// that the memory range [ptr, ptr + count * size_of::<T>()) is within valid
    /// allocated memory boundaries. Incorrect usage can lead to undefined behavior.
    pub unsafe fn validate_range<T>(ptr: *const T, count: usize) -> UtilsResult<()> {
        if ptr.is_null() {
            return Err(UtilsError::InvalidParameter("Null pointer".to_string()));
        }

        // Basic overflow check
        let end_ptr = unsafe { ptr.add(count) };
        if end_ptr < ptr {
            return Err(UtilsError::InvalidParameter("Pointer overflow".to_string()));
        }

        Ok(())
    }

    /// Validate memory alignment
    pub fn validate_alignment<T>(ptr: *const T, required_alignment: usize) -> UtilsResult<()> {
        if !MemoryAlignment::is_aligned(ptr, required_alignment) {
            return Err(UtilsError::InvalidParameter(format!(
                "Pointer not aligned to {required_alignment} byte boundary"
            )));
        }
        Ok(())
    }

    /// Validate buffer bounds
    pub fn validate_buffer_access(
        buffer_size: usize,
        offset: usize,
        access_size: usize,
    ) -> UtilsResult<()> {
        if offset >= buffer_size {
            return Err(UtilsError::InvalidParameter(format!(
                "Offset {offset} exceeds buffer size {buffer_size}"
            )));
        }

        if offset + access_size > buffer_size {
            return Err(UtilsError::InvalidParameter(format!(
                "Access range {}..{} exceeds buffer size {}",
                offset,
                offset + access_size,
                buffer_size
            )));
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::System;

    #[test]
    fn test_tracking_allocator_stats() {
        let allocator = TrackingAllocator::new(System);
        let initial_stats = allocator.stats();
        assert_eq!(initial_stats.allocation_count, 0);
        assert_eq!(initial_stats.current_allocated, 0);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool: MemoryPool<u64> = MemoryPool::new(10);

        {
            let item1 = pool.allocate().unwrap();
            *item1 = 42;
            assert_eq!(*item1, 42);
        }
        assert_eq!(pool.used(), 1);

        {
            let item2 = pool.allocate().unwrap();
            *item2 = 84;
            assert_eq!(*item2, 84);
        }
        assert_eq!(pool.used(), 2);

        // Note: In a real implementation, deallocate would need the item reference
        // For this test, we'll just check that the pool tracks allocation correctly
        assert_eq!(pool.capacity(), 10);
    }

    #[test]
    fn test_leak_detector() {
        let detector = LeakDetector::new();
        let ptr = Box::into_raw(Box::new(42u64));

        detector.track_allocation(ptr as *mut u8, 8);
        assert_eq!(detector.total_leaked_bytes(), 8);

        detector.track_deallocation(ptr as *mut u8);
        assert_eq!(detector.total_leaked_bytes(), 0);

        unsafe { drop(Box::from_raw(ptr)) };
    }

    #[test]
    fn test_gc_helper() {
        let gc = GcHelper::new(42u64);
        assert_eq!(gc.ref_count(), 1);

        let strong_ref = gc.get_ref();
        assert_eq!(gc.ref_count(), 2);
        assert_eq!(*strong_ref, 42);

        let weak_ref = gc.get_weak_ref();
        assert!(weak_ref.upgrade().is_some());

        drop(strong_ref);
        assert_eq!(gc.ref_count(), 1);
    }

    #[test]
    fn test_memory_monitor() {
        let mut monitor = MemoryMonitor::new();
        assert_eq!(monitor.peak_memory(), 0);
        assert_eq!(monitor.current_memory(), 0);

        monitor.update(1024);
        assert_eq!(monitor.peak_memory(), 1024);
        assert_eq!(monitor.current_memory(), 1024);

        monitor.update(512);
        assert_eq!(monitor.peak_memory(), 1024);
        assert_eq!(monitor.current_memory(), 512);

        monitor.update(2048);
        assert_eq!(monitor.peak_memory(), 2048);
        assert_eq!(monitor.current_memory(), 2048);

        assert_eq!(monitor.average_memory(), (1024.0 + 512.0 + 2048.0) / 3.0);
    }

    #[test]
    fn test_safe_vec() {
        let mut safe_vec = SafeVec::new();
        safe_vec.push(1);
        safe_vec.push(2);
        safe_vec.push(3);

        // Valid access
        assert_eq!(*safe_vec.get(0).unwrap(), 1);
        assert_eq!(*safe_vec.get(2).unwrap(), 3);

        // Invalid access
        assert!(safe_vec.get(5).is_err());

        // Safe slice
        let slice = safe_vec.safe_slice(1, 3).unwrap();
        assert_eq!(slice, &[2, 3]);

        // Invalid slice
        assert!(safe_vec.safe_slice(2, 5).is_err());
        assert!(safe_vec.safe_slice(3, 2).is_err());
    }

    #[test]
    fn test_safe_buffer() {
        let mut buffer = SafeBuffer::new(5, 0);

        // Write and read
        buffer.write(0, 42).unwrap();
        buffer.write(1, 84).unwrap();

        assert_eq!(*buffer.read(0).unwrap(), 42);
        assert_eq!(*buffer.read(1).unwrap(), 84);
        assert_eq!(buffer.size(), 2);

        // Append
        buffer.append(100).unwrap();
        buffer.append(200).unwrap();
        buffer.append(300).unwrap();

        assert!(buffer.is_full());
        assert!(buffer.append(400).is_err()); // Should fail - buffer full

        // Buffer overflow protection
        assert!(buffer.write(10, 500).is_err()); // Should fail - out of bounds
    }

    #[test]
    fn test_safe_ptr() {
        let ptr = SafePtr::new(42);
        assert!(ptr.is_valid());
        assert_eq!(ptr.ref_count(), 1);

        // Clone increases ref count
        let _ptr2 = ptr.clone();
        assert_eq!(ptr.ref_count(), 2);

        // Read value
        {
            let guard = ptr.try_read().unwrap();
            assert_eq!(*guard, Some(42));
        }

        // Take value
        let value = ptr.take().unwrap();
        assert_eq!(value, Some(42));
        assert!(!ptr.is_valid());
    }

    #[test]
    fn test_memory_alignment() {
        // Test alignment checking
        let data = 42u64;
        let ptr = &data as *const u64;

        assert!(MemoryAlignment::is_aligned(ptr, 8)); // u64 should be 8-byte aligned
        assert_eq!(MemoryAlignment::alignment_of::<u64>(), 8);

        // Test aligned size calculation
        assert_eq!(MemoryAlignment::aligned_size(10, 8), 16);
        assert_eq!(MemoryAlignment::aligned_size(16, 8), 16);
        assert_eq!(MemoryAlignment::aligned_size(17, 8), 24);
    }

    #[test]
    fn test_stack_guard() {
        use std::sync::Arc;

        let cleanup_called = Arc::new(Mutex::new(false));
        let cleanup_called_clone = cleanup_called.clone();

        {
            let _guard = StackGuard::new(|| {
                *cleanup_called_clone.lock().unwrap() = true;
            });

            // Cleanup not called yet
            assert!(!*cleanup_called.lock().unwrap());
        } // Guard drops here

        // Cleanup should be called now
        assert!(*cleanup_called.lock().unwrap());
    }

    #[test]
    fn test_memory_validator() {
        // Test null pointer validation
        let null_ptr: *const u8 = std::ptr::null();
        assert!(unsafe { MemoryValidator::validate_range(null_ptr, 10) }.is_err());

        // Test valid pointer
        let data = [1u8, 2, 3, 4, 5];
        let ptr = data.as_ptr();
        assert!(unsafe { MemoryValidator::validate_range(ptr, 5) }.is_ok());

        // Test alignment validation
        let aligned_ptr = &42u64 as *const u64;
        assert!(MemoryValidator::validate_alignment(aligned_ptr, 8).is_ok());

        // Test buffer access validation
        assert!(MemoryValidator::validate_buffer_access(10, 0, 5).is_ok());
        assert!(MemoryValidator::validate_buffer_access(10, 5, 5).is_ok());
        assert!(MemoryValidator::validate_buffer_access(10, 10, 1).is_err()); // Offset too large
        assert!(MemoryValidator::validate_buffer_access(10, 8, 5).is_err()); // Access exceeds buffer
    }

    #[test]
    fn test_defer_macro() {
        use std::sync::Arc;

        let cleanup_called = Arc::new(Mutex::new(false));
        let cleanup_called_clone = cleanup_called.clone();

        {
            defer!({
                *cleanup_called_clone.lock().unwrap() = true;
            });

            // Cleanup not called yet
            assert!(!*cleanup_called.lock().unwrap());
        } // Deferred cleanup happens here

        // Cleanup should be called now
        assert!(*cleanup_called.lock().unwrap());
    }
}
