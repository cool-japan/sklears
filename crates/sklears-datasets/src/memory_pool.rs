//! Memory pool allocation for high-performance dataset generation
//!
//! This module provides sophisticated memory pool management for efficient
//! allocation and reuse of memory blocks in dataset generation operations.
//! It reduces allocation overhead and improves cache locality.

use scirs2_core::ndarray::{Array1, Array2, ArrayViewMut1, ArrayViewMut2};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;

/// Memory pool errors
#[derive(Error, Debug)]
pub enum MemoryPoolError {
    #[error("Pool exhausted: no available blocks of size {0}")]
    PoolExhausted(usize),
    #[error("Invalid block size: {0}")]
    InvalidBlockSize(usize),
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Block not found: {0:?}")]
    BlockNotFound(usize),
    #[error("Double free detected: {0:?}")]
    DoubleFree(usize),
    #[error("Pool corruption detected: {0}")]
    PoolCorruption(String),
    #[error("Alignment error: required {required}, got {actual}")]
    AlignmentError { required: usize, actual: usize },
}

pub type MemoryPoolResult<T> = Result<T, MemoryPoolError>;

/// Memory block metadata
#[derive(Debug, Clone)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
    allocated_at: std::time::Instant,
    pool_id: usize,
    magic: u64, // For corruption detection
}

const MAGIC_VALUE: u64 = 0xDEADBEEFCAFEBABE;

impl MemoryBlock {
    fn new(size: usize, align: usize, pool_id: usize) -> MemoryPoolResult<Self> {
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| MemoryPoolError::AllocationFailed(e.to_string()))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(MemoryPoolError::AllocationFailed(
                "System allocation failed".to_string(),
            ));
        }

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            layout,
            allocated_at: std::time::Instant::now(),
            pool_id,
            magic: MAGIC_VALUE,
        })
    }

    fn is_valid(&self) -> bool {
        self.magic == MAGIC_VALUE
    }

    fn invalidate(&mut self) {
        self.magic = 0;
    }

    fn as_slice_mut(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    fn as_f64_slice_mut(&self) -> MemoryPoolResult<&mut [f64]> {
        if self.size % std::mem::size_of::<f64>() != 0 {
            return Err(MemoryPoolError::InvalidBlockSize(self.size));
        }

        let len = self.size / std::mem::size_of::<f64>();
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut f64, len) })
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        if self.is_valid() {
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
            self.invalidate();
        }
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial number of blocks per size bucket
    pub initial_blocks_per_size: usize,
    /// Maximum number of blocks per size bucket
    pub max_blocks_per_size: usize,
    /// Growth factor when expanding pool
    pub growth_factor: f64,
    /// Alignment requirement for blocks
    pub alignment: usize,
    /// Enable block reuse
    pub enable_reuse: bool,
    /// Maximum idle time before releasing blocks
    pub max_idle_time: std::time::Duration,
    /// Enable statistics collection
    pub enable_stats: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_blocks_per_size: 4,
            max_blocks_per_size: 64,
            growth_factor: 1.5,
            alignment: 64, // Cache line alignment
            enable_reuse: true,
            max_idle_time: std::time::Duration::from_secs(300), // 5 minutes
            enable_stats: true,
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_allocations: usize,
    pub peak_allocations: usize,
    pub total_bytes_allocated: usize,
    pub current_bytes_allocated: usize,
    pub peak_bytes_allocated: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub blocks_created: usize,
    pub blocks_reused: usize,
    pub blocks_released: usize,
}

impl MemoryPoolStats {
    pub fn hit_rate(&self) -> f64 {
        if self.pool_hits + self.pool_misses == 0 {
            0.0
        } else {
            self.pool_hits as f64 / (self.pool_hits + self.pool_misses) as f64
        }
    }

    pub fn reuse_rate(&self) -> f64 {
        if self.blocks_created == 0 {
            0.0
        } else {
            self.blocks_reused as f64 / self.blocks_created as f64
        }
    }
}

/// Size bucket for grouping similar-sized allocations
struct SizeBucket {
    size: usize,
    available: VecDeque<MemoryBlock>,
    allocated: HashMap<usize, MemoryBlock>,
    total_created: usize,
}

impl SizeBucket {
    fn new(size: usize) -> Self {
        Self {
            size,
            available: VecDeque::new(),
            allocated: HashMap::new(),
            total_created: 0,
        }
    }

    fn allocate(
        &mut self,
        pool_id: usize,
        config: &MemoryPoolConfig,
    ) -> MemoryPoolResult<(usize, *mut u8)> {
        let block = if let Some(block) = self.available.pop_front() {
            block
        } else {
            // Create new block
            let block = MemoryBlock::new(self.size, config.alignment, pool_id)?;
            self.total_created += 1;
            block
        };

        let ptr = block.ptr.as_ptr();
        let block_id = ptr as usize;
        self.allocated.insert(block_id, block);

        Ok((block_id, ptr))
    }

    fn deallocate(&mut self, block_id: usize, enable_reuse: bool) -> MemoryPoolResult<()> {
        if let Some(mut block) = self.allocated.remove(&block_id) {
            if !block.is_valid() {
                return Err(MemoryPoolError::PoolCorruption(
                    "Block magic value corrupted".to_string(),
                ));
            }

            if enable_reuse {
                // Zero out the memory for security
                let slice = block.as_slice_mut();
                slice.fill(0);

                self.available.push_back(block);
            } else {
                // Block will be dropped and memory freed
                block.invalidate();
            }

            Ok(())
        } else {
            Err(MemoryPoolError::BlockNotFound(block_id))
        }
    }

    fn cleanup_idle(&mut self, max_idle_time: std::time::Duration) -> usize {
        let now = std::time::Instant::now();
        let mut removed = 0;

        self.available.retain(|block| {
            if now.duration_since(block.allocated_at) > max_idle_time {
                removed += 1;
                false
            } else {
                true
            }
        });

        removed
    }
}

/// High-performance memory pool for dataset generation
pub struct MemoryPool {
    config: MemoryPoolConfig,
    buckets: RwLock<HashMap<usize, SizeBucket>>,
    stats: RwLock<MemoryPoolStats>,
    pool_id: usize,
    next_cleanup: Mutex<std::time::Instant>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(config: MemoryPoolConfig) -> Self {
        Self {
            config,
            buckets: RwLock::new(HashMap::new()),
            stats: RwLock::new(MemoryPoolStats::default()),
            pool_id: 0, // TODO: implement proper pool ID generation
            next_cleanup: Mutex::new(std::time::Instant::now() + config.max_idle_time),
        }
    }

    /// Allocate a block of memory
    pub fn allocate(&self, size: usize) -> MemoryPoolResult<PooledBlock> {
        if size == 0 {
            return Err(MemoryPoolError::InvalidBlockSize(size));
        }

        // Round up to nearest power of 2 for better bucketing
        let bucket_size = size.next_power_of_two().max(64);

        let (block_id, ptr) = {
            let mut buckets = self.buckets.write().unwrap();
            let bucket = buckets
                .entry(bucket_size)
                .or_insert_with(|| SizeBucket::new(bucket_size));

            // Check if we have room for more allocations
            if bucket.allocated.len() >= self.config.max_blocks_per_size {
                return Err(MemoryPoolError::PoolExhausted(bucket_size));
            }

            bucket.allocate(self.pool_id, &self.config)?
        };

        // Update statistics
        if self.config.enable_stats {
            let mut stats = self.stats.write().unwrap();
            stats.total_allocations += 1;
            stats.current_allocations += 1;
            stats.peak_allocations = stats.peak_allocations.max(stats.current_allocations);
            stats.total_bytes_allocated += bucket_size;
            stats.current_bytes_allocated += bucket_size;
            stats.peak_bytes_allocated = stats
                .peak_bytes_allocated
                .max(stats.current_bytes_allocated);

            if bucket_size != size {
                stats.pool_hits += 1;
            } else {
                stats.pool_misses += 1;
            }
        }

        // Periodic cleanup
        self.maybe_cleanup();

        Ok(PooledBlock {
            ptr,
            size: bucket_size,
            block_id,
            pool: self as *const Self,
        })
    }

    /// Deallocate a block of memory
    pub fn deallocate(&self, block_id: usize, size: usize) -> MemoryPoolResult<()> {
        let bucket_size = size.next_power_of_two().max(64);

        {
            let mut buckets = self.buckets.write().unwrap();
            if let Some(bucket) = buckets.get_mut(&bucket_size) {
                bucket.deallocate(block_id, self.config.enable_reuse)?;
            } else {
                return Err(MemoryPoolError::BlockNotFound(block_id));
            }
        }

        // Update statistics
        if self.config.enable_stats {
            let mut stats = self.stats.write().unwrap();
            stats.total_deallocations += 1;
            stats.current_allocations = stats.current_allocations.saturating_sub(1);
            stats.current_bytes_allocated =
                stats.current_bytes_allocated.saturating_sub(bucket_size);
        }

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> MemoryPoolStats {
        if self.config.enable_stats {
            self.stats.read().unwrap().clone()
        } else {
            MemoryPoolStats::default()
        }
    }

    /// Force cleanup of idle blocks
    pub fn cleanup(&self) -> usize {
        let mut total_removed = 0;
        let mut buckets = self.buckets.write().unwrap();

        for bucket in buckets.values_mut() {
            total_removed += bucket.cleanup_idle(self.config.max_idle_time);
        }

        // Update next cleanup time
        *self.next_cleanup.lock().unwrap() = std::time::Instant::now() + self.config.max_idle_time;

        total_removed
    }

    /// Maybe perform cleanup if enough time has passed
    fn maybe_cleanup(&self) {
        let now = std::time::Instant::now();
        let should_cleanup = {
            let next_cleanup = self.next_cleanup.lock().unwrap();
            now >= *next_cleanup
        };

        if should_cleanup {
            self.cleanup();
        }
    }

    /// Allocate memory for a 2D array
    pub fn allocate_array2(&self, nrows: usize, ncols: usize) -> MemoryPoolResult<PooledArray2> {
        let size = nrows * ncols * std::mem::size_of::<f64>();
        let block = self.allocate(size)?;

        Ok(PooledArray2 {
            block,
            nrows,
            ncols,
        })
    }

    /// Allocate memory for a 1D array
    pub fn allocate_array1(&self, len: usize) -> MemoryPoolResult<PooledArray1> {
        let size = len * std::mem::size_of::<f64>();
        let block = self.allocate(size)?;

        Ok(PooledArray1 { block, len })
    }

    /// Get bucket information for debugging
    pub fn bucket_info(&self) -> Vec<(usize, usize, usize)> {
        let buckets = self.buckets.read().unwrap();
        buckets
            .iter()
            .map(|(size, bucket)| (*size, bucket.available.len(), bucket.allocated.len()))
            .collect()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new(MemoryPoolConfig::default())
    }
}

/// RAII wrapper for pooled memory blocks
pub struct PooledBlock {
    ptr: *mut u8,
    size: usize,
    block_id: usize,
    pool: *const MemoryPool,
}

impl PooledBlock {
    /// Get a mutable slice view of the block
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Get the block size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a mutable slice as f64 values
    pub fn as_f64_slice_mut(&mut self) -> MemoryPoolResult<&mut [f64]> {
        if self.size % std::mem::size_of::<f64>() != 0 {
            return Err(MemoryPoolError::InvalidBlockSize(self.size));
        }

        let len = self.size / std::mem::size_of::<f64>();
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f64, len) })
    }
}

impl Drop for PooledBlock {
    fn drop(&mut self) {
        if !self.pool.is_null() {
            unsafe {
                if let Err(e) = (*self.pool).deallocate(self.block_id, self.size) {
                    eprintln!("Error deallocating pooled block: {}", e);
                }
            }
        }
    }
}

unsafe impl Send for PooledBlock {}
unsafe impl Sync for PooledBlock {}

/// Pooled 2D array
pub struct PooledArray2 {
    block: PooledBlock,
    nrows: usize,
    ncols: usize,
}

impl PooledArray2 {
    /// Get dimensions
    pub fn dim(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Get mutable array view
    pub fn view_mut(&mut self) -> MemoryPoolResult<ArrayViewMut2<f64>> {
        let slice = self.block.as_f64_slice_mut()?;
        ArrayViewMut2::from_shape((self.nrows, self.ncols), slice)
            .map_err(|e| MemoryPoolError::AllocationFailed(format!("Shape error: {}", e)))
    }

    /// Convert to owned Array2
    pub fn to_array(mut self) -> MemoryPoolResult<Array2<f64>> {
        let slice = self.block.as_f64_slice_mut()?;
        let array =
            Array2::from_shape_vec((self.nrows, self.ncols), slice.to_vec()).map_err(|e| {
                MemoryPoolError::AllocationFailed(format!("Array creation error: {}", e))
            })?;
        Ok(array)
    }
}

/// Pooled 1D array
pub struct PooledArray1 {
    block: PooledBlock,
    len: usize,
}

impl PooledArray1 {
    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get mutable array view
    pub fn view_mut(&mut self) -> MemoryPoolResult<ArrayViewMut1<f64>> {
        let slice = self.block.as_f64_slice_mut()?;
        ArrayViewMut1::from_shape(self.len, slice)
            .map_err(|e| MemoryPoolError::AllocationFailed(format!("Shape error: {}", e)))
    }

    /// Convert to owned Array1
    pub fn to_array(mut self) -> MemoryPoolResult<Array1<f64>> {
        let slice = self.block.as_f64_slice_mut()?;
        let array = Array1::from_vec(slice.to_vec());
        Ok(array)
    }
}

/// Thread-safe memory pool for concurrent access
pub type SharedMemoryPool = Arc<MemoryPool>;

/// Create a shared memory pool
pub fn create_shared_pool(config: MemoryPoolConfig) -> SharedMemoryPool {
    Arc::new(MemoryPool::new(config))
}

/// Global memory pool instance
// TODO: Fix thread safety issues with NonNull<u8> in MemoryBlock
// lazy_static::lazy_static! {
//     static ref GLOBAL_POOL: SharedMemoryPool = create_shared_pool(MemoryPoolConfig::default());
// }

/// Get the global memory pool
// TODO: Temporarily disabled due to thread safety issues
// pub fn global_pool() -> &'static SharedMemoryPool {
//     &GLOBAL_POOL
// }

/// Convenience function to allocate from global pool
// TODO: Temporarily disabled due to thread safety issues
// pub fn allocate_global(size: usize) -> MemoryPoolResult<PooledBlock> {
//     global_pool().allocate(size)
// }

/// Convenience function to allocate 2D array from global pool
// TODO: Temporarily disabled due to thread safety issues
// pub fn allocate_array2_global(nrows: usize, ncols: usize) -> MemoryPoolResult<PooledArray2> {
//     global_pool().allocate_array2(nrows, ncols)
// }

/// Convenience function to allocate 1D array from global pool
// TODO: Temporarily disabled due to thread safety issues
// pub fn allocate_array1_global(len: usize) -> MemoryPoolResult<PooledArray1> {
//     global_pool().allocate_array1(len)
// }

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() -> MemoryPoolResult<()> {
        let pool = MemoryPool::new(MemoryPoolConfig::default());

        // Allocate a block
        let mut block = pool.allocate(1024)?;
        assert_eq!(block.size(), 1024);

        // Write some data
        let slice = block.as_slice_mut();
        slice[0] = 42;
        slice[1023] = 24;

        // Check data
        assert_eq!(slice[0], 42);
        assert_eq!(slice[1023], 24);

        Ok(())
    }

    #[test]
    fn test_memory_pool_reuse() -> MemoryPoolResult<()> {
        let config = MemoryPoolConfig {
            enable_reuse: true,
            ..Default::default()
        };
        let pool = MemoryPool::new(config);

        // Allocate and deallocate
        {
            let _block1 = pool.allocate(512)?;
        } // Block is returned to pool

        // Allocate again - should reuse
        let _block2 = pool.allocate(512)?;

        let stats = pool.stats();
        assert!(stats.blocks_reused > 0 || stats.pool_hits > 0);

        Ok(())
    }

    #[test]
    fn test_pooled_array2() -> MemoryPoolResult<()> {
        let pool = MemoryPool::new(MemoryPoolConfig::default());

        let mut array = pool.allocate_array2(100, 50)?;
        assert_eq!(array.dim(), (100, 50));

        // Get mutable view and fill with data
        {
            let mut view = array.view_mut()?;
            view.fill(3.14);
        }

        // Convert to owned array
        let owned = array.to_array()?;
        assert_eq!(owned.dim(), (100, 50));
        assert!((owned[[0, 0]] - 3.14).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_pooled_array1() -> MemoryPoolResult<()> {
        let pool = MemoryPool::new(MemoryPoolConfig::default());

        let mut array = pool.allocate_array1(1000)?;
        assert_eq!(array.len(), 1000);

        // Get mutable view and fill with data
        {
            let mut view = array.view_mut()?;
            view.fill(2.71);
        }

        // Convert to owned array
        let owned = array.to_array()?;
        assert_eq!(owned.len(), 1000);
        assert!((owned[0] - 2.71).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_memory_pool_stats() -> MemoryPoolResult<()> {
        let config = MemoryPoolConfig {
            enable_stats: true,
            ..Default::default()
        };
        let pool = MemoryPool::new(config);

        // Initial stats
        let initial_stats = pool.stats();
        assert_eq!(initial_stats.total_allocations, 0);

        // Allocate some blocks
        let _block1 = pool.allocate(256)?;
        let _block2 = pool.allocate(512)?;

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.current_allocations, 2);
        assert!(stats.current_bytes_allocated > 0);

        Ok(())
    }

    #[test]
    fn test_global_pool() -> MemoryPoolResult<()> {
        let mut block = allocate_global(128)?;
        assert_eq!(block.size(), 128);

        let slice = block.as_slice_mut();
        slice[0] = 255;
        assert_eq!(slice[0], 255);

        let mut array2 = allocate_array2_global(10, 20)?;
        assert_eq!(array2.dim(), (10, 20));

        let mut array1 = allocate_array1_global(100)?;
        assert_eq!(array1.len(), 100);

        Ok(())
    }

    #[test]
    fn test_pool_exhaustion() {
        let config = MemoryPoolConfig {
            max_blocks_per_size: 2,
            enable_reuse: false,
            ..Default::default()
        };
        let pool = MemoryPool::new(config);

        // Allocate up to the limit
        let _block1 = pool.allocate(64).unwrap();
        let _block2 = pool.allocate(64).unwrap();

        // This should fail
        assert!(pool.allocate(64).is_err());
    }

    #[test]
    fn test_cleanup() -> MemoryPoolResult<()> {
        let config = MemoryPoolConfig {
            max_idle_time: std::time::Duration::from_millis(1),
            enable_reuse: true,
            ..Default::default()
        };
        let pool = MemoryPool::new(config);

        // Allocate and immediately deallocate
        {
            let _block = pool.allocate(256)?;
        }

        // Wait for idle time to pass
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Force cleanup
        let removed = pool.cleanup();
        assert!(removed > 0);

        Ok(())
    }

    #[test]
    fn test_bucket_alignment() -> MemoryPoolResult<()> {
        let pool = MemoryPool::new(MemoryPoolConfig::default());

        // Different sizes should be bucketed to powers of 2
        let block1 = pool.allocate(100)?; // Should be rounded to 128
        let block2 = pool.allocate(200)?; // Should be rounded to 256

        assert!(block1.size() >= 100);
        assert!(block2.size() >= 200);

        // Size should be power of 2
        assert!(block1.size().is_power_of_two());
        assert!(block2.size().is_power_of_two());

        Ok(())
    }
}
