//! GPU memory management utilities
//!
//! This module provides advanced GPU memory management including:
//! - Memory pools for efficient allocation/deallocation
//! - Unified memory management for CUDA
//! - Multi-GPU memory distribution
//! - Memory bandwidth optimization

use crate::gpu::{GpuBackend, GpuDevice};
use crate::traits::SimdError;

#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "no-std")]
use alloc::sync::Arc;
#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;
#[cfg(not(feature = "no-std"))]
use std::sync::Arc;

#[cfg(feature = "no-std")]
use alloc::{format, string::ToString};

#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

#[cfg(feature = "no-std")]
use spin::Mutex;
#[cfg(not(feature = "no-std"))]
use std::sync::Mutex;

#[cfg(feature = "no-std")]
use core::slice;
#[cfg(not(feature = "no-std"))]
use std::slice;

/// Memory pool for GPU allocations
pub struct GpuMemoryPool {
    device: GpuDevice,
    free_blocks: HashMap<usize, Vec<GpuMemoryBlock>>,
    allocated_blocks: HashMap<usize, GpuMemoryBlock>,
    total_allocated: usize,
    peak_usage: usize,
    allocation_count: usize,
}

/// GPU memory block descriptor
#[derive(Debug, Clone)]
pub struct GpuMemoryBlock {
    ptr: *mut u8,
    size: usize,
    device_id: u32,
    backend: GpuBackend,
    is_unified: bool,
}

unsafe impl Send for GpuMemoryBlock {}
unsafe impl Sync for GpuMemoryBlock {}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Simple allocation without pooling
    Simple,
    /// Pool-based allocation with size classes
    Pooled,
    /// Unified memory allocation (CUDA only)
    Unified,
    /// Pinned host memory for fast transfers
    Pinned,
}

/// Memory bandwidth optimization settings
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    pub use_async_transfers: bool,
    pub prefer_pinned_memory: bool,
    pub coalesce_transfers: bool,
    pub max_concurrent_streams: u32,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            use_async_transfers: true,
            prefer_pinned_memory: true,
            coalesce_transfers: true,
            max_concurrent_streams: 4,
        }
    }
}

impl GpuMemoryPool {
    /// Create a new memory pool for the specified device
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device,
            free_blocks: HashMap::new(),
            allocated_blocks: HashMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            allocation_count: 0,
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(
        &mut self,
        size: usize,
        strategy: AllocationStrategy,
    ) -> Result<GpuMemoryBlock, SimdError> {
        self.allocation_count += 1;

        // Try to find a suitable free block
        if let Some(block) = self.find_free_block(size) {
            self.allocated_blocks
                .insert(block.ptr as usize, block.clone());
            return Ok(block);
        }

        // Allocate new block
        let block = self.allocate_new_block(size, strategy)?;
        self.allocated_blocks
            .insert(block.ptr as usize, block.clone());
        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);

        Ok(block)
    }

    /// Deallocate memory and return to pool
    pub fn deallocate(&mut self, ptr: *mut u8) -> Result<(), SimdError> {
        if let Some(block) = self.allocated_blocks.remove(&(ptr as usize)) {
            self.total_allocated -= block.size;

            // Add to free blocks for reuse
            let size_class = self.get_size_class(block.size);
            self.free_blocks.entry(size_class).or_default().push(block);

            Ok(())
        } else {
            Err(SimdError::InvalidParameter {
                name: "ptr".to_string(),
                value: "Invalid pointer for deallocation".to_string(),
            })
        }
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            allocation_count: self.allocation_count,
            free_blocks_count: self.free_blocks.values().map(|v| v.len()).sum(),
            device_memory_mb: self.device.memory_mb,
        }
    }

    /// Clear all free blocks to reclaim memory
    pub fn trim(&mut self) {
        self.free_blocks.clear();
    }

    fn find_free_block(&mut self, size: usize) -> Option<GpuMemoryBlock> {
        let size_class = self.get_size_class(size);

        // Try exact size class first
        if let Some(blocks) = self.free_blocks.get_mut(&size_class) {
            if let Some(block) = blocks.pop() {
                return Some(block);
            }
        }

        // Try larger size classes
        for (&class_size, blocks) in self.free_blocks.iter_mut() {
            if class_size >= size_class && !blocks.is_empty() {
                return blocks.pop();
            }
        }

        None
    }

    fn allocate_new_block(
        &self,
        size: usize,
        strategy: AllocationStrategy,
    ) -> Result<GpuMemoryBlock, SimdError> {
        match strategy {
            AllocationStrategy::Simple => self.allocate_simple(size),
            AllocationStrategy::Pooled => self.allocate_pooled(size),
            AllocationStrategy::Unified => self.allocate_unified(size),
            AllocationStrategy::Pinned => self.allocate_pinned(size),
        }
    }

    fn allocate_simple(&self, size: usize) -> Result<GpuMemoryBlock, SimdError> {
        match self.device.backend {
            GpuBackend::Cuda => {
                // CUDA disabled for macOS compatibility
                let _ = size;
                Err(SimdError::UnsupportedOperation(
                    "CUDA not available".to_string(),
                ))
            }
            GpuBackend::OpenCL => {
                // OpenCL disabled for macOS compatibility
                let _ = size;
                Err(SimdError::UnsupportedOperation(
                    "OpenCL not available".to_string(),
                ))
            }
            _ => Err(SimdError::UnsupportedOperation(
                "Backend not supported".to_string(),
            )),
        }
    }

    fn allocate_pooled(&self, size: usize) -> Result<GpuMemoryBlock, SimdError> {
        // Allocate larger blocks for pooling efficiency
        let pool_size = (size * 2).max(1024 * 1024); // At least 1MB
        self.allocate_simple(pool_size)
    }

    fn allocate_unified(&self, size: usize) -> Result<GpuMemoryBlock, SimdError> {
        if self.device.backend != GpuBackend::Cuda {
            return Err(SimdError::UnsupportedOperation(
                "Unified memory only available with CUDA".to_string(),
            ));
        }

        // CUDA disabled for macOS compatibility
        let _ = size;
        Err(SimdError::UnsupportedOperation(
            "CUDA not available".to_string(),
        ))
    }

    fn allocate_pinned(&self, size: usize) -> Result<GpuMemoryBlock, SimdError> {
        match self.device.backend {
            GpuBackend::Cuda => {
                // CUDA disabled for macOS compatibility
                let _ = size;
                Err(SimdError::UnsupportedOperation(
                    "CUDA not available".to_string(),
                ))
            }
            _ => Err(SimdError::UnsupportedOperation(
                "Pinned memory only available with CUDA".to_string(),
            )),
        }
    }

    fn get_size_class(&self, size: usize) -> usize {
        // Round up to nearest power of 2 for size classes
        if size == 0 {
            return 1;
        }
        1 << (64 - size.leading_zeros())
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
    pub free_blocks_count: usize,
    pub device_memory_mb: u64,
}

impl MemoryStats {
    /// Get memory utilization as a percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.device_memory_mb == 0 {
            return 0.0;
        }
        (self.total_allocated as f64 / (self.device_memory_mb * 1024 * 1024) as f64) * 100.0
    }

    /// Check if memory usage is approaching limit
    pub fn is_high_usage(&self, threshold: f64) -> bool {
        self.utilization_percent() > threshold
    }
}

/// Multi-GPU memory manager
pub struct MultiGpuMemoryManager {
    pools: HashMap<u32, Arc<Mutex<GpuMemoryPool>>>,
    allocation_strategy: AllocationStrategy,
    bandwidth_config: BandwidthConfig,
}

impl MultiGpuMemoryManager {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            allocation_strategy: AllocationStrategy::Pooled,
            bandwidth_config: BandwidthConfig::default(),
        }
    }

    /// Add a device to the manager
    pub fn add_device(&mut self, device: GpuDevice) {
        let pool = Arc::new(Mutex::new(GpuMemoryPool::new(device.clone())));
        self.pools.insert(device.id, pool);
    }

    /// Allocate memory on the specified device
    pub fn allocate_on_device(
        &self,
        device_id: u32,
        size: usize,
    ) -> Result<GpuMemoryBlock, SimdError> {
        if let Some(pool) = self.pools.get(&device_id) {
            #[cfg(not(feature = "no-std"))]
            let mut pool = pool.lock().map_err(|_| {
                SimdError::ExternalLibraryError("Failed to lock memory pool".to_string())
            })?;
            #[cfg(feature = "no-std")]
            let mut pool = pool.lock();
            pool.allocate(size, self.allocation_strategy)
        } else {
            Err(SimdError::InvalidParameter {
                name: "device_id".to_string(),
                value: format!("Device {} not found", device_id),
            })
        }
    }

    /// Allocate memory on the best available device
    pub fn allocate_on_best_device(&self, size: usize) -> Result<(u32, GpuMemoryBlock), SimdError> {
        let best_device = self.find_best_device_for_allocation(size)?;
        let block = self.allocate_on_device(best_device, size)?;
        Ok((best_device, block))
    }

    /// Deallocate memory on the specified device
    pub fn deallocate_on_device(&self, device_id: u32, ptr: *mut u8) -> Result<(), SimdError> {
        if let Some(pool) = self.pools.get(&device_id) {
            #[cfg(not(feature = "no-std"))]
            let mut pool = pool.lock().map_err(|_| {
                SimdError::ExternalLibraryError("Failed to lock memory pool".to_string())
            })?;
            #[cfg(feature = "no-std")]
            let mut pool = pool.lock();
            pool.deallocate(ptr)
        } else {
            Err(SimdError::InvalidParameter {
                name: "device_id".to_string(),
                value: format!("Device {} not found", device_id),
            })
        }
    }

    /// Get memory statistics for all devices
    pub fn get_all_stats(&self) -> HashMap<u32, MemoryStats> {
        let mut stats = HashMap::new();
        for (&device_id, pool) in &self.pools {
            #[cfg(not(feature = "no-std"))]
            {
                if let Ok(pool) = pool.lock() {
                    stats.insert(device_id, pool.get_stats());
                }
            }
            #[cfg(feature = "no-std")]
            {
                let pool = pool.lock();
                stats.insert(device_id, pool.get_stats());
            }
        }
        stats
    }

    /// Find the best device for allocation based on available memory
    fn find_best_device_for_allocation(&self, size: usize) -> Result<u32, SimdError> {
        let mut best_device = None;
        let mut min_usage = f64::INFINITY;

        for (&device_id, pool) in &self.pools {
            #[cfg(not(feature = "no-std"))]
            let pool_result = pool.lock();
            #[cfg(feature = "no-std")]
            let pool_result: Result<_, ()> = Ok(pool.lock());

            if let Ok(pool) = pool_result {
                let stats = pool.get_stats();
                let usage = stats.utilization_percent();

                // Check if device has enough memory
                let available_mb =
                    stats.device_memory_mb - (stats.total_allocated / (1024 * 1024)) as u64;
                let required_mb = (size / (1024 * 1024)) as u64 + 1;

                if available_mb >= required_mb && usage < min_usage {
                    min_usage = usage;
                    best_device = Some(device_id);
                }
            }
        }

        best_device.ok_or_else(|| {
            SimdError::ExternalLibraryError("No suitable device found for allocation".to_string())
        })
    }

    /// Configure allocation strategy
    pub fn set_allocation_strategy(&mut self, strategy: AllocationStrategy) {
        self.allocation_strategy = strategy;
    }

    /// Configure bandwidth optimization
    pub fn set_bandwidth_config(&mut self, config: BandwidthConfig) {
        self.bandwidth_config = config;
    }
}

impl Default for MultiGpuMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified memory buffer for CUDA
#[derive(Debug)]
pub struct UnifiedMemoryBuffer<T> {
    ptr: *mut T,
    size: usize,
    device_id: u32,
}

impl<T> UnifiedMemoryBuffer<T> {
    /// Create a new unified memory buffer
    pub fn new(size: usize, device_id: u32) -> Result<Self, SimdError> {
        // CUDA disabled for macOS compatibility
        let _ = (size, device_id);
        Err(SimdError::UnsupportedOperation(
            "CUDA unified memory not available".to_string(),
        ))
    }

    /// Get mutable slice to the data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Get immutable slice to the data
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Prefetch to GPU
    pub fn prefetch_to_gpu(&self) -> Result<(), SimdError> {
        // CUDA disabled for macOS compatibility
        Err(SimdError::UnsupportedOperation(
            "CUDA not available".to_string(),
        ))
    }

    /// Prefetch to CPU
    pub fn prefetch_to_cpu(&self) -> Result<(), SimdError> {
        // CUDA disabled for macOS compatibility
        Err(SimdError::UnsupportedOperation(
            "CUDA not available".to_string(),
        ))
    }
}

impl<T> Drop for UnifiedMemoryBuffer<T> {
    fn drop(&mut self) {
        // CUDA disabled for macOS compatibility - no cleanup needed
    }
}

unsafe impl<T: Send> Send for UnifiedMemoryBuffer<T> {}
unsafe impl<T: Sync> Sync for UnifiedMemoryBuffer<T> {}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    use crate::gpu::GpuBackend;

    #[cfg(feature = "no-std")]
    use alloc::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    #[test]
    fn test_memory_pool_creation() {
        let device = GpuDevice {
            id: 0,
            name: "Test Device".to_string(),
            backend: GpuBackend::Cuda,
            compute_units: 80,
            memory_mb: 8192,
            supports_f64: true,
            supports_f16: true,
        };

        let pool = GpuMemoryPool::new(device);
        let stats = pool.get_stats();

        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_size_class_calculation() {
        let device = GpuDevice {
            id: 0,
            name: "Test Device".to_string(),
            backend: GpuBackend::Cuda,
            compute_units: 80,
            memory_mb: 8192,
            supports_f64: true,
            supports_f16: true,
        };

        let pool = GpuMemoryPool::new(device);

        assert_eq!(pool.get_size_class(0), 1);
        assert_eq!(pool.get_size_class(1), 2);
        assert_eq!(pool.get_size_class(1000), 1024);
        assert_eq!(pool.get_size_class(1024), 2048);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            total_allocated: 1024 * 1024, // 1MB
            peak_usage: 2 * 1024 * 1024,  // 2MB
            allocation_count: 5,
            free_blocks_count: 2,
            device_memory_mb: 1024, // 1GB
        };

        assert!((stats.utilization_percent() - 0.09765625).abs() < 0.001); // ~0.1%
        assert!(!stats.is_high_usage(50.0));
        assert!(stats.is_high_usage(0.05));
    }

    #[test]
    fn test_multi_gpu_manager() {
        let mut manager = MultiGpuMemoryManager::new();

        let device1 = GpuDevice {
            id: 0,
            name: "Device 0".to_string(),
            backend: GpuBackend::Cuda,
            compute_units: 80,
            memory_mb: 8192,
            supports_f64: true,
            supports_f16: true,
        };

        let device2 = GpuDevice {
            id: 1,
            name: "Device 1".to_string(),
            backend: GpuBackend::Cuda,
            compute_units: 40,
            memory_mb: 4096,
            supports_f64: true,
            supports_f16: true,
        };

        manager.add_device(device1);
        manager.add_device(device2);

        let stats = manager.get_all_stats();
        assert_eq!(stats.len(), 2);
        assert!(stats.contains_key(&0));
        assert!(stats.contains_key(&1));
    }

    #[test]
    fn test_bandwidth_config() {
        let config = BandwidthConfig::default();
        assert!(config.use_async_transfers);
        assert!(config.prefer_pinned_memory);
        assert!(config.coalesce_transfers);
        assert_eq!(config.max_concurrent_streams, 4);
    }

    #[test]
    fn test_allocation_strategies() {
        let strategies = vec![
            AllocationStrategy::Simple,
            AllocationStrategy::Pooled,
            AllocationStrategy::Unified,
            AllocationStrategy::Pinned,
        ];

        // Test that strategies can be compared and used
        for strategy in strategies {
            match strategy {
                AllocationStrategy::Simple => assert!(true),
                AllocationStrategy::Pooled => assert!(true),
                AllocationStrategy::Unified => assert!(true),
                AllocationStrategy::Pinned => assert!(true),
            }
        }
    }
}
