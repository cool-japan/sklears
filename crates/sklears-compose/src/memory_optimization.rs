//! Memory optimization and monitoring for pipeline execution
//!
//! This module provides memory-efficient pipeline execution, memory monitoring,
//! garbage collection optimization, and memory pool management.

use scirs2_core::ndarray::{s, Array2, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::alloc::{self, Layout};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::mem;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

/// Memory usage tracking
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Current allocated memory in bytes
    pub allocated: u64,
    /// Peak memory usage in bytes
    pub peak: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Number of deallocations
    pub deallocations: u64,
    /// Memory fragmentation ratio
    pub fragmentation: f64,
    /// Last update timestamp
    pub updated_at: SystemTime,
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self {
            allocated: 0,
            peak: 0,
            allocations: 0,
            deallocations: 0,
            fragmentation: 0.0,
            updated_at: SystemTime::now(),
        }
    }
}

impl MemoryUsage {
    /// Update memory statistics
    pub fn update(&mut self, allocated: u64, allocations: u64, deallocations: u64) {
        self.allocated = allocated;
        self.allocations = allocations;
        self.deallocations = deallocations;

        if allocated > self.peak {
            self.peak = allocated;
        }

        // Simple fragmentation calculation
        if allocations > 0 {
            self.fragmentation = (allocations - deallocations) as f64 / allocations as f64;
        }

        self.updated_at = SystemTime::now();
    }

    /// Get current utilization ratio (0.0 - 1.0)
    #[must_use]
    pub fn utilization(&self, total_available: u64) -> f64 {
        if total_available == 0 {
            0.0
        } else {
            self.allocated as f64 / total_available as f64
        }
    }

    /// Check if memory usage is critical
    #[must_use]
    pub fn is_critical(&self, threshold: f64, total_available: u64) -> bool {
        self.utilization(total_available) > threshold
    }
}

/// Memory monitor for tracking system memory usage
pub struct MemoryMonitor {
    /// Current memory usage
    usage: Arc<RwLock<MemoryUsage>>,
    /// Monitoring configuration
    config: MemoryMonitorConfig,
    /// Usage history
    history: Arc<RwLock<VecDeque<MemoryUsage>>>,
    /// Monitoring thread handle
    monitor_thread: Option<JoinHandle<()>>,
    /// Running flag
    is_running: Arc<Mutex<bool>>,
    /// Callbacks for memory events
    callbacks: Arc<RwLock<Vec<Box<dyn Fn(&MemoryUsage) + Send + Sync>>>>,
}

impl std::fmt::Debug for MemoryMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryMonitor")
            .field("usage", &self.usage)
            .field("config", &self.config)
            .field("history", &self.history)
            .field("monitor_thread", &self.monitor_thread.is_some())
            .field("is_running", &self.is_running)
            .field(
                "callbacks",
                &format!("{} callbacks", self.callbacks.read().unwrap().len()),
            )
            .finish()
    }
}

/// Memory monitoring configuration
#[derive(Debug, Clone)]
pub struct MemoryMonitorConfig {
    /// Monitoring interval
    pub interval: Duration,
    /// Warning threshold (0.0 - 1.0)
    pub warning_threshold: f64,
    /// Critical threshold (0.0 - 1.0)
    pub critical_threshold: f64,
    /// Maximum history entries
    pub max_history: usize,
    /// Enable automatic garbage collection
    pub auto_gc: bool,
    /// GC trigger threshold
    pub gc_threshold: f64,
}

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            warning_threshold: 0.7,
            critical_threshold: 0.9,
            max_history: 3600, // 1 hour at 1-second intervals
            auto_gc: true,
            gc_threshold: 0.8,
        }
    }
}

impl MemoryMonitor {
    /// Create a new memory monitor
    #[must_use]
    pub fn new(config: MemoryMonitorConfig) -> Self {
        Self {
            usage: Arc::new(RwLock::new(MemoryUsage::default())),
            config,
            history: Arc::new(RwLock::new(VecDeque::new())),
            monitor_thread: None,
            is_running: Arc::new(Mutex::new(false)),
            callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start monitoring
    pub fn start(&mut self) -> SklResult<()> {
        {
            let mut running = self.is_running.lock().unwrap();
            if *running {
                return Ok(());
            }
            *running = true;
        }

        let usage = Arc::clone(&self.usage);
        let history = Arc::clone(&self.history);
        let callbacks = Arc::clone(&self.callbacks);
        let is_running = Arc::clone(&self.is_running);
        let config = self.config.clone();

        let handle = thread::spawn(move || {
            Self::monitor_loop(usage, history, callbacks, is_running, config);
        });

        self.monitor_thread = Some(handle);
        Ok(())
    }

    /// Stop monitoring
    pub fn stop(&mut self) -> SklResult<()> {
        {
            let mut running = self.is_running.lock().unwrap();
            *running = false;
        }

        if let Some(handle) = self.monitor_thread.take() {
            handle.join().map_err(|_| SklearsError::InvalidData {
                reason: "Failed to join monitor thread".to_string(),
            })?;
        }

        Ok(())
    }

    /// Main monitoring loop
    fn monitor_loop(
        usage: Arc<RwLock<MemoryUsage>>,
        history: Arc<RwLock<VecDeque<MemoryUsage>>>,
        callbacks: Arc<RwLock<Vec<Box<dyn Fn(&MemoryUsage) + Send + Sync>>>>,
        is_running: Arc<Mutex<bool>>,
        config: MemoryMonitorConfig,
    ) {
        while *is_running.lock().unwrap() {
            // Get current system memory usage
            let (allocated, allocations, deallocations) = Self::get_system_memory_info();

            // Update usage statistics
            {
                let mut current_usage = usage.write().unwrap();
                current_usage.update(allocated, allocations, deallocations);

                // Add to history
                {
                    let mut hist = history.write().unwrap();
                    hist.push_back(current_usage.clone());

                    // Limit history size
                    while hist.len() > config.max_history {
                        hist.pop_front();
                    }
                }

                // Check thresholds and trigger callbacks
                let total_memory = Self::get_total_system_memory();
                let utilization = current_usage.utilization(total_memory);

                if config.auto_gc && utilization > config.gc_threshold {
                    Self::trigger_garbage_collection();
                }

                // Notify callbacks
                let cb_list = callbacks.read().unwrap();
                for callback in cb_list.iter() {
                    callback(&current_usage);
                }
            }

            thread::sleep(config.interval);
        }
    }

    /// Get system memory information (simplified implementation)
    fn get_system_memory_info() -> (u64, u64, u64) {
        // In a real implementation, this would use platform-specific APIs
        // For now, return dummy values
        (1024 * 1024 * 100, 1000, 900) // 100MB allocated, 1000 allocs, 900 deallocs
    }

    /// Get total system memory (simplified implementation)
    fn get_total_system_memory() -> u64 {
        // In a real implementation, this would query system memory
        1024 * 1024 * 1024 * 8 // 8GB
    }

    /// Trigger garbage collection
    fn trigger_garbage_collection() {
        // Force garbage collection (simplified)
        // In Rust, this might involve dropping unused data structures
        // or calling custom cleanup functions
    }

    /// Get current memory usage
    #[must_use]
    pub fn current_usage(&self) -> MemoryUsage {
        let usage = self.usage.read().unwrap();
        usage.clone()
    }

    /// Get memory usage history
    #[must_use]
    pub fn usage_history(&self) -> Vec<MemoryUsage> {
        let history = self.history.read().unwrap();
        history.iter().cloned().collect()
    }

    /// Add memory event callback
    pub fn add_callback(&self, callback: Box<dyn Fn(&MemoryUsage) + Send + Sync>) {
        let mut callbacks = self.callbacks.write().unwrap();
        callbacks.push(callback);
    }

    /// Check if memory usage is above threshold
    #[must_use]
    pub fn is_above_threshold(&self, threshold: f64) -> bool {
        let usage = self.usage.read().unwrap();
        let total = Self::get_total_system_memory();
        usage.utilization(total) > threshold
    }

    /// Get memory statistics summary
    #[must_use]
    pub fn get_statistics(&self) -> MemoryStatistics {
        let usage = self.usage.read().unwrap();
        let history = self.history.read().unwrap();

        let avg_allocated = if history.is_empty() {
            usage.allocated
        } else {
            history.iter().map(|u| u.allocated).sum::<u64>() / history.len() as u64
        };

        let max_allocated = history
            .iter()
            .map(|u| u.allocated)
            .max()
            .unwrap_or(usage.allocated);
        let min_allocated = history
            .iter()
            .map(|u| u.allocated)
            .min()
            .unwrap_or(usage.allocated);

        MemoryStatistics {
            current: usage.clone(),
            average_allocated: avg_allocated,
            max_allocated,
            min_allocated,
            total_system_memory: Self::get_total_system_memory(),
            samples_count: history.len(),
        }
    }
}

/// Memory statistics summary
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Current memory usage
    pub current: MemoryUsage,
    /// Average allocated memory
    pub average_allocated: u64,
    /// Maximum allocated memory
    pub max_allocated: u64,
    /// Minimum allocated memory
    pub min_allocated: u64,
    /// Total system memory
    pub total_system_memory: u64,
    /// Number of samples
    pub samples_count: usize,
}

/// Memory pool for efficient allocation and reuse
#[derive(Debug)]
pub struct MemoryPool {
    config: MemoryPoolConfig,
    available_blocks: Arc<RwLock<BTreeMap<usize, Vec<MemoryBlock>>>>,
    allocated_blocks: Arc<RwLock<HashMap<*mut u8, MemoryBlock>>>,
    statistics: Arc<RwLock<PoolStatistics>>,
    monitor: Option<MemoryMonitor>,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,
    /// Maximum pool size in bytes
    pub max_size: usize,
    /// Block size classes
    pub size_classes: Vec<usize>,
    /// Enable automatic expansion
    pub auto_expand: bool,
    /// Expansion factor
    pub expansion_factor: f64,
    /// Enable memory compaction
    pub compaction_enabled: bool,
    /// Compaction threshold
    pub compaction_threshold: f64,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024 * 10, // 10MB
            max_size: 1024 * 1024 * 100,    // 100MB
            size_classes: vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            auto_expand: true,
            expansion_factor: 1.5,
            compaction_enabled: true,
            compaction_threshold: 0.7,
        }
    }
}

/// Memory block in the pool
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block pointer
    pub ptr: *mut u8,
    /// Block size
    pub size: usize,
    /// Allocation timestamp
    pub allocated_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Reference count
    pub ref_count: usize,
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    /// Total allocated bytes
    pub total_allocated: usize,
    /// Total available bytes
    pub total_available: usize,
    /// Number of allocations
    pub allocations: u64,
    /// Number of deallocations
    pub deallocations: u64,
    /// Cache hit rate
    pub hit_rate: f64,
    /// Fragmentation ratio
    pub fragmentation: f64,
    /// Pool utilization
    pub utilization: f64,
}

impl Default for PoolStatistics {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            total_available: 0,
            allocations: 0,
            deallocations: 0,
            hit_rate: 0.0,
            fragmentation: 0.0,
            utilization: 0.0,
        }
    }
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(config: MemoryPoolConfig) -> SklResult<Self> {
        let mut pool = Self {
            config,
            available_blocks: Arc::new(RwLock::new(BTreeMap::new())),
            allocated_blocks: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(PoolStatistics::default())),
            monitor: None,
        };

        // Initialize pool with initial blocks
        pool.initialize_pool()?;

        Ok(pool)
    }

    /// Initialize the memory pool
    fn initialize_pool(&mut self) -> SklResult<()> {
        let mut available = self.available_blocks.write().unwrap();

        for &size_class in &self.config.size_classes {
            let blocks_per_class =
                self.config.initial_size / (size_class * self.config.size_classes.len());
            let mut blocks = Vec::with_capacity(blocks_per_class);

            for _ in 0..blocks_per_class {
                let layout = Layout::from_size_align(size_class, std::mem::align_of::<u8>())
                    .map_err(|_| SklearsError::InvalidData {
                        reason: "Invalid memory layout".to_string(),
                    })?;

                unsafe {
                    let ptr = alloc::alloc(layout);
                    if ptr.is_null() {
                        return Err(SklearsError::InvalidData {
                            reason: "Memory allocation failed".to_string(),
                        });
                    }

                    blocks.push(MemoryBlock {
                        ptr,
                        size: size_class,
                        allocated_at: SystemTime::now(),
                        last_accessed: SystemTime::now(),
                        ref_count: 0,
                    });
                }
            }

            available.insert(size_class, blocks);
        }

        Ok(())
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> SklResult<*mut u8> {
        let size_class = self.find_size_class(size);
        let mut available = self.available_blocks.write().unwrap();
        let mut allocated = self.allocated_blocks.write().unwrap();
        let mut stats = self.statistics.write().unwrap();

        if let Some(blocks) = available.get_mut(&size_class) {
            if let Some(mut block) = blocks.pop() {
                // Found available block
                block.allocated_at = SystemTime::now();
                block.last_accessed = SystemTime::now();
                block.ref_count = 1;

                let ptr = block.ptr;
                allocated.insert(ptr, block);

                stats.allocations += 1;
                stats.total_allocated += size_class;
                stats.hit_rate = stats.allocations as f64 / (stats.allocations + 1) as f64;

                return Ok(ptr);
            }
        }

        // No available block, allocate new one if auto-expand is enabled
        if self.config.auto_expand {
            let layout =
                Layout::from_size_align(size_class, std::mem::align_of::<u8>()).map_err(|_| {
                    SklearsError::InvalidData {
                        reason: "Invalid memory layout".to_string(),
                    }
                })?;

            unsafe {
                let ptr = alloc::alloc(layout);
                if ptr.is_null() {
                    return Err(SklearsError::InvalidData {
                        reason: "Memory allocation failed".to_string(),
                    });
                }

                let block = MemoryBlock {
                    ptr,
                    size: size_class,
                    allocated_at: SystemTime::now(),
                    last_accessed: SystemTime::now(),
                    ref_count: 1,
                };

                allocated.insert(ptr, block);
                stats.allocations += 1;
                stats.total_allocated += size_class;

                Ok(ptr)
            }
        } else {
            Err(SklearsError::InvalidData {
                reason: "Memory pool exhausted".to_string(),
            })
        }
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&self, ptr: *mut u8) -> SklResult<()> {
        let mut available = self.available_blocks.write().unwrap();
        let mut allocated = self.allocated_blocks.write().unwrap();
        let mut stats = self.statistics.write().unwrap();

        if let Some(mut block) = allocated.remove(&ptr) {
            block.ref_count = 0;
            block.last_accessed = SystemTime::now();

            let size_class = block.size;
            available.entry(size_class).or_default().push(block);

            stats.deallocations += 1;
            stats.total_allocated = stats.total_allocated.saturating_sub(size_class);

            Ok(())
        } else {
            Err(SklearsError::InvalidData {
                reason: "Invalid pointer for deallocation".to_string(),
            })
        }
    }

    /// Find appropriate size class for allocation
    fn find_size_class(&self, size: usize) -> usize {
        self.config
            .size_classes
            .iter()
            .find(|&&class_size| class_size >= size)
            .copied()
            .unwrap_or_else(|| {
                // Round up to next power of 2
                let mut class_size = 1;
                while class_size < size {
                    class_size <<= 1;
                }
                class_size
            })
    }

    /// Compact the memory pool
    pub fn compact(&self) -> SklResult<()> {
        let available = self.available_blocks.write().unwrap();
        let mut stats = self.statistics.write().unwrap();

        let total_blocks: usize = available.values().map(std::vec::Vec::len).sum();
        let fragmentation = if total_blocks > 0 {
            1.0 - (available.len() as f64 / total_blocks as f64)
        } else {
            0.0
        };

        if fragmentation > self.config.compaction_threshold {
            // Perform compaction (simplified)
            // In a real implementation, this would reorganize memory blocks
            stats.fragmentation = fragmentation;
        }

        Ok(())
    }

    /// Get pool statistics
    #[must_use]
    pub fn statistics(&self) -> PoolStatistics {
        let stats = self.statistics.read().unwrap();
        stats.clone()
    }

    /// Enable memory monitoring
    pub fn enable_monitoring(&mut self, config: MemoryMonitorConfig) -> SklResult<()> {
        let mut monitor = MemoryMonitor::new(config);
        monitor.start()?;
        self.monitor = Some(monitor);
        Ok(())
    }

    /// Clear unused blocks (garbage collection)
    pub fn garbage_collect(&self) -> SklResult<usize> {
        let mut available = self.available_blocks.write().unwrap();
        let mut freed_blocks = 0;

        for (_, blocks) in available.iter_mut() {
            let old_len = blocks.len();

            // Keep only recently accessed blocks
            let cutoff = SystemTime::now() - Duration::from_secs(300); // 5 minutes
            blocks.retain(|block| block.last_accessed > cutoff);

            freed_blocks += old_len - blocks.len();
        }

        Ok(freed_blocks)
    }
}

/// Memory-efficient data structure for streaming data
#[derive(Debug)]
pub struct StreamingBuffer<T> {
    /// Ring buffer for data
    buffer: Vec<Option<T>>,
    /// Buffer capacity
    capacity: usize,
    /// Current write position
    write_pos: usize,
    /// Current read position
    read_pos: usize,
    /// Number of elements in buffer
    count: usize,
    /// Memory pool for allocations
    memory_pool: Option<Arc<MemoryPool>>,
}

impl<T> StreamingBuffer<T> {
    /// Create a new streaming buffer
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }
        Self {
            buffer,
            capacity,
            write_pos: 0,
            read_pos: 0,
            count: 0,
            memory_pool: None,
        }
    }

    /// Create streaming buffer with memory pool
    #[must_use]
    pub fn with_memory_pool(capacity: usize, memory_pool: Arc<MemoryPool>) -> Self {
        let mut buffer = Self::new(capacity);
        buffer.memory_pool = Some(memory_pool);
        buffer
    }

    /// Push an element to the buffer
    pub fn push(&mut self, item: T) -> Option<T> {
        let old_item = self.buffer[self.write_pos].take();
        self.buffer[self.write_pos] = Some(item);

        self.write_pos = (self.write_pos + 1) % self.capacity;

        if self.count < self.capacity {
            self.count += 1;
        } else {
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }

        old_item
    }

    /// Pop an element from the buffer
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }

        let item = self.buffer[self.read_pos].take();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.count -= 1;

        item
    }

    /// Get current buffer size
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if buffer is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.write_pos = 0;
        self.read_pos = 0;
        self.count = 0;
    }

    /// Get memory usage of the buffer
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.capacity * mem::size_of::<Option<T>>()
    }
}

/// Memory-efficient array operations
pub struct MemoryEfficientOps;

impl MemoryEfficientOps {
    /// In-place array transformation to reduce memory allocations
    pub fn transform_inplace<F>(array: &mut Array2<f64>, transform_fn: F)
    where
        F: Fn(f64) -> f64,
    {
        array.mapv_inplace(transform_fn);
    }

    /// Batch processing with controlled memory usage
    pub fn batch_process<F, R>(
        data: &Array2<f64>,
        batch_size: usize,
        process_fn: F,
    ) -> SklResult<Vec<R>>
    where
        F: Fn(ArrayView2<f64>) -> SklResult<R>,
    {
        let mut results = Vec::new();
        let n_rows = data.nrows();

        for chunk_start in (0..n_rows).step_by(batch_size) {
            let chunk_end = std::cmp::min(chunk_start + batch_size, n_rows);
            let batch = data.slice(s![chunk_start..chunk_end, ..]);

            let result = process_fn(batch)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Memory-efficient matrix multiplication using chunking
    pub fn chunked_matmul(
        a: &Array2<f64>,
        b: &Array2<f64>,
        chunk_size: usize,
    ) -> SklResult<Array2<f64>> {
        if a.ncols() != b.nrows() {
            return Err(SklearsError::InvalidData {
                reason: "Matrix dimensions don't match for multiplication".to_string(),
            });
        }

        let mut result = Array2::zeros((a.nrows(), b.ncols()));

        for i_chunk in (0..a.nrows()).step_by(chunk_size) {
            let i_end = std::cmp::min(i_chunk + chunk_size, a.nrows());

            for j_chunk in (0..b.ncols()).step_by(chunk_size) {
                let j_end = std::cmp::min(j_chunk + chunk_size, b.ncols());

                for k_chunk in (0..a.ncols()).step_by(chunk_size) {
                    let k_end = std::cmp::min(k_chunk + chunk_size, a.ncols());

                    let a_chunk = a.slice(s![i_chunk..i_end, k_chunk..k_end]);
                    let b_chunk = b.slice(s![k_chunk..k_end, j_chunk..j_end]);

                    let mut result_chunk = result.slice_mut(s![i_chunk..i_end, j_chunk..j_end]);

                    // Perform chunk multiplication
                    for (i, a_row) in a_chunk.rows().into_iter().enumerate() {
                        for (j, b_col) in b_chunk.columns().into_iter().enumerate() {
                            result_chunk[[i, j]] += a_row.dot(&b_col);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Reduce memory footprint by using lower precision when possible
    #[must_use]
    pub fn optimize_precision(array: &Array2<f64>, tolerance: f64) -> Array2<f32> {
        array.mapv(|x| {
            if x.abs() < tolerance {
                0.0f32
            } else {
                x as f32
            }
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_usage() {
        let mut usage = MemoryUsage::default();
        usage.update(1024, 10, 5);

        assert_eq!(usage.allocated, 1024);
        assert_eq!(usage.allocations, 10);
        assert_eq!(usage.deallocations, 5);
        assert_eq!(usage.peak, 1024);
    }

    #[test]
    fn test_memory_monitor_creation() {
        let config = MemoryMonitorConfig::default();
        let monitor = MemoryMonitor::new(config);

        let usage = monitor.current_usage();
        assert_eq!(usage.allocated, 0);
    }

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config).unwrap();

        let stats = pool.statistics();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.deallocations, 0);
    }

    #[test]
    fn test_streaming_buffer() {
        let mut buffer = StreamingBuffer::new(3);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 3);

        let old_item = buffer.push(4); // Should evict 1
        assert_eq!(old_item, Some(1));

        let popped = buffer.pop();
        assert_eq!(popped, Some(2));
    }

    #[test]
    fn test_memory_efficient_ops() {
        let mut array = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Test in-place transformation
        MemoryEfficientOps::transform_inplace(&mut array, |x| x * 2.0);
        assert_eq!(array[[0, 0]], 2.0);
        assert_eq!(array[[1, 1]], 8.0);

        // Test precision optimization
        let array_f64 = Array2::from_shape_vec((2, 2), vec![1.0, 0.000001, 3.0, 0.000002]).unwrap();
        let array_f32 = MemoryEfficientOps::optimize_precision(&array_f64, 0.00001);
        assert_eq!(array_f32[[0, 1]], 0.0f32); // Small value should be zeroed
        assert_eq!(array_f32[[1, 0]], 3.0f32); // Large value should be preserved
    }

    #[test]
    fn test_batch_processing() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let results = MemoryEfficientOps::batch_process(&data, 2, |batch| Ok(batch.sum())).unwrap();

        assert_eq!(results.len(), 2); // 4 rows / 2 batch_size = 2 batches
        assert_eq!(results[0], 10.0); // Sum of first batch: 1+2+3+4
        assert_eq!(results[1], 26.0); // Sum of second batch: 5+6+7+8
    }

    #[test]
    fn test_chunked_matmul() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = MemoryEfficientOps::chunked_matmul(&a, &b, 2).unwrap();

        // Expected result of matrix multiplication
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 22.0); // 1*1 + 2*3 + 3*5
        assert_eq!(result[[0, 1]], 28.0); // 1*2 + 2*4 + 3*6
        assert_eq!(result[[1, 0]], 49.0); // 4*1 + 5*3 + 6*5
        assert_eq!(result[[1, 1]], 64.0); // 4*2 + 5*4 + 6*6
    }
}
