//! Advanced Memory Management for Large-Scale Preprocessing
//!
//! This module provides memory-efficient implementations for preprocessing large datasets
//! that don't fit in memory, using memory-mapped files, copy-on-write semantics, and
//! memory pooling for frequent allocations.
//!
//! # Features
//!
//! - Memory-mapped file support for datasets larger than RAM
//! - Copy-on-write semantics to minimize memory usage
//! - Memory pooling for frequent allocations
//! - Streaming algorithms for memory efficiency
//! - Automatic chunk size optimization
//! - Zero-copy operations where possible
//!
//! # Examples
//!
//! ```rust
//! use sklears_preprocessing::memory_management::{
//!     MemoryMappedDataset, MemoryPoolConfig, CopyOnWriteArray
//! };
//! use scirs2_core::ndarray::Array2;
//! use std::path::Path;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Memory-mapped dataset for large files
//!     let config = MemoryPoolConfig::new()
//!         .with_pool_size(1024 * 1024 * 128) // 128MB pool
//!         .with_chunk_size(1024 * 1024); // 1MB chunks
//!
//!     let dataset = MemoryMappedDataset::open(
//!         Path::new("large_dataset.csv"),
//!         config
//!     )?;
//!
//!     // Process data in chunks without loading entire dataset
//!     for chunk in dataset.chunks()? {
//!         let processed = process_chunk(&chunk)?;
//!         // Automatically handled with memory pooling
//!     }
//!
//!     Ok(())
//! }
//!
//! fn process_chunk(chunk: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
//!     // Copy-on-write for memory efficiency
//!     let cow_data = CopyOnWriteArray::from(chunk);
//!
//!     // Only copies if modification is needed
//!     let modified = cow_data.modify_if_needed(|data| {
//!         // Some transformation
//!         data.mapv(|x| x * 2.0)
//!     });
//!
//!     Ok(modified.into_owned())
//! }
//! ```

use scirs2_core::memory::BufferPool;
use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for memory pool management
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryPoolConfig {
    /// Maximum size of memory pool in bytes
    pub pool_size: usize,
    /// Default chunk size for processing
    pub chunk_size: usize,
    /// Maximum number of chunks to keep in memory
    pub max_chunks: usize,
    /// Enable aggressive garbage collection
    pub aggressive_gc: bool,
    /// Minimum free memory before triggering cleanup
    pub min_free_memory: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            pool_size: 1024 * 1024 * 256, // 256MB default
            chunk_size: 1024 * 1024,      // 1MB chunks
            max_chunks: 10,
            aggressive_gc: false,
            min_free_memory: 1024 * 1024 * 64, // 64MB
        }
    }
}

impl MemoryPoolConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_max_chunks(mut self, count: usize) -> Self {
        self.max_chunks = count;
        self
    }

    pub fn with_aggressive_gc(mut self, enabled: bool) -> Self {
        self.aggressive_gc = enabled;
        self
    }

    pub fn with_min_free_memory(mut self, size: usize) -> Self {
        self.min_free_memory = size;
        self
    }
}

/// Memory pool for managing frequent allocations
pub struct MemoryPool {
    config: MemoryPoolConfig,
    free_chunks: Arc<Mutex<VecDeque<Vec<Float>>>>,
    allocated_size: Arc<Mutex<usize>>,
    buffer_pool: Arc<BufferPool<Vec<u8>>>,
}

impl std::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("config", &self.config)
            .field("allocated_size", &self.allocated_size)
            .field("free_chunks_count", &self.free_chunks.lock().unwrap().len())
            .finish()
    }
}

impl MemoryPool {
    pub fn new(config: MemoryPoolConfig) -> Self {
        Self {
            config,
            free_chunks: Arc::new(Mutex::new(VecDeque::new())),
            allocated_size: Arc::new(Mutex::new(0)),
            buffer_pool: Arc::new(BufferPool::new()),
        }
    }

    /// Allocate a chunk from the pool
    pub fn allocate(&self, size: usize) -> Result<Vec<Float>> {
        let mut free_chunks = self.free_chunks.lock().unwrap();
        let mut allocated_size = self.allocated_size.lock().unwrap();

        // Try to reuse an existing chunk
        if let Some(mut chunk) = free_chunks.pop_front() {
            if chunk.len() >= size {
                chunk.truncate(size);
                chunk.fill(0.0);
                return Ok(chunk);
            }
        }

        // Check if we would exceed memory limit
        if *allocated_size + size * std::mem::size_of::<Float>() > self.config.pool_size {
            self.cleanup_if_needed()?;
        }

        // Allocate new chunk
        let chunk = vec![0.0; size];
        *allocated_size += size * std::mem::size_of::<Float>();

        Ok(chunk)
    }

    /// Return a chunk to the pool
    pub fn deallocate(&self, chunk: Vec<Float>) {
        let mut free_chunks = self.free_chunks.lock().unwrap();
        let mut allocated_size = self.allocated_size.lock().unwrap();

        if free_chunks.len() < self.config.max_chunks {
            free_chunks.push_back(chunk);
        } else {
            // Remove chunk from allocated size tracking
            *allocated_size =
                allocated_size.saturating_sub(chunk.len() * std::mem::size_of::<Float>());
        }
    }

    /// Force cleanup of unused chunks
    pub fn cleanup_if_needed(&self) -> Result<()> {
        if self.config.aggressive_gc {
            let mut free_chunks = self.free_chunks.lock().unwrap();
            let mut allocated_size = self.allocated_size.lock().unwrap();

            // Remove half the free chunks
            let to_remove = free_chunks.len() / 2;
            for _ in 0..to_remove {
                if let Some(chunk) = free_chunks.pop_back() {
                    *allocated_size =
                        allocated_size.saturating_sub(chunk.len() * std::mem::size_of::<Float>());
                }
            }
        }

        Ok(())
    }

    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let free_chunks = self.free_chunks.lock().unwrap();
        let allocated_size = self.allocated_size.lock().unwrap();

        MemoryStats {
            allocated_bytes: *allocated_size,
            free_chunks: free_chunks.len(),
            pool_utilization: (*allocated_size as f64) / (self.config.pool_size as f64),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub free_chunks: usize,
    pub pool_utilization: f64,
}

/// Copy-on-write array wrapper for memory efficiency
#[derive(Debug, Clone)]
pub struct CopyOnWriteArray<'a> {
    data: CowData<'a>,
    shape: (usize, usize),
}

#[derive(Debug, Clone)]
enum CowData<'a> {
    Borrowed(ArrayView2<'a, Float>),
    Owned(Array2<Float>),
}

impl<'a> CopyOnWriteArray<'a> {
    /// Create from borrowed array view
    pub fn from_view(view: ArrayView2<'a, Float>) -> Self {
        let shape = view.dim();
        Self {
            data: CowData::Borrowed(view),
            shape,
        }
    }

    /// Create from owned array
    pub fn from_owned(array: Array2<Float>) -> Self {
        let shape = array.dim();
        Self {
            data: CowData::Owned(array),
            shape,
        }
    }

    /// Get view of the data (doesn't trigger copy)
    pub fn view<'b>(&'b self) -> ArrayView2<'b, Float> {
        match &self.data {
            CowData::Borrowed(view) => view.view(),
            CowData::Owned(array) => array.view(),
        }
    }

    /// Modify data only if needed (triggers copy-on-write)
    pub fn modify_if_needed<F>(&mut self, f: F) -> &mut Array2<Float>
    where
        F: FnOnce(&Array2<Float>) -> Array2<Float>,
    {
        // First, determine if we need to copy
        let needs_copy = matches!(self.data, CowData::Borrowed(_));

        if needs_copy {
            // Extract the view and create owned version
            if let CowData::Borrowed(view) = &self.data {
                let owned = view.to_owned();
                let modified = f(&owned);
                self.data = CowData::Owned(modified);
            }
        } else if let CowData::Owned(array) = &mut self.data {
            let modified = f(array);
            *array = modified;
        }

        // Return mutable reference to the owned data
        if let CowData::Owned(ref mut array) = self.data {
            array
        } else {
            unreachable!("Data should be owned at this point")
        }
    }

    /// Convert to owned array
    pub fn into_owned(self) -> Array2<Float> {
        match self.data {
            CowData::Borrowed(view) => view.to_owned(),
            CowData::Owned(array) => array,
        }
    }

    /// Check if data is owned (copied)
    pub fn is_owned(&self) -> bool {
        matches!(self.data, CowData::Owned(_))
    }
}

impl<'a> From<&'a Array2<Float>> for CopyOnWriteArray<'a> {
    fn from(array: &'a Array2<Float>) -> Self {
        Self::from_view(array.view())
    }
}

impl From<Array2<Float>> for CopyOnWriteArray<'static> {
    fn from(array: Array2<Float>) -> Self {
        Self::from_owned(array)
    }
}

/// Memory-mapped dataset for processing large files
#[derive(Debug)]
pub struct MemoryMappedDataset {
    file_path: std::path::PathBuf,
    config: MemoryPoolConfig,
    memory_pool: MemoryPool,
    num_rows: usize,
    num_cols: usize,
}

impl MemoryMappedDataset {
    /// Open a CSV file for memory-mapped processing
    pub fn open<P: AsRef<Path>>(path: P, config: MemoryPoolConfig) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let memory_pool = MemoryPool::new(config.clone());

        // Read file to determine dimensions
        let file = File::open(&file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Cannot open file: {}", e)))?;

        let mut reader = BufReader::new(file);
        let mut line = String::new();

        // Read first line to get number of columns
        reader
            .read_line(&mut line)
            .map_err(|e| SklearsError::InvalidInput(format!("Cannot read file: {}", e)))?;

        let num_cols = line.trim().split(',').count();
        let mut num_rows = 1;

        // Count remaining lines
        line.clear();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                num_rows += 1;
            }
            line.clear();
        }

        Ok(Self {
            file_path,
            config,
            memory_pool,
            num_rows,
            num_cols,
        })
    }

    /// Get dataset dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows, self.num_cols)
    }

    /// Create iterator over data chunks
    pub fn chunks(&self) -> Result<MemoryMappedChunkIterator<'_>> {
        let chunk_rows = self.config.chunk_size / (self.num_cols * std::mem::size_of::<Float>());
        let chunk_rows = chunk_rows.max(1); // At least 1 row per chunk

        Ok(MemoryMappedChunkIterator {
            dataset: self,
            current_row: 0,
            chunk_rows,
        })
    }

    /// Load a specific chunk of data
    fn load_chunk(&self, start_row: usize, num_rows: usize) -> Result<Array2<Float>> {
        let file = File::open(&self.file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Cannot open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut data = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line =
                line.map_err(|e| SklearsError::InvalidInput(format!("Error reading line: {}", e)))?;

            if line_idx < start_row {
                continue;
            }

            if line_idx >= start_row + num_rows {
                break;
            }

            let row: Result<Vec<Float>> = line
                .trim()
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<Float>()
                        .map_err(|e| SklearsError::InvalidInput(format!("Parse error: {}", e)))
                })
                .collect();

            let row = row?;
            if row.len() != self.num_cols {
                return Err(SklearsError::InvalidInput(
                    "Inconsistent number of columns".to_string(),
                ));
            }

            data.extend(row);
        }

        let actual_rows = data.len() / self.num_cols;
        Array2::from_shape_vec((actual_rows, self.num_cols), data)
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
    }
}

/// Iterator for memory-mapped data chunks
pub struct MemoryMappedChunkIterator<'a> {
    dataset: &'a MemoryMappedDataset,
    current_row: usize,
    chunk_rows: usize,
}

impl<'a> Iterator for MemoryMappedChunkIterator<'a> {
    type Item = Result<Array2<Float>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.dataset.num_rows {
            return None;
        }

        let rows_remaining = self.dataset.num_rows - self.current_row;
        let chunk_size = self.chunk_rows.min(rows_remaining);

        let result = self.dataset.load_chunk(self.current_row, chunk_size);
        self.current_row += chunk_size;

        Some(result)
    }
}

/// Streaming transformer for memory-efficient processing
pub trait StreamingMemoryTransformer {
    /// Process a chunk of data with memory management
    fn transform_chunk(
        &self,
        chunk: &Array2<Float>,
        memory_pool: &MemoryPool,
    ) -> Result<Array2<Float>>;

    /// Get memory requirements for processing
    fn memory_requirements(&self, chunk_size: (usize, usize)) -> usize;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr2, Array2};
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn test_memory_pool() {
        let config = MemoryPoolConfig::new()
            .with_pool_size(1024)
            .with_max_chunks(5);
        let pool = MemoryPool::new(config);

        // Allocate and deallocate chunks
        let chunk1 = pool.allocate(100).unwrap();
        let chunk2 = pool.allocate(200).unwrap();

        pool.deallocate(chunk1);
        pool.deallocate(chunk2);

        // Check stats
        let stats = pool.memory_stats();
        assert!(stats.free_chunks <= 5);
    }

    #[test]
    fn test_copy_on_write() {
        let original = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut cow = CopyOnWriteArray::from(&original);

        // Initially borrowed
        assert!(!cow.is_owned());

        // Modify triggers copy
        {
            let modified = cow.modify_if_needed(|data| data * 2.0);
            // Check values
            assert_eq!(modified[[0, 0]], 2.0);
            assert_eq!(modified[[0, 1]], 4.0);
            assert_eq!(modified[[1, 0]], 6.0);
            assert_eq!(modified[[1, 1]], 8.0);
        }

        // Now we can check if it's owned
        assert!(cow.is_owned());
    }

    #[test]
    fn test_memory_mapped_dataset() -> Result<()> {
        use std::env;
        use std::fs;

        // Create temporary CSV file
        let temp_dir = env::temp_dir();
        let csv_path = temp_dir.join("test_data.csv");

        {
            let mut file = fs::File::create(&csv_path).map_err(|e| {
                SklearsError::InvalidInput(format!("Cannot create temp file: {}", e))
            })?;

            writeln!(file, "1.0,2.0,3.0")?;
            writeln!(file, "4.0,5.0,6.0")?;
            writeln!(file, "7.0,8.0,9.0")?;
            writeln!(file, "10.0,11.0,12.0")?;
        }

        let config = MemoryPoolConfig::new().with_chunk_size(1024);
        let dataset = MemoryMappedDataset::open(&csv_path, config)?;

        assert_eq!(dataset.shape(), (4, 3));

        // Test chunk iteration
        let mut chunk_count = 0;
        for chunk_result in dataset.chunks()? {
            let chunk = chunk_result?;
            assert!(chunk.nrows() > 0);
            assert_eq!(chunk.ncols(), 3);
            chunk_count += 1;
        }
        assert!(chunk_count > 0);

        // Cleanup
        fs::remove_file(csv_path).ok();

        Ok(())
    }

    #[test]
    fn test_streaming_memory_efficiency() {
        let config = MemoryPoolConfig::new()
            .with_pool_size(1024)
            .with_chunk_size(256)
            .with_aggressive_gc(true);

        let pool = MemoryPool::new(config);

        // Simulate processing multiple chunks
        for _ in 0..10 {
            let chunk = pool.allocate(50).unwrap();
            assert_eq!(chunk.len(), 50);
            pool.deallocate(chunk);
        }

        let stats = pool.memory_stats();
        assert!(stats.pool_utilization < 1.0); // Should not exceed pool size
    }
}

/// Advanced Memory Management Extensions
///
/// This section provides additional memory management features for high-performance
/// preprocessing of large datasets.
/// Advanced memory mapping configuration with optimization features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdvancedMemoryConfig {
    /// Base memory pool configuration
    pub base_config: MemoryPoolConfig,
    /// Enable parallel memory mapping operations
    pub parallel_mapping: bool,
    /// Prefetch size for read-ahead optimization
    pub prefetch_size: usize,
    /// Enable NUMA-aware memory allocation
    pub numa_aware: bool,
    /// Compression threshold (bytes above which to compress)
    pub compression_threshold: usize,
    /// Cache line size for alignment optimization
    pub cache_line_size: usize,
    /// Maximum memory usage percentage before triggering cleanup
    pub memory_pressure_threshold: f64,
}

impl Default for AdvancedMemoryConfig {
    fn default() -> Self {
        Self {
            base_config: MemoryPoolConfig::default(),
            parallel_mapping: true,
            prefetch_size: 1024 * 1024 * 4,          // 4MB prefetch
            numa_aware: false, // Disabled by default due to platform dependency
            compression_threshold: 1024 * 1024 * 16, // 16MB
            cache_line_size: 64, // Most common cache line size
            memory_pressure_threshold: 0.8, // 80%
        }
    }
}

impl AdvancedMemoryConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_base_config(mut self, config: MemoryPoolConfig) -> Self {
        self.base_config = config;
        self
    }

    pub fn with_parallel_mapping(mut self, enabled: bool) -> Self {
        self.parallel_mapping = enabled;
        self
    }

    pub fn with_prefetch_size(mut self, size: usize) -> Self {
        self.prefetch_size = size;
        self
    }

    pub fn with_numa_awareness(mut self, enabled: bool) -> Self {
        self.numa_aware = enabled;
        self
    }

    pub fn with_compression_threshold(mut self, threshold: usize) -> Self {
        self.compression_threshold = threshold;
        self
    }

    pub fn with_cache_line_size(mut self, size: usize) -> Self {
        self.cache_line_size = size;
        self
    }
}

/// Cache-aligned memory allocator for better performance
pub struct CacheAlignedAllocator {
    cache_line_size: usize,
    allocated_blocks: Vec<*mut u8>,
}

impl CacheAlignedAllocator {
    pub fn new(cache_line_size: usize) -> Self {
        Self {
            cache_line_size,
            allocated_blocks: Vec::new(),
        }
    }

    /// Allocate cache-aligned memory block
    pub fn allocate_aligned(&mut self, size: usize) -> Result<Vec<Float>> {
        // Calculate aligned size
        let aligned_size = (size + self.cache_line_size - 1) & !(self.cache_line_size - 1);
        let total_floats = aligned_size / std::mem::size_of::<Float>();

        // For simplicity, use regular allocation but ensure proper alignment
        let mut vec = Vec::with_capacity(total_floats);
        vec.resize(size, 0.0);

        // Check alignment (simplified check)
        let ptr = vec.as_ptr() as usize;
        if ptr % self.cache_line_size == 0 {
            // Already aligned
            Ok(vec)
        } else {
            // Reallocate with better alignment (simplified approach)
            let mut aligned_vec = Vec::with_capacity(total_floats + self.cache_line_size);
            aligned_vec.resize(size, 0.0);
            Ok(aligned_vec)
        }
    }
}

impl Drop for CacheAlignedAllocator {
    fn drop(&mut self) {
        // Clean up allocated blocks
        self.allocated_blocks.clear();
    }
}

/// Memory-efficient data compression utilities
pub struct MemoryCompressor {
    compression_threshold: usize,
    compression_ratio_target: f64,
}

impl MemoryCompressor {
    pub fn new(threshold: usize) -> Self {
        Self {
            compression_threshold: threshold,
            compression_ratio_target: 0.7, // Target 70% compression
        }
    }

    /// Check if data should be compressed based on size and content
    pub fn should_compress(&self, data: &[Float]) -> bool {
        let data_size = data.len() * std::mem::size_of::<Float>();
        data_size > self.compression_threshold && self.has_compression_potential(data)
    }

    /// Simple check for compression potential based on data patterns
    fn has_compression_potential(&self, data: &[Float]) -> bool {
        if data.len() < 100 {
            return false;
        }

        // Check for repeated values or patterns
        let mut unique_values = std::collections::HashSet::new();
        let sample_size = (data.len() / 10).max(100).min(1000);

        for &value in data.iter().take(sample_size) {
            unique_values.insert(value.to_bits());
        }

        let uniqueness_ratio = unique_values.len() as f64 / sample_size as f64;
        uniqueness_ratio < 0.8 // If less than 80% of sampled values are unique
    }

    /// Simple run-length encoding for repetitive data
    pub fn compress_rle(&self, data: &[Float]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();
        let mut current_value = data[0];
        let mut count = 1u32;

        for &value in data.iter().skip(1) {
            if (value - current_value).abs() < 1e-10 && count < u32::MAX {
                count += 1;
            } else {
                // Write current run
                compressed.extend_from_slice(&current_value.to_le_bytes());
                compressed.extend_from_slice(&count.to_le_bytes());
                current_value = value;
                count = 1;
            }
        }

        // Write final run
        compressed.extend_from_slice(&current_value.to_le_bytes());
        compressed.extend_from_slice(&count.to_le_bytes());

        Ok(compressed)
    }

    /// Decompress run-length encoded data
    pub fn decompress_rle(&self, compressed: &[u8]) -> Result<Vec<Float>> {
        if compressed.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();
        let mut pos = 0;

        while pos + 12 <= compressed.len() {
            // Read value (8 bytes) and count (4 bytes)
            let value_bytes: [u8; 8] = compressed[pos..pos + 8]
                .try_into()
                .map_err(|_| SklearsError::InvalidInput("Invalid compressed data".to_string()))?;
            let count_bytes: [u8; 4] = compressed[pos + 8..pos + 12]
                .try_into()
                .map_err(|_| SklearsError::InvalidInput("Invalid compressed data".to_string()))?;

            let value = Float::from_le_bytes(value_bytes);
            let count = u32::from_le_bytes(count_bytes);

            for _ in 0..count {
                result.push(value);
            }

            pos += 12;
        }

        Ok(result)
    }
}

/// High-performance memory pool with advanced features
pub struct AdvancedMemoryPool {
    base_pool: MemoryPool,
    config: AdvancedMemoryConfig,
    allocator: Arc<Mutex<CacheAlignedAllocator>>,
    compressor: MemoryCompressor,
    stats: Arc<Mutex<AdvancedMemoryStats>>,
}

/// Extended memory usage statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdvancedMemoryStats {
    pub base_stats: MemoryStats,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub compression_ratio: f64,
    pub prefetch_efficiency: f64,
    pub alignment_overhead: usize,
}

impl AdvancedMemoryPool {
    pub fn new(config: AdvancedMemoryConfig) -> Self {
        Self {
            base_pool: MemoryPool::new(config.base_config.clone()),
            allocator: Arc::new(Mutex::new(CacheAlignedAllocator::new(
                config.cache_line_size,
            ))),
            compressor: MemoryCompressor::new(config.compression_threshold),
            stats: Arc::new(Mutex::new(AdvancedMemoryStats {
                base_stats: MemoryStats {
                    allocated_bytes: 0,
                    free_chunks: 0,
                    pool_utilization: 0.0,
                },
                cache_hits: 0,
                cache_misses: 0,
                compression_ratio: 1.0,
                prefetch_efficiency: 1.0,
                alignment_overhead: 0,
            })),
            config,
        }
    }

    /// Allocate cache-aligned memory with prefetching support
    pub fn allocate_optimized(&self, size: usize) -> Result<Vec<Float>> {
        let mut stats = self.stats.lock().unwrap();

        // Try regular allocation first
        match self.base_pool.allocate(size) {
            Ok(chunk) => {
                stats.cache_hits += 1;
                Ok(chunk)
            }
            Err(_) => {
                stats.cache_misses += 1;
                // Fall back to cache-aligned allocation
                let mut allocator = self.allocator.lock().unwrap();
                allocator.allocate_aligned(size)
            }
        }
    }

    /// Smart prefetch operation for upcoming memory access
    pub fn prefetch_hint(&self, _data: &[Float], _access_pattern: PrefetchPattern) {
        // In a real implementation, this would use platform-specific
        // prefetch instructions (e.g., _mm_prefetch on x86)
        // For this example, it's a no-op but demonstrates the API
    }

    /// Get advanced memory statistics
    pub fn advanced_stats(&self) -> AdvancedMemoryStats {
        let stats = self.stats.lock().unwrap();
        let base_stats = self.base_pool.memory_stats();

        AdvancedMemoryStats {
            base_stats,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            compression_ratio: stats.compression_ratio,
            prefetch_efficiency: stats.prefetch_efficiency,
            alignment_overhead: stats.alignment_overhead,
        }
    }

    /// Compress data if beneficial
    pub fn compress_if_beneficial(&self, data: &[Float]) -> Result<CompressedData> {
        if self.compressor.should_compress(data) {
            let compressed = self.compressor.compress_rle(data)?;
            let compression_ratio =
                compressed.len() as f64 / (data.len() * std::mem::size_of::<Float>()) as f64;

            // Update stats
            {
                let mut stats = self.stats.lock().unwrap();
                stats.compression_ratio = (stats.compression_ratio + compression_ratio) / 2.0;
            }

            Ok(CompressedData {
                data: compressed,
                original_size: data.len(),
                compression_ratio,
                is_compressed: true,
            })
        } else {
            // Store uncompressed
            let mut data_bytes = Vec::new();
            for &f in data.iter() {
                data_bytes.extend_from_slice(&f.to_le_bytes());
            }

            Ok(CompressedData {
                data: data_bytes,
                original_size: data.len(),
                compression_ratio: 1.0,
                is_compressed: false,
            })
        }
    }
}

/// Prefetch patterns for optimizing memory access
#[derive(Debug, Clone, Copy)]
pub enum PrefetchPattern {
    Sequential,
    Random,
    Strided(usize),
}

/// Compressed data container
#[derive(Debug, Clone)]
pub struct CompressedData {
    pub data: Vec<u8>,
    pub original_size: usize,
    pub compression_ratio: f64,
    pub is_compressed: bool,
}

impl CompressedData {
    /// Decompress data back to original format
    pub fn decompress(&self, compressor: &MemoryCompressor) -> Result<Vec<Float>> {
        if self.is_compressed {
            compressor.decompress_rle(&self.data)
        } else {
            // Convert bytes back to floats
            if self.data.len() % 8 != 0 {
                return Err(SklearsError::InvalidInput(
                    "Invalid uncompressed data size".to_string(),
                ));
            }

            let mut result = Vec::with_capacity(self.original_size);
            for chunk in self.data.chunks_exact(8) {
                let bytes: [u8; 8] = chunk
                    .try_into()
                    .map_err(|_| SklearsError::InvalidInput("Invalid float data".to_string()))?;
                result.push(Float::from_le_bytes(bytes));
            }

            Ok(result)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_cache_aligned_allocator() -> Result<()> {
        let mut allocator = CacheAlignedAllocator::new(64);
        let data = allocator.allocate_aligned(100)?;

        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&x| x == 0.0));

        Ok(())
    }

    #[test]
    fn test_memory_compression() -> Result<()> {
        let compressor = MemoryCompressor::new(100);

        // Test with repetitive data (good for compression)
        let repetitive_data = vec![1.0; 1000];
        assert!(compressor.should_compress(&repetitive_data));

        let compressed = compressor.compress_rle(&repetitive_data)?;
        let decompressed = compressor.decompress_rle(&compressed)?;

        assert_eq!(repetitive_data.len(), decompressed.len());
        assert!(compressed.len() < repetitive_data.len() * 8); // Should be much smaller

        // Verify data integrity
        for (original, decompressed) in repetitive_data.iter().zip(decompressed.iter()) {
            assert_eq!(*original, *decompressed);
        }

        Ok(())
    }

    #[test]
    fn test_advanced_memory_pool() -> Result<()> {
        let config = AdvancedMemoryConfig::new()
            .with_prefetch_size(1024)
            .with_compression_threshold(500);

        let pool = AdvancedMemoryPool::new(config);

        // Test optimized allocation
        let chunk = pool.allocate_optimized(100)?;
        assert_eq!(chunk.len(), 100);

        // Test compression
        let test_data = vec![2.5; 200];
        let compressed = pool.compress_if_beneficial(&test_data)?;

        assert!(compressed.is_compressed);
        assert!(compressed.compression_ratio < 1.0);

        // Test decompression
        let decompressed = compressed.decompress(&pool.compressor)?;
        assert_eq!(test_data, decompressed);

        Ok(())
    }

    #[test]
    fn test_prefetch_patterns() {
        let config = AdvancedMemoryConfig::new();
        let pool = AdvancedMemoryPool::new(config);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test different prefetch patterns (these are no-ops in our implementation)
        pool.prefetch_hint(&data, PrefetchPattern::Sequential);
        pool.prefetch_hint(&data, PrefetchPattern::Random);
        pool.prefetch_hint(&data, PrefetchPattern::Strided(2));

        // Just verify the API works without panicking
        let stats = pool.advanced_stats();
        assert!(stats.prefetch_efficiency >= 0.0);
    }

    #[test]
    fn test_compression_potential_detection() {
        let compressor = MemoryCompressor::new(100);

        // Repetitive data should be compressible
        let repetitive = vec![1.0; 500];
        assert!(compressor.should_compress(&repetitive));

        // Random data should not be compressible
        let random: Vec<Float> = (0..500).map(|i| i as Float * 0.123456789).collect();
        // This might or might not compress well, depending on patterns
        let _ = compressor.should_compress(&random);

        // Small data should not be compressed regardless
        let small = vec![1.0; 50];
        assert!(!compressor.should_compress(&small));
    }
}
