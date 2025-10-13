//! Memory-efficient dataset storage and access
//!
//! This module provides memory-mapped dataset storage for handling large datasets
//! that don't fit in memory, along with zero-copy dataset views and arena allocation.

use memmap2::{Mmap, MmapMut, MmapOptions};
use scirs2_core::ndarray::{
    Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2,
};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Memory management errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Memory mapping error: {0}")]
    Mmap(String),
    #[error("Invalid dataset format: {0}")]
    InvalidFormat(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    #[error("Index out of bounds: {index} >= {len}")]
    IndexOutOfBounds { index: usize, len: usize },
}

pub type MemoryResult<T> = Result<T, MemoryError>;

/// Memory-mapped dataset header for metadata
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DatasetHeader {
    magic: [u8; 8],      // "SKLEARS\0"
    version: u32,        // Format version
    n_samples: u64,      // Number of samples
    n_features: u64,     // Number of features
    data_offset: u64,    // Offset to feature data
    targets_offset: u64, // Offset to target data
    feature_size: u32,   // Size of each feature element (8 for f64)
    target_size: u32,    // Size of each target element
    checksum: u64,       // Simple checksum for validation
}

impl DatasetHeader {
    const MAGIC: [u8; 8] = *b"SKLEARS\0";
    const VERSION: u32 = 1;

    fn new(n_samples: usize, n_features: usize, has_targets: bool) -> Self {
        let header_size = std::mem::size_of::<DatasetHeader>() as u64;
        let feature_data_size = (n_samples * n_features * 8) as u64;
        let targets_offset = if has_targets {
            header_size + feature_data_size
        } else {
            0
        };

        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            n_samples: n_samples as u64,
            n_features: n_features as u64,
            data_offset: header_size,
            targets_offset,
            feature_size: 8,                              // f64
            target_size: if has_targets { 8 } else { 0 }, // f64 for regression, could be i32 for classification
            checksum: 0,                                  // Will be calculated later
        }
    }

    fn validate(&self) -> MemoryResult<()> {
        if self.magic != Self::MAGIC {
            return Err(MemoryError::InvalidFormat(
                "Invalid magic number".to_string(),
            ));
        }
        if self.version != Self::VERSION {
            return Err(MemoryError::InvalidFormat(format!(
                "Unsupported version: {}",
                self.version
            )));
        }
        Ok(())
    }
}

/// Memory-mapped dataset for efficient large dataset access
pub struct MmapDataset {
    file: File,
    mmap: Mmap,
    header: DatasetHeader,
    path: PathBuf,
}

impl MmapDataset {
    /// Create a new memory-mapped dataset from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> MemoryResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < std::mem::size_of::<DatasetHeader>() {
            return Err(MemoryError::InvalidFormat("File too small".to_string()));
        }

        // Read header
        let header = unsafe { std::ptr::read(mmap.as_ptr() as *const DatasetHeader) };
        header.validate()?;

        Ok(Self {
            file,
            mmap,
            header,
            path,
        })
    }

    /// Create a new memory-mapped dataset file
    pub fn create<P: AsRef<Path>>(
        path: P,
        features: &Array2<f64>,
        targets: Option<&Array1<f64>>,
    ) -> MemoryResult<Self> {
        let path = path.as_ref().to_path_buf();
        let (n_samples, n_features) = features.dim();
        let has_targets = targets.is_some();

        let header = DatasetHeader::new(n_samples, n_features, has_targets);
        let total_size = header.data_offset
            + (n_samples * n_features * 8) as u64
            + if has_targets {
                (n_samples * 8) as u64
            } else {
                0
            };

        // Create file and set size
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(total_size)?;
        file.seek(SeekFrom::Start(0))?;

        // Write header
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const DatasetHeader as *const u8,
                std::mem::size_of::<DatasetHeader>(),
            )
        };
        file.write_all(header_bytes)?;

        // Write feature data
        file.seek(SeekFrom::Start(header.data_offset))?;
        let features_slice = features
            .as_slice()
            .ok_or_else(|| MemoryError::InvalidFormat("Features not contiguous".to_string()))?;
        let features_bytes = unsafe {
            std::slice::from_raw_parts(
                features_slice.as_ptr() as *const u8,
                features_slice.len() * 8,
            )
        };
        file.write_all(features_bytes)?;

        // Write targets if provided
        if let Some(targets) = targets {
            file.seek(SeekFrom::Start(header.targets_offset))?;
            let targets_slice = targets
                .as_slice()
                .ok_or_else(|| MemoryError::InvalidFormat("Targets not contiguous".to_string()))?;
            let targets_bytes = unsafe {
                std::slice::from_raw_parts(
                    targets_slice.as_ptr() as *const u8,
                    targets_slice.len() * 8,
                )
            };
            file.write_all(targets_bytes)?;
        }

        file.sync_all()?;

        // Re-open as memory-mapped
        drop(file);
        Self::from_file(path)
    }

    /// Get dataset dimensions
    pub fn shape(&self) -> (usize, usize) {
        (
            self.header.n_samples as usize,
            self.header.n_features as usize,
        )
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.header.n_samples as usize
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.header.n_features as usize
    }

    /// Check if targets are available
    pub fn has_targets(&self) -> bool {
        self.header.targets_offset != 0
    }

    /// Get a view of the entire feature matrix
    pub fn features(&self) -> MemoryResult<ArrayView2<f64>> {
        let (n_samples, n_features) = self.shape();
        let data_ptr =
            unsafe { self.mmap.as_ptr().add(self.header.data_offset as usize) as *const f64 };
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, n_samples * n_features) };

        ArrayView2::from_shape((n_samples, n_features), data_slice)
            .map_err(|e| MemoryError::InvalidFormat(format!("Shape error: {}", e)))
    }

    /// Get a view of a specific sample (row)
    pub fn sample(&self, index: usize) -> MemoryResult<ArrayView1<f64>> {
        if index >= self.n_samples() {
            return Err(MemoryError::IndexOutOfBounds {
                index,
                len: self.n_samples(),
            });
        }

        let n_features = self.n_features();
        let offset = self.header.data_offset as usize + index * n_features * 8;
        let data_ptr = unsafe { self.mmap.as_ptr().add(offset) as *const f64 };
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, n_features) };

        ArrayView1::from_shape(n_features, data_slice)
            .map_err(|e| MemoryError::InvalidFormat(format!("Shape error: {}", e)))
    }

    /// Get a view of targets if available
    pub fn targets(&self) -> MemoryResult<Option<ArrayView1<f64>>> {
        if !self.has_targets() {
            return Ok(None);
        }

        let n_samples = self.n_samples();
        let data_ptr =
            unsafe { self.mmap.as_ptr().add(self.header.targets_offset as usize) as *const f64 };
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, n_samples) };

        ArrayView1::from_shape(n_samples, data_slice)
            .map(Some)
            .map_err(|e| MemoryError::InvalidFormat(format!("Shape error: {}", e)))
    }

    /// Get a batch of samples
    pub fn batch(&self, start: usize, size: usize) -> MemoryResult<ArrayView2<f64>> {
        if start + size > self.n_samples() {
            return Err(MemoryError::IndexOutOfBounds {
                index: start + size,
                len: self.n_samples(),
            });
        }

        let n_features = self.n_features();
        let offset = self.header.data_offset as usize + start * n_features * 8;
        let data_ptr = unsafe { self.mmap.as_ptr().add(offset) as *const f64 };
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, size * n_features) };

        ArrayView2::from_shape((size, n_features), data_slice)
            .map_err(|e| MemoryError::InvalidFormat(format!("Shape error: {}", e)))
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get file size in bytes
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }
}

/// Iterator over batches of a memory-mapped dataset
pub struct MmapBatchIterator<'a> {
    dataset: &'a MmapDataset,
    batch_size: usize,
    current: usize,
}

impl<'a> MmapBatchIterator<'a> {
    fn new(dataset: &'a MmapDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current: 0,
        }
    }
}

impl<'a> Iterator for MmapBatchIterator<'a> {
    type Item = MemoryResult<ArrayView2<'a, f64>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.n_samples() {
            return None;
        }

        let remaining = self.dataset.n_samples() - self.current;
        let batch_size = self.batch_size.min(remaining);

        let result = self.dataset.batch(self.current, batch_size);
        self.current += batch_size;

        Some(result)
    }
}

impl MmapDataset {
    /// Create an iterator over batches
    pub fn batches(&self, batch_size: usize) -> MmapBatchIterator {
        MmapBatchIterator::new(self, batch_size)
    }
}

/// Mutable memory-mapped dataset for write operations
pub struct MmapDatasetMut {
    file: File,
    mmap: MmapMut,
    header: DatasetHeader,
    path: PathBuf,
}

impl MmapDatasetMut {
    /// Create a mutable memory-mapped dataset
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_samples: usize,
        n_features: usize,
        has_targets: bool,
    ) -> MemoryResult<Self> {
        let path = path.as_ref().to_path_buf();
        let header = DatasetHeader::new(n_samples, n_features, has_targets);
        let total_size = header.data_offset
            + (n_samples * n_features * 8) as u64
            + if has_targets {
                (n_samples * 8) as u64
            } else {
                0
            };

        // Create file
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(total_size)?;
        file.seek(SeekFrom::Start(0))?;

        // Write header
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const DatasetHeader as *const u8,
                std::mem::size_of::<DatasetHeader>(),
            )
        };
        file.write_all(header_bytes)?;
        file.sync_all()?;

        // Memory map
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            file,
            mmap,
            header,
            path,
        })
    }

    /// Get mutable view of features
    pub fn features_mut(&mut self) -> MemoryResult<ArrayViewMut2<f64>> {
        let (n_samples, n_features) = (
            self.header.n_samples as usize,
            self.header.n_features as usize,
        );
        let data_ptr =
            unsafe { self.mmap.as_mut_ptr().add(self.header.data_offset as usize) as *mut f64 };
        let data_slice =
            unsafe { std::slice::from_raw_parts_mut(data_ptr, n_samples * n_features) };

        ArrayViewMut2::from_shape((n_samples, n_features), data_slice)
            .map_err(|e| MemoryError::InvalidFormat(format!("Shape error: {}", e)))
    }

    /// Get mutable view of a specific sample
    pub fn sample_mut(&mut self, index: usize) -> MemoryResult<ArrayViewMut1<f64>> {
        if index >= self.header.n_samples as usize {
            return Err(MemoryError::IndexOutOfBounds {
                index,
                len: self.header.n_samples as usize,
            });
        }

        let n_features = self.header.n_features as usize;
        let offset = self.header.data_offset as usize + index * n_features * 8;
        let data_ptr = unsafe { self.mmap.as_mut_ptr().add(offset) as *mut f64 };
        let data_slice = unsafe { std::slice::from_raw_parts_mut(data_ptr, n_features) };

        ArrayViewMut1::from_shape(n_features, data_slice)
            .map_err(|e| MemoryError::InvalidFormat(format!("Shape error: {}", e)))
    }

    /// Sync changes to disk
    pub fn sync(&self) -> MemoryResult<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Get dataset shape
    pub fn shape(&self) -> (usize, usize) {
        (
            self.header.n_samples as usize,
            self.header.n_features as usize,
        )
    }
}

/// Arena allocator for efficient memory management of multiple datasets
pub struct DatasetArena {
    capacity: usize,
    used: usize,
    buffer: Vec<u8>,
    allocations: Vec<ArenaAllocation>,
    // Memory alignment for efficient access
    alignment: usize,
    // Free list for deallocated blocks
    free_blocks: Vec<(usize, usize)>, // (offset, size) pairs
}

/// Information about an arena allocation
#[derive(Debug, Clone)]
pub struct ArenaAllocation {
    offset: usize,
    size: usize,
    id: usize,
    allocated_at: std::time::Instant,
    name: Option<String>,
}

impl DatasetArena {
    /// Create a new arena with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self::new_with_alignment(capacity, 8) // Default to 8-byte alignment for f64
    }

    /// Create a new arena with specified capacity and alignment
    pub fn new_with_alignment(capacity: usize, alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "Alignment must be a power of two"
        );
        Self {
            capacity,
            used: 0,
            buffer: Vec::with_capacity(capacity),
            allocations: Vec::new(),
            alignment,
            free_blocks: Vec::new(),
        }
    }

    /// Allocate space for a dataset
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        self.allocate_named(size, None)
    }

    /// Allocate space with a name for debugging
    pub fn allocate_named(&mut self, size: usize, name: Option<String>) -> Option<usize> {
        // Try to find a suitable free block first
        if let Some(block_idx) = self.find_free_block(size) {
            let (offset, block_size) = self.free_blocks.remove(block_idx);

            // If the block is larger than needed, split it
            if block_size > size {
                let remaining_offset = offset + size;
                let remaining_size = block_size - size;
                self.free_blocks.push((remaining_offset, remaining_size));
            }

            let id = self.allocations.len();
            self.allocations.push(ArenaAllocation {
                offset,
                size,
                id,
                allocated_at: std::time::Instant::now(),
                name,
            });

            return Some(offset);
        }

        // Align the offset
        let aligned_offset = self.align_offset(self.used);
        let total_size = aligned_offset - self.used + size;

        if aligned_offset + size > self.capacity {
            return None;
        }

        // Resize buffer if needed
        if self.buffer.len() < aligned_offset + size {
            self.buffer.resize(aligned_offset + size, 0);
        }

        let id = self.allocations.len();
        self.allocations.push(ArenaAllocation {
            offset: aligned_offset,
            size,
            id,
            allocated_at: std::time::Instant::now(),
            name,
        });

        self.used = aligned_offset + size;
        Some(aligned_offset)
    }

    /// Find a suitable free block for the given size
    fn find_free_block(&self, size: usize) -> Option<usize> {
        self.free_blocks
            .iter()
            .enumerate()
            .find(|(_, (_, block_size))| *block_size >= size)
            .map(|(idx, _)| idx)
    }

    /// Align an offset to the arena's alignment
    fn align_offset(&self, offset: usize) -> usize {
        (offset + self.alignment - 1) & !(self.alignment - 1)
    }

    /// Deallocate a block by ID
    pub fn deallocate(&mut self, allocation_id: usize) -> bool {
        if let Some(alloc_idx) = self.allocations.iter().position(|a| a.id == allocation_id) {
            let allocation = self.allocations.remove(alloc_idx);
            self.free_blocks.push((allocation.offset, allocation.size));

            // Sort free blocks by offset for potential merging
            self.free_blocks.sort_by_key(|(offset, _)| *offset);
            self.merge_free_blocks();

            true
        } else {
            false
        }
    }

    /// Merge adjacent free blocks
    fn merge_free_blocks(&mut self) {
        if self.free_blocks.len() <= 1 {
            return;
        }

        let mut merged = Vec::new();
        let mut current = self.free_blocks[0];

        for &block in &self.free_blocks[1..] {
            if current.0 + current.1 == block.0 {
                // Adjacent blocks, merge them
                current.1 += block.1;
            } else {
                merged.push(current);
                current = block;
            }
        }
        merged.push(current);
        self.free_blocks = merged;
    }

    /// Get a slice for the allocation
    pub fn get_slice(&self, offset: usize, size: usize) -> Option<&[u8]> {
        if offset + size <= self.buffer.len() {
            Some(&self.buffer[offset..offset + size])
        } else {
            None
        }
    }

    /// Get a mutable slice for the allocation
    pub fn get_slice_mut(&mut self, offset: usize, size: usize) -> Option<&mut [u8]> {
        if offset + size <= self.buffer.len() {
            Some(&mut self.buffer[offset..offset + size])
        } else {
            None
        }
    }

    /// Get current usage statistics
    pub fn usage(&self) -> ArenaUsageStats {
        let active_size: usize = self.allocations.iter().map(|a| a.size).sum();
        let free_size: usize = self.free_blocks.iter().map(|(_, size)| *size).sum();
        let utilization = if self.capacity > 0 {
            active_size as f64 / self.capacity as f64
        } else {
            0.0
        };
        let fragmentation = if active_size + free_size > 0 {
            free_size as f64 / (active_size + free_size) as f64
        } else {
            0.0
        };

        ArenaUsageStats {
            total_capacity: self.capacity,
            active_size,
            free_size,
            wasted_size: self.used - active_size - free_size,
            utilization,
            fragmentation,
            active_allocations: self.allocations.len(),
            free_blocks: self.free_blocks.len(),
        }
    }

    /// Get allocation information
    pub fn get_allocation(&self, id: usize) -> Option<&ArenaAllocation> {
        self.allocations.iter().find(|a| a.id == id)
    }

    /// List all active allocations
    pub fn active_allocations(&self) -> &[ArenaAllocation] {
        &self.allocations
    }

    /// Get free blocks information
    pub fn free_blocks(&self) -> &[(usize, usize)] {
        &self.free_blocks
    }

    /// Compact the arena by moving all allocations to the beginning
    pub fn compact(&mut self) {
        if self.allocations.is_empty() {
            self.reset();
            return;
        }

        // Sort allocations by offset
        self.allocations.sort_by_key(|a| a.offset);

        let mut new_offset = 0;
        let mut moves = Vec::new();

        for allocation in &mut self.allocations {
            let aligned_offset = self.align_offset(new_offset);
            if allocation.offset != aligned_offset {
                moves.push((allocation.offset, aligned_offset, allocation.size));
                allocation.offset = aligned_offset;
            }
            new_offset = aligned_offset + allocation.size;
        }

        // Perform the actual moves
        for (old_offset, new_offset, size) in moves {
            let src = old_offset;
            let dst = new_offset;
            if src != dst {
                // Safe to use copy_within since we're moving data within the same buffer
                self.buffer.copy_within(src..src + size, dst);
            }
        }

        self.used = new_offset;
        self.free_blocks.clear();
    }

    /// Reset the arena (clears all allocations)
    pub fn reset(&mut self) {
        self.used = 0;
        self.allocations.clear();
        self.free_blocks.clear();
        self.buffer.clear();
    }

    /// Create a dataset view from an allocation
    pub fn dataset_view<'a>(
        &'a self,
        allocation_id: usize,
        n_samples: usize,
        n_features: usize,
    ) -> MemoryResult<ArrayView2<'a, f64>> {
        let allocation = self
            .get_allocation(allocation_id)
            .ok_or_else(|| MemoryError::InvalidFormat("Allocation not found".to_string()))?;

        let expected_size = n_samples * n_features * std::mem::size_of::<f64>();
        if allocation.size < expected_size {
            return Err(MemoryError::DimensionMismatch {
                expected: format!("at least {} bytes", expected_size),
                actual: format!("{} bytes", allocation.size),
            });
        }

        let slice = self
            .get_slice(allocation.offset, expected_size)
            .ok_or_else(|| MemoryError::InvalidFormat("Cannot access allocation".to_string()))?;

        let data_slice = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const f64, n_samples * n_features)
        };

        ArrayView2::from_shape((n_samples, n_features), data_slice)
            .map_err(|e| MemoryError::InvalidFormat(format!("Shape error: {}", e)))
    }
}

/// Statistics about arena usage
#[derive(Debug, Clone)]
pub struct ArenaUsageStats {
    pub total_capacity: usize,
    pub active_size: usize,
    pub free_size: usize,
    pub wasted_size: usize,
    pub utilization: f64,
    pub fragmentation: f64,
    pub active_allocations: usize,
    pub free_blocks: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;
    use std::env;

    #[test]
    fn test_mmap_dataset_creation() -> MemoryResult<()> {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_dataset.mmap");

        // Create test data
        let features =
            Array::from_shape_vec((100, 4), (0..400).map(|x| x as f64).collect()).unwrap();
        let targets = Array1::from_shape_vec(100, (0..100).map(|x| x as f64).collect()).unwrap();

        // Create memory-mapped dataset
        let dataset = MmapDataset::create(&path, &features, Some(&targets))?;

        // Verify dimensions
        assert_eq!(dataset.shape(), (100, 4));
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.has_targets());

        // Test feature access
        let features_view = dataset.features()?;
        assert_eq!(features_view.dim(), (100, 4));
        assert_eq!(features_view[[0, 0]], 0.0);
        assert_eq!(features_view[[99, 3]], 399.0);

        // Test sample access
        let sample = dataset.sample(50)?;
        assert_eq!(sample.len(), 4);
        assert_eq!(sample[0], 200.0); // 50 * 4 + 0

        // Test targets
        let targets_view = dataset.targets()?.unwrap();
        assert_eq!(targets_view.len(), 100);
        assert_eq!(targets_view[50], 50.0);

        // Test batch access
        let batch = dataset.batch(10, 5)?;
        assert_eq!(batch.dim(), (5, 4));

        // Clean up
        std::fs::remove_file(&path).ok();

        Ok(())
    }

    #[test]
    fn test_mmap_dataset_batches() -> MemoryResult<()> {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_batch_dataset.mmap");

        let features = Array::from_shape_vec((20, 3), (0..60).map(|x| x as f64).collect()).unwrap();
        let dataset = MmapDataset::create(&path, &features, None)?;

        let mut batch_count = 0;
        let batch_size = 7;

        for batch_result in dataset.batches(batch_size) {
            let batch = batch_result?;
            batch_count += 1;

            match batch_count {
                1 => assert_eq!(batch.dim(), (7, 3)),
                2 => assert_eq!(batch.dim(), (7, 3)),
                3 => assert_eq!(batch.dim(), (6, 3)), // Last batch is smaller
                _ => panic!("Too many batches"),
            }
        }

        assert_eq!(batch_count, 3);

        // Clean up
        std::fs::remove_file(&path).ok();

        Ok(())
    }

    #[test]
    fn test_mmap_dataset_mut() -> MemoryResult<()> {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_mut_dataset.mmap");

        // Create mutable dataset
        let mut dataset = MmapDatasetMut::create(&path, 10, 3, false)?;
        assert_eq!(dataset.shape(), (10, 3));

        // Write data
        {
            let mut features = dataset.features_mut()?;
            for i in 0..10 {
                for j in 0..3 {
                    features[[i, j]] = (i * 3 + j) as f64;
                }
            }
        }

        // Test sample modification
        {
            let mut sample = dataset.sample_mut(5)?;
            sample[0] = 999.0;
        }

        dataset.sync()?;

        // Verify by reading back
        drop(dataset);
        let read_dataset = MmapDataset::from_file(&path)?;
        let features = read_dataset.features()?;

        assert_eq!(features[[5, 0]], 999.0);
        assert_eq!(features[[5, 1]], 16.0); // 5 * 3 + 1
        assert_eq!(features[[9, 2]], 29.0); // 9 * 3 + 2

        // Clean up
        std::fs::remove_file(&path).ok();

        Ok(())
    }

    #[test]
    fn test_arena_allocator() {
        let mut arena = DatasetArena::new(1024);

        // Test allocation
        let offset1 = arena.allocate(100).unwrap();
        let offset2 = arena.allocate(200).unwrap();

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 100);

        // Test usage
        let (used, capacity, utilization) = arena.usage();
        assert_eq!(used, 300);
        assert_eq!(capacity, 1024);
        assert!((utilization - 300.0 / 1024.0).abs() < f64::EPSILON);

        // Test slice access
        if let Some(slice) = arena.get_slice_mut(offset1, 100) {
            slice[0] = 42;
        }

        if let Some(slice) = arena.get_slice(offset1, 100) {
            assert_eq!(slice[0], 42);
        }

        // Test capacity exceeded
        assert!(arena.allocate(800).is_none()); // Would exceed capacity

        // Test reset
        arena.reset();
        let (used, _, _) = arena.usage();
        assert_eq!(used, 0);
    }

    #[test]
    fn test_dataset_header_validation() {
        let valid_header = DatasetHeader::new(100, 10, true);
        assert!(valid_header.validate().is_ok());

        let mut invalid_header = valid_header;
        invalid_header.magic = *b"INVALID\0";
        assert!(invalid_header.validate().is_err());

        let mut wrong_version = valid_header;
        wrong_version.version = 999;
        assert!(wrong_version.validate().is_err());
    }
}
