//! Memory-mapped kernel matrices for large-scale SVM training
//!
//! This module provides memory-mapped implementations of kernel matrices that allow
//! efficient access to kernel values without loading the entire matrix into memory.
//! This is essential for training SVMs on very large datasets where the kernel matrix
//! would be too large to fit in available RAM.

use crate::kernels::Kernel;
use memmap2::{Mmap, MmapMut, MmapOptions};
use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Configuration for memory-mapped kernel matrices
#[derive(Debug, Clone)]
pub struct MemoryMappedKernelConfig {
    /// Path to store the memory-mapped file
    pub file_path: PathBuf,
    /// Cache size for frequently accessed kernel values (in elements)
    pub cache_size: usize,
    /// Block size for chunked access (in elements)
    pub block_size: usize,
    /// Whether to precompute the entire kernel matrix
    pub precompute_all: bool,
    /// Compression level for on-disk storage (0-9, 0=no compression)
    pub compression_level: u8,
    /// Whether to use read-only mode
    pub read_only: bool,
    /// Number of threads for parallel computation
    pub num_threads: usize,
}

impl Default for MemoryMappedKernelConfig {
    fn default() -> Self {
        #[cfg(feature = "parallel")]
        let default_threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let default_threads = num_cpus::get();

        Self {
            file_path: PathBuf::from("/tmp/sklears_kernel_matrix.dat"),
            cache_size: 10000,
            block_size: 1000,
            precompute_all: false,
            compression_level: 0,
            read_only: false,
            num_threads: default_threads,
        }
    }
}

/// Memory-mapped kernel matrix
pub struct MemoryMappedKernelMatrix {
    config: MemoryMappedKernelConfig,
    file: File,
    mmap: Option<Mmap>,
    mmap_mut: Option<MmapMut>,
    dimensions: (usize, usize),
    kernel: Box<dyn Kernel>,
    cache: Arc<Mutex<LRUCache<(usize, usize), Float>>>,
    x_data: Option<Array2<Float>>, // Store original data for on-demand computation
    header_size: usize,
}

impl MemoryMappedKernelMatrix {
    /// Create a new memory-mapped kernel matrix
    pub fn new(
        kernel: Box<dyn Kernel>,
        dimensions: (usize, usize),
        config: MemoryMappedKernelConfig,
    ) -> Result<Self> {
        let header_size = std::mem::size_of::<KernelMatrixHeader>();

        let file = if config.read_only {
            File::open(&config.file_path)?
        } else {
            OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&config.file_path)?
        };

        let cache = Arc::new(Mutex::new(LRUCache::new(config.cache_size)));

        let mut matrix = Self {
            config,
            file,
            mmap: None,
            mmap_mut: None,
            dimensions,
            kernel,
            cache,
            x_data: None,
            header_size,
        };

        matrix.initialize()?;
        Ok(matrix)
    }

    /// Initialize the memory-mapped file
    fn initialize(&mut self) -> Result<()> {
        let total_elements = self.dimensions.0 * self.dimensions.1;
        let data_size = total_elements * std::mem::size_of::<Float>();
        let total_size = self.header_size + data_size;

        // Resize file to accommodate header + data
        self.file.set_len(total_size as u64)?;

        // Write header
        let header = KernelMatrixHeader {
            rows: self.dimensions.0 as u64,
            cols: self.dimensions.1 as u64,
            element_size: std::mem::size_of::<Float>() as u32,
            compression_level: self.config.compression_level,
            version: 1,
        };

        self.write_header(&header)?;

        // Create memory mapping
        if self.config.read_only {
            let mmap = unsafe {
                MmapOptions::new()
                    .offset(self.header_size as u64)
                    .len(data_size)
                    .map(&self.file)?
            };
            self.mmap = Some(mmap);
        } else {
            let mmap_mut = unsafe {
                MmapOptions::new()
                    .offset(self.header_size as u64)
                    .len(data_size)
                    .map_mut(&self.file)?
            };
            self.mmap_mut = Some(mmap_mut);
        }

        Ok(())
    }

    /// Set the original data for on-demand computation
    pub fn set_data(&mut self, x: Array2<Float>) {
        self.x_data = Some(x);
    }

    /// Precompute the entire kernel matrix
    pub fn precompute(&mut self) -> Result<()> {
        if self.x_data.is_none() {
            return Err(SklearsError::InvalidInput(
                "No data provided for precomputation".to_string(),
            ));
        }

        let x = self.x_data.as_ref().unwrap().clone();
        let (rows, cols) = self.dimensions;

        // Sequential computation for now (to avoid borrowing issues)
        for i in 0..rows {
            for j in 0..cols {
                let k_val = if i <= j {
                    // Only compute upper triangle for symmetric matrices
                    self.kernel.compute(x.row(i), x.row(j))
                } else {
                    // Use symmetry
                    self.get_raw(j, i)?
                };

                self.set_raw(i, j, k_val)?;
                if i != j {
                    self.set_raw(j, i, k_val)?; // Symmetric
                }
            }
        }

        self.flush()?;
        Ok(())
    }

    /// Get kernel value at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Result<Float> {
        if i >= self.dimensions.0 || j >= self.dimensions.1 {
            return Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for matrix of size ({}, {})",
                i, j, self.dimensions.0, self.dimensions.1
            )));
        }

        // Check cache first
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(&value) = cache.get(&(i, j)) {
                return Ok(value);
            }
        }

        // Try to get from memory-mapped file
        let value = if let Ok(precomputed) = self.get_raw(i, j) {
            precomputed
        } else if let Some(ref x) = self.x_data {
            // Compute on-demand
            let val = self.kernel.compute(x.row(i), x.row(j));

            // Cache the computed value
            if let Ok(mut cache) = self.cache.lock() {
                cache.put((i, j), val);
                if i != j {
                    cache.put((j, i), val); // Symmetric
                }
            }

            val
        } else {
            return Err(SklearsError::InvalidInput(
                "No precomputed data or original data available".to_string(),
            ));
        };

        Ok(value)
    }

    /// Set kernel value at position (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: Float) -> Result<()> {
        if i >= self.dimensions.0 || j >= self.dimensions.1 {
            return Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds",
                i, j
            )));
        }

        self.set_raw(i, j, value)?;

        // Update cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.put((i, j), value);
        }

        Ok(())
    }

    /// Get a block of kernel values
    pub fn get_block(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<Array2<Float>> {
        if row_end > self.dimensions.0 || col_end > self.dimensions.1 {
            return Err(SklearsError::InvalidInput(
                "Block indices out of bounds".to_string(),
            ));
        }

        let block_rows = row_end - row_start;
        let block_cols = col_end - col_start;
        let mut block = Array2::zeros((block_rows, block_cols));

        for i in 0..block_rows {
            for j in 0..block_cols {
                block[[i, j]] = self.get(row_start + i, col_start + j)?;
            }
        }

        Ok(block)
    }

    /// Set a block of kernel values
    pub fn set_block(
        &mut self,
        row_start: usize,
        col_start: usize,
        block: &Array2<Float>,
    ) -> Result<()> {
        let (block_rows, block_cols) = block.dim();

        if row_start + block_rows > self.dimensions.0 || col_start + block_cols > self.dimensions.1
        {
            return Err(SklearsError::InvalidInput(
                "Block would exceed matrix bounds".to_string(),
            ));
        }

        for i in 0..block_rows {
            for j in 0..block_cols {
                self.set(row_start + i, col_start + j, block[[i, j]])?;
            }
        }

        Ok(())
    }

    /// Get raw value from memory-mapped file
    fn get_raw(&self, i: usize, j: usize) -> Result<Float> {
        let offset = (i * self.dimensions.1 + j) * std::mem::size_of::<Float>();

        if let Some(ref mmap) = self.mmap {
            let bytes = &mmap[offset..offset + std::mem::size_of::<Float>()];
            let value =
                Float::from_le_bytes(bytes.try_into().map_err(|_| {
                    SklearsError::InvalidInput("Failed to read raw value".to_string())
                })?);
            Ok(value)
        } else if let Some(ref mmap_mut) = self.mmap_mut {
            let bytes = &mmap_mut[offset..offset + std::mem::size_of::<Float>()];
            let value =
                Float::from_le_bytes(bytes.try_into().map_err(|_| {
                    SklearsError::InvalidInput("Failed to read raw value".to_string())
                })?);
            Ok(value)
        } else {
            Err(SklearsError::InvalidInput(
                "No memory mapping available".to_string(),
            ))
        }
    }

    /// Set raw value in memory-mapped file
    fn set_raw(&mut self, i: usize, j: usize, value: Float) -> Result<()> {
        let offset = (i * self.dimensions.1 + j) * std::mem::size_of::<Float>();

        if let Some(ref mut mmap_mut) = self.mmap_mut {
            let bytes = value.to_le_bytes();
            mmap_mut[offset..offset + std::mem::size_of::<Float>()].copy_from_slice(&bytes);
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(
                "Memory mapping not available for writing".to_string(),
            ))
        }
    }

    /// Write header to file
    fn write_header(&mut self, header: &KernelMatrixHeader) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                header as *const KernelMatrixHeader as *const u8,
                std::mem::size_of::<KernelMatrixHeader>(),
            )
        };

        self.file.write_all(header_bytes)?;

        Ok(())
    }

    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mmap_mut) = self.mmap_mut {
            mmap_mut.flush()?;
        }
        Ok(())
    }

    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    /// Get cache hit rate statistics
    pub fn cache_stats(&self) -> Result<CacheStats> {
        if let Ok(cache) = self.cache.lock() {
            Ok(CacheStats {
                size: cache.len(),
                capacity: cache.cap(),
                hit_rate: 0.0, // Would need to track hits/misses for accurate rate
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Failed to access cache".to_string(),
            ))
        }
    }
}

/// Header structure for the memory-mapped kernel matrix file
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct KernelMatrixHeader {
    rows: u64,
    cols: u64,
    element_size: u32,
    compression_level: u8,
    version: u8,
}

/// Cache statistics
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f64,
}

/// Sparse memory-mapped kernel matrix for very large, sparse datasets
pub struct SparseMemoryMappedKernelMatrix {
    dense_matrix: MemoryMappedKernelMatrix,
    sparse_indices: HashMap<(usize, usize), usize>,
    sparsity_threshold: Float,
}

impl SparseMemoryMappedKernelMatrix {
    /// Create a new sparse memory-mapped kernel matrix
    pub fn new(
        kernel: Box<dyn Kernel>,
        dimensions: (usize, usize),
        config: MemoryMappedKernelConfig,
        sparsity_threshold: Float,
    ) -> Result<Self> {
        let dense_matrix = MemoryMappedKernelMatrix::new(kernel, dimensions, config)?;

        Ok(Self {
            dense_matrix,
            sparse_indices: HashMap::new(),
            sparsity_threshold,
        })
    }

    /// Get kernel value, using sparsity
    pub fn get(&self, i: usize, j: usize) -> Result<Float> {
        if let Some(&sparse_idx) = self.sparse_indices.get(&(i, j)) {
            // This is a stored sparse value
            self.dense_matrix.get_raw(
                sparse_idx / self.dense_matrix.dimensions.1,
                sparse_idx % self.dense_matrix.dimensions.1,
            )
        } else {
            // Either dense value or implicit zero
            let value = self.dense_matrix.get(i, j)?;
            if value.abs() < self.sparsity_threshold {
                Ok(0.0)
            } else {
                Ok(value)
            }
        }
    }

    /// Set kernel value with sparsity consideration
    pub fn set(&mut self, i: usize, j: usize, value: Float) -> Result<()> {
        if value.abs() < self.sparsity_threshold {
            // Remove from sparse indices if it exists
            self.sparse_indices.remove(&(i, j));
            Ok(())
        } else {
            self.dense_matrix.set(i, j, value)
        }
    }

    /// Get sparsity ratio
    pub fn sparsity_ratio(&self) -> f64 {
        let total_elements = self.dense_matrix.dimensions.0 * self.dense_matrix.dimensions.1;
        let sparse_elements = self.sparse_indices.len();
        1.0 - (sparse_elements as f64 / total_elements as f64)
    }
}

/// Simple LRU Cache implementation
struct LRUCache<K, V> {
    map: HashMap<K, V>,
    capacity: usize,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> LRUCache<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity {
            // Simple eviction - remove a random entry
            if let Some(key_to_remove) = self.map.keys().next().cloned() {
                self.map.remove(&key_to_remove);
            }
        }
        self.map.insert(key, value);
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn cap(&self) -> usize {
        self.capacity
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::LinearKernel;

    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_mapped_kernel_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut config = MemoryMappedKernelConfig::default();
        config.file_path = temp_file.path().to_path_buf();

        let kernel = Box::new(LinearKernel);
        let matrix = MemoryMappedKernelMatrix::new(kernel, (100, 100), config);
        assert!(matrix.is_ok());
    }

    #[test]
    fn test_kernel_value_set_get() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut config = MemoryMappedKernelConfig::default();
        config.file_path = temp_file.path().to_path_buf();

        let kernel = Box::new(LinearKernel);
        let mut matrix = MemoryMappedKernelMatrix::new(kernel, (10, 10), config).unwrap();

        matrix.set(0, 0, 1.5).unwrap();
        assert_eq!(matrix.get(0, 0).unwrap(), 1.5);
    }

    #[test]
    fn test_sparse_memory_mapped_kernel() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut config = MemoryMappedKernelConfig::default();
        config.file_path = temp_file.path().to_path_buf();

        let kernel = Box::new(LinearKernel);
        let sparse_matrix = SparseMemoryMappedKernelMatrix::new(kernel, (10, 10), config, 0.1);
        assert!(sparse_matrix.is_ok());
    }
}
