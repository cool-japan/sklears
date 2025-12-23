//! Cache-friendly feature layouts for improved performance
//!
//! This module provides optimized data layouts and access patterns to maximize
//! CPU cache utilization during kernel approximation computations.
//!
//! # Key Optimizations
//!
//! - **Blocked Matrix Operations**: Tiles large matrices into cache-friendly blocks
//! - **Data Alignment**: Ensures proper alignment for SIMD operations
//! - **Memory Layouts**: Provides both row-major and column-major layouts
//! - **Prefetching**: Hints for CPU prefetching to reduce cache misses
//! - **Structure of Arrays (SoA)**: Optimized layout for vectorized operations

use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sklears_core::error::SklearsError;
use std::alloc::{alloc, dealloc, Layout};

/// Memory layout strategy for feature matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryLayout {
    /// Row-major layout (C-style) - better for row-wise operations
    #[default]
    RowMajor,
    /// Column-major layout (Fortran-style) - better for column-wise operations
    ColumnMajor,
    /// Blocked layout with specified tile size - optimal for cache locality
    Blocked { tile_size: usize },
    /// Structure of Arrays - separate arrays for each feature
    StructureOfArrays,
}

/// Configuration for cache optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Preferred memory layout
    pub layout: MemoryLayout,
    /// L1 cache size in bytes (default: 32KB)
    pub l1_cache_size: usize,
    /// L2 cache size in bytes (default: 256KB)
    pub l2_cache_size: usize,
    /// L3 cache size in bytes (default: 8MB)
    pub l3_cache_size: usize,
    /// Cache line size in bytes (default: 64)
    pub cache_line_size: usize,
    /// Enable prefetching hints
    pub enable_prefetch: bool,
    /// Alignment requirement in bytes (default: 64 for AVX-512)
    pub alignment: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            layout: MemoryLayout::default(),
            l1_cache_size: 32 * 1024,       // 32 KB
            l2_cache_size: 256 * 1024,      // 256 KB
            l3_cache_size: 8 * 1024 * 1024, // 8 MB
            cache_line_size: 64,
            enable_prefetch: true,
            alignment: 64, // AVX-512 alignment
        }
    }
}

impl CacheConfig {
    /// Calculate optimal tile size based on cache sizes
    pub fn optimal_tile_size(&self) -> usize {
        // Use L2 cache size to determine tile size
        // Assume we need space for 3 tiles (A, B, C matrices)
        let bytes_per_element = std::mem::size_of::<f64>();
        let available_cache = self.l2_cache_size / 3;
        let elements_per_tile = available_cache / bytes_per_element;

        // Calculate square tile dimension
        let tile_dim = (elements_per_tile as f64).sqrt() as usize;

        // Round down to nearest multiple of cache line size
        let elements_per_line = self.cache_line_size / bytes_per_element;
        (tile_dim / elements_per_line) * elements_per_line
    }

    /// Calculate optimal block size for vector operations
    pub fn optimal_block_size(&self) -> usize {
        // Target L1 cache to keep working set in fastest cache
        let bytes_per_element = std::mem::size_of::<f64>();
        let available_cache = self.l1_cache_size / 2; // Keep some room for other data
        available_cache / bytes_per_element
    }
}

/// Cache-friendly feature matrix wrapper
pub struct CacheFriendlyMatrix {
    /// The underlying data
    data: Vec<f64>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
    /// Memory layout
    layout: MemoryLayout,
    /// Cache configuration
    config: CacheConfig,
}

impl CacheFriendlyMatrix {
    /// Create a new cache-friendly matrix from an Array2
    pub fn from_array(array: &Array2<f64>, config: CacheConfig) -> Result<Self, SklearsError> {
        let (n_rows, n_cols) = array.dim();
        let layout = config.layout;

        let data = match layout {
            MemoryLayout::RowMajor => array.iter().copied().collect(),
            MemoryLayout::ColumnMajor => {
                let mut data = Vec::with_capacity(n_rows * n_cols);
                for col in 0..n_cols {
                    for row in 0..n_rows {
                        data.push(array[[row, col]]);
                    }
                }
                data
            }
            MemoryLayout::Blocked { tile_size } => Self::convert_to_blocked(array, tile_size)?,
            MemoryLayout::StructureOfArrays => array.iter().copied().collect(),
        };

        Ok(Self {
            data,
            n_rows,
            n_cols,
            layout,
            config,
        })
    }

    /// Convert array to blocked layout
    fn convert_to_blocked(array: &Array2<f64>, tile_size: usize) -> Result<Vec<f64>, SklearsError> {
        let (n_rows, n_cols) = array.dim();
        let mut data = vec![0.0; n_rows * n_cols];

        let n_row_blocks = (n_rows + tile_size - 1) / tile_size;
        let n_col_blocks = (n_cols + tile_size - 1) / tile_size;

        let mut offset = 0;
        for block_row in 0..n_row_blocks {
            for block_col in 0..n_col_blocks {
                let row_start = block_row * tile_size;
                let row_end = (row_start + tile_size).min(n_rows);
                let col_start = block_col * tile_size;
                let col_end = (col_start + tile_size).min(n_cols);

                for row in row_start..row_end {
                    for col in col_start..col_end {
                        data[offset] = array[[row, col]];
                        offset += 1;
                    }
                }
            }
        }

        Ok(data)
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<f64, SklearsError> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(SklearsError::InvalidInput(
                "Index out of bounds".to_string(),
            ));
        }

        let idx = match self.layout {
            MemoryLayout::RowMajor => row * self.n_cols + col,
            MemoryLayout::ColumnMajor => col * self.n_rows + row,
            MemoryLayout::Blocked { tile_size } => {
                let block_row = row / tile_size;
                let block_col = col / tile_size;
                let in_block_row = row % tile_size;
                let in_block_col = col % tile_size;

                let n_col_blocks = (self.n_cols + tile_size - 1) / tile_size;
                let block_idx = block_row * n_col_blocks + block_col;
                let block_offset = block_idx * tile_size * tile_size;

                block_offset + in_block_row * tile_size + in_block_col
            }
            MemoryLayout::StructureOfArrays => row * self.n_cols + col,
        };

        Ok(self.data[idx])
    }

    /// Convert back to Array2
    pub fn to_array(&self) -> Result<Array2<f64>, SklearsError> {
        let mut array = Array2::zeros((self.n_rows, self.n_cols));

        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                array[[row, col]] = self.get(row, col)?;
            }
        }

        Ok(array)
    }

    /// Perform cache-friendly matrix-vector multiplication
    pub fn dot_vector(&self, vector: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        if vector.len() != self.n_cols {
            return Err(SklearsError::InvalidInput(
                "Vector length must match number of columns".to_string(),
            ));
        }

        let mut result = Array1::zeros(self.n_rows);

        match self.layout {
            MemoryLayout::RowMajor => {
                // Row-major is naturally cache-friendly for this operation
                for row in 0..self.n_rows {
                    let mut sum = 0.0;
                    for col in 0..self.n_cols {
                        sum += self.data[row * self.n_cols + col] * vector[col];
                    }
                    result[row] = sum;
                }
            }
            MemoryLayout::ColumnMajor => {
                // Use column-wise accumulation
                for col in 0..self.n_cols {
                    let v_col = vector[col];
                    for row in 0..self.n_rows {
                        result[row] += self.data[col * self.n_rows + row] * v_col;
                    }
                }
            }
            MemoryLayout::Blocked { tile_size } => {
                // Process in blocks for cache efficiency
                for row_block in (0..self.n_rows).step_by(tile_size) {
                    let row_end = (row_block + tile_size).min(self.n_rows);

                    for col_block in (0..self.n_cols).step_by(tile_size) {
                        let col_end = (col_block + tile_size).min(self.n_cols);

                        for row in row_block..row_end {
                            let mut sum = 0.0;
                            for col in col_block..col_end {
                                sum += self.get(row, col)? * vector[col];
                            }
                            result[row] += sum;
                        }
                    }
                }
            }
            MemoryLayout::StructureOfArrays => {
                // Similar to row-major
                for row in 0..self.n_rows {
                    let mut sum = 0.0;
                    for col in 0..self.n_cols {
                        sum += self.data[row * self.n_cols + col] * vector[col];
                    }
                    result[row] = sum;
                }
            }
        }

        Ok(result)
    }

    /// Get dimensions
    pub fn dim(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }
}

/// Aligned memory allocator for SIMD operations
pub struct AlignedBuffer {
    ptr: *mut f64,
    len: usize,
    alignment: usize,
}

impl AlignedBuffer {
    /// Create a new aligned buffer
    pub fn new(len: usize, alignment: usize) -> Result<Self, SklearsError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(SklearsError::InvalidInput(
                "Alignment must be a power of 2".to_string(),
            ));
        }

        let layout = Layout::from_size_align(len * std::mem::size_of::<f64>(), alignment)
            .map_err(|e| SklearsError::InvalidInput(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe { alloc(layout) as *mut f64 };

        if ptr.is_null() {
            return Err(SklearsError::InvalidInput(
                "Failed to allocate aligned memory".to_string(),
            ));
        }

        // Initialize to zero
        unsafe {
            std::ptr::write_bytes(ptr, 0, len);
        }

        Ok(Self {
            ptr,
            len,
            alignment,
        })
    }

    /// Get a slice view of the buffer
    pub fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get a mutable slice view of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert to Array1
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(self.as_slice().to_vec())
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let layout =
                Layout::from_size_align(self.len * std::mem::size_of::<f64>(), self.alignment)
                    .expect("Invalid layout in drop");
            unsafe {
                dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

/// Cache-aware feature transformation strategy
pub trait CacheAwareTransform {
    /// Transform features with cache optimization
    fn transform_cached(
        &self,
        features: &Array2<f64>,
        config: &CacheConfig,
    ) -> Result<Array2<f64>, SklearsError>;
}

/// Utility functions for cache optimization
pub mod utils {
    use super::*;

    /// Prefetch data at the given address (hint to CPU)
    #[inline(always)]
    pub fn prefetch_read(addr: *const f64) {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "sse")]
            unsafe {
                use std::arch::x86_64::_mm_prefetch;
                use std::arch::x86_64::_MM_HINT_T0;
                _mm_prefetch(addr as *const i8, _MM_HINT_T0);
            }
        }

        // For non-x86 or without SSE, this is a no-op
        let _ = addr;
    }

    /// Transpose matrix with cache blocking
    pub fn transpose_blocked(
        matrix: &Array2<f64>,
        tile_size: usize,
    ) -> Result<Array2<f64>, SklearsError> {
        let (n_rows, n_cols) = matrix.dim();
        let mut result = Array2::zeros((n_cols, n_rows));

        for row_block in (0..n_rows).step_by(tile_size) {
            for col_block in (0..n_cols).step_by(tile_size) {
                let row_end = (row_block + tile_size).min(n_rows);
                let col_end = (col_block + tile_size).min(n_cols);

                for row in row_block..row_end {
                    for col in col_block..col_end {
                        result[[col, row]] = matrix[[row, col]];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Calculate optimal number of threads based on data size and cache
    pub fn optimal_thread_count(
        n_samples: usize,
        n_features: usize,
        config: &CacheConfig,
    ) -> usize {
        let data_size = n_samples * n_features * std::mem::size_of::<f64>();
        let num_cpus = num_cpus::get();

        // If data fits in L3 cache, use fewer threads
        if data_size <= config.l3_cache_size {
            (num_cpus / 2).max(1)
        } else {
            num_cpus
        }
    }

    /// Align size to cache line boundary
    pub fn align_to_cache_line(size: usize, cache_line_size: usize) -> usize {
        ((size + cache_line_size - 1) / cache_line_size) * cache_line_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.cache_line_size, 64);
        assert!(config.enable_prefetch);
    }

    #[test]
    fn test_optimal_tile_size() {
        let config = CacheConfig::default();
        let tile_size = config.optimal_tile_size();
        assert!(tile_size > 0);
        assert!(tile_size < 1024); // Reasonable size
    }

    #[test]
    fn test_cache_friendly_matrix_row_major() {
        let array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let config = CacheConfig {
            layout: MemoryLayout::RowMajor,
            ..Default::default()
        };

        let matrix = CacheFriendlyMatrix::from_array(&array, config).unwrap();
        assert_eq!(matrix.dim(), (2, 3));
        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(1, 2).unwrap(), 6.0);
    }

    #[test]
    fn test_cache_friendly_matrix_column_major() {
        let array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let config = CacheConfig {
            layout: MemoryLayout::ColumnMajor,
            ..Default::default()
        };

        let matrix = CacheFriendlyMatrix::from_array(&array, config).unwrap();
        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(1, 2).unwrap(), 6.0);
    }

    #[test]
    fn test_cache_friendly_matrix_blocked() {
        let array = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];
        let config = CacheConfig {
            layout: MemoryLayout::Blocked { tile_size: 2 },
            ..Default::default()
        };

        let matrix = CacheFriendlyMatrix::from_array(&array, config).unwrap();
        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(3, 3).unwrap(), 16.0);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let array = array![[1.0, 2.0], [3.0, 4.0]];
        let vector = array![1.0, 2.0];
        let config = CacheConfig::default();

        let matrix = CacheFriendlyMatrix::from_array(&array, config).unwrap();
        let result = matrix.dot_vector(&vector).unwrap();

        assert_eq!(result[0], 5.0); // 1*1 + 2*2
        assert_eq!(result[1], 11.0); // 3*1 + 4*2
    }

    #[test]
    fn test_aligned_buffer() {
        let buffer = AlignedBuffer::new(10, 64).unwrap();
        assert_eq!(buffer.len(), 10);
        assert!(!buffer.is_empty());

        let slice = buffer.as_slice();
        assert_eq!(slice.len(), 10);
    }

    #[test]
    fn test_transpose_blocked() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let transposed = utils::transpose_blocked(&matrix, 2).unwrap();

        assert_eq!(transposed.dim(), (3, 2));
        assert_eq!(transposed[[0, 0]], 1.0);
        assert_eq!(transposed[[2, 1]], 6.0);
    }

    #[test]
    fn test_optimal_thread_count() {
        let config = CacheConfig::default();
        let threads = utils::optimal_thread_count(1000, 100, &config);
        assert!(threads > 0);
        assert!(threads <= num_cpus::get());
    }

    #[test]
    fn test_align_to_cache_line() {
        let aligned = utils::align_to_cache_line(100, 64);
        assert_eq!(aligned, 128);
        assert_eq!(aligned % 64, 0);
    }
}
