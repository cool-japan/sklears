//! Performance Optimization Utilities
//!
//! This module provides performance optimization techniques including:
//! - Cache-friendly matrix layouts and operations
//! - Memory-aligned data structures
//! - Prefetching for large matrix operations
//! - SIMD-friendly data arrangements
//! - Memory pooling and arena allocation

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

/// Cache line size for modern CPUs (typically 64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Alignment for SIMD operations (32 bytes for AVX)
const SIMD_ALIGNMENT: usize = 32;

/// Cache-friendly matrix configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Tile size for matrix operations
    pub tile_size: usize,

    /// Prefetch distance
    pub prefetch_distance: usize,

    /// Use memory alignment
    pub use_alignment: bool,

    /// Block size for blocked algorithms
    pub block_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            tile_size: 64,
            prefetch_distance: 8,
            use_alignment: true,
            block_size: 128,
        }
    }
}

/// Cache-friendly matrix wrapper
pub struct CacheFriendlyMatrix {
    data: Vec<Float>,
    nrows: usize,
    ncols: usize,
    layout: MatrixLayout,
    config: CacheConfig,
}

/// Matrix storage layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    /// Row-major layout (C-style)
    RowMajor,

    /// Column-major layout (Fortran-style)
    ColumnMajor,

    /// Blocked/tiled layout for cache efficiency
    Blocked,
}

impl CacheFriendlyMatrix {
    /// Create new cache-friendly matrix from Array2
    pub fn new(array: Array2<Float>, layout: MatrixLayout, config: CacheConfig) -> Self {
        let (nrows, ncols) = array.dim();

        let data = match layout {
            MatrixLayout::RowMajor => array.into_raw_vec(),
            MatrixLayout::ColumnMajor => {
                // Transpose to column-major
                let mut data = Vec::with_capacity(nrows * ncols);
                for j in 0..ncols {
                    for i in 0..nrows {
                        data.push(array[[i, j]]);
                    }
                }
                data
            }
            MatrixLayout::Blocked => {
                // Block layout for cache efficiency
                Self::convert_to_blocked(&array, config.block_size)
            }
        };

        Self {
            data,
            nrows,
            ncols,
            layout,
            config,
        }
    }

    /// Convert to blocked layout
    fn convert_to_blocked(array: &Array2<Float>, block_size: usize) -> Vec<Float> {
        let (nrows, ncols) = array.dim();
        let mut blocked = Vec::with_capacity(nrows * ncols);

        let n_row_blocks = (nrows + block_size - 1) / block_size;
        let n_col_blocks = (ncols + block_size - 1) / block_size;

        for block_i in 0..n_row_blocks {
            for block_j in 0..n_col_blocks {
                let row_start = block_i * block_size;
                let row_end = (row_start + block_size).min(nrows);
                let col_start = block_j * block_size;
                let col_end = (col_start + block_size).min(ncols);

                for i in row_start..row_end {
                    for j in col_start..col_end {
                        blocked.push(array[[i, j]]);
                    }
                }

                // Pad to full block size
                let block_elements = (row_end - row_start) * (col_end - col_start);
                let full_block = block_size * block_size;
                for _ in block_elements..full_block {
                    blocked.push(0.0);
                }
            }
        }

        blocked
    }

    /// Get element at position
    pub fn get(&self, i: usize, j: usize) -> Float {
        match self.layout {
            MatrixLayout::RowMajor => self.data[i * self.ncols + j],
            MatrixLayout::ColumnMajor => self.data[j * self.nrows + i],
            MatrixLayout::Blocked => {
                let block_size = self.config.block_size;
                let block_i = i / block_size;
                let block_j = j / block_size;
                let local_i = i % block_size;
                let local_j = j % block_size;

                let n_col_blocks = (self.ncols + block_size - 1) / block_size;
                let block_idx = block_i * n_col_blocks + block_j;
                let block_start = block_idx * block_size * block_size;
                let offset = local_i * block_size + local_j;

                self.data[block_start + offset]
            }
        }
    }

    /// Convert back to Array2
    pub fn to_array(&self) -> Array2<Float> {
        let mut array = Array2::zeros((self.nrows, self.ncols));

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                array[[i, j]] = self.get(i, j);
            }
        }

        array
    }

    /// Cache-friendly matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ncols != other.nrows {
            return Err(SklearsError::InvalidInput(format!(
                "Dimension mismatch: {} x {} cannot multiply {} x {}",
                self.nrows, self.ncols, other.nrows, other.ncols
            )));
        }

        let mut result_data = vec![0.0; self.nrows * other.ncols];

        // Tiled matrix multiplication for cache efficiency
        let tile_size = self.config.tile_size;

        for i_tile in (0..self.nrows).step_by(tile_size) {
            for j_tile in (0..other.ncols).step_by(tile_size) {
                for k_tile in (0..self.ncols).step_by(tile_size) {
                    // Process tile
                    let i_end = (i_tile + tile_size).min(self.nrows);
                    let j_end = (j_tile + tile_size).min(other.ncols);
                    let k_end = (k_tile + tile_size).min(self.ncols);

                    for i in i_tile..i_end {
                        for j in j_tile..j_end {
                            let mut sum = 0.0;
                            for k in k_tile..k_end {
                                sum += self.get(i, k) * other.get(k, j);
                            }
                            let idx = i * other.ncols + j;
                            result_data[idx] += sum;
                        }
                    }
                }
            }
        }

        Ok(Self {
            data: result_data,
            nrows: self.nrows,
            ncols: other.ncols,
            layout: MatrixLayout::RowMajor,
            config: self.config.clone(),
        })
    }

    /// Get matrix dimensions
    pub fn dim(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

/// Memory pool for matrix allocations
pub struct MemoryPool {
    /// Pool of available buffers
    buffers: Vec<Vec<Float>>,

    /// Buffer size
    buffer_size: usize,

    /// Maximum pool size
    max_pool_size: usize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(buffer_size: usize, max_pool_size: usize) -> Self {
        Self {
            buffers: Vec::new(),
            buffer_size,
            max_pool_size,
        }
    }

    /// Acquire buffer from pool
    pub fn acquire(&mut self) -> Vec<Float> {
        self.buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.buffer_size))
    }

    /// Return buffer to pool
    pub fn release(&mut self, mut buffer: Vec<Float>) {
        if self.buffers.len() < self.max_pool_size {
            buffer.clear();
            self.buffers.push(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Clear pool
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            available_buffers: self.buffers.len(),
            buffer_size: self.buffer_size,
            total_allocated_bytes: self.buffers.len()
                * self.buffer_size
                * std::mem::size_of::<Float>(),
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub available_buffers: usize,
    pub buffer_size: usize,
    pub total_allocated_bytes: usize,
}

/// Aligned memory allocator for SIMD operations
pub struct AlignedAllocator;

impl AlignedAllocator {
    /// Allocate aligned memory
    pub fn allocate(size: usize, alignment: usize) -> Result<*mut Float> {
        if size == 0 {
            return Ok(ptr::null_mut());
        }

        unsafe {
            let layout = Layout::from_size_align(size * std::mem::size_of::<Float>(), alignment)
                .map_err(|e| SklearsError::InvalidInput(format!("Invalid layout: {}", e)))?;

            let ptr = alloc(layout) as *mut Float;

            if ptr.is_null() {
                return Err(SklearsError::InvalidInput("Allocation failed".to_string()));
            }

            Ok(ptr)
        }
    }

    /// Deallocate aligned memory
    pub unsafe fn deallocate(ptr: *mut Float, size: usize, alignment: usize) {
        if ptr.is_null() || size == 0 {
            return;
        }

        let layout =
            Layout::from_size_align_unchecked(size * std::mem::size_of::<Float>(), alignment);

        dealloc(ptr as *mut u8, layout);
    }

    /// Create aligned vector
    pub fn aligned_vec(size: usize, alignment: usize) -> Result<Vec<Float>> {
        let ptr = Self::allocate(size, alignment)?;

        unsafe {
            // Initialize to zero
            ptr::write_bytes(ptr, 0, size);

            // Create Vec from raw parts
            // Note: This is a simplified version - real implementation would need custom Drop
            Ok(Vec::from_raw_parts(ptr, size, size))
        }
    }
}

/// Prefetch hint for cache optimization
#[inline(always)]
pub fn prefetch_hint<T>(_addr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            #[cfg(target_feature = "sse")]
            {
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                    _addr as *const i8,
                );
            }
        }
    }
    // For other architectures, this is a no-op
}

/// Performance profiler for decomposition operations
pub struct PerformanceProfiler {
    measurements: Vec<Measurement>,
}

#[derive(Debug, Clone)]
pub struct Measurement {
    pub name: String,
    pub duration_ns: u128,
    pub memory_bytes: usize,
}

impl PerformanceProfiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    /// Start timing an operation
    pub fn start(&mut self, name: &str) -> TimingGuard {
        TimingGuard {
            name: name.to_string(),
            start: std::time::Instant::now(),
        }
    }

    /// Record measurement
    pub fn record(&mut self, name: String, duration_ns: u128, memory_bytes: usize) {
        self.measurements.push(Measurement {
            name,
            duration_ns,
            memory_bytes,
        });
    }

    /// Get measurements
    pub fn measurements(&self) -> &[Measurement] {
        &self.measurements
    }

    /// Clear measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }

    /// Generate report
    pub fn report(&self) -> String {
        let mut report = String::from("Performance Report:\n");
        report.push_str("===================\n\n");

        for m in &self.measurements {
            report.push_str(&format!(
                "{}: {:.2} ms, {} bytes\n",
                m.name,
                m.duration_ns as f64 / 1_000_000.0,
                m.memory_bytes
            ));
        }

        let total_time: u128 = self.measurements.iter().map(|m| m.duration_ns).sum();
        let total_memory: usize = self.measurements.iter().map(|m| m.memory_bytes).sum();

        report.push_str(&format!(
            "\nTotal: {:.2} ms, {} bytes\n",
            total_time as f64 / 1_000_000.0,
            total_memory
        ));

        report
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing guard for automatic measurement
pub struct TimingGuard {
    name: String,
    start: std::time::Instant,
}

impl TimingGuard {
    /// Finish timing and return duration
    pub fn finish(self) -> (String, u128) {
        let duration = self.start.elapsed().as_nanos();
        (self.name, duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_friendly_matrix_creation() {
        let array =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let cache_matrix = CacheFriendlyMatrix::new(
            array.clone(),
            MatrixLayout::RowMajor,
            CacheConfig::default(),
        );

        assert_eq!(cache_matrix.dim(), (3, 3));
        assert_eq!(cache_matrix.get(0, 0), 1.0);
        assert_eq!(cache_matrix.get(2, 2), 9.0);
    }

    #[test]
    fn test_cache_friendly_matmul() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let cache_a = CacheFriendlyMatrix::new(a, MatrixLayout::RowMajor, CacheConfig::default());
        let cache_b = CacheFriendlyMatrix::new(b, MatrixLayout::RowMajor, CacheConfig::default());

        let result = cache_a.matmul(&cache_b).unwrap();
        assert_eq!(result.dim(), (2, 2));

        // Expected result: [[58, 64], [139, 154]]
        assert!((result.get(0, 0) - 58.0).abs() < 1e-6);
        assert!((result.get(0, 1) - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(100, 5);

        // Acquire and release buffers
        let buf1 = pool.acquire();
        let buf2 = pool.acquire();

        pool.release(buf1);
        pool.release(buf2);

        let stats = pool.stats();
        assert_eq!(stats.available_buffers, 2);

        // Acquire again should reuse buffer
        let _buf3 = pool.acquire();
        let stats = pool.stats();
        assert_eq!(stats.available_buffers, 1);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();

        let guard = profiler.start("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let (name, duration) = guard.finish();

        profiler.record(name, duration, 1024);

        let measurements = profiler.measurements();
        assert_eq!(measurements.len(), 1);
        assert!(measurements[0].duration_ns > 0);
    }

    #[test]
    fn test_column_major_layout() {
        let array = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let cache_matrix = CacheFriendlyMatrix::new(
            array.clone(),
            MatrixLayout::ColumnMajor,
            CacheConfig::default(),
        );

        assert_eq!(cache_matrix.get(0, 0), 1.0);
        assert_eq!(cache_matrix.get(0, 1), 2.0);
        assert_eq!(cache_matrix.get(1, 0), 3.0);
        assert_eq!(cache_matrix.get(1, 1), 4.0);
    }

    #[test]
    fn test_profiler_report() {
        let mut profiler = PerformanceProfiler::new();

        profiler.record("op1".to_string(), 1_000_000, 1024);
        profiler.record("op2".to_string(), 2_000_000, 2048);

        let report = profiler.report();
        assert!(report.contains("op1"));
        assert!(report.contains("op2"));
        assert!(report.contains("Total"));
    }
}
