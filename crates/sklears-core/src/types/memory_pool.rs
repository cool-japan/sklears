/// Memory pool and efficient memory management for machine learning operations
///
/// This module provides memory pooling and efficient memory management utilities
/// specifically designed for machine learning workloads. It includes features for
/// reducing memory allocations, zero-copy operations, and GPU memory management.

use crate::types::{FloatBounds, Numeric};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Thread-safe memory pool for reducing allocations in ML operations
pub struct MemoryPool<T> {
    buffers: Arc<Mutex<VecDeque<Vec<T>>>>,
    max_buffers: usize,
    buffer_size: usize,
    _marker: PhantomData<T>,
}

impl<T> MemoryPool<T>
where
    T: Clone + Default,
{
    /// Create a new memory pool with specified capacity and buffer size
    pub fn new(max_buffers: usize, buffer_size: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(VecDeque::new())),
            max_buffers,
            buffer_size,
            _marker: PhantomData,
        }
    }

    /// Get a buffer from the pool, or allocate a new one if none available
    pub fn get_buffer(&self) -> PooledBuffer<T> {
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let buffer = if let Some(mut buf) = buffers.pop_front() {
            buf.clear();
            buf.resize(self.buffer_size, T::default());
            buf
        } else {
            vec![T::default(); self.buffer_size]
        };

        PooledBuffer {
            buffer,
            pool: Arc::clone(&self.buffers),
            max_buffers: self.max_buffers,
        }
    }

    /// Get the current number of pooled buffers
    pub fn available_buffers(&self) -> usize {
        self.buffers.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        self.buffers.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// A buffer that automatically returns to the pool when dropped
pub struct PooledBuffer<T> {
    buffer: Vec<T>,
    pool: Arc<Mutex<VecDeque<Vec<T>>>>,
    max_buffers: usize,
}

impl<T> PooledBuffer<T> {
    /// Get a reference to the underlying buffer
    pub fn as_slice(&self) -> &[T] {
        &self.buffer
    }

    /// Get a mutable reference to the underlying buffer
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buffer
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Resize the buffer (may require reallocation)
    pub fn resize(&mut self, new_size: usize, value: T)
    where
        T: Clone,
    {
        self.buffer.resize(new_size, value);
    }
}

impl<T> std::ops::Deref for PooledBuffer<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T> std::ops::DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl<T> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < self.max_buffers {
            pool.push_back(std::mem::take(&mut self.buffer));
        }
    }
}

/// Zero-copy wrapper for borrowed array data
pub struct ZeroCopyArray<'a, T> {
    data: &'a [T],
    shape: (usize, usize),
}

impl<'a, T> ZeroCopyArray<'a, T> {
    /// Create a new zero-copy array from borrowed data
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Data length {} doesn't match shape ({}, {})",
                data.len(),
                rows,
                cols
            ));
        }

        Ok(Self {
            data,
            shape: (rows, cols),
        })
    }

    /// Get the shape of the array
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Get a row as a slice
    pub fn row(&self, idx: usize) -> Result<&[T], String> {
        if idx >= self.nrows() {
            return Err(format!("Row index {} out of bounds", idx));
        }

        let start = idx * self.ncols();
        let end = start + self.ncols();
        Ok(&self.data[start..end])
    }

    /// Get an element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<&T, String> {
        if row >= self.nrows() || col >= self.ncols() {
            return Err(format!("Index ({}, {}) out of bounds", row, col));
        }

        Ok(&self.data[row * self.ncols() + col])
    }

    /// Create a view of a submatrix
    pub fn submatrix(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Result<ZeroCopySubArray<'a, T>, String> {
        if row_start >= row_end || col_start >= col_end {
            return Err("Invalid submatrix range".to_string());
        }

        if row_end > self.nrows() || col_end > self.ncols() {
            return Err("Submatrix range out of bounds".to_string());
        }

        Ok(ZeroCopySubArray {
            parent: self,
            row_range: (row_start, row_end),
            col_range: (col_start, col_end),
        })
    }
}

/// Zero-copy submatrix view
pub struct ZeroCopySubArray<'a, T> {
    parent: &'a ZeroCopyArray<'a, T>,
    row_range: (usize, usize),
    col_range: (usize, usize),
}

impl<'a, T> ZeroCopySubArray<'a, T> {
    /// Get the shape of the submatrix
    pub fn shape(&self) -> (usize, usize) {
        (
            self.row_range.1 - self.row_range.0,
            self.col_range.1 - self.col_range.0,
        )
    }

    /// Get an element at (row, col) within the submatrix
    pub fn get(&self, row: usize, col: usize) -> Result<&T, String> {
        let (sub_rows, sub_cols) = self.shape();
        if row >= sub_rows || col >= sub_cols {
            return Err(format!("Index ({}, {}) out of bounds for submatrix", row, col));
        }

        self.parent.get(
            self.row_range.0 + row,
            self.col_range.0 + col,
        )
    }
}

/// Memory-mapped array for large datasets
pub struct MemoryMappedArray<T> {
    _data: Vec<T>, // Placeholder for actual mmap implementation
    shape: (usize, usize),
}

impl<T> MemoryMappedArray<T>
where
    T: Clone + Default + bytemuck::Pod,
{
    /// Create a new memory-mapped array from file
    pub fn from_file(_path: &str, rows: usize, cols: usize) -> Result<Self, String> {
        // Placeholder implementation - would use actual memory mapping in production
        Ok(Self {
            _data: vec![T::default(); rows * cols],
            shape: (rows, cols),
        })
    }

    /// Get the shape of the array
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Chunked processing for large datasets that don't fit in memory
pub struct ChunkedProcessor<T> {
    chunk_size: usize,
    overlap: usize,
    _marker: PhantomData<T>,
}

impl<T> ChunkedProcessor<T>
where
    T: Clone,
{
    /// Create a new chunked processor with specified chunk size and overlap
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
            _marker: PhantomData,
        }
    }

    /// Process data in chunks with the provided function
    pub fn process_chunks<F, R>(&self, data: &[T], mut processor: F) -> Vec<R>
    where
        F: FnMut(&[T]) -> R,
    {
        let mut results = Vec::new();
        let mut start = 0;

        while start < data.len() {
            let end = std::cmp::min(start + self.chunk_size, data.len());
            let chunk = &data[start..end];
            results.push(processor(chunk));

            // Move to next chunk with overlap consideration
            if end == data.len() {
                break;
            }
            start += self.chunk_size - self.overlap;
        }

        results
    }

    /// Process data in chunks with parallel execution
    pub fn process_chunks_parallel<F, R>(&self, data: &[T], processor: F) -> Vec<R>
    where
        F: Fn(&[T]) -> R + Send + Sync,
        R: Send,
        T: Send + Sync,
    {
        use rayon::prelude::*;

        let chunk_size = self.chunk_size;
        let overlap = self.overlap;

        let chunks: Vec<_> = {
            let mut chunks = Vec::new();
            let mut start = 0;

            while start < data.len() {
                let end = std::cmp::min(start + chunk_size, data.len());
                chunks.push((start, end));

                if end == data.len() {
                    break;
                }
                start += chunk_size - overlap;
            }

            chunks
        };

        chunks
            .par_iter()
            .map(|(start, end)| processor(&data[*start..*end]))
            .collect()
    }
}

/// Cache-friendly memory layout for ML data structures
pub struct CacheFriendlyArray<T> {
    data: Vec<T>,
    shape: (usize, usize),
    layout: MemoryLayout,
}

/// Memory layout strategies for different access patterns
#[derive(Debug, Clone, Copy)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout - good for row-wise operations
    RowMajor,
    /// Column-major (Fortran-style) layout - good for column-wise operations
    ColumnMajor,
    /// Tiled layout for cache efficiency
    Tiled { tile_size: usize },
}

impl<T> CacheFriendlyArray<T>
where
    T: Clone + Default,
{
    /// Create a new cache-friendly array with specified layout
    pub fn new(rows: usize, cols: usize, layout: MemoryLayout) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            shape: (rows, cols),
            layout,
        }
    }

    /// Get element at (row, col) considering the memory layout
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        let index = self.calculate_index(row, col)?;
        self.data.get(index)
    }

    /// Set element at (row, col) considering the memory layout
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), String> {
        let index = self.calculate_index(row, col)
            .ok_or_else(|| format!("Index ({}, {}) out of bounds", row, col))?;

        if let Some(element) = self.data.get_mut(index) {
            *element = value;
            Ok(())
        } else {
            Err("Internal indexing error".to_string())
        }
    }

    /// Calculate the linear index based on the memory layout
    fn calculate_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return None;
        }

        let index = match self.layout {
            MemoryLayout::RowMajor => row * self.shape.1 + col,
            MemoryLayout::ColumnMajor => col * self.shape.0 + row,
            MemoryLayout::Tiled { tile_size } => {
                let tile_row = row / tile_size;
                let tile_col = col / tile_size;
                let in_tile_row = row % tile_size;
                let in_tile_col = col % tile_size;

                let tiles_per_row = (self.shape.1 + tile_size - 1) / tile_size;
                let tile_index = tile_row * tiles_per_row + tile_col;
                let in_tile_index = in_tile_row * tile_size + in_tile_col;

                tile_index * tile_size * tile_size + in_tile_index
            }
        };

        Some(index)
    }

    /// Get the memory layout of this array
    pub fn layout(&self) -> MemoryLayout {
        self.layout
    }

    /// Get the shape of the array
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

/// Global memory pool for commonly used buffer sizes
pub struct GlobalBufferPool;

thread_local! {
    static F32_POOL: RefCell<MemoryPool<f32>> = RefCell::new(MemoryPool::new(16, 1024));
    static F64_POOL: RefCell<MemoryPool<f64>> = RefCell::new(MemoryPool::new(16, 1024));
    static I32_POOL: RefCell<MemoryPool<i32>> = RefCell::new(MemoryPool::new(8, 1024));
}

impl GlobalBufferPool {
    /// Get a f32 buffer from the global pool
    pub fn get_f32_buffer() -> PooledBuffer<f32> {
        F32_POOL.with(|pool| pool.borrow().get_buffer())
    }

    /// Get a f64 buffer from the global pool
    pub fn get_f64_buffer() -> PooledBuffer<f64> {
        F64_POOL.with(|pool| pool.borrow().get_buffer())
    }

    /// Get an i32 buffer from the global pool
    pub fn get_i32_buffer() -> PooledBuffer<i32> {
        I32_POOL.with(|pool| pool.borrow().get_buffer())
    }

    /// Clear all global pools
    pub fn clear_all() {
        F32_POOL.with(|pool| pool.borrow().clear());
        F64_POOL.with(|pool| pool.borrow().clear());
        I32_POOL.with(|pool| pool.borrow().clear());
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::<f64>::new(2, 10);

        assert_eq!(pool.available_buffers(), 0);

        {
            let buffer1 = pool.get_buffer();
            assert_eq!(buffer1.len(), 10);
            assert_eq!(pool.available_buffers(), 0);

            let buffer2 = pool.get_buffer();
            assert_eq!(buffer2.len(), 10);
            assert_eq!(pool.available_buffers(), 0);
        } // buffers drop here

        assert_eq!(pool.available_buffers(), 2);
    }

    #[test]
    fn test_zero_copy_array() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let array = ZeroCopyArray::new(&data, 2, 3).expect("expected valid value");

        assert_eq!(array.shape(), (2, 3));
        assert_eq!(array.nrows(), 2);
        assert_eq!(array.ncols(), 3);

        assert_eq!(*array.get(0, 0).expect("get should succeed"), 1);
        assert_eq!(*array.get(1, 2).expect("get should succeed"), 6);

        let row = array.row(1).expect("row should succeed");
        assert_eq!(row, &[4, 5, 6]);

        let sub = array.submatrix(0, 2, 1, 3).expect("submatrix should succeed");
        assert_eq!(sub.shape(), (2, 2));
        assert_eq!(*sub.get(0, 0).expect("get should succeed"), 2);
        assert_eq!(*sub.get(1, 1).expect("get should succeed"), 6);
    }

    #[test]
    fn test_chunked_processor() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let processor = ChunkedProcessor::new(4, 1);

        let results = processor.process_chunks(&data, |chunk| chunk.len());

        // Should have chunks of sizes: 4, 4, 3 (with overlap)
        assert!(results.len() >= 2);
        assert_eq!(results[0], 4);
    }

    #[test]
    fn test_cache_friendly_array() {
        let mut array = CacheFriendlyArray::new(3, 3, MemoryLayout::RowMajor);

        array.set(1, 1, 42).expect("set should succeed");
        assert_eq!(*array.get(1, 1).expect("get should succeed"), 42);

        let mut tiled_array = CacheFriendlyArray::new(4, 4, MemoryLayout::Tiled { tile_size: 2 });
        tiled_array.set(0, 0, 1).expect("set should succeed");
        tiled_array.set(0, 1, 2).expect("set should succeed");
        tiled_array.set(1, 0, 3).expect("set should succeed");
        tiled_array.set(1, 1, 4).expect("set should succeed");

        assert_eq!(*tiled_array.get(0, 0).expect("get should succeed"), 1);
        assert_eq!(*tiled_array.get(0, 1).expect("get should succeed"), 2);
        assert_eq!(*tiled_array.get(1, 0).expect("get should succeed"), 3);
        assert_eq!(*tiled_array.get(1, 1).expect("get should succeed"), 4);
    }

    #[test]
    fn test_global_buffer_pool() {
        let buffer1 = GlobalBufferPool::get_f32_buffer();
        assert!(!buffer1.is_empty());

        let buffer2 = GlobalBufferPool::get_f64_buffer();
        assert!(!buffer2.is_empty());

        let buffer3 = GlobalBufferPool::get_i32_buffer();
        assert!(!buffer3.is_empty());

        GlobalBufferPool::clear_all();
    }
}