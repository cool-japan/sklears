//! Memory-mapped arrays for out-of-core computation
//!
//! This module provides memory-mapped array functionality for processing datasets
//! that are larger than available RAM. It supports both read-only and read-write
//! memory-mapped files with chunked processing capabilities.

use memmap2::{Mmap, MmapMut, MmapOptions};
use sklears_core::error::SklearsError;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::slice;

/// Configuration for memory-mapped arrays
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Whether to use read-write mapping (default: false for read-only)
    pub read_write: bool,
    /// Number of elements to process in each chunk
    pub chunk_size: usize,
    /// Whether to prefault pages (load all pages into memory)
    pub prefault: bool,
    /// Advice for memory access pattern
    pub advice: MmapAdvice,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            read_write: false,
            chunk_size: 65536, // 64k elements
            prefault: false,
            advice: MmapAdvice::Sequential,
        }
    }
}

/// Memory access pattern advice for optimization
#[derive(Debug, Clone, Copy)]
pub enum MmapAdvice {
    /// Normal random access pattern
    Normal,
    /// Sequential access pattern (forward)
    Sequential,
    /// Random access pattern
    Random,
    /// Will need this data soon
    WillNeed,
    /// Don't need this data anymore
    DontNeed,
}

/// Memory-mapped matrix for out-of-core linear algebra
pub struct MmapMatrix {
    mmap: Mmap,
    rows: usize,
    cols: usize,
    config: MmapConfig,
}

/// Mutable memory-mapped matrix for out-of-core computation
pub struct MmapMatrixMut {
    mmap: MmapMut,
    rows: usize,
    cols: usize,
    config: MmapConfig,
}

/// Memory-mapped vector
pub struct MmapVector {
    mmap: Mmap,
    len: usize,
    config: MmapConfig,
}

/// Mutable memory-mapped vector
pub struct MmapVectorMut {
    mmap: MmapMut,
    len: usize,
    config: MmapConfig,
}

impl MmapMatrix {
    /// Create a new memory-mapped matrix from file
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        rows: usize,
        cols: usize,
        config: MmapConfig,
    ) -> Result<Self, SklearsError> {
        let file = File::open(path)?;

        let expected_size = rows * cols * std::mem::size_of::<f64>();
        let file_size = file.metadata()?.len() as usize;

        if file_size < expected_size {
            return Err(SklearsError::InvalidInput(format!(
                "File size {} is smaller than expected size {} for {}x{} matrix",
                file_size, expected_size, rows, cols
            )));
        }

        let mut mmap_options = MmapOptions::new();
        if config.prefault {
            mmap_options.populate();
        }

        let mmap = unsafe { mmap_options.map(&file)? };

        let matrix = Self {
            mmap,
            rows,
            cols,
            config,
        };

        matrix.apply_advice()?;
        Ok(matrix)
    }

    /// Get dimensions of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get matrix element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<f64, SklearsError> {
        if row >= self.rows || col >= self.cols {
            return Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

        let index = row * self.cols + col;
        let data = self.as_slice();
        Ok(data[index])
    }

    /// Get a row as a slice
    pub fn get_row(&self, row: usize) -> Result<&[f64], SklearsError> {
        if row >= self.rows {
            return Err(SklearsError::InvalidInput(format!(
                "Row index {} out of bounds for matrix with {} rows",
                row, self.rows
            )));
        }

        let start = row * self.cols;
        let end = start + self.cols;
        let data = self.as_slice();
        Ok(&data[start..end])
    }

    /// Get the entire matrix as a slice
    pub fn as_slice(&self) -> &[f64] {
        let ptr = self.mmap.as_ptr() as *const f64;
        let len = self.rows * self.cols;
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Process matrix in chunks with a callback function
    pub fn process_chunks<F, R>(&self, mut callback: F) -> Result<Vec<R>, SklearsError>
    where
        F: FnMut(&[f64], usize, usize) -> Result<R, SklearsError>,
    {
        let mut results = Vec::new();
        let chunk_rows = (self.config.chunk_size / self.cols).max(1);

        for start_row in (0..self.rows).step_by(chunk_rows) {
            let end_row = (start_row + chunk_rows).min(self.rows);
            let chunk_size = (end_row - start_row) * self.cols;

            let start_idx = start_row * self.cols;
            let end_idx = start_idx + chunk_size;

            let data = self.as_slice();
            let chunk = &data[start_idx..end_idx];

            let result = callback(chunk, start_row, end_row - start_row)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Apply memory access advice
    fn apply_advice(&self) -> Result<(), SklearsError> {
        #[cfg(unix)]
        {
            let advice = match self.config.advice {
                MmapAdvice::Normal => libc::MADV_NORMAL,
                MmapAdvice::Sequential => libc::MADV_SEQUENTIAL,
                MmapAdvice::Random => libc::MADV_RANDOM,
                MmapAdvice::WillNeed => libc::MADV_WILLNEED,
                MmapAdvice::DontNeed => libc::MADV_DONTNEED,
            };

            let result = unsafe {
                libc::madvise(
                    self.mmap.as_ptr() as *mut libc::c_void,
                    self.mmap.len(),
                    advice,
                )
            };

            if result != 0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Failed to apply memory advice: {}",
                    io::Error::last_os_error()
                )));
            }
        }

        Ok(())
    }
}

impl MmapMatrixMut {
    /// Create a new mutable memory-mapped matrix
    pub fn create_file<P: AsRef<Path>>(
        path: P,
        rows: usize,
        cols: usize,
        config: MmapConfig,
    ) -> Result<Self, SklearsError> {
        let file_size = rows * cols * std::mem::size_of::<f64>();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(file_size as u64)?;

        let mut mmap_options = MmapOptions::new();
        if config.prefault {
            mmap_options.populate();
        }

        let mmap = unsafe { mmap_options.map_mut(&file)? };

        let matrix = Self {
            mmap,
            rows,
            cols,
            config,
        };

        matrix.apply_advice()?;
        Ok(matrix)
    }

    /// From existing file (read-write)
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        rows: usize,
        cols: usize,
        config: MmapConfig,
    ) -> Result<Self, SklearsError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let expected_size = rows * cols * std::mem::size_of::<f64>();
        let file_size = file.metadata()?.len() as usize;

        if file_size < expected_size {
            return Err(SklearsError::InvalidInput(format!(
                "File size {} is smaller than expected size {} for {}x{} matrix",
                file_size, expected_size, rows, cols
            )));
        }

        let mut mmap_options = MmapOptions::new();
        if config.prefault {
            mmap_options.populate();
        }

        let mmap = unsafe { mmap_options.map_mut(&file)? };

        let matrix = Self {
            mmap,
            rows,
            cols,
            config,
        };

        matrix.apply_advice()?;
        Ok(matrix)
    }

    /// Get dimensions of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Set matrix element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), SklearsError> {
        if row >= self.rows || col >= self.cols {
            return Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

        let index = row * self.cols + col;
        let data = self.as_mut_slice();
        data[index] = value;
        Ok(())
    }

    /// Get matrix element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<f64, SklearsError> {
        if row >= self.rows || col >= self.cols {
            return Err(SklearsError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

        let index = row * self.cols + col;
        let data = self.as_slice();
        Ok(data[index])
    }

    /// Get the entire matrix as a slice
    pub fn as_slice(&self) -> &[f64] {
        let ptr = self.mmap.as_ptr() as *const f64;
        let len = self.rows * self.cols;
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Get the entire matrix as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        let ptr = self.mmap.as_mut_ptr() as *mut f64;
        let len = self.rows * self.cols;
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }

    /// Get a mutable row
    pub fn get_row_mut(&mut self, row: usize) -> Result<&mut [f64], SklearsError> {
        if row >= self.rows {
            return Err(SklearsError::InvalidInput(format!(
                "Row index {} out of bounds for matrix with {} rows",
                row, self.rows
            )));
        }

        let start = row * self.cols;
        let end = start + self.cols;
        let data = self.as_mut_slice();
        Ok(&mut data[start..end])
    }

    /// Process matrix in chunks with a mutable callback function
    pub fn process_chunks_mut<F, R>(&mut self, mut callback: F) -> Result<Vec<R>, SklearsError>
    where
        F: FnMut(&mut [f64], usize, usize) -> Result<R, SklearsError>,
    {
        let mut results = Vec::new();
        let chunk_rows = (self.config.chunk_size / self.cols).max(1);

        for start_row in (0..self.rows).step_by(chunk_rows) {
            let end_row = (start_row + chunk_rows).min(self.rows);
            let chunk_size = (end_row - start_row) * self.cols;

            let start_idx = start_row * self.cols;
            let _end_idx = start_idx + chunk_size;

            // We need to get mutable slice for each chunk separately
            // to avoid borrow checker issues
            let ptr = self.mmap.as_mut_ptr() as *mut f64;
            let chunk = unsafe { slice::from_raw_parts_mut(ptr.add(start_idx), chunk_size) };

            let result = callback(chunk, start_row, end_row - start_row)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Flush changes to disk
    pub fn flush(&self) -> Result<(), SklearsError> {
        Ok(self.mmap.flush()?)
    }

    /// Apply memory access advice
    fn apply_advice(&self) -> Result<(), SklearsError> {
        #[cfg(unix)]
        {
            let advice = match self.config.advice {
                MmapAdvice::Normal => libc::MADV_NORMAL,
                MmapAdvice::Sequential => libc::MADV_SEQUENTIAL,
                MmapAdvice::Random => libc::MADV_RANDOM,
                MmapAdvice::WillNeed => libc::MADV_WILLNEED,
                MmapAdvice::DontNeed => libc::MADV_DONTNEED,
            };

            let result = unsafe {
                libc::madvise(
                    self.mmap.as_ptr() as *mut libc::c_void,
                    self.mmap.len(),
                    advice,
                )
            };

            if result != 0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Failed to apply memory advice: {}",
                    io::Error::last_os_error()
                )));
            }
        }

        Ok(())
    }
}

impl MmapVector {
    /// Create memory-mapped vector from file
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        len: usize,
        config: MmapConfig,
    ) -> Result<Self, SklearsError> {
        let file = File::open(path)?;

        let expected_size = len * std::mem::size_of::<f64>();
        let file_size = file.metadata()?.len() as usize;

        if file_size < expected_size {
            return Err(SklearsError::InvalidInput(format!(
                "File size {} is smaller than expected size {} for vector of length {}",
                file_size, expected_size, len
            )));
        }

        let mut mmap_options = MmapOptions::new();
        if config.prefault {
            mmap_options.populate();
        }

        let mmap = unsafe { mmap_options.map(&file)? };

        Ok(Self { mmap, len, config })
    }

    /// Get vector length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Result<f64, SklearsError> {
        if index >= self.len {
            return Err(SklearsError::InvalidInput(format!(
                "Index {} out of bounds for vector of length {}",
                index, self.len
            )));
        }

        let data = self.as_slice();
        Ok(data[index])
    }

    /// Get vector as slice
    pub fn as_slice(&self) -> &[f64] {
        let ptr = self.mmap.as_ptr() as *const f64;
        unsafe { slice::from_raw_parts(ptr, self.len) }
    }

    /// Process vector in chunks
    pub fn process_chunks<F, R>(&self, mut callback: F) -> Result<Vec<R>, SklearsError>
    where
        F: FnMut(&[f64], usize) -> Result<R, SklearsError>,
    {
        let mut results = Vec::new();

        for start in (0..self.len).step_by(self.config.chunk_size) {
            let end = (start + self.config.chunk_size).min(self.len);
            let data = self.as_slice();
            let chunk = &data[start..end];

            let result = callback(chunk, start)?;
            results.push(result);
        }

        Ok(results)
    }
}

impl MmapVectorMut {
    /// Create mutable memory-mapped vector
    pub fn create_file<P: AsRef<Path>>(
        path: P,
        len: usize,
        config: MmapConfig,
    ) -> Result<Self, SklearsError> {
        let file_size = len * std::mem::size_of::<f64>();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(file_size as u64)?;

        let mut mmap_options = MmapOptions::new();
        if config.prefault {
            mmap_options.populate();
        }

        let mmap = unsafe { mmap_options.map_mut(&file)? };

        Ok(Self { mmap, len, config })
    }

    /// Get vector length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: f64) -> Result<(), SklearsError> {
        if index >= self.len {
            return Err(SklearsError::InvalidInput(format!(
                "Index {} out of bounds for vector of length {}",
                index, self.len
            )));
        }

        let data = self.as_mut_slice();
        data[index] = value;
        Ok(())
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Result<f64, SklearsError> {
        if index >= self.len {
            return Err(SklearsError::InvalidInput(format!(
                "Index {} out of bounds for vector of length {}",
                index, self.len
            )));
        }

        let data = self.as_slice();
        Ok(data[index])
    }

    /// Get vector as slice
    pub fn as_slice(&self) -> &[f64] {
        let ptr = self.mmap.as_ptr() as *const f64;
        unsafe { slice::from_raw_parts(ptr, self.len) }
    }

    /// Get vector as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        let ptr = self.mmap.as_mut_ptr() as *mut f64;
        unsafe { slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Flush changes to disk
    pub fn flush(&self) -> Result<(), SklearsError> {
        Ok(self.mmap.flush()?)
    }
}

/// Utility functions for memory-mapped operations
pub struct MmapUtils;

impl MmapUtils {
    /// Copy data from regular array to memory-mapped file
    pub fn array_to_mmap_file<P: AsRef<Path>>(data: &[f64], path: P) -> Result<(), SklearsError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let bytes = unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        file.write_all(bytes)?;

        Ok(())
    }

    /// Copy data from memory-mapped file to regular array
    pub fn mmap_file_to_array<P: AsRef<Path>>(
        path: P,
        len: usize,
    ) -> Result<Vec<f64>, SklearsError> {
        let config = MmapConfig::default();
        let mmap_vec = MmapVector::from_file(path, len, config)?;
        Ok(mmap_vec.as_slice().to_vec())
    }

    /// Get recommended chunk size based on available memory
    pub fn recommend_chunk_size(
        matrix_rows: usize,
        matrix_cols: usize,
        available_memory_gb: f64,
    ) -> usize {
        let element_size = std::mem::size_of::<f64>();
        let available_bytes = (available_memory_gb * 1024.0 * 1024.0 * 1024.0) as usize;

        // Use 50% of available memory for chunks
        let target_chunk_bytes = available_bytes / 2;
        let mut elements_per_chunk = target_chunk_bytes / element_size;

        // Ensure we process at least one complete row
        let min_chunk_size = matrix_cols.max(1);
        elements_per_chunk = elements_per_chunk.max(min_chunk_size);

        // Never request more elements than exist in the matrix
        let total_elements = matrix_rows.saturating_mul(matrix_cols).max(min_chunk_size);
        elements_per_chunk = elements_per_chunk.min(total_elements);

        // Clamp to a reasonable upper bound to avoid excessively large chunks
        let max_reasonable = 1_000_000usize.max(min_chunk_size);
        elements_per_chunk.min(max_reasonable)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_matrix_creation() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_matrix.dat");

        // Create test data
        let rows = 100;
        let cols = 50;
        let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();

        // Write to file
        MmapUtils::array_to_mmap_file(&data, &file_path).unwrap();

        // Create memory-mapped matrix
        let config = MmapConfig::default();
        let mmap_matrix = MmapMatrix::from_file(&file_path, rows, cols, config).unwrap();

        // Test access
        assert_eq!(mmap_matrix.shape(), (rows, cols));
        assert_eq!(mmap_matrix.get(0, 0).unwrap(), 0.0);
        assert_eq!(mmap_matrix.get(1, 0).unwrap(), cols as f64);

        // Test row access
        let row = mmap_matrix.get_row(0).unwrap();
        assert_eq!(row.len(), cols);
        assert_eq!(row[0], 0.0);
        assert_eq!(row[1], 1.0);
    }

    #[test]
    fn test_mmap_matrix_chunked_processing() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_matrix_chunks.dat");

        let rows = 1000;
        let cols = 10;
        let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();

        MmapUtils::array_to_mmap_file(&data, &file_path).unwrap();

        let config = MmapConfig {
            chunk_size: 500, // Process 50 rows at a time (500 elements / 10 cols)
            ..MmapConfig::default()
        };

        let mmap_matrix = MmapMatrix::from_file(&file_path, rows, cols, config).unwrap();

        let results = mmap_matrix
            .process_chunks(|chunk, start_row, num_rows| Ok((chunk.len(), start_row, num_rows)))
            .unwrap();

        // Should have 20 chunks (1000 rows / 50 rows per chunk)
        assert_eq!(results.len(), 20);
        assert_eq!(results[0], (500, 0, 50)); // First chunk
        assert_eq!(results[19], (500, 950, 50)); // Last chunk
    }

    #[test]
    fn test_mmap_vector() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_vector.dat");

        let len = 10000;
        let data: Vec<f64> = (0..len).map(|i| i as f64).collect();

        MmapUtils::array_to_mmap_file(&data, &file_path).unwrap();

        let config = MmapConfig::default();
        let mmap_vec = MmapVector::from_file(&file_path, len, config).unwrap();

        assert_eq!(mmap_vec.len(), len);
        assert_eq!(mmap_vec.get(0).unwrap(), 0.0);
        assert_eq!(mmap_vec.get(9999).unwrap(), 9999.0);

        // Test chunked processing
        let results = mmap_vec
            .process_chunks(|chunk, start_idx| Ok((chunk.len(), start_idx, chunk[0])))
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].1, 0); // First chunk starts at index 0
        assert_eq!(results[0].2, 0.0); // First element is 0.0
    }

    #[test]
    fn test_mmap_matrix_mut() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_matrix_mut.dat");

        let rows = 10;
        let cols = 5;
        let config = MmapConfig::default();

        let mut mmap_matrix = MmapMatrixMut::create_file(&file_path, rows, cols, config).unwrap();

        // Test setting values
        mmap_matrix.set(0, 0, 42.0).unwrap();
        mmap_matrix.set(1, 2, 3.15).unwrap();

        // Test getting values
        assert_eq!(mmap_matrix.get(0, 0).unwrap(), 42.0);
        assert_eq!(mmap_matrix.get(1, 2).unwrap(), 3.15);

        // Test flushing
        mmap_matrix.flush().unwrap();
    }

    #[test]
    fn test_chunk_size_recommendation() {
        let rows = 10000;
        let cols = 100;
        let available_gb = 4.0;

        let chunk_size = MmapUtils::recommend_chunk_size(rows, cols, available_gb);

        // Should be at least matrix_cols
        assert!(chunk_size >= cols);

        // Should be reasonable (not too large)
        let max_reasonable = 1000000; // 1M elements
        assert!(chunk_size <= max_reasonable);
    }
}
