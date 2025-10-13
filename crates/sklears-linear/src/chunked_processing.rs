//! Chunked processing for memory-constrained environments
//!
//! This module provides utilities for processing large datasets in chunks,
//! particularly useful when working with datasets that don't fit in memory.
//! It integrates with memory-mapped arrays and provides various chunking strategies.

use crate::mmap_arrays::{MmapMatrix, MmapVector};
use sklears_core::error::SklearsError;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

/// Configuration for chunked processing
#[derive(Debug, Clone)]
pub struct ChunkProcessingConfig {
    /// Maximum memory to use for processing (in bytes)
    pub max_memory_bytes: usize,
    /// Number of parallel threads for processing chunks
    pub num_threads: usize,
    /// Whether to use overlap between chunks for continuity
    pub use_overlap: bool,
    /// Size of overlap between chunks (only used if use_overlap is true)
    pub overlap_size: usize,
    /// Whether to process chunks in-place to save memory
    pub in_place_processing: bool,
    /// Buffer size for inter-chunk communication
    pub buffer_size: usize,
}

impl Default for ChunkProcessingConfig {
    fn default() -> Self {
        let available_memory = Self::estimate_available_memory();
        Self {
            max_memory_bytes: available_memory / 2, // Use 50% of available memory
            num_threads: num_cpus::get(),
            use_overlap: false,
            overlap_size: 0,
            in_place_processing: true,
            buffer_size: 1024,
        }
    }
}

impl ChunkProcessingConfig {
    /// Estimate available system memory in bytes
    pub fn estimate_available_memory() -> usize {
        // Default to 4GB if we can't detect system memory
        const DEFAULT_MEMORY: usize = 4 * 1024 * 1024 * 1024;

        #[cfg(unix)]
        {
            unsafe {
                let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
                let page_size = libc::sysconf(libc::_SC_PAGE_SIZE);
                if pages > 0 && page_size > 0 {
                    return (pages * page_size) as usize;
                }
            }
        }

        #[cfg(windows)]
        {
            use std::mem;
            use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

            let mut mem_status: MEMORYSTATUSEX = unsafe { mem::zeroed() };
            mem_status.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;

            unsafe {
                if GlobalMemoryStatusEx(&mut mem_status) != 0 {
                    return mem_status.ullTotalPhys as usize;
                }
            }
        }

        DEFAULT_MEMORY
    }

    /// Calculate optimal chunk size based on memory constraints
    pub fn calculate_chunk_size(&self, element_size: usize, num_features: usize) -> usize {
        // Reserve some memory for intermediate calculations
        let usable_memory = (self.max_memory_bytes as f64 * 0.8) as usize;

        // Calculate how many samples we can fit in memory
        let bytes_per_sample = num_features * element_size;
        let max_samples = usable_memory / bytes_per_sample;

        // Ensure we have at least 1 sample, but limit to reasonable chunk sizes
        max_samples.max(1).min(100000)
    }
}

/// Result of chunked processing operation
#[derive(Debug, Clone)]
pub struct ChunkProcessingResult<T> {
    /// Results from each chunk
    pub chunk_results: Vec<T>,
    /// Total number of chunks processed
    pub num_chunks: usize,
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Memory usage statistics for chunked processing
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage during processing
    pub peak_memory_bytes: usize,
    /// Average memory usage per chunk
    pub avg_memory_per_chunk: usize,
    /// Number of times memory limits were exceeded
    pub memory_limit_violations: usize,
}

/// Trait for chunked processing operations
pub trait ChunkedProcessor<Input, Output> {
    /// Process a single chunk of data
    fn process_chunk(&mut self, chunk: &Input, chunk_index: usize) -> Result<Output, SklearsError>;

    /// Combine results from multiple chunks
    fn combine_results(&self, results: Vec<Output>) -> Result<Output, SklearsError>;

    /// Called before processing starts (optional setup)
    fn setup(&mut self) -> Result<(), SklearsError> {
        Ok(())
    }

    /// Called after processing completes (optional cleanup)
    fn cleanup(&mut self) -> Result<(), SklearsError> {
        Ok(())
    }
}

/// Chunked linear regression processor
pub struct ChunkedLinearRegression {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Accumulated XTX matrix
    pub xtx: Option<Vec<Vec<f64>>>,
    /// Accumulated XTy vector
    pub xty: Option<Vec<f64>>,
    /// Number of features
    pub n_features: Option<usize>,
    /// Sample count for averaging
    pub n_samples: usize,
}

impl ChunkedLinearRegression {
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            fit_intercept,
            xtx: None,
            xty: None,
            n_features: None,
            n_samples: 0,
        }
    }

    fn accumulate_normal_equations(
        &mut self,
        x_chunk: &[f64],
        y_chunk: &[f64],
        chunk_rows: usize,
        chunk_cols: usize,
    ) -> Result<(), SklearsError> {
        // Initialize on first chunk
        if self.n_features.is_none() {
            let features = if self.fit_intercept {
                chunk_cols + 1
            } else {
                chunk_cols
            };
            self.n_features = Some(features);
            self.xtx = Some(vec![vec![0.0; features]; features]);
            self.xty = Some(vec![0.0; features]);
        }

        let n_features = self.n_features.unwrap();
        let xtx = self.xtx.as_mut().unwrap();
        let xty = self.xty.as_mut().unwrap();

        // Accumulate XTX and XTy
        for i in 0..chunk_rows {
            // Get row from X
            let x_row = &x_chunk[i * chunk_cols..(i + 1) * chunk_cols];
            let y_val = y_chunk[i];

            // Add intercept if needed
            let x_extended: Vec<f64> = if self.fit_intercept {
                let mut x_ext = vec![1.0];
                x_ext.extend_from_slice(x_row);
                x_ext
            } else {
                x_row.to_vec()
            };

            // Update XTX
            for j in 0..n_features {
                for k in 0..n_features {
                    xtx[j][k] += x_extended[j] * x_extended[k];
                }
            }

            // Update XTy
            for j in 0..n_features {
                xty[j] += x_extended[j] * y_val;
            }
        }

        self.n_samples += chunk_rows;
        Ok(())
    }

    fn solve_normal_equations(&self) -> Result<Vec<f64>, SklearsError> {
        let xtx = self.xtx.as_ref().ok_or_else(|| SklearsError::InvalidData {
            reason: "XTX not initialized".to_string(),
        })?;
        let xty = self.xty.as_ref().ok_or_else(|| SklearsError::InvalidData {
            reason: "XTy not initialized".to_string(),
        })?;

        let n_features = xtx.len();

        // Simple Gaussian elimination for solving XTX * beta = XTy
        let mut a = xtx.clone();
        let mut b = xty.clone();

        // Forward elimination
        for i in 0..n_features {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n_features {
                if a[k][i].abs() > a[max_row][i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                a.swap(i, max_row);
                b.swap(i, max_row);
            }

            // Check for singular matrix
            if a[i][i].abs() < 1e-12 {
                return Err(SklearsError::NumericalError(
                    "Singular matrix in normal equations".to_string(),
                ));
            }

            // Eliminate column
            for k in (i + 1)..n_features {
                let factor = a[k][i] / a[i][i];
                for j in i..n_features {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }

        // Back substitution
        let mut x = vec![0.0; n_features];
        for i in (0..n_features).rev() {
            x[i] = b[i];
            for j in (i + 1)..n_features {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        Ok(x)
    }
}

impl ChunkedProcessor<(Vec<f64>, Vec<f64>, usize, usize), Vec<f64>> for ChunkedLinearRegression {
    fn process_chunk(
        &mut self,
        chunk: &(Vec<f64>, Vec<f64>, usize, usize),
        _chunk_index: usize,
    ) -> Result<Vec<f64>, SklearsError> {
        let (x_data, y_data, rows, cols) = chunk;
        self.accumulate_normal_equations(x_data, y_data, *rows, *cols)?;

        // Return empty vector as we're accumulating
        Ok(vec![])
    }

    fn combine_results(&self, _results: Vec<Vec<f64>>) -> Result<Vec<f64>, SklearsError> {
        // Solve the accumulated normal equations
        self.solve_normal_equations()
    }
}

/// Chunked processor for matrix operations
pub struct ChunkedMatrixProcessor;

impl ChunkedMatrixProcessor {
    /// Process large matrix multiplication using chunks
    pub fn chunked_matrix_multiply(
        a: &MmapMatrix,
        b: &MmapMatrix,
        config: &ChunkProcessingConfig,
    ) -> Result<Vec<Vec<f64>>, SklearsError> {
        let (a_rows, a_cols) = a.shape();
        let (b_rows, b_cols) = b.shape();

        if a_cols != b_rows {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimensions don't match for multiplication: {}x{} * {}x{}",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        // Calculate chunk sizes
        let element_size = std::mem::size_of::<f64>();
        let chunk_size = config.calculate_chunk_size(element_size, a_cols);
        let chunk_rows = (chunk_size / a_cols).max(1);

        // Initialize result matrix
        let mut result = vec![vec![0.0; b_cols]; a_rows];

        // Process in chunks
        for start_row in (0..a_rows).step_by(chunk_rows) {
            let end_row = (start_row + chunk_rows).min(a_rows);

            // Process chunk of A
            for i in start_row..end_row {
                let a_row = a.get_row(i)?;

                for j in 0..b_cols {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        sum += a_row[k] * b.get(k, j)?;
                    }
                    result[i][j] = sum;
                }
            }
        }

        Ok(result)
    }

    /// Process large matrix-vector multiplication using chunks
    pub fn chunked_matrix_vector_multiply(
        matrix: &MmapMatrix,
        vector: &MmapVector,
        config: &ChunkProcessingConfig,
    ) -> Result<Vec<f64>, SklearsError> {
        let (rows, cols) = matrix.shape();

        if cols != vector.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix-vector dimensions don't match: {}x{} * {}",
                rows,
                cols,
                vector.len()
            )));
        }

        // Calculate chunk size
        let element_size = std::mem::size_of::<f64>();
        let chunk_size = config.calculate_chunk_size(element_size, cols);
        let chunk_rows = (chunk_size / cols).max(1);

        let mut result = vec![0.0; rows];

        // Process in chunks
        for start_row in (0..rows).step_by(chunk_rows) {
            let end_row = (start_row + chunk_rows).min(rows);

            for i in start_row..end_row {
                let matrix_row = matrix.get_row(i)?;
                let vector_slice = vector.as_slice();

                let mut dot_product = 0.0;
                for j in 0..cols {
                    dot_product += matrix_row[j] * vector_slice[j];
                }
                result[i] = dot_product;
            }
        }

        Ok(result)
    }
}

/// Parallel chunked processor using multiple threads
pub struct ParallelChunkedProcessor {
    config: ChunkProcessingConfig,
}

impl ParallelChunkedProcessor {
    pub fn new(config: ChunkProcessingConfig) -> Self {
        Self { config }
    }

    /// Process chunks in parallel using multiple threads
    pub fn process_parallel<T, R, F>(
        &self,
        data_chunks: Vec<T>,
        processor: F,
    ) -> Result<ChunkProcessingResult<R>, SklearsError>
    where
        T: Send + 'static + Clone,
        R: Send + 'static,
        F: FnMut(T, usize) -> Result<R, SklearsError> + Send + 'static + Clone,
    {
        let start_time = std::time::Instant::now();
        let num_chunks = data_chunks.len();

        if num_chunks == 0 {
            return Ok(ChunkProcessingResult {
                chunk_results: vec![],
                num_chunks: 0,
                processing_time_ms: 0,
                memory_stats: MemoryStats {
                    peak_memory_bytes: 0,
                    avg_memory_per_chunk: 0,
                    memory_limit_violations: 0,
                },
            });
        }

        // Create channels for communication
        let (tx, rx) = mpsc::channel();

        // Spawn worker threads
        let mut handles = vec![];
        let chunks_per_thread =
            (num_chunks + self.config.num_threads - 1) / self.config.num_threads;

        // Convert data_chunks to indexed chunks first
        let indexed_chunks: Vec<(usize, T)> = data_chunks.into_iter().enumerate().collect();

        for thread_id in 0..self.config.num_threads {
            let start_idx = thread_id * chunks_per_thread;
            let end_idx = ((thread_id + 1) * chunks_per_thread).min(num_chunks);

            if start_idx >= num_chunks {
                break;
            }

            let thread_chunks: Vec<(usize, T)> = indexed_chunks
                .iter()
                .skip(start_idx)
                .take(end_idx - start_idx)
                .map(|(idx, chunk)| (*idx, chunk.clone()))
                .collect();

            let tx_clone = tx.clone();
            let mut processor_clone = processor.clone();

            let handle = thread::spawn(move || {
                for (chunk_idx, chunk) in thread_chunks {
                    match processor_clone(chunk, chunk_idx) {
                        Ok(result) => {
                            if tx_clone.send((chunk_idx, Ok(result))).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx_clone.send((chunk_idx, Err(e)));
                            break;
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Drop the original sender
        drop(tx);

        // Collect results
        let mut results = vec![];
        for _ in 0..num_chunks {
            match rx.recv() {
                Ok((chunk_idx, result)) => match result {
                    Ok(r) => results.push((chunk_idx, r)),
                    Err(e) => return Err(e),
                },
                Err(_) => break,
            }
        }

        // Wait for all threads to complete
        for handle in handles {
            handle
                .join()
                .map_err(|_| SklearsError::Other("Thread join failed".to_string()))?;
        }

        // Sort results by chunk index
        results.sort_by_key(|(idx, _)| *idx);
        let chunk_results: Vec<R> = results.into_iter().map(|(_, r)| r).collect();

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ChunkProcessingResult {
            chunk_results,
            num_chunks,
            processing_time_ms: processing_time,
            memory_stats: MemoryStats {
                peak_memory_bytes: self.config.max_memory_bytes,
                avg_memory_per_chunk: self.config.max_memory_bytes / num_chunks.max(1),
                memory_limit_violations: 0,
            },
        })
    }
}

/// Utility functions for chunked processing
pub struct ChunkedProcessingUtils;

impl ChunkedProcessingUtils {
    /// Split data into chunks based on memory constraints
    pub fn calculate_optimal_chunks(
        total_samples: usize,
        n_features: usize,
        max_memory_bytes: usize,
    ) -> (usize, usize) {
        let element_size = std::mem::size_of::<f64>();
        let bytes_per_sample = n_features * element_size;

        // Reserve 20% of memory for overhead
        let usable_memory = (max_memory_bytes as f64 * 0.8) as usize;

        let samples_per_chunk = (usable_memory / bytes_per_sample).max(1);
        let num_chunks = (total_samples + samples_per_chunk - 1) / samples_per_chunk;

        (samples_per_chunk, num_chunks)
    }

    /// Estimate memory usage for a given chunk size
    pub fn estimate_memory_usage(
        chunk_size: usize,
        n_features: usize,
        overhead_factor: f64,
    ) -> usize {
        let element_size = std::mem::size_of::<f64>();
        let base_memory = chunk_size * n_features * element_size;
        (base_memory as f64 * (1.0 + overhead_factor)) as usize
    }

    /// Create memory-efficient data iterator
    pub fn create_chunk_iterator(
        matrix: Arc<MmapMatrix>,
        chunk_size: usize,
    ) -> ChunkedDataIterator {
        ChunkedDataIterator::new(matrix, chunk_size)
    }
}

/// Iterator for processing data in chunks
pub struct ChunkedDataIterator {
    matrix: Arc<MmapMatrix>,
    chunk_size: usize,
    current_pos: usize,
    total_rows: usize,
}

impl ChunkedDataIterator {
    fn new(matrix: Arc<MmapMatrix>, chunk_size: usize) -> Self {
        let (total_rows, _) = matrix.shape();
        Self {
            matrix,
            chunk_size,
            current_pos: 0,
            total_rows,
        }
    }
}

impl Iterator for ChunkedDataIterator {
    type Item = Result<(Vec<f64>, usize, usize), SklearsError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.total_rows {
            return None;
        }

        let end_pos = (self.current_pos + self.chunk_size).min(self.total_rows);
        let actual_chunk_size = end_pos - self.current_pos;
        let (_, cols) = self.matrix.shape();

        // Extract chunk data
        let mut chunk_data = Vec::with_capacity(actual_chunk_size * cols);

        for row in self.current_pos..end_pos {
            match self.matrix.get_row(row) {
                Ok(row_data) => chunk_data.extend_from_slice(row_data),
                Err(e) => return Some(Err(e)),
            }
        }

        let result = Ok((chunk_data, self.current_pos, actual_chunk_size));
        self.current_pos = end_pos;
        Some(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmap_arrays::{MmapConfig, MmapUtils};
    use tempfile::tempdir;

    #[test]
    fn test_chunk_processing_config() {
        let config = ChunkProcessingConfig::default();

        assert!(config.max_memory_bytes > 0);
        assert!(config.num_threads > 0);

        let chunk_size = config.calculate_chunk_size(8, 100);
        assert!(chunk_size > 0);
    }

    #[test]
    fn test_chunked_linear_regression() {
        let mut processor = ChunkedLinearRegression::new(true);

        // Create test data - avoid multicollinearity
        let x_data = vec![
            1.0, 3.0, // sample 1
            2.0, 1.0, // sample 2
            3.0, 4.0, // sample 3
            4.0, 2.0, // sample 4
            5.0, 6.0, // sample 5
        ]; // 5x2 matrix with non-collinear columns
        let y_data = vec![3.0, 5.0, 11.0, 10.0, 23.0]; // 5x1 vector

        let chunk = (x_data, y_data, 5, 2);
        processor.process_chunk(&chunk, 0).unwrap();

        let coefficients = processor.combine_results(vec![]).unwrap();
        assert_eq!(coefficients.len(), 3); // intercept + 2 features
    }

    #[test]
    fn test_chunked_matrix_multiplication() {
        let dir = tempdir().unwrap();
        let file_a = dir.path().join("matrix_a.dat");
        let file_b = dir.path().join("matrix_b.dat");

        // Create test matrices A (3x2) and B (2x3)
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        MmapUtils::array_to_mmap_file(&a_data, &file_a).unwrap();
        MmapUtils::array_to_mmap_file(&b_data, &file_b).unwrap();

        let config = MmapConfig::default();
        let matrix_a = MmapMatrix::from_file(&file_a, 3, 2, config.clone()).unwrap();
        let matrix_b = MmapMatrix::from_file(&file_b, 2, 3, config).unwrap();

        let proc_config = ChunkProcessingConfig::default();
        let result =
            ChunkedMatrixProcessor::chunked_matrix_multiply(&matrix_a, &matrix_b, &proc_config)
                .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_chunked_data_iterator() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.dat");

        let rows = 100;
        let cols = 5;
        let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();

        MmapUtils::array_to_mmap_file(&data, &file_path).unwrap();

        let config = MmapConfig::default();
        let matrix = Arc::new(MmapMatrix::from_file(&file_path, rows, cols, config).unwrap());

        let chunk_size = 5; // Process 5 rows at a time
        let iterator = ChunkedProcessingUtils::create_chunk_iterator(matrix, chunk_size);

        let chunks: Result<Vec<_>, _> = iterator.collect();
        let chunks = chunks.unwrap();

        assert_eq!(chunks.len(), 20); // 100 rows / 5 rows per chunk
        assert_eq!(chunks[0].1, 0); // First chunk starts at position 0
        assert_eq!(chunks[0].2, 5); // First chunk has 5 rows
    }

    #[test]
    fn test_parallel_chunked_processor() {
        let config = ChunkProcessingConfig {
            num_threads: 2,
            ..Default::default()
        };
        let processor = ParallelChunkedProcessor::new(config);

        let data_chunks: Vec<i32> = (0..10).collect();

        let result = processor
            .process_parallel(data_chunks, |chunk, _idx| Ok(chunk * 2))
            .unwrap();

        assert_eq!(result.num_chunks, 10);
        assert_eq!(result.chunk_results.len(), 10);
        assert_eq!(result.chunk_results[0], 0);
        assert_eq!(result.chunk_results[9], 18);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let memory_usage = ChunkedProcessingUtils::estimate_memory_usage(
            1000, // chunk_size
            50,   // n_features
            0.2,  // 20% overhead
        );

        let expected = 1000.0 * 50.0 * 8.0 * 1.2; // 1000 * 50 * sizeof(f64) * 1.2
        assert_eq!(memory_usage, expected as usize);
    }
}
