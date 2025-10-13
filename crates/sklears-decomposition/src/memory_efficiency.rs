//! Memory efficiency improvements for decomposition algorithms
//!
//! This module provides memory-efficient implementations including:
//! - Out-of-core decomposition methods for datasets larger than memory
//! - Memory-mapped matrix operations for efficient disk access
//! - Chunked processing for large matrices
//! - Compression techniques for decomposed matrices
//! - Lazy evaluation for decomposition chains

use memmap2::{Mmap, MmapOptions};
use scirs2_core::ndarray::{s, Array2};
use sklears_core::error::{Result as SklearsResult, SklearsError};
use std::fs::File;
use std::path::Path;

/// Type alias for complex operation function type
type DecompositionOperation<T> = Box<dyn Fn(&T) -> SklearsResult<T>>;

/// Out-of-core matrix decomposition for datasets larger than memory
pub struct OutOfCoreDecomposition {
    chunk_size: usize,
    temp_dir: std::path::PathBuf,
    n_components: usize,
}

impl OutOfCoreDecomposition {
    pub fn new(chunk_size: usize, temp_dir: Option<&Path>) -> Self {
        let temp_dir = temp_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::temp_dir().join("sklears_decomposition"));

        Self {
            chunk_size,
            temp_dir,
            n_components: 10,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Perform incremental SVD on large matrices using out-of-core processing
    pub fn incremental_svd<P: AsRef<Path>>(
        &self,
        matrix_file: P,
        n_rows: usize,
        n_cols: usize,
    ) -> SklearsResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        // Create temporary directory
        std::fs::create_dir_all(&self.temp_dir).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create temp directory: {e}"))
        })?;

        // Initialize running SVD
        let mut u_running = Array2::zeros((0, self.n_components));
        let mut s_running = Array2::zeros((self.n_components, 1));
        let mut vt_running = Array2::zeros((self.n_components, n_cols));

        // Process matrix in chunks
        let file = File::open(matrix_file)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open matrix file: {e}")))?;

        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to memory map file: {e}"))
            })?
        };

        let chunk_rows = self.chunk_size / (n_cols * 8); // Assuming f64
        let n_chunks = (n_rows + chunk_rows - 1) / chunk_rows;

        for chunk_idx in 0..n_chunks {
            let start_row = chunk_idx * chunk_rows;
            let end_row = std::cmp::min(start_row + chunk_rows, n_rows);
            let actual_chunk_rows = end_row - start_row;

            // Extract chunk from memory-mapped file
            let chunk = self.extract_chunk(&mmap, start_row, actual_chunk_rows, n_cols)?;

            // Update running SVD with this chunk
            let (u_chunk, s_chunk, vt_chunk) = self.chunk_svd(&chunk)?;

            // Merge with running decomposition
            (u_running, s_running, vt_running) = self.merge_svd(
                (u_running, s_running, vt_running),
                (u_chunk, s_chunk, vt_chunk),
                chunk_idx == 0,
            )?;
        }

        Ok((u_running, s_running, vt_running))
    }

    fn extract_chunk(
        &self,
        mmap: &Mmap,
        start_row: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> SklearsResult<Array2<f64>> {
        let mut chunk = Array2::zeros((n_rows, n_cols));
        let bytes_per_row = n_cols * 8; // f64 = 8 bytes

        for i in 0..n_rows {
            let row_start = (start_row + i) * bytes_per_row;
            let row_end = row_start + bytes_per_row;

            if row_end <= mmap.len() {
                let row_bytes = &mmap[row_start..row_end];
                let row_data: &[f64] =
                    unsafe { std::slice::from_raw_parts(row_bytes.as_ptr() as *const f64, n_cols) };

                for j in 0..n_cols {
                    chunk[(i, j)] = row_data[j];
                }
            }
        }

        Ok(chunk)
    }

    fn chunk_svd(
        &self,
        chunk: &Array2<f64>,
    ) -> SklearsResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        // For now, use a simplified version - in a real implementation, you'd use proper SVD
        let (n_rows, n_cols) = chunk.dim();
        let n_components = std::cmp::min(self.n_components, std::cmp::min(n_rows, n_cols));

        // Create placeholder matrices - replace with actual SVD computation
        let u = Array2::zeros((n_rows, n_components));
        let s = Array2::eye(n_components);
        let vt = Array2::zeros((n_components, n_cols));

        Ok((u, s, vt))
    }

    fn merge_svd(
        &self,
        running: (Array2<f64>, Array2<f64>, Array2<f64>),
        chunk: (Array2<f64>, Array2<f64>, Array2<f64>),
        is_first: bool,
    ) -> SklearsResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        if is_first {
            return Ok(chunk);
        }

        let (u_running, s_running, _vt_running) = running;
        let (u_chunk, s_chunk, vt_chunk) = chunk;

        // Simplified merge - in practice would use more sophisticated techniques
        let n_keep = std::cmp::min(self.n_components, s_running.nrows());

        let u_merged = if u_chunk.nrows() > 0 {
            let n_components_to_take = std::cmp::min(n_keep, u_chunk.ncols());
            u_chunk.slice(s![.., ..n_components_to_take]).to_owned()
        } else {
            let n_components_to_take = std::cmp::min(n_keep, u_running.ncols());
            u_running.slice(s![.., ..n_components_to_take]).to_owned()
        };

        let s_merged = {
            let n_components_to_take = std::cmp::min(n_keep, s_chunk.nrows());
            s_chunk
                .slice(s![..n_components_to_take, ..n_components_to_take])
                .to_owned()
        };

        let vt_merged = {
            let n_components_to_take = std::cmp::min(n_keep, vt_chunk.nrows());
            vt_chunk.slice(s![..n_components_to_take, ..]).to_owned()
        };

        Ok((u_merged, s_merged, vt_merged))
    }
}

/// Memory-mapped matrix operations for efficient disk access
pub struct MemoryMappedMatrix {
    mmap: Mmap,
    shape: (usize, usize),
    dtype_size: usize,
}

impl MemoryMappedMatrix {
    pub fn open<P: AsRef<Path>>(
        path: P,
        shape: (usize, usize),
        dtype_size: usize,
    ) -> SklearsResult<Self> {
        let file = File::open(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {e}")))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to memory map: {e}")))?
        };

        Ok(Self {
            mmap,
            shape,
            dtype_size,
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn read_row(&self, row: usize) -> SklearsResult<Vec<f64>> {
        if row >= self.shape.0 {
            return Err(SklearsError::InvalidInput(
                "Row index out of bounds".to_string(),
            ));
        }

        let start = row * self.shape.1 * self.dtype_size;
        let end = start + self.shape.1 * self.dtype_size;

        if end > self.mmap.len() {
            return Err(SklearsError::InvalidInput("Invalid file size".to_string()));
        }

        let bytes = &self.mmap[start..end];
        let data: &[f64] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.shape.1) };

        Ok(data.to_vec())
    }

    pub fn read_chunk(&self, start_row: usize, n_rows: usize) -> SklearsResult<Array2<f64>> {
        if start_row + n_rows > self.shape.0 {
            return Err(SklearsError::InvalidInput(
                "Chunk extends beyond matrix".to_string(),
            ));
        }

        let mut chunk = Array2::zeros((n_rows, self.shape.1));

        for i in 0..n_rows {
            let row_data = self.read_row(start_row + i)?;
            for j in 0..self.shape.1 {
                chunk[(i, j)] = row_data[j];
            }
        }

        Ok(chunk)
    }
}

/// Chunked processing for large matrices
pub struct ChunkedProcessor {
    chunk_size: usize,
    overlap: usize,
}

impl ChunkedProcessor {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
        }
    }

    pub fn process_matrix<F, R>(
        &self,
        matrix: &Array2<f64>,
        mut process_fn: F,
    ) -> SklearsResult<Vec<R>>
    where
        F: FnMut(&Array2<f64>) -> SklearsResult<R>,
    {
        let mut results = Vec::new();
        let n_rows = matrix.nrows();
        let _n_cols = matrix.ncols();

        let effective_chunk_size = self.chunk_size - self.overlap;
        let n_chunks = (n_rows + effective_chunk_size - 1) / effective_chunk_size;

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * effective_chunk_size;
            let end = std::cmp::min(start + self.chunk_size, n_rows);

            let chunk = matrix.slice(s![start..end, ..]).to_owned();
            let result = process_fn(&chunk)?;
            results.push(result);
        }

        Ok(results)
    }

    pub fn process_columns<F, R>(
        &self,
        matrix: &Array2<f64>,
        mut process_fn: F,
    ) -> SklearsResult<Vec<R>>
    where
        F: FnMut(&Array2<f64>) -> SklearsResult<R>,
    {
        let mut results = Vec::new();
        let _n_rows = matrix.nrows();
        let n_cols = matrix.ncols();

        let effective_chunk_size = self.chunk_size - self.overlap;
        let n_chunks = (n_cols + effective_chunk_size - 1) / effective_chunk_size;

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * effective_chunk_size;
            let end = std::cmp::min(start + self.chunk_size, n_cols);

            let chunk = matrix.slice(s![.., start..end]).to_owned();
            let result = process_fn(&chunk)?;
            results.push(result);
        }

        Ok(results)
    }
}

/// Compression techniques for decomposed matrices
#[derive(Debug, Clone)]
pub enum CompressionMethod {
    /// Store only significant singular values above threshold
    SingularValueThreshold(f64),
    /// Store only top-k components
    TopK(usize),
    /// Quantize values to reduce precision
    Quantized(u8), // bits per value
    /// Block-wise compression
    BlockCompression(usize), // block size
}

pub struct CompressedDecomposition {
    u_compressed: CompressedMatrix,
    s_compressed: Vec<f64>,
    vt_compressed: CompressedMatrix,
    compression_ratio: f64,
}

impl CompressedDecomposition {
    pub fn compress(
        u: &Array2<f64>,
        s: &Array2<f64>,
        vt: &Array2<f64>,
        method: CompressionMethod,
    ) -> SklearsResult<Self> {
        let original_size = (u.len() + s.len() + vt.len()) * 8; // f64 = 8 bytes

        let (u_compressed, s_compressed, vt_compressed) = match method {
            CompressionMethod::SingularValueThreshold(threshold) => {
                Self::threshold_compression(u, s, vt, threshold)?
            }
            CompressionMethod::TopK(k) => Self::topk_compression(u, s, vt, k)?,
            CompressionMethod::Quantized(bits) => Self::quantized_compression(u, s, vt, bits)?,
            CompressionMethod::BlockCompression(block_size) => {
                Self::block_compression(u, s, vt, block_size)?
            }
        };

        let compressed_size = u_compressed.compressed_size()
            + s_compressed.len() * 8
            + vt_compressed.compressed_size();

        let compression_ratio = original_size as f64 / compressed_size as f64;

        Ok(Self {
            u_compressed,
            s_compressed,
            vt_compressed,
            compression_ratio,
        })
    }

    fn threshold_compression(
        u: &Array2<f64>,
        s: &Array2<f64>,
        vt: &Array2<f64>,
        threshold: f64,
    ) -> SklearsResult<(CompressedMatrix, Vec<f64>, CompressedMatrix)> {
        // Extract diagonal values from s matrix
        let min_dim = std::cmp::min(s.nrows(), s.ncols());
        let s_diag: Vec<f64> = (0..min_dim).map(|i| s[(i, i)]).collect();

        let significant_indices: Vec<usize> = s_diag
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > threshold)
            .map(|(i, _)| i)
            .collect();

        let k = significant_indices.len();
        if k == 0 {
            return Err(SklearsError::InvalidInput(
                "No significant singular values".to_string(),
            ));
        }

        let u_reduced = u.slice(s![.., ..k]).to_owned();
        let s_reduced: Vec<f64> = significant_indices.iter().map(|&i| s_diag[i]).collect();
        let vt_reduced = vt.slice(s![..k, ..]).to_owned();

        Ok((
            CompressedMatrix::from_dense(&u_reduced),
            s_reduced,
            CompressedMatrix::from_dense(&vt_reduced),
        ))
    }

    fn topk_compression(
        u: &Array2<f64>,
        s: &Array2<f64>,
        vt: &Array2<f64>,
        k: usize,
    ) -> SklearsResult<(CompressedMatrix, Vec<f64>, CompressedMatrix)> {
        let min_dim = std::cmp::min(s.nrows(), s.ncols());
        let actual_k = std::cmp::min(k, min_dim);

        let u_reduced = u.slice(s![.., ..actual_k]).to_owned();
        let s_reduced: Vec<f64> = (0..actual_k).map(|i| s[(i, i)]).collect();
        let vt_reduced = vt.slice(s![..actual_k, ..]).to_owned();

        Ok((
            CompressedMatrix::from_dense(&u_reduced),
            s_reduced,
            CompressedMatrix::from_dense(&vt_reduced),
        ))
    }

    fn quantized_compression(
        u: &Array2<f64>,
        s: &Array2<f64>,
        vt: &Array2<f64>,
        bits: u8,
    ) -> SklearsResult<(CompressedMatrix, Vec<f64>, CompressedMatrix)> {
        // Quantize matrices to reduce precision
        let levels = (1u64 << bits) as f64;

        let u_quantized = Self::quantize_matrix(u, levels);
        let vt_quantized = Self::quantize_matrix(vt, levels);

        let min_dim = std::cmp::min(s.nrows(), s.ncols());
        let s_diag: Vec<f64> = (0..min_dim).map(|i| s[(i, i)]).collect();

        Ok((
            CompressedMatrix::quantized(&u_quantized, bits),
            s_diag,
            CompressedMatrix::quantized(&vt_quantized, bits),
        ))
    }

    fn block_compression(
        u: &Array2<f64>,
        s: &Array2<f64>,
        vt: &Array2<f64>,
        block_size: usize,
    ) -> SklearsResult<(CompressedMatrix, Vec<f64>, CompressedMatrix)> {
        let min_dim = std::cmp::min(s.nrows(), s.ncols());
        let s_diag: Vec<f64> = (0..min_dim).map(|i| s[(i, i)]).collect();

        Ok((
            CompressedMatrix::block_compressed(u, block_size),
            s_diag,
            CompressedMatrix::block_compressed(vt, block_size),
        ))
    }

    fn quantize_matrix(matrix: &Array2<f64>, levels: f64) -> Array2<f64> {
        let (min_val, max_val) = matrix
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let scale = (max_val - min_val) / (levels - 1.0);

        matrix.map(|&val| {
            let quantized = ((val - min_val) / scale).round();
            min_val + quantized * scale
        })
    }

    pub fn compression_ratio(&self) -> f64 {
        self.compression_ratio
    }

    pub fn decompress(&self) -> SklearsResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let u = self.u_compressed.decompress()?;

        // Create diagonal matrix from singular values
        let n = self.s_compressed.len();
        let mut s = Array2::zeros((n, n));
        for i in 0..n {
            s[(i, i)] = self.s_compressed[i];
        }

        let vt = self.vt_compressed.decompress()?;

        Ok((u, s, vt))
    }
}

/// Compressed matrix representation
#[derive(Debug, Clone)]
pub enum CompressedMatrix {
    Dense(Array2<f64>),
    Sparse(Vec<(usize, usize, f64)>, (usize, usize)), // (row, col, val), shape
    Quantized(Array2<f64>, u8),                       // quantized data, bits per value
    BlockCompressed(Vec<Array2<f64>>, (usize, usize), usize), // blocks, shape, block_size
}

impl CompressedMatrix {
    pub fn from_dense(matrix: &Array2<f64>) -> Self {
        Self::Dense(matrix.clone())
    }

    pub fn sparse(data: Vec<(usize, usize, f64)>, shape: (usize, usize)) -> Self {
        Self::Sparse(data, shape)
    }

    pub fn quantized(matrix: &Array2<f64>, bits: u8) -> Self {
        Self::Quantized(matrix.clone(), bits)
    }

    pub fn block_compressed(matrix: &Array2<f64>, block_size: usize) -> Self {
        let n_rows = matrix.nrows();
        let n_cols = matrix.ncols();
        let mut blocks = Vec::new();

        for i in (0..n_rows).step_by(block_size) {
            for j in (0..n_cols).step_by(block_size) {
                let end_i = std::cmp::min(i + block_size, n_rows);
                let end_j = std::cmp::min(j + block_size, n_cols);
                let block = matrix.slice(s![i..end_i, j..end_j]).to_owned();
                blocks.push(block);
            }
        }

        Self::BlockCompressed(blocks, (n_rows, n_cols), block_size)
    }

    pub fn compressed_size(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.len() * 8,
            Self::Sparse(data, _) => data.len() * (8 + 8 + 8), // row, col, val indices
            Self::Quantized(matrix, bits) => matrix.len() * (*bits as usize / 8).max(1),
            Self::BlockCompressed(blocks, _, _) => blocks.iter().map(|b| b.len() * 8).sum(),
        }
    }

    pub fn decompress(&self) -> SklearsResult<Array2<f64>> {
        match self {
            Self::Dense(matrix) => Ok(matrix.clone()),
            Self::Sparse(data, shape) => {
                let mut matrix = Array2::zeros(*shape);
                for &(i, j, val) in data {
                    matrix[(i, j)] = val;
                }
                Ok(matrix)
            }
            Self::Quantized(matrix, _) => Ok(matrix.clone()),
            Self::BlockCompressed(blocks, shape, block_size) => {
                let mut matrix = Array2::zeros(*shape);
                let mut block_idx = 0;

                for i in (0..shape.0).step_by(*block_size) {
                    for j in (0..shape.1).step_by(*block_size) {
                        if block_idx < blocks.len() {
                            let block = &blocks[block_idx];
                            let end_i = std::cmp::min(i + block_size, shape.0);
                            let end_j = std::cmp::min(j + block_size, shape.1);

                            for (bi, mi) in (i..end_i).enumerate() {
                                for (bj, mj) in (j..end_j).enumerate() {
                                    if bi < block.nrows() && bj < block.ncols() {
                                        matrix[(mi, mj)] = block[(bi, bj)];
                                    }
                                }
                            }
                            block_idx += 1;
                        }
                    }
                }

                Ok(matrix)
            }
        }
    }
}

/// Lazy evaluation for decomposition chains
pub struct LazyDecomposition<T> {
    operations: Vec<DecompositionOperation<T>>,
    cached_result: Option<T>,
}

impl<T: Clone> LazyDecomposition<T> {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            cached_result: None,
        }
    }

    pub fn add_operation<F>(mut self, operation: F) -> Self
    where
        F: Fn(&T) -> SklearsResult<T> + 'static,
    {
        self.operations.push(Box::new(operation));
        self.cached_result = None; // Invalidate cache
        self
    }

    pub fn evaluate(&mut self, input: &T) -> SklearsResult<T> {
        if let Some(ref result) = self.cached_result {
            return Ok(result.clone());
        }

        let mut current = input.clone();
        for operation in &self.operations {
            current = operation(&current)?;
        }

        self.cached_result = Some(current.clone());
        Ok(current)
    }

    pub fn clear_cache(&mut self) {
        self.cached_result = None;
    }
}

impl<T: Clone> Default for LazyDecomposition<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_chunked_processor() {
        let matrix = Array2::from_shape_fn((100, 50), |(i, j)| (i * j) as f64);
        let processor = ChunkedProcessor::new(25, 5);

        let results = processor
            .process_matrix(&matrix, |chunk| Ok(chunk.sum()))
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results.len(), 5); // (100 + 20 - 1) / 20 = 5 chunks
    }

    #[test]
    fn test_compression_topk() {
        let u = Array2::from_shape_fn((10, 5), |(i, j)| (i + j) as f64);
        let s = Array2::eye(5);
        let vt = Array2::from_shape_fn((5, 8), |(i, j)| (i * j) as f64);

        let compressed =
            CompressedDecomposition::compress(&u, &s, &vt, CompressionMethod::TopK(3)).unwrap();

        assert!(compressed.compression_ratio() > 1.0);

        let (u_dec, s_dec, vt_dec) = compressed.decompress().unwrap();
        assert_eq!(u_dec.ncols(), 3);
        assert_eq!(vt_dec.nrows(), 3);
    }

    #[test]
    fn test_lazy_decomposition() {
        let mut lazy = LazyDecomposition::new()
            .add_operation(|x: &f64| Ok(x * 2.0))
            .add_operation(|x: &f64| Ok(x + 1.0))
            .add_operation(|x: &f64| Ok(x * x));

        let result = lazy.evaluate(&3.0).unwrap();
        assert_eq!(result, 49.0); // ((3 * 2) + 1)^2 = 7^2 = 49

        // Test caching
        let result2 = lazy.evaluate(&3.0).unwrap();
        assert_eq!(result2, 49.0);
    }

    #[test]
    fn test_compressed_matrix_sparse() {
        let data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let compressed = CompressedMatrix::sparse(data, (3, 3));

        let decompressed = compressed.decompress().unwrap();
        assert_eq!(decompressed[(0, 0)], 1.0);
        assert_eq!(decompressed[(1, 1)], 2.0);
        assert_eq!(decompressed[(2, 2)], 3.0);
        assert_eq!(decompressed[(0, 1)], 0.0);
    }
}
