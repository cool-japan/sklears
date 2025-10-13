//! Compressed kernel representations for memory-efficient SVM training
//!
//! This module provides various compression techniques for kernel matrices to reduce
//! memory usage while maintaining acceptable approximation quality. The approaches include:
//! - Low-rank approximations
//! - Quantization methods
//! - Sparse representations
//! - Hierarchical compression

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::convert::TryFrom;

/// Configuration for compressed kernel representations
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression method to use
    pub method: CompressionMethod,
    /// Target compression ratio (0.0 to 1.0)
    pub compression_ratio: Float,
    /// Quality threshold for lossy compression
    pub quality_threshold: Float,
    /// Number of components for low-rank approximations
    pub num_components: Option<usize>,
    /// Quantization levels for quantized compression
    pub quantization_levels: usize,
    /// Block size for hierarchical compression
    pub block_size: usize,
    /// Whether to use adaptive compression
    pub adaptive: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            method: CompressionMethod::LowRank,
            compression_ratio: 0.1,
            quality_threshold: 1e-6,
            num_components: None,
            quantization_levels: 256,
            block_size: 1000,
            adaptive: true,
        }
    }
}

/// Compression methods for kernel matrices
#[derive(Debug, Clone, Copy)]
pub enum CompressionMethod {
    /// Low-rank matrix approximation using SVD
    LowRank,
    /// Quantization-based compression
    Quantized,
    /// Sparse matrix representation
    Sparse,
    /// Hierarchical block compression
    Hierarchical,
    /// Adaptive compression combining multiple methods
    Adaptive,
    /// Nyström method for low-rank approximation
    Nystrom,
    /// Random Fourier features
    RandomFourier,
}

/// Compressed kernel matrix trait
pub trait CompressedKernelMatrix: Send + Sync {
    /// Get kernel value at position (i, j)
    fn get(&self, i: usize, j: usize) -> Result<Float>;

    /// Get a block of kernel values
    fn get_block(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<Array2<Float>>;

    /// Get matrix dimensions
    fn dimensions(&self) -> (usize, usize);

    /// Get compression ratio achieved
    fn compression_ratio(&self) -> Float;

    /// Get approximation error
    fn approximation_error(&self) -> Float;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// Low-rank compressed kernel matrix using SVD
pub struct LowRankKernelMatrix {
    left_factors: Array2<Float>,
    right_factors: Array2<Float>,
    singular_values: Array1<Float>,
    dimensions: (usize, usize),
    approximation_error: Float,
    original_memory: usize,
}

impl LowRankKernelMatrix {
    /// Create low-rank approximation of kernel matrix
    pub fn new(kernel_matrix: &Array2<Float>, num_components: usize) -> Result<Self> {
        let dimensions = kernel_matrix.dim();
        let original_memory = dimensions.0 * dimensions.1 * std::mem::size_of::<Float>();

        // Perform SVD
        let (mut u, mut s, mut vt) = Self::svd_decomposition(kernel_matrix, num_components)?;

        // Determine the number of components that actually provide compression.
        let (m, n) = dimensions;
        let mut effective_components = s.len();
        let element_size = std::mem::size_of::<Float>();
        let mut compressed_memory =
            (m * effective_components + n * effective_components) * element_size;

        while effective_components > 1 && compressed_memory >= original_memory {
            effective_components -= 1;
            compressed_memory =
                (m * effective_components + n * effective_components) * element_size;
        }

        if effective_components != s.len() {
            u = u.slice(s![.., ..effective_components]).to_owned();
            s = s.slice(s![..effective_components]).to_owned();
            vt = vt.slice(s![..effective_components, ..]).to_owned();
        }

        // Compute approximation error
        let approximation_error = Self::compute_approximation_error(kernel_matrix, &u, &s, &vt)?;

        // Store scaled factors to avoid keeping the diagonal separately
        let mut left_factors = u;
        let mut right_factors = vt;
        for (idx, singular) in s.iter().enumerate() {
            if *singular <= 0.0 {
                left_factors.column_mut(idx).fill(0.0);
                right_factors.row_mut(idx).fill(0.0);
            } else {
                let scale = singular.sqrt();
                left_factors.column_mut(idx).mapv_inplace(|val| val * scale);
                right_factors.row_mut(idx).mapv_inplace(|val| val * scale);
            }
        }

        Ok(Self {
            left_factors,
            right_factors,
            singular_values: s,
            dimensions,
            approximation_error,
            original_memory,
        })
    }

    /// Perform truncated SVD decomposition
    fn svd_decomposition(
        matrix: &Array2<Float>,
        k: usize,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        // For this implementation, we'll use a simplified approach
        // In practice, you would use more sophisticated SVD libraries like LAPACK
        let (m, n) = matrix.dim();
        let rank = k.min(m).min(n);

        // Compute eigendecomposition of A^T A for right singular vectors
        let ata = matrix.t().dot(matrix);
        let (eigenvals, eigenvecs) = Self::compute_eigenpairs(&ata, rank)?;

        // Compute singular values
        let mut s = Array1::zeros(rank);
        for i in 0..rank {
            s[i] = eigenvals[i].sqrt();
        }

        // Compute left singular vectors
        let mut u = Array2::zeros((m, rank));
        for i in 0..rank {
            if s[i] > 1e-12 {
                let v_col = eigenvecs.column(i);
                let u_col = matrix.dot(&v_col) / s[i];
                u.column_mut(i).assign(&u_col);
            }
        }

        // Right singular vectors are eigenvectors
        let vt = eigenvecs.t().to_owned();

        Ok((u, s, vt.slice(s![..rank, ..]).to_owned()))
    }

    /// Simplified eigendecomposition (in practice, use LAPACK)
    fn compute_eigenpairs(
        matrix: &Array2<Float>,
        k: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let mut eigenvals = Array1::zeros(k);
        let mut eigenvecs = Array2::zeros((n, k));

        // Power iteration for largest eigenvalues/eigenvectors
        for i in 0..k {
            let mut v = Array1::from_elem(n, 1.0 / (n as Float).sqrt());

            // Deflate by previous eigenvectors
            for j in 0..i {
                let proj = v.dot(&eigenvecs.column(j));
                v = &v - &(&eigenvecs.column(j).to_owned() * proj);
            }

            // Power iteration
            for _ in 0..100 {
                let mut v_new = matrix.dot(&v);

                // Deflate
                for j in 0..i {
                    let proj = v_new.dot(&eigenvecs.column(j));
                    v_new = &v_new - &(&eigenvecs.column(j).to_owned() * proj);
                }

                let norm = v_new.iter().map(|x| x * x).sum::<Float>().sqrt();
                if norm < 1e-12 {
                    break;
                }
                v_new /= norm;

                if (&v_new - &v).iter().map(|x| x.abs()).sum::<Float>() < 1e-8 {
                    break;
                }
                v = v_new;
            }

            let eigenval = v.dot(&matrix.dot(&v));
            eigenvals[i] = eigenval;
            eigenvecs.column_mut(i).assign(&v);
        }

        Ok((eigenvals, eigenvecs))
    }

    /// Compute approximation error
    fn compute_approximation_error(
        original: &Array2<Float>,
        u: &Array2<Float>,
        s: &Array1<Float>,
        vt: &Array2<Float>,
    ) -> Result<Float> {
        let (m, n) = original.dim();
        let mut total_error = 0.0;
        let mut count = 0;

        // Sample-based error computation to avoid full reconstruction
        for i in (0..m).step_by(m / 100 + 1) {
            for j in (0..n).step_by(n / 100 + 1) {
                let original_val = original[[i, j]];
                let mut approx_val = 0.0;

                for k in 0..u.ncols() {
                    approx_val += u[[i, k]] * s[k] * vt[[k, j]];
                }

                total_error += (original_val - approx_val).powi(2);
                count += 1;
            }
        }

        Ok((total_error / count as Float).sqrt())
    }
}

impl CompressedKernelMatrix for LowRankKernelMatrix {
    fn get(&self, i: usize, j: usize) -> Result<Float> {
        if i >= self.dimensions.0 || j >= self.dimensions.1 {
            return Err(SklearsError::InvalidInput(
                "Index out of bounds".to_string(),
            ));
        }

        let mut value = 0.0;
        for k in 0..self.left_factors.ncols() {
            value += self.left_factors[[i, k]] * self.right_factors[[k, j]];
        }

        Ok(value)
    }

    fn get_block(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<Array2<Float>> {
        let block_rows = row_end - row_start;
        let block_cols = col_end - col_start;
        let mut block = Array2::zeros((block_rows, block_cols));

        let u_block = self.left_factors.slice(s![row_start..row_end, ..]);
        let vt_block = self.right_factors.slice(s![.., col_start..col_end]);

        for k in 0..self.left_factors.ncols() {
            let u_col = u_block.column(k);
            let vt_row = vt_block.row(k);

            for i in 0..block_rows {
                for j in 0..block_cols {
                    block[[i, j]] += u_col[i] * vt_row[j];
                }
            }
        }

        Ok(block)
    }

    fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    fn compression_ratio(&self) -> Float {
        let compressed_memory =
            (self.left_factors.len() + self.right_factors.len()) * std::mem::size_of::<Float>();
        compressed_memory as Float / self.original_memory as Float
    }

    fn approximation_error(&self) -> Float {
        self.approximation_error
    }

    fn memory_usage(&self) -> usize {
        (self.left_factors.len() + self.right_factors.len()) * std::mem::size_of::<Float>()
    }
}

/// Quantized kernel matrix for discrete compression
pub struct QuantizedKernelMatrix {
    quantized_data: Vec<u8>,
    min_value: Float,
    max_value: Float,
    quantization_levels: usize,
    dimensions: (usize, usize),
    original_memory: usize,
}

impl QuantizedKernelMatrix {
    /// Create quantized representation of kernel matrix
    pub fn new(kernel_matrix: &Array2<Float>, quantization_levels: usize) -> Result<Self> {
        let dimensions = kernel_matrix.dim();
        let original_memory = dimensions.0 * dimensions.1 * std::mem::size_of::<Float>();

        let min_value = kernel_matrix.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_value = kernel_matrix
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        let range = max_value - min_value;
        let scale = (quantization_levels - 1) as Float / range;

        let quantized_data: Vec<u8> = kernel_matrix
            .iter()
            .map(|&val| {
                let normalized = (val - min_value) * scale;
                normalized
                    .round()
                    .min((quantization_levels - 1) as Float)
                    .max(0.0) as u8
            })
            .collect();

        Ok(Self {
            quantized_data,
            min_value,
            max_value,
            quantization_levels,
            dimensions,
            original_memory,
        })
    }

    /// Dequantize a value
    fn dequantize(&self, quantized: u8) -> Float {
        let range = self.max_value - self.min_value;
        let scale = range / (self.quantization_levels - 1) as Float;
        self.min_value + quantized as Float * scale
    }
}

impl CompressedKernelMatrix for QuantizedKernelMatrix {
    fn get(&self, i: usize, j: usize) -> Result<Float> {
        if i >= self.dimensions.0 || j >= self.dimensions.1 {
            return Err(SklearsError::InvalidInput(
                "Index out of bounds".to_string(),
            ));
        }

        let idx = i * self.dimensions.1 + j;
        let quantized = self.quantized_data[idx];
        Ok(self.dequantize(quantized))
    }

    fn get_block(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<Array2<Float>> {
        let block_rows = row_end - row_start;
        let block_cols = col_end - col_start;
        let mut block = Array2::zeros((block_rows, block_cols));

        for i in 0..block_rows {
            for j in 0..block_cols {
                let global_i = row_start + i;
                let global_j = col_start + j;
                let idx = global_i * self.dimensions.1 + global_j;
                block[[i, j]] = self.dequantize(self.quantized_data[idx]);
            }
        }

        Ok(block)
    }

    fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    fn compression_ratio(&self) -> Float {
        let compressed_memory = self.quantized_data.len() * std::mem::size_of::<u8>() +
                               2 * std::mem::size_of::<Float>() + // min/max values
                               std::mem::size_of::<usize>(); // quantization levels
        compressed_memory as Float / self.original_memory as Float
    }

    fn approximation_error(&self) -> Float {
        let range = self.max_value - self.min_value;
        range / (self.quantization_levels - 1) as Float / 2.0 // Maximum quantization error
    }

    fn memory_usage(&self) -> usize {
        self.quantized_data.len() * std::mem::size_of::<u8>()
            + 2 * std::mem::size_of::<Float>()
            + std::mem::size_of::<usize>()
    }
}

/// Sparse kernel matrix representation
pub struct SparseKernelMatrix {
    values: Vec<Float>,
    row_indices: Vec<u32>,
    col_indices: Vec<u32>,
    dimensions: (usize, usize),
    sparsity_threshold: Float,
    original_memory: usize,
}

impl SparseKernelMatrix {
    /// Create sparse representation of kernel matrix
    pub fn new(kernel_matrix: &Array2<Float>, sparsity_threshold: Float) -> Result<Self> {
        let dimensions = kernel_matrix.dim();
        let original_memory = dimensions.0 * dimensions.1 * std::mem::size_of::<Float>();

        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for i in 0..dimensions.0 {
            for j in 0..dimensions.1 {
                let val = kernel_matrix[[i, j]];
                if val.abs() >= sparsity_threshold {
                    values.push(val);
                    row_indices.push(u32::try_from(i).map_err(|_| {
                        SklearsError::InvalidInput(
                            "Sparse kernel row index exceeds 32-bit storage".to_string(),
                        )
                    })?);
                    col_indices.push(u32::try_from(j).map_err(|_| {
                        SklearsError::InvalidInput(
                            "Sparse kernel column index exceeds 32-bit storage".to_string(),
                        )
                    })?);
                }
            }
        }

        Ok(Self {
            values,
            row_indices,
            col_indices,
            dimensions,
            sparsity_threshold,
            original_memory,
        })
    }

    /// Find index for (i, j) in sparse representation
    fn find_index(&self, i: usize, j: usize) -> Option<usize> {
        for (idx, (&row, &col)) in self.row_indices.iter().zip(&self.col_indices).enumerate() {
            if row as usize == i && col as usize == j {
                return Some(idx);
            }
        }
        None
    }
}

impl CompressedKernelMatrix for SparseKernelMatrix {
    fn get(&self, i: usize, j: usize) -> Result<Float> {
        if i >= self.dimensions.0 || j >= self.dimensions.1 {
            return Err(SklearsError::InvalidInput(
                "Index out of bounds".to_string(),
            ));
        }

        if let Some(idx) = self.find_index(i, j) {
            Ok(self.values[idx])
        } else {
            Ok(0.0) // Implicit zero
        }
    }

    fn get_block(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<Array2<Float>> {
        let block_rows = row_end - row_start;
        let block_cols = col_end - col_start;
        let mut block = Array2::zeros((block_rows, block_cols));

        for (idx, (&row, &col)) in self.row_indices.iter().zip(&self.col_indices).enumerate() {
            let row = row as usize;
            let col = col as usize;
            if row >= row_start && row < row_end && col >= col_start && col < col_end {
                block[[row - row_start, col - col_start]] = self.values[idx];
            }
        }

        Ok(block)
    }

    fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    fn compression_ratio(&self) -> Float {
        let compressed_memory = self.values.len() * std::mem::size_of::<Float>()
            + self.row_indices.len() * std::mem::size_of::<u32>()
            + self.col_indices.len() * std::mem::size_of::<u32>();
        compressed_memory as Float / self.original_memory as Float
    }

    fn approximation_error(&self) -> Float {
        self.sparsity_threshold
    }

    fn memory_usage(&self) -> usize {
        self.values.len() * std::mem::size_of::<Float>()
            + self.row_indices.len() * std::mem::size_of::<u32>()
            + self.col_indices.len() * std::mem::size_of::<u32>()
    }
}

/// Hierarchical compressed kernel matrix using block compression
pub struct HierarchicalKernelMatrix {
    blocks: Vec<Vec<Box<dyn CompressedKernelMatrix>>>,
    block_size: usize,
    dimensions: (usize, usize),
    compression_config: CompressionConfig,
}

impl HierarchicalKernelMatrix {
    /// Create hierarchical compression of kernel matrix
    pub fn new(kernel_matrix: &Array2<Float>, config: CompressionConfig) -> Result<Self> {
        let dimensions = kernel_matrix.dim();
        let block_size = config.block_size;

        let num_block_rows = (dimensions.0 + block_size - 1) / block_size;
        let num_block_cols = (dimensions.1 + block_size - 1) / block_size;

        let mut blocks = Vec::with_capacity(num_block_rows);

        for i in 0..num_block_rows {
            let mut block_row = Vec::with_capacity(num_block_cols);

            for j in 0..num_block_cols {
                let row_start = i * block_size;
                let row_end = ((i + 1) * block_size).min(dimensions.0);
                let col_start = j * block_size;
                let col_end = ((j + 1) * block_size).min(dimensions.1);

                let block = kernel_matrix.slice(s![row_start..row_end, col_start..col_end]);
                let compressed_block = Self::compress_block(&block.to_owned(), &config)?;
                block_row.push(compressed_block);
            }

            blocks.push(block_row);
        }

        Ok(Self {
            blocks,
            block_size,
            dimensions,
            compression_config: config,
        })
    }

    /// Compress individual block based on configuration
    fn compress_block(
        block: &Array2<Float>,
        config: &CompressionConfig,
    ) -> Result<Box<dyn CompressedKernelMatrix>> {
        match config.method {
            CompressionMethod::LowRank => {
                let k = config
                    .num_components
                    .unwrap_or(block.nrows().min(block.ncols()) / 4);
                Ok(Box::new(LowRankKernelMatrix::new(block, k)?))
            }
            CompressionMethod::Quantized => Ok(Box::new(QuantizedKernelMatrix::new(
                block,
                config.quantization_levels,
            )?)),
            CompressionMethod::Sparse => Ok(Box::new(SparseKernelMatrix::new(
                block,
                config.quality_threshold,
            )?)),
            CompressionMethod::Adaptive => {
                // Choose best compression method for this block
                Self::adaptive_compression(block, config)
            }
            _ => {
                // Default to low-rank
                let k = config
                    .num_components
                    .unwrap_or(block.nrows().min(block.ncols()) / 4);
                Ok(Box::new(LowRankKernelMatrix::new(block, k)?))
            }
        }
    }

    /// Adaptive compression choosing the best method
    fn adaptive_compression(
        block: &Array2<Float>,
        config: &CompressionConfig,
    ) -> Result<Box<dyn CompressedKernelMatrix>> {
        // Try different compression methods and choose the best
        let methods = [
            CompressionMethod::LowRank,
            CompressionMethod::Quantized,
            CompressionMethod::Sparse,
        ];

        let mut best_compression: Option<Box<dyn CompressedKernelMatrix>> = None;
        let mut best_score = Float::NEG_INFINITY;

        for method in &methods {
            let mut test_config = config.clone();
            test_config.method = *method;

            if let Ok(compressed) = Self::compress_block(block, &test_config) {
                // Score based on compression ratio and error
                let compression_ratio = compressed.compression_ratio();
                let error = compressed.approximation_error();
                let score = compression_ratio - error * 10.0; // Weight error more heavily

                if score > best_score {
                    best_score = score;
                    best_compression = Some(compressed);
                }
            }
        }

        best_compression.ok_or_else(|| {
            SklearsError::InvalidInput("No compression method succeeded".to_string())
        })
    }

    /// Get block indices for global position
    fn get_block_indices(&self, i: usize, j: usize) -> (usize, usize, usize, usize) {
        let block_i = i / self.block_size;
        let block_j = j / self.block_size;
        let local_i = i % self.block_size;
        let local_j = j % self.block_size;
        (block_i, block_j, local_i, local_j)
    }
}

impl CompressedKernelMatrix for HierarchicalKernelMatrix {
    fn get(&self, i: usize, j: usize) -> Result<Float> {
        if i >= self.dimensions.0 || j >= self.dimensions.1 {
            return Err(SklearsError::InvalidInput(
                "Index out of bounds".to_string(),
            ));
        }

        let (block_i, block_j, local_i, local_j) = self.get_block_indices(i, j);
        self.blocks[block_i][block_j].get(local_i, local_j)
    }

    fn get_block(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<Array2<Float>> {
        let block_rows = row_end - row_start;
        let block_cols = col_end - col_start;
        let mut result = Array2::zeros((block_rows, block_cols));

        for i in 0..block_rows {
            for j in 0..block_cols {
                result[[i, j]] = self.get(row_start + i, col_start + j)?;
            }
        }

        Ok(result)
    }

    fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    fn compression_ratio(&self) -> Float {
        let total_memory: usize = self
            .blocks
            .iter()
            .flat_map(|row| row.iter())
            .map(|block| block.memory_usage())
            .sum();

        let original_memory = self.dimensions.0 * self.dimensions.1 * std::mem::size_of::<Float>();
        total_memory as Float / original_memory as Float
    }

    fn approximation_error(&self) -> Float {
        // Average error across all blocks
        let total_error: Float = self
            .blocks
            .iter()
            .flat_map(|row| row.iter())
            .map(|block| block.approximation_error())
            .sum();

        let num_blocks = self.blocks.len() * self.blocks[0].len();
        total_error / num_blocks as Float
    }

    fn memory_usage(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|row| row.iter())
            .map(|block| block.memory_usage())
            .sum()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_rank_compression() {
        let matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.8, 0.6, 0.4, 0.8, 1.0, 0.7, 0.5, 0.6, 0.7, 1.0, 0.6, 0.4, 0.5, 0.6, 1.0,
            ],
        )
        .unwrap();

        let compressed = LowRankKernelMatrix::new(&matrix, 2).unwrap();
        assert_eq!(compressed.dimensions(), (4, 4));
        assert!(compressed.compression_ratio() < 1.0);
    }

    #[test]
    fn test_quantized_compression() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0])
                .unwrap();

        let compressed = QuantizedKernelMatrix::new(&matrix, 16).unwrap();
        assert_eq!(compressed.dimensions(), (3, 3));
        assert!(compressed.compression_ratio() < 1.0);
    }

    #[test]
    fn test_sparse_compression() {
        let matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.8, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();

        let compressed = SparseKernelMatrix::new(&matrix, 0.1).unwrap();
        assert_eq!(compressed.dimensions(), (4, 4));
        assert!(compressed.compression_ratio() < 1.0);
    }

    #[test]
    fn test_hierarchical_compression() {
        let matrix = Array2::eye(8);
        let mut config = CompressionConfig::default();
        config.block_size = 4;
        config.method = CompressionMethod::LowRank;

        let compressed = HierarchicalKernelMatrix::new(&matrix, config).unwrap();
        assert_eq!(compressed.dimensions(), (8, 8));
    }
}
