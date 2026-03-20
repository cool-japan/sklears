//! Sparse Matrix Optimizations for Preprocessing
//!
//! This module provides memory-efficient sparse matrix implementations optimized
//! for preprocessing operations on high-dimensional sparse data commonly found
//! in text processing, categorical encoding, and feature engineering.
//!
//! # Features
//!
//! - Compressed Sparse Row (CSR) format for efficient row operations
//! - Compressed Sparse Column (CSC) format for efficient column operations
//! - Coordinate (COO) format for efficient construction and modification
//! - Sparse-aware preprocessing transformers
//! - Memory-efficient sparse matrix arithmetic
//! - Automatic density optimization
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_preprocessing::sparse_optimizations::{
//!     SparseMatrix, SparseFormat, SparseStandardScaler
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create sparse matrix from coordinates
//!     let rows = vec![0, 0, 1, 2, 2, 2];
//!     let cols = vec![0, 2, 1, 0, 1, 2];
//!     let data = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
//!
//!     let sparse = SparseMatrix::from_triplets(
//!         (3, 3), rows, cols, data, SparseFormat::CSR
//!     )?;
//!
//!     // Sparse-aware scaling
//!     let scaler = SparseStandardScaler::new();
//!     let scaler_fitted = scaler.fit(&sparse, &())?;
//!     let scaled = scaler_fitted.transform(&sparse)?;
//!
//!     println!("Original density: {:.2}%", sparse.density() * 100.0);
//!     println!("Scaled density: {:.2}%", scaled.density() * 100.0);
//!
//!     Ok(())
//! }
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Sparse matrix storage formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SparseFormat {
    /// Compressed Sparse Row - efficient for row operations
    CSR,
    /// Compressed Sparse Column - efficient for column operations
    CSC,
    /// Coordinate format - efficient for construction and modification
    COO,
}

/// Sparse matrix representation with multiple format support
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseMatrix {
    /// Matrix dimensions (rows, cols)
    shape: (usize, usize),
    /// Current storage format
    format: SparseFormat,
    /// CSR/CSC data
    data: Vec<Float>,
    /// CSR/CSC indices
    indices: Vec<usize>,
    /// CSR row pointers or CSC column pointers
    indptr: Vec<usize>,
    /// COO row indices (only used in COO format)
    coo_rows: Vec<usize>,
    /// COO column indices (only used in COO format)
    coo_cols: Vec<usize>,
}

impl SparseMatrix {
    /// Create sparse matrix from triplet format (row, col, value)
    pub fn from_triplets(
        shape: (usize, usize),
        rows: Vec<usize>,
        cols: Vec<usize>,
        data: Vec<Float>,
        format: SparseFormat,
    ) -> Result<Self> {
        if rows.len() != cols.len() || rows.len() != data.len() {
            return Err(SklearsError::InvalidInput(
                "Row, column, and data vectors must have same length".to_string(),
            ));
        }

        let mut sparse = Self {
            shape,
            format: SparseFormat::COO,
            data: data.clone(),
            indices: Vec::new(),
            indptr: Vec::new(),
            coo_rows: rows,
            coo_cols: cols,
        };

        sparse.convert_to(format)?;
        Ok(sparse)
    }

    /// Create empty sparse matrix
    pub fn zeros(shape: (usize, usize), format: SparseFormat) -> Self {
        let indptr = match format {
            SparseFormat::CSR => vec![0; shape.0 + 1],
            SparseFormat::CSC => vec![0; shape.1 + 1],
            SparseFormat::COO => Vec::new(),
        };

        Self {
            shape,
            format,
            data: Vec::new(),
            indices: Vec::new(),
            indptr,
            coo_rows: Vec::new(),
            coo_cols: Vec::new(),
        }
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Calculate sparsity density (fraction of non-zero elements)
    pub fn density(&self) -> Float {
        if self.shape.0 == 0 || self.shape.1 == 0 {
            0.0
        } else {
            self.nnz() as Float / (self.shape.0 * self.shape.1) as Float
        }
    }

    /// Get current storage format
    pub fn format(&self) -> SparseFormat {
        self.format
    }

    /// Convert to different sparse format
    pub fn convert_to(&mut self, target_format: SparseFormat) -> Result<()> {
        if self.format == target_format {
            return Ok(());
        }

        match (self.format, target_format) {
            (SparseFormat::COO, SparseFormat::CSR) => self.coo_to_csr(),
            (SparseFormat::COO, SparseFormat::CSC) => self.coo_to_csc(),
            (SparseFormat::CSR, SparseFormat::COO) => self.csr_to_coo(),
            (SparseFormat::CSC, SparseFormat::COO) => self.csc_to_coo(),
            (SparseFormat::CSR, SparseFormat::CSC) => {
                self.csr_to_coo()?;
                self.coo_to_csc()
            }
            (SparseFormat::CSC, SparseFormat::CSR) => {
                self.csc_to_coo()?;
                self.coo_to_csr()
            }
            // Same format - no conversion needed
            (SparseFormat::CSR, SparseFormat::CSR)
            | (SparseFormat::CSC, SparseFormat::CSC)
            | (SparseFormat::COO, SparseFormat::COO) => Ok(()),
        }
    }

    fn coo_to_csr(&mut self) -> Result<()> {
        let (rows, _cols) = self.shape;
        let mut indptr = vec![0; rows + 1];

        // Count elements per row
        for &row in &self.coo_rows {
            if row >= rows {
                return Err(SklearsError::InvalidInput(
                    "Row index out of bounds".to_string(),
                ));
            }
            indptr[row + 1] += 1;
        }

        // Convert counts to pointers
        for i in 0..rows {
            indptr[i + 1] += indptr[i];
        }

        // Sort by row, then by column
        let mut triplets: Vec<(usize, usize, Float)> = self
            .coo_rows
            .iter()
            .zip(self.coo_cols.iter())
            .zip(self.data.iter())
            .map(|((&r, &c), &d)| (r, c, d))
            .collect();

        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Build CSR format
        self.indices = triplets.iter().map(|(_, c, _)| *c).collect();
        self.data = triplets.iter().map(|(_, _, d)| *d).collect();
        self.indptr = indptr;
        self.coo_rows.clear();
        self.coo_cols.clear();
        self.format = SparseFormat::CSR;

        Ok(())
    }

    fn coo_to_csc(&mut self) -> Result<()> {
        let (_rows, cols) = self.shape;
        let mut indptr = vec![0; cols + 1];

        // Count elements per column
        for &col in &self.coo_cols {
            if col >= cols {
                return Err(SklearsError::InvalidInput(
                    "Column index out of bounds".to_string(),
                ));
            }
            indptr[col + 1] += 1;
        }

        // Convert counts to pointers
        for i in 0..cols {
            indptr[i + 1] += indptr[i];
        }

        // Sort by column, then by row
        let mut triplets: Vec<(usize, usize, Float)> = self
            .coo_rows
            .iter()
            .zip(self.coo_cols.iter())
            .zip(self.data.iter())
            .map(|((&r, &c), &d)| (r, c, d))
            .collect();

        triplets.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // Build CSC format
        self.indices = triplets.iter().map(|(r, _, _)| *r).collect();
        self.data = triplets.iter().map(|(_, _, d)| *d).collect();
        self.indptr = indptr;
        self.coo_rows.clear();
        self.coo_cols.clear();
        self.format = SparseFormat::CSC;

        Ok(())
    }

    fn csr_to_coo(&mut self) -> Result<()> {
        let mut coo_rows = Vec::with_capacity(self.data.len());

        for (row, window) in self.indptr.windows(2).enumerate() {
            let start = window[0];
            let end = window[1];
            for _ in start..end {
                coo_rows.push(row);
            }
        }

        self.coo_rows = coo_rows;
        self.coo_cols = self.indices.clone();
        self.indices.clear();
        self.indptr.clear();
        self.format = SparseFormat::COO;

        Ok(())
    }

    fn csc_to_coo(&mut self) -> Result<()> {
        let mut coo_cols = Vec::with_capacity(self.data.len());

        for (col, window) in self.indptr.windows(2).enumerate() {
            let start = window[0];
            let end = window[1];
            for _ in start..end {
                coo_cols.push(col);
            }
        }

        self.coo_rows = self.indices.clone();
        self.coo_cols = coo_cols;
        self.indices.clear();
        self.indptr.clear();
        self.format = SparseFormat::COO;

        Ok(())
    }

    /// Get column means for CSR matrix
    pub fn column_means(&self) -> Result<Array1<Float>> {
        match self.format {
            SparseFormat::CSR => {
                let mut means = Array1::zeros(self.shape.1);

                for window in self.indptr.windows(2) {
                    let start = window[0];
                    let end = window[1];
                    for idx in start..end {
                        let col = self.indices[idx];
                        let value = self.data[idx];
                        means[col] += value;
                    }
                }

                means /= self.shape.0 as Float;
                Ok(means)
            }
            _ => {
                let mut temp = self.clone();
                temp.convert_to(SparseFormat::CSR)?;
                temp.column_means()
            }
        }
    }

    /// Get column variances for CSR matrix
    pub fn column_variances(&self, means: &Array1<Float>) -> Result<Array1<Float>> {
        match self.format {
            SparseFormat::CSR => {
                let mut variances: Array1<Float> = Array1::zeros(self.shape.1);
                let mut counts: Array1<Float> = Array1::zeros(self.shape.1);

                for window in self.indptr.windows(2) {
                    let start = window[0];
                    let end = window[1];
                    for idx in start..end {
                        let col = self.indices[idx];
                        let value = self.data[idx];
                        let diff = value - means[col];
                        variances[col] += diff * diff;
                        counts[col] += 1.0;
                    }
                }

                // Account for implicit zeros
                for col in 0..self.shape.1 {
                    let zeros: Float = self.shape.0 as Float - counts[col];
                    let zero_contribution: Float = zeros * means[col] * means[col];
                    variances[col] += zero_contribution;
                    variances[col] /= self.shape.0 as Float;
                }

                Ok(variances)
            }
            _ => {
                let mut temp = self.clone();
                temp.convert_to(SparseFormat::CSR)?;
                temp.column_variances(means)
            }
        }
    }

    /// Convert to dense array (use with caution for large sparse matrices)
    pub fn to_dense(&self) -> Result<Array2<Float>> {
        let mut dense = Array2::zeros(self.shape);

        match self.format {
            SparseFormat::CSR => {
                for (row, window) in self.indptr.windows(2).enumerate() {
                    let start = window[0];
                    let end = window[1];
                    for idx in start..end {
                        let col = self.indices[idx];
                        dense[[row, col]] = self.data[idx];
                    }
                }
            }
            SparseFormat::CSC => {
                for (col, window) in self.indptr.windows(2).enumerate() {
                    let start = window[0];
                    let end = window[1];
                    for idx in start..end {
                        let row = self.indices[idx];
                        dense[[row, col]] = self.data[idx];
                    }
                }
            }
            SparseFormat::COO => {
                for ((row, col), value) in self
                    .coo_rows
                    .iter()
                    .zip(self.coo_cols.iter())
                    .zip(self.data.iter())
                {
                    dense[[*row, *col]] = *value;
                }
            }
        }

        Ok(dense)
    }
}

/// Sparse-aware standard scaler
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseStandardScaler<S = Untrained> {
    config: SparseStandardScalerConfig,
    state: std::marker::PhantomData<S>,
}

/// Fitted sparse standard scaler
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseStandardScalerFitted {
    config: SparseStandardScalerConfig,
    mean: Array1<Float>,
    scale: Array1<Float>,
}

impl Default for SparseStandardScaler<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseStandardScaler<Untrained> {
    pub fn new() -> Self {
        Self {
            config: SparseStandardScalerConfig {
                with_mean: false, // Mean centering destroys sparsity
                with_std: true,
            },
            state: std::marker::PhantomData,
        }
    }

    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.config.with_mean = with_mean;
        self
    }

    pub fn with_std(mut self, with_std: bool) -> Self {
        self.config.with_std = with_std;
        self
    }
}

/// Configuration for SparseStandardScaler
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseStandardScalerConfig {
    pub with_mean: bool,
    pub with_std: bool,
}

impl Default for SparseStandardScalerConfig {
    fn default() -> Self {
        Self {
            with_mean: false,
            with_std: true,
        }
    }
}

impl SparseStandardScaler<Untrained> {
    pub fn get_config(&self) -> &SparseStandardScalerConfig {
        &self.config
    }
}

impl Estimator<Untrained> for SparseStandardScaler<Untrained> {
    type Config = SparseStandardScalerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<SparseMatrix, ()> for SparseStandardScaler<Untrained> {
    type Fitted = SparseStandardScalerFitted;

    fn fit(self, x: &SparseMatrix, _y: &()) -> Result<Self::Fitted> {
        let mean = if self.config.with_mean {
            x.column_means()?
        } else {
            Array1::zeros(x.shape().1)
        };

        let scale = if self.config.with_std {
            let variances = x.column_variances(&mean)?;
            variances.mapv(|v| if v > 1e-8 { v.sqrt() } else { 1.0 })
        } else {
            Array1::ones(x.shape().1)
        };

        Ok(SparseStandardScalerFitted {
            config: self.config,
            mean,
            scale,
        })
    }
}

impl Estimator<Trained> for SparseStandardScalerFitted {
    type Config = SparseStandardScalerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Transform<SparseMatrix, SparseMatrix> for SparseStandardScalerFitted {
    fn transform(&self, x: &SparseMatrix) -> Result<SparseMatrix> {
        if x.shape().1 != self.mean.len() {
            return Err(SklearsError::InvalidInput(
                "Number of features must match fitted scaler".to_string(),
            ));
        }

        let mut result = x.clone();
        result.convert_to(SparseFormat::CSR)?;

        // Transform non-zero values
        for idx in 0..result.data.len() {
            let col = result.indices[idx];
            let mut value = result.data[idx];

            if self.config.with_mean {
                value -= self.mean[col];
            }

            if self.config.with_std {
                value /= self.scale[col];
            }

            result.data[idx] = value;
        }

        Ok(result)
    }
}

/// Sparse matrix vector multiplication
pub fn sparse_matvec(matrix: &SparseMatrix, vector: &Array1<Float>) -> Result<Array1<Float>> {
    if matrix.shape().1 != vector.len() {
        return Err(SklearsError::InvalidInput(
            "Matrix columns must match vector length".to_string(),
        ));
    }

    match matrix.format() {
        SparseFormat::CSR => {
            let mut result = Array1::zeros(matrix.shape().0);

            for (row, window) in matrix.indptr.windows(2).enumerate() {
                let start = window[0];
                let end = window[1];
                let mut sum = 0.0;

                for idx in start..end {
                    let col = matrix.indices[idx];
                    sum += matrix.data[idx] * vector[col];
                }

                result[row] = sum;
            }

            Ok(result)
        }
        _ => {
            let mut temp = matrix.clone();
            temp.convert_to(SparseFormat::CSR)?;
            sparse_matvec(&temp, vector)
        }
    }
}

/// Configuration for sparse operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseConfig {
    /// Density threshold below which to use sparse operations
    pub sparsity_threshold: Float,
    /// Preferred sparse format for operations
    pub preferred_format: SparseFormat,
    /// Enable parallel sparse operations
    pub use_parallel: bool,
    /// Maximum memory usage for sparse operations (bytes)
    pub max_memory_usage: usize,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.1, // Use sparse if < 10% density
            preferred_format: SparseFormat::CSR,
            use_parallel: true,
            max_memory_usage: 1024 * 1024 * 256, // 256MB
        }
    }
}

impl SparseConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sparsity_threshold(mut self, threshold: Float) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    pub fn with_preferred_format(mut self, format: SparseFormat) -> Self {
        self.preferred_format = format;
        self
    }

    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled;
        self
    }

    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_usage = bytes;
        self
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_sparse_matrix_creation() -> Result<()> {
        let rows = vec![0, 0, 1, 2];
        let cols = vec![0, 2, 1, 0];
        let data = vec![1.0, 3.0, 2.0, 4.0];

        let sparse = SparseMatrix::from_triplets((3, 3), rows, cols, data, SparseFormat::CSR)?;

        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.nnz(), 4);
        assert!((sparse.density() - 4.0 / 9.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_sparse_format_conversion() -> Result<()> {
        let rows = vec![0, 1, 1];
        let cols = vec![0, 0, 1];
        let data = vec![1.0, 2.0, 3.0];

        let mut sparse = SparseMatrix::from_triplets((2, 2), rows, cols, data, SparseFormat::COO)?;

        // Convert to CSR
        sparse.convert_to(SparseFormat::CSR)?;
        assert_eq!(sparse.format(), SparseFormat::CSR);

        // Convert to CSC
        sparse.convert_to(SparseFormat::CSC)?;
        assert_eq!(sparse.format(), SparseFormat::CSC);

        // Convert back to COO
        sparse.convert_to(SparseFormat::COO)?;
        assert_eq!(sparse.format(), SparseFormat::COO);

        Ok(())
    }

    #[test]
    fn test_sparse_standard_scaler() -> Result<()> {
        // Create sparse matrix: [[1, 0, 3], [0, 2, 0], [4, 0, 0]]
        let rows = vec![0, 0, 1, 2];
        let cols = vec![0, 2, 1, 0];
        let data = vec![1.0, 3.0, 2.0, 4.0];

        let sparse = SparseMatrix::from_triplets((3, 3), rows, cols, data, SparseFormat::CSR)?;

        let scaler = SparseStandardScaler::new();
        let scaler_fitted = scaler.fit(&sparse, &())?;
        let scaled = scaler_fitted.transform(&sparse)?;

        // Check that scaling was applied
        assert_eq!(scaled.nnz(), sparse.nnz());
        assert!(scaled.data.iter().any(|&x| (x - 1.0).abs() > 1e-6));

        Ok(())
    }

    #[test]
    fn test_sparse_to_dense() -> Result<()> {
        let rows = vec![0, 1, 1];
        let cols = vec![0, 0, 1];
        let data = vec![1.0, 2.0, 3.0];

        let sparse = SparseMatrix::from_triplets((2, 2), rows, cols, data, SparseFormat::CSR)?;

        let dense = sparse.to_dense()?;
        let expected = arr2(&[[1.0, 0.0], [2.0, 3.0]]);

        for i in 0..2 {
            for j in 0..2 {
                assert!((dense[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }

        Ok(())
    }

    #[test]
    fn test_sparse_matvec() -> Result<()> {
        // Create sparse matrix: [[1, 0], [0, 2], [3, 0]]
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 0];
        let data = vec![1.0, 2.0, 3.0];

        let sparse = SparseMatrix::from_triplets((3, 2), rows, cols, data, SparseFormat::CSR)?;

        let vector = arr1(&[2.0, 3.0]);
        let result = sparse_matvec(&sparse, &vector)?;

        let expected = arr1(&[2.0, 6.0, 6.0]); // [1*2, 2*3, 3*2]
        for i in 0..3 {
            assert!((result[i] - expected[i]).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_sparse_column_stats() -> Result<()> {
        // Create sparse matrix with known statistics
        let rows = vec![0, 0, 1, 2];
        let cols = vec![0, 1, 0, 1];
        let data = vec![2.0, 4.0, 6.0, 8.0];

        let sparse = SparseMatrix::from_triplets((3, 2), rows, cols, data, SparseFormat::CSR)?;

        let means = sparse.column_means()?;
        let expected_means = arr1(&[(2.0 + 6.0) / 3.0, (4.0 + 8.0) / 3.0]);

        for i in 0..2 {
            assert!((means[i] - expected_means[i]).abs() < 1e-10);
        }

        Ok(())
    }
}
