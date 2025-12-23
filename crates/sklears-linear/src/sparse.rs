//! Sparse matrix support for linear models
//!
//! This module provides efficient sparse matrix operations for large-scale
//! linear models using the sprs crate. It includes:
//! - Sparse matrix traits and wrappers
//! - Conversion utilities between dense and sparse formats
//! - Sparse implementations of key algorithms (coordinate descent, etc.)
//! - Memory-efficient operations for high-dimensional sparse data

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "sparse")]
use scirs2_core::numeric::SparseElement;
#[cfg(feature = "sparse")]
use scirs2_sparse::CsrMatrix;
use sklears_core::{
    error::{Result, SklearsError},
    types::FloatBounds,
};

/// Either type for sparse matrix operations
#[derive(Debug, Clone)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

/// Configuration for sparse matrix operations
#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub sparsity_threshold: f64,
    pub min_sparsity_ratio: f64,
    pub max_dense_memory_ratio: f64,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 1e-8,
            min_sparsity_ratio: 0.1,
            max_dense_memory_ratio: 0.5,
        }
    }
}

/// Analysis of matrix sparsity patterns
#[derive(Debug, Clone)]
pub struct SparsityAnalysis {
    pub sparsity_ratio: f64,
    pub memory_savings: f64,
    pub recommended_format: String,
}

/// Sparse matrix operations trait
#[cfg(feature = "sparse")]
pub trait SparseMatrix<T: FloatBounds> {
    /// Number of rows
    fn nrows(&self) -> usize;

    /// Number of columns
    fn ncols(&self) -> usize;

    /// Number of non-zero elements
    fn nnz(&self) -> usize;

    /// Sparsity ratio (nnz / (nrows * ncols))
    fn sparsity(&self) -> f64 {
        let total_elements = self.nrows() as f64 * self.ncols() as f64;
        if total_elements > 0.0 {
            self.nnz() as f64 / total_elements
        } else {
            0.0
        }
    }

    /// Matrix-vector multiplication: self * x
    fn matvec(&self, x: &Array1<T>) -> Result<Array1<T>>;

    /// Transposed matrix-vector multiplication: self^T * x
    fn transp_matvec(&self, x: &Array1<T>) -> Result<Array1<T>>;

    /// Get a specific row as a sparse vector
    fn row(&self, i: usize) -> Result<CsrMatrix<T>>;

    /// Get a specific column as a sparse vector
    fn col(&self, j: usize) -> Result<CsrMatrix<T>>;

    /// Convert to dense matrix (use with caution for large matrices)
    fn to_dense(&self) -> Result<Array2<T>>;
}

/// Wrapper for CSR sparse matrices
#[cfg(feature = "sparse")]
#[derive(Clone)]
pub struct SparseMatrixCSR<T: FloatBounds> {
    inner: CsrMatrix<T>,
}

#[cfg(feature = "sparse")]
impl<T: FloatBounds + SparseElement> SparseMatrixCSR<T> {
    /// Create a new sparse matrix from CSR format
    pub fn new(inner: CsrMatrix<T>) -> Self {
        Self { inner }
    }

    /// Create from triplet format (row indices, col indices, values)
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize, T)],
    ) -> Result<Self> {
        let csmat = CsrMatrix::try_from_triplets(nrows, ncols, triplets)
            .map_err(|e| SklearsError::Other(format!("Failed to create sparse matrix: {:?}", e)))?;
        Ok(Self::new(csmat))
    }

    /// Get inner CSR matrix
    pub fn inner(&self) -> &CsrMatrix<T> {
        &self.inner
    }

    /// Create from dense matrix with sparsity threshold
    pub fn from_dense(dense: &Array2<T>, threshold: T) -> Self {
        let (nrows, ncols) = dense.dim();
        let mut triplets = Vec::new();

        // Iterate through dense matrix and collect non-zero elements
        for i in 0..nrows {
            for j in 0..ncols {
                let val = dense[[i, j]];
                // Check if value is above threshold (non-zero)
                if val.abs() > threshold {
                    triplets.push((i, j, val));
                }
            }
        }

        // Use try_from_triplets to construct CSR matrix
        let csmat = CsrMatrix::try_from_triplets(nrows, ncols, &triplets).unwrap_or_else(|_| {
            // Create empty matrix as fallback
            CsrMatrix::try_from_triplets(nrows, ncols, &[]).unwrap()
        });

        Self::new(csmat)
    }
}

#[cfg(feature = "sparse")]
impl<T: FloatBounds + SparseElement> SparseMatrix<T> for SparseMatrixCSR<T> {
    fn nrows(&self) -> usize {
        self.inner.rows()
    }

    fn ncols(&self) -> usize {
        self.inner.cols()
    }

    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    fn matvec(&self, x: &Array1<T>) -> Result<Array1<T>> {
        if x.len() != self.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Vector length {} does not match matrix columns {}",
                x.len(),
                self.ncols()
            )));
        }

        let mut result = Array1::zeros(self.nrows());

        // Standard CSR matrix-vector multiplication: y = A * x
        for row_idx in 0..self.nrows() {
            let row_start = self.inner.indptr[row_idx];
            let row_end = self.inner.indptr[row_idx + 1];

            let mut sum = T::default();
            for j in row_start..row_end {
                let col_idx = self.inner.indices[j];
                let val = self.inner.data[j];
                sum += val * x[col_idx];
            }
            result[row_idx] = sum;
        }

        Ok(result)
    }

    fn transp_matvec(&self, x: &Array1<T>) -> Result<Array1<T>> {
        if x.len() != self.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Vector length {} does not match matrix rows {}",
                x.len(),
                self.nrows()
            )));
        }

        let mut result = Array1::zeros(self.ncols());

        // Transposed CSR matrix-vector multiplication: y = A^T * x
        // For each row in A, scatter x[row] * A[row,:] into result
        for row_idx in 0..self.nrows() {
            let row_start = self.inner.indptr[row_idx];
            let row_end = self.inner.indptr[row_idx + 1];

            let x_val = x[row_idx];
            for j in row_start..row_end {
                let col_idx = self.inner.indices[j];
                let val = self.inner.data[j];
                result[col_idx] += val * x_val;
            }
        }

        Ok(result)
    }

    fn row(&self, i: usize) -> Result<CsrMatrix<T>> {
        if i >= self.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Row index {} out of bounds for matrix with {} rows",
                i,
                self.nrows()
            )));
        }

        // For now, return a simple error - implementing row extraction requires more complex logic
        Err(SklearsError::NotImplemented(
            "Row extraction not yet implemented".to_string(),
        ))
    }

    fn col(&self, j: usize) -> Result<CsrMatrix<T>> {
        if j >= self.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Column index {} out of bounds for matrix with {} columns",
                j,
                self.ncols()
            )));
        }

        // For now, return a simple error - implementing column extraction requires more complex logic
        Err(SklearsError::NotImplemented(
            "Column extraction not yet implemented".to_string(),
        ))
    }

    fn to_dense(&self) -> Result<Array2<T>> {
        let mut dense = Array2::zeros((self.nrows(), self.ncols()));

        // Convert CSR sparse matrix to dense format
        for row_idx in 0..self.nrows() {
            let row_start = self.inner.indptr[row_idx];
            let row_end = self.inner.indptr[row_idx + 1];

            for j in row_start..row_end {
                let col_idx = self.inner.indices[j];
                let val = self.inner.data[j];
                dense[[row_idx, col_idx]] = val;
            }
        }

        Ok(dense)
    }
}

/// Coordinate descent solver for sparse matrices
#[cfg(feature = "sparse")]
pub struct SparseCoordinateDescentSolver<T> {
    pub alpha: T,
    pub l1_ratio: T,
    pub max_iter: usize,
    pub tol: T,
    pub cyclic: bool,
    pub sparse_config: SparseConfig,
}

#[cfg(feature = "sparse")]
impl<T: FloatBounds> SparseCoordinateDescentSolver<T> {
    pub fn new(alpha: T, l1_ratio: T, max_iter: usize, tol: T) -> Self {
        Self {
            alpha,
            l1_ratio,
            max_iter,
            tol,
            cyclic: true,
            sparse_config: SparseConfig::default(),
        }
    }

    pub fn fit_lasso(&self, _x: &SparseMatrixCSR<T>, _y: &Array1<T>) -> Result<Array1<T>> {
        // TODO: Implement sparse LASSO coordinate descent
        Err(SklearsError::NotImplemented(
            "Sparse LASSO not yet implemented".to_string(),
        ))
    }

    pub fn fit_ridge(&self, _x: &SparseMatrixCSR<T>, _y: &Array1<T>) -> Result<Array1<T>> {
        // TODO: Implement sparse Ridge regression
        Err(SklearsError::NotImplemented(
            "Sparse Ridge regression not yet implemented".to_string(),
        ))
    }

    pub fn fit_elastic_net(&self, _x: &SparseMatrixCSR<T>, _y: &Array1<T>) -> Result<Array1<T>> {
        // TODO: Implement sparse Elastic Net
        Err(SklearsError::NotImplemented(
            "Sparse Elastic Net not yet implemented".to_string(),
        ))
    }

    /// Solve sparse LASSO regression
    pub fn solve_sparse_lasso(
        &self,
        _x: &SparseMatrixCSR<T>,
        _y: &Array1<T>,
        _alpha: T,
        _fit_intercept: bool,
    ) -> Result<(Array1<T>, T)> {
        // TODO: Implement sparse LASSO solving
        Err(SklearsError::NotImplemented(
            "solve_sparse_lasso not yet implemented".to_string(),
        ))
    }

    /// Solve sparse Elastic Net regression
    pub fn solve_sparse_elastic_net(
        &self,
        _x: &SparseMatrixCSR<T>,
        _y: &Array1<T>,
        _alpha: T,
        _l1_ratio: T,
        _fit_intercept: bool,
    ) -> Result<(Array1<T>, T)> {
        // TODO: Implement sparse Elastic Net solving
        Err(SklearsError::NotImplemented(
            "solve_sparse_elastic_net not yet implemented".to_string(),
        ))
    }
}

/// Convenience functions for sparse matrix operations
#[cfg(feature = "sparse")]
pub mod utils {
    use super::*;

    /// Check if a dense matrix should be converted to sparse based on sparsity ratio
    pub fn should_use_sparse<T: FloatBounds>(dense: &Array2<T>, config: &SparseConfig) -> bool {
        let total_elements = dense.len() as f64;
        let non_zero = dense
            .iter()
            .filter(|&&x| x.abs() > T::from(config.sparsity_threshold).unwrap())
            .count() as f64;
        let sparsity = non_zero / total_elements;
        sparsity < config.min_sparsity_ratio
    }

    /// Convert dense to sparse if beneficial
    pub fn auto_sparse<T: FloatBounds + SparseElement>(
        dense: &Array2<T>,
        threshold: T,
    ) -> Result<SparseMatrixCSR<T>> {
        let config = SparseConfig::default();
        if should_use_sparse(dense, &config) {
            Ok(SparseMatrixCSR::from_dense(dense, threshold))
        } else {
            Err(SklearsError::InvalidInput(
                "Matrix is not sparse enough to benefit from sparse format".to_string(),
            ))
        }
    }
}
