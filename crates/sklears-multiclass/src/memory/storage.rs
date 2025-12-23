//! Sparse and efficient storage formats

use scirs2_core::ndarray::{Array1, Array2};

/// Sparse matrix format (CSR - Compressed Sparse Row)
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Non-zero values
    pub values: Vec<f64>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Row pointers
    pub row_ptrs: Vec<usize>,
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
}

impl SparseMatrix {
    /// Create a new sparse matrix from dense array
    pub fn from_dense(array: &Array2<f64>, threshold: f64) -> Self {
        let (nrows, ncols) = array.dim();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..nrows {
            let mut nnz_in_row = 0;
            for j in 0..ncols {
                let val = array[[i, j]];
                if val.abs() > threshold {
                    values.push(val);
                    col_indices.push(j);
                    nnz_in_row += 1;
                }
            }
            row_ptrs.push(row_ptrs.last().unwrap() + nnz_in_row);
        }

        Self {
            values,
            col_indices,
            row_ptrs,
            shape: (nrows, ncols),
        }
    }

    /// Convert back to dense array
    pub fn to_dense(&self) -> Array2<f64> {
        let (nrows, ncols) = self.shape;
        let mut array = Array2::zeros((nrows, ncols));

        for i in 0..nrows {
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];

            for idx in start..end {
                let j = self.col_indices[idx];
                array[[i, j]] = self.values[idx];
            }
        }

        array
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio
    pub fn sparsity(&self) -> f64 {
        let total_elements = self.shape.0 * self.shape.1;
        if total_elements == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / total_elements as f64)
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.values.len() * std::mem::size_of::<f64>()
            + self.col_indices.len() * std::mem::size_of::<usize>()
            + self.row_ptrs.len() * std::mem::size_of::<usize>()
    }

    /// Get compression ratio compared to dense
    pub fn compression_ratio(&self) -> f64 {
        let dense_size = self.shape.0 * self.shape.1 * std::mem::size_of::<f64>();
        dense_size as f64 / self.memory_size() as f64
    }
}

/// Sparse vector (for 1D arrays)
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Non-zero values
    pub values: Vec<f64>,
    /// Indices of non-zero values
    pub indices: Vec<usize>,
    /// Vector length
    pub len: usize,
}

impl SparseVector {
    /// Create from dense array
    pub fn from_dense(array: &Array1<f64>, threshold: f64) -> Self {
        let mut values = Vec::new();
        let mut indices = Vec::new();

        for (i, &val) in array.iter().enumerate() {
            if val.abs() > threshold {
                values.push(val);
                indices.push(i);
            }
        }

        Self {
            values,
            indices,
            len: array.len(),
        }
    }

    /// Convert to dense array
    pub fn to_dense(&self) -> Array1<f64> {
        let mut array = Array1::zeros(self.len);

        for (val, &idx) in self.values.iter().zip(self.indices.iter()) {
            array[idx] = *val;
        }

        array
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio
    pub fn sparsity(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / self.len as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_sparse_matrix_dense_conversion() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]];

        let sparse = SparseMatrix::from_dense(&dense, 1e-10);
        let recovered = sparse.to_dense();

        assert_eq!(dense, recovered);
    }

    #[test]
    fn test_sparse_matrix_nnz() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]];

        let sparse = SparseMatrix::from_dense(&dense, 1e-10);
        assert_eq!(sparse.nnz(), 4);
    }

    #[test]
    fn test_sparse_matrix_sparsity() {
        let dense = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let sparse = SparseMatrix::from_dense(&dense, 1e-10);
        // 8 out of 9 elements are zero
        assert!((sparse.sparsity() - 8.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_vector() {
        let dense = array![1.0, 0.0, 2.0, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense, 1e-10);
        let recovered = sparse.to_dense();

        assert_eq!(dense, recovered);
        assert_eq!(sparse.nnz(), 3);
    }

    #[test]
    fn test_sparse_compression_ratio() {
        let dense = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];

        let sparse = SparseMatrix::from_dense(&dense, 1e-10);
        // Should have significant compression for sparse data
        assert!(sparse.compression_ratio() > 1.0);
    }

    #[test]
    fn test_sparse_threshold() {
        let dense = array![[1.0, 0.001, 2.0], [0.002, 0.0, 0.003]];

        // With high threshold, small values are treated as zero
        let sparse = SparseMatrix::from_dense(&dense, 0.01);
        assert_eq!(sparse.nnz(), 2); // Only 1.0 and 2.0
    }
}
