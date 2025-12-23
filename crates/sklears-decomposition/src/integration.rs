//! Enhanced Integration and Interoperability Module
//!
//! This module provides advanced integration features for sklears-decomposition,
//! including support for various data formats, frameworks, and interoperability layers.
//!
//! Features:
//! - DataFrame support (conceptual interface for polars/pandas-like structures)
//! - Sparse matrix handling
//! - Arrow format compatibility
//! - Batch processing utilities
//! - Data conversion helpers

use crate::s;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// DataFrame-like interface for decomposition input
pub trait DataFrameInterface {
    /// Get number of rows
    fn nrows(&self) -> usize;

    /// Get number of columns
    fn ncols(&self) -> usize;

    /// Get column names
    fn column_names(&self) -> Vec<String>;

    /// Convert to dense Array2
    fn to_array2(&self) -> Result<Array2<Float>>;

    /// Get column by name
    fn column(&self, name: &str) -> Result<Array1<Float>>;

    /// Select columns by names
    fn select(&self, columns: &[String]) -> Result<Box<dyn DataFrameInterface>>;

    /// Get shape
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

/// Simple DataFrame implementation
#[derive(Debug, Clone)]
pub struct SimpleDataFrame {
    data: Array2<Float>,
    column_names: Vec<String>,
}

impl SimpleDataFrame {
    /// Create new DataFrame from Array2 and column names
    pub fn new(data: Array2<Float>, column_names: Vec<String>) -> Result<Self> {
        if data.ncols() != column_names.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Column count mismatch: {} columns but {} names",
                data.ncols(),
                column_names.len()
            )));
        }

        Ok(Self { data, column_names })
    }

    /// Create from Array2 with auto-generated column names
    pub fn from_array(data: Array2<Float>) -> Self {
        let ncols = data.ncols();
        let column_names = (0..ncols).map(|i| format!("col_{}", i)).collect();

        Self { data, column_names }
    }

    /// Get underlying array
    pub fn as_array(&self) -> &Array2<Float> {
        &self.data
    }

    /// Consume and return underlying array
    pub fn into_array(self) -> Array2<Float> {
        self.data
    }
}

impl DataFrameInterface for SimpleDataFrame {
    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn ncols(&self) -> usize {
        self.data.ncols()
    }

    fn column_names(&self) -> Vec<String> {
        self.column_names.clone()
    }

    fn to_array2(&self) -> Result<Array2<Float>> {
        Ok(self.data.clone())
    }

    fn column(&self, name: &str) -> Result<Array1<Float>> {
        let idx = self
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Column '{}' not found", name)))?;

        Ok(self.data.column(idx).to_owned())
    }

    fn select(&self, columns: &[String]) -> Result<Box<dyn DataFrameInterface>> {
        let indices: Vec<usize> = columns
            .iter()
            .map(|name| {
                self.column_names
                    .iter()
                    .position(|n| n == name)
                    .ok_or_else(|| {
                        SklearsError::InvalidInput(format!("Column '{}' not found", name))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let selected_data = self.data.select(scirs2_core::ndarray::Axis(1), &indices);
        let selected_names = columns.to_vec();

        Ok(Box::new(SimpleDataFrame::new(
            selected_data,
            selected_names,
        )?))
    }
}

/// Sparse matrix representation (COO format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrix {
    /// Row indices
    pub row_indices: Vec<usize>,

    /// Column indices
    pub col_indices: Vec<usize>,

    /// Non-zero values
    pub values: Vec<Float>,

    /// Matrix shape (rows, cols)
    pub shape: (usize, usize),

    /// Number of non-zero elements
    pub nnz: usize,
}

impl SparseMatrix {
    /// Create new sparse matrix in COO format
    pub fn new(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<Float>,
        shape: (usize, usize),
    ) -> Result<Self> {
        if row_indices.len() != col_indices.len() || row_indices.len() != values.len() {
            return Err(SklearsError::InvalidInput(
                "Sparse matrix indices and values must have same length".to_string(),
            ));
        }

        let nnz = values.len();

        // Validate indices
        for &row_idx in &row_indices {
            if row_idx >= shape.0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Row index {} out of bounds for shape {:?}",
                    row_idx, shape
                )));
            }
        }

        for &col_idx in &col_indices {
            if col_idx >= shape.1 {
                return Err(SklearsError::InvalidInput(format!(
                    "Column index {} out of bounds for shape {:?}",
                    col_idx, shape
                )));
            }
        }

        Ok(Self {
            row_indices,
            col_indices,
            values,
            shape,
            nnz,
        })
    }

    /// Convert to dense Array2
    pub fn to_dense(&self) -> Array2<Float> {
        let mut dense = Array2::zeros(self.shape);

        for i in 0..self.nnz {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            let val = self.values[i];
            dense[[row, col]] = val;
        }

        dense
    }

    /// Create sparse matrix from dense array
    pub fn from_dense(array: &Array2<Float>, threshold: Float) -> Self {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let shape = array.dim();

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let val = array[[i, j]];
                if val.abs() > threshold {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(val);
                }
            }
        }

        let nnz = values.len();

        Self {
            row_indices,
            col_indices,
            values,
            shape,
            nnz,
        }
    }

    /// Get sparsity ratio (fraction of zero elements)
    pub fn sparsity(&self) -> Float {
        let total_elements = self.shape.0 * self.shape.1;
        1.0 - (self.nnz as Float / total_elements as Float)
    }

    /// Transpose sparse matrix
    pub fn transpose(&self) -> Self {
        Self {
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            values: self.values.clone(),
            shape: (self.shape.1, self.shape.0),
            nnz: self.nnz,
        }
    }
}

/// Batch processor for large datasets
pub struct BatchProcessor {
    batch_size: usize,
    overlap: usize,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(batch_size: usize, overlap: usize) -> Self {
        Self {
            batch_size,
            overlap,
        }
    }

    /// Split data into batches
    pub fn split<'a>(&self, data: &'a Array2<Float>) -> Vec<Array2<Float>> {
        let n_samples = data.nrows();
        let mut batches = Vec::new();

        let stride = self.batch_size - self.overlap;
        let mut start = 0;

        while start < n_samples {
            let end = (start + self.batch_size).min(n_samples);
            let batch = data.slice(s![start..end, ..]).to_owned();
            batches.push(batch);

            if end >= n_samples {
                break;
            }

            start += stride;
        }

        batches
    }

    /// Process data in batches with a callback function
    pub fn process_batches<F, R>(&self, data: &Array2<Float>, mut processor: F) -> Vec<R>
    where
        F: FnMut(&Array2<Float>, usize) -> R,
    {
        let batches = self.split(data);
        batches
            .iter()
            .enumerate()
            .map(|(idx, batch)| processor(batch, idx))
            .collect()
    }
}

/// Data conversion utilities
pub struct DataConverter;

impl DataConverter {
    /// Normalize array to zero mean and unit variance
    pub fn standardize(data: &Array2<Float>) -> (Array2<Float>, Array1<Float>, Array1<Float>) {
        let mean = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let centered = data - &mean;

        let variance = centered
            .mapv(|x| x.powi(2))
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .unwrap();

        let std = variance.mapv(|v| v.sqrt().max(1e-8));
        let standardized = &centered / &std;

        (standardized, mean, std)
    }

    /// Min-max normalization to [0, 1] range
    pub fn min_max_scale(data: &Array2<Float>) -> (Array2<Float>, Array1<Float>, Array1<Float>) {
        let min_vals = data.fold_axis(scirs2_core::ndarray::Axis(0), Float::INFINITY, |&a, &b| {
            a.min(b)
        });

        let max_vals = data.fold_axis(
            scirs2_core::ndarray::Axis(0),
            Float::NEG_INFINITY,
            |&a, &b| a.max(b),
        );

        let range = &max_vals - &min_vals;
        let safe_range = range.mapv(|r| r.max(1e-8));

        let scaled = (data - &min_vals) / &safe_range;

        (scaled, min_vals, max_vals)
    }

    /// Robust scaling using median and IQR
    pub fn robust_scale(data: &Array2<Float>) -> (Array2<Float>, Array1<Float>, Array1<Float>) {
        let n_features = data.ncols();
        let mut medians = Array1::zeros(n_features);
        let mut iqrs = Array1::zeros(n_features);

        for (i, col) in data.axis_iter(scirs2_core::ndarray::Axis(1)).enumerate() {
            let mut sorted: Vec<Float> = col.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = sorted.len();
            let median = sorted[n / 2];
            let q1 = sorted[n / 4];
            let q3 = sorted[3 * n / 4];
            let iqr = (q3 - q1).max(1e-8);

            medians[i] = median;
            iqrs[i] = iqr;
        }

        let scaled = (data - &medians) / &iqrs;

        (scaled, medians, iqrs)
    }

    /// Convert to non-negative by clipping
    pub fn to_nonnegative(data: &Array2<Float>) -> Array2<Float> {
        data.mapv(|x| x.max(0.0))
    }

    /// Log transform with offset to handle zeros
    pub fn log_transform(data: &Array2<Float>, offset: Float) -> Array2<Float> {
        data.mapv(|x| (x + offset).ln())
    }
}

/// Memory-mapped array wrapper (conceptual - for documentation)
#[derive(Debug)]
pub struct MemoryMappedArray {
    shape: (usize, usize),
    dtype: String,
    path: String,
}

impl MemoryMappedArray {
    /// Create reference to memory-mapped array
    pub fn new(path: String, shape: (usize, usize), dtype: String) -> Self {
        Self { shape, dtype, path }
    }

    /// Get shape
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Note: Actual implementation would use memmap2 crate
    /// This is a conceptual placeholder
    pub fn to_array(&self) -> Result<Array2<Float>> {
        Err(SklearsError::InvalidInput(
            "Memory-mapped array loading not yet implemented - use memmap2 crate".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_simple_dataframe() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let columns = vec!["A".to_string(), "B".to_string()];

        let df = SimpleDataFrame::new(data, columns).unwrap();

        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.column_names(), vec!["A", "B"]);
    }

    #[test]
    fn test_dataframe_from_array() {
        let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let df = SimpleDataFrame::from_array(data);

        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 3);
    }

    #[test]
    fn test_dataframe_column() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let columns = vec!["A".to_string(), "B".to_string()];
        let df = SimpleDataFrame::new(data, columns).unwrap();

        let col_a = df.column("A").unwrap();
        assert_eq!(col_a.len(), 3);
        assert_eq!(col_a[0], 1.0);
    }

    #[test]
    fn test_sparse_matrix_creation() {
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 2.0, 3.0];
        let shape = (3, 3);

        let sparse = SparseMatrix::new(row_indices, col_indices, values, shape).unwrap();

        assert_eq!(sparse.nnz, 3);
        assert_eq!(sparse.shape, (3, 3));
    }

    #[test]
    fn test_sparse_to_dense() {
        let row_indices = vec![0, 1, 1];
        let col_indices = vec![0, 0, 1];
        let values = vec![1.0, 2.0, 3.0];
        let shape = (2, 2);

        let sparse = SparseMatrix::new(row_indices, col_indices, values, shape).unwrap();
        let dense = sparse.to_dense();

        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[1, 0]], 2.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[0, 1]], 0.0);
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let sparse = SparseMatrix::from_dense(&dense, 0.5);

        assert_eq!(sparse.nnz, 2);
        assert!(sparse.sparsity() > 0.0);
    }

    #[test]
    fn test_batch_processor() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as Float).collect()).unwrap();

        let processor = BatchProcessor::new(4, 1);
        let batches = processor.split(&data);

        assert!(batches.len() >= 3);
        assert_eq!(batches[0].nrows(), 4);
    }

    #[test]
    fn test_data_converter_standardize() {
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((20, 5), |_| rng.gen_range(0.0..10.0));

        let (standardized, mean, std) = DataConverter::standardize(&data);

        assert_eq!(standardized.dim(), data.dim());
        assert_eq!(mean.len(), 5);
        assert_eq!(std.len(), 5);

        // Check that standardized data has approximately zero mean
        let new_mean = standardized
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .unwrap();
        for &m in new_mean.iter() {
            assert!(m.abs() < 0.1); // Should be close to zero
        }
    }

    #[test]
    fn test_data_converter_min_max() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let (scaled, _min_vals, _max_vals) = DataConverter::min_max_scale(&data);

        assert_eq!(scaled.dim(), data.dim());

        // Check range is [0, 1]
        let min_scaled =
            scaled.fold_axis(scirs2_core::ndarray::Axis(0), Float::INFINITY, |&a, &b| {
                a.min(b)
            });
        let max_scaled = scaled.fold_axis(
            scirs2_core::ndarray::Axis(0),
            Float::NEG_INFINITY,
            |&a, &b| a.max(b),
        );

        for &val in min_scaled.iter() {
            assert!(val >= -0.01); // Should be >= 0
        }
        for &val in max_scaled.iter() {
            assert!(val <= 1.01); // Should be <= 1
        }
    }

    #[test]
    fn test_data_converter_to_nonnegative() {
        let data = Array2::from_shape_vec((2, 2), vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
        let nonneg = DataConverter::to_nonnegative(&data);

        for &val in nonneg.iter() {
            assert!(val >= 0.0);
        }
    }
}
