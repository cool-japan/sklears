//! Sparse matrix support for tree algorithms
//!
//! This module provides efficient sparse matrix representations and operations
//! for handling high-dimensional data with many zero values.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Compressed Sparse Row (CSR) matrix representation
///
/// Efficiently stores sparse matrices by storing only non-zero values
/// along with their row and column indices.
#[derive(Debug, Clone)]
pub struct CSRMatrix {
    /// Non-zero values in row-major order
    pub data: Vec<Float>,
    /// Column indices for each value in data
    pub indices: Vec<usize>,
    /// Pointers to the start of each row in data/indices arrays
    pub indptr: Vec<usize>,
    /// Number of rows in the matrix
    pub nrows: usize,
    /// Number of columns in the matrix
    pub ncols: usize,
}

impl CSRMatrix {
    /// Create a new empty CSR matrix with given dimensions
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            data: Vec::new(),
            indices: Vec::new(),
            indptr: vec![0; nrows + 1],
            nrows,
            ncols,
        }
    }

    /// Create CSR matrix from dense ndarray
    pub fn from_dense(dense: &Array2<Float>) -> Self {
        let (nrows, ncols) = dense.dim();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0; nrows + 1];

        for (row_idx, row) in dense.outer_iter().enumerate() {
            indptr[row_idx] = data.len();

            for (col_idx, &value) in row.iter().enumerate() {
                if value.abs() > Float::EPSILON {
                    data.push(value);
                    indices.push(col_idx);
                }
            }
        }
        indptr[nrows] = data.len();

        Self {
            data,
            indices,
            indptr,
            nrows,
            ncols,
        }
    }

    /// Convert CSR matrix back to dense ndarray
    pub fn to_dense(&self) -> Array2<Float> {
        let mut dense = Array2::<Float>::zeros((self.nrows, self.ncols));

        for row_idx in 0..self.nrows {
            let start = self.indptr[row_idx];
            let end = self.indptr[row_idx + 1];

            for data_idx in start..end {
                let col_idx = self.indices[data_idx];
                let value = self.data[data_idx];
                dense[[row_idx, col_idx]] = value;
            }
        }

        dense
    }

    /// Get a specific element from the matrix
    pub fn get(&self, row: usize, col: usize) -> Float {
        if row >= self.nrows || col >= self.ncols {
            return 0.0;
        }

        let start = self.indptr[row];
        let end = self.indptr[row + 1];

        for data_idx in start..end {
            if self.indices[data_idx] == col {
                return self.data[data_idx];
            }
            if self.indices[data_idx] > col {
                break;
            }
        }

        0.0
    }

    /// Set a specific element in the matrix
    pub fn set(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        if row >= self.nrows || col >= self.ncols {
            return Err(SklearsError::InvalidInput(
                "Row or column index out of bounds".to_string(),
            ));
        }

        let start = self.indptr[row];
        let end = self.indptr[row + 1];

        // Find insertion point
        let mut insert_pos = end;
        for data_idx in start..end {
            if self.indices[data_idx] == col {
                // Update existing value
                if value.abs() > Float::EPSILON {
                    self.data[data_idx] = value;
                } else {
                    // Remove zero value
                    self.data.remove(data_idx);
                    self.indices.remove(data_idx);
                    for i in (row + 1)..=self.nrows {
                        self.indptr[i] -= 1;
                    }
                }
                return Ok(());
            }
            if self.indices[data_idx] > col {
                insert_pos = data_idx;
                break;
            }
        }

        // Insert new non-zero value
        if value.abs() > Float::EPSILON {
            self.data.insert(insert_pos, value);
            self.indices.insert(insert_pos, col);
            for i in (row + 1)..=self.nrows {
                self.indptr[i] += 1;
            }
        }

        Ok(())
    }

    /// Get a row as a sparse vector
    pub fn get_row(&self, row: usize) -> SparseVector {
        if row >= self.nrows {
            return SparseVector::new(self.ncols);
        }

        let start = self.indptr[row];
        let end = self.indptr[row + 1];

        let indices = self.indices[start..end].to_vec();
        let data = self.data[start..end].to_vec();

        SparseVector {
            indices,
            data,
            length: self.ncols,
        }
    }

    /// Calculate sparsity ratio (proportion of zero elements)
    pub fn sparsity(&self) -> Float {
        let total_elements = self.nrows * self.ncols;
        let non_zero_elements = self.data.len();
        1.0 - (non_zero_elements as Float / total_elements as Float)
    }

    /// Matrix-vector multiplication with sparse vector
    pub fn matvec_sparse(&self, vector: &SparseVector) -> Result<Array1<Float>> {
        if vector.length != self.ncols {
            return Err(SklearsError::InvalidInput(
                "Vector length must match matrix columns".to_string(),
            ));
        }

        let mut result = Array1::<Float>::zeros(self.nrows);

        // Create a map for fast vector lookups
        let mut vector_map: HashMap<usize, Float> = HashMap::new();
        for (i, &idx) in vector.indices.iter().enumerate() {
            vector_map.insert(idx, vector.data[i]);
        }

        for row_idx in 0..self.nrows {
            let start = self.indptr[row_idx];
            let end = self.indptr[row_idx + 1];

            let mut dot_product = 0.0;
            for data_idx in start..end {
                let col_idx = self.indices[data_idx];
                let matrix_value = self.data[data_idx];

                if let Some(&vector_value) = vector_map.get(&col_idx) {
                    dot_product += matrix_value * vector_value;
                }
            }
            result[row_idx] = dot_product;
        }

        Ok(result)
    }

    /// Matrix-vector multiplication with dense vector
    pub fn matvec_dense(&self, vector: &Array1<Float>) -> Result<Array1<Float>> {
        if vector.len() != self.ncols {
            return Err(SklearsError::InvalidInput(
                "Vector length must match matrix columns".to_string(),
            ));
        }

        let mut result = Array1::<Float>::zeros(self.nrows);

        for row_idx in 0..self.nrows {
            let start = self.indptr[row_idx];
            let end = self.indptr[row_idx + 1];

            let mut dot_product = 0.0;
            for data_idx in start..end {
                let col_idx = self.indices[data_idx];
                let matrix_value = self.data[data_idx];
                dot_product += matrix_value * vector[col_idx];
            }
            result[row_idx] = dot_product;
        }

        Ok(result)
    }

    /// Apply a function to all non-zero elements
    pub fn map_nonzero<F>(&mut self, f: F)
    where
        F: Fn(Float) -> Float,
    {
        for value in &mut self.data {
            *value = f(*value);
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<Float>()
            + self.indices.len() * std::mem::size_of::<usize>()
            + self.indptr.len() * std::mem::size_of::<usize>()
    }
}

/// Sparse vector representation
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Non-zero values
    pub data: Vec<Float>,
    /// Indices of non-zero values
    pub indices: Vec<usize>,
    /// Total length of the vector
    pub length: usize,
}

impl SparseVector {
    /// Create a new empty sparse vector
    pub fn new(length: usize) -> Self {
        Self {
            data: Vec::new(),
            indices: Vec::new(),
            length,
        }
    }

    /// Create sparse vector from dense array
    pub fn from_dense(dense: &Array1<Float>) -> Self {
        let mut data = Vec::new();
        let mut indices = Vec::new();

        for (idx, &value) in dense.iter().enumerate() {
            if value.abs() > Float::EPSILON {
                data.push(value);
                indices.push(idx);
            }
        }

        Self {
            data,
            indices,
            length: dense.len(),
        }
    }

    /// Convert to dense array
    pub fn to_dense(&self) -> Array1<Float> {
        let mut dense = Array1::<Float>::zeros(self.length);

        for (i, &idx) in self.indices.iter().enumerate() {
            dense[idx] = self.data[i];
        }

        dense
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Float {
        if index >= self.length {
            return 0.0;
        }

        for (i, &idx) in self.indices.iter().enumerate() {
            if idx == index {
                return self.data[i];
            }
            if idx > index {
                break;
            }
        }

        0.0
    }

    /// Dot product with another sparse vector
    pub fn dot_sparse(&self, other: &SparseVector) -> Float {
        if self.length != other.length {
            return 0.0;
        }

        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            let idx_a = self.indices[i];
            let idx_b = other.indices[j];

            if idx_a == idx_b {
                result += self.data[i] * other.data[j];
                i += 1;
                j += 1;
            } else if idx_a < idx_b {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }

    /// Dot product with dense vector
    pub fn dot_dense(&self, dense: &Array1<Float>) -> Float {
        if dense.len() != self.length {
            return 0.0;
        }

        let mut result = 0.0;
        for (i, &idx) in self.indices.iter().enumerate() {
            result += self.data[i] * dense[idx];
        }

        result
    }

    /// L2 norm of the vector
    pub fn norm(&self) -> Float {
        self.data.iter().map(|&x| x * x).sum::<Float>().sqrt()
    }

    /// Sparsity ratio
    pub fn sparsity(&self) -> Float {
        1.0 - (self.data.len() as Float / self.length as Float)
    }
}

/// Sparse tree data structure for efficient tree operations on sparse data
#[derive(Debug, Clone)]
pub struct SparseTreeData {
    /// Sparse feature matrix (samples x features)
    pub x_sparse: CSRMatrix,
    /// Target values
    pub y: Array1<Float>,
    /// Sample weights (optional)
    pub sample_weight: Option<Array1<Float>>,
    /// Feature indices that are actually used in splits
    pub active_features: Vec<usize>,
}

impl SparseTreeData {
    /// Create new sparse tree data
    pub fn new(x_sparse: CSRMatrix, y: Array1<Float>) -> Result<Self> {
        if x_sparse.nrows != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Find active features (features with at least one non-zero value)
        let mut active_features = Vec::new();
        let mut feature_used = vec![false; x_sparse.ncols];

        for &col_idx in &x_sparse.indices {
            if !feature_used[col_idx] {
                feature_used[col_idx] = true;
                active_features.push(col_idx);
            }
        }
        active_features.sort_unstable();

        Ok(Self {
            x_sparse,
            y,
            sample_weight: None,
            active_features,
        })
    }

    /// Set sample weights
    pub fn with_sample_weight(mut self, weights: Array1<Float>) -> Result<Self> {
        if weights.len() != self.x_sparse.nrows {
            return Err(SklearsError::InvalidInput(
                "Sample weights length must match number of samples".to_string(),
            ));
        }
        self.sample_weight = Some(weights);
        Ok(self)
    }

    /// Get feature values for a specific feature across all samples
    pub fn get_feature_column(&self, feature_idx: usize) -> SparseVector {
        let mut data = Vec::new();
        let mut indices = Vec::new();

        for row_idx in 0..self.x_sparse.nrows {
            let value = self.x_sparse.get(row_idx, feature_idx);
            if value.abs() > Float::EPSILON {
                data.push(value);
                indices.push(row_idx);
            }
        }

        SparseVector {
            data,
            indices,
            length: self.x_sparse.nrows,
        }
    }

    /// Calculate feature importance based on non-zero value frequency
    pub fn calculate_sparse_feature_importance(&self) -> Array1<Float> {
        let mut importance = Array1::<Float>::zeros(self.x_sparse.ncols);

        for feature_idx in &self.active_features {
            let feature_col = self.get_feature_column(*feature_idx);
            let frequency = feature_col.data.len() as Float / self.x_sparse.nrows as Float;
            let magnitude = feature_col.data.iter().map(|&x| x.abs()).sum::<Float>();

            // Combine frequency and magnitude for importance score
            importance[*feature_idx] = frequency * magnitude;
        }

        // Normalize importance scores
        let max_importance = importance.iter().fold(0.0f64, |max, &x| max.max(x));
        if max_importance > Float::EPSILON {
            importance /= max_importance;
        }

        importance
    }

    /// Get subset of data for specific sample indices
    pub fn subset(&self, sample_indices: &[usize]) -> Result<Self> {
        let new_nrows = sample_indices.len();
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0; new_nrows + 1];

        for (new_row_idx, &old_row_idx) in sample_indices.iter().enumerate() {
            if old_row_idx >= self.x_sparse.nrows {
                return Err(SklearsError::InvalidInput(
                    "Sample index out of bounds".to_string(),
                ));
            }

            new_indptr[new_row_idx] = new_data.len();

            let start = self.x_sparse.indptr[old_row_idx];
            let end = self.x_sparse.indptr[old_row_idx + 1];

            for data_idx in start..end {
                new_data.push(self.x_sparse.data[data_idx]);
                new_indices.push(self.x_sparse.indices[data_idx]);
            }
        }
        new_indptr[new_nrows] = new_data.len();

        let new_x_sparse = CSRMatrix {
            data: new_data,
            indices: new_indices,
            indptr: new_indptr,
            nrows: new_nrows,
            ncols: self.x_sparse.ncols,
        };

        let new_y = Array1::from_iter(sample_indices.iter().map(|&i| self.y[i]));

        let new_sample_weight = self
            .sample_weight
            .as_ref()
            .map(|weights| Array1::from_iter(sample_indices.iter().map(|&i| weights[i])));

        Ok(Self {
            x_sparse: new_x_sparse,
            y: new_y,
            sample_weight: new_sample_weight,
            active_features: self.active_features.clone(),
        })
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.x_sparse.memory_usage()
            + self.y.len() * std::mem::size_of::<Float>()
            + self
                .sample_weight
                .as_ref()
                .map_or(0, |w| w.len() * std::mem::size_of::<Float>())
            + self.active_features.len() * std::mem::size_of::<usize>()
    }
}

/// Utilities for working with sparse data in tree algorithms
pub struct SparseTreeUtils;

impl SparseTreeUtils {
    /// Find best threshold for a sparse feature
    ///
    /// This function efficiently finds the best split threshold for a sparse feature
    /// by only considering non-zero values and the implicit zeros.
    pub fn find_best_sparse_threshold(
        feature_values: &SparseVector,
        targets: &Array1<Float>,
        sample_weight: Option<&Array1<Float>>,
    ) -> Result<(Float, Float)> {
        if feature_values.length != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Feature values and targets must have same length".to_string(),
            ));
        }

        if feature_values.data.is_empty() {
            return Ok((0.0, 0.0)); // No split possible
        }

        // Collect (value, target, weight) tuples for non-zero values
        let mut value_target_pairs: Vec<(Float, Float, Float)> = Vec::new();

        for (i, &idx) in feature_values.indices.iter().enumerate() {
            let weight = sample_weight.map_or(1.0, |w| w[idx]);
            value_target_pairs.push((feature_values.data[i], targets[idx], weight));
        }

        // Sort by feature value
        value_target_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Calculate statistics for zero values
        let mut zero_target_sum = 0.0;
        let mut zero_weight_sum = 0.0;
        let mut zero_count = 0;

        let mut used_indices = vec![false; targets.len()];
        for &idx in &feature_values.indices {
            used_indices[idx] = true;
        }

        for (idx, &is_used) in used_indices.iter().enumerate() {
            if !is_used {
                let weight = sample_weight.map_or(1.0, |w| w[idx]);
                zero_target_sum += targets[idx] * weight;
                zero_weight_sum += weight;
                zero_count += 1;
            }
        }

        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        // Try splits between consecutive non-zero values
        for i in 0..value_target_pairs.len() - 1 {
            let current_value = value_target_pairs[i].0;
            let next_value = value_target_pairs[i + 1].0;

            if (next_value - current_value).abs() < Float::EPSILON {
                continue; // Skip if values are essentially the same
            }

            let threshold = (current_value + next_value) / 2.0;

            // Calculate left and right statistics
            let mut left_sum = zero_target_sum; // Zeros go to left
            let mut left_weight = zero_weight_sum;
            let mut right_sum = 0.0;
            let mut right_weight = 0.0;

            for j in 0..value_target_pairs.len() {
                let (value, target, weight) = value_target_pairs[j];
                if value <= threshold {
                    left_sum += target * weight;
                    left_weight += weight;
                } else {
                    right_sum += target * weight;
                    right_weight += weight;
                }
            }

            // Calculate variance reduction (simple version)
            if left_weight > 0.0 && right_weight > 0.0 {
                let left_mean = left_sum / left_weight;
                let right_mean = right_sum / right_weight;
                let total_weight = left_weight + right_weight;
                let total_mean = (left_sum + right_sum) / total_weight;

                // Weighted variance reduction
                let gain = (left_weight / total_weight) * (left_mean - total_mean).powi(2)
                    + (right_weight / total_weight) * (right_mean - total_mean).powi(2);

                if gain > best_gain {
                    best_gain = gain;
                    best_threshold = threshold;
                }
            }
        }

        // Also try threshold at 0 (separating zeros from non-zeros)
        if !value_target_pairs.is_empty() {
            let threshold = 0.0;

            let left_sum = zero_target_sum;
            let left_weight = zero_weight_sum;
            let right_sum: Float = value_target_pairs
                .iter()
                .map(|(_, target, weight)| target * weight)
                .sum();
            let right_weight: Float = value_target_pairs
                .iter()
                .map(|(_, _, weight)| *weight)
                .sum();

            if left_weight > 0.0 && right_weight > 0.0 {
                let left_mean = left_sum / left_weight;
                let right_mean = right_sum / right_weight;
                let total_weight = left_weight + right_weight;
                let total_mean = (left_sum + right_sum) / total_weight;

                let gain = (left_weight / total_weight) * (left_mean - total_mean).powi(2)
                    + (right_weight / total_weight) * (right_mean - total_mean).powi(2);

                if gain > best_gain {
                    best_gain = gain;
                    best_threshold = threshold;
                }
            }
        }

        Ok((best_threshold, best_gain))
    }

    /// Convert dense data to sparse format if beneficial
    ///
    /// Returns sparse format only if sparsity is above the threshold
    pub fn densify_if_beneficial(
        sparse_data: &CSRMatrix,
        sparsity_threshold: Float,
    ) -> Option<Array2<Float>> {
        if sparse_data.sparsity() < sparsity_threshold {
            Some(sparse_data.to_dense())
        } else {
            None
        }
    }

    /// Estimate memory savings from using sparse format
    pub fn estimate_memory_savings(dense: &Array2<Float>) -> (usize, usize, Float) {
        let (nrows, ncols) = dense.dim();
        let dense_memory = nrows * ncols * std::mem::size_of::<Float>();

        let sparse = CSRMatrix::from_dense(dense);
        let sparse_memory = sparse.memory_usage();

        let savings_ratio = 1.0 - (sparse_memory as Float / dense_memory as Float);

        (dense_memory, sparse_memory, savings_ratio)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_csr_matrix_creation() {
        let dense = arr2(&[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]]);

        let sparse = CSRMatrix::from_dense(&dense);

        assert_eq!(sparse.data, vec![1.0, 3.0, 2.0, 4.0, 5.0]);
        assert_eq!(sparse.indices, vec![0, 2, 1, 0, 2]);
        assert_eq!(sparse.indptr, vec![0, 2, 3, 5]);
    }

    #[test]
    fn test_csr_matrix_get_set() {
        let dense = arr2(&[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]]);

        let mut sparse = CSRMatrix::from_dense(&dense);

        assert_eq!(sparse.get(0, 0), 1.0);
        assert_eq!(sparse.get(0, 1), 0.0);
        assert_eq!(sparse.get(1, 1), 2.0);

        sparse.set(0, 1, 7.0).unwrap();
        assert_eq!(sparse.get(0, 1), 7.0);
    }

    #[test]
    fn test_sparse_vector_operations() {
        let dense = Array1::from(vec![1.0, 0.0, 3.0, 0.0, 5.0]);
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.data, vec![1.0, 3.0, 5.0]);
        assert_eq!(sparse.indices, vec![0, 2, 4]);

        assert_eq!(sparse.get(0), 1.0);
        assert_eq!(sparse.get(1), 0.0);
        assert_eq!(sparse.get(2), 3.0);

        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed, dense);
    }

    #[test]
    fn test_sparse_matvec() {
        let dense = arr2(&[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]]);

        let sparse_matrix = CSRMatrix::from_dense(&dense);
        let vector = Array1::from(vec![1.0, 2.0, 3.0]);

        let result_sparse = sparse_matrix.matvec_dense(&vector).unwrap();
        let result_dense = dense.dot(&vector);

        for (a, b) in result_sparse.iter().zip(result_dense.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
