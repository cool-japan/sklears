//! Memory-efficient storage for sparse output representations
//!
//! This module provides optimized data structures and algorithms for scenarios where
//! multi-output predictions are sparse (most outputs are zero or inactive).
//! Common in multi-label classification where each instance typically has only a few active labels.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::fmt;

/// Compressed Sparse Row (CSR) format for efficient sparse matrix storage
#[derive(Debug, Clone)]
pub struct CSRMatrix<T: Clone> {
    /// Non-zero values stored in row-major order
    pub data: Vec<T>,
    /// Column indices for each non-zero value
    pub indices: Vec<usize>,
    /// Pointers to the start of each row in data/indices
    pub indptr: Vec<usize>,
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
}

impl<T: Clone + Default + PartialEq> CSRMatrix<T> {
    /// Create a new empty CSR matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: Vec::new(),
            indices: Vec::new(),
            indptr: vec![0; rows + 1],
            shape: (rows, cols),
        }
    }

    /// Create CSR matrix from dense array
    pub fn from_dense(dense: &ArrayView2<T>) -> Self
    where
        T: Clone + Default + PartialEq + Copy,
    {
        let (rows, cols) = dense.dim();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0; rows + 1];

        for row in 0..rows {
            for col in 0..cols {
                let val = dense[[row, col]];
                if val != T::default() {
                    data.push(val);
                    indices.push(col);
                }
            }
            indptr[row + 1] = data.len();
        }

        Self {
            data,
            indices,
            indptr,
            shape: (rows, cols),
        }
    }

    /// Convert back to dense array
    pub fn to_dense(&self) -> Array2<T>
    where
        T: Clone + Default,
    {
        let (rows, cols) = self.shape;
        let mut dense = Array2::from_elem((rows, cols), T::default());

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                let col = self.indices[idx];
                let val = self.data[idx].clone();
                dense[[row, col]] = val;
            }
        }

        dense
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Calculate sparsity ratio (fraction of non-zero elements)
    pub fn sparsity(&self) -> f64 {
        let total_elements = self.shape.0 * self.shape.1;
        if total_elements == 0 {
            0.0
        } else {
            self.nnz() as f64 / total_elements as f64
        }
    }

    /// Get values for a specific row
    pub fn get_row(&self, row: usize) -> Vec<(usize, T)> {
        if row >= self.shape.0 {
            return Vec::new();
        }

        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        let mut row_data = Vec::new();

        for idx in start..end {
            let col = self.indices[idx];
            let val = self.data[idx].clone();
            row_data.push((col, val));
        }

        row_data
    }

    /// Set a value at specific row and column
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if row >= self.shape.0 || col >= self.shape.1 {
            return;
        }

        let start = self.indptr[row];
        let end = self.indptr[row + 1];

        // Find if the element already exists
        for idx in start..end {
            if self.indices[idx] == col {
                if value == T::default() {
                    // Remove the element
                    self.data.remove(idx);
                    self.indices.remove(idx);
                    // Update indptr for all following rows
                    for r in (row + 1)..=self.shape.0 {
                        self.indptr[r] -= 1;
                    }
                } else {
                    // Update the value
                    self.data[idx] = value;
                }
                return;
            }
            if self.indices[idx] > col {
                // Insert at this position
                if value != T::default() {
                    self.data.insert(idx, value);
                    self.indices.insert(idx, col);
                    // Update indptr for all following rows
                    for r in (row + 1)..=self.shape.0 {
                        self.indptr[r] += 1;
                    }
                }
                return;
            }
        }

        // Append at the end of this row
        if value != T::default() {
            self.data.insert(end, value);
            self.indices.insert(end, col);
            // Update indptr for all following rows
            for r in (row + 1)..=self.shape.0 {
                self.indptr[r] += 1;
            }
        }
    }
}

/// Memory-efficient sparse multi-output predictor
#[derive(Debug, Clone)]
pub struct SparseMultiOutput<S = Untrained> {
    state: S,
    /// Sparsity threshold - values below this are considered zero
    sparsity_threshold: f64,
    /// Whether to use compressed storage for predictions
    use_compression: bool,
}

/// Trained state for sparse multi-output predictor
#[derive(Debug, Clone)]
pub struct SparseMultiOutputTrained {
    pub coefficients: CSRMatrix<f64>,
    pub bias: HashMap<usize, f64>,
    pub feature_means: Array1<f64>,
    pub feature_stds: Array1<f64>,
    pub n_features: usize,
    pub n_outputs: usize,
    pub sparsity_ratio: f64,
}

impl SparseMultiOutput<Untrained> {
    /// Create a new sparse multi-output predictor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            sparsity_threshold: 1e-6,
            use_compression: true,
        }
    }

    /// Set the sparsity threshold
    pub fn sparsity_threshold(mut self, threshold: f64) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Enable or disable compression
    pub fn use_compression(mut self, use_compression: bool) -> Self {
        self.use_compression = use_compression;
        self
    }
}

impl Default for SparseMultiOutput<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SparseMultiOutput<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for SparseMultiOutput<SparseMultiOutputTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, f64>> for SparseMultiOutput<Untrained> {
    type Fitted = SparseMultiOutput<SparseMultiOutputTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView2<'_, f64>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let (n_samples_y, n_outputs) = y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Convert X to f64 for consistency
        let X_f64 = X.mapv(|x| x);

        // Compute feature statistics for standardization
        let mut feature_means = Array1::zeros(n_features);
        let mut feature_stds = Array1::zeros(n_features);

        for feature in 0..n_features {
            let col = X_f64.column(feature);
            feature_means[feature] = col.sum() / n_samples as f64;

            let variance = col
                .iter()
                .map(|&x| (x - feature_means[feature]).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            feature_stds[feature] = variance.sqrt().max(1e-8); // Avoid division by zero
        }

        // Standardize X
        let mut X_std = X_f64.clone();
        for feature in 0..n_features {
            let mut col = X_std.column_mut(feature);
            col -= feature_means[feature];
            col /= feature_stds[feature];
        }

        // Train sparse linear models using coordinate descent
        let mut coefficients_dense = Array2::zeros((n_outputs, n_features));
        let mut bias = HashMap::new();

        for output in 0..n_outputs {
            let y_target = y.column(output);

            // Simple ridge regression for each output
            let mut weights = Array1::zeros(n_features);
            let intercept = y_target.mean().unwrap_or(0.0);

            // Coordinate descent iterations
            for _iter in 0..100 {
                let mut converged = true;

                // Update each weight
                for feature in 0..n_features {
                    let old_weight = weights[feature];

                    // Compute residuals without this feature
                    let mut residual_sum = 0.0;
                    for sample in 0..n_samples {
                        let mut pred = intercept;
                        for other_feature in 0..n_features {
                            if other_feature != feature {
                                pred += weights[other_feature] * X_std[[sample, other_feature]];
                            }
                        }
                        let residual = y_target[sample] - pred;
                        residual_sum += residual * X_std[[sample, feature]];
                    }

                    // Feature variance (standardized features have variance 1)
                    let feature_var = n_samples as f64;

                    // Ridge penalty
                    let lambda = 0.01;
                    let new_weight = residual_sum / (feature_var + lambda);

                    // Apply sparsity threshold
                    weights[feature] = if new_weight.abs() < self.sparsity_threshold {
                        0.0
                    } else {
                        new_weight
                    };

                    if (weights[feature] - old_weight).abs() > 1e-6 {
                        converged = false;
                    }
                }

                if converged {
                    break;
                }
            }

            // Store results
            for feature in 0..n_features {
                coefficients_dense[[output, feature]] = weights[feature];
            }

            if intercept.abs() > self.sparsity_threshold {
                bias.insert(output, intercept);
            }
        }

        // Convert to sparse format
        let coefficients = CSRMatrix::from_dense(&coefficients_dense.view());
        let sparsity_ratio = coefficients.sparsity();

        Ok(SparseMultiOutput {
            state: SparseMultiOutputTrained {
                coefficients,
                bias,
                feature_means,
                feature_stds,
                n_features,
                n_outputs,
                sparsity_ratio,
            },
            sparsity_threshold: self.sparsity_threshold,
            use_compression: self.use_compression,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<f64>> for SparseMultiOutput<SparseMultiOutputTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features, n_features
            )));
        }

        // Standardize input features
        let mut X_std = X.mapv(|x| x);
        for feature in 0..n_features {
            let mut col = X_std.column_mut(feature);
            col -= self.state.feature_means[feature];
            col /= self.state.feature_stds[feature];
        }

        let mut predictions = Array2::zeros((n_samples, self.state.n_outputs));

        // Sparse matrix-vector multiplication
        for output in 0..self.state.n_outputs {
            let output_coeffs = self.state.coefficients.get_row(output);
            let intercept = *self.state.bias.get(&output).unwrap_or(&0.0);

            for sample in 0..n_samples {
                let mut pred = intercept;

                // Only compute for non-zero coefficients
                for &(feature, coeff) in &output_coeffs {
                    pred += coeff * X_std[[sample, feature]];
                }

                predictions[[sample, output]] = pred;
            }
        }

        Ok(predictions)
    }
}

impl SparseMultiOutput<SparseMultiOutputTrained> {
    /// Get the sparsity ratio of the coefficient matrix
    pub fn sparsity_ratio(&self) -> f64 {
        self.state.sparsity_ratio
    }

    /// Get the number of non-zero coefficients
    pub fn nnz_coefficients(&self) -> usize {
        self.state.coefficients.nnz()
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        let dense_size = self.state.n_outputs * self.state.n_features * 8; // 8 bytes per f64
        let sparse_size = self.state.coefficients.data.len() * 8 + // data values
                         self.state.coefficients.indices.len() * 8 + // column indices
                         self.state.coefficients.indptr.len() * 8; // row pointers

        let compression_ratio = if dense_size > 0 {
            sparse_size as f64 / dense_size as f64
        } else {
            1.0
        };

        MemoryUsage {
            dense_size_bytes: dense_size,
            sparse_size_bytes: sparse_size,
            compression_ratio,
            memory_saved_bytes: dense_size.saturating_sub(sparse_size),
        }
    }

    /// Get coefficients for a specific output (sparse representation)
    pub fn get_output_coefficients(&self, output: usize) -> Vec<(usize, f64)> {
        if output >= self.state.n_outputs {
            return Vec::new();
        }

        self.state.coefficients.get_row(output)
    }

    /// Get the bias for a specific output
    pub fn get_output_bias(&self, output: usize) -> f64 {
        *self.state.bias.get(&output).unwrap_or(&0.0)
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Size of dense representation in bytes
    pub dense_size_bytes: usize,
    /// Size of sparse representation in bytes
    pub sparse_size_bytes: usize,
    /// Compression ratio (sparse_size / dense_size)
    pub compression_ratio: f64,
    /// Memory saved in bytes
    pub memory_saved_bytes: usize,
}

impl fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,
            "Memory Usage - Dense: {} bytes, Sparse: {} bytes, Compression: {:.3}x, Saved: {} bytes",
            self.dense_size_bytes,
            self.sparse_size_bytes,
            self.compression_ratio,
            self.memory_saved_bytes
        )
    }
}

/// Utility functions for sparse output analysis
pub mod sparse_utils {
    use super::*;

    /// Analyze sparsity patterns in output data
    pub fn analyze_output_sparsity(y: &ArrayView2<f64>, threshold: f64) -> SparsityAnalysis {
        let (n_samples, n_outputs) = y.dim();
        let mut total_elements = 0;
        let mut zero_elements = 0;
        let mut output_sparsities = Vec::with_capacity(n_outputs);

        for output in 0..n_outputs {
            let col = y.column(output);
            let output_zeros = col.iter().filter(|&&x| x.abs() <= threshold).count();
            let output_sparsity = output_zeros as f64 / n_samples as f64;
            output_sparsities.push(output_sparsity);

            total_elements += n_samples;
            zero_elements += output_zeros;
        }

        let overall_sparsity = zero_elements as f64 / total_elements as f64;
        let avg_sparsity = output_sparsities.iter().sum::<f64>() / n_outputs as f64;
        let min_sparsity = output_sparsities
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_sparsity = output_sparsities
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        SparsityAnalysis {
            overall_sparsity,
            avg_sparsity,
            min_sparsity,
            max_sparsity,
            output_sparsities,
            total_elements,
            zero_elements,
        }
    }

    /// Recommend whether to use sparse storage based on data characteristics
    pub fn recommend_sparse_storage(y: &ArrayView2<f64>, threshold: f64) -> StorageRecommendation {
        let analysis = analyze_output_sparsity(y, threshold);

        let should_use_sparse = analysis.overall_sparsity > 0.5; // More than 50% zeros
        let expected_compression = if should_use_sparse {
            // Estimate compression based on sparsity
            1.0 - analysis.overall_sparsity + 0.1 // Add overhead estimate
        } else {
            1.0
        };

        StorageRecommendation {
            should_use_sparse,
            expected_compression_ratio: expected_compression,
            sparsity_analysis: analysis,
        }
    }
}

/// Sparsity analysis results
#[derive(Debug, Clone)]
pub struct SparsityAnalysis {
    /// Overall fraction of zero elements
    pub overall_sparsity: f64,
    /// Average sparsity across outputs
    pub avg_sparsity: f64,
    /// Minimum sparsity among outputs
    pub min_sparsity: f64,
    /// Maximum sparsity among outputs
    pub max_sparsity: f64,
    /// Sparsity for each output
    pub output_sparsities: Vec<f64>,
    /// Total number of elements
    pub total_elements: usize,
    /// Number of zero elements
    pub zero_elements: usize,
}

/// Storage recommendation based on data analysis
#[derive(Debug, Clone)]
pub struct StorageRecommendation {
    /// Whether sparse storage is recommended
    pub should_use_sparse: bool,
    /// Expected compression ratio
    pub expected_compression_ratio: f64,
    /// Detailed sparsity analysis
    pub sparsity_analysis: SparsityAnalysis,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    fn test_csr_matrix_basic() {
        let dense = array![[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 0.0]];
        let csr = CSRMatrix::from_dense(&dense.view());

        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.data, vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(csr.indices, vec![0, 2, 1, 0]);
        assert_eq!(csr.indptr, vec![0, 2, 3, 4]);

        let reconstructed = csr.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(dense[[i, j]], reconstructed[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_csr_sparsity() {
        let dense = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]];
        let csr = CSRMatrix::from_dense(&dense.view());

        assert_abs_diff_eq!(csr.sparsity(), 2.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_sparse_multi_output_basic() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![
            [1.0, 0.0, 0.1],
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.2]
        ];

        let model = SparseMultiOutput::new().sparsity_threshold(0.05);
        let trained = model.fit(&X.view(), &y.view()).unwrap();

        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[4, 3]);

        // Check that model learned something reasonable
        assert!(trained.sparsity_ratio() < 1.0); // Should have some non-zero coefficients
        println!("Sparsity ratio: {}", trained.sparsity_ratio());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_sparse_memory_efficiency() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0]
        ];
        // Highly sparse output - most values are zero
        let y = array![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
        ];

        let model = SparseMultiOutput::new().sparsity_threshold(1e-6);
        let trained = model.fit(&X.view(), &y.view()).unwrap();

        let memory_usage = trained.memory_usage();
        println!("{}", memory_usage);

        // Should achieve significant compression for sparse data
        assert!(memory_usage.compression_ratio < 0.8); // At least 20% compression
        assert!(memory_usage.memory_saved_bytes > 0);
    }

    #[test]
    fn test_sparsity_analysis() {
        let y = array![
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0]
        ];

        let analysis = sparse_utils::analyze_output_sparsity(&y.view(), 1e-6);

        // Actually 11 out of 16 elements are zero (5 non-zero: 1.0, 2.0, 3.0, 1.0, 2.0)
        assert_abs_diff_eq!(analysis.overall_sparsity, 11.0 / 16.0, epsilon = 1e-10);
        assert_eq!(analysis.total_elements, 16);
        assert_eq!(analysis.zero_elements, 11);
        assert_eq!(analysis.output_sparsities.len(), 4);
    }

    #[test]
    fn test_storage_recommendation() {
        // Sparse data
        let y_sparse = array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ];

        let recommendation = sparse_utils::recommend_sparse_storage(&y_sparse.view(), 1e-6);
        assert!(recommendation.should_use_sparse);
        assert!(recommendation.expected_compression_ratio < 1.0);

        // Dense data
        let y_dense = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
        ];

        let recommendation = sparse_utils::recommend_sparse_storage(&y_dense.view(), 1e-6);
        assert!(!recommendation.should_use_sparse);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_sparse_coefficient_access() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0, 0.0], [0.0, 2.0]];

        let model = SparseMultiOutput::new().sparsity_threshold(1e-3);
        let trained = model.fit(&X.view(), &y.view()).unwrap();

        // Test coefficient access for each output
        for output in 0..2 {
            let coeffs = trained.get_output_coefficients(output);
            let bias = trained.get_output_bias(output);

            println!("Output {}: coeffs = {:?}, bias = {}", output, coeffs, bias);

            // Should have some coefficients
            assert!(!coeffs.is_empty() || bias.abs() > 1e-6);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_edge_cases() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];

        // All zeros - coefficients should be small due to regularization
        let y_zeros = array![[0.0, 0.0], [0.0, 0.0]];
        let model = SparseMultiOutput::new().sparsity_threshold(1e-3);
        let trained = model.fit(&X.view(), &y_zeros.view()).unwrap();

        // Check that predictions are close to zero
        let pred_zeros = trained.predict(&X.view()).unwrap();
        for i in 0..pred_zeros.nrows() {
            for j in 0..pred_zeros.ncols() {
                assert!(
                    pred_zeros[[i, j]].abs() < 0.1,
                    "Prediction should be close to zero: {}",
                    pred_zeros[[i, j]]
                );
            }
        }

        println!("Zero data sparsity ratio: {}", trained.sparsity_ratio());

        // Single feature
        let X_single = array![[1.0], [2.0]];
        let y_single = array![[1.0], [2.0]];
        let model_single = SparseMultiOutput::new();
        let trained_single = model_single
            .fit(&X_single.view(), &y_single.view())
            .unwrap();
        let pred = trained_single.predict(&X_single.view()).unwrap();
        assert_eq!(pred.shape(), &[2, 1]);

        // Test with many zero outputs
        let X_many = array![[1.0, 2.0], [3.0, 4.0]];
        let y_many_sparse = array![[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0]];
        let model_many = SparseMultiOutput::new().sparsity_threshold(1e-6);
        let trained_many = model_many
            .fit(&X_many.view(), &y_many_sparse.view())
            .unwrap();

        // Just check that training completed and we can make predictions
        let pred_many = trained_many.predict(&X_many.view()).unwrap();
        assert_eq!(pred_many.shape(), &[2, 5]);

        println!(
            "Many sparse outputs sparsity ratio: {}",
            trained_many.sparsity_ratio()
        );
    }
}
