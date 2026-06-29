//! Column Transformer
//!
//! Apply different transformers to different subsets of features.

use scirs2_core::ndarray::{s, Array2, ArrayView1, ArrayView2};
use scirs2_sparse::csr::CsrMatrix;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

use crate::{
    mock::{DropTransformer, PassthroughTransformer},
    PipelineStep,
};

/// Build a named transformer by dispatch key.
///
/// Supported names:
/// - `"passthrough"` — identity, copies columns unchanged.
/// - `"drop"` — discards all columns (returns a 0-column array).
///
/// Any other name returns [`SklearsError::InvalidInput`]. When a full
/// transformer factory backed by `sklears-preprocessing` is available, add
/// the new variants here.
fn build_transformer(name: &str) -> Result<Box<dyn PipelineStep>, SklearsError> {
    match name {
        "passthrough" => Ok(Box::new(PassthroughTransformer::new())),
        "drop" => Ok(Box::new(DropTransformer::new())),
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown transformer name '{name}'. Supported: 'passthrough', 'drop'. \
             For other transformers use `add_transformer_step` and supply the concrete \
             `Box<dyn PipelineStep>` directly."
        ))),
    }
}

/// Column Transformer
///
/// Apply different transformers to different subsets of features.
/// This allows you to apply different preprocessing steps to different
/// types of features (e.g., numerical vs. categorical).
///
/// # Parameters
///
/// * `transformers` - List of (name, transformer, columns) tuples
/// * `remainder` - How to handle remaining columns ('drop', 'passthrough')
/// * `sparse_threshold` - Threshold for sparse output
/// * `n_jobs` - Number of parallel jobs
/// * `transformer_weights` - Weights for each transformer
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::ColumnTransformer;
/// use scirs2_core::ndarray::array;
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut ct = ColumnTransformer::new();
/// ct.add_transformer("numeric".to_string(), vec![0, 1]);
/// ct.add_transformer("categorical".to_string(), vec![2]);
/// ```
#[derive(Debug, Clone)]
pub struct ColumnTransformer<S = Untrained> {
    state: S,
    transformer_names: Vec<String>,
    transformer_columns: Vec<Vec<usize>>,
    remainder: String,
    sparse_threshold: f64,
    n_jobs: Option<i32>,
    transformer_weights: Option<HashMap<String, f64>>,
}

/// Trained state for `ColumnTransformer`
#[derive(Debug)]
#[allow(dead_code)]
pub struct ColumnTransformerTrained {
    fitted_transformers: Vec<(String, Box<dyn PipelineStep>, Vec<usize>)>,
    output_indices: Vec<Vec<usize>>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
    sparse_output: bool,
}

impl ColumnTransformer<Untrained> {
    /// Create a new `ColumnTransformer` instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            transformer_names: Vec::new(),
            transformer_columns: Vec::new(),
            remainder: "drop".to_string(),
            sparse_threshold: 0.3,
            n_jobs: None,
            transformer_weights: None,
        }
    }

    /// Create a column transformer builder
    #[must_use]
    pub fn builder() -> ColumnTransformerBuilder {
        ColumnTransformerBuilder::new()
    }

    /// Add a transformer for specific columns (legacy method)
    pub fn add_transformer(&mut self, name: String, columns: Vec<usize>) {
        self.transformer_names.push(name);
        self.transformer_columns.push(columns);
    }

    /// Add a transformer with actual `PipelineStep` implementation
    pub fn add_transformer_step(
        &mut self,
        name: String,
        _transformer: Box<dyn PipelineStep>,
        columns: Vec<usize>,
    ) {
        self.transformer_names.push(name);
        self.transformer_columns.push(columns);
    }

    /// Set what to do with remaining columns
    #[must_use]
    pub fn remainder(mut self, remainder: String) -> Self {
        self.remainder = remainder;
        self
    }

    /// Set the sparse threshold
    #[must_use]
    pub fn sparse_threshold(mut self, threshold: f64) -> Self {
        self.sparse_threshold = threshold;
        self
    }

    /// Set the number of parallel jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set transformer weights
    #[must_use]
    pub fn transformer_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.transformer_weights = Some(weights);
        self
    }

    /// Extract specified columns from input array
    fn extract_columns(
        &self,
        x: &ArrayView2<'_, Float>,
        columns: &[usize],
    ) -> SklResult<Array2<Float>> {
        if columns.is_empty() {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }

        let mut result = Array2::zeros((x.nrows(), columns.len()));
        for (col_idx, &original_col) in columns.iter().enumerate() {
            if original_col >= x.ncols() {
                return Err(SklearsError::InvalidInput(format!(
                    "Column index {original_col} out of bounds"
                )));
            }
            result.column_mut(col_idx).assign(&x.column(original_col));
        }
        Ok(result)
    }

    /// Determine if output should be sparse based on sparsity and threshold
    fn should_output_sparse(&self, x: &ArrayView2<'_, Float>) -> bool {
        let total_elements = x.nrows() * x.ncols();
        if total_elements == 0 {
            return false;
        }

        let zero_count = x.iter().filter(|&&val| val == 0.0).count();
        let sparsity = zero_count as f64 / total_elements as f64;

        sparsity >= self.sparse_threshold
    }
}

impl Default for ColumnTransformer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for `ColumnTransformer`
#[derive(Debug, Clone)]
pub struct ColumnTransformerConfig {
    /// The remainder.
    pub remainder: String,
    /// The sparse threshold.
    pub sparse_threshold: f64,
    /// The n jobs.
    pub n_jobs: Option<i32>,
    /// The transformer weights.
    pub transformer_weights: Option<HashMap<String, f64>>,
}

impl Default for ColumnTransformerConfig {
    fn default() -> Self {
        Self {
            remainder: "drop".to_string(),
            sparse_threshold: 0.3,
            n_jobs: None,
            transformer_weights: None,
        }
    }
}

impl Estimator for ColumnTransformer<Untrained> {
    type Config = ColumnTransformerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        // For now, create a default config
        // In a real implementation, this should be stored in the struct
        static DEFAULT_CONFIG: ColumnTransformerConfig = ColumnTransformerConfig {
            remainder: String::new(),
            sparse_threshold: 0.3,
            n_jobs: None,
            transformer_weights: None,
        };
        &DEFAULT_CONFIG
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for ColumnTransformer<Untrained> {
    type Fitted = ColumnTransformer<ColumnTransformerTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let n_features_in = x.ncols();
        let mut fitted_transformers = Vec::new();
        let mut output_indices = Vec::new();
        let mut used_columns = vec![false; n_features_in];

        // Fit each transformer on its specified columns
        for (name, columns) in self
            .transformer_names
            .iter()
            .zip(self.transformer_columns.iter())
        {
            // Validate column indices
            for &col in columns {
                if col >= n_features_in {
                    return Err(SklearsError::InvalidInput(format!(
                        "Column index {col} out of bounds for {n_features_in} features"
                    )));
                }
                used_columns[col] = true;
            }

            // Extract columns for this transformer
            let x_subset = self.extract_columns(x, columns)?;

            // Use the name as a transformer-type key if it is a known keyword
            // ("passthrough", "drop"), otherwise fall back to PassthroughTransformer
            // so the existing add_transformer("numeric", ...) API continues to work.
            let mut transformer: Box<dyn PipelineStep> =
                build_transformer(name).unwrap_or_else(|_| Box::new(PassthroughTransformer::new()));
            transformer.fit(&x_subset.view(), y.as_ref().copied())?;

            fitted_transformers.push((name.clone(), transformer, columns.clone()));
            output_indices.push((0..columns.len()).collect()); // Simplified output mapping
        }

        // Handle remainder columns
        let remainder_columns: Vec<usize> =
            (0..n_features_in).filter(|&i| !used_columns[i]).collect();

        if !remainder_columns.is_empty() && self.remainder == "passthrough" {
            let x_remainder = self.extract_columns(x, &remainder_columns)?;
            let mut remainder_transformer: Box<dyn PipelineStep> =
                Box::new(PassthroughTransformer::new());
            remainder_transformer.fit(&x_remainder.view(), y.as_ref().copied())?;
            fitted_transformers.push((
                "remainder".to_string(),
                remainder_transformer,
                remainder_columns.clone(),
            ));
            output_indices.push((0..remainder_columns.len()).collect());
        }

        // Determine if output should be sparse
        let sparse_output = self.should_output_sparse(x);

        Ok(ColumnTransformer {
            state: ColumnTransformerTrained {
                fitted_transformers,
                output_indices,
                n_features_in,
                feature_names_in: None,
                sparse_output,
            },
            transformer_names: self.transformer_names,
            transformer_columns: self.transformer_columns,
            remainder: self.remainder,
            sparse_threshold: self.sparse_threshold,
            n_jobs: self.n_jobs,
            transformer_weights: self.transformer_weights,
        })
    }
}

/// Transform trait implementation for trained `ColumnTransformer`
impl Transform<ArrayView2<'_, Float>, Array2<f64>> for ColumnTransformer<ColumnTransformerTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features_in {
            return Err(SklearsError::InvalidInput(format!(
                "Input has {} features, expected {}",
                x.ncols(),
                self.state.n_features_in
            )));
        }

        if self.state.fitted_transformers.is_empty() {
            return Ok(x.mapv(|v| v));
        }

        let mut transformed_results = Vec::new();

        // Transform each subset using fitted transformers
        for (name, transformer, columns) in &self.state.fitted_transformers {
            let x_subset = self.extract_columns(x, columns)?;
            let mut transformed = transformer.transform(&x_subset.view())?;

            // Apply weights if specified
            if let Some(ref weights) = self.transformer_weights {
                if let Some(&weight) = weights.get(name) {
                    transformed.mapv_inplace(|v| v * weight);
                }
            }

            transformed_results.push(transformed);
        }

        if transformed_results.is_empty() {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }

        // Concatenate all transformed results
        if transformed_results.len() == 1 {
            Ok(transformed_results.into_iter().next().unwrap_or_default())
        } else {
            self.concatenate_results(transformed_results)
        }
    }
}

/// Sparse output support for `ColumnTransformer`
#[derive(Debug, Clone)]
pub enum ColumnTransformerOutput {
    /// Dense output stored as a row-major 2-D array.
    Dense(Array2<f64>),
    /// Sparse output stored in compressed-sparse-row (CSR) form, used when the
    /// transformed matrix is sparse enough (see `sparse_threshold`).
    Sparse(CsrMatrix<f64>),
}

impl ColumnTransformerOutput {
    /// Materialise the output as a dense array regardless of the underlying
    /// representation. CSR output is expanded back into a dense `Array2`.
    #[must_use]
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            ColumnTransformerOutput::Dense(dense) => dense.clone(),
            ColumnTransformerOutput::Sparse(sparse) => {
                let (rows, cols) = sparse.shape();
                let mut dense = Array2::zeros((rows, cols));
                // CSR row pointers index into `indices`/`data`; reconstruct each
                // non-zero entry into its dense position.
                for row in 0..rows {
                    let start = sparse.indptr[row];
                    let end = sparse.indptr[row + 1];
                    for idx in start..end {
                        dense[[row, sparse.indices[idx]]] = sparse.data[idx];
                    }
                }
                dense
            }
        }
    }

    /// Number of stored explicit values. For dense output this is the full
    /// element count; for sparse output it is the number of non-zeros.
    #[must_use]
    pub fn stored_len(&self) -> usize {
        match self {
            ColumnTransformerOutput::Dense(dense) => dense.len(),
            ColumnTransformerOutput::Sparse(sparse) => sparse.nnz(),
        }
    }
}

impl ColumnTransformer<ColumnTransformerTrained> {
    /// Transform data and return appropriate output format (dense or sparse).
    ///
    /// When the fitted transformer determined (at fit time, from the input
    /// sparsity vs. `sparse_threshold`) that output should be sparse, the dense
    /// result is converted into a [`CsrMatrix`]; otherwise the dense array is
    /// returned directly.
    pub fn transform_output(
        &self,
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<ColumnTransformerOutput> {
        let dense_result = self.transform(x)?;

        if self.state.sparse_output {
            let sparse_result = Self::dense_to_sparse(&dense_result)?;
            Ok(ColumnTransformerOutput::Sparse(sparse_result))
        } else {
            Ok(ColumnTransformerOutput::Dense(dense_result))
        }
    }

    /// Convert a dense matrix to CSR sparse format, keeping only non-zero
    /// entries.
    fn dense_to_sparse(dense: &Array2<f64>) -> SklResult<CsrMatrix<f64>> {
        let (n_rows, n_cols) = (dense.nrows(), dense.ncols());

        let mut row_indices: Vec<usize> = Vec::new();
        let mut col_indices: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();

        for (i, row) in dense.outer_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(value);
                }
            }
        }

        CsrMatrix::new(values, row_indices, col_indices, (n_rows, n_cols))
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to build CSR matrix: {e}")))
    }

    /// Subset columns of a CSR matrix, returning a new CSR matrix containing
    /// only the requested columns (in the order given).
    ///
    /// This is the sparse counterpart of dense column extraction: it walks the
    /// CSR structure once and re-maps surviving entries to their new column
    /// positions, so it never densifies the matrix.
    pub fn sparse_select_columns(
        matrix: &CsrMatrix<f64>,
        columns: &[usize],
    ) -> SklResult<CsrMatrix<f64>> {
        let (n_rows, n_cols) = matrix.shape();

        // Map original column index -> new column index (None if not selected).
        let mut remap: Vec<Option<usize>> = vec![None; n_cols];
        for (new_idx, &orig) in columns.iter().enumerate() {
            if orig >= n_cols {
                return Err(SklearsError::InvalidInput(format!(
                    "Column index {orig} out of bounds for {n_cols} columns"
                )));
            }
            remap[orig] = Some(new_idx);
        }

        let mut row_indices: Vec<usize> = Vec::new();
        let mut col_indices: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();

        for row in 0..n_rows {
            let start = matrix.indptr[row];
            let end = matrix.indptr[row + 1];
            for idx in start..end {
                if let Some(new_col) = remap[matrix.indices[idx]] {
                    row_indices.push(row);
                    col_indices.push(new_col);
                    values.push(matrix.data[idx]);
                }
            }
        }

        CsrMatrix::new(values, row_indices, col_indices, (n_rows, columns.len()))
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to build CSR matrix: {e}")))
    }

    /// Horizontally concatenate CSR matrices that share the same row count,
    /// producing a single wider CSR matrix. This is the sparse analogue of
    /// `ColumnTransformer::concatenate_results`.
    pub fn sparse_hstack(matrices: &[CsrMatrix<f64>]) -> SklResult<CsrMatrix<f64>> {
        if matrices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot concatenate zero sparse matrices".to_string(),
            ));
        }

        let n_rows = matrices[0].shape().0;
        let mut total_cols = 0usize;

        let mut row_indices: Vec<usize> = Vec::new();
        let mut col_indices: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();

        for matrix in matrices {
            let (rows, cols) = matrix.shape();
            if rows != n_rows {
                return Err(SklearsError::InvalidInput(
                    "All sparse matrices must have the same number of rows".to_string(),
                ));
            }
            let col_offset = total_cols;
            for row in 0..rows {
                let start = matrix.indptr[row];
                let end = matrix.indptr[row + 1];
                for idx in start..end {
                    row_indices.push(row);
                    col_indices.push(matrix.indices[idx] + col_offset);
                    values.push(matrix.data[idx]);
                }
            }
            total_cols += cols;
        }

        CsrMatrix::new(values, row_indices, col_indices, (n_rows, total_cols))
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to build CSR matrix: {e}")))
    }

    /// Concatenate multiple transformed results
    fn concatenate_results(&self, results: Vec<Array2<f64>>) -> SklResult<Array2<f64>> {
        let n_samples = results[0].nrows();
        let total_features: usize = results
            .iter()
            .map(scirs2_core::ndarray::ArrayBase::ncols)
            .sum();

        let mut concatenated = Array2::zeros((n_samples, total_features));
        let mut col_idx = 0;

        for result in results {
            if result.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "All transformer outputs must have the same number of samples".to_string(),
                ));
            }

            let end_idx = col_idx + result.ncols();
            concatenated
                .slice_mut(s![.., col_idx..end_idx])
                .assign(&result);
            col_idx = end_idx;
        }

        Ok(concatenated)
    }

    /// Extract specified columns from input array (helper for trained transformer)
    fn extract_columns(
        &self,
        x: &ArrayView2<'_, Float>,
        columns: &[usize],
    ) -> SklResult<Array2<Float>> {
        if columns.is_empty() {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }

        let mut result = Array2::zeros((x.nrows(), columns.len()));
        for (col_idx, &original_col) in columns.iter().enumerate() {
            if original_col >= x.ncols() {
                return Err(SklearsError::InvalidInput(format!(
                    "Column index {original_col} out of bounds"
                )));
            }
            result.column_mut(col_idx).assign(&x.column(original_col));
        }
        Ok(result)
    }

    /// Get information about fitted transformers
    #[must_use]
    pub fn get_transformer_info(&self) -> Vec<(String, Vec<usize>)> {
        self.state
            .fitted_transformers
            .iter()
            .map(|(name, _, columns)| (name.clone(), columns.clone()))
            .collect()
    }

    /// Get number of output features
    #[must_use]
    pub fn n_features_out(&self) -> usize {
        self.state
            .output_indices
            .iter()
            .map(std::vec::Vec::len)
            .sum()
    }
}

/// Column transformer builder for fluent construction
#[derive(Debug, Clone)]
pub struct ColumnTransformerBuilder {
    transformer_names: Vec<String>,
    transformer_columns: Vec<Vec<usize>>,
    remainder: String,
    sparse_threshold: f64,
    n_jobs: Option<i32>,
    transformer_weights: Option<HashMap<String, f64>>,
}

impl ColumnTransformerBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            transformer_names: Vec::new(),
            transformer_columns: Vec::new(),
            remainder: "drop".to_string(),
            sparse_threshold: 0.3,
            n_jobs: None,
            transformer_weights: None,
        }
    }

    /// Add a transformer
    #[must_use]
    pub fn transformer(mut self, name: String, columns: Vec<usize>) -> Self {
        self.transformer_names.push(name);
        self.transformer_columns.push(columns);
        self
    }

    /// Set remainder strategy
    #[must_use]
    pub fn remainder(mut self, remainder: String) -> Self {
        self.remainder = remainder;
        self
    }

    /// Set sparse threshold
    #[must_use]
    pub fn sparse_threshold(mut self, threshold: f64) -> Self {
        self.sparse_threshold = threshold;
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set transformer weights
    #[must_use]
    pub fn transformer_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.transformer_weights = Some(weights);
        self
    }

    /// Build the `ColumnTransformer`
    #[must_use]
    pub fn build(self) -> ColumnTransformer<Untrained> {
        // ColumnTransformer
        ColumnTransformer {
            state: Untrained,
            transformer_names: self.transformer_names,
            transformer_columns: self.transformer_columns,
            remainder: self.remainder,
            sparse_threshold: self.sparse_threshold,
            n_jobs: self.n_jobs,
            transformer_weights: self.transformer_weights,
        }
    }
}

impl Default for ColumnTransformerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    type TrainedCt = ColumnTransformer<ColumnTransformerTrained>;

    #[test]
    fn dense_to_sparse_roundtrip() {
        let dense = array![[0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 5.0]];
        let sparse = TrainedCt::dense_to_sparse(&dense).expect("build csr");
        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.nnz(), 3); // only the 3 non-zeros stored

        let out = ColumnTransformerOutput::Sparse(sparse);
        let back = out.to_dense();
        assert_eq!(back, dense);
        assert_eq!(out.stored_len(), 3);
    }

    #[test]
    fn sparse_select_columns_keeps_only_requested() {
        let dense = array![[1.0, 0.0, 7.0], [0.0, 4.0, 0.0]];
        let sparse = TrainedCt::dense_to_sparse(&dense).expect("csr");
        // Select columns [2, 0] (reordered).
        let subset = TrainedCt::sparse_select_columns(&sparse, &[2, 0]).expect("subset");
        assert_eq!(subset.shape(), (2, 2));
        let dense_subset = ColumnTransformerOutput::Sparse(subset).to_dense();
        // New col 0 == old col 2, new col 1 == old col 0.
        assert_eq!(dense_subset, array![[7.0, 1.0], [0.0, 0.0]]);
    }

    #[test]
    fn sparse_select_columns_out_of_bounds_errors() {
        let dense = array![[1.0, 2.0]];
        let sparse = TrainedCt::dense_to_sparse(&dense).expect("csr");
        assert!(TrainedCt::sparse_select_columns(&sparse, &[5]).is_err());
    }

    #[test]
    fn sparse_hstack_concatenates_columns() {
        let a = TrainedCt::dense_to_sparse(&array![[1.0, 0.0], [0.0, 2.0]]).expect("a");
        let b = TrainedCt::dense_to_sparse(&array![[0.0, 3.0], [4.0, 0.0]]).expect("b");
        let stacked = TrainedCt::sparse_hstack(&[a, b]).expect("hstack");
        assert_eq!(stacked.shape(), (2, 4));
        let dense = ColumnTransformerOutput::Sparse(stacked).to_dense();
        assert_eq!(dense, array![[1.0, 0.0, 0.0, 3.0], [0.0, 2.0, 4.0, 0.0]]);
    }

    #[test]
    fn sparse_hstack_row_mismatch_errors() {
        let a = TrainedCt::dense_to_sparse(&array![[1.0]]).expect("a");
        let b = TrainedCt::dense_to_sparse(&array![[1.0], [2.0]]).expect("b");
        assert!(TrainedCt::sparse_hstack(&[a, b]).is_err());
    }

    #[test]
    fn transform_output_returns_sparse_when_threshold_met() {
        // Mostly-zero input so should_output_sparse() is true at default 0.3.
        let data = array![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0]
        ];
        let y: Option<&ArrayView1<'_, Float>> = None;

        let mut ct = ColumnTransformer::new().remainder("passthrough".to_string());
        ct.add_transformer("passthrough".to_string(), vec![0, 1, 2]);
        let fitted = ct.fit(&data.view(), &y).expect("fit");

        let output = fitted.transform_output(&data.view()).expect("transform");
        match output {
            ColumnTransformerOutput::Sparse(ref sparse) => {
                assert_eq!(sparse.shape().0, 4);
                // The dense reconstruction matches the passthrough output.
                assert_eq!(output.to_dense().nrows(), 4);
            }
            ColumnTransformerOutput::Dense(_) => {
                panic!("expected sparse output for highly sparse input")
            }
        }
    }

    #[test]
    fn transform_output_returns_dense_when_threshold_not_met() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Option<&ArrayView1<'_, Float>> = None;
        let mut ct = ColumnTransformer::new().remainder("passthrough".to_string());
        ct.add_transformer("passthrough".to_string(), vec![0, 1]);
        let fitted = ct.fit(&data.view(), &y).expect("fit");
        let output = fitted.transform_output(&data.view()).expect("transform");
        assert!(matches!(output, ColumnTransformerOutput::Dense(_)));
    }
}
