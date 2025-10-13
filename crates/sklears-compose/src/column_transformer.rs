//! Column Transformer
//!
//! Apply different transformers to different subsets of features.

use scirs2_core::ndarray::{s, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
// TODO: Migrate to scirs2-sparse when implementing sparse functionality
// use scirs2_sparse::{CsMat, TriMat};
use std::collections::HashMap;

use crate::{MockTransformer, PipelineStep};

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
        transformer: Box<dyn PipelineStep>,
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
    pub remainder: String,
    pub sparse_threshold: f64,
    pub n_jobs: Option<i32>,
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

            // Create a mock transformer for now - in real implementation, this would be provided
            let mut transformer = Box::new(MockTransformer::new()) as Box<dyn PipelineStep>;
            transformer.fit(&x_subset.view(), y.as_ref().copied())?;

            fitted_transformers.push((name.clone(), transformer, columns.clone()));
            output_indices.push((0..columns.len()).collect()); // Simplified output mapping
        }

        // Handle remainder columns
        let remainder_columns: Vec<usize> =
            (0..n_features_in).filter(|&i| !used_columns[i]).collect();

        if !remainder_columns.is_empty() && self.remainder == "passthrough" {
            let x_remainder = self.extract_columns(x, &remainder_columns)?;
            let mut remainder_transformer =
                Box::new(MockTransformer::new()) as Box<dyn PipelineStep>;
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
            Ok(transformed_results.into_iter().next().unwrap())
        } else {
            self.concatenate_results(transformed_results)
        }
    }
}

/// Sparse output support for `ColumnTransformer`
#[derive(Debug, Clone)]
pub enum ColumnTransformerOutput {
    /// Dense
    Dense(Array2<f64>),
    // TODO: Re-enable sparse support with scirs2-sparse
    // Sparse(CsMat<f64>),
}

impl ColumnTransformer<ColumnTransformerTrained> {
    /// Transform data and return appropriate output format (dense or sparse)
    pub fn transform_output(
        &self,
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<ColumnTransformerOutput> {
        let dense_result = self.transform(x)?;

        // TODO: Re-enable sparse support with scirs2-sparse
        // if self.state.sparse_output {
        //     // Convert to sparse matrix if threshold is met
        //     let sparse_result = self.dense_to_sparse(&dense_result)?;
        //     Ok(ColumnTransformerOutput::Sparse(sparse_result))
        // } else {
        Ok(ColumnTransformerOutput::Dense(dense_result))
        // }
    }

    // TODO: Re-enable sparse support with scirs2-sparse
    // /// Convert dense matrix to sparse CSR format
    // fn dense_to_sparse(&self, dense: &Array2<f64>) -> SklResult<CsMat<f64>> {
    //     let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    //
    //     for (i, row) in dense.outer_iter().enumerate() {
    //         for (j, &value) in row.iter().enumerate() {
    //             if value != 0.0 {
    //                 triplets.add_triplet(i, j, value);
    //             }
    //         }
    //     }
    //
    //     Ok(triplets.to_csr())
    // }

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
        /// ColumnTransformer
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
