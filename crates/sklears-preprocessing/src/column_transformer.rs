//! Column Transformer
//!
//! This module provides ColumnTransformer which applies different transformers
//! to specific columns of a dataset.

use scirs2_core::ndarray::{s, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// For floating point comparison in HashSet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OrderedFloat(u64);

impl From<Float> for OrderedFloat {
    fn from(val: Float) -> Self {
        OrderedFloat(val.to_bits())
    }
}

/// Column selector type
#[derive(Debug, Clone)]
pub enum ColumnSelector {
    /// Select columns by indices
    Indices(Vec<usize>),
    /// Select columns by name (when working with named columns)
    Names(Vec<String>),
    /// Select columns by data type (would require runtime type checking)
    DataType(DataType),
    /// Select all remaining columns
    Remainder,
}

/// Data type enum for column selection
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Numeric,
    Categorical,
    Boolean,
}

/// Strategy for handling remaining columns
#[derive(Debug, Clone)]
pub enum RemainderStrategy {
    /// Drop remaining columns
    Drop,
    /// Pass through remaining columns unchanged
    Passthrough,
    /// Apply a specific transformer to remaining columns
    Transform(Box<dyn TransformerWrapper>),
}

/// Strategy for handling errors during column transformations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnErrorStrategy {
    /// Stop on first error
    StopOnError,
    /// Skip failed transformers and continue with others
    SkipOnError,
    /// Use fallback transformer for failed columns
    Fallback,
    /// Replace failed columns with zeros
    ReplaceWithZeros,
    /// Replace failed columns with NaN values
    ReplaceWithNaN,
}

impl Default for ColumnErrorStrategy {
    fn default() -> Self {
        Self::StopOnError
    }
}

impl Default for RemainderStrategy {
    fn default() -> Self {
        Self::Drop
    }
}

/// Trait for transformer wrappers to enable dynamic dispatch
pub trait TransformerWrapper: Send + Sync + std::fmt::Debug {
    fn fit_transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>>;
    fn transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>>;
    fn get_n_features_out(&self) -> Option<usize>;
    fn clone_box(&self) -> Box<dyn TransformerWrapper>;
}

impl Clone for Box<dyn TransformerWrapper> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A transformer step in the column transformer
#[derive(Debug, Clone)]
pub struct TransformerStep {
    /// Name of the transformer step
    pub name: String,
    /// The column selector
    pub columns: ColumnSelector,
    /// The transformer (boxed for dynamic dispatch)
    pub transformer: Box<dyn TransformerWrapper>,
}

/// Configuration for ColumnTransformer
#[derive(Debug, Clone)]
pub struct ColumnTransformerConfig {
    /// Strategy for handling remaining columns
    pub remainder: RemainderStrategy,
    /// Whether to preserve column order in output
    pub preserve_order: bool,
    /// Whether to use parallel processing
    pub n_jobs: Option<usize>,
    /// Whether to validate input
    pub validate_input: bool,
    /// Strategy for handling transformation errors
    pub error_strategy: ColumnErrorStrategy,
    /// Enable parallel processing for transformers
    pub parallel_execution: bool,
    /// Fallback transformer for error handling
    pub fallback_transformer: Option<Box<dyn TransformerWrapper>>,
}

impl Default for ColumnTransformerConfig {
    fn default() -> Self {
        Self {
            remainder: RemainderStrategy::Drop,
            preserve_order: false,
            n_jobs: None,
            validate_input: true,
            error_strategy: ColumnErrorStrategy::StopOnError,
            parallel_execution: false,
            fallback_transformer: None,
        }
    }
}

/// ColumnTransformer applies different transformers to different columns
#[derive(Debug)]
pub struct ColumnTransformer<State = Untrained> {
    config: ColumnTransformerConfig,
    transformers: Vec<TransformerStep>,
    state: PhantomData<State>,
    // Fitted parameters
    fitted_transformers_: Option<Vec<TransformerStep>>,
    feature_names_in_: Option<Vec<String>>,
    n_features_in_: Option<usize>,
    output_indices_: Option<HashMap<String, Vec<usize>>>,
    remainder_indices_: Option<Vec<usize>>,
}

/// Result of a column transformation attempt
#[derive(Debug)]
struct ColumnTransformResult {
    transformer_name: String,
    column_indices: Vec<usize>,
    result: Result<Array2<Float>>,
    original_indices: Vec<usize>,
}

// Shared methods for both Untrained and Trained states
impl<State> ColumnTransformer<State> {
    /// Apply transformer with error handling
    fn apply_transformer_with_error_handling(
        &self,
        step: &TransformerStep,
        _data: &Array2<Float>,
        subset: &Array2<Float>,
        is_fit_transform: bool,
        resolved_indices: &[usize],
    ) -> ColumnTransformResult {
        let column_indices = resolved_indices.to_vec();

        let transform_result = if is_fit_transform {
            step.transformer.fit_transform_wrapper(subset)
        } else {
            step.transformer.transform_wrapper(subset)
        };

        let final_result = match transform_result {
            Ok(transformed) => Ok(transformed),
            Err(error) => {
                // Apply error strategy
                match self.config.error_strategy {
                    ColumnErrorStrategy::StopOnError => Err(error),
                    ColumnErrorStrategy::SkipOnError => {
                        eprintln!(
                            "Warning: Transformer '{}' failed on columns {:?}: {}. Skipping...",
                            step.name, column_indices, error
                        );
                        // Return empty result to indicate skipping
                        Ok(Array2::zeros((subset.nrows(), 0)))
                    }
                    ColumnErrorStrategy::Fallback => {
                        if let Some(ref fallback) = self.config.fallback_transformer {
                            eprintln!("Warning: Transformer '{}' failed on columns {:?}: {}. Using fallback...", 
                                    step.name, column_indices, error);
                            if is_fit_transform {
                                fallback.fit_transform_wrapper(subset)
                            } else {
                                fallback.transform_wrapper(subset)
                            }
                        } else {
                            eprintln!("Warning: Transformer '{}' failed on columns {:?}: {}. No fallback available, passing through...", 
                                    step.name, column_indices, error);
                            Ok(subset.clone())
                        }
                    }
                    ColumnErrorStrategy::ReplaceWithZeros => {
                        eprintln!("Warning: Transformer '{}' failed on columns {:?}: {}. Replacing with zeros...", 
                                step.name, column_indices, error);
                        Ok(Array2::zeros(subset.dim()))
                    }
                    ColumnErrorStrategy::ReplaceWithNaN => {
                        eprintln!("Warning: Transformer '{}' failed on columns {:?}: {}. Replacing with NaN...", 
                                step.name, column_indices, error);
                        Ok(Array2::from_elem(subset.dim(), Float::NAN))
                    }
                }
            }
        };

        ColumnTransformResult {
            transformer_name: step.name.clone(),
            column_indices: column_indices.clone(),
            result: final_result,
            original_indices: column_indices,
        }
    }
}

impl ColumnTransformer<Untrained> {
    /// Create a new ColumnTransformer
    pub fn new() -> Self {
        Self {
            config: ColumnTransformerConfig::default(),
            transformers: Vec::new(),
            state: PhantomData,
            fitted_transformers_: None,
            feature_names_in_: None,
            n_features_in_: None,
            output_indices_: None,
            remainder_indices_: None,
        }
    }

    /// Add a transformer for specific columns
    pub fn add_transformer<T>(mut self, name: &str, transformer: T, columns: ColumnSelector) -> Self
    where
        T: TransformerWrapper + 'static,
    {
        self.transformers.push(TransformerStep {
            name: name.to_string(),
            columns,
            transformer: Box::new(transformer),
        });
        self
    }

    /// Set the remainder strategy
    pub fn remainder(mut self, strategy: RemainderStrategy) -> Self {
        self.config.remainder = strategy;
        self
    }

    /// Set whether to preserve column order
    pub fn preserve_order(mut self, preserve: bool) -> Self {
        self.config.preserve_order = preserve;
        self
    }

    /// Set number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Set input validation
    pub fn validate_input(mut self, validate: bool) -> Self {
        self.config.validate_input = validate;
        self
    }

    /// Set error handling strategy
    pub fn error_strategy(mut self, strategy: ColumnErrorStrategy) -> Self {
        self.config.error_strategy = strategy;
        self
    }

    /// Enable/disable parallel execution
    pub fn parallel_execution(mut self, parallel: bool) -> Self {
        self.config.parallel_execution = parallel;
        self
    }

    /// Set fallback transformer for error handling
    pub fn fallback_transformer<T>(mut self, transformer: T) -> Self
    where
        T: TransformerWrapper + 'static,
    {
        self.config.fallback_transformer = Some(Box::new(transformer));
        self
    }

    /// Resolve column indices from selectors
    fn resolve_columns(&self, selector: &ColumnSelector, n_features: usize) -> Result<Vec<usize>> {
        match selector {
            ColumnSelector::Indices(indices) => {
                // Validate indices
                for &idx in indices {
                    if idx >= n_features {
                        return Err(SklearsError::InvalidInput(format!(
                            "Column index {} is out of bounds for {} features",
                            idx, n_features
                        )));
                    }
                }
                Ok(indices.clone())
            }
            ColumnSelector::Names(_names) => {
                // For now, return error as we need named column support
                Err(SklearsError::NotImplemented(
                    "Named column selection not yet implemented".to_string(),
                ))
            }
            ColumnSelector::DataType(_dtype) => {
                // DataType selection requires training data, handled in resolve_columns_with_data
                Err(SklearsError::InvalidInput(
                    "DataType column selection requires training data. Use resolve_columns_with_data.".to_string(),
                ))
            }
            ColumnSelector::Remainder => {
                // This should be handled separately
                Ok(Vec::new())
            }
        }
    }

    /// Resolve column indices from selectors with access to training data
    fn resolve_columns_with_data(
        &self,
        selector: &ColumnSelector,
        data: &Array2<Float>,
    ) -> Result<Vec<usize>> {
        let (_, n_features) = data.dim();

        match selector {
            ColumnSelector::Indices(indices) => {
                // Validate indices
                for &idx in indices {
                    if idx >= n_features {
                        return Err(SklearsError::InvalidInput(format!(
                            "Column index {} is out of bounds for {} features",
                            idx, n_features
                        )));
                    }
                }
                Ok(indices.clone())
            }
            ColumnSelector::Names(_names) => {
                // For now, return error as we need named column support
                Err(SklearsError::NotImplemented(
                    "Named column selection not yet implemented".to_string(),
                ))
            }
            ColumnSelector::DataType(dtype) => self.infer_columns_by_dtype_with_data(dtype, data),
            ColumnSelector::Remainder => {
                // This should be handled separately
                Ok(Vec::new())
            }
        }
    }

    /// Infer column indices by data type using heuristics (without training data)
    fn infer_columns_by_dtype(&self, _dtype: &DataType, _n_features: usize) -> Result<Vec<usize>> {
        // This method cannot work without training data
        Err(SklearsError::InvalidInput(
            "Data type column selection requires training data context. \
             Use resolve_columns_with_data during fit."
                .to_string(),
        ))
    }

    /// Infer column indices by data type using heuristics on training data
    fn infer_columns_by_dtype_with_data(
        &self,
        dtype: &DataType,
        data: &Array2<Float>,
    ) -> Result<Vec<usize>> {
        let (_n_samples, n_features) = data.dim();
        let mut matching_columns = Vec::new();

        for col_idx in 0..n_features {
            let column = data.column(col_idx);
            let column_type = self.infer_column_type(&column);

            if column_type == *dtype {
                matching_columns.push(col_idx);
            }
        }

        Ok(matching_columns)
    }

    /// Infer the data type of a single column using heuristics
    fn infer_column_type(&self, column: &scirs2_core::ndarray::ArrayView1<Float>) -> DataType {
        let unique_values: std::collections::HashSet<_> =
            column.iter().map(|&x| OrderedFloat::from(x)).collect();

        let n_unique = unique_values.len();
        let n_total = column.len();

        // Check if column is boolean (only 0.0 and 1.0 values)
        if n_unique <= 2 {
            let zero_bits = OrderedFloat::from(0.0);
            let one_bits = OrderedFloat::from(1.0);
            if unique_values
                .iter()
                .all(|&x| x == zero_bits || x == one_bits)
            {
                return DataType::Boolean;
            }
        }

        // Heuristic for categorical vs numeric
        // If the ratio of unique values to total values is low, consider it categorical
        let unique_ratio = n_unique as f64 / n_total as f64;

        // Use a more balanced approach: categorical if either condition is met
        // but with more conservative thresholds
        if (unique_ratio < 0.6 && n_unique <= 5) || unique_ratio < 0.2 {
            DataType::Categorical
        } else {
            DataType::Numeric
        }
    }

    /// Get indices of columns that are not selected by any transformer
    fn get_remainder_indices(&self, data: &Array2<Float>) -> Result<Vec<usize>> {
        let (_, n_features) = data.dim();
        let mut used_indices = std::collections::HashSet::new();

        // Collect all used indices
        for step in &self.transformers {
            let indices = match &step.columns {
                ColumnSelector::DataType(_) => {
                    self.resolve_columns_with_data(&step.columns, data)?
                }
                _ => self.resolve_columns(&step.columns, n_features)?,
            };
            for idx in indices {
                used_indices.insert(idx);
            }
        }

        // Return unused indices
        Ok((0..n_features)
            .filter(|i| !used_indices.contains(i))
            .collect())
    }
}

impl Default for ColumnTransformer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ColumnTransformer<Untrained> {
    type Config = ColumnTransformerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for ColumnTransformer<Trained> {
    type Config = ColumnTransformerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for ColumnTransformer<Untrained> {
    type Fitted = ColumnTransformer<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit transformer on empty dataset".to_string(),
            ));
        }

        // Get remainder indices
        let remainder_indices = self.get_remainder_indices(x)?;

        // Prepare transformer steps with resolved indices
        let mut transformer_tasks: Vec<(TransformerStep, Vec<usize>)> = Vec::new();

        for step in &self.transformers {
            // Resolve column indices - use data-aware method for DataType selectors
            let indices = match &step.columns {
                ColumnSelector::DataType(_) => self.resolve_columns_with_data(&step.columns, x)?,
                _ => self.resolve_columns(&step.columns, n_features)?,
            };

            if !indices.is_empty() {
                transformer_tasks.push((step.clone(), indices));
            }
        }

        // Apply transformers with parallel processing and error handling
        let transform_results: Vec<ColumnTransformResult> = if self.config.parallel_execution
            && transformer_tasks.len() > 1
        {
            #[cfg(feature = "parallel")]
            {
                transformer_tasks
                    .into_par_iter()
                    .map(|(step, indices)| {
                        let subset = x.select(Axis(1), &indices);
                        self.apply_transformer_with_error_handling(
                            &step, x, &subset, true, &indices,
                        )
                    })
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                // Fallback to sequential processing
                transformer_tasks
                    .into_iter()
                    .map(|(step, indices)| {
                        let subset = x.select(Axis(1), &indices);
                        self.apply_transformer_with_error_handling(
                            &step, x, &subset, true, &indices,
                        )
                    })
                    .collect()
            }
        } else {
            // Sequential processing
            transformer_tasks
                .into_iter()
                .map(|(step, indices)| {
                    let subset = x.select(Axis(1), &indices);
                    self.apply_transformer_with_error_handling(&step, x, &subset, true, &indices)
                })
                .collect()
        };

        // Process results and create fitted transformers
        let mut fitted_transformers = Vec::new();
        let mut output_indices = HashMap::new();

        for transform_result in transform_results {
            match transform_result.result {
                Ok(transformed) => {
                    if transformed.ncols() > 0 {
                        // Skip empty results (from SkipOnError)
                        // Store output indices mapping
                        let output_cols = (0..transformed.ncols()).collect();
                        output_indices
                            .insert(transform_result.transformer_name.clone(), output_cols);

                        // Create fitted transformer step
                        let transformer_name = transform_result.transformer_name.clone();
                        fitted_transformers.push(TransformerStep {
                            name: transformer_name.clone(),
                            columns: ColumnSelector::Indices(transform_result.original_indices),
                            transformer: self
                                .transformers
                                .iter()
                                .find(|s| s.name == transformer_name)
                                .unwrap()
                                .transformer
                                .clone_box(),
                        });
                    }
                }
                Err(e) => {
                    // If we reach here, it means StopOnError was used
                    return Err(SklearsError::TransformError(format!(
                        "Transformer '{}' failed: {}",
                        transform_result.transformer_name, e
                    )));
                }
            }
        }

        Ok(ColumnTransformer {
            config: self.config,
            transformers: self.transformers,
            state: PhantomData,
            fitted_transformers_: Some(fitted_transformers),
            feature_names_in_: None,
            n_features_in_: Some(n_features),
            output_indices_: Some(output_indices),
            remainder_indices_: Some(remainder_indices),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for ColumnTransformer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if Some(n_features) != self.n_features_in_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_.unwrap_or(0),
                actual: n_features,
            });
        }

        let fitted_transformers = self.fitted_transformers_.as_ref().unwrap();
        let remainder_indices = self.remainder_indices_.as_ref().unwrap();

        // Prepare transformer tasks for parallel processing
        let transformer_tasks: Vec<&TransformerStep> = fitted_transformers.iter().collect();

        // Apply transformers with parallel processing and error handling
        let transform_results: Vec<ColumnTransformResult> =
            if self.config.parallel_execution && transformer_tasks.len() > 1 {
                #[cfg(feature = "parallel")]
                {
                    transformer_tasks
                        .into_par_iter()
                        .filter_map(|step| {
                            if let ColumnSelector::Indices(indices) = &step.columns {
                                if !indices.is_empty() {
                                    let subset = x.select(Axis(1), indices);
                                    Some(self.apply_transformer_with_error_handling(
                                        step, x, &subset, false, indices,
                                    ))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                }
                #[cfg(not(feature = "parallel"))]
                {
                    // Fallback to sequential processing
                    transformer_tasks
                        .into_iter()
                        .filter_map(|step| {
                            if let ColumnSelector::Indices(indices) = &step.columns {
                                if !indices.is_empty() {
                                    let subset = x.select(Axis(1), indices);
                                    Some(self.apply_transformer_with_error_handling(
                                        step, x, &subset, false, indices,
                                    ))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                }
            } else {
                // Sequential processing
                transformer_tasks
                    .into_iter()
                    .filter_map(|step| {
                        if let ColumnSelector::Indices(indices) = &step.columns {
                            if !indices.is_empty() {
                                let subset = x.select(Axis(1), indices);
                                Some(self.apply_transformer_with_error_handling(
                                    step, x, &subset, false, indices,
                                ))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect()
            };

        // Process results and create column outputs
        let mut column_outputs: Vec<(usize, Array2<Float>)> = Vec::new();

        for transform_result in transform_results {
            match transform_result.result {
                Ok(transformed) => {
                    if transformed.ncols() > 0 {
                        // Skip empty results (from SkipOnError)
                        // For each original column index, store its min value to maintain order
                        let min_index = *transform_result.original_indices.iter().min().unwrap();
                        column_outputs.push((min_index, transformed));
                    }
                }
                Err(e) => {
                    // If we reach here, it means StopOnError was used
                    return Err(SklearsError::TransformError(format!(
                        "Transformer '{}' failed: {}",
                        transform_result.transformer_name, e
                    )));
                }
            }
        }

        // Handle remainder columns
        if !remainder_indices.is_empty() {
            let remainder_data = x.select(Axis(1), remainder_indices);

            let transformed_remainder = match &self.config.remainder {
                RemainderStrategy::Drop => {
                    None // remainder is dropped
                }
                RemainderStrategy::Passthrough => Some(remainder_data),
                RemainderStrategy::Transform(transformer) => {
                    let transformed = transformer.transform_wrapper(&remainder_data)?;
                    Some(transformed)
                }
            };

            if let Some(remainder_output) = transformed_remainder {
                // Add remainder with the minimum remainder index
                if let Some(&min_remainder_index) = remainder_indices.iter().min() {
                    column_outputs.push((min_remainder_index, remainder_output));
                }
            }
        }

        // Sort by original column indices to maintain proper ordering
        column_outputs.sort_by_key(|(idx, _)| *idx);

        // Concatenate all output parts in the correct order
        if column_outputs.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No output from any transformer".to_string(),
            ));
        }

        // Calculate total columns
        let total_cols: usize = column_outputs.iter().map(|(_, arr)| arr.ncols()).sum();
        let mut result = Array2::zeros((n_samples, total_cols));

        // Concatenate in order
        let mut col_offset = 0;
        for (_, part) in column_outputs {
            let part_cols = part.ncols();
            result
                .slice_mut(s![.., col_offset..col_offset + part_cols])
                .assign(&part);
            col_offset += part_cols;
        }

        Ok(result)
    }
}

impl ColumnTransformer<Trained> {
    /// Get the number of features seen during fitting
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the output indices mapping
    pub fn output_indices(&self) -> &HashMap<String, Vec<usize>> {
        self.output_indices_.as_ref().unwrap()
    }

    /// Get the remainder indices
    pub fn remainder_indices(&self) -> &Vec<usize> {
        self.remainder_indices_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Mock transformer for testing
    #[derive(Debug, Clone)]
    struct MockTransformer {
        scale: Float,
    }

    impl TransformerWrapper for MockTransformer {
        fn fit_transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            Ok(x * self.scale)
        }

        fn transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            Ok(x * self.scale)
        }

        fn get_n_features_out(&self) -> Option<usize> {
            None // Same as input
        }

        fn clone_box(&self) -> Box<dyn TransformerWrapper> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_column_transformer_basic() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "scale_first_two",
                MockTransformer { scale: 2.0 },
                ColumnSelector::Indices(vec![0, 1]),
            )
            .remainder(RemainderStrategy::Passthrough);

        let fitted_ct = ct.fit(&x, &()).unwrap();
        let result = fitted_ct.transform(&x).unwrap();

        // First two columns should be scaled by 2, last column passed through
        assert_eq!(result.dim(), (3, 3));
        assert_eq!(result[[0, 0]], 2.0); // 1.0 * 2.0
        assert_eq!(result[[0, 1]], 4.0); // 2.0 * 2.0
        assert_eq!(result[[0, 2]], 3.0); // 3.0 (passthrough)
    }

    #[test]
    fn test_column_transformer_drop_remainder() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "scale_middle",
                MockTransformer { scale: 3.0 },
                ColumnSelector::Indices(vec![1, 2]),
            )
            .remainder(RemainderStrategy::Drop);

        let fitted_ct = ct.fit(&x, &()).unwrap();
        let result = fitted_ct.transform(&x).unwrap();

        // Only middle two columns should remain (scaled by 3)
        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 6.0); // 2.0 * 3.0
        assert_eq!(result[[0, 1]], 9.0); // 3.0 * 3.0
    }

    #[test]
    fn test_column_transformer_multiple_transformers() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "scale_first",
                MockTransformer { scale: 2.0 },
                ColumnSelector::Indices(vec![0]),
            )
            .add_transformer(
                "scale_last",
                MockTransformer { scale: 0.5 },
                ColumnSelector::Indices(vec![3]),
            )
            .remainder(RemainderStrategy::Passthrough);

        let fitted_ct = ct.fit(&x, &()).unwrap();
        let result = fitted_ct.transform(&x).unwrap();

        // Should have 4 columns: [scaled_first, middle_two_passthrough, scaled_last]
        assert_eq!(result.dim(), (2, 4));
        assert_eq!(result[[0, 0]], 2.0); // 1.0 * 2.0 (first transformer)
        assert_eq!(result[[0, 1]], 2.0); // 2.0 (passthrough)
        assert_eq!(result[[0, 2]], 3.0); // 3.0 (passthrough)
        assert_eq!(result[[0, 3]], 2.0); // 4.0 * 0.5 (second transformer)
    }

    #[test]
    fn test_column_transformer_empty_data() {
        let x_empty: Array2<Float> = Array2::zeros((0, 3));

        let ct = ColumnTransformer::new().add_transformer(
            "test",
            MockTransformer { scale: 1.0 },
            ColumnSelector::Indices(vec![0]),
        );

        let result = ct.fit(&x_empty, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_column_transformer_invalid_indices() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let ct = ColumnTransformer::new().add_transformer(
            "invalid",
            MockTransformer { scale: 1.0 },
            ColumnSelector::Indices(vec![0, 5]), // Index 5 doesn't exist
        );

        let result = ct.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_column_type_inference() {
        let ct = ColumnTransformer::new();

        // Test boolean detection (strict 0.0 and 1.0 only)
        let bool_col = scirs2_core::ndarray::array![0.0, 1.0, 0.0, 1.0, 0.0];
        let bool_type = ct.infer_column_type(&bool_col.view());
        assert_eq!(bool_type, DataType::Boolean);

        // Test categorical detection (few unique values)
        let cat_col = scirs2_core::ndarray::array![1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let cat_type = ct.infer_column_type(&cat_col.view());
        assert_eq!(cat_type, DataType::Categorical);

        // Test numeric detection (many unique values)
        let num_col =
            scirs2_core::ndarray::array![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0];
        let num_type = ct.infer_column_type(&num_col.view());
        assert_eq!(num_type, DataType::Numeric);
    }

    // Failing transformer for testing error handling
    #[derive(Debug, Clone)]
    struct FailingTransformer {
        should_fail: bool,
    }

    impl TransformerWrapper for FailingTransformer {
        fn fit_transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            if self.should_fail {
                Err(SklearsError::InvalidInput(
                    "Intentional failure for testing".to_string(),
                ))
            } else {
                Ok(x * 2.0)
            }
        }

        fn transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            if self.should_fail {
                Err(SklearsError::InvalidInput(
                    "Intentional failure for testing".to_string(),
                ))
            } else {
                Ok(x * 2.0)
            }
        }

        fn get_n_features_out(&self) -> Option<usize> {
            None
        }

        fn clone_box(&self) -> Box<dyn TransformerWrapper> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_column_transformer_error_handling_stop_on_error() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "failing",
                FailingTransformer { should_fail: true },
                ColumnSelector::Indices(vec![0]),
            )
            .error_strategy(ColumnErrorStrategy::StopOnError);

        let result = ct.fit(&x, &());
        assert!(result.is_err(), "Should fail with StopOnError");
    }

    #[test]
    fn test_column_transformer_error_handling_skip_on_error() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "failing",
                FailingTransformer { should_fail: true },
                ColumnSelector::Indices(vec![0]),
            )
            .add_transformer(
                "working",
                MockTransformer { scale: 2.0 },
                ColumnSelector::Indices(vec![1]),
            )
            .error_strategy(ColumnErrorStrategy::SkipOnError)
            .remainder(RemainderStrategy::Passthrough);

        let fitted_ct = ct.fit(&x, &()).unwrap();
        let result = fitted_ct.transform(&x).unwrap();

        // Should have 2 columns: working transformer output + remainder
        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 4.0); // 2.0 * 2.0 (working transformer)
        assert_eq!(result[[0, 1]], 3.0); // 3.0 (remainder passthrough)
    }

    #[test]
    fn test_column_transformer_error_handling_replace_with_zeros() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "failing",
                FailingTransformer { should_fail: true },
                ColumnSelector::Indices(vec![0]),
            )
            .error_strategy(ColumnErrorStrategy::ReplaceWithZeros)
            .remainder(RemainderStrategy::Passthrough);

        let fitted_ct = ct.fit(&x, &()).unwrap();
        let result = fitted_ct.transform(&x).unwrap();

        // Should have 3 columns: zeros (replacement) + remainder passthrough
        assert_eq!(result.dim(), (2, 3));
        assert_eq!(result[[0, 0]], 0.0); // Replaced with zero
        assert_eq!(result[[0, 1]], 2.0); // Remainder passthrough
        assert_eq!(result[[0, 2]], 3.0); // Remainder passthrough
    }

    #[test]
    fn test_column_transformer_error_handling_fallback() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "failing",
                FailingTransformer { should_fail: true },
                ColumnSelector::Indices(vec![0]),
            )
            .error_strategy(ColumnErrorStrategy::Fallback)
            .fallback_transformer(MockTransformer { scale: 0.5 })
            .remainder(RemainderStrategy::Passthrough);

        let fitted_ct = ct.fit(&x, &()).unwrap();
        let result = fitted_ct.transform(&x).unwrap();

        // Should have 3 columns: fallback transformer output + remainder
        assert_eq!(result.dim(), (2, 3));
        assert_eq!(result[[0, 0]], 0.5); // 1.0 * 0.5 (fallback)
        assert_eq!(result[[0, 1]], 2.0); // Remainder passthrough
        assert_eq!(result[[0, 2]], 3.0); // Remainder passthrough
    }

    #[test]
    fn test_column_transformer_parallel_execution() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "scale_first",
                MockTransformer { scale: 2.0 },
                ColumnSelector::Indices(vec![0]),
            )
            .add_transformer(
                "scale_second",
                MockTransformer { scale: 3.0 },
                ColumnSelector::Indices(vec![1]),
            )
            .parallel_execution(true)
            .remainder(RemainderStrategy::Passthrough);

        let fitted_ct = ct.fit(&x, &()).unwrap();
        let result = fitted_ct.transform(&x).unwrap();

        // Should have 4 columns: 2 transformed + 2 remainder
        assert_eq!(result.dim(), (2, 4));
        assert_eq!(result[[0, 0]], 2.0); // 1.0 * 2.0
        assert_eq!(result[[0, 1]], 6.0); // 2.0 * 3.0
        assert_eq!(result[[0, 2]], 3.0); // Remainder
        assert_eq!(result[[0, 3]], 4.0); // Remainder
    }

    #[test]
    fn test_column_transformer_dtype_selection() {
        // Create data with very clear column types:
        // Col 0: Numeric (many unique continuous values)
        // Col 1: Boolean (strict 0.0/1.0 only)
        // Col 2: Categorical (few repeated values)
        let x = array![
            [1.23456, 0.0, 1.0],
            [2.78901, 1.0, 1.0],
            [3.45678, 0.0, 2.0],
            [4.98765, 1.0, 1.0],
            [5.12345, 0.0, 2.0],
            [6.67890, 1.0, 3.0],
            [7.11111, 0.0, 1.0],
            [8.22222, 1.0, 2.0],
        ];

        // Test Boolean column selection
        let ct_bool = ColumnTransformer::new().add_transformer(
            "bool_transformer",
            MockTransformer { scale: 10.0 },
            ColumnSelector::DataType(DataType::Boolean),
        );

        let fitted_ct_bool = ct_bool.fit(&x, &()).unwrap();
        let result_bool = fitted_ct_bool.transform(&x).unwrap();

        // Should have 1 column (boolean column scaled by 10)
        assert_eq!(result_bool.dim(), (8, 1));
        assert_eq!(result_bool[[0, 0]], 0.0); // 0.0 * 10.0
        assert_eq!(result_bool[[1, 0]], 10.0); // 1.0 * 10.0

        // Test Categorical column selection
        let ct_cat = ColumnTransformer::new().add_transformer(
            "cat_transformer",
            MockTransformer { scale: 0.1 },
            ColumnSelector::DataType(DataType::Categorical),
        );

        let fitted_ct_cat = ct_cat.fit(&x, &()).unwrap();
        let result_cat = fitted_ct_cat.transform(&x).unwrap();

        // Should have 1 column (categorical column scaled by 0.1)
        assert_eq!(result_cat.dim(), (8, 1));
        assert_eq!(result_cat[[0, 0]], 0.1); // 1.0 * 0.1

        // Test Numeric column selection
        let ct_num = ColumnTransformer::new().add_transformer(
            "num_transformer",
            MockTransformer { scale: 2.0 },
            ColumnSelector::DataType(DataType::Numeric),
        );

        let fitted_ct_num = ct_num.fit(&x, &()).unwrap();
        let result_num = fitted_ct_num.transform(&x).unwrap();

        // Should have 1 column (numeric column scaled by 2.0)
        assert_eq!(result_num.dim(), (8, 1));
        let expected_first = 1.23456 * 2.0;
        assert!((result_num[[0, 0]] - expected_first).abs() < 1e-10)
    }
}
