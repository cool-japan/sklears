//! Feature Union
//!
//! This module provides FeatureUnion which combines the output of multiple
//! transformers by applying them all to the same input data and concatenating results.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Re-export the TransformerWrapper trait from column_transformer
pub use crate::column_transformer::TransformerWrapper;

/// Feature selection strategy for FeatureUnion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureSelectionStrategy {
    /// No feature selection (keep all features)
    None,
    /// Select top k features based on variance
    VarianceThreshold(Float),
    /// Select top k features by count
    TopK(usize),
    /// Select features with importance above threshold
    ImportanceThreshold(Float),
    /// Select top percentage of features
    TopPercentile(Float),
}

impl Default for FeatureSelectionStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// Feature importance calculation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureImportanceMethod {
    /// Use variance as importance measure
    Variance,
    /// Use absolute mean as importance measure
    AbsoluteMean,
    /// Use L1 norm as importance measure
    L1Norm,
    /// Use L2 norm as importance measure
    L2Norm,
    /// Use correlation with first principal component
    PrincipalComponent,
}

impl Default for FeatureImportanceMethod {
    fn default() -> Self {
        Self::Variance
    }
}

/// A transformer step in the feature union
#[derive(Debug, Clone)]
pub struct FeatureUnionStep {
    /// Name of the transformer step
    pub name: String,
    /// The transformer (boxed for dynamic dispatch)
    pub transformer: Box<dyn TransformerWrapper>,
    /// Weight for this transformer's output (optional)
    pub weight: Option<Float>,
}

/// Configuration for FeatureUnion
#[derive(Debug, Clone)]
pub struct FeatureUnionConfig {
    /// Whether to use parallel processing
    pub n_jobs: Option<usize>,
    /// Whether to validate input
    pub validate_input: bool,
    /// Whether to preserve transformer order in output
    pub preserve_order: bool,
    /// Feature selection strategy
    pub feature_selection: FeatureSelectionStrategy,
    /// Feature importance calculation method
    pub importance_method: FeatureImportanceMethod,
    /// Whether to enable feature selection
    pub enable_feature_selection: bool,
}

impl Default for FeatureUnionConfig {
    fn default() -> Self {
        Self {
            n_jobs: None,
            validate_input: true,
            preserve_order: true,
            feature_selection: FeatureSelectionStrategy::None,
            importance_method: FeatureImportanceMethod::Variance,
            enable_feature_selection: false,
        }
    }
}

/// FeatureUnion concatenates the results of multiple transformers
///
/// Unlike ColumnTransformer which applies different transformers to different columns,
/// FeatureUnion applies all transformers to the same input data and concatenates
/// their outputs column-wise.
///
/// This is useful for creating feature combinations, such as applying both
/// PCA and polynomial features to the same input.
#[derive(Debug)]
pub struct FeatureUnion<State = Untrained> {
    config: FeatureUnionConfig,
    transformers: Vec<FeatureUnionStep>,
    state: PhantomData<State>,
    // Fitted parameters
    fitted_transformers_: Option<Vec<FeatureUnionStep>>,
    n_features_in_: Option<usize>,
    n_features_out_: Option<usize>,
    transformer_weights_: Option<Vec<Float>>,
    // Feature selection parameters
    selected_features_: Option<Vec<usize>>,
    feature_importances_: Option<Array1<Float>>,
    feature_names_: Option<Vec<String>>,
}

impl FeatureUnion<Untrained> {
    /// Create a new FeatureUnion
    pub fn new() -> Self {
        Self {
            config: FeatureUnionConfig::default(),
            transformers: Vec::new(),
            state: PhantomData,
            fitted_transformers_: None,
            n_features_in_: None,
            n_features_out_: None,
            transformer_weights_: None,
            selected_features_: None,
            feature_importances_: None,
            feature_names_: None,
        }
    }

    /// Add a transformer to the union
    pub fn add_transformer<T>(mut self, name: &str, transformer: T) -> Self
    where
        T: TransformerWrapper + 'static,
    {
        self.transformers.push(FeatureUnionStep {
            name: name.to_string(),
            transformer: Box::new(transformer),
            weight: None,
        });
        self
    }

    /// Add a weighted transformer to the union
    pub fn add_weighted_transformer<T>(mut self, name: &str, transformer: T, weight: Float) -> Self
    where
        T: TransformerWrapper + 'static,
    {
        self.transformers.push(FeatureUnionStep {
            name: name.to_string(),
            transformer: Box::new(transformer),
            weight: Some(weight),
        });
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

    /// Set whether to preserve transformer order
    pub fn preserve_order(mut self, preserve: bool) -> Self {
        self.config.preserve_order = preserve;
        self
    }

    /// Set feature selection strategy
    pub fn feature_selection(mut self, strategy: FeatureSelectionStrategy) -> Self {
        self.config.feature_selection = strategy;
        self.config.enable_feature_selection = !matches!(strategy, FeatureSelectionStrategy::None);
        self
    }

    /// Set feature importance calculation method
    pub fn importance_method(mut self, method: FeatureImportanceMethod) -> Self {
        self.config.importance_method = method;
        self
    }

    /// Enable or disable feature selection
    pub fn enable_feature_selection(mut self, enable: bool) -> Self {
        self.config.enable_feature_selection = enable;
        self
    }

    /// Calculate feature importance scores for the transformed data
    fn calculate_feature_importance(
        &self,
        data: &Array2<Float>,
        method: FeatureImportanceMethod,
    ) -> Array1<Float> {
        let n_features = data.ncols();
        let mut importances = Array1::zeros(n_features);

        match method {
            FeatureImportanceMethod::Variance => {
                for (i, col) in data.columns().into_iter().enumerate() {
                    let mean = col.mean().unwrap_or(0.0);
                    let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<Float>()
                        / (col.len() as Float);
                    importances[i] = variance;
                }
            }
            FeatureImportanceMethod::AbsoluteMean => {
                for (i, col) in data.columns().into_iter().enumerate() {
                    let abs_mean =
                        col.iter().map(|&x| x.abs()).sum::<Float>() / (col.len() as Float);
                    importances[i] = abs_mean;
                }
            }
            FeatureImportanceMethod::L1Norm => {
                for (i, col) in data.columns().into_iter().enumerate() {
                    let l1_norm = col.iter().map(|&x| x.abs()).sum::<Float>();
                    importances[i] = l1_norm;
                }
            }
            FeatureImportanceMethod::L2Norm => {
                for (i, col) in data.columns().into_iter().enumerate() {
                    let l2_norm = col.iter().map(|&x| x * x).sum::<Float>().sqrt();
                    importances[i] = l2_norm;
                }
            }
            FeatureImportanceMethod::PrincipalComponent => {
                // Simplified correlation with the first principal component (mean-centered sum)
                let means: Vec<Float> = (0..n_features)
                    .map(|i| data.column(i).mean().unwrap_or(0.0))
                    .collect();

                // Calculate first principal component as weighted sum of all features
                let mut pc1: Array1<Float> = Array1::zeros(data.nrows());
                for (i, col) in data.columns().into_iter().enumerate() {
                    let mean = means[i];
                    for (row_idx, &val) in col.iter().enumerate() {
                        pc1[row_idx] += (val - mean) / (n_features as Float).sqrt();
                    }
                }

                // Calculate correlation of each feature with PC1
                for (i, col) in data.columns().into_iter().enumerate() {
                    let mean = means[i];
                    let centered_col: Vec<Float> = col.iter().map(|&x| x - mean).collect();
                    let correlation: Float = centered_col
                        .iter()
                        .zip(pc1.iter())
                        .map(|(&x, &y): (&Float, &Float)| x * y)
                        .sum::<Float>()
                        / ((data.nrows() - 1) as Float);
                    importances[i] = correlation.abs();
                }
            }
        }

        importances
    }

    /// Select features based on the configured strategy
    fn select_features(
        &self,
        importances: &Array1<Float>,
        strategy: FeatureSelectionStrategy,
    ) -> Vec<usize> {
        let n_features = importances.len();
        let mut feature_indices: Vec<(usize, Float)> = importances
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        match strategy {
            FeatureSelectionStrategy::None => (0..n_features).collect(),
            FeatureSelectionStrategy::VarianceThreshold(threshold) => feature_indices
                .into_iter()
                .filter_map(|(idx, score)| if score >= threshold { Some(idx) } else { None })
                .collect(),
            FeatureSelectionStrategy::TopK(k) => {
                feature_indices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                feature_indices
                    .into_iter()
                    .take(k.min(n_features))
                    .map(|(idx, _)| idx)
                    .collect()
            }
            FeatureSelectionStrategy::ImportanceThreshold(threshold) => feature_indices
                .into_iter()
                .filter_map(|(idx, score)| if score >= threshold { Some(idx) } else { None })
                .collect(),
            FeatureSelectionStrategy::TopPercentile(percentile) => {
                if percentile <= 0.0 || percentile > 100.0 {
                    return (0..n_features).collect();
                }
                feature_indices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let k = ((n_features as Float * percentile / 100.0).ceil() as usize).max(1);
                feature_indices
                    .into_iter()
                    .take(k)
                    .map(|(idx, _)| idx)
                    .collect()
            }
        }
    }
}

impl Default for FeatureUnion<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FeatureUnion<Untrained> {
    type Config = FeatureUnionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for FeatureUnion<Trained> {
    type Config = FeatureUnionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for FeatureUnion<Untrained> {
    type Fitted = FeatureUnion<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit FeatureUnion on empty dataset".to_string(),
            ));
        }

        if self.transformers.is_empty() {
            return Err(SklearsError::InvalidInput(
                "FeatureUnion requires at least one transformer".to_string(),
            ));
        }

        // Fit each transformer and calculate output dimensions
        let mut fitted_transformers = Vec::new();
        let mut transformer_weights = Vec::new();
        let mut all_transformed_data = Vec::new();

        for step in &self.transformers {
            // Fit and transform to get the fitted transformer and output shape
            let transformed = step.transformer.fit_transform_wrapper(x)?;

            // Validate that all transformers return the same number of samples
            if transformed.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Transformer '{}' returned {} samples, expected {}",
                    step.name,
                    transformed.nrows(),
                    n_samples
                )));
            }

            // Store the weight (default to 1.0 if not specified)
            transformer_weights.push(step.weight.unwrap_or(1.0));

            // Apply weight if specified
            let mut weighted_transformed = transformed;
            let weight = step.weight.unwrap_or(1.0);
            if (weight - 1.0).abs() > Float::EPSILON {
                weighted_transformed *= weight;
            }

            all_transformed_data.push(weighted_transformed);

            // Create fitted transformer step (clone for now)
            fitted_transformers.push(FeatureUnionStep {
                name: step.name.clone(),
                transformer: step.transformer.clone_box(),
                weight: step.weight,
            });
        }

        // Concatenate all transformed data for feature selection
        let concatenated_data = concatenate_features(all_transformed_data)?;

        // Perform feature selection if enabled
        let (selected_features, feature_importances, total_output_features) = if self
            .config
            .enable_feature_selection
        {
            let importances = self
                .calculate_feature_importance(&concatenated_data, self.config.importance_method);
            let selected = self.select_features(&importances, self.config.feature_selection);
            let n_selected = selected.len();
            (Some(selected), Some(importances), n_selected)
        } else {
            (None, None, concatenated_data.ncols())
        };

        Ok(FeatureUnion {
            config: self.config,
            transformers: self.transformers,
            state: PhantomData,
            fitted_transformers_: Some(fitted_transformers),
            n_features_in_: Some(n_features),
            n_features_out_: Some(total_output_features),
            transformer_weights_: Some(transformer_weights),
            selected_features_: selected_features,
            feature_importances_: feature_importances,
            feature_names_: None, // Could be set if feature names are provided
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for FeatureUnion<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if Some(n_features) != self.n_features_in_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_.unwrap_or(0),
                actual: n_features,
            });
        }

        let fitted_transformers = self.fitted_transformers_.as_ref().unwrap();
        let transformer_weights = self.transformer_weights_.as_ref().unwrap();

        if fitted_transformers.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No fitted transformers available".to_string(),
            ));
        }

        // Apply each transformer and collect results
        let mut transformed_parts = Vec::new();

        for (i, step) in fitted_transformers.iter().enumerate() {
            // Transform the input
            let mut transformed = step.transformer.transform_wrapper(x)?;

            // Validate output shape
            if transformed.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Transformer '{}' returned {} samples, expected {}",
                    step.name,
                    transformed.nrows(),
                    n_samples
                )));
            }

            // Apply weight if specified
            let weight = transformer_weights[i];
            if (weight - 1.0).abs() > Float::EPSILON {
                transformed *= weight;
            }

            transformed_parts.push(transformed);
        }

        // Concatenate all transformed parts along the feature axis
        let concatenated = concatenate_features(transformed_parts)?;

        // Apply feature selection if enabled
        if let Some(ref selected_features) = self.selected_features_ {
            if selected_features.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "No features were selected during fitting".to_string(),
                ));
            }

            // Select only the chosen features
            let selected_data = concatenated.select(Axis(1), selected_features);
            Ok(selected_data)
        } else {
            Ok(concatenated)
        }
    }
}

/// Helper function to concatenate arrays along the feature (column) axis
fn concatenate_features(parts: Vec<Array2<Float>>) -> Result<Array2<Float>> {
    if parts.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No arrays to concatenate".to_string(),
        ));
    }

    if parts.len() == 1 {
        return Ok(parts.into_iter().next().unwrap());
    }

    // Calculate total columns
    let total_cols: usize = parts.iter().map(|p| p.ncols()).sum();
    let n_rows = parts[0].nrows();

    // Create result array
    let mut result = Array2::zeros((n_rows, total_cols));

    // Copy each part into the result
    let mut col_offset = 0;
    for part in parts {
        let part_cols = part.ncols();
        result
            .slice_mut(scirs2_core::ndarray::s![
                ..,
                col_offset..col_offset + part_cols
            ])
            .assign(&part);
        col_offset += part_cols;
    }

    Ok(result)
}

impl FeatureUnion<Trained> {
    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the number of output features
    pub fn n_features_out(&self) -> usize {
        self.n_features_out_.unwrap()
    }

    /// Get the fitted transformers
    pub fn get_transformers(&self) -> &Vec<FeatureUnionStep> {
        self.fitted_transformers_.as_ref().unwrap()
    }

    /// Get the weights used for each transformer
    pub fn get_weights(&self) -> &Vec<Float> {
        self.transformer_weights_.as_ref().unwrap()
    }

    /// Get the selected feature indices (if feature selection is enabled)
    pub fn get_selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get the feature importance scores (if feature selection is enabled)
    pub fn get_feature_importances(&self) -> Option<&Array1<Float>> {
        self.feature_importances_.as_ref()
    }

    /// Get the number of features that were selected
    pub fn n_features_selected(&self) -> usize {
        self.selected_features_
            .as_ref()
            .map(|features| features.len())
            .unwrap_or_else(|| self.n_features_out())
    }

    /// Check if feature selection is enabled
    pub fn is_feature_selection_enabled(&self) -> bool {
        self.selected_features_.is_some()
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
        output_features: Option<usize>,
    }

    impl TransformerWrapper for MockTransformer {
        fn fit_transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            self.transform_wrapper(x)
        }

        fn transform_wrapper(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
            let result = x * self.scale;

            // If output_features is specified, duplicate or reduce features
            if let Some(out_features) = self.output_features {
                let n_rows = result.nrows();
                let mut output = Array2::zeros((n_rows, out_features));

                for i in 0..out_features {
                    let source_col = i % result.ncols();
                    output.column_mut(i).assign(&result.column(source_col));
                }

                Ok(output)
            } else {
                Ok(result)
            }
        }

        fn get_n_features_out(&self) -> Option<usize> {
            self.output_features
        }

        fn clone_box(&self) -> Box<dyn TransformerWrapper> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_feature_union_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let fu = FeatureUnion::new()
            .add_transformer(
                "scale_by_2",
                MockTransformer {
                    scale: 2.0,
                    output_features: None,
                },
            )
            .add_transformer(
                "scale_by_3",
                MockTransformer {
                    scale: 3.0,
                    output_features: None,
                },
            );

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let result = fitted_fu.transform(&x).unwrap();

        // Should have 4 features: [original*2, original*3]
        assert_eq!(result.dim(), (3, 4));

        // First transformer output (scale by 2)
        assert_eq!(result[[0, 0]], 2.0); // 1.0 * 2
        assert_eq!(result[[0, 1]], 4.0); // 2.0 * 2

        // Second transformer output (scale by 3)
        assert_eq!(result[[0, 2]], 3.0); // 1.0 * 3
        assert_eq!(result[[0, 3]], 6.0); // 2.0 * 3
    }

    #[test]
    fn test_feature_union_weighted() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let fu = FeatureUnion::new().add_weighted_transformer(
            "weighted",
            MockTransformer {
                scale: 1.0,
                output_features: None,
            },
            2.0,
        );

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let result = fitted_fu.transform(&x).unwrap();

        // Features should be scaled by weight (2.0)
        assert_eq!(result[[0, 0]], 2.0); // 1.0 * 1.0 * 2.0
        assert_eq!(result[[0, 1]], 4.0); // 2.0 * 1.0 * 2.0
    }

    #[test]
    fn test_feature_union_different_output_sizes() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let fu = FeatureUnion::new()
            .add_transformer(
                "identity",
                MockTransformer {
                    scale: 1.0,
                    output_features: None,
                },
            ) // 2 features out
            .add_transformer(
                "expand",
                MockTransformer {
                    scale: 1.0,
                    output_features: Some(3),
                },
            ); // 3 features out

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let result = fitted_fu.transform(&x).unwrap();

        // Should have 5 features total (2 + 3)
        assert_eq!(result.dim(), (2, 5));
        assert_eq!(fitted_fu.n_features_out(), 5);
    }

    #[test]
    fn test_feature_union_empty_transformers() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let fu = FeatureUnion::new();

        let result = fu.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_union_empty_data() {
        let x_empty: Array2<Float> = Array2::zeros((0, 2));

        let fu = FeatureUnion::new().add_transformer(
            "test",
            MockTransformer {
                scale: 1.0,
                output_features: None,
            },
        );

        let result = fu.fit(&x_empty, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_union_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0],];

        let x_test = array![
            [1.0, 2.0, 3.0], // Wrong number of features
            [4.0, 5.0, 6.0],
        ];

        let fu = FeatureUnion::new().add_transformer(
            "test",
            MockTransformer {
                scale: 1.0,
                output_features: None,
            },
        );

        let fitted_fu = fu.fit(&x_train, &()).unwrap();
        let result = fitted_fu.transform(&x_test);

        assert!(result.is_err());
        if let Err(SklearsError::FeatureMismatch { expected, actual }) = result {
            assert_eq!(expected, 2);
            assert_eq!(actual, 3);
        } else {
            panic!("Expected FeatureMismatch error");
        }
    }

    #[test]
    fn test_concatenate_features() {
        let part1 = array![[1.0, 2.0], [3.0, 4.0],];

        let part2 = array![[5.0], [6.0],];

        let part3 = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0],];

        let parts = vec![part1, part2, part3];
        let result = concatenate_features(parts).unwrap();

        assert_eq!(result.dim(), (2, 6)); // 2 + 1 + 3 columns

        // Check values
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[0, 2]], 5.0);
        assert_eq!(result[[0, 3]], 7.0);
        assert_eq!(result[[0, 4]], 8.0);
        assert_eq!(result[[0, 5]], 9.0);
    }

    #[test]
    fn test_feature_selection_variance_threshold() {
        let x = array![
            [1.0, 1.0, 1.0, 2.0], // Low variance features + high variance
            [1.1, 1.0, 1.0, 4.0],
            [0.9, 1.0, 1.0, 6.0],
            [1.0, 1.0, 1.0, 8.0],
        ];

        let fu = FeatureUnion::new()
            .add_transformer(
                "identity",
                MockTransformer {
                    scale: 1.0,
                    output_features: None,
                },
            )
            .feature_selection(FeatureSelectionStrategy::VarianceThreshold(0.1))
            .importance_method(FeatureImportanceMethod::Variance);

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let _result = fitted_fu.transform(&x).unwrap();

        // Should select only features with variance > 0.1 (likely just the last column)
        assert!(fitted_fu.is_feature_selection_enabled());
        assert!(fitted_fu.n_features_selected() <= 4);
        assert!(fitted_fu.get_feature_importances().is_some());
        assert!(fitted_fu.get_selected_features().is_some());
    }

    #[test]
    fn test_feature_selection_top_k() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fu = FeatureUnion::new()
            .add_transformer(
                "scale_by_2",
                MockTransformer {
                    scale: 2.0,
                    output_features: None,
                },
            )
            .add_transformer(
                "scale_by_3",
                MockTransformer {
                    scale: 3.0,
                    output_features: None,
                },
            )
            .feature_selection(FeatureSelectionStrategy::TopK(2))
            .importance_method(FeatureImportanceMethod::L2Norm);

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let result = fitted_fu.transform(&x).unwrap();

        // Should select exactly 2 features
        assert_eq!(fitted_fu.n_features_selected(), 2);
        assert_eq!(result.ncols(), 2);
        assert!(fitted_fu.is_feature_selection_enabled());
    }

    #[test]
    fn test_feature_selection_top_percentile() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fu = FeatureUnion::new()
            .add_transformer(
                "expand",
                MockTransformer {
                    scale: 1.0,
                    output_features: Some(6),
                },
            )
            .feature_selection(FeatureSelectionStrategy::TopPercentile(50.0))
            .importance_method(FeatureImportanceMethod::AbsoluteMean);

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let result = fitted_fu.transform(&x).unwrap();

        // Should select 50% of features (3 out of 6)
        assert_eq!(fitted_fu.n_features_selected(), 3);
        assert_eq!(result.ncols(), 3);
        assert!(fitted_fu.is_feature_selection_enabled());
    }

    #[test]
    fn test_feature_selection_disabled() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fu = FeatureUnion::new()
            .add_transformer(
                "identity",
                MockTransformer {
                    scale: 1.0,
                    output_features: None,
                },
            )
            .feature_selection(FeatureSelectionStrategy::None);

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let result = fitted_fu.transform(&x).unwrap();

        // Should keep all features
        assert!(!fitted_fu.is_feature_selection_enabled());
        assert_eq!(fitted_fu.n_features_selected(), 2);
        assert_eq!(result.ncols(), 2);
        assert!(fitted_fu.get_feature_importances().is_none());
        assert!(fitted_fu.get_selected_features().is_none());
    }

    #[test]
    fn test_feature_importance_methods() {
        let x = array![[1.0, 0.0, 10.0], [2.0, 0.0, 20.0], [3.0, 0.0, 30.0],];

        let fu = FeatureUnion::new()
            .add_transformer(
                "identity",
                MockTransformer {
                    scale: 1.0,
                    output_features: None,
                },
            )
            .enable_feature_selection(true)
            .importance_method(FeatureImportanceMethod::Variance);

        let fitted_fu = fu.fit(&x, &()).unwrap();
        let importances = fitted_fu.get_feature_importances().unwrap();

        // Third column should have highest variance
        assert!(importances[2] > importances[0]);
        assert!(importances[0] > importances[1]); // Second column has zero variance
    }

    #[test]
    fn test_get_methods() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let fu = FeatureUnion::new()
            .add_weighted_transformer(
                "test1",
                MockTransformer {
                    scale: 1.0,
                    output_features: None,
                },
                2.0,
            )
            .add_transformer(
                "test2",
                MockTransformer {
                    scale: 1.0,
                    output_features: Some(3),
                },
            );

        let fitted_fu = fu.fit(&x, &()).unwrap();

        assert_eq!(fitted_fu.n_features_in(), 2);
        assert_eq!(fitted_fu.n_features_out(), 5); // 2 + 3
        assert_eq!(fitted_fu.get_transformers().len(), 2);
        assert_eq!(fitted_fu.get_weights(), &vec![2.0, 1.0]);
    }
}
