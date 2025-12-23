//! Feature extraction pipelines
//!
//! This module provides implementations for composing multiple feature extractors
//! into unified pipelines for complex feature engineering workflows.

use crate::feature_traits::{FeatureExtractor, FeatureUnion};
use scirs2_core::ndarray::{Array2, Axis};
use sklears_core::{error::Result as SklResult, prelude::SklearsError, types::Float};

// ============================================================================
// Sequential Pipeline
// ============================================================================

/// Sequential feature extraction pipeline
///
/// Executes extractors in sequence, passing the output of one extractor
/// as input to the next (when compatible).
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::pipelines::SequentialPipeline;
/// use sklears_feature_extraction::text::{EmotionDetector, SentimentAnalyzer};
///
/// // Note: This example demonstrates the API structure
/// // Actual chaining would require compatible input/output types
/// ```
#[derive(Debug)]
pub struct SequentialPipeline<T> {
    stages: Vec<T>,
    stage_names: Vec<String>,
}

impl<T> SequentialPipeline<T> {
    /// Create a new sequential pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            stage_names: Vec::new(),
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: T, name: String) -> Self {
        self.stages.push(stage);
        self.stage_names.push(name);
        self
    }

    /// Get the number of stages
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl<T> Default for SequentialPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Parallel Feature Union
// ============================================================================

/// Parallel feature union for combining multiple extractors
///
/// Extracts features using multiple extractors in parallel and concatenates
/// the results horizontally into a single feature matrix.
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::pipelines::ParallelFeatureUnion;
/// use sklears_feature_extraction::text::{EmotionDetector, SentimentAnalyzer};
/// use sklears_feature_extraction::feature_traits::FeatureUnion;
///
/// let union = ParallelFeatureUnion::new()
///     .add_extractor(EmotionDetector::new(), "emotion".to_string())
///     .add_extractor(SentimentAnalyzer::new(), "sentiment".to_string());
///
/// let documents = vec![
///     "I am happy!".to_string(),
///     "This is sad.".to_string(),
/// ];
///
/// let combined_features = union.extract_union(&documents).unwrap();
/// assert_eq!(combined_features.nrows(), 2);
/// assert_eq!(combined_features.ncols(), 14 + 5); // emotion (14) + sentiment (5)
/// ```
#[derive(Debug, Clone)]
pub struct ParallelFeatureUnion<E> {
    extractors: Vec<E>,
    extractor_names: Vec<String>,
}

impl<E> ParallelFeatureUnion<E> {
    /// Create a new parallel feature union
    pub fn new() -> Self {
        Self {
            extractors: Vec::new(),
            extractor_names: Vec::new(),
        }
    }

    /// Add an extractor to the union
    pub fn add_extractor(mut self, extractor: E, name: String) -> Self {
        self.extractors.push(extractor);
        self.extractor_names.push(name);
        self
    }

    /// Get the number of extractors
    pub fn len(&self) -> usize {
        self.extractors.len()
    }

    /// Check if union is empty
    pub fn is_empty(&self) -> bool {
        self.extractors.is_empty()
    }

    /// Get extractor names
    pub fn extractor_names(&self) -> &[String] {
        &self.extractor_names
    }
}

impl<E> Default for ParallelFeatureUnion<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> FeatureUnion for ParallelFeatureUnion<E>
where
    E: FeatureExtractor<Input = String, Output = Array2<Float>>,
{
    type Input = String;
    type Output = Array2<Float>;

    fn extract_union(&self, input: &[Self::Input]) -> SklResult<Self::Output> {
        if self.extractors.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Feature union has no extractors".to_string(),
            ));
        }

        if input.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Extract features from all extractors
        let mut feature_matrices: Vec<Array2<Float>> = Vec::new();

        for extractor in &self.extractors {
            let features = extractor.extract_features(input)?;
            feature_matrices.push(features);
        }

        // Validate all matrices have the same number of rows
        let n_samples = feature_matrices[0].nrows();
        for (i, matrix) in feature_matrices.iter().enumerate() {
            if matrix.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Extractor {} produced {} samples, expected {}",
                    i,
                    matrix.nrows(),
                    n_samples
                )));
            }
        }

        // Concatenate horizontally
        let mut combined = feature_matrices[0].clone();
        for matrix in &feature_matrices[1..] {
            combined =
                scirs2_core::ndarray::concatenate(Axis(1), &[combined.view(), matrix.view()])
                    .map_err(|e| {
                        SklearsError::InvalidInput(format!("Concatenation failed: {}", e))
                    })?;
        }

        Ok(combined)
    }

    fn n_extractors(&self) -> usize {
        self.extractors.len()
    }

    fn feature_splits(&self) -> Vec<usize> {
        let mut splits = vec![0];
        let mut cumsum = 0;

        for extractor in &self.extractors {
            if let Some(n_features) = extractor.n_features() {
                cumsum += n_features;
                splits.push(cumsum);
            }
        }

        splits
    }
}

// ============================================================================
// Weighted Feature Union
// ============================================================================

/// Weighted feature union that applies weights to each extractor's features
///
/// Similar to ParallelFeatureUnion but applies importance weights to features
/// from each extractor before concatenation.
#[derive(Debug, Clone)]
pub struct WeightedFeatureUnion<E> {
    extractors: Vec<E>,
    weights: Vec<Float>,
    extractor_names: Vec<String>,
}

impl<E> WeightedFeatureUnion<E> {
    /// Create a new weighted feature union
    pub fn new() -> Self {
        Self {
            extractors: Vec::new(),
            weights: Vec::new(),
            extractor_names: Vec::new(),
        }
    }

    /// Add an extractor with a weight
    pub fn add_weighted_extractor(mut self, extractor: E, weight: Float, name: String) -> Self {
        self.extractors.push(extractor);
        self.weights.push(weight);
        self.extractor_names.push(name);
        self
    }

    /// Get the number of extractors
    pub fn len(&self) -> usize {
        self.extractors.len()
    }

    /// Check if union is empty
    pub fn is_empty(&self) -> bool {
        self.extractors.is_empty()
    }

    /// Get weights
    pub fn weights(&self) -> &[Float] {
        &self.weights
    }
}

impl<E> Default for WeightedFeatureUnion<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> FeatureUnion for WeightedFeatureUnion<E>
where
    E: FeatureExtractor<Input = String, Output = Array2<Float>>,
{
    type Input = String;
    type Output = Array2<Float>;

    fn extract_union(&self, input: &[Self::Input]) -> SklResult<Self::Output> {
        if self.extractors.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Feature union has no extractors".to_string(),
            ));
        }

        if input.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Extract features from all extractors and apply weights
        let mut feature_matrices: Vec<Array2<Float>> = Vec::new();

        for (extractor, weight) in self.extractors.iter().zip(&self.weights) {
            let mut features = extractor.extract_features(input)?;
            // Apply weight
            features *= *weight;
            feature_matrices.push(features);
        }

        // Validate all matrices have the same number of rows
        let n_samples = feature_matrices[0].nrows();
        for (i, matrix) in feature_matrices.iter().enumerate() {
            if matrix.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Extractor {} produced {} samples, expected {}",
                    i,
                    matrix.nrows(),
                    n_samples
                )));
            }
        }

        // Concatenate horizontally
        let mut combined = feature_matrices[0].clone();
        for matrix in &feature_matrices[1..] {
            combined =
                scirs2_core::ndarray::concatenate(Axis(1), &[combined.view(), matrix.view()])
                    .map_err(|e| {
                        SklearsError::InvalidInput(format!("Concatenation failed: {}", e))
                    })?;
        }

        Ok(combined)
    }

    fn n_extractors(&self) -> usize {
        self.extractors.len()
    }

    fn feature_splits(&self) -> Vec<usize> {
        let mut splits = vec![0];
        let mut cumsum = 0;

        for extractor in &self.extractors {
            if let Some(n_features) = extractor.n_features() {
                cumsum += n_features;
                splits.push(cumsum);
            }
        }

        splits
    }
}

// ============================================================================
// Feature Selector
// ============================================================================

/// Feature selector that selects a subset of features
#[derive(Debug, Clone)]
pub struct IndexFeatureSelector {
    selected_indices: Vec<usize>,
}

impl IndexFeatureSelector {
    /// Create a new feature selector
    pub fn new(selected_indices: Vec<usize>) -> Self {
        Self { selected_indices }
    }

    /// Select features from a feature matrix
    pub fn select_features(&self, features: &Array2<Float>) -> SklResult<Array2<Float>> {
        if self.selected_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected".to_string(),
            ));
        }

        let n_features = features.ncols();
        for &idx in &self.selected_indices {
            if idx >= n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature index {} out of range (max: {})",
                    idx,
                    n_features - 1
                )));
            }
        }

        // Select columns
        let selected_cols: Vec<_> = self
            .selected_indices
            .iter()
            .map(|&idx| features.column(idx).to_owned())
            .collect();

        let n_samples = features.nrows();
        let n_selected = selected_cols.len();
        let mut result = Array2::zeros((n_samples, n_selected));

        for (i, col) in selected_cols.iter().enumerate() {
            result.column_mut(i).assign(col);
        }

        Ok(result)
    }

    /// Get selected feature indices
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::EmotionDetector;

    #[test]
    fn test_parallel_feature_union_basic() {
        // Test with multiple instances of the same extractor type
        let detector1 = EmotionDetector::new();
        let detector2 = EmotionDetector::new().min_confidence(0.2);

        let union = ParallelFeatureUnion::new()
            .add_extractor(detector1, "emotion1".to_string())
            .add_extractor(detector2, "emotion2".to_string());

        let documents = vec!["I am happy!".to_string(), "This is sad.".to_string()];

        let features = union.extract_union(&documents).unwrap();

        assert_eq!(features.nrows(), 2);
        assert_eq!(features.ncols(), 14 + 14); // emotion (14) + emotion (14)
    }

    #[test]
    fn test_parallel_feature_union_feature_splits() {
        let union = ParallelFeatureUnion::new()
            .add_extractor(EmotionDetector::new(), "emotion1".to_string())
            .add_extractor(EmotionDetector::new(), "emotion2".to_string());

        let splits = union.feature_splits();
        assert_eq!(splits, vec![0, 14, 28]); // 0, 14 (emotion), 28 (emotion + emotion)
    }

    #[test]
    fn test_weighted_feature_union() {
        let union = WeightedFeatureUnion::new()
            .add_weighted_extractor(EmotionDetector::new(), 2.0, "emotion1".to_string())
            .add_weighted_extractor(EmotionDetector::new(), 0.5, "emotion2".to_string());

        let documents = vec!["I am happy!".to_string()];

        let features = union.extract_union(&documents).unwrap();

        assert_eq!(features.nrows(), 1);
        assert_eq!(features.ncols(), 14 + 14);
    }

    #[test]
    fn test_feature_selector() {
        let features = Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();

        let selector = IndexFeatureSelector::new(vec![0, 2, 4]);
        let selected = selector.select_features(&features).unwrap();

        assert_eq!(selected.nrows(), 2);
        assert_eq!(selected.ncols(), 3);
        assert_eq!(selected[[0, 0]], 1.0);
        assert_eq!(selected[[0, 1]], 3.0);
        assert_eq!(selected[[0, 2]], 5.0);
    }

    #[test]
    fn test_feature_selector_out_of_range() {
        let features = Array2::zeros((2, 5));
        let selector = IndexFeatureSelector::new(vec![10]);

        assert!(selector.select_features(&features).is_err());
    }
}
