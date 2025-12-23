//! Custom transformers and advanced feature extraction
//!
//! This module provides specialized transformers for advanced feature extraction
//! including mixed-type data handling, sketching algorithms, and custom encodings.

use crate::*;
// use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use sklears_core::prelude::{Estimator, Fit, SklearsError, Transform};
use sklears_core::traits::Untrained;
// use std::hash::Hash;

/// Mixed-type feature extractor for handling heterogeneous data
///
/// This extractor can handle different types of features (numerical, categorical, binary, ordinal)
/// and apply appropriate transformations to each type, creating a unified feature matrix.
#[derive(Debug, Clone)]
pub struct MixedTypeFeatureExtractor {
    /// Types of features for each column
    pub feature_types: Vec<FeatureType>,
    /// Whether to normalize numerical features
    pub normalize_numerical: bool,
    /// Whether to handle missing values
    pub handle_missing_values: bool,
    /// Strategy for handling missing values
    pub missing_value_strategy: MissingValueStrategy,
    /// Categorical encoding strategy
    pub encoding_strategy: CategoricalEncoding,
    /// Whether to include interaction features
    pub include_interaction_features: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

/// Types of features that can be handled
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    /// Continuous numerical features
    Numerical,
    /// Categorical features with no inherent order
    Categorical,
    /// Binary features (0/1)
    Binary,
    /// Ordinal features with inherent order
    Ordinal,
    /// Date/time features
    DateTime,
}

/// Strategies for handling missing values
#[derive(Debug, Clone, Copy)]
pub enum MissingValueStrategy {
    /// Replace with mean (for numerical) or mode (for categorical)
    Mean,
    /// Replace with median (for numerical) or mode (for categorical)
    Median,
    /// Replace with mode (most frequent value)
    Mode,
    /// Replace with zero
    Zero,
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
}

/// Categorical encoding strategies
#[derive(Debug, Clone, Copy)]
pub enum CategoricalEncoding {
    /// One-hot encoding
    OneHot,
    /// Label encoding (ordinal integers)
    LabelEncoding,
    /// Target encoding (mean of target for each category)
    TargetEncoding,
    /// Binary encoding
    BinaryEncoding,
    /// Frequency encoding
    FrequencyEncoding,
}

impl MixedTypeFeatureExtractor {
    /// Create a new mixed-type feature extractor
    pub fn new() -> Self {
        Self {
            feature_types: Vec::new(),
            normalize_numerical: true,
            handle_missing_values: true,
            missing_value_strategy: MissingValueStrategy::Mean,
            encoding_strategy: CategoricalEncoding::OneHot,
            include_interaction_features: false,
            random_state: None,
        }
    }

    /// Set feature types for each column
    pub fn feature_types(mut self, types: Vec<FeatureType>) -> Self {
        self.feature_types = types;
        self
    }

    /// Set whether to normalize numerical features
    pub fn normalize_numerical(mut self, normalize: bool) -> Self {
        self.normalize_numerical = normalize;
        self
    }

    /// Set whether to handle missing values
    pub fn handle_missing_values(mut self, handle: bool) -> Self {
        self.handle_missing_values = handle;
        self
    }

    /// Set missing value strategy
    pub fn missing_value_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.missing_value_strategy = strategy;
        self
    }

    /// Set categorical encoding strategy
    pub fn encoding_strategy(mut self, strategy: CategoricalEncoding) -> Self {
        self.encoding_strategy = strategy;
        self
    }

    /// Set whether to include interaction features
    pub fn include_interaction_features(mut self, include: bool) -> Self {
        self.include_interaction_features = include;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Handle missing values in data
    fn handle_missing(
        &self,
        data: &Array2<f64>,
        col_idx: usize,
        feature_type: FeatureType,
    ) -> Array1<f64> {
        let column = data.column(col_idx);
        let mut result = Array1::zeros(column.len());

        // Find non-missing values
        let valid_values: Vec<f64> = column.iter().filter(|&&x| x.is_finite()).copied().collect();

        if valid_values.is_empty() {
            return result; // All zeros if no valid values
        }

        let replacement_value = match (self.missing_value_strategy, feature_type) {
            (MissingValueStrategy::Mean, FeatureType::Numerical) => {
                valid_values.iter().sum::<f64>() / valid_values.len() as f64
            }
            (MissingValueStrategy::Median, FeatureType::Numerical) => {
                let mut sorted = valid_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
            (MissingValueStrategy::Mode, _) => {
                // Find most frequent value
                let mut counts = std::collections::HashMap::new();
                for &val in &valid_values {
                    *counts.entry(val as i64).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(val, _)| val as f64)
                    .unwrap_or(0.0)
            }
            (MissingValueStrategy::Zero, _) => 0.0,
            (MissingValueStrategy::ForwardFill, _) => {
                // Use first valid value
                valid_values[0]
            }
            (MissingValueStrategy::BackwardFill, _) => {
                // Use last valid value
                valid_values[valid_values.len() - 1]
            }
            _ => valid_values.iter().sum::<f64>() / valid_values.len() as f64, // Default to mean
        };

        // Replace missing values
        for (i, &value) in column.iter().enumerate() {
            result[i] = if value.is_finite() {
                value
            } else {
                replacement_value
            };
        }

        result
    }

    /// Normalize numerical features
    fn normalize_column(&self, column: &Array1<f64>) -> Array1<f64> {
        let mean = column.mean().unwrap_or(0.0);
        let std = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
        let std = std.sqrt();

        if std == 0.0 {
            Array1::zeros(column.len())
        } else {
            column.mapv(|x| (x - mean) / std)
        }
    }

    /// Encode categorical features
    fn encode_categorical(&self, column: &Array1<f64>) -> Array2<f64> {
        match self.encoding_strategy {
            CategoricalEncoding::OneHot => {
                // Find unique values
                let mut unique_values: Vec<i64> = column
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                unique_values.sort();

                let n_categories = unique_values.len();
                let mut encoded = Array2::zeros((column.len(), n_categories));

                for (row_idx, &value) in column.iter().enumerate() {
                    if let Some(cat_idx) = unique_values.iter().position(|&x| x == value as i64) {
                        encoded[[row_idx, cat_idx]] = 1.0;
                    }
                }

                encoded
            }
            CategoricalEncoding::LabelEncoding => {
                // Map each unique value to an integer
                let mut unique_values: Vec<i64> = column
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                unique_values.sort();

                let mut encoded = Array2::zeros((column.len(), 1));
                for (row_idx, &value) in column.iter().enumerate() {
                    if let Some(label) = unique_values.iter().position(|&x| x == value as i64) {
                        encoded[[row_idx, 0]] = label as f64;
                    }
                }

                encoded
            }
            CategoricalEncoding::TargetEncoding => {
                // For simplicity, use the value itself (would normally use target mean)
                let mut encoded = Array2::zeros((column.len(), 1));
                for (row_idx, &value) in column.iter().enumerate() {
                    encoded[[row_idx, 0]] = value;
                }
                encoded
            }
            _ => {
                // Default to label encoding
                let mut encoded = Array2::zeros((column.len(), 1));
                for (row_idx, &value) in column.iter().enumerate() {
                    encoded[[row_idx, 0]] = value;
                }
                encoded
            }
        }
    }

    /// Process a single feature column
    fn process_feature(
        &self,
        data: &Array2<f64>,
        col_idx: usize,
        feature_type: FeatureType,
    ) -> Array2<f64> {
        // Handle missing values
        let column = if self.handle_missing_values {
            self.handle_missing(data, col_idx, feature_type)
        } else {
            data.column(col_idx).to_owned()
        };

        match feature_type {
            FeatureType::Numerical => {
                let processed = if self.normalize_numerical {
                    self.normalize_column(&column)
                } else {
                    column
                };
                processed.insert_axis(Axis(1))
            }
            FeatureType::Categorical => self.encode_categorical(&column),
            FeatureType::Binary => {
                // Ensure binary values are 0 or 1
                let binary = column.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 });
                binary.insert_axis(Axis(1))
            }
            FeatureType::Ordinal => {
                // Treat as numerical but preserve order
                let processed = if self.normalize_numerical {
                    self.normalize_column(&column)
                } else {
                    column
                };
                processed.insert_axis(Axis(1))
            }
            FeatureType::DateTime => {
                // For simplicity, treat as numerical
                let processed = if self.normalize_numerical {
                    self.normalize_column(&column)
                } else {
                    column
                };
                processed.insert_axis(Axis(1))
            }
        }
    }

    /// Transform the data according to feature types
    fn transform_data(&self, data: &Array2<f64>) -> SklResult<Array2<f64>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.feature_types.len() != data.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of feature types ({}) doesn't match number of columns ({})",
                self.feature_types.len(),
                data.ncols()
            )));
        }

        let mut processed_features = Vec::new();

        // Process each feature according to its type
        for (col_idx, &feature_type) in self.feature_types.iter().enumerate() {
            let processed = self.process_feature(data, col_idx, feature_type);
            processed_features.push(processed);
        }

        // Concatenate all processed features
        if processed_features.is_empty() {
            return Ok(Array2::zeros((data.nrows(), 0)));
        }

        let total_cols: usize = processed_features.iter().map(|f| f.ncols()).sum();
        let mut result = Array2::zeros((data.nrows(), total_cols));

        let mut col_offset = 0;
        for feature_matrix in processed_features {
            let n_cols = feature_matrix.ncols();
            result
                .slice_mut(s![.., col_offset..col_offset + n_cols])
                .assign(&feature_matrix);
            col_offset += n_cols;
        }

        Ok(result)
    }
}

/// Fitted mixed-type feature extractor
pub struct FittedMixedTypeFeatureExtractor {
    extractor: MixedTypeFeatureExtractor,
    transformation_stats: std::collections::HashMap<String, f64>,
}

impl FittedMixedTypeFeatureExtractor {
    /// Get transformation statistics
    pub fn transformation_statistics(&self) -> &std::collections::HashMap<String, f64> {
        &self.transformation_stats
    }
}

impl Estimator<Untrained> for MixedTypeFeatureExtractor {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for MixedTypeFeatureExtractor {
    type Fitted = FittedMixedTypeFeatureExtractor;

    fn fit(self, X: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if X.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Set default feature types if not specified
        let feature_types = if self.feature_types.is_empty() {
            vec![FeatureType::Numerical; X.ncols()]
        } else {
            self.feature_types.clone()
        };

        // Calculate transformation statistics
        let mut stats = std::collections::HashMap::new();

        let n_numerical = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Numerical)
            .count();
        let n_categorical = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Categorical)
            .count();
        let n_binary = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Binary)
            .count();
        let n_ordinal = feature_types
            .iter()
            .filter(|&&t| t == FeatureType::Ordinal)
            .count();

        stats.insert("n_numerical_features".to_string(), n_numerical as f64);
        stats.insert("n_categorical_features".to_string(), n_categorical as f64);
        stats.insert("n_binary_features".to_string(), n_binary as f64);
        stats.insert("n_ordinal_features".to_string(), n_ordinal as f64);
        stats.insert("total_input_features".to_string(), X.ncols() as f64);

        // Estimate output feature count (simplified)
        let estimated_output_features = match self.encoding_strategy {
            CategoricalEncoding::OneHot => {
                // Estimate expanded size for one-hot encoding
                let categorical_expansion = n_categorical * 3; // Assume avg 3 categories
                n_numerical + n_binary + n_ordinal + categorical_expansion
            }
            _ => feature_types.len(),
        };

        stats.insert(
            "total_output_features".to_string(),
            estimated_output_features as f64,
        );

        let updated_extractor = MixedTypeFeatureExtractor {
            feature_types,
            ..self
        };

        Ok(FittedMixedTypeFeatureExtractor {
            extractor: updated_extractor,
            transformation_stats: stats,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedMixedTypeFeatureExtractor {
    fn transform(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.extractor.transform_data(X)
    }
}

impl Default for MixedTypeFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Count-Min Sketch for approximate frequency counting
///
/// A probabilistic data structure that provides an approximate count of elements
/// in a data stream using a small amount of memory. Useful for detecting heavy
/// hitters and approximate frequency analysis.
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::custom_transformers::CountMinSketch;
/// use sklears_core::traits::Transform;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| (x % 50) as f64).collect()).unwrap();
/// let sketch = CountMinSketch::new()
///     .width(100)
///     .depth(5)
///     .include_frequency_statistics(true);
///
/// let features = sketch.transform(&data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    include_frequency_statistics: bool,
    include_sketch_properties: bool,
    hash_seed: u64,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch
    pub fn new() -> Self {
        Self {
            width: 1000,
            depth: 5,
            include_frequency_statistics: true,
            include_sketch_properties: true,
            hash_seed: 42,
        }
    }

    /// Set the width of the sketch (number of buckets per hash function)
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the depth of the sketch (number of hash functions)
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Set whether to include frequency statistics
    pub fn include_frequency_statistics(mut self, include: bool) -> Self {
        self.include_frequency_statistics = include;
        self
    }

    /// Set whether to include sketch properties
    pub fn include_sketch_properties(mut self, include: bool) -> Self {
        self.include_sketch_properties = include;
        self
    }

    /// Set the hash seed for reproducibility
    pub fn hash_seed(mut self, seed: u64) -> Self {
        self.hash_seed = seed;
        self
    }

    /// Hash function implementation
    fn hash(&self, item: u64, hash_idx: usize) -> usize {
        // Simple hash function combining item and hash index
        let combined = item.wrapping_add((hash_idx as u64).wrapping_mul(self.hash_seed));
        let hash_val = combined.wrapping_mul(0x9e3779b97f4a7c15_u64);
        (hash_val as usize) % self.width
    }

    /// Build the Count-Min Sketch from data
    fn build_sketch(&self, data: &Array2<f64>) -> Array2<u32> {
        let mut sketch = Array2::zeros((self.depth, self.width));

        // Process each data point
        for row in data.outer_iter() {
            for &value in row.iter() {
                // Convert value to integer for hashing
                let item = (value * 1000.0) as u64; // Scale to preserve some precision

                // Update all hash functions
                for hash_idx in 0..self.depth {
                    let bucket = self.hash(item, hash_idx);
                    sketch[[hash_idx, bucket]] += 1;
                }
            }
        }

        sketch
    }

    /// Estimate frequency of an item
    fn estimate_frequency(&self, sketch: &Array2<u32>, item: f64) -> u32 {
        let item_hash = (item * 1000.0) as u64;
        let mut min_count = u32::MAX;

        for hash_idx in 0..self.depth {
            let bucket = self.hash(item_hash, hash_idx);
            min_count = min_count.min(sketch[[hash_idx, bucket]]);
        }

        min_count
    }

    /// Extract features from the sketch
    fn extract_sketch_features(&self, sketch: &Array2<u32>, data: &Array2<f64>) -> Vec<f64> {
        let mut features = Vec::new();

        if self.include_frequency_statistics {
            // Total count
            let total_count: u32 = sketch.sum();
            features.push(total_count as f64);

            // Non-zero buckets
            let non_zero_buckets = sketch.iter().filter(|&&x| x > 0).count();
            features.push(non_zero_buckets as f64);

            // Load factor (fraction of non-zero buckets)
            let load_factor = non_zero_buckets as f64 / (self.width * self.depth) as f64;
            features.push(load_factor);

            // Maximum bucket count
            let max_count = sketch.iter().max().copied().unwrap_or(0);
            features.push(max_count as f64);

            // Average bucket count (excluding zeros)
            let avg_count = if non_zero_buckets > 0 {
                total_count as f64 / non_zero_buckets as f64
            } else {
                0.0
            };
            features.push(avg_count);
        }

        if self.include_sketch_properties {
            // Sketch dimensions
            features.push(self.width as f64);
            features.push(self.depth as f64);

            // Collision rate estimate (approximate)
            let unique_items = data
                .iter()
                .map(|&x| (x * 1000.0) as u64)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let collision_rate = if unique_items > 0 {
                1.0 - (unique_items as f64 / (self.width as f64))
            } else {
                0.0
            };
            features.push(collision_rate.max(0.0));
        }

        features
    }
}

impl Default for CountMinSketch {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array1<Float>> for CountMinSketch {
    fn transform(&self, data: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot create sketch from empty data".to_string(),
            ));
        }

        // Convert to f64 for internal processing
        let data_owned = data.mapv(|x| x).to_owned();

        let sketch = self.build_sketch(&data_owned);
        let features = self.extract_sketch_features(&sketch, &data_owned);

        Ok(Array1::from_vec(
            features.into_iter().map(|x| x as Float).collect(),
        ))
    }
}
