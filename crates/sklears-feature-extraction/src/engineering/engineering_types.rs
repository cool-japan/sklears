//! Common type definitions and enumerations for feature engineering
//!
//! This module provides core types used across the engineering modules,
//! ensuring consistency and reducing code duplication.

use crate::*;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Types of features that can be handled by mixed-type extractors
///
/// This enumeration defines the different types of features that can be
/// processed by feature engineering algorithms, each requiring different
/// handling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureType {
    /// Continuous numerical features (e.g., height, weight, temperature)
    Numerical,
    /// Categorical features with no inherent order (e.g., color, country)
    Categorical,
    /// Binary features with only two states (0/1, true/false)
    Binary,
    /// Ordinal features with inherent order (e.g., size: small/medium/large)
    Ordinal,
    /// Date/time features requiring temporal processing
    DateTime,
    /// Text features requiring natural language processing
    Text,
}

/// Strategies for handling missing values in datasets
///
/// Different strategies are appropriate for different data types and
/// analysis requirements. Each strategy has trade-offs in terms of
/// computational cost and statistical validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueStrategy {
    /// Replace with mean (for numerical) or mode (for categorical)
    /// Fast and simple, preserves overall distribution
    Mean,
    /// Replace with median (for numerical) or mode (for categorical)
    /// More robust to outliers than mean
    Median,
    /// Replace with mode (most frequent value)
    /// Best for categorical data
    Mode,
    /// Replace with zero
    /// Simple but can introduce bias
    Zero,
    /// Forward fill - use previous non-missing value
    /// Good for time series data
    ForwardFill,
    /// Backward fill - use next non-missing value
    /// Good for time series data
    BackwardFill,
    /// Use linear interpolation between non-missing values
    /// Good for smooth numerical data
    Interpolate,
    /// Drop rows/columns with missing values
    /// Reduces dataset size but avoids bias
    Drop,
}

/// Categorical encoding strategies for converting categorical data to numerical
///
/// Different encoding strategies are appropriate for different types of
/// categorical data and machine learning algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalEncoding {
    OneHot,
    LabelEncoding,
    TargetEncoding,
    BinaryEncoding,
    FrequencyEncoding,
    HashEncoding,
}

/// Transform types for fast mathematical transformations
///
/// These represent different classes of fast transforms that can be
/// applied to data for feature extraction and dimensionality reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformType {
    /// Fast Fourier Transform - frequency domain analysis
    FFT,
    /// Discrete Cosine Transform - energy compaction
    DCT,
    /// Walsh-Hadamard Transform - digital signal processing
    Walsh,
    /// Haar Transform - wavelet analysis
    Haar,
    /// Discrete Wavelet Transform - time-frequency analysis
    DWT,
    /// Random Fourier Features - kernel approximation
    RFF,
}

/// Distance metrics for similarity and clustering operations
///
/// Different distance metrics capture different notions of similarity
/// and are appropriate for different data types and analysis goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceMetric {
    /// Euclidean distance - standard geometric distance
    Euclidean,
    /// Manhattan distance - sum of absolute differences
    Manhattan,
    /// Cosine distance - angle between vectors
    Cosine,
    /// Hamming distance - number of differing positions
    Hamming,
    /// Jaccard distance - set similarity measure
    Jaccard,
    /// Minkowski distance with configurable p parameter
    Minkowski(u32),
    /// Mahalanobis distance - covariance-weighted
    Mahalanobis,
}

/// Sampling strategies for data selection and reduction
///
/// Different sampling strategies preserve different aspects of the
/// original data distribution and are suitable for different scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Simple random sampling - uniform probability
    Random,
    /// Stratified sampling - maintain class proportions
    Stratified,
    /// Systematic sampling - select every nth item
    Systematic,
    /// Importance sampling - weight by importance
    Importance,
    /// Reservoir sampling - maintain sample as stream arrives
    Reservoir,
    /// Bootstrap sampling - sample with replacement
    Bootstrap,
    /// Cluster-based sampling - representative from each cluster
    Cluster,
}

/// Aggregation functions for feature summarization
///
/// These functions reduce collections of values to single representative
/// values, useful for creating summary statistics and reducing dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationFunction {
    /// Mean (average) value
    Mean,
    /// Median (middle) value
    Median,
    /// Standard deviation
    Std,
    /// Variance
    Variance,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Sum of values
    Sum,
    /// Count of values
    Count,
    /// Skewness (asymmetry measure)
    Skewness,
    /// Kurtosis (tail heaviness measure)
    Kurtosis,
    /// Specified percentile
    Percentile(u8),
    /// Range (max - min)
    Range,
}

/// Feature selection criteria for dimensionality reduction
///
/// Different criteria evaluate features based on different statistical
/// properties and relationships with target variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionCriterion {
    /// Select k best features by score
    KBest(usize),
    /// Select features above threshold
    Threshold(u32), // scaled threshold * 1000 for integer storage
    /// Select top percentile of features
    Percentile(u8),
    /// Variance-based selection
    VarianceThreshold,
    /// Mutual information-based selection
    MutualInformation,
    /// Chi-squared test-based selection
    ChiSquared,
    /// ANOVA F-test-based selection
    FTest,
}

/// Common configuration for engineering operations
///
/// This struct provides standard configuration options that are
/// commonly used across different feature engineering operations.
#[derive(Debug, Clone)]
pub struct EngineeringConfig {
    /// Random seed for reproducible results
    pub random_state: Option<u64>,
    /// Number of parallel jobs to use (None for automatic)
    pub n_jobs: Option<usize>,
    /// Whether to use verbose output
    pub verbose: bool,
    /// Memory limit for operations (in bytes)
    pub memory_limit: Option<usize>,
    /// Chunk size for batch processing
    pub chunk_size: usize,
}

impl Default for EngineeringConfig {
    fn default() -> Self {
        Self {
            random_state: Some(42),
            n_jobs: None,
            verbose: false,
            memory_limit: None,
            chunk_size: 1000,
        }
    }
}

/// Type alias for feature importance scores
pub type FeatureImportance = HashMap<String, Float>;

/// Type alias for feature names
pub type FeatureNames = Vec<String>;

/// Type alias for categorical mappings
pub type CategoryMapping = HashMap<String, HashMap<String, usize>>;

/// Result type for engineering operations
pub type EngineeringResult<T> = Result<T, SklearsError>;

/// Statistics for a single feature
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// Feature name
    pub name: String,
    /// Feature type
    pub feature_type: FeatureType,
    /// Number of non-missing values
    pub count: usize,
    /// Number of unique values
    pub unique_count: usize,
    /// Mean value (for numerical features)
    pub mean: Option<Float>,
    /// Standard deviation (for numerical features)
    pub std: Option<Float>,
    /// Minimum value (for numerical features)
    pub min: Option<Float>,
    /// Maximum value (for numerical features)
    pub max: Option<Float>,
    /// Most frequent value
    pub mode: Option<String>,
    /// Fraction of missing values
    pub missing_fraction: Float,
}

impl FeatureStatistics {
    /// Create new feature statistics
    pub fn new(name: String, feature_type: FeatureType) -> Self {
        Self {
            name,
            feature_type,
            count: 0,
            unique_count: 0,
            mean: None,
            std: None,
            min: None,
            max: None,
            mode: None,
            missing_fraction: 0.0,
        }
    }

    /// Check if feature is categorical
    pub fn is_categorical(&self) -> bool {
        matches!(
            self.feature_type,
            FeatureType::Categorical | FeatureType::Binary | FeatureType::Ordinal
        )
    }

    /// Check if feature is numerical
    pub fn is_numerical(&self) -> bool {
        matches!(self.feature_type, FeatureType::Numerical)
    }

    /// Check if feature has high cardinality (many unique values)
    pub fn is_high_cardinality(&self) -> bool {
        if self.count == 0 {
            return false;
        }
        let cardinality_ratio = self.unique_count as Float / self.count as Float;
        cardinality_ratio > 0.5 // More than 50% unique values
    }
}

/// Feature transformation metadata
#[derive(Debug, Clone)]
pub struct TransformationMetadata {
    /// Original feature names
    pub input_features: FeatureNames,
    /// Generated feature names
    pub output_features: FeatureNames,
    /// Transformation parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Feature statistics before transformation
    pub input_statistics: Vec<FeatureStatistics>,
    /// Transformation timestamp
    pub timestamp: String,
}

impl TransformationMetadata {
    /// Create new transformation metadata
    pub fn new(input_features: FeatureNames, output_features: FeatureNames) -> Self {
        Self {
            input_features,
            output_features,
            parameters: HashMap::new(),
            input_statistics: Vec::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Add parameter to metadata
    pub fn add_parameter<T: serde::Serialize>(&mut self, key: String, value: T) {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.parameters.insert(key, json_value);
        }
    }

    /// Get number of input features
    pub fn input_dimension(&self) -> usize {
        self.input_features.len()
    }

    /// Get number of output features
    pub fn output_dimension(&self) -> usize {
        self.output_features.len()
    }

    /// Calculate dimensionality change ratio
    pub fn dimensionality_ratio(&self) -> Float {
        if self.input_features.is_empty() {
            return 0.0;
        }
        self.output_features.len() as Float / self.input_features.len() as Float
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_type_properties() {
        assert_eq!(FeatureType::Numerical, FeatureType::Numerical);
        assert_ne!(FeatureType::Numerical, FeatureType::Categorical);
    }

    #[test]
    fn test_missing_value_strategy() {
        let strategy = MissingValueStrategy::Mean;
        assert_eq!(strategy, MissingValueStrategy::Mean);
    }

    #[test]
    fn test_feature_statistics() {
        let stats = FeatureStatistics::new("test".to_string(), FeatureType::Numerical);
        assert!(stats.is_numerical());
        assert!(!stats.is_categorical());
        assert!(!stats.is_high_cardinality());
    }

    #[test]
    fn test_engineering_config_default() {
        let config = EngineeringConfig::default();
        assert_eq!(config.random_state, Some(42));
        assert_eq!(config.chunk_size, 1000);
        assert!(!config.verbose);
    }

    #[test]
    fn test_transformation_metadata() {
        let input = vec!["feature1".to_string(), "feature2".to_string()];
        let output = vec!["transformed1".to_string(), "transformed2".to_string(), "transformed3".to_string()];

        let metadata = TransformationMetadata::new(input, output);
        assert_eq!(metadata.input_dimension(), 2);
        assert_eq!(metadata.output_dimension(), 3);
        assert_eq!(metadata.dimensionality_ratio(), 1.5);
    }

    #[test]
    fn test_distance_metric_equality() {
        assert_eq!(DistanceMetric::Euclidean, DistanceMetric::Euclidean);
        assert_eq!(DistanceMetric::Minkowski(2), DistanceMetric::Minkowski(2));
        assert_ne!(DistanceMetric::Minkowski(1), DistanceMetric::Minkowski(2));
    }

    #[test]
    fn test_selection_criterion() {
        let criterion = SelectionCriterion::KBest(10);
        match criterion {
            SelectionCriterion::KBest(k) => assert_eq!(k, 10),
            _ => panic!("Wrong criterion type"),
        }
    }
}