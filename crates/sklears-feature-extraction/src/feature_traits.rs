//! Trait-based feature extraction framework
//!
//! This module provides a unified trait system for feature extractors,
//! enabling composable, type-safe feature extraction pipelines.

use sklears_core::{error::Result as SklResult, types::Float};
use std::fmt::Debug;

// ============================================================================
// Core Feature Extraction Traits
// ============================================================================

/// Core trait for feature extraction from raw data
///
/// This trait provides a unified interface for all feature extractors in the crate,
/// enabling composable pipelines and consistent API design.
///
/// # Type Parameters
///
/// - `Input`: The type of input data (e.g., String, Vec`<u8>`, Array2`<Float>`)
/// - `Output`: The type of extracted features (typically Array2`<Float>`)
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::feature_traits::FeatureExtractor;
/// use scirs2_core::ndarray::Array2;
///
/// struct MyExtractor;
///
/// impl FeatureExtractor for MyExtractor {
///     type Input = String;
///     type Output = Array2`<f64>`;
///
///     fn extract_features(&self, input: &[Self::Input]) -> sklears_core::error::Result<Self::Output> {
///         // Implementation here
///         Ok(Array2::zeros((input.len(), 10)))
///     }
///
///     fn feature_names(&self) -> Option<`Vec<String>`> {
///         Some((0..10).map(|i| format!("feature_{}", i)).collect())
///     }
/// }
/// ```
pub trait FeatureExtractor: Debug + Send + Sync {
    /// Input data type
    type Input;

    /// Output feature type
    type Output;

    /// Extract features from input data
    ///
    /// # Arguments
    ///
    /// * `input` - Slice of input data to extract features from
    ///
    /// # Returns
    ///
    /// * `Ok(features)` - Extracted feature matrix
    /// * `Err(e)` - Error if extraction fails
    fn extract_features(&self, input: &[Self::Input]) -> SklResult<Self::Output>;

    /// Get feature names (if available)
    ///
    /// Returns a vector of feature names for interpretability.
    /// If feature names are not meaningful, returns None.
    fn feature_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Get the number of features produced by this extractor
    fn n_features(&self) -> Option<usize> {
        None
    }

    /// Validate input data before extraction
    ///
    /// Default implementation always returns Ok(()).
    /// Override to add custom validation logic.
    fn validate_input(&self, _input: &[Self::Input]) -> SklResult<()> {
        Ok(())
    }
}

/// Trait for feature extractors that can be configured
///
/// This trait allows feature extractors to expose their configuration
/// and support dynamic reconfiguration.
pub trait ConfigurableExtractor: FeatureExtractor {
    /// Configuration type
    type Config: Clone + Debug;

    /// Get current configuration
    fn config(&self) -> Self::Config;

    /// Create a new extractor with updated configuration
    fn with_config(&self, config: Self::Config) -> Self;
}

/// Trait for feature extractors that can be fitted to data
///
/// Some extractors need to learn parameters from data before extraction
/// (e.g., vocabulary for text vectorizers, PCA components).
pub trait FittableExtractor: FeatureExtractor {
    /// Fitted state type
    type Fitted: FeatureExtractor<Input = Self::Input, Output = Self::Output>;

    /// Fit the extractor to training data
    ///
    /// # Arguments
    ///
    /// * `data` - Training data to fit on
    ///
    /// # Returns
    ///
    /// * `Ok(fitted)` - Fitted extractor ready for transformation
    /// * `Err(e)` - Error if fitting fails
    fn fit(&self, data: &[Self::Input]) -> SklResult<Self::Fitted>;
}

/// Trait for incremental/online feature extraction
///
/// Allows processing data in batches without loading everything into memory.
pub trait StreamingExtractor: FeatureExtractor {
    /// Process a single batch of data
    ///
    /// # Arguments
    ///
    /// * `batch` - Batch of data to process
    ///
    /// # Returns
    ///
    /// * `Ok(features)` - Features for this batch
    /// * `Err(e)` - Error if extraction fails
    fn extract_batch(&mut self, batch: &[Self::Input]) -> SklResult<Self::Output>;

    /// Reset the extractor state for a new stream
    fn reset(&mut self);

    /// Finalize extraction and get accumulated results
    fn finalize(&mut self) -> SklResult<Self::Output>;
}

/// Trait for feature extractors that support parallel processing
pub trait ParallelExtractor: FeatureExtractor {
    /// Extract features using parallel processing
    ///
    /// # Arguments
    ///
    /// * `input` - Input data to process
    /// * `n_jobs` - Number of parallel jobs (None = auto-detect)
    ///
    /// # Returns
    ///
    /// * `Ok(features)` - Extracted features
    /// * `Err(e)` - Error if extraction fails
    fn extract_parallel(
        &self,
        input: &[Self::Input],
        n_jobs: Option<usize>,
    ) -> SklResult<Self::Output>;
}

// ============================================================================
// Feature Transformation Traits
// ============================================================================

/// Trait for composable feature transformations
///
/// Allows chaining multiple feature extractors in a pipeline.
pub trait FeatureTransformer: Debug + Send + Sync {
    /// Input feature type
    type Input;

    /// Output feature type
    type Output;

    /// Transform features
    ///
    /// # Arguments
    ///
    /// * `features` - Input features to transform
    ///
    /// # Returns
    ///
    /// * `Ok(transformed)` - Transformed features
    /// * `Err(e)` - Error if transformation fails
    fn transform(&self, features: &Self::Input) -> SklResult<Self::Output>;

    /// Check if transformer is invertible
    fn is_invertible(&self) -> bool {
        false
    }

    /// Inverse transform (if supported)
    ///
    /// # Arguments
    ///
    /// * `features` - Transformed features to invert
    ///
    /// # Returns
    ///
    /// * `Ok(original)` - Original features
    /// * `Err(e)` - Error if not invertible or transformation fails
    fn inverse_transform(&self, _features: &Self::Output) -> SklResult<Self::Input> {
        Err(sklears_core::prelude::SklearsError::NotImplemented(
            "Inverse transform not supported".to_string(),
        ))
    }
}

/// Trait for feature selectors
///
/// Selects a subset of features based on some criterion.
pub trait FeatureSelector: Debug + Send + Sync {
    /// Get indices of selected features
    fn selected_features(&self) -> Vec<usize>;

    /// Get number of selected features
    fn n_selected(&self) -> usize {
        self.selected_features().len()
    }

    /// Check if a feature is selected
    fn is_selected(&self, feature_idx: usize) -> bool {
        self.selected_features().contains(&feature_idx)
    }
}

// ============================================================================
// Domain-Specific Traits
// ============================================================================

/// Trait for text feature extractors
pub trait TextFeatureExtractor: FeatureExtractor<Input = String> {
    /// Get vocabulary (if applicable)
    fn vocabulary(&self) -> Option<Vec<String>> {
        None
    }

    /// Get vocabulary size
    fn vocabulary_size(&self) -> Option<usize> {
        self.vocabulary().map(|v| v.len())
    }
}

/// Trait for image feature extractors
pub trait ImageFeatureExtractor: FeatureExtractor {
    /// Get expected image dimensions (height, width, channels)
    fn image_shape(&self) -> Option<(usize, usize, usize)> {
        None
    }

    /// Whether extractor supports color images
    fn supports_color(&self) -> bool {
        true
    }

    /// Whether extractor supports grayscale images
    fn supports_grayscale(&self) -> bool {
        true
    }
}

/// Trait for time series feature extractors
pub trait TimeSeriesFeatureExtractor: FeatureExtractor {
    /// Get window size (if applicable)
    fn window_size(&self) -> Option<usize> {
        None
    }

    /// Get step size for sliding windows (if applicable)
    fn step_size(&self) -> Option<usize> {
        None
    }

    /// Whether extractor supports variable-length sequences
    fn supports_variable_length(&self) -> bool {
        false
    }
}

/// Trait for graph feature extractors
pub trait GraphFeatureExtractor: FeatureExtractor {
    /// Whether extractor supports directed graphs
    fn supports_directed(&self) -> bool {
        true
    }

    /// Whether extractor supports weighted graphs
    fn supports_weighted(&self) -> bool {
        true
    }

    /// Whether extractor supports attributed graphs
    fn supports_attributes(&self) -> bool {
        false
    }
}

// ============================================================================
// Pipeline and Composition Traits
// ============================================================================

/// Trait for feature extraction pipelines
///
/// Combines multiple extractors and transformers into a single pipeline.
pub trait FeaturePipeline: Debug + Send + Sync {
    /// Input data type
    type Input;

    /// Output feature type
    type Output;

    /// Execute the full pipeline
    ///
    /// # Arguments
    ///
    /// * `input` - Input data
    ///
    /// # Returns
    ///
    /// * `Ok(features)` - Final extracted features
    /// * `Err(e)` - Error if pipeline fails
    fn execute(&self, input: &[Self::Input]) -> SklResult<Self::Output>;

    /// Get number of stages in the pipeline
    fn n_stages(&self) -> usize;

    /// Get stage names (if available)
    fn stage_names(&self) -> Vec<String> {
        (0..self.n_stages())
            .map(|i| format!("stage_{}", i))
            .collect()
    }
}

/// Trait for feature union (combining multiple extractors)
///
/// Concatenates features from multiple extractors horizontally.
pub trait FeatureUnion: Debug + Send + Sync {
    /// Input data type
    type Input;

    /// Output feature type
    type Output;

    /// Extract and combine features from all extractors
    ///
    /// # Arguments
    ///
    /// * `input` - Input data
    ///
    /// # Returns
    ///
    /// * `Ok(combined)` - Concatenated features
    /// * `Err(e)` - Error if extraction fails
    fn extract_union(&self, input: &[Self::Input]) -> SklResult<Self::Output>;

    /// Get number of extractors in the union
    fn n_extractors(&self) -> usize;

    /// Get feature split points (cumulative feature counts)
    fn feature_splits(&self) -> Vec<usize>;
}

// ============================================================================
// Metadata Traits
// ============================================================================

/// Trait for extractors that provide metadata about features
pub trait FeatureMetadata {
    /// Get feature importance scores (if available)
    fn feature_importances(&self) -> Option<Vec<Float>> {
        None
    }

    /// Get feature types (categorical, numerical, etc.)
    fn feature_types(&self) -> Option<Vec<String>> {
        None
    }

    /// Get feature statistics (min, max, mean, std, etc.)
    fn feature_statistics(&self) -> Option<Vec<(Float, Float, Float, Float)>> {
        None
    }

    /// Get feature descriptions
    fn feature_descriptions(&self) -> Option<Vec<String>> {
        None
    }
}

/// Trait for extractors that can estimate computational complexity
pub trait ComplexityEstimator {
    /// Estimate time complexity as a string (e.g., "O(n)", "O(n^2)")
    fn time_complexity(&self) -> String {
        "O(n)".to_string()
    }

    /// Estimate space complexity as a string
    fn space_complexity(&self) -> String {
        "O(n)".to_string()
    }

    /// Estimate number of operations for given input size
    fn estimate_operations(&self, _input_size: usize) -> Option<usize> {
        None
    }
}

// ============================================================================
// Utility Traits
// ============================================================================

/// Trait for extractors that support serialization
#[cfg(feature = "serde")]
pub trait SerializableExtractor:
    FeatureExtractor + serde::Serialize + serde::de::DeserializeOwned
{
}

/// Trait for extractors that can be cloned
pub trait ClonableExtractor: FeatureExtractor {
    /// Clone the extractor
    fn clone_extractor(
        &self,
    ) -> Box<dyn FeatureExtractor<Input = Self::Input, Output = Self::Output>>;
}

// ============================================================================
// Helper Types
// ============================================================================

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Whether to validate input before extraction
    pub validate_input: bool,
    /// Whether to normalize features
    pub normalize: bool,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Number of parallel jobs (None = auto)
    pub n_jobs: Option<usize>,
    /// Whether to cache intermediate results
    pub cache: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            validate_input: true,
            normalize: false,
            parallel: false,
            n_jobs: None,
            cache: false,
        }
    }
}

/// Result of feature extraction with metadata
#[derive(Debug, Clone)]
pub struct ExtractionResult<T> {
    /// Extracted features
    pub features: T,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Number of features
    pub n_features: usize,
    /// Number of samples
    pub n_samples: usize,
    /// Extraction metadata
    pub metadata: ExtractionMetadata,
}

/// Metadata about feature extraction
#[derive(Debug, Clone, Default)]
pub struct ExtractionMetadata {
    /// Time taken for extraction (in milliseconds)
    pub duration_ms: Option<f64>,
    /// Number of features before selection/transformation
    pub original_n_features: Option<usize>,
    /// Whether features were normalized
    pub normalized: bool,
    /// Whether extraction used parallel processing
    pub parallel: bool,
    /// Custom metadata key-value pairs
    pub custom: std::collections::HashMap<String, String>,
}

impl<T> ExtractionResult<T> {
    /// Create a new extraction result
    pub fn new(features: T, n_features: usize, n_samples: usize) -> Self {
        Self {
            features,
            feature_names: None,
            n_features,
            n_samples,
            metadata: ExtractionMetadata::default(),
        }
    }

    /// Add feature names
    pub fn with_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: ExtractionMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}
