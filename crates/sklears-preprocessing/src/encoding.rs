//! Data encoding and categorical feature transformation utilities
//!
//! This module provides comprehensive data encoding implementations including
//! label encoding, one-hot encoding, ordinal encoding, binary encoding, hash encoding,
//! frequency encoding, target encoding, feature hashing, categorical transformations,
//! cardinality reduction, embedding-based encoding, statistical encoding, smoothing techniques,
//! regularization methods, cross-validation encoding, time-aware encoding, and
//! high-performance categorical feature processing pipelines. All algorithms have been
//! refactored into focused modules for better maintainability and comply with SciRS2 Policy.

// FIXME: These modules are not implemented yet - commenting out to allow compilation
// // Core encoding types and base structures
// mod encoding_core;
// pub use encoding_core::{
//     EncodingProcessor, EncodingConfig, EncodingValidator, EncodingEstimator,
//     EncodingTransformer, EncodingAnalyzer, CategoricalProcessor, FeatureEncoder
// };

// // Label encoding and categorical to numerical transformation
// mod label_encoding;
// pub use label_encoding::{
//     LabelEncoder, LabelEncodingConfig, LabelEncodingValidator, CategoricalToNumerical,
//     StringEncoder, ClassEncoder, IndexMapping, LabelTransformer,
//     InverseLabelEncoder, LabelMappingAnalyzer, MultiLabelEncoder
// };

// // One-hot encoding and sparse representation
// mod onehot_encoding;
// pub use onehot_encoding::{
//     OneHotEncoder, OneHotConfig, OneHotValidator, SparseOneHot,
//     DenseOneHot, BinaryIndicator, CategoricalExpansion, OneHotTransformer,
//     SparseMatrixEncoder, OneHotOptimizer, MemoryEfficientOneHot
// };

// // Ordinal encoding and rank-based transformation
// mod ordinal_encoding;
// pub use ordinal_encoding::{
//     OrdinalEncoder, OrdinalConfig, OrdinalValidator, RankBasedEncoder,
//     OrderedCategorical, OrdinalMapping, CategoryRanking, OrdinalTransformer,
//     CustomOrderEncoder, SequentialEncoder, OrdinalOptimizer
// };

// // Binary encoding and bit-based representation
// mod binary_encoding;
// pub use binary_encoding::{
//     BinaryEncoder, BinaryEncoderConfig, BinaryValidator, BitEncoder,
//     BinaryRepresentation, BitVectorEncoder, CompactBinaryEncoder,
//     BinaryFeatureGenerator, BinaryTransformer, BinaryOptimizer
// };

// // Hash encoding and feature hashing
// mod hash_encoding;
// pub use hash_encoding::{
//     HashEncoder, HashEncoderConfig, HashValidator, FeatureHashing,
//     HashingTrick, CollisionHandling, HashFunction, MurmurHashEncoder,
//     CityHashEncoder, HashOptimizer, ConsistentHashing, HashAnalyzer
// };

// // Frequency encoding and count-based transformation
// mod frequency_encoding;
// pub use frequency_encoding::{
//     FrequencyEncoder, FrequencyEncoderConfig, FrequencyValidator, CountEncoder,
//     CategoryFrequency, FrequencyTransformer, CountBasedEncoder, RareCategoryHandler,
//     FrequencyBinning, FrequencyOptimizer, StatisticalFrequencyEncoder
// };

// // Target encoding and statistical encoding
// mod target_encoding;
// pub use target_encoding::{
//     TargetEncoder, TargetEncodingConfig, TargetValidator, MeanTargetEncoder,
//     BayesianTargetEncoder, SmoothTargetEncoder, CrossValidationTargetEncoder,
//     RegularizedTargetEncoder, TargetStatistics, TargetOptimizer, LeaveOneOutEncoder
// };

// // Embedding-based encoding and learned representations
// mod embedding_encoding;
// pub use embedding_encoding::{
//     EmbeddingEncoder, EmbeddingConfig, EmbeddingValidator, LearnedEmbedding,
//     CategoricalEmbedding, NeuralEmbedding, Word2VecEncoder, AutoencoderEmbedding,
//     EmbeddingTransformer, DimensionalityReducedEmbedding, EmbeddingOptimizer
// };

// // High-cardinality encoding and dimensionality reduction
// mod cardinality_reduction;
// pub use cardinality_reduction::{
//     CardinalityReducer, CardinalityConfig, CardinalityValidator, HighCardinalityHandler,
//     RareCategoryGrouping, TopKCategorySelector, FrequencyBasedReduction,
//     HierarchicalGrouping, CardinalityOptimizer, CategoryConsolidator
// };

// FIXME: Additional modules not implemented yet - commenting out to allow compilation
// // Time-aware encoding and temporal features
// mod temporal_encoding;
// pub use temporal_encoding::{
//     TemporalEncoder, TemporalConfig, TemporalValidator, TimeAwareEncoder,
//     SeasonalEncoder, CyclicEncoder, DateTimeEncoder, TimeSeriesEncoder,
//     TemporalFeatureExtractor, TimestampEncoder, TemporalOptimizer
// };

// // Cross-validation and robust encoding methods
// mod crossval_encoding;
// pub use crossval_encoding::{
//     CrossValidationEncoder, CVEncodingConfig, CVValidator, FoldBasedEncoder,
//     KFoldEncoder, StratifiedEncoder, TimeSeriesCVEncoder, RobustEncoder,
//     LeaveOneOutEncoder, CrossValidationOptimizer, ValidationAwareEncoder
// };

// // Smoothing and regularization techniques
// mod smoothing_techniques;
// pub use smoothing_techniques::{
//     SmoothingEncoder, SmoothingConfig, SmoothingValidator, BayesianSmoothing,
//     LaplaceSmoothing, JamesSteinSmoothing, EmpiricalBayesSmoothing,
//     AdaptiveSmoothing, SmoothingOptimizer, RegularizedSmoothing
// };

// // Performance optimization and computational efficiency
// mod performance_optimization;
// pub use performance_optimization::{
//     EncodingPerformanceOptimizer, ComputationalEfficiency, MemoryOptimizer,
//     AlgorithmicOptimizer, CacheOptimizer, ParallelEncodingProcessor
// };

// // Utilities and helper functions
// mod encoding_utilities;
// pub use encoding_utilities::{
//     EncodingUtilities, CategoricalMathUtils, EncodingAnalysisUtils, ValidationUtils,
//     ComputationalUtils, HelperFunctions, StatisticalUtils, UtilityValidator
// };

// FIXME: Re-exports commented out since modules are not implemented
// // Re-export main encoding classes for backwards compatibility
// pub use label_encoding::LabelEncoder;
// pub use onehot_encoding::OneHotEncoder;
// pub use ordinal_encoding::OrdinalEncoder;
// pub use binary_encoding::{BinaryEncoder, BinaryEncoderConfig};
// pub use hash_encoding::{HashEncoder, HashEncoderConfig};
// pub use frequency_encoding::{FrequencyEncoder, FrequencyEncoderConfig};
// pub use target_encoding::TargetEncoder;
// pub use embedding_encoding::EmbeddingEncoder;
// pub use cardinality_reduction::CardinalityReducer;

// FIXME: Re-export common configurations and types (commented out until modules are implemented)
// pub use encoding_core::EncodingConfig;
// pub use label_encoding::LabelEncodingConfig;
// pub use onehot_encoding::OneHotConfig;
// pub use ordinal_encoding::OrdinalConfig;
// pub use binary_encoding::BinaryEncoderConfig;
// pub use hash_encoding::HashEncoderConfig;
// pub use frequency_encoding::FrequencyEncoderConfig;
// pub use target_encoding::TargetEncodingConfig;
// pub use embedding_encoding::EmbeddingConfig;
// pub use temporal_encoding::TemporalConfig;

// Actual implementations of encoding functionality

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for BinaryEncoder
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BinaryEncoderConfig {
    /// Whether to drop the first binary column to avoid collinearity
    pub drop_first: bool,
    /// How to handle unknown categories during transform
    pub handle_unknown: UnknownStrategy,
    /// Whether to use base-2 encoding (true) or natural binary representation (false)
    pub use_base2: bool,
}

impl Default for BinaryEncoderConfig {
    fn default() -> Self {
        Self {
            drop_first: false,
            handle_unknown: UnknownStrategy::Error,
            use_base2: true,
        }
    }
}

/// Strategy for handling unknown categories
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UnknownStrategy {
    /// Raise an error when unknown category is encountered
    Error,
    /// Assign unknown categories to a special "unknown" encoding
    Ignore,
    /// Use all zeros for unknown categories
    Zero,
}

/// Binary encoder for high-cardinality categorical features
pub struct BinaryEncoder<State = Untrained> {
    config: BinaryEncoderConfig,
    fitted_state: Option<BinaryEncoderFitted>,
    state: PhantomData<State>,
}

/// Fitted state of BinaryEncoder
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BinaryEncoderFitted {
    config: BinaryEncoderConfig,
    /// Mapping from category to binary index
    #[allow(dead_code)]
    category_mapping: HashMap<String, usize>,
    /// Number of binary columns needed
    #[allow(dead_code)]
    n_binary_cols: usize,
    /// Categories seen during fitting
    #[allow(dead_code)]
    categories: Vec<String>,
}

impl BinaryEncoder<Untrained> {
    /// Create a new BinaryEncoder
    pub fn new() -> Self {
        Self {
            config: BinaryEncoderConfig::default(),
            fitted_state: None,
            state: PhantomData,
        }
    }

    /// Set whether to drop the first column
    pub fn drop_first(mut self, drop_first: bool) -> Self {
        self.config.drop_first = drop_first;
        self
    }

    /// Set the strategy for handling unknown categories
    pub fn handle_unknown(mut self, strategy: UnknownStrategy) -> Self {
        self.config.handle_unknown = strategy;
        self
    }

    /// Set whether to use base-2 encoding
    pub fn use_base2(mut self, use_base2: bool) -> Self {
        self.config.use_base2 = use_base2;
        self
    }
}

impl Default for BinaryEncoder<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BinaryEncoder<Untrained> {
    type Config = BinaryEncoderConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for BinaryEncoder<Trained> {
    type Config = BinaryEncoderConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.fitted_state().config
    }
}

impl BinaryEncoder<Trained> {
    fn fitted_state(&self) -> &BinaryEncoderFitted {
        self.fitted_state
            .as_ref()
            .expect("BinaryEncoder<Trained> must have fitted_state")
    }
}

impl Fit<Vec<String>, ()> for BinaryEncoder<Untrained> {
    type Fitted = BinaryEncoder<Trained>;

    fn fit(self, x: &Vec<String>, _y: &()) -> Result<Self::Fitted> {
        // Extract unique categories and sort them for deterministic encoding
        let mut categories = x.clone();
        categories.sort();
        categories.dedup();

        let n_categories = categories.len();

        // Calculate number of binary columns needed for encoding
        // log2(n_categories) rounded up gives the minimum bits needed
        let n_binary_cols = if n_categories <= 1 {
            1
        } else {
            (n_categories as f64).log2().ceil() as usize
        };

        // Create mapping from category to its index
        let category_mapping: HashMap<String, usize> = categories
            .iter()
            .enumerate()
            .map(|(i, cat)| (cat.clone(), i))
            .collect();

        // Create fitted state
        let fitted_state = BinaryEncoderFitted {
            config: self.config.clone(),
            category_mapping,
            n_binary_cols,
            categories,
        };

        // Return trained encoder with fitted state
        Ok(BinaryEncoder {
            config: self.config,
            fitted_state: Some(fitted_state),
            state: PhantomData,
        })
    }
}

/// Configuration for HashEncoder
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HashEncoderConfig {
    /// Number of hash buckets
    pub n_components: usize,
    /// Hash function to use
    pub hash_method: HashMethod,
    /// Whether to use signed hash (can have negative values)
    pub alternate_sign: bool,
}

impl Default for HashEncoderConfig {
    fn default() -> Self {
        Self {
            n_components: 32,
            hash_method: HashMethod::Md5,
            alternate_sign: true,
        }
    }
}

/// Hash function options
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HashMethod {
    /// MD5 hash function
    Md5,
    /// Simple modulo hash
    Modulo,
}

/// Hash encoder for categorical features using feature hashing
pub struct HashEncoder<State = Untrained> {
    config: HashEncoderConfig,
    state: PhantomData<State>,
}

impl HashEncoder<Untrained> {
    /// Create a new HashEncoder
    pub fn new() -> Self {
        Self {
            config: HashEncoderConfig::default(),
            state: PhantomData,
        }
    }

    /// Set the number of hash components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the hash method
    pub fn hash_method(mut self, method: HashMethod) -> Self {
        self.config.hash_method = method;
        self
    }
}

impl Default for HashEncoder<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Frequency encoder configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrequencyEncoderConfig {
    /// Whether to normalize frequencies to probabilities
    pub normalize: bool,
    /// Strategy for handling rare categories
    pub rare_strategy: RareStrategy,
    /// Threshold for considering categories as rare
    pub rare_threshold: usize,
}

impl Default for FrequencyEncoderConfig {
    fn default() -> Self {
        Self {
            normalize: false,
            rare_strategy: RareStrategy::Keep,
            rare_threshold: 1,
        }
    }
}

/// Strategy for handling rare categories
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RareStrategy {
    /// Keep rare categories as-is
    Keep,
    /// Group rare categories together
    Group,
    /// Replace rare categories with mean frequency
    MeanFrequency,
}

/// Frequency encoder transforms categories to their occurrence frequencies
pub struct FrequencyEncoder<State = Untrained> {
    config: FrequencyEncoderConfig,
    state: PhantomData<State>,
}

impl FrequencyEncoder<Untrained> {
    /// Create a new FrequencyEncoder
    pub fn new() -> Self {
        Self {
            config: FrequencyEncoderConfig::default(),
            state: PhantomData,
        }
    }

    /// Set whether to normalize frequencies
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set the rare category strategy
    pub fn rare_strategy(mut self, strategy: RareStrategy) -> Self {
        self.config.rare_strategy = strategy;
        self
    }
}

impl Default for FrequencyEncoder<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for CategoricalEmbedding
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CategoricalEmbeddingConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Learning rate for training
    pub learning_rate: Float,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
}

impl Default for CategoricalEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 50,
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 32,
        }
    }
}

/// Categorical embedding using neural network-style embeddings
pub struct CategoricalEmbedding<State = Untrained> {
    config: CategoricalEmbeddingConfig,
    state: PhantomData<State>,
}

impl CategoricalEmbedding<Untrained> {
    /// Create a new CategoricalEmbedding
    pub fn new() -> Self {
        Self {
            config: CategoricalEmbeddingConfig::default(),
            state: PhantomData,
        }
    }

    /// Set the embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.config.embedding_dim = dim;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }
}

impl Default for CategoricalEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

// Full implementations for the core encoders

/// Learned state for `LabelEncoder`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LabelEncoderFitted {
    /// Sorted unique classes in the order they are assigned indices
    classes: Vec<String>,
    /// Map from class string to integer index
    class_to_index: HashMap<String, usize>,
}

/// Label encoder: maps string (or string-convertible) labels to integers.
///
/// Equivalent to scikit-learn's `LabelEncoder`. After `fit`, each unique class
/// is assigned a contiguous integer 0..n_classes in sorted order.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LabelEncoder {
    /// Fitted state (None before fit)
    fitted: Option<LabelEncoderFitted>,
}

impl LabelEncoder {
    /// Create a new unfitted `LabelEncoder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the encoder to the given labels.
    ///
    /// Labels are sorted and deduplicated; the resulting order determines the
    /// integer encoding.
    pub fn fit(&mut self, y: &[&str]) -> Result<&mut Self> {
        let mut classes: Vec<String> = y.iter().map(|&s| s.to_string()).collect();
        classes.sort();
        classes.dedup();

        let class_to_index: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        self.fitted = Some(LabelEncoderFitted {
            classes,
            class_to_index,
        });
        Ok(self)
    }

    /// Transform labels to integer codes.
    ///
    /// Returns `Err` for any label not seen during fit.
    pub fn transform(&self, y: &[&str]) -> Result<Vec<usize>> {
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "LabelEncoder has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        y.iter()
            .map(|&label| {
                fitted.class_to_index.get(label).copied().ok_or_else(|| {
                    SklearsError::InvalidInput(format!(
                        "Unknown label '{}' encountered during transform",
                        label
                    ))
                })
            })
            .collect()
    }

    /// Map integer codes back to string labels.
    ///
    /// Returns `Err` if any index is out of range.
    pub fn inverse_transform(&self, y: &[usize]) -> Result<Vec<String>> {
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "LabelEncoder has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        y.iter()
            .map(|&idx| {
                fitted.classes.get(idx).cloned().ok_or_else(|| {
                    SklearsError::InvalidInput(format!(
                        "Index {} is out of range (n_classes = {})",
                        idx,
                        fitted.classes.len()
                    ))
                })
            })
            .collect()
    }

    /// Fit and immediately transform the given labels.
    pub fn fit_transform(&mut self, y: &[&str]) -> Result<Vec<usize>> {
        self.fit(y)?;
        self.transform(y)
    }

    /// Return the learned classes in sorted order (None before fit).
    pub fn classes_(&self) -> Option<&[String]> {
        self.fitted.as_ref().map(|f| f.classes.as_slice())
    }

    /// Number of classes (None before fit).
    pub fn n_classes(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.classes.len())
    }
}

// ---------------------------------------------------------------------------
// OneHotEncoder
// ---------------------------------------------------------------------------

/// Drop strategy for `OneHotEncoder`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DropStrategy {
    /// Keep all categories (no column dropped)
    #[default]
    None,
    /// Drop the first category per feature to avoid multicollinearity
    First,
    /// Drop the category if the feature is binary (2 categories), keep all otherwise
    IfBinary,
}

/// Per-feature fitted data for `OneHotEncoder`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct OneHotFeature {
    /// Sorted unique categories for this feature column
    categories: Vec<String>,
    /// Number of output columns contributed by this feature
    n_out_cols: usize,
}

/// One-hot encoder for categorical features stored as numeric codes.
///
/// Equivalent to scikit-learn's `OneHotEncoder`. Input is `Array2<Float>`
/// where each element is an integer category code (0.0, 1.0, 2.0, …).
/// Output is an indicator matrix.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OneHotEncoder {
    /// Drop strategy
    drop: DropStrategy,
    /// Fitted per-feature state (None before fit)
    features_: Option<Vec<OneHotFeature>>,
}

impl OneHotEncoder {
    /// Create a new `OneHotEncoder` with no dropping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the drop strategy.
    pub fn drop(mut self, strategy: DropStrategy) -> Self {
        self.drop = strategy;
        self
    }

    /// Fit the encoder to the given integer-coded feature matrix.
    ///
    /// Each column of `x` is treated as a categorical feature; unique values
    /// are collected and sorted.
    pub fn fit(&mut self, x: &scirs2_core::ndarray::Array2<Float>) -> Result<&mut Self> {
        let (_, n_cols) = x.dim();

        let mut features = Vec::with_capacity(n_cols);
        for j in 0..n_cols {
            let col = x.column(j);
            let mut unique_vals: Vec<String> =
                col.iter().map(|&v| format!("{}", v as i64)).collect();
            unique_vals.sort();
            unique_vals.dedup();

            let n_cats = unique_vals.len();
            let n_out_cols = match self.drop {
                DropStrategy::None => n_cats,
                DropStrategy::First => n_cats.saturating_sub(1),
                DropStrategy::IfBinary => {
                    if n_cats == 2 {
                        1
                    } else {
                        n_cats
                    }
                }
            };

            features.push(OneHotFeature {
                categories: unique_vals,
                n_out_cols,
            });
        }

        self.features_ = Some(features);
        Ok(self)
    }

    /// Transform integer-coded feature matrix to one-hot indicator matrix.
    ///
    /// Output shape is `(n_samples, sum_of_n_out_cols_per_feature)`.
    pub fn transform(
        &self,
        x: &scirs2_core::ndarray::Array2<Float>,
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        let features = self.features_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "OneHotEncoder has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != features.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: features.len(),
                actual: n_cols,
            });
        }

        let total_out_cols: usize = features.iter().map(|f| f.n_out_cols).sum();
        let mut out = scirs2_core::ndarray::Array2::zeros((n_rows, total_out_cols));

        let mut out_col_offset = 0_usize;
        for (j, feat) in features.iter().enumerate() {
            for i in 0..n_rows {
                let val_code = format!("{}", x[[i, j]] as i64);
                let cat_idx = feat
                    .categories
                    .iter()
                    .position(|c| c == &val_code)
                    .ok_or_else(|| {
                        SklearsError::InvalidInput(format!(
                            "Unknown category '{}' in feature column {} during transform",
                            val_code, j
                        ))
                    })?;

                // Determine which output column to set, applying drop strategy
                let effective_idx = match self.drop {
                    DropStrategy::None => Some(cat_idx),
                    DropStrategy::First => {
                        if cat_idx == 0 {
                            None // dropped
                        } else {
                            Some(cat_idx - 1)
                        }
                    }
                    DropStrategy::IfBinary => {
                        if feat.categories.len() == 2 {
                            // Only output column for the second category
                            if cat_idx == 0 {
                                None
                            } else {
                                Some(0)
                            }
                        } else {
                            Some(cat_idx)
                        }
                    }
                };

                if let Some(local_idx) = effective_idx {
                    if local_idx < feat.n_out_cols {
                        out[[i, out_col_offset + local_idx]] = 1.0;
                    }
                }
            }
            out_col_offset += feat.n_out_cols;
        }

        Ok(out)
    }

    /// Fit and immediately transform the given matrix.
    pub fn fit_transform(
        &mut self,
        x: &scirs2_core::ndarray::Array2<Float>,
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Return the learned per-feature categories (None before fit).
    pub fn categories_(&self) -> Option<Vec<&[String]>> {
        self.features_
            .as_ref()
            .map(|fs| fs.iter().map(|f| f.categories.as_slice()).collect())
    }
}

/// Ordinal encoder for categorical features with inherent ordering
#[derive(Debug, Clone, Default)]
pub struct OrdinalEncoder {
    // Placeholder - should implement proper ordinal encoding
}

/// Target encoder using target statistics for categorical encoding
#[derive(Debug, Clone, Default)]
pub struct TargetEncoder {
    // Placeholder - should implement proper target encoding
}
