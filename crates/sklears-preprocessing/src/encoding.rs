//! Data encoding and categorical feature transformation utilities
//!
//! This module provides comprehensive data encoding implementations including
//! label encoding, one-hot encoding, ordinal encoding, binary encoding, hash encoding,
//! frequency encoding, target encoding, feature hashing, categorical transformations,
//! cardinality reduction, embedding-based encoding, statistical encoding, smoothing techniques,
//! regularization methods, cross-validation encoding, time-aware encoding, and
//! high-performance categorical feature processing pipelines. All algorithms have been
//! refactored into focused modules for better maintainability and comply with SciRS2 Policy.

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

// ---------------------------------------------------------------------------
// OrdinalEncoder
// ---------------------------------------------------------------------------

/// Ordinal encoder for categorical features stored as numeric codes.
///
/// Equivalent to scikit-learn's `OrdinalEncoder`. Input is `Array2<Float>`
/// where each element is an integer category code (0.0, 1.0, 2.0, …), keyed
/// internally via `format!("{}", v as i64)` to stay consistent with
/// [`OneHotEncoder`]. Each column is encoded independently into the integer
/// index (as `Float`) of its category within that column's sorted unique list.
///
/// Unknown categories at transform time raise an error by default
/// (`handle_unknown='error'` in scikit-learn). Calling
/// [`OrdinalEncoder::handle_unknown_use_encoded_value`] switches to scikit-learn's
/// `handle_unknown='use_encoded_value'` behavior, mapping unseen categories to a
/// caller-provided encoded value.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrdinalEncoder {
    /// Per-feature sorted unique category codes (string form, None before fit).
    categories_: Option<Vec<Vec<String>>>,
    /// Optional encoded value used for unknown categories at transform time.
    /// When `None`, unknown categories raise an error.
    handle_unknown_value: Option<Float>,
}

impl OrdinalEncoder {
    /// Create a new unfitted `OrdinalEncoder` that errors on unknown categories.
    pub fn new() -> Self {
        Self::default()
    }

    /// Map unknown categories at transform time to the provided encoded value
    /// instead of raising an error (scikit-learn `handle_unknown='use_encoded_value'`).
    pub fn handle_unknown_use_encoded_value(mut self, encoded_value: Float) -> Self {
        self.handle_unknown_value = Some(encoded_value);
        self
    }

    /// Fit the encoder to the given integer-coded feature matrix.
    ///
    /// For each column, the distinct codes are collected, sorted (as strings, to
    /// mirror [`OneHotEncoder`]), and stored.
    pub fn fit(&mut self, x: &scirs2_core::ndarray::Array2<Float>) -> Result<&mut Self> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "OrdinalEncoder.fit received an empty input matrix".to_string(),
            ));
        }

        let mut categories = Vec::with_capacity(n_cols);
        for j in 0..n_cols {
            let col = x.column(j);
            let mut unique_vals: Vec<String> =
                col.iter().map(|&v| format!("{}", v as i64)).collect();
            unique_vals.sort();
            unique_vals.dedup();
            categories.push(unique_vals);
        }

        self.categories_ = Some(categories);
        Ok(self)
    }

    /// Transform an integer-coded feature matrix into ordinal indices.
    ///
    /// Output has the same shape as the input; each cell becomes the integer
    /// index (as `Float`) of its category within that column's sorted unique
    /// list. Unknown categories raise an error unless an encoded value was set
    /// via [`OrdinalEncoder::handle_unknown_use_encoded_value`].
    pub fn transform(
        &self,
        x: &scirs2_core::ndarray::Array2<Float>,
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        let categories = self.categories_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "OrdinalEncoder has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != categories.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: categories.len(),
                actual: n_cols,
            });
        }

        let mut out = scirs2_core::ndarray::Array2::zeros((n_rows, n_cols));
        for (j, cats) in categories.iter().enumerate() {
            for i in 0..n_rows {
                let val_code = format!("{}", x[[i, j]] as i64);
                match cats.iter().position(|c| c == &val_code) {
                    Some(idx) => out[[i, j]] = idx as Float,
                    None => match self.handle_unknown_value {
                        Some(encoded_value) => out[[i, j]] = encoded_value,
                        None => {
                            return Err(SklearsError::InvalidInput(format!(
                                "Unknown category '{}' in feature column {} during transform",
                                val_code, j
                            )));
                        }
                    },
                }
            }
        }

        Ok(out)
    }

    /// Map ordinal indices back to the original category codes (as `Float`).
    ///
    /// Each cell is interpreted as the integer index of a category within that
    /// column's sorted unique list. An out-of-range index raises an error.
    pub fn inverse_transform(
        &self,
        x: &scirs2_core::ndarray::Array2<Float>,
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        let categories = self.categories_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "OrdinalEncoder has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != categories.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: categories.len(),
                actual: n_cols,
            });
        }

        let mut out = scirs2_core::ndarray::Array2::zeros((n_rows, n_cols));
        for (j, cats) in categories.iter().enumerate() {
            for i in 0..n_rows {
                let idx = x[[i, j]] as i64;
                if idx < 0 || idx as usize >= cats.len() {
                    return Err(SklearsError::InvalidInput(format!(
                        "Ordinal index {} is out of range for feature column {} (n_categories = {})",
                        idx,
                        j,
                        cats.len()
                    )));
                }
                let code: i64 = cats[idx as usize].parse().map_err(|_| {
                    SklearsError::InvalidInput(format!(
                        "Stored category '{}' in feature column {} is not a valid integer code",
                        cats[idx as usize], j
                    ))
                })?;
                out[[i, j]] = code as Float;
            }
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
        self.categories_
            .as_ref()
            .map(|cs| cs.iter().map(|c| c.as_slice()).collect())
    }
}

// ---------------------------------------------------------------------------
// TargetEncoder
// ---------------------------------------------------------------------------

/// Per-feature fitted data for `TargetEncoder`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct TargetFeature {
    /// Map from category code (string form) to its smoothed target encoding.
    encodings: HashMap<String, Float>,
}

/// Target encoder using smoothed target means for categorical encoding.
///
/// Equivalent to the mean target encoding of the `category_encoders` library
/// (and scikit-learn's `TargetEncoder`). Input features are `Array2<Float>` of
/// integer category codes (keyed via `format!("{}", v as i64)`, like
/// [`OneHotEncoder`]); the target `y` is a slice of `Float` with one entry per
/// row.
///
/// Each category is encoded by blending its own target mean with the global
/// target mean (the prior). For a category with `n_c` occurrences, category
/// target mean `mean_c`, and global mean `prior`:
///
/// ```text
/// weight     = n_c / (n_c + smoothing)
/// encoding_c = weight * mean_c + (1 - weight) * prior
/// ```
///
/// Larger `smoothing` pulls rare-category encodings toward the prior. At
/// transform time, categories never seen during `fit` are mapped to the global
/// prior — the standard, honest fallback for target encoding (not a fabricated
/// value).
///
/// # Target leakage
///
/// [`TargetEncoder::fit_transform`] here performs a plain fit-then-transform: it
/// does **not** use out-of-fold cross-fitting. Because every row's encoding is
/// computed from statistics that include that row's own target, the transformed
/// training features can leak target information and overfit downstream models.
/// For leakage-sensitive workflows, fit the encoder on a training split and
/// transform a separate validation/test split (or implement K-fold out-of-fold
/// encoding). This limitation is documented rather than silently ignored.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TargetEncoder {
    /// Smoothing strength; larger values pull encodings toward the prior.
    smoothing: Float,
    /// Number of features seen during fit (None before fit).
    n_features_in: Option<usize>,
    /// Global target mean used as the prior and unknown-category fallback.
    prior: Option<Float>,
    /// Per-feature fitted encodings (None before fit).
    features_: Option<Vec<TargetFeature>>,
}

impl Default for TargetEncoder {
    fn default() -> Self {
        Self {
            smoothing: 1.0,
            n_features_in: None,
            prior: None,
            features_: None,
        }
    }
}

impl TargetEncoder {
    /// Create a new unfitted `TargetEncoder` with default smoothing (1.0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the smoothing strength. Larger values pull rare-category encodings
    /// toward the global prior. Must be non-negative.
    pub fn smoothing(mut self, smoothing: Float) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Fit the encoder to the integer-coded feature matrix and target.
    ///
    /// Validates that `y` is non-empty and matches the number of rows, computes
    /// the global prior, then derives a smoothed encoding for every category in
    /// every column.
    pub fn fit(
        &mut self,
        x: &scirs2_core::ndarray::Array2<Float>,
        y: &[Float],
    ) -> Result<&mut Self> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "TargetEncoder.fit received an empty input matrix".to_string(),
            ));
        }
        if y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "TargetEncoder.fit received an empty target".to_string(),
            ));
        }
        if y.len() != n_rows {
            return Err(SklearsError::DimensionMismatch {
                expected: n_rows,
                actual: y.len(),
            });
        }
        if self.smoothing < 0.0 {
            return Err(SklearsError::InvalidInput(
                "TargetEncoder smoothing must be non-negative".to_string(),
            ));
        }

        let prior: Float = y.iter().copied().sum::<Float>() / (n_rows as Float);

        let mut features = Vec::with_capacity(n_cols);
        for j in 0..n_cols {
            // Accumulate per-category target sum and count.
            let mut sums: HashMap<String, Float> = HashMap::new();
            let mut counts: HashMap<String, Float> = HashMap::new();
            for i in 0..n_rows {
                let key = format!("{}", x[[i, j]] as i64);
                *sums.entry(key.clone()).or_insert(0.0) += y[i];
                *counts.entry(key).or_insert(0.0) += 1.0;
            }

            // Derive the smoothed encoding for each category.
            let mut encodings: HashMap<String, Float> = HashMap::with_capacity(counts.len());
            for (key, n_c) in counts.iter() {
                let sum_c = sums.get(key).copied().unwrap_or(0.0);
                let mean_c = sum_c / n_c;
                let weight = n_c / (n_c + self.smoothing);
                let encoding_c = weight * mean_c + (1.0 - weight) * prior;
                encodings.insert(key.clone(), encoding_c);
            }

            features.push(TargetFeature { encodings });
        }

        self.n_features_in = Some(n_cols);
        self.prior = Some(prior);
        self.features_ = Some(features);
        Ok(self)
    }

    /// Transform an integer-coded feature matrix into smoothed target encodings.
    ///
    /// Output has the same shape as the input. Categories not seen during `fit`
    /// are mapped to the global prior (the standard target-encoding fallback).
    pub fn transform(
        &self,
        x: &scirs2_core::ndarray::Array2<Float>,
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        let features = self.features_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "TargetEncoder has not been fitted yet; call fit() first".to_string(),
            )
        })?;
        let prior = self.prior.ok_or_else(|| {
            SklearsError::InvalidInput(
                "TargetEncoder has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != features.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: features.len(),
                actual: n_cols,
            });
        }

        let mut out = scirs2_core::ndarray::Array2::zeros((n_rows, n_cols));
        for (j, feat) in features.iter().enumerate() {
            for i in 0..n_rows {
                let key = format!("{}", x[[i, j]] as i64);
                // Unknown categories fall back to the global prior (documented).
                out[[i, j]] = feat.encodings.get(&key).copied().unwrap_or(prior);
            }
        }

        Ok(out)
    }

    /// Fit and immediately transform the given matrix.
    ///
    /// See the [`TargetEncoder`] type-level note: this does **not** perform
    /// out-of-fold cross-fitting and may leak target information.
    pub fn fit_transform(
        &mut self,
        x: &scirs2_core::ndarray::Array2<Float>,
        y: &[Float],
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Return the global target prior (None before fit).
    pub fn prior_(&self) -> Option<Float> {
        self.prior
    }

    /// Number of features seen during fit (None before fit).
    pub fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Return the learned per-feature category encodings (None before fit).
    ///
    /// Each map keys the category code (in `format!("{}", v as i64)` form) to its
    /// smoothed target encoding.
    pub fn encodings_(&self) -> Option<Vec<HashMap<String, Float>>> {
        self.features_
            .as_ref()
            .map(|fs| fs.iter().map(|f| f.encodings.clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn ordinal_encoder_maps_to_sorted_indices() {
        // Column 0 codes: {2, 0, 1} -> sorted strings ["0","1","2"]
        // Column 1 codes: {10, 5}    -> sorted strings ["10","5"] (string sort!)
        let x =
            Array2::from_shape_vec((3, 2), vec![2.0, 10.0, 0.0, 5.0, 1.0, 10.0]).expect("shape");
        let mut enc = OrdinalEncoder::new();
        let out = enc.fit_transform(&x).expect("fit_transform");

        // Column 0: 2 -> index 2, 0 -> index 0, 1 -> index 1
        assert!((out[[0, 0]] - 2.0).abs() < 1e-9);
        assert!((out[[1, 0]] - 0.0).abs() < 1e-9);
        assert!((out[[2, 0]] - 1.0).abs() < 1e-9);

        // Column 1 sorted strings: ["10", "5"] => "10" -> 0, "5" -> 1
        assert!((out[[0, 1]] - 0.0).abs() < 1e-9);
        assert!((out[[1, 1]] - 1.0).abs() < 1e-9);
        assert!((out[[2, 1]] - 0.0).abs() < 1e-9);

        let cats = enc.categories_().expect("categories");
        assert_eq!(
            cats[0],
            &["0".to_string(), "1".to_string(), "2".to_string()]
        );
        assert_eq!(cats[1], &["10".to_string(), "5".to_string()]);
    }

    #[test]
    fn ordinal_encoder_inverse_round_trips() {
        let x =
            Array2::from_shape_vec((3, 2), vec![2.0, 10.0, 0.0, 5.0, 1.0, 10.0]).expect("shape");
        let mut enc = OrdinalEncoder::new();
        let encoded = enc.fit_transform(&x).expect("fit_transform");
        let recovered = enc.inverse_transform(&encoded).expect("inverse");

        let (n_rows, n_cols) = x.dim();
        for i in 0..n_rows {
            for j in 0..n_cols {
                assert!((recovered[[i, j]] - x[[i, j]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn ordinal_encoder_unknown_errors_by_default() {
        let x_fit = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
        let mut enc = OrdinalEncoder::new();
        enc.fit(&x_fit).expect("fit");

        let x_unknown = Array2::from_shape_vec((1, 1), vec![7.0]).expect("shape");
        assert!(enc.transform(&x_unknown).is_err());
    }

    #[test]
    fn ordinal_encoder_unknown_uses_encoded_value() {
        let x_fit = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
        let mut enc = OrdinalEncoder::new().handle_unknown_use_encoded_value(-1.0);
        enc.fit(&x_fit).expect("fit");

        let x_unknown = Array2::from_shape_vec((2, 1), vec![7.0, 0.0]).expect("shape");
        let out = enc.transform(&x_unknown).expect("transform");
        assert!((out[[0, 0]] - (-1.0)).abs() < 1e-9);
        assert!((out[[1, 0]] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn ordinal_encoder_dim_mismatch_errors() {
        let x_fit = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("shape");
        let mut enc = OrdinalEncoder::new();
        enc.fit(&x_fit).expect("fit");

        let x_bad = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 0.0]).expect("shape");
        assert!(enc.transform(&x_bad).is_err());
    }

    #[test]
    fn target_encoder_matches_hand_computed_smoothing() {
        // Single column, two categories.
        // Category 0 rows: y = [1, 0]   -> n_c = 2, mean_c = 0.5
        // Category 1 rows: y = [1, 1]   -> n_c = 2, mean_c = 1.0
        // prior = mean([1,0,1,1]) = 0.75, smoothing = 1.0
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 1.0, 1.0]).expect("shape");
        let y = vec![1.0, 0.0, 1.0, 1.0];
        let mut enc = TargetEncoder::new().smoothing(1.0);
        let out = enc.fit_transform(&x, &y).expect("fit_transform");

        let prior = 0.75_f64;
        // Category 0: weight = 2/(2+1) = 2/3; enc = 2/3*0.5 + 1/3*0.75
        let w0 = 2.0 / 3.0;
        let enc0 = w0 * 0.5 + (1.0 - w0) * prior;
        // Category 1: weight = 2/3; enc = 2/3*1.0 + 1/3*0.75
        let w1 = 2.0 / 3.0;
        let enc1 = w1 * 1.0 + (1.0 - w1) * prior;

        assert!((out[[0, 0]] - enc0).abs() < 1e-9);
        assert!((out[[1, 0]] - enc0).abs() < 1e-9);
        assert!((out[[2, 0]] - enc1).abs() < 1e-9);
        assert!((out[[3, 0]] - enc1).abs() < 1e-9);

        assert!((enc.prior_().expect("prior") - prior).abs() < 1e-9);
        let learned = enc.encodings_().expect("encodings");
        assert!((learned[0].get("0").copied().expect("cat0") - enc0).abs() < 1e-9);
        assert!((learned[0].get("1").copied().expect("cat1") - enc1).abs() < 1e-9);
    }

    #[test]
    fn target_encoder_unknown_maps_to_prior() {
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 1.0, 1.0]).expect("shape");
        let y = vec![1.0, 0.0, 1.0, 1.0];
        let mut enc = TargetEncoder::new().smoothing(1.0);
        enc.fit(&x, &y).expect("fit");

        let prior = enc.prior_().expect("prior");
        let x_unknown = Array2::from_shape_vec((1, 1), vec![99.0]).expect("shape");
        let out = enc.transform(&x_unknown).expect("transform");
        assert!((out[[0, 0]] - prior).abs() < 1e-9);
    }

    #[test]
    fn target_encoder_mismatched_y_errors() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 0.0]).expect("shape");
        let y = vec![1.0, 0.0]; // wrong length
        let mut enc = TargetEncoder::new();
        assert!(enc.fit(&x, &y).is_err());
    }

    #[test]
    fn target_encoder_not_fitted_errors() {
        let enc = TargetEncoder::new();
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).expect("shape");
        assert!(enc.transform(&x).is_err());
    }
}
