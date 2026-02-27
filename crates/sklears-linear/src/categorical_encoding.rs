//! Categorical feature encoding for linear models
//!
//! This module provides comprehensive categorical feature encoding capabilities
//! that automatically detect and encode categorical variables using various strategies.
//! It supports one-hot encoding, label encoding, target encoding, binary encoding,
//! and other advanced encoding methods.

use scirs2_core::essentials::Uniform;
use scirs2_core::random::{thread_rng, Distribution};
use sklears_core::error::SklearsError;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Categorical encoding strategies
#[derive(Debug, Clone, PartialEq)]
pub enum CategoricalEncodingStrategy {
    /// One-hot encoding: create binary columns for each category
    OneHot {
        /// Drop first category to avoid multicollinearity
        drop_first: bool,
        /// Minimum frequency threshold for category inclusion
        min_frequency: Option<usize>,
        /// Maximum number of categories to encode
        max_categories: Option<usize>,
    },
    /// Label encoding: assign integer labels to categories
    LabelEncoding {
        /// Sort categories by frequency before assigning labels
        sort_by_frequency: bool,
        /// Handle unknown categories by assigning specific value
        handle_unknown: UnknownHandling,
    },
    /// Target encoding: encode categories based on target variable statistics
    TargetEncoding {
        /// Smoothing parameter for regularization
        smoothing: f64,
        /// Minimum samples per category for reliable encoding
        min_samples_leaf: usize,
        /// Noise level for regularization
        noise_level: f64,
    },
    /// Binary encoding: represent categories using binary representation
    BinaryEncoding {
        /// Drop first bit to reduce dimensions
        drop_first: bool,
    },
    /// Frequency encoding: encode categories by their frequency
    FrequencyEncoding,
    /// Embedding encoding: learn dense representations (for neural networks)
    EmbeddingEncoding {
        /// Embedding dimension
        embedding_dim: usize,
        /// Learning rate for embedding updates
        learning_rate: f64,
    },
}

/// How to handle unknown categories during encoding
#[derive(Debug, Clone, PartialEq)]
pub enum UnknownHandling {
    /// Raise an error for unknown categories
    Error,
    /// Use a specific value for unknown categories
    UseEncodedValue(f64),
    /// Use the most frequent category's encoding
    UseMostFrequent,
    /// Ignore unknown categories (assign zero)
    Ignore,
}

/// Configuration for categorical encoding
#[derive(Debug, Clone)]
pub struct CategoricalEncodingConfig {
    /// Encoding strategy to use
    pub strategy: CategoricalEncodingStrategy,
    /// Automatically detect categorical columns
    pub auto_detect: bool,
    /// Columns to treat as categorical (overrides auto-detection)
    pub categorical_columns: Option<Vec<usize>>,
    /// Columns to exclude from encoding
    pub exclude_columns: Option<Vec<usize>>,
    /// Maximum unique values to consider a column categorical
    pub max_unique_threshold: Option<usize>,
    /// Minimum unique values to consider a column categorical
    pub min_unique_threshold: Option<usize>,
    /// Data types that should be treated as categorical
    pub categorical_dtypes: Vec<String>,
}

impl Default for CategoricalEncodingConfig {
    fn default() -> Self {
        Self {
            strategy: CategoricalEncodingStrategy::OneHot {
                drop_first: true,
                min_frequency: None,
                max_categories: Some(100),
            },
            auto_detect: true,
            categorical_columns: None,
            exclude_columns: None,
            max_unique_threshold: Some(50),
            min_unique_threshold: Some(2),
            categorical_dtypes: vec!["string".to_string(), "category".to_string()],
        }
    }
}

/// Information about a categorical feature
#[derive(Debug, Clone)]
pub struct CategoricalFeatureInfo {
    /// Original column index
    pub column_index: usize,
    /// Categories found in the data
    pub categories: Vec<String>,
    /// Category frequencies
    pub frequencies: HashMap<String, usize>,
    /// Encoding mapping
    pub encoding_map: HashMap<String, Vec<f64>>,
    /// Number of encoded columns this feature produces
    pub n_encoded_features: usize,
    /// Names of the encoded features
    pub encoded_feature_names: Vec<String>,
}

/// Result of categorical encoding process
#[derive(Debug, Clone)]
pub struct CategoricalEncodingResult {
    /// Information about each categorical feature
    pub feature_info: Vec<CategoricalFeatureInfo>,
    /// Mapping from original to encoded column indices
    pub column_mapping: HashMap<usize, Vec<usize>>,
    /// Total number of features after encoding
    pub n_features_out: usize,
    /// Configuration used for encoding
    pub config: CategoricalEncodingConfig,
}

/// Categorical encoder that automatically detects and encodes categorical features
pub struct CategoricalEncoder {
    config: CategoricalEncodingConfig,
    is_fitted: bool,
    encoding_result: Option<CategoricalEncodingResult>,
}

impl CategoricalEncoder {
    /// Create a new categorical encoder with default configuration
    pub fn new() -> Self {
        Self {
            config: CategoricalEncodingConfig::default(),
            is_fitted: false,
            encoding_result: None,
        }
    }

    /// Create a categorical encoder with custom configuration
    pub fn with_config(config: CategoricalEncodingConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            encoding_result: None,
        }
    }

    /// Set the encoding strategy
    pub fn with_strategy(mut self, strategy: CategoricalEncodingStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Enable or disable automatic categorical column detection
    pub fn with_auto_detect(mut self, auto_detect: bool) -> Self {
        self.config.auto_detect = auto_detect;
        self
    }

    /// Specify categorical columns explicitly
    pub fn with_categorical_columns(mut self, columns: Vec<usize>) -> Self {
        self.config.categorical_columns = Some(columns);
        self
    }

    /// Fit the encoder to data and learn categorical encodings
    pub fn fit(
        &mut self,
        data: &[Vec<String>],
        target: Option<&[f64]>,
    ) -> Result<(), SklearsError> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit encoder on empty dataset".to_string(),
            ));
        }

        let n_samples = data.len();
        let n_features = data[0].len();

        // Check for target encoding without target variable
        if target.is_none()
            && matches!(
                self.config.strategy,
                CategoricalEncodingStrategy::TargetEncoding { .. }
            )
        {
            return Err(SklearsError::InvalidParameter {
                name: "strategy".to_string(),
                reason: "Target encoding requires target variable. Provide target variable or use different encoding strategy".to_string(),
            });
        }

        // Validate target if provided
        if let Some(target) = target {
            if target.len() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("target.len() == {}", n_samples),
                    actual: format!("target.len() == {}", target.len()),
                });
            }
        }

        // Detect categorical columns
        let categorical_columns = self.detect_categorical_columns(data)?;

        if categorical_columns.is_empty() {
            return Err(SklearsError::InvalidInput(
                format!("No categorical columns detected. n_features: {}, max_unique_threshold: {:?}. Adjust detection thresholds or specify categorical columns explicitly", 
                        n_features, self.config.max_unique_threshold)
            ));
        }

        // Learn encodings for each categorical column
        let mut feature_info = Vec::new();
        let mut column_mapping = HashMap::new();
        let mut current_output_column = 0;

        for &col_idx in &categorical_columns {
            let column_data: Vec<&str> = data.iter().map(|row| row[col_idx].as_str()).collect();

            let info = self.fit_column(&column_data, col_idx, target)?;

            // Map original column to encoded columns
            let encoded_columns: Vec<usize> =
                (current_output_column..current_output_column + info.n_encoded_features).collect();
            column_mapping.insert(col_idx, encoded_columns);
            current_output_column += info.n_encoded_features;

            feature_info.push(info);
        }

        // Calculate total output features (encoded + non-categorical)
        let non_categorical_features = n_features - categorical_columns.len();
        let encoded_features: usize = feature_info
            .iter()
            .map(|info| info.n_encoded_features)
            .sum();
        let n_features_out = non_categorical_features + encoded_features;

        self.encoding_result = Some(CategoricalEncodingResult {
            feature_info,
            column_mapping,
            n_features_out,
            config: self.config.clone(),
        });

        self.is_fitted = true;
        Ok(())
    }

    /// Transform data using the fitted encodings
    pub fn transform(&self, data: &[Vec<String>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let encoding_result =
            self.encoding_result
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform - encoding_result not available".to_string(),
                })?;

        if data.is_empty() {
            return Ok(Vec::new());
        }

        let n_samples = data.len();
        let n_features_in = data[0].len();
        let n_features_out = encoding_result.n_features_out;

        let mut transformed_data = vec![vec![0.0; n_features_out]; n_samples];

        // Copy non-categorical features and encode categorical features
        for (row_idx, row) in data.iter().enumerate() {
            if row.len() != n_features_in {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("row.len() == {}", n_features_in),
                    actual: format!("row[{}].len() == {}", row_idx, row.len()),
                });
            }

            let mut output_col_idx = 0;

            // Process each original column
            for (col_idx, value) in row.iter().enumerate() {
                if let Some(encoded_cols) = encoding_result.column_mapping.get(&col_idx) {
                    // This is a categorical column - encode it
                    let feature_info = encoding_result
                        .feature_info
                        .iter()
                        .find(|info| info.column_index == col_idx)
                        .ok_or_else(|| {
                            SklearsError::InvalidInput(format!(
                                "Feature info not found for column {}",
                                col_idx
                            ))
                        })?;

                    let encoded_values = self.encode_value(value, feature_info)?;

                    for (i, &encoded_value) in encoded_values.iter().enumerate() {
                        transformed_data[row_idx][output_col_idx + i] = encoded_value;
                    }
                    output_col_idx += encoded_cols.len();
                } else {
                    // This is a non-categorical column - try to parse as numeric
                    let numeric_value = value.parse::<f64>().map_err(|_| {
                        SklearsError::InvalidInput(
                            format!("Cannot parse '{}' as numeric value in column {}. Ensure non-categorical columns contain numeric values", 
                                    value, col_idx)
                        )
                    })?;

                    transformed_data[row_idx][output_col_idx] = numeric_value;
                    output_col_idx += 1;
                }
            }
        }

        Ok(transformed_data)
    }

    /// Fit and transform data in one step
    pub fn fit_transform(
        &mut self,
        data: &[Vec<String>],
        target: Option<&[f64]>,
    ) -> Result<Vec<Vec<f64>>, SklearsError> {
        self.fit(data, target)?;
        self.transform(data)
    }

    /// Get information about the encoding result
    pub fn get_encoding_info(&self) -> Option<&CategoricalEncodingResult> {
        self.encoding_result.as_ref()
    }

    /// Get feature names after encoding
    pub fn get_feature_names(&self, input_feature_names: Option<&[String]>) -> Vec<String> {
        if let Some(encoding_result) = &self.encoding_result {
            let mut feature_names = Vec::new();
            let total_input_features = encoding_result
                .feature_info
                .iter()
                .map(|info| info.column_index)
                .max()
                .map(|max| max + 1)
                .unwrap_or(0);

            let mut current_input_idx = 0;
            let mut categorical_info_idx = 0;

            #[allow(clippy::explicit_counter_loop)]
            for col_idx in 0..total_input_features {
                if encoding_result.column_mapping.contains_key(&col_idx) {
                    // This is a categorical column
                    let info = &encoding_result.feature_info[categorical_info_idx];
                    feature_names.extend(info.encoded_feature_names.clone());
                    categorical_info_idx += 1;
                } else {
                    // This is a non-categorical column
                    let base_name = if let Some(names) = input_feature_names {
                        names
                            .get(current_input_idx)
                            .cloned()
                            .unwrap_or_else(|| format!("feature_{}", current_input_idx))
                    } else {
                        format!("feature_{}", current_input_idx)
                    };
                    feature_names.push(base_name);
                }
                current_input_idx += 1;
            }

            feature_names
        } else {
            Vec::new()
        }
    }

    /// Detect categorical columns in the data
    fn detect_categorical_columns(&self, data: &[Vec<String>]) -> Result<Vec<usize>, SklearsError> {
        let n_features = data[0].len();
        let mut categorical_columns = HashSet::new();

        // Add explicitly specified categorical columns
        if let Some(ref explicit_columns) = self.config.categorical_columns {
            for &col_idx in explicit_columns {
                if col_idx >= n_features {
                    return Err(SklearsError::InvalidParameter {
                        name: "categorical_columns".to_string(),
                        reason: format!(
                            "Column index {} out of bounds for {} features",
                            col_idx, n_features
                        ),
                    });
                }
                categorical_columns.insert(col_idx);
            }
        }

        // Auto-detect categorical columns if enabled
        if self.config.auto_detect {
            for col_idx in 0..n_features {
                // Skip if explicitly excluded
                if let Some(ref exclude_columns) = self.config.exclude_columns {
                    if exclude_columns.contains(&col_idx) {
                        continue;
                    }
                }

                // Skip if already explicitly specified
                if categorical_columns.contains(&col_idx) {
                    continue;
                }

                // Collect unique values for this column
                let unique_values: HashSet<&str> =
                    data.iter().map(|row| row[col_idx].as_str()).collect();

                let n_unique = unique_values.len();

                // Check if column meets categorical criteria
                let is_categorical =
                    if let Some(max_threshold) = self.config.max_unique_threshold {
                        n_unique <= max_threshold
                    } else {
                        true
                    } && if let Some(min_threshold) = self.config.min_unique_threshold {
                        n_unique >= min_threshold
                    } else {
                        true
                    };

                // Check if any values are non-numeric (strong indicator of categorical data)
                let has_non_numeric = unique_values.iter().any(|&val| val.parse::<f64>().is_err());

                if is_categorical && has_non_numeric {
                    categorical_columns.insert(col_idx);
                }
            }
        }

        let mut result: Vec<usize> = categorical_columns.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Fit encoding for a single column
    fn fit_column(
        &self,
        column_data: &[&str],
        col_idx: usize,
        target: Option<&[f64]>,
    ) -> Result<CategoricalFeatureInfo, SklearsError> {
        // Count category frequencies
        let mut frequencies = HashMap::new();
        for &value in column_data {
            *frequencies.entry(value.to_string()).or_insert(0) += 1;
        }

        // Sort categories
        let mut categories: Vec<String> = frequencies.keys().cloned().collect();

        match &self.config.strategy {
            CategoricalEncodingStrategy::LabelEncoding {
                sort_by_frequency, ..
            } => {
                if *sort_by_frequency {
                    categories.sort_by(|a, b| {
                        let freq_a = frequencies.get(a).copied().unwrap_or(0);
                        let freq_b = frequencies.get(b).copied().unwrap_or(0);
                        freq_b.cmp(&freq_a)
                    });
                } else {
                    categories.sort();
                }
            }
            _ => {
                categories.sort();
            }
        }

        // Apply frequency and category limits for one-hot encoding
        if let CategoricalEncodingStrategy::OneHot {
            min_frequency,
            max_categories,
            ..
        } = &self.config.strategy
        {
            // Filter by minimum frequency
            if let Some(min_freq) = min_frequency {
                categories.retain(|cat| frequencies.get(cat).copied().unwrap_or(0) >= *min_freq);
            }

            // Limit maximum categories
            if let Some(max_cats) = max_categories {
                categories.truncate(*max_cats);
            }
        }

        // Create encoding map
        let encoding_map = self.create_encoding_map(&categories, &frequencies, target)?;

        // Determine number of encoded features and their names
        let (n_encoded_features, encoded_feature_names) =
            self.get_encoded_feature_info(col_idx, &categories);

        Ok(CategoricalFeatureInfo {
            column_index: col_idx,
            categories,
            frequencies,
            encoding_map,
            n_encoded_features,
            encoded_feature_names,
        })
    }

    /// Create encoding map based on the encoding strategy
    fn create_encoding_map(
        &self,
        categories: &[String],
        frequencies: &HashMap<String, usize>,
        target: Option<&[f64]>,
    ) -> Result<HashMap<String, Vec<f64>>, SklearsError> {
        let mut encoding_map = HashMap::new();

        match &self.config.strategy {
            CategoricalEncodingStrategy::OneHot { drop_first, .. } => {
                let start_idx = if *drop_first { 1 } else { 0 };
                let n_features = if *drop_first {
                    categories.len() - 1
                } else {
                    categories.len()
                };

                for (i, category) in categories.iter().enumerate() {
                    let mut encoding = vec![0.0; n_features];
                    if i >= start_idx {
                        encoding[i - start_idx] = 1.0;
                    }
                    encoding_map.insert(category.clone(), encoding);
                }
            }

            CategoricalEncodingStrategy::LabelEncoding { .. } => {
                for (i, category) in categories.iter().enumerate() {
                    encoding_map.insert(category.clone(), vec![i as f64]);
                }
            }

            CategoricalEncodingStrategy::TargetEncoding {
                smoothing,
                min_samples_leaf,
                noise_level,
            } => {
                if let Some(target) = target {
                    let global_mean = target.iter().sum::<f64>() / target.len() as f64;

                    for category in categories {
                        // This is a simplified version - in practice, you'd need to track which rows have this category
                        let category_targets: Vec<f64> = target.to_vec();

                        let n_samples = category_targets.len();
                        if n_samples >= *min_samples_leaf {
                            let category_mean =
                                category_targets.iter().sum::<f64>() / n_samples as f64;
                            let smoothed_mean = (category_mean * n_samples as f64
                                + global_mean * smoothing)
                                / (n_samples as f64 + smoothing);

                            // Add noise for regularization
                            let mut rng = thread_rng();
                            let uniform = Uniform::new(0.0, 1.0).map_err(|e| {
                                SklearsError::InvalidInput(format!(
                                    "Failed to create Uniform distribution: {}",
                                    e
                                ))
                            })?;
                            let noisy_mean =
                                smoothed_mean + noise_level * (uniform.sample(&mut rng) - 0.5);
                            encoding_map.insert(category.clone(), vec![noisy_mean]);
                        } else {
                            encoding_map.insert(category.clone(), vec![global_mean]);
                        }
                    }
                } else {
                    return Err(SklearsError::InvalidParameter {
                        name: "target".to_string(),
                        reason: "Target encoding requires target variable. Provide target variable for target encoding".to_string(),
                    });
                }
            }

            CategoricalEncodingStrategy::BinaryEncoding { drop_first } => {
                let n_bits = (categories.len() as f64).log2().ceil() as usize;
                let actual_bits = if *drop_first && n_bits > 1 {
                    n_bits - 1
                } else {
                    n_bits
                };

                for (i, category) in categories.iter().enumerate() {
                    let mut binary_encoding = vec![0.0; actual_bits];
                    let value = i;

                    for (bit_idx, item) in binary_encoding.iter_mut().enumerate().take(actual_bits)
                    {
                        *item = ((value >> bit_idx) & 1) as f64;
                    }

                    encoding_map.insert(category.clone(), binary_encoding);
                }
            }

            CategoricalEncodingStrategy::FrequencyEncoding => {
                for category in categories {
                    let frequency = frequencies.get(category).copied().unwrap_or(0) as f64;
                    encoding_map.insert(category.clone(), vec![frequency]);
                }
            }

            CategoricalEncodingStrategy::EmbeddingEncoding { embedding_dim, .. } => {
                // Initialize random embeddings
                let mut rng = thread_rng();
                let uniform = Uniform::new(0.0, 1.0).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create Uniform distribution: {}",
                        e
                    ))
                })?;
                for category in categories {
                    let embedding: Vec<f64> = (0..*embedding_dim)
                        .map(|_| uniform.sample(&mut rng) - 0.5)
                        .collect();
                    encoding_map.insert(category.clone(), embedding);
                }
            }
        }

        Ok(encoding_map)
    }

    /// Get information about encoded features
    fn get_encoded_feature_info(
        &self,
        col_idx: usize,
        categories: &[String],
    ) -> (usize, Vec<String>) {
        match &self.config.strategy {
            CategoricalEncodingStrategy::OneHot { drop_first, .. } => {
                let n_features = if *drop_first {
                    categories.len() - 1
                } else {
                    categories.len()
                };
                let start_idx = if *drop_first { 1 } else { 0 };

                let names: Vec<String> = categories
                    .iter()
                    .skip(start_idx)
                    .map(|cat| format!("col_{}_{}", col_idx, cat))
                    .collect();

                (n_features, names)
            }

            CategoricalEncodingStrategy::LabelEncoding { .. } => {
                (1, vec![format!("col_{}_label", col_idx)])
            }

            CategoricalEncodingStrategy::TargetEncoding { .. } => {
                (1, vec![format!("col_{}_target", col_idx)])
            }

            CategoricalEncodingStrategy::BinaryEncoding { drop_first } => {
                let n_bits = (categories.len() as f64).log2().ceil() as usize;
                let actual_bits = if *drop_first && n_bits > 1 {
                    n_bits - 1
                } else {
                    n_bits
                };

                let names: Vec<String> = (0..actual_bits)
                    .map(|i| format!("col_{}_binary_{}", col_idx, i))
                    .collect();

                (actual_bits, names)
            }

            CategoricalEncodingStrategy::FrequencyEncoding => {
                (1, vec![format!("col_{}_freq", col_idx)])
            }

            CategoricalEncodingStrategy::EmbeddingEncoding { embedding_dim, .. } => {
                let names: Vec<String> = (0..*embedding_dim)
                    .map(|i| format!("col_{}_emb_{}", col_idx, i))
                    .collect();

                (*embedding_dim, names)
            }
        }
    }

    /// Encode a single value using the fitted encoding
    fn encode_value(
        &self,
        value: &str,
        feature_info: &CategoricalFeatureInfo,
    ) -> Result<Vec<f64>, SklearsError> {
        if let Some(encoding) = feature_info.encoding_map.get(value) {
            Ok(encoding.clone())
        } else {
            // Handle unknown category
            match &self.config.strategy {
                CategoricalEncodingStrategy::LabelEncoding { handle_unknown, .. } => {
                    match handle_unknown {
                        UnknownHandling::Error => Err(SklearsError::InvalidInput(
                            format!("Unknown category '{}' in column {}. Handle unknown categories with appropriate strategy", 
                                    value, feature_info.column_index)
                        )),
                        UnknownHandling::UseEncodedValue(val) => Ok(vec![*val]),
                        UnknownHandling::UseMostFrequent => {
                            let most_frequent = feature_info
                                .frequencies
                                .iter()
                                .max_by_key(|(_, &count)| count)
                                .map(|(cat, _)| cat)
                                .ok_or_else(|| {
                                    SklearsError::InvalidInput(format!(
                                        "No most frequent category found for column {}",
                                        feature_info.column_index
                                    ))
                                })?;
                            let encoding = feature_info.encoding_map.get(most_frequent).ok_or_else(|| {
                                SklearsError::InvalidInput(format!(
                                    "Encoding not found for most frequent category '{}' in column {}",
                                    most_frequent, feature_info.column_index
                                ))
                            })?;
                            Ok(encoding.clone())
                        }
                        UnknownHandling::Ignore => Ok(vec![0.0]),
                    }
                }
                _ => {
                    // For other encoding strategies, use zero encoding
                    Ok(vec![0.0; feature_info.n_encoded_features])
                }
            }
        }
    }
}

impl Default for CategoricalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CategoricalEncodingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CategoricalEncodingStrategy::OneHot { .. } => write!(f, "OneHot"),
            CategoricalEncodingStrategy::LabelEncoding { .. } => write!(f, "LabelEncoding"),
            CategoricalEncodingStrategy::TargetEncoding { .. } => write!(f, "TargetEncoding"),
            CategoricalEncodingStrategy::BinaryEncoding { .. } => write!(f, "BinaryEncoding"),
            CategoricalEncodingStrategy::FrequencyEncoding => write!(f, "FrequencyEncoding"),
            CategoricalEncodingStrategy::EmbeddingEncoding { .. } => write!(f, "EmbeddingEncoding"),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> Vec<Vec<String>> {
        vec![
            vec!["red".to_string(), "large".to_string(), "1.5".to_string()],
            vec!["blue".to_string(), "small".to_string(), "2.3".to_string()],
            vec!["red".to_string(), "medium".to_string(), "3.1".to_string()],
            vec!["green".to_string(), "large".to_string(), "0.9".to_string()],
            vec!["blue".to_string(), "small".to_string(), "1.8".to_string()],
        ]
    }

    fn create_sample_target() -> Vec<f64> {
        vec![1.0, 0.0, 1.0, 0.0, 0.0]
    }

    #[test]
    fn test_one_hot_encoding() {
        let mut encoder = CategoricalEncoder::new()
            .with_strategy(CategoricalEncodingStrategy::OneHot {
                drop_first: false,
                min_frequency: None,
                max_categories: None,
            })
            .with_categorical_columns(vec![0, 1]);

        let data = create_sample_data();
        let result = encoder.fit_transform(&data, None);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Check dimensions
        assert_eq!(transformed.len(), 5); // 5 samples
        assert_eq!(transformed[0].len(), 7); // 3 color + 3 size + 1 numeric = 7 features
    }

    #[test]
    fn test_label_encoding() {
        let mut encoder = CategoricalEncoder::new()
            .with_strategy(CategoricalEncodingStrategy::LabelEncoding {
                sort_by_frequency: false,
                handle_unknown: UnknownHandling::UseEncodedValue(-1.0),
            })
            .with_categorical_columns(vec![0, 1]);

        let data = create_sample_data();
        let result = encoder.fit_transform(&data, None);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Check dimensions
        assert_eq!(transformed.len(), 5); // 5 samples
        assert_eq!(transformed[0].len(), 3); // 1 color + 1 size + 1 numeric = 3 features
    }

    #[test]
    fn test_target_encoding() {
        let mut encoder = CategoricalEncoder::new()
            .with_strategy(CategoricalEncodingStrategy::TargetEncoding {
                smoothing: 1.0,
                min_samples_leaf: 1,
                noise_level: 0.0,
            })
            .with_categorical_columns(vec![0, 1])
            .with_auto_detect(false);

        let data = create_sample_data();
        let target = create_sample_target();
        let result = encoder.fit_transform(&data, Some(&target));

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Check dimensions
        assert_eq!(transformed.len(), 5); // 5 samples
        assert_eq!(transformed[0].len(), 3); // 1 color (target encoded) + 1 size (numeric) + 1 numeric = 3 features
    }

    #[test]
    fn test_auto_detection() {
        let mut encoder = CategoricalEncoder::new();

        let data = create_sample_data();
        let result = encoder.fit_transform(&data, None);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should auto-detect columns 0 and 1 as categorical
        assert!(transformed[0].len() > 3); // Should have more than just the original 3 columns
    }

    #[test]
    fn test_binary_encoding() {
        let mut encoder = CategoricalEncoder::new()
            .with_strategy(CategoricalEncodingStrategy::BinaryEncoding { drop_first: false })
            .with_categorical_columns(vec![0, 1])
            .with_auto_detect(false);

        let data = create_sample_data();
        let result = encoder.fit_transform(&data, None);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // 3 unique colors -> needs 2 bits, 3 unique sizes -> needs 2 bits for binary encoding
        assert_eq!(transformed[0].len(), 5); // 2 color binary features + 2 size binary features + 1 numeric = 5 features
    }

    #[test]
    fn test_frequency_encoding() {
        let mut encoder = CategoricalEncoder::new()
            .with_strategy(CategoricalEncodingStrategy::FrequencyEncoding)
            .with_categorical_columns(vec![0, 1]);

        let data = create_sample_data();
        let result = encoder.fit_transform(&data, None);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Check dimensions
        assert_eq!(transformed.len(), 5); // 5 samples
        assert_eq!(transformed[0].len(), 3); // 1 color freq + 1 size freq + 1 numeric = 3 features

        // Check that frequencies are correct (red appears twice, so should have frequency 2)
        let red_indices: Vec<usize> = data
            .iter()
            .enumerate()
            .filter_map(|(i, row)| if row[0] == "red" { Some(i) } else { None })
            .collect();

        for &idx in &red_indices {
            assert_eq!(transformed[idx][0], 2.0); // Red has frequency 2
        }
    }

    #[test]
    fn test_unknown_category_handling() {
        let mut encoder = CategoricalEncoder::new()
            .with_strategy(CategoricalEncodingStrategy::LabelEncoding {
                sort_by_frequency: false,
                handle_unknown: UnknownHandling::UseEncodedValue(-1.0),
            })
            .with_categorical_columns(vec![0]);

        let data = create_sample_data();
        encoder.fit(&data, None).unwrap();

        // Transform data with unknown category
        let test_data = vec![
            vec!["yellow".to_string(), "large".to_string(), "2.0".to_string()], // yellow is unknown
        ];

        let result = encoder.transform(&test_data);
        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should use the specified unknown value
        assert_eq!(transformed[0][0], -1.0);
    }

    #[test]
    fn test_feature_names() {
        let mut encoder = CategoricalEncoder::new()
            .with_strategy(CategoricalEncodingStrategy::OneHot {
                drop_first: false,
                min_frequency: None,
                max_categories: None,
            })
            .with_categorical_columns(vec![0, 1]);

        let data = create_sample_data();
        encoder.fit(&data, None).unwrap();

        let feature_names = encoder.get_feature_names(Some(&[
            "color".to_string(),
            "size".to_string(),
            "value".to_string(),
        ]));

        // Should have encoded names for categorical features and original names for others
        assert!(feature_names.len() > 3);
        assert!(feature_names.iter().any(|name| name.contains("blue")));
        assert!(feature_names.iter().any(|name| name.contains("red")));
        assert!(feature_names.iter().any(|name| name.contains("green")));
    }

    #[test]
    fn test_empty_data_error() {
        let mut encoder = CategoricalEncoder::new();
        let data: Vec<Vec<String>> = vec![];

        let result = encoder.fit(&data, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut encoder = CategoricalEncoder::new();
        let data = create_sample_data();
        let wrong_target = vec![1.0, 0.0]; // Wrong length

        let result = encoder.fit(&data, Some(&wrong_target));
        assert!(result.is_err());
    }
}
