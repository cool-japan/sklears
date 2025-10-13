//! Serialization support for explanation results
//!
//! This module provides serialization and deserialization capabilities
//! for explanation results, allowing them to be saved to disk and loaded later.

use crate::types::*;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Serializable wrapper for explanation results
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializableExplanationResult {
    /// Unique identifier for this explanation
    pub id: String,
    /// Explanation method used
    pub method: String,
    /// Timestamp when explanation was generated
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Feature importance values
    pub feature_importance: Vec<Float>,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// SHAP values (if computed)
    pub shap_values: Option<Vec<Vec<Float>>>,
    /// Confidence intervals
    pub confidence_intervals: Option<Vec<(Float, Float)>>,
    /// Model information
    pub model_info: ModelMetadata,
    /// Dataset information
    pub dataset_info: DatasetMetadata,
    /// Explanation configuration
    pub config: ExplanationConfiguration,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Model metadata for serialization
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelMetadata {
    /// Model type
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Training accuracy (if available)
    pub training_accuracy: Option<Float>,
    /// Validation accuracy (if available)
    pub validation_accuracy: Option<Float>,
    /// Number of parameters
    pub num_parameters: Option<usize>,
    /// Additional model-specific metadata
    pub additional_info: HashMap<String, String>,
}

/// Dataset metadata for serialization
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DatasetMetadata {
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Dataset name
    pub name: Option<String>,
    /// Feature types
    pub feature_types: Option<Vec<String>>,
    /// Target variable info
    pub target_info: Option<String>,
    /// Data statistics
    pub statistics: Option<DataStatistics>,
}

/// Data statistics
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DataStatistics {
    /// Feature means
    pub feature_means: Vec<Float>,
    /// Feature standard deviations
    pub feature_stds: Vec<Float>,
    /// Feature min values
    pub feature_mins: Vec<Float>,
    /// Feature max values
    pub feature_maxs: Vec<Float>,
    /// Missing value counts
    pub missing_counts: Vec<usize>,
}

/// Explanation configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExplanationConfiguration {
    /// Method name
    pub method_name: String,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Random seed (if used)
    pub random_seed: Option<u64>,
    /// Computation time
    pub computation_time_ms: Option<u64>,
    /// Number of samples used
    pub num_samples_used: Option<usize>,
}

/// Serialization format options
#[derive(Clone, Debug, PartialEq)]
pub enum SerializationFormat {
    /// JSON format
    Json,
    /// Binary format (MessagePack)
    Binary,
    /// CSV format (limited functionality)
    Csv,
    /// Parquet format (for large datasets)
    Parquet,
}

/// Compression options
#[derive(Clone, Debug, PartialEq)]
pub enum CompressionType {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// Zstd compression
    Zstd,
}

/// Serialization configuration
#[derive(Clone, Debug)]
pub struct SerializationConfig {
    /// Output format
    pub format: SerializationFormat,
    /// Compression type
    pub compression: CompressionType,
    /// Include raw data
    pub include_raw_data: bool,
    /// Include intermediate results
    pub include_intermediate: bool,
    /// Precision for floating point numbers
    pub float_precision: usize,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            format: SerializationFormat::Json,
            compression: CompressionType::None,
            include_raw_data: false,
            include_intermediate: false,
            float_precision: 6,
        }
    }
}

impl SerializableExplanationResult {
    /// Create a new serializable explanation result
    pub fn new(
        id: String,
        method: String,
        feature_importance: Array1<Float>,
        feature_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            id,
            method,
            timestamp: chrono::Utc::now(),
            feature_importance: feature_importance.to_vec(),
            feature_names,
            shap_values: None,
            confidence_intervals: None,
            model_info: ModelMetadata::default(),
            dataset_info: DatasetMetadata::default(),
            config: ExplanationConfiguration::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add SHAP values to the result
    pub fn with_shap_values(mut self, shap_values: Array2<Float>) -> Self {
        self.shap_values = Some(
            shap_values
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
        );
        self
    }

    /// Add confidence intervals
    pub fn with_confidence_intervals(mut self, intervals: Vec<(Float, Float)>) -> Self {
        self.confidence_intervals = Some(intervals);
        self
    }

    /// Add model metadata
    pub fn with_model_info(mut self, model_info: ModelMetadata) -> Self {
        self.model_info = model_info;
        self
    }

    /// Add dataset metadata
    pub fn with_dataset_info(mut self, dataset_info: DatasetMetadata) -> Self {
        self.dataset_info = dataset_info;
        self
    }

    /// Add configuration information
    pub fn with_config(mut self, config: ExplanationConfiguration) -> Self {
        self.config = config;
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get feature importance as Array1
    pub fn get_feature_importance(&self) -> Array1<Float> {
        Array1::from_vec(self.feature_importance.clone())
    }

    /// Get SHAP values as Array2 (if available)
    pub fn get_shap_values(&self) -> Option<Array2<Float>> {
        self.shap_values.as_ref().map(|values| {
            let rows = values.len();
            let cols = values.first().map(|row| row.len()).unwrap_or(0);
            let flat: Vec<Float> = values.iter().flatten().copied().collect();
            Array2::from_shape_vec((rows, cols), flat).unwrap()
        })
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> crate::SklResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to serialize to JSON: {}", e)))
    }

    /// Deserialize from JSON string
    pub fn from_json(json: &str) -> crate::SklResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to deserialize from JSON: {}", e))
        })
    }

    /// Save to file
    pub fn save_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        config: &SerializationConfig,
    ) -> crate::SklResult<()> {
        let content = match config.format {
            SerializationFormat::Json => self.to_json()?,
            SerializationFormat::Binary => {
                return Err(SklearsError::InvalidInput(
                    "Binary format not yet implemented".to_string(),
                ));
            }
            SerializationFormat::Csv => self.to_csv()?,
            SerializationFormat::Parquet => {
                return Err(SklearsError::InvalidInput(
                    "Parquet format not yet implemented".to_string(),
                ));
            }
        };

        // Apply compression if needed
        let final_content = match config.compression {
            CompressionType::None => content.into_bytes(),
            CompressionType::Gzip => {
                return Err(SklearsError::InvalidInput(
                    "Gzip compression not yet implemented".to_string(),
                ));
            }
            CompressionType::Lz4 => {
                return Err(SklearsError::InvalidInput(
                    "LZ4 compression not yet implemented".to_string(),
                ));
            }
            CompressionType::Zstd => {
                return Err(SklearsError::InvalidInput(
                    "Zstd compression not yet implemented".to_string(),
                ));
            }
        };

        fs::write(path, final_content)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load from file
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        config: &SerializationConfig,
    ) -> crate::SklResult<Self> {
        let content = fs::read(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read file: {}", e)))?;

        // Decompress if needed
        let decompressed_content = match config.compression {
            CompressionType::None => String::from_utf8(content).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to decode UTF-8: {}", e))
            })?,
            CompressionType::Gzip => {
                return Err(SklearsError::InvalidInput(
                    "Gzip decompression not yet implemented".to_string(),
                ));
            }
            CompressionType::Lz4 => {
                return Err(SklearsError::InvalidInput(
                    "LZ4 decompression not yet implemented".to_string(),
                ));
            }
            CompressionType::Zstd => {
                return Err(SklearsError::InvalidInput(
                    "Zstd decompression not yet implemented".to_string(),
                ));
            }
        };

        match config.format {
            SerializationFormat::Json => Self::from_json(&decompressed_content),
            SerializationFormat::Binary => Err(SklearsError::InvalidInput(
                "Binary format not yet implemented".to_string(),
            )),
            SerializationFormat::Csv => Self::from_csv(&decompressed_content),
            SerializationFormat::Parquet => Err(SklearsError::InvalidInput(
                "Parquet format not yet implemented".to_string(),
            )),
        }
    }

    /// Convert to CSV format (simplified)
    pub fn to_csv(&self) -> crate::SklResult<String> {
        let mut csv = String::new();

        // Header
        csv.push_str("feature_index,feature_name,importance\n");

        // Data rows
        for (idx, importance) in self.feature_importance.iter().enumerate() {
            let feature_name = self
                .feature_names
                .as_ref()
                .and_then(|names| names.get(idx))
                .map(|s| s.as_str())
                .unwrap_or("unknown");

            csv.push_str(&format!("{},{},{}\n", idx, feature_name, importance));
        }

        Ok(csv)
    }

    /// Load from CSV format (simplified)
    pub fn from_csv(csv: &str) -> crate::SklResult<Self> {
        let lines: Vec<&str> = csv.lines().collect();
        if lines.is_empty() {
            return Err(SklearsError::InvalidInput("Empty CSV content".to_string()));
        }

        let mut feature_importance = Vec::new();
        let mut feature_names = Vec::new();

        // Skip header and parse data
        for line in lines.iter().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                let importance: Float = parts[2].parse().map_err(|_| {
                    SklearsError::InvalidInput("Invalid importance value in CSV".to_string())
                })?;
                feature_importance.push(importance);
                feature_names.push(parts[1].to_string());
            }
        }

        Ok(Self::new(
            "csv_import".to_string(),
            "unknown".to_string(),
            Array1::from_vec(feature_importance),
            Some(feature_names),
        ))
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> SerializationSummary {
        let importance_array = self.get_feature_importance();

        SerializationSummary {
            method: self.method.clone(),
            num_features: self.feature_importance.len(),
            timestamp: self.timestamp,
            max_importance: importance_array
                .iter()
                .cloned()
                .fold(Float::NEG_INFINITY, Float::max),
            min_importance: importance_array
                .iter()
                .cloned()
                .fold(Float::INFINITY, Float::min),
            mean_importance: importance_array.mean().unwrap_or(0.0),
            std_importance: importance_array.std(0.0),
            has_shap_values: self.shap_values.is_some(),
            has_confidence_intervals: self.confidence_intervals.is_some(),
        }
    }
}

/// Summary information about an explanation
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializationSummary {
    /// Method used
    pub method: String,
    /// Number of features
    pub num_features: usize,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Maximum importance value
    pub max_importance: Float,
    /// Minimum importance value
    pub min_importance: Float,
    /// Mean importance value
    pub mean_importance: Float,
    /// Standard deviation of importance
    pub std_importance: Float,
    /// Whether SHAP values are available
    pub has_shap_values: bool,
    /// Whether confidence intervals are available
    pub has_confidence_intervals: bool,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            model_type: "unknown".to_string(),
            version: "1.0.0".to_string(),
            training_accuracy: None,
            validation_accuracy: None,
            num_parameters: None,
            additional_info: HashMap::new(),
        }
    }
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self {
            num_samples: 0,
            num_features: 0,
            name: None,
            feature_types: None,
            target_info: None,
            statistics: None,
        }
    }
}

impl Default for ExplanationConfiguration {
    fn default() -> Self {
        Self {
            method_name: "unknown".to_string(),
            parameters: HashMap::new(),
            random_seed: None,
            computation_time_ms: None,
            num_samples_used: None,
        }
    }
}

/// Batch serialization for multiple explanation results
pub struct ExplanationBatch {
    /// List of explanation results
    pub results: Vec<SerializableExplanationResult>,
    /// Batch metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ExplanationBatch {
    /// Create a new explanation batch
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Add an explanation result to the batch
    pub fn add_result(&mut self, result: SerializableExplanationResult) {
        self.results.push(result);
    }

    /// Add metadata to the batch
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Save batch to directory
    pub fn save_to_directory<P: AsRef<Path>>(
        &self,
        directory: P,
        config: &SerializationConfig,
    ) -> crate::SklResult<()> {
        let dir_path = directory.as_ref();
        fs::create_dir_all(dir_path).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create directory: {}", e))
        })?;

        // Save each result as a separate file
        for (idx, result) in self.results.iter().enumerate() {
            let filename = format!("explanation_{:03}.json", idx);
            let file_path = dir_path.join(filename);
            result.save_to_file(file_path, config)?;
        }

        // Save batch metadata
        let metadata_path = dir_path.join("batch_metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to serialize metadata: {}", e))
        })?;
        fs::write(metadata_path, metadata_json)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write metadata: {}", e)))?;

        Ok(())
    }

    /// Load batch from directory
    pub fn load_from_directory<P: AsRef<Path>>(
        directory: P,
        config: &SerializationConfig,
    ) -> crate::SklResult<Self> {
        let dir_path = directory.as_ref();

        let mut batch = ExplanationBatch::new();

        // Load all explanation files
        let entries = fs::read_dir(dir_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to read directory entry: {}", e))
            })?;

            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with("explanation_") && filename.ends_with(".json") {
                    let result = SerializableExplanationResult::load_from_file(&path, config)?;
                    batch.add_result(result);
                }
            }
        }

        // Load metadata if available
        let metadata_path = dir_path.join("batch_metadata.json");
        if metadata_path.exists() {
            let metadata_content = fs::read_to_string(metadata_path).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to read metadata: {}", e))
            })?;

            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_content)
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to parse metadata: {}", e))
                })?;

            batch.metadata = metadata;
        }

        Ok(batch)
    }

    /// Get summary of all results in the batch
    pub fn get_batch_summary(&self) -> BatchSummary {
        let summaries: Vec<SerializationSummary> = self
            .results
            .iter()
            .map(|result| result.get_summary())
            .collect();

        let methods: Vec<String> = summaries.iter().map(|s| s.method.clone()).collect();
        let unique_methods: std::collections::HashSet<String> = methods.iter().cloned().collect();

        BatchSummary {
            num_results: self.results.len(),
            methods_used: unique_methods.into_iter().collect(),
            created_at: self.created_at,
            total_features: summaries.iter().map(|s| s.num_features).sum(),
            has_metadata: !self.metadata.is_empty(),
        }
    }
}

/// Batch summary information
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BatchSummary {
    /// Number of results in the batch
    pub num_results: usize,
    /// Methods used in the batch
    pub methods_used: Vec<String>,
    /// Batch creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Total number of features across all results
    pub total_features: usize,
    /// Whether the batch has metadata
    pub has_metadata: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;
    use tempfile::tempdir;

    #[test]
    fn test_serializable_explanation_result_creation() {
        let feature_importance = array![0.5, 0.3, 0.2];
        let feature_names = vec![
            "feature1".to_string(),
            "feature2".to_string(),
            "feature3".to_string(),
        ];

        let result = SerializableExplanationResult::new(
            "test_id".to_string(),
            "permutation".to_string(),
            feature_importance.clone(),
            Some(feature_names.clone()),
        );

        assert_eq!(result.id, "test_id");
        assert_eq!(result.method, "permutation");
        assert_eq!(result.feature_importance, vec![0.5, 0.3, 0.2]);
        assert_eq!(result.feature_names, Some(feature_names));
    }

    #[test]
    fn test_shap_values_conversion() {
        let feature_importance = array![0.5, 0.3];
        let shap_values = array![[0.1, 0.2], [0.3, 0.4]];

        let result = SerializableExplanationResult::new(
            "test_id".to_string(),
            "shap".to_string(),
            feature_importance,
            None,
        )
        .with_shap_values(shap_values.clone());

        let recovered_shap = result.get_shap_values().unwrap();
        assert_eq!(recovered_shap, shap_values);
    }

    #[test]
    fn test_json_serialization() {
        let feature_importance = array![0.5, 0.3, 0.2];
        let result = SerializableExplanationResult::new(
            "test_id".to_string(),
            "permutation".to_string(),
            feature_importance,
            None,
        );

        let json = result.to_json().unwrap();
        let recovered = SerializableExplanationResult::from_json(&json).unwrap();

        assert_eq!(result.id, recovered.id);
        assert_eq!(result.method, recovered.method);
        assert_eq!(result.feature_importance, recovered.feature_importance);
    }

    #[test]
    fn test_csv_serialization() {
        let feature_importance = array![0.5, 0.3, 0.2];
        let feature_names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];

        let result = SerializableExplanationResult::new(
            "test_id".to_string(),
            "permutation".to_string(),
            feature_importance,
            Some(feature_names),
        );

        let csv = result.to_csv().unwrap();
        let recovered = SerializableExplanationResult::from_csv(&csv).unwrap();

        assert_eq!(
            result.feature_importance.len(),
            recovered.feature_importance.len()
        );
        for (a, b) in result
            .feature_importance
            .iter()
            .zip(recovered.feature_importance.iter())
        {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_file_save_and_load() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_explanation.json");

        let feature_importance = array![0.5, 0.3, 0.2];
        let result = SerializableExplanationResult::new(
            "test_id".to_string(),
            "permutation".to_string(),
            feature_importance,
            None,
        );

        let config = SerializationConfig::default();
        result.save_to_file(&file_path, &config).unwrap();

        let loaded = SerializableExplanationResult::load_from_file(&file_path, &config).unwrap();
        assert_eq!(result.id, loaded.id);
        assert_eq!(result.method, loaded.method);
        assert_eq!(result.feature_importance, loaded.feature_importance);
    }

    #[test]
    fn test_explanation_summary() {
        let feature_importance = array![0.5, 0.3, 0.8, 0.1];
        let result = SerializableExplanationResult::new(
            "test_id".to_string(),
            "permutation".to_string(),
            feature_importance,
            None,
        );

        let summary = result.get_summary();
        assert_eq!(summary.method, "permutation");
        assert_eq!(summary.num_features, 4);
        assert_eq!(summary.max_importance, 0.8);
        assert_eq!(summary.min_importance, 0.1);
        assert!(!summary.has_shap_values);
        assert!(!summary.has_confidence_intervals);
    }

    #[test]
    fn test_explanation_batch() {
        let mut batch = ExplanationBatch::new();

        let result1 = SerializableExplanationResult::new(
            "test_1".to_string(),
            "permutation".to_string(),
            array![0.5, 0.3],
            None,
        );

        let result2 = SerializableExplanationResult::new(
            "test_2".to_string(),
            "shap".to_string(),
            array![0.2, 0.8],
            None,
        );

        batch.add_result(result1);
        batch.add_result(result2);
        batch.add_metadata("experiment".to_string(), "test_run".to_string());

        let summary = batch.get_batch_summary();
        assert_eq!(summary.num_results, 2);
        assert!(summary.methods_used.contains(&"permutation".to_string()));
        assert!(summary.methods_used.contains(&"shap".to_string()));
        assert!(summary.has_metadata);
    }

    #[test]
    fn test_batch_save_and_load() {
        let temp_dir = tempdir().unwrap();

        let mut batch = ExplanationBatch::new();
        batch.add_result(SerializableExplanationResult::new(
            "test_1".to_string(),
            "permutation".to_string(),
            array![0.5, 0.3],
            None,
        ));
        batch.add_metadata("experiment".to_string(), "test_batch".to_string());

        let config = SerializationConfig::default();
        batch.save_to_directory(temp_dir.path(), &config).unwrap();

        let loaded_batch = ExplanationBatch::load_from_directory(temp_dir.path(), &config).unwrap();
        assert_eq!(loaded_batch.results.len(), 1);
        assert_eq!(
            loaded_batch.metadata.get("experiment"),
            Some(&"test_batch".to_string())
        );
    }

    #[test]
    fn test_serialization_config_default() {
        let config = SerializationConfig::default();
        assert_eq!(config.format, SerializationFormat::Json);
        assert_eq!(config.compression, CompressionType::None);
        assert!(!config.include_raw_data);
        assert_eq!(config.float_precision, 6);
    }

    #[test]
    fn test_model_metadata_default() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.model_type, "unknown");
        assert_eq!(metadata.version, "1.0.0");
        assert!(metadata.training_accuracy.is_none());
    }

    #[test]
    fn test_dataset_metadata_default() {
        let metadata = DatasetMetadata::default();
        assert_eq!(metadata.num_samples, 0);
        assert_eq!(metadata.num_features, 0);
        assert!(metadata.name.is_none());
    }

    #[test]
    fn test_with_methods() {
        let feature_importance = array![0.5, 0.3];
        let model_info = ModelMetadata {
            model_type: "linear_regression".to_string(),
            ..Default::default()
        };

        let result = SerializableExplanationResult::new(
            "test_id".to_string(),
            "permutation".to_string(),
            feature_importance,
            None,
        )
        .with_model_info(model_info.clone())
        .with_metadata("key1".to_string(), "value1".to_string());

        assert_eq!(result.model_info.model_type, "linear_regression");
        assert_eq!(result.metadata.get("key1"), Some(&"value1".to_string()));
    }
}
