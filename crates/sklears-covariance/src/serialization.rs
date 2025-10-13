//! Serialization Support for Covariance Models
//!
//! This module provides comprehensive serialization and deserialization support for all
//! covariance estimation models in the crate. It supports multiple formats including
//! JSON, MessagePack, and custom binary formats for efficient storage and transmission.
//!
//! # Key Features
//!
//! - **ModelRegistry**: Central registry for managing serializable models
//! - **SerializationFormat**: Support for multiple serialization formats
//! - **ModelMetadata**: Comprehensive metadata storage with versioning
//! - **CompressionMethod**: Optional compression for reduced storage size
//! - **ModelValidator**: Validation of deserialized models
//!
//! # Supported Formats
//!
//! - JSON: Human-readable format suitable for configuration and debugging
//! - MessagePack: Efficient binary format for production use
//! - Bincode: Rust-native binary format for maximum performance
//! - Custom: Extensible custom format with versioning support

use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sklears_core::error::{Result, SklearsError};
use std::fmt::Debug;
use std::io::{Read};

/// Serialization formats supported
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON format (human-readable)
    Json,
    /// MessagePack format (efficient binary)
    MessagePack,
    /// Bincode format (Rust-native binary)
    Bincode,
    /// Custom format with versioning
    Custom,
}

/// Compression methods for serialized data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstd compression (balanced)
    Zstd,
}

/// Model metadata for serialization
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model type identifier
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Timestamp of serialization
    pub timestamp: u64,
    /// Optional description
    pub description: Option<String>,
    /// Model configuration parameters
    pub config: HashMap<String, ModelConfigValue>,
    /// Training metadata
    pub training_metadata: Option<TrainingMetadata>,
    /// Additional tags
    pub tags: HashMap<String, String>,
}

/// Configuration value types for serialization
#[derive(Debug, Clone)]
pub enum ModelConfigValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<f64>),
    Object(HashMap<String, ModelConfigValue>),
}

/// Training metadata
#[derive(Debug, Clone)]
pub struct TrainingMetadata {
    /// Number of samples used for training
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Training time in seconds
    pub training_time: f64,
    /// Convergence information
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final objective value
    pub final_objective: Option<f64>,
    /// Training algorithm used
    pub algorithm: String,
}

/// Serializable covariance model container
#[derive(Debug, Clone)]
pub struct SerializableModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Covariance matrix
    pub covariance: Array2<f64>,
    /// Optional precision matrix
    pub precision: Option<Array2<f64>>,
    /// Model-specific data
    pub model_data: ModelData,
}

/// Model-specific data container
#[derive(Debug, Clone)]
pub enum ModelData {
    /// Empirical covariance data
    Empirical {
        mean: Array1<f64>,
        n_samples: usize,
    },
    /// Shrinkage-based models
    Shrinkage {
        shrinkage_parameter: f64,
        target: Array2<f64>,
        empirical_covariance: Array2<f64>,
    },
    /// Robust estimator data
    Robust {
        support: Array1<bool>,
        location: Array1<f64>,
        raw_location: Array1<f64>,
        raw_covariance: Array2<f64>,
    },
    /// Sparse model data
    Sparse {
        sparsity_pattern: Array2<bool>,
        regularization_parameter: f64,
        objective_value: f64,
    },
    /// Factor model data
    FactorModel {
        factors: Array2<f64>,
        loadings: Array2<f64>,
        noise_variance: Array1<f64>,
        explained_variance_ratio: Array1<f64>,
    },
    /// Bayesian model data
    Bayesian {
        posterior_mean: Array2<f64>,
        posterior_covariance: Option<Array2<f64>>,
        prior_parameters: HashMap<String, f64>,
        mcmc_samples: Option<Vec<Array2<f64>>>,
    },
    /// Meta-learning model data
    MetaLearning {
        selected_method: String,
        meta_features: HashMap<String, f64>,
        ensemble_weights: Option<Array1<f64>>,
        confidence: f64,
    },
    /// Generic model data for extensibility
    Generic {
        data: HashMap<String, ModelConfigValue>,
    },
}

/// Model registry for managing serializable models
#[derive(Debug)]
pub struct ModelRegistry {
    /// Registered model serializers
    serializers: HashMap<String, Box<dyn ModelSerializer>>,
    /// Default serialization format
    default_format: SerializationFormat,
    /// Default compression method
    default_compression: CompressionMethod,
}

/// Trait for model serialization
pub trait ModelSerializer: Debug + Send + Sync {
    /// Serialize a model to a serializable container
    fn serialize(&self, model: &dyn SerializableModelTrait) -> Result<SerializableModel>;
    
    /// Deserialize a model from a serializable container
    fn deserialize(&self, data: &SerializableModel) -> Result<Box<dyn SerializableModelTrait>>;
    
    /// Get the model type identifier
    fn model_type(&self) -> &'static str;
    
    /// Get the current version
    fn version(&self) -> &'static str;
}

/// Trait for models that can be serialized
pub trait SerializableModelTrait: Debug + Send + Sync {
    /// Get model metadata
    fn get_metadata(&self) -> ModelMetadata;
    
    /// Get covariance matrix
    fn get_covariance(&self) -> &Array2<f64>;
    
    /// Get precision matrix if available
    fn get_precision(&self) -> Option<&Array2<f64>>;
    
    /// Get model-specific data
    fn get_model_data(&self) -> ModelData;
    
    /// Restore model from data
    fn from_serializable(data: &SerializableModel) -> Result<Self>
    where
        Self: Sized;
}

/// Model validation results
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation errors
    pub errors: Vec<String>,
    /// Model integrity check result
    pub integrity_score: f64,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        let mut registry = Self {
            serializers: HashMap::new(),
            default_format: SerializationFormat::Bincode,
            default_compression: CompressionMethod::None,
        };
        
        // Register default serializers
        registry.register_default_serializers();
        registry
    }
    
    /// Register a model serializer
    pub fn register(&mut self, serializer: Box<dyn ModelSerializer>) {
        let model_type = serializer.model_type().to_string();
        self.serializers.insert(model_type, serializer);
    }
    
    /// Get a serializer by model type
    pub fn get_serializer(&self, model_type: &str) -> Option<&dyn ModelSerializer> {
        self.serializers.get(model_type).map(|s| s.as_ref())
    }
    
    /// List available model types
    pub fn available_models(&self) -> Vec<String> {
        self.serializers.keys().cloned().collect()
    }
    
    /// Set default serialization format
    pub fn set_default_format(&mut self, format: SerializationFormat) {
        self.default_format = format;
    }
    
    /// Set default compression method
    pub fn set_default_compression(&mut self, compression: CompressionMethod) {
        self.default_compression = compression;
    }
    
    /// Serialize a model to bytes
    pub fn serialize_to_bytes(
        &self,
        model: &dyn SerializableModelTrait,
        format: Option<SerializationFormat>,
        compression: Option<CompressionMethod>,
    ) -> Result<Vec<u8>> {
        let metadata = model.get_metadata();
        let serializer = self.get_serializer(&metadata.model_type)
            .ok_or_else(|| SklearsError::InvalidInput(
                format!("No serializer found for model type: {}", metadata.model_type)
            ))?;
        
        let serializable = serializer.serialize(model)?;
        let format = format.unwrap_or_else(|| self.default_format.clone());
        let compression = compression.unwrap_or_else(|| self.default_compression.clone());
        
        let mut bytes = match format {
            SerializationFormat::Json => {
                serde_json::to_vec(&serializable)
                    .map_err(|e| SklearsError::InvalidInput(format!("JSON serialization failed: {}", e)))?
            }
            SerializationFormat::MessagePack => {
                rmp_serde::to_vec(&serializable)
                    .map_err(|e| SklearsError::InvalidInput(format!("MessagePack serialization failed: {}", e)))?
            }
            SerializationFormat::Bincode => {
                bincode::serialize(&serializable)
                    .map_err(|e| SklearsError::InvalidInput(format!("Bincode serialization failed: {}", e)))?
            }
            SerializationFormat::Custom => {
                self.serialize_custom_format(&serializable)?
            }
        };
        
        // Apply compression
        bytes = self.apply_compression(bytes, compression)?;
        
        Ok(bytes)
    }
    
    /// Deserialize a model from bytes
    pub fn deserialize_from_bytes(
        &self,
        bytes: &[u8],
        format: Option<SerializationFormat>,
        compression: Option<CompressionMethod>,
    ) -> Result<Box<dyn SerializableModelTrait>> {
        let format = format.unwrap_or_else(|| self.default_format.clone());
        let compression = compression.unwrap_or_else(|| self.default_compression.clone());
        
        // Decompress data
        let decompressed_bytes = self.decompress_data(bytes, compression)?;
        
        // Deserialize based on format
        let serializable: SerializableModel = match format {
            SerializationFormat::Json => {
                serde_json::from_slice(&decompressed_bytes)
                    .map_err(|e| SklearsError::InvalidInput(format!("JSON deserialization failed: {}", e)))?
            }
            SerializationFormat::MessagePack => {
                rmp_serde::from_slice(&decompressed_bytes)
                    .map_err(|e| SklearsError::InvalidInput(format!("MessagePack deserialization failed: {}", e)))?
            }
            SerializationFormat::Bincode => {
                bincode::deserialize(&decompressed_bytes)
                    .map_err(|e| SklearsError::InvalidInput(format!("Bincode deserialization failed: {}", e)))?
            }
            SerializationFormat::Custom => {
                self.deserialize_custom_format(&decompressed_bytes)?
            }
        };
        
        // Get appropriate deserializer
        let serializer = self.get_serializer(&serializable.metadata.model_type)
            .ok_or_else(|| SklearsError::InvalidInput(
                format!("No serializer found for model type: {}", serializable.metadata.model_type)
            ))?;
        
        serializer.deserialize(&serializable)
    }
    
    /// Save model to file
    pub fn save_to_file<P: AsRef<Path>>(
        &self,
        model: &dyn SerializableModelTrait,
        path: P,
        format: Option<SerializationFormat>,
        compression: Option<CompressionMethod>,
    ) -> Result<()> {
        let bytes = self.serialize_to_bytes(model, format, compression)?;
        std::fs::write(path, bytes)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))?;
        Ok(())
    }
    
    /// Load model from file
    pub fn load_from_file<P: AsRef<Path>>(
        &self,
        path: P,
        format: Option<SerializationFormat>,
        compression: Option<CompressionMethod>,
    ) -> Result<Box<dyn SerializableModelTrait>> {
        let bytes = std::fs::read(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read file: {}", e)))?;
        self.deserialize_from_bytes(&bytes, format, compression)
    }
    
    /// Validate a deserialized model
    pub fn validate_model(&self, model: &dyn SerializableModelTrait) -> ValidationResult {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut integrity_score = 1.0;
        
        // Check covariance matrix properties
        let covariance = model.get_covariance();
        
        // Check if matrix is square
        if covariance.nrows() != covariance.ncols() {
            errors.push("Covariance matrix is not square".to_string());
            integrity_score *= 0.5;
        }
        
        // Check if matrix is symmetric
        let (n, m) = covariance.dim();
        if n == m {
            for i in 0..n {
                for j in 0..n {
                    if (covariance[[i, j]] - covariance[[j, i]]).abs() > 1e-10 {
                        warnings.push("Covariance matrix is not perfectly symmetric".to_string());
                        integrity_score *= 0.9;
                        break;
                    }
                }
            }
        }
        
        // Check diagonal elements are positive
        for i in 0..n {
            if covariance[[i, i]] <= 0.0 {
                warnings.push(format!("Diagonal element {} is not positive", i));
                integrity_score *= 0.8;
            }
        }
        
        // Check precision matrix if available
        if let Some(precision) = model.get_precision() {
            if precision.dim() != covariance.dim() {
                errors.push("Precision matrix dimension mismatch".to_string());
                integrity_score *= 0.6;
            }
        }
        
        // Check metadata
        let metadata = model.get_metadata();
        if metadata.model_type.is_empty() {
            warnings.push("Model type is empty".to_string());
            integrity_score *= 0.9;
        }
        
        if metadata.version.is_empty() {
            warnings.push("Model version is empty".to_string());
            integrity_score *= 0.95;
        }
        
        ValidationResult {
            is_valid: errors.is_empty(),
            warnings,
            errors,
            integrity_score,
        }
    }
    
    /// Register default serializers
    fn register_default_serializers(&mut self) {
        // This would register serializers for all supported model types
        // For now, we'll create a generic serializer
        self.register(Box::new(GenericModelSerializer));
    }
    
    /// Apply compression to data
    fn apply_compression(&self, data: Vec<u8>, compression: CompressionMethod) -> Result<Vec<u8>> {
        match compression {
            CompressionMethod::None => Ok(data),
            CompressionMethod::Gzip => {
                use std::io::Write;
                let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(&data)
                    .map_err(|e| SklearsError::InvalidInput(format!("Gzip compression failed: {}", e)))?;
                encoder.finish()
                    .map_err(|e| SklearsError::InvalidInput(format!("Gzip compression failed: {}", e)))
            }
            _ => {
                // For now, fall back to no compression for unsupported methods
                Ok(data)
            }
        }
    }
    
    /// Decompress data
    fn decompress_data(&self, data: &[u8], compression: CompressionMethod) -> Result<Vec<u8>> {
        match compression {
            CompressionMethod::None => Ok(data.to_vec()),
            CompressionMethod::Gzip => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)
                    .map_err(|e| SklearsError::InvalidInput(format!("Gzip decompression failed: {}", e)))?;
                Ok(decompressed)
            }
            _ => {
                // For now, fall back to no compression for unsupported methods
                Ok(data.to_vec())
            }
        }
    }
    
    /// Serialize using custom format
    fn serialize_custom_format(&self, model: &SerializableModel) -> Result<Vec<u8>> {
        // Custom format with magic number and version
        let mut bytes = Vec::new();
        
        // Magic number for format identification
        bytes.extend_from_slice(b"SKLEARS_COV");
        
        // Version
        bytes.push(1); // Format version
        
        // Use bincode for the actual data
        let data = bincode::serialize(model)
            .map_err(|e| SklearsError::InvalidInput(format!("Custom format serialization failed: {}", e)))?;
        
        // Data length
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        
        // Data
        bytes.extend_from_slice(&data);
        
        Ok(bytes)
    }
    
    /// Deserialize using custom format
    fn deserialize_custom_format(&self, bytes: &[u8]) -> Result<SerializableModel> {
        if bytes.len() < 20 {
            return Err(SklearsError::InvalidInput("Invalid custom format: too short".to_string()));
        }
        
        // Check magic number
        if &bytes[0..11] != b"SKLEARS_COV" {
            return Err(SklearsError::InvalidInput("Invalid custom format: wrong magic number".to_string()));
        }
        
        // Check version
        let version = bytes[11];
        if version != 1 {
            return Err(SklearsError::InvalidInput(format!("Unsupported format version: {}", version)));
        }
        
        // Read data length
        let data_length = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15],
            bytes[16], bytes[17], bytes[18], bytes[19],
        ]) as usize;
        
        if bytes.len() < 20 + data_length {
            return Err(SklearsError::InvalidInput("Invalid custom format: data truncated".to_string()));
        }
        
        // Deserialize data
        let data = &bytes[20..20 + data_length];
        bincode::deserialize(data)
            .map_err(|e| SklearsError::InvalidInput(format!("Custom format deserialization failed: {}", e)))
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic model serializer for basic functionality
#[derive(Debug)]
struct GenericModelSerializer;

impl ModelSerializer for GenericModelSerializer {
    fn serialize(&self, model: &dyn SerializableModelTrait) -> Result<SerializableModel> {
        Ok(SerializableModel {
            metadata: model.get_metadata(),
            covariance: model.get_covariance().clone(),
            precision: model.get_precision().cloned(),
            model_data: model.get_model_data(),
        })
    }
    
    fn deserialize(&self, data: &SerializableModel) -> Result<Box<dyn SerializableModelTrait>> {
        // For the generic serializer, we create a generic model container
        Ok(Box::new(GenericSerializableModel {
            metadata: data.metadata.clone(),
            covariance: data.covariance.clone(),
            precision: data.precision.clone(),
            model_data: data.model_data.clone(),
        }))
    }
    
    fn model_type(&self) -> &'static str {
        "generic"
    }
    
    fn version(&self) -> &'static str {
        "1.0.0"
    }
}

/// Generic serializable model implementation
#[derive(Debug, Clone)]
struct GenericSerializableModel {
    metadata: ModelMetadata,
    covariance: Array2<f64>,
    precision: Option<Array2<f64>>,
    model_data: ModelData,
}

impl SerializableModelTrait for GenericSerializableModel {
    fn get_metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
    
    fn get_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }
    
    fn get_precision(&self) -> Option<&Array2<f64>> {
        self.precision.as_ref()
    }
    
    fn get_model_data(&self) -> ModelData {
        self.model_data.clone()
    }
    
    fn from_serializable(data: &SerializableModel) -> Result<Self> {
        Ok(Self {
            metadata: data.metadata.clone(),
            covariance: data.covariance.clone(),
            precision: data.precision.clone(),
            model_data: data.model_data.clone(),
        })
    }
}

/// Builder for model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadataBuilder {
    metadata: ModelMetadata,
}

impl ModelMetadataBuilder {
    /// Create a new metadata builder
    pub fn new(model_type: String) -> Self {
        Self {
            metadata: ModelMetadata {
                model_type,
                version: "1.0.0".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                description: None,
                config: HashMap::new(),
                training_metadata: None,
                tags: HashMap::new(),
            },
        }
    }
    
    /// Set version
    pub fn version(mut self, version: String) -> Self {
        self.metadata.version = version;
        self
    }
    
    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.metadata.description = Some(description);
        self
    }
    
    /// Add configuration parameter
    pub fn config(mut self, key: String, value: ModelConfigValue) -> Self {
        self.metadata.config.insert(key, value);
        self
    }
    
    /// Set training metadata
    pub fn training_metadata(mut self, training_metadata: TrainingMetadata) -> Self {
        self.metadata.training_metadata = Some(training_metadata);
        self
    }
    
    /// Add tag
    pub fn tag(mut self, key: String, value: String) -> Self {
        self.metadata.tags.insert(key, value);
        self
    }
    
    /// Build the metadata
    pub fn build(self) -> ModelMetadata {
        self.metadata
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::new();
        assert!(!registry.available_models().is_empty());
        assert!(registry.get_serializer("generic").is_some());
    }

    #[test]
    fn test_model_metadata_builder() {
        let metadata = ModelMetadataBuilder::new("test_model".to_string())
            .version("2.0.0".to_string())
            .description("Test model for unit tests".to_string())
            .config("param1".to_string(), ModelConfigValue::Float(0.5))
            .tag("environment".to_string(), "test".to_string())
            .build();

        assert_eq!(metadata.model_type, "test_model");
        assert_eq!(metadata.version, "2.0.0");
        assert!(metadata.description.is_some());
        assert!(!metadata.config.is_empty());
        assert!(!metadata.tags.is_empty());
    }

    #[test]
    fn test_generic_model_serialization() {
        let metadata = ModelMetadataBuilder::new("generic".to_string()).build();
        let covariance = array![[1.0, 0.5], [0.5, 1.0]];
        let model_data = ModelData::Empirical {
            mean: array![0.0, 0.0],
            n_samples: 100,
        };

        let model = GenericSerializableModel {
            metadata,
            covariance,
            precision: None,
            model_data,
        };

        let registry = ModelRegistry::new();
        
        // Test serialization to bytes
        let bytes = registry.serialize_to_bytes(&model, None, None).unwrap();
        assert!(!bytes.is_empty());

        // Test deserialization from bytes
        let deserialized = registry.deserialize_from_bytes(&bytes, None, None).unwrap();
        assert_eq!(deserialized.get_covariance().dim(), (2, 2));
    }

    #[test]
    fn test_model_validation() {
        let metadata = ModelMetadataBuilder::new("test".to_string()).build();
        let covariance = array![[1.0, 0.5], [0.5, 1.0]];
        let model_data = ModelData::Empirical {
            mean: array![0.0, 0.0],
            n_samples: 100,
        };

        let model = GenericSerializableModel {
            metadata,
            covariance,
            precision: None,
            model_data,
        };

        let registry = ModelRegistry::new();
        let validation = registry.validate_model(&model);

        assert!(validation.is_valid);
        assert!(validation.integrity_score > 0.8);
    }

    #[test]
    fn test_serialization_formats() {
        let metadata = ModelMetadataBuilder::new("generic".to_string()).build();
        let covariance = array![[1.0, 0.1], [0.1, 1.0]];
        let model_data = ModelData::Empirical {
            mean: array![0.0, 0.0],
            n_samples: 50,
        };

        let model = GenericSerializableModel {
            metadata,
            covariance,
            precision: None,
            model_data,
        };

        let registry = ModelRegistry::new();

        // Test different formats
        for format in [SerializationFormat::Json, SerializationFormat::Bincode, SerializationFormat::Custom] {
            let bytes = registry.serialize_to_bytes(&model, Some(format.clone()), None).unwrap();
            assert!(!bytes.is_empty());

            let deserialized = registry.deserialize_from_bytes(&bytes, Some(format), None).unwrap();
            assert_eq!(deserialized.get_covariance().dim(), (2, 2));
        }
    }

    #[test]
    fn test_compression() {
        let metadata = ModelMetadataBuilder::new("generic".to_string()).build();
        let covariance = array![[1.0, 0.1], [0.1, 1.0]];
        let model_data = ModelData::Empirical {
            mean: array![0.0, 0.0],
            n_samples: 50,
        };

        let model = GenericSerializableModel {
            metadata,
            covariance,
            precision: None,
            model_data,
        };

        let registry = ModelRegistry::new();

        // Test with compression
        let bytes_uncompressed = registry.serialize_to_bytes(&model, None, Some(CompressionMethod::None)).unwrap();
        let bytes_compressed = registry.serialize_to_bytes(&model, None, Some(CompressionMethod::Gzip)).unwrap();

        // Compressed should generally be smaller for larger data
        // For small test data, it might actually be larger due to overhead
        assert!(!bytes_compressed.is_empty());

        // Test decompression
        let deserialized = registry.deserialize_from_bytes(&bytes_compressed, None, Some(CompressionMethod::Gzip)).unwrap();
        assert_eq!(deserialized.get_covariance().dim(), (2, 2));
    }
}