//! Model Export and Serialization
//!
//! This module provides functionality to export trained models to various formats
//! for deployment and interoperability.

pub mod onnx;
pub mod serialization;

pub use onnx::*;
pub use serialization::*;

use sklears_core::error::{Result as SklResult, SklearsError};
use std::path::Path;

/// Export format for models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// ONNX format for cross-platform deployment
    ONNX,
    /// JSON format for human-readable inspection
    JSON,
    /// Binary format for efficient storage
    Binary,
    /// TensorFlow Lite (future support)
    TFLite,
}

/// Model export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Export format
    pub format: ExportFormat,
    /// Model name
    pub model_name: String,
    /// Model version
    pub version: String,
    /// Include metadata
    pub include_metadata: bool,
    /// Compress output
    pub compress: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::ONNX,
            model_name: "multiclass_model".to_string(),
            version: "1.0".to_string(),
            include_metadata: true,
            compress: false,
        }
    }
}

/// Trait for exportable models
pub trait Exportable {
    /// Export model to a file
    fn export_to_file(&self, path: &Path, config: &ExportConfig) -> SklResult<()>;

    /// Export model to bytes
    fn export_to_bytes(&self, config: &ExportConfig) -> SklResult<Vec<u8>>;

    /// Get model metadata
    fn export_metadata(&self) -> SklResult<ModelMetadata>;
}

/// Model metadata for export
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model type
    pub model_type: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of classes
    pub n_classes: usize,
    /// Creation timestamp
    pub created_at: String,
    /// Additional properties
    pub properties: std::collections::HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new metadata
    pub fn new(name: String, version: String, model_type: String) -> Self {
        Self {
            name,
            version,
            model_type,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            n_classes: 0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            properties: std::collections::HashMap::new(),
        }
    }

    /// Add a property
    pub fn add_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
    }

    /// Serialize to JSON
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> SklResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to serialize metadata: {}", e)))
    }

    #[cfg(not(feature = "serde"))]
    pub fn to_json(&self) -> SklResult<String> {
        Err(SklearsError::InvalidInput(
            "serde feature not enabled".to_string(),
        ))
    }
}

// Conditional implementations for serde
#[cfg(feature = "serde")]
use serde::Serialize;

#[cfg(feature = "serde")]
impl Serialize for ModelMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ModelMetadata", 8)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("version", &self.version)?;
        state.serialize_field("model_type", &self.model_type)?;
        state.serialize_field("input_shape", &self.input_shape)?;
        state.serialize_field("output_shape", &self.output_shape)?;
        state.serialize_field("n_classes", &self.n_classes)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("properties", &self.properties)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::ONNX);
        assert_eq!(config.model_name, "multiclass_model");
        assert!(config.include_metadata);
    }

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new(
            "test_model".to_string(),
            "1.0".to_string(),
            "OneVsRest".to_string(),
        );

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.version, "1.0");
        assert_eq!(metadata.model_type, "OneVsRest");
    }

    #[test]
    fn test_model_metadata_properties() {
        let mut metadata =
            ModelMetadata::new("test".to_string(), "1.0".to_string(), "OvR".to_string());

        metadata.add_property("optimizer".to_string(), "sgd".to_string());
        assert_eq!(
            metadata.properties.get("optimizer"),
            Some(&"sgd".to_string())
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_metadata_json_serialization() {
        let metadata = ModelMetadata::new("test".to_string(), "1.0".to_string(), "OvR".to_string());

        let json = metadata.to_json();
        assert!(json.is_ok());
    }
}
