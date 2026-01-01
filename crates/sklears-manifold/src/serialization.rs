//! Serialization support for manifold learning models
//!
//! This module provides functionality to save and load trained manifold learning models
//! to and from various formats (JSON, binary, etc.).

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Serializable representation of a trained manifold learning model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerializableModel {
    /// Algorithm type (e.g., "TSNE", "UMAP", "PCA", etc.)
    pub algorithm: String,
    /// Model hyperparameters
    pub parameters: HashMap<String, SerializableParam>,
    /// Trained model state
    pub state: ModelState,
    /// Metadata about the training process
    pub metadata: ModelMetadata,
}

/// Serializable parameter types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SerializableParam {
    /// Integer parameter
    Int(i64),
    /// Float parameter
    Float(f64),
    /// String parameter
    String(String),
    /// Boolean parameter
    Bool(bool),
    /// Array of floats
    FloatArray(Vec<f64>),
    /// Array of integers
    IntArray(Vec<i64>),
}

/// Serializable model state containing trained parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelState {
    /// Embedding matrix (if applicable)
    pub embedding: Option<Vec<Vec<f64>>>,
    /// Projection matrix (if applicable)
    pub projection_matrix: Option<Vec<Vec<f64>>>,
    /// Learned weights (if applicable)
    pub weights: Option<Vec<f64>>,
    /// Eigenvalues (if applicable)
    pub eigenvalues: Option<Vec<f64>>,
    /// Eigenvectors (if applicable)
    pub eigenvectors: Option<Vec<Vec<f64>>>,
    /// Custom state data
    pub custom_data: HashMap<String, SerializableParam>,
}

/// Metadata about the training process
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelMetadata {
    /// Training timestamp
    pub timestamp: String,
    /// Number of samples used for training
    pub n_samples: usize,
    /// Number of features in input data
    pub n_features: usize,
    /// Number of components in output
    pub n_components: usize,
    /// Training time in seconds
    pub training_time: Option<f64>,
    /// Final loss/objective value
    pub final_loss: Option<f64>,
    /// Number of iterations performed
    pub n_iterations: Option<usize>,
    /// Convergence status
    pub converged: Option<bool>,
    /// Version of the implementation
    pub version: String,
}

/// Trait for models that can be serialized
pub trait Serializable {
    /// Convert the model to a serializable format
    fn to_serializable(&self) -> SklResult<SerializableModel>;

    /// Load a model from a serializable format
    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized;
}

/// Serialization format options
#[derive(Debug, Clone, Copy)]
pub enum SerializationFormat {
    /// JSON format (human-readable)
    Json,
    /// Binary format (compact)
    Binary,
    /// MessagePack format (efficient)
    MessagePack,
}

/// Model serialization utilities
pub struct ModelSerializer;

impl ModelSerializer {
    /// Save a serializable model to a file
    pub fn save_to_file<P: AsRef<Path>>(
        model: &SerializableModel,
        path: P,
        format: SerializationFormat,
    ) -> SklResult<()> {
        let mut file = File::create(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        match format {
            SerializationFormat::Json => {
                let json_str = serde_json::to_string_pretty(model).map_err(|e| {
                    SklearsError::InvalidInput(format!("JSON serialization failed: {}", e))
                })?;
                file.write_all(json_str.as_bytes()).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to write file: {}", e))
                })?;
            }
            SerializationFormat::Binary => {
                let binary_data = oxicode::serde::encode_to_vec(model, oxicode::config::standard())
                    .map_err(|e| {
                        SklearsError::InvalidInput(format!("Binary serialization failed: {}", e))
                    })?;
                file.write_all(&binary_data).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to write file: {}", e))
                })?;
            }
            SerializationFormat::MessagePack => {
                let msgpack_data = rmp_serde::to_vec(model).map_err(|e| {
                    SklearsError::InvalidInput(format!("MessagePack serialization failed: {}", e))
                })?;
                file.write_all(&msgpack_data).map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to write file: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Load a serializable model from a file
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> SklResult<SerializableModel> {
        let mut file = File::open(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read file: {}", e)))?;

        match format {
            SerializationFormat::Json => {
                let json_str = String::from_utf8(buffer)
                    .map_err(|e| SklearsError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;
                serde_json::from_str(&json_str).map_err(|e| {
                    SklearsError::InvalidInput(format!("JSON deserialization failed: {}", e))
                })
            }
            SerializationFormat::Binary => {
                let (model, _bytes_read) =
                    oxicode::serde::decode_from_slice(&buffer, oxicode::config::standard())
                        .map_err(|e| {
                            SklearsError::InvalidInput(format!(
                                "Binary deserialization failed: {}",
                                e
                            ))
                        })?;
                Ok(model)
            }
            SerializationFormat::MessagePack => rmp_serde::from_slice(&buffer).map_err(|e| {
                SklearsError::InvalidInput(format!("MessagePack deserialization failed: {}", e))
            }),
        }
    }

    /// Save a model to a JSON string
    pub fn to_json_string(model: &SerializableModel) -> SklResult<String> {
        serde_json::to_string_pretty(model)
            .map_err(|e| SklearsError::InvalidInput(format!("JSON serialization failed: {}", e)))
    }

    /// Load a model from a JSON string
    pub fn from_json_string(json_str: &str) -> SklResult<SerializableModel> {
        serde_json::from_str(json_str)
            .map_err(|e| SklearsError::InvalidInput(format!("JSON deserialization failed: {}", e)))
    }

    /// Save a model to binary bytes
    pub fn to_binary(model: &SerializableModel) -> SklResult<Vec<u8>> {
        oxicode::serde::encode_to_vec(model, oxicode::config::standard())
            .map_err(|e| SklearsError::InvalidInput(format!("Binary serialization failed: {}", e)))
    }

    /// Load a model from binary bytes
    pub fn from_binary(data: &[u8]) -> SklResult<SerializableModel> {
        let (model, _bytes_read) =
            oxicode::serde::decode_from_slice(data, oxicode::config::standard()).map_err(|e| {
                SklearsError::InvalidInput(format!("Binary deserialization failed: {}", e))
            })?;
        Ok(model)
    }
}

/// Utility functions for converting between ndarray and serializable formats
pub struct ArrayConverter;

impl ArrayConverter {
    /// Convert ndarray Array2 to serializable format
    pub fn array2_to_vec(arr: &Array2<f64>) -> Vec<Vec<f64>> {
        arr.rows().into_iter().map(|row| row.to_vec()).collect()
    }

    /// Convert serializable format to ndarray Array2
    pub fn vec_to_array2(vec: &[Vec<f64>]) -> SklResult<Array2<f64>> {
        if vec.is_empty() {
            return Err(SklearsError::InvalidInput("Empty vector".to_string()));
        }

        let n_rows = vec.len();
        let n_cols = vec[0].len();

        // Check that all rows have the same length
        for (i, row) in vec.iter().enumerate() {
            if row.len() != n_cols {
                return Err(SklearsError::InvalidInput(format!(
                    "Row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    n_cols
                )));
            }
        }

        // Flatten the vector and create the array
        let flat_vec: Vec<f64> = vec.iter().flatten().copied().collect();
        Array2::from_shape_vec((n_rows, n_cols), flat_vec)
            .map_err(|e| SklearsError::InvalidInput(format!("Array creation failed: {}", e)))
    }

    /// Convert ndarray Array1 to serializable format
    pub fn array1_to_vec(arr: &Array1<f64>) -> Vec<f64> {
        arr.to_vec()
    }

    /// Convert serializable format to ndarray Array1
    pub fn vec_to_array1(vec: &[f64]) -> Array1<f64> {
        Array1::from_vec(vec.to_vec())
    }
}

/// Builder for creating serializable models
pub struct SerializableModelBuilder {
    algorithm: String,
    parameters: HashMap<String, SerializableParam>,
    state: ModelState,
    metadata: ModelMetadata,
}

impl SerializableModelBuilder {
    /// Create a new builder for the specified algorithm
    pub fn new(algorithm: &str) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            parameters: HashMap::new(),
            state: ModelState {
                embedding: None,
                projection_matrix: None,
                weights: None,
                eigenvalues: None,
                eigenvectors: None,
                custom_data: HashMap::new(),
            },
            metadata: ModelMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                n_samples: 0,
                n_features: 0,
                n_components: 0,
                training_time: None,
                final_loss: None,
                n_iterations: None,
                converged: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }

    /// Add a parameter to the model
    pub fn parameter(mut self, name: &str, value: SerializableParam) -> Self {
        self.parameters.insert(name.to_string(), value);
        self
    }

    /// Set the embedding matrix
    pub fn embedding(mut self, embedding: &Array2<f64>) -> Self {
        self.state.embedding = Some(ArrayConverter::array2_to_vec(embedding));
        self
    }

    /// Set the projection matrix
    pub fn projection_matrix(mut self, matrix: &Array2<f64>) -> Self {
        self.state.projection_matrix = Some(ArrayConverter::array2_to_vec(matrix));
        self
    }

    /// Set the weights
    pub fn weights(mut self, weights: &Array1<f64>) -> Self {
        self.state.weights = Some(ArrayConverter::array1_to_vec(weights));
        self
    }

    /// Set the eigenvalues
    pub fn eigenvalues(mut self, eigenvalues: &Array1<f64>) -> Self {
        self.state.eigenvalues = Some(ArrayConverter::array1_to_vec(eigenvalues));
        self
    }

    /// Set the eigenvectors
    pub fn eigenvectors(mut self, eigenvectors: &Array2<f64>) -> Self {
        self.state.eigenvectors = Some(ArrayConverter::array2_to_vec(eigenvectors));
        self
    }

    /// Set custom state data
    pub fn custom_data(mut self, key: &str, value: SerializableParam) -> Self {
        self.state.custom_data.insert(key.to_string(), value);
        self
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Build the serializable model
    pub fn build(self) -> SerializableModel {
        SerializableModel {
            algorithm: self.algorithm,
            parameters: self.parameters,
            state: self.state,
            metadata: self.metadata,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_array_conversion() {
        let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = ArrayConverter::array2_to_vec(&arr);
        let arr2 = ArrayConverter::vec_to_array2(&vec).unwrap();

        assert_eq!(arr, arr2);
    }

    #[test]
    fn test_serializable_model_builder() {
        let embedding = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let model = SerializableModelBuilder::new("TSNE")
            .parameter("n_components", SerializableParam::Int(2))
            .parameter("perplexity", SerializableParam::Float(30.0))
            .embedding(&embedding)
            .build();

        assert_eq!(model.algorithm, "TSNE");
        assert!(model.state.embedding.is_some());
        assert_eq!(model.parameters.len(), 2);
    }

    #[test]
    fn test_json_serialization() {
        let model = SerializableModelBuilder::new("PCA")
            .parameter("n_components", SerializableParam::Int(2))
            .build();

        let json_str = ModelSerializer::to_json_string(&model).unwrap();
        let loaded_model = ModelSerializer::from_json_string(&json_str).unwrap();

        assert_eq!(model.algorithm, loaded_model.algorithm);
        assert_eq!(model.parameters.len(), loaded_model.parameters.len());
    }

    #[test]
    fn test_binary_serialization() {
        let model = SerializableModelBuilder::new("UMAP")
            .parameter("n_neighbors", SerializableParam::Int(15))
            .parameter("min_dist", SerializableParam::Float(0.1))
            .build();

        let binary_data = ModelSerializer::to_binary(&model).unwrap();
        let loaded_model = ModelSerializer::from_binary(&binary_data).unwrap();

        assert_eq!(model.algorithm, loaded_model.algorithm);
        assert_eq!(model.parameters.len(), loaded_model.parameters.len());
    }
}
