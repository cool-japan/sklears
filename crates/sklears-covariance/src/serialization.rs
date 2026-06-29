//! Serialization support for covariance estimation models.
//!
//! This module provides infrastructure to convert fitted covariance estimators
//! into a format-agnostic [`SerializableModel`] and to persist that model to and
//! from JSON, a Rust-native binary format (oxicode) and MessagePack.
//!
//! The design mirrors the approach used by `sklears-manifold`: covariance and
//! precision matrices are stored as nested `Vec<Vec<f64>>` (and `Vec<f64>` for
//! the location vector) so that no dependency on the `ndarray` `serde` feature is
//! required. The binary and in-memory representations round-trip bit-for-bit; the
//! JSON text format is written exactly but, because `serde_json` is built here
//! without its `float_roundtrip` feature, parsing may differ from the original
//! `f64` by up to one ULP.
//!
//! Per-estimator [`Serializable`] implementations live in
//! `serialization_impl`.

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Serializable, format-agnostic representation of a fitted covariance model.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerializableModel {
    /// Estimator identifier (e.g. `"EmpiricalCovariance"`, `"LedoitWolf"`).
    pub algorithm: String,
    /// Estimator hyperparameters.
    pub parameters: HashMap<String, SerializableParam>,
    /// Fitted model state (matrices and vectors).
    pub state: ModelState,
    /// Metadata describing the serialized artifact.
    pub metadata: ModelMetadata,
}

/// Serializable parameter value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SerializableParam {
    /// Integer parameter.
    Int(i64),
    /// Floating-point parameter.
    Float(f64),
    /// String parameter.
    String(String),
    /// Boolean parameter.
    Bool(bool),
    /// Array of floating-point values.
    FloatArray(Vec<f64>),
    /// Array of integer values.
    IntArray(Vec<i64>),
}

/// Fitted covariance model state.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelState {
    /// Estimated covariance matrix.
    pub covariance: Option<Vec<Vec<f64>>>,
    /// Estimated precision (inverse covariance) matrix, if stored.
    pub precision: Option<Vec<Vec<f64>>>,
    /// Location (mean) vector.
    pub location: Option<Vec<f64>>,
    /// Support mask for robust estimators (e.g. MinCovDet).
    pub support: Option<Vec<bool>>,
    /// Per-sample distances for robust estimators (e.g. MinCovDet).
    pub distances: Option<Vec<f64>>,
    /// Additional estimator-specific state.
    pub custom_data: HashMap<String, SerializableParam>,
}

/// Metadata describing a serialized covariance model.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelMetadata {
    /// Unix timestamp (seconds) at which the artifact was created.
    pub timestamp: u64,
    /// Number of features the estimator was fitted on.
    pub n_features: usize,
    /// Number of samples used during fitting, if known.
    pub n_samples: Option<usize>,
    /// Version of the crate that produced the artifact.
    pub version: String,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0),
            n_features: 0,
            n_samples: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Trait implemented by fitted estimators that support serialization.
pub trait Serializable {
    /// Convert the fitted model into a [`SerializableModel`].
    fn to_serializable(&self) -> SklResult<SerializableModel>;

    /// Reconstruct a fitted model from a [`SerializableModel`].
    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized;
}

/// Supported on-disk serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Human-readable JSON.
    Json,
    /// Compact Rust-native binary (oxicode).
    Binary,
    /// MessagePack binary.
    MessagePack,
}

/// Conversions between `ndarray` containers and serializable representations.
pub struct ArrayConverter;

impl ArrayConverter {
    /// Convert an [`Array2<f64>`] into a row-major nested vector.
    pub fn array2_to_vec(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
        matrix.rows().into_iter().map(|row| row.to_vec()).collect()
    }

    /// Convert a nested vector back into an [`Array2<f64>`].
    pub fn vec_to_array2(rows: &[Vec<f64>]) -> SklResult<Array2<f64>> {
        if rows.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        let n_rows = rows.len();
        let n_cols = rows[0].len();

        for (index, row) in rows.iter().enumerate() {
            if row.len() != n_cols {
                return Err(SklearsError::InvalidInput(format!(
                    "Row {} has {} columns, expected {}",
                    index,
                    row.len(),
                    n_cols
                )));
            }
        }

        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        Array2::from_shape_vec((n_rows, n_cols), flat).map_err(|error| {
            SklearsError::InvalidInput(format!("Failed to rebuild matrix: {}", error))
        })
    }

    /// Convert an [`Array1<f64>`] into a vector.
    pub fn array1_to_vec(vector: &Array1<f64>) -> Vec<f64> {
        vector.to_vec()
    }

    /// Convert a slice back into an [`Array1<f64>`].
    pub fn vec_to_array1(values: &[f64]) -> Array1<f64> {
        Array1::from_vec(values.to_vec())
    }

    /// Convert an [`Array1<bool>`] into a vector.
    pub fn array1_bool_to_vec(vector: &Array1<bool>) -> Vec<bool> {
        vector.to_vec()
    }

    /// Convert a slice of booleans back into an [`Array1<bool>`].
    pub fn vec_to_array1_bool(values: &[bool]) -> Array1<bool> {
        Array1::from_vec(values.to_vec())
    }
}

/// Builder for assembling a [`SerializableModel`].
pub struct SerializableModelBuilder {
    algorithm: String,
    parameters: HashMap<String, SerializableParam>,
    state: ModelState,
    metadata: ModelMetadata,
}

impl SerializableModelBuilder {
    /// Create a new builder for the given estimator identifier.
    pub fn new(algorithm: &str) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            parameters: HashMap::new(),
            state: ModelState::default(),
            metadata: ModelMetadata::default(),
        }
    }

    /// Record a hyperparameter.
    pub fn parameter(mut self, name: &str, value: SerializableParam) -> Self {
        self.parameters.insert(name.to_string(), value);
        self
    }

    /// Store the covariance matrix.
    pub fn covariance(mut self, matrix: &Array2<f64>) -> Self {
        self.metadata.n_features = matrix.nrows();
        self.state.covariance = Some(ArrayConverter::array2_to_vec(matrix));
        self
    }

    /// Store the precision matrix.
    pub fn precision(mut self, matrix: &Array2<f64>) -> Self {
        self.state.precision = Some(ArrayConverter::array2_to_vec(matrix));
        self
    }

    /// Store the location (mean) vector.
    pub fn location(mut self, vector: &Array1<f64>) -> Self {
        self.state.location = Some(ArrayConverter::array1_to_vec(vector));
        self
    }

    /// Store the support mask.
    pub fn support(mut self, mask: &Array1<bool>) -> Self {
        self.state.support = Some(ArrayConverter::array1_bool_to_vec(mask));
        self
    }

    /// Store per-sample distances.
    pub fn distances(mut self, distances: &Array1<f64>) -> Self {
        self.state.distances = Some(ArrayConverter::array1_to_vec(distances));
        self
    }

    /// Record the number of samples used during fitting.
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.metadata.n_samples = Some(n_samples);
        self
    }

    /// Build the [`SerializableModel`].
    pub fn build(self) -> SerializableModel {
        SerializableModel {
            algorithm: self.algorithm,
            parameters: self.parameters,
            state: self.state,
            metadata: self.metadata,
        }
    }
}

/// File and byte-level serialization helpers.
pub struct ModelSerializer;

impl ModelSerializer {
    /// Serialize a model to bytes in the requested format.
    pub fn to_bytes(model: &SerializableModel, format: SerializationFormat) -> SklResult<Vec<u8>> {
        match format {
            SerializationFormat::Json => serde_json::to_vec_pretty(model).map_err(|error| {
                SklearsError::InvalidInput(format!("JSON serialization failed: {}", error))
            }),
            SerializationFormat::Binary => Self::to_binary(model),
            SerializationFormat::MessagePack => rmp_serde::to_vec(model).map_err(|error| {
                SklearsError::InvalidInput(format!("MessagePack serialization failed: {}", error))
            }),
        }
    }

    /// Deserialize a model from bytes in the given format.
    pub fn from_bytes(bytes: &[u8], format: SerializationFormat) -> SklResult<SerializableModel> {
        match format {
            SerializationFormat::Json => serde_json::from_slice(bytes).map_err(|error| {
                SklearsError::InvalidInput(format!("JSON deserialization failed: {}", error))
            }),
            SerializationFormat::Binary => Self::from_binary(bytes),
            SerializationFormat::MessagePack => rmp_serde::from_slice(bytes).map_err(|error| {
                SklearsError::InvalidInput(format!("MessagePack deserialization failed: {}", error))
            }),
        }
    }

    /// Serialize a model to the compact Rust-native binary format.
    pub fn to_binary(model: &SerializableModel) -> SklResult<Vec<u8>> {
        oxicode::serde::encode_to_vec(model, oxicode::config::standard()).map_err(|error| {
            SklearsError::InvalidInput(format!("Binary serialization failed: {}", error))
        })
    }

    /// Deserialize a model from the compact Rust-native binary format.
    pub fn from_binary(bytes: &[u8]) -> SklResult<SerializableModel> {
        let (model, _read) = oxicode::serde::decode_from_slice(bytes, oxicode::config::standard())
            .map_err(|error| {
                SklearsError::InvalidInput(format!("Binary deserialization failed: {}", error))
            })?;
        Ok(model)
    }

    /// Serialize a model to a JSON string.
    pub fn to_json_string(model: &SerializableModel) -> SklResult<String> {
        serde_json::to_string_pretty(model).map_err(|error| {
            SklearsError::InvalidInput(format!("JSON serialization failed: {}", error))
        })
    }

    /// Deserialize a model from a JSON string.
    pub fn from_json_string(json: &str) -> SklResult<SerializableModel> {
        serde_json::from_str(json).map_err(|error| {
            SklearsError::InvalidInput(format!("JSON deserialization failed: {}", error))
        })
    }

    /// Persist a model to a file using the requested format.
    pub fn save_to_file<P: AsRef<Path>>(
        model: &SerializableModel,
        path: P,
        format: SerializationFormat,
    ) -> SklResult<()> {
        let bytes = Self::to_bytes(model, format)?;
        let mut file = File::create(path).map_err(|error| {
            SklearsError::InvalidInput(format!("Failed to create file: {}", error))
        })?;
        file.write_all(&bytes).map_err(|error| {
            SklearsError::InvalidInput(format!("Failed to write file: {}", error))
        })?;
        Ok(())
    }

    /// Load a model from a file using the given format.
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> SklResult<SerializableModel> {
        let mut file = File::open(path).map_err(|error| {
            SklearsError::InvalidInput(format!("Failed to open file: {}", error))
        })?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|error| {
            SklearsError::InvalidInput(format!("Failed to read file: {}", error))
        })?;
        Self::from_bytes(&buffer, format)
    }
}

/// Read an [`Array2<f64>`] from a [`ModelState`] covariance slot.
pub(crate) fn require_covariance(state: &ModelState) -> SklResult<Array2<f64>> {
    let rows = state.covariance.as_ref().ok_or_else(|| {
        SklearsError::InvalidInput("Serialized model is missing the covariance matrix".to_string())
    })?;
    ArrayConverter::vec_to_array2(rows)
}

/// Read an optional precision matrix from a [`ModelState`].
pub(crate) fn optional_precision(state: &ModelState) -> SklResult<Option<Array2<f64>>> {
    match state.precision.as_ref() {
        Some(rows) => Ok(Some(ArrayConverter::vec_to_array2(rows)?)),
        None => Ok(None),
    }
}

/// Read the location vector from a [`ModelState`].
pub(crate) fn require_location(state: &ModelState) -> SklResult<Array1<f64>> {
    let values = state.location.as_ref().ok_or_else(|| {
        SklearsError::InvalidInput("Serialized model is missing the location vector".to_string())
    })?;
    Ok(ArrayConverter::vec_to_array1(values))
}

/// Read an `f64` parameter from a serialized model.
pub(crate) fn float_param(model: &SerializableModel, name: &str) -> SklResult<f64> {
    match model.parameters.get(name) {
        Some(SerializableParam::Float(value)) => Ok(*value),
        Some(SerializableParam::Int(value)) => Ok(*value as f64),
        Some(_) => Err(SklearsError::InvalidInput(format!(
            "Parameter '{}' must be a floating-point value",
            name
        ))),
        None => Err(SklearsError::InvalidInput(format!(
            "Serialized model is missing parameter '{}'",
            name
        ))),
    }
}

/// Read an `i64`-backed `usize` parameter from a serialized model.
pub(crate) fn usize_param(model: &SerializableModel, name: &str) -> SklResult<usize> {
    match model.parameters.get(name) {
        Some(SerializableParam::Int(value)) if *value >= 0 => Ok(*value as usize),
        Some(SerializableParam::Int(_)) => Err(SklearsError::InvalidInput(format!(
            "Parameter '{}' must be a non-negative integer",
            name
        ))),
        Some(_) => Err(SklearsError::InvalidInput(format!(
            "Parameter '{}' must be an integer",
            name
        ))),
        None => Err(SklearsError::InvalidInput(format!(
            "Serialized model is missing parameter '{}'",
            name
        ))),
    }
}

/// Verify that a serialized model carries the expected algorithm tag.
pub(crate) fn check_algorithm(model: &SerializableModel, expected: &str) -> SklResult<()> {
    if model.algorithm != expected {
        return Err(SklearsError::InvalidInput(format!(
            "Expected algorithm '{}' but found '{}'",
            expected, model.algorithm
        )));
    }
    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_array2_round_trip() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let encoded = ArrayConverter::array2_to_vec(&matrix);
        let decoded = ArrayConverter::vec_to_array2(&encoded).expect("matrix should rebuild");
        assert_eq!(matrix, decoded);
    }

    #[test]
    fn test_vec_to_array2_rejects_ragged_rows() {
        let ragged = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(ArrayConverter::vec_to_array2(&ragged).is_err());
    }

    #[test]
    fn test_builder_records_state() {
        let covariance = array![[2.0, 0.5], [0.5, 1.5]];
        let location = array![0.1, -0.2];
        let model = SerializableModelBuilder::new("EmpiricalCovariance")
            .parameter("store_precision", SerializableParam::Bool(true))
            .covariance(&covariance)
            .location(&location)
            .n_samples(42)
            .build();

        assert_eq!(model.algorithm, "EmpiricalCovariance");
        assert!(model.state.covariance.is_some());
        assert_eq!(model.metadata.n_features, 2);
        assert_eq!(model.metadata.n_samples, Some(42));
        assert_eq!(
            model.parameters.get("store_precision"),
            Some(&SerializableParam::Bool(true))
        );
    }

    #[test]
    fn test_serializer_byte_round_trips() {
        let covariance = array![[1.0, 0.25], [0.25, 1.0]];
        let location = array![0.0, 0.0];
        let model = SerializableModelBuilder::new("EmpiricalCovariance")
            .covariance(&covariance)
            .location(&location)
            .build();

        for format in [
            SerializationFormat::Json,
            SerializationFormat::Binary,
            SerializationFormat::MessagePack,
        ] {
            let bytes =
                ModelSerializer::to_bytes(&model, format).expect("serialization should succeed");
            assert!(!bytes.is_empty());
            let restored = ModelSerializer::from_bytes(&bytes, format)
                .expect("deserialization should succeed");
            assert_eq!(restored.algorithm, model.algorithm);
            assert_eq!(restored.metadata.n_features, model.metadata.n_features);
        }
    }
}
