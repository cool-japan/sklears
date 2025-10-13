//! Serialization support for linear models
//!
//! This module provides comprehensive serialization and deserialization support
//! for all linear model types using serde, enabling model persistence and
//! deployment in production environments.

#[cfg(feature = "serde")]
pub mod serde_support {
    use nalgebra::{DMatrix, DVector};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufReader, BufWriter};
    use std::path::Path;

    /// Serializable wrapper for DMatrix
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableMatrix {
        pub nrows: usize,
        pub ncols: usize,
        pub data: Vec<f64>,
    }

    impl From<&DMatrix<f64>> for SerializableMatrix {
        fn from(matrix: &DMatrix<f64>) -> Self {
            Self {
                nrows: matrix.nrows(),
                ncols: matrix.ncols(),
                data: matrix.iter().cloned().collect(),
            }
        }
    }

    impl From<SerializableMatrix> for DMatrix<f64> {
        fn from(ser_matrix: SerializableMatrix) -> Self {
            DMatrix::from_vec(ser_matrix.nrows, ser_matrix.ncols, ser_matrix.data)
        }
    }

    /// Serializable wrapper for DVector
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableVector {
        pub data: Vec<f64>,
    }

    impl From<&DVector<f64>> for SerializableVector {
        fn from(vector: &DVector<f64>) -> Self {
            Self {
                data: vector.iter().cloned().collect(),
            }
        }
    }

    impl From<SerializableVector> for DVector<f64> {
        fn from(ser_vector: SerializableVector) -> Self {
            DVector::from_vec(ser_vector.data)
        }
    }

    /// Serializable linear regression model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableLinearRegression {
        pub coefficients: Option<SerializableVector>,
        pub intercept: Option<f64>,
        pub config: SerializableLinearRegressionConfig,
        pub n_features: Option<usize>,
        pub model_type: String,
        pub version: String,
        pub timestamp: String,
    }

    /// Serializable linear regression configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableLinearRegressionConfig {
        pub fit_intercept: bool,
        pub solver: String,
    }

    /// Serializable Ridge regression model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableRidgeRegression {
        pub coefficients: Option<SerializableVector>,
        pub intercept: Option<f64>,
        pub alpha: f64,
        pub best_alpha: Option<f64>,
        pub cv_scores: Option<HashMap<String, Vec<f64>>>,
        pub config: SerializableRidgeConfig,
        pub n_features: Option<usize>,
        pub model_type: String,
        pub version: String,
        pub timestamp: String,
    }

    /// Serializable Ridge configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableRidgeConfig {
        pub alphas: Vec<f64>,
        pub fit_intercept: bool,
        pub cv_folds: usize,
        pub scoring: String,
        pub solver: String,
    }

    /// Serializable Lasso regression model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableLassoRegression {
        pub coefficients: Option<SerializableVector>,
        pub intercept: Option<f64>,
        pub alpha: f64,
        pub best_alpha: Option<f64>,
        pub cv_scores: Option<HashMap<String, Vec<f64>>>,
        pub config: SerializableLassoConfig,
        pub n_features: Option<usize>,
        pub sparsity_level: Option<f64>,
        pub model_type: String,
        pub version: String,
        pub timestamp: String,
    }

    /// Serializable Lasso configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableLassoConfig {
        pub alphas: Vec<f64>,
        pub fit_intercept: bool,
        pub cv_folds: usize,
        pub max_iter: usize,
        pub tolerance: f64,
        pub scoring: String,
    }

    /// Serializable multi-output regression model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableMultiOutputRegression {
        pub coefficients: Option<SerializableMatrix>,
        pub intercept: Option<SerializableVector>,
        pub target_correlations: Option<SerializableMatrix>,
        pub rank_factors: Option<(SerializableMatrix, SerializableMatrix)>,
        pub chain_order: Option<Vec<usize>>,
        pub config: SerializableMultiOutputConfig,
        pub n_features: Option<usize>,
        pub n_targets: Option<usize>,
        pub training_loss: Option<f64>,
        pub model_type: String,
        pub version: String,
        pub timestamp: String,
    }

    /// Serializable multi-output configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableMultiOutputConfig {
        pub alpha: f64,
        pub l1_ratio: f64,
        pub strategy: String,
        pub max_iter: usize,
        pub tolerance: f64,
        pub rank: Option<usize>,
        pub model_correlations: bool,
        pub fit_intercept: bool,
    }

    /// Serializable constrained optimization problem
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableConstrainedOptimization {
        pub solution: Option<SerializableVector>,
        pub objective_value: Option<f64>,
        pub lambda_inequality: Option<SerializableVector>,
        pub lambda_equality: Option<SerializableVector>,
        pub active_constraints: Option<Vec<usize>>,
        pub constraint_violation: Option<f64>,
        pub converged: Option<bool>,
        pub iterations: Option<usize>,
        pub config: SerializableConstrainedConfig,
        pub model_type: String,
        pub version: String,
        pub timestamp: String,
    }

    /// Serializable constrained optimization configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableConstrainedConfig {
        pub max_iter: usize,
        pub tolerance: f64,
        pub feasibility_tolerance: f64,
        pub barrier_parameter: f64,
        pub barrier_reduction: f64,
        pub step_size: f64,
        pub backtrack_factor: f64,
        pub verbose: bool,
    }

    /// Model metadata for versioning and compatibility
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelMetadata {
        pub model_type: String,
        pub version: String,
        pub library_version: String,
        pub timestamp: String,
        pub training_info: Option<TrainingInfo>,
        pub performance_metrics: Option<PerformanceMetrics>,
        pub feature_info: Option<FeatureInfo>,
    }

    /// Training information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrainingInfo {
        pub n_samples: usize,
        pub n_features: usize,
        pub n_targets: Option<usize>,
        pub training_time_seconds: Option<f64>,
        pub convergence_info: Option<ConvergenceInfo>,
    }

    /// Performance metrics
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerformanceMetrics {
        pub training_score: Option<f64>,
        pub validation_score: Option<f64>,
        pub cross_validation_scores: Option<Vec<f64>>,
        pub feature_importance: Option<Vec<f64>>,
    }

    /// Feature information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FeatureInfo {
        pub feature_names: Option<Vec<String>>,
        pub feature_types: Option<Vec<String>>,
        pub scaling_info: Option<ScalingInfo>,
    }

    /// Scaling information for preprocessing
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ScalingInfo {
        pub method: String,
        pub means: Option<Vec<f64>>,
        pub stds: Option<Vec<f64>>,
        pub mins: Option<Vec<f64>>,
        pub maxs: Option<Vec<f64>>,
    }

    /// Convergence information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConvergenceInfo {
        pub converged: bool,
        pub n_iterations: usize,
        pub final_loss: f64,
        pub tolerance_achieved: f64,
    }

    /// Serialization format enum
    #[derive(Debug, Clone, Copy)]
    pub enum SerializationFormat {
        Json,
        Bincode,
        MessagePack,
    }

    /// Model serialization and deserialization utilities
    pub struct ModelSerializer;

    impl ModelSerializer {
        /// Serialize a model to JSON format
        pub fn to_json<T: Serialize>(model: &T) -> Result<String, serde_json::Error> {
            serde_json::to_string_pretty(model)
        }

        /// Deserialize a model from JSON format
        pub fn from_json<T: for<'de> Deserialize<'de>>(json: &str) -> Result<T, serde_json::Error> {
            serde_json::from_str(json)
        }

        /// Serialize a model to binary format using bincode
        pub fn to_binary<T: Serialize>(model: &T) -> Result<Vec<u8>, bincode::Error> {
            bincode::serialize(model)
        }

        /// Deserialize a model from binary format using bincode
        pub fn from_binary<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T, bincode::Error> {
            bincode::deserialize(data)
        }

        /// Save a model to file
        pub fn save_to_file<T: Serialize, P: AsRef<Path>>(
            model: &T,
            path: P,
            format: SerializationFormat,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);

            match format {
                SerializationFormat::Json => {
                    serde_json::to_writer_pretty(writer, model)?;
                }
                SerializationFormat::Bincode => {
                    bincode::serialize_into(writer, model)?;
                }
                SerializationFormat::MessagePack => {
                    rmp_serde::encode::write(&mut writer, model)?;
                }
            }

            Ok(())
        }

        /// Load a model from file
        pub fn load_from_file<T: for<'de> Deserialize<'de>, P: AsRef<Path>>(
            path: P,
            format: SerializationFormat,
        ) -> Result<T, Box<dyn std::error::Error>> {
            let file = File::open(path)?;
            let reader = BufReader::new(file);

            let model = match format {
                SerializationFormat::Json => serde_json::from_reader(reader)?,
                SerializationFormat::Bincode => bincode::deserialize_from(reader)?,
                SerializationFormat::MessagePack => rmp_serde::decode::from_read(reader)?,
            };

            Ok(model)
        }

        /// Validate model compatibility
        pub fn validate_compatibility(
            metadata: &ModelMetadata,
            expected_version: &str,
        ) -> Result<(), String> {
            if metadata.library_version != expected_version {
                return Err(format!(
                "Version mismatch: model was trained with version {}, but current version is {}",
                metadata.library_version, expected_version
            ));
            }
            Ok(())
        }

        /// Get current timestamp
        pub fn current_timestamp() -> String {
            chrono::Utc::now().to_rfc3339()
        }

        /// Get library version
        pub fn library_version() -> String {
            env!("CARGO_PKG_VERSION").to_string()
        }

        /// Create model metadata
        pub fn create_metadata(
            model_type: &str,
            training_info: Option<TrainingInfo>,
            performance_metrics: Option<PerformanceMetrics>,
            feature_info: Option<FeatureInfo>,
        ) -> ModelMetadata {
            ModelMetadata {
                model_type: model_type.to_string(),
                version: "1.0".to_string(),
                library_version: Self::library_version(),
                timestamp: Self::current_timestamp(),
                training_info,
                performance_metrics,
                feature_info,
            }
        }
    }

    /// Convenience traits for model serialization
    pub trait SerializableModel: Serialize + for<'de> Deserialize<'de> {
        /// Save model to file with metadata
        fn save<P: AsRef<Path>>(
            &self,
            path: P,
            format: SerializationFormat,
        ) -> Result<(), Box<dyn std::error::Error>> {
            ModelSerializer::save_to_file(self, path, format)
        }

        /// Load model from file
        fn load<P: AsRef<Path>>(
            path: P,
            format: SerializationFormat,
        ) -> Result<Self, Box<dyn std::error::Error>>
        where
            Self: Sized,
        {
            ModelSerializer::load_from_file(path, format)
        }

        /// Convert to JSON string
        fn to_json(&self) -> Result<String, serde_json::Error> {
            ModelSerializer::to_json(self)
        }

        /// Create from JSON string
        fn from_json(json: &str) -> Result<Self, serde_json::Error>
        where
            Self: Sized,
        {
            ModelSerializer::from_json(json)
        }

        /// Convert to binary format
        fn to_binary(&self) -> Result<Vec<u8>, bincode::Error> {
            ModelSerializer::to_binary(self)
        }

        /// Create from binary format
        fn from_binary(data: &[u8]) -> Result<Self, bincode::Error>
        where
            Self: Sized,
        {
            ModelSerializer::from_binary(data)
        }
    }

    // Implement SerializableModel for all serializable model types
    impl SerializableModel for SerializableLinearRegression {}
    impl SerializableModel for SerializableRidgeRegression {}
    impl SerializableModel for SerializableLassoRegression {}
    impl SerializableModel for SerializableMultiOutputRegression {}
    impl SerializableModel for SerializableConstrainedOptimization {}

    /// Model registry for tracking serialized models
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelRegistry {
        pub models: HashMap<String, ModelMetadata>,
        pub registry_version: String,
        pub created_at: String,
        pub updated_at: String,
    }

    impl ModelRegistry {
        /// Create a new model registry
        pub fn new() -> Self {
            let now = ModelSerializer::current_timestamp();
            Self {
                models: HashMap::new(),
                registry_version: "1.0".to_string(),
                created_at: now.clone(),
                updated_at: now,
            }
        }

        /// Register a model
        pub fn register_model(&mut self, model_id: &str, metadata: ModelMetadata) {
            self.models.insert(model_id.to_string(), metadata);
            self.updated_at = ModelSerializer::current_timestamp();
        }

        /// Get model metadata
        pub fn get_model(&self, model_id: &str) -> Option<&ModelMetadata> {
            self.models.get(model_id)
        }

        /// List all models
        pub fn list_models(&self) -> Vec<&str> {
            self.models.keys().map(|k| k.as_str()).collect()
        }

        /// Remove a model
        pub fn remove_model(&mut self, model_id: &str) -> Option<ModelMetadata> {
            self.updated_at = ModelSerializer::current_timestamp();
            self.models.remove(model_id)
        }

        /// Save registry to file
        pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
            ModelSerializer::save_to_file(self, path, SerializationFormat::Json)
        }

        /// Load registry from file
        pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
            ModelSerializer::load_from_file(path, SerializationFormat::Json)
        }
    }

    impl Default for ModelRegistry {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Model versioning utilities
    pub struct ModelVersioning;

    impl ModelVersioning {
        /// Check if a model format is compatible
        pub fn is_compatible(model_version: &str, current_version: &str) -> bool {
            // Simple semantic versioning check
            let model_parts: Vec<&str> = model_version.split('.').collect();
            let current_parts: Vec<&str> = current_version.split('.').collect();

            if model_parts.len() >= 2 && current_parts.len() >= 2 {
                // Major version must match, minor version can be backwards compatible
                model_parts[0] == current_parts[0]
                    && model_parts[1].parse::<u32>().unwrap_or(0)
                        <= current_parts[1].parse::<u32>().unwrap_or(0)
            } else {
                model_version == current_version
            }
        }

        /// Migrate model between versions (placeholder for future implementation)
        pub fn migrate_model<T>(
            _model: T,
            _from_version: &str,
            _to_version: &str,
        ) -> Result<T, String> {
            // This would contain version-specific migration logic
            Err("Model migration not implemented yet".to_string())
        }
    }
} // End of serde_support module

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::serde_support::*;
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_serializable_matrix_conversion() {
        let original = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let serializable = SerializableMatrix::from(&original);
        let converted: DMatrix<f64> = serializable.into();

        assert_eq!(original, converted);
    }

    #[test]
    fn test_serializable_vector_conversion() {
        let original = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let serializable = SerializableVector::from(&original);
        let converted: DVector<f64> = serializable.into();

        assert_eq!(original, converted);
    }

    #[test]
    fn test_json_serialization() {
        let model = SerializableLinearRegression {
            coefficients: Some(SerializableVector {
                data: vec![1.0, 2.0, 3.0],
            }),
            intercept: Some(0.5),
            config: SerializableLinearRegressionConfig {
                fit_intercept: true,
                solver: "normal".to_string(),
            },
            n_features: Some(3),
            model_type: "LinearRegression".to_string(),
            version: "1.0".to_string(),
            timestamp: "2023-01-01T00:00:00Z".to_string(),
        };

        let json = model.to_json().unwrap();
        let deserialized: SerializableLinearRegression =
            SerializableLinearRegression::from_json(&json).unwrap();

        assert_eq!(
            model.coefficients.as_ref().unwrap().data,
            deserialized.coefficients.as_ref().unwrap().data
        );
        assert_eq!(model.intercept, deserialized.intercept);
        assert_eq!(model.n_features, deserialized.n_features);
    }

    #[test]
    fn test_binary_serialization() {
        let model = SerializableRidgeRegression {
            coefficients: Some(SerializableVector {
                data: vec![1.0, 2.0, 3.0],
            }),
            intercept: Some(0.5),
            alpha: 1.0,
            best_alpha: Some(1.0),
            cv_scores: None,
            config: SerializableRidgeConfig {
                alphas: vec![0.1, 1.0, 10.0],
                fit_intercept: true,
                cv_folds: 5,
                scoring: "r2".to_string(),
                solver: "auto".to_string(),
            },
            n_features: Some(3),
            model_type: "RidgeRegression".to_string(),
            version: "1.0".to_string(),
            timestamp: "2023-01-01T00:00:00Z".to_string(),
        };

        let binary = model.to_binary().unwrap();
        let deserialized: SerializableRidgeRegression =
            SerializableRidgeRegression::from_binary(&binary).unwrap();

        assert_eq!(model.alpha, deserialized.alpha);
        assert_eq!(model.config.alphas, deserialized.config.alphas);
    }

    #[test]
    fn test_file_serialization() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_model.json");

        let model = SerializableLassoRegression {
            coefficients: Some(SerializableVector {
                data: vec![1.0, 0.0, 2.0],
            }),
            intercept: Some(0.1),
            alpha: 0.1,
            best_alpha: Some(0.1),
            cv_scores: None,
            config: SerializableLassoConfig {
                alphas: vec![0.01, 0.1, 1.0],
                fit_intercept: true,
                cv_folds: 5,
                max_iter: 1000,
                tolerance: 1e-4,
                scoring: "r2".to_string(),
            },
            n_features: Some(3),
            sparsity_level: Some(0.33),
            model_type: "LassoRegression".to_string(),
            version: "1.0".to_string(),
            timestamp: "2023-01-01T00:00:00Z".to_string(),
        };

        // Save model
        model.save(&file_path, SerializationFormat::Json).unwrap();
        assert!(file_path.exists());

        // Load model
        let loaded_model: SerializableLassoRegression =
            SerializableLassoRegression::load(&file_path, SerializationFormat::Json).unwrap();

        assert_eq!(model.alpha, loaded_model.alpha);
        assert_eq!(model.sparsity_level, loaded_model.sparsity_level);
        assert_eq!(model.config.max_iter, loaded_model.config.max_iter);
    }

    #[test]
    fn test_model_registry() {
        let mut registry = ModelRegistry::new();

        let metadata = ModelMetadata {
            model_type: "LinearRegression".to_string(),
            version: "1.0".to_string(),
            library_version: "0.1.0".to_string(),
            timestamp: "2023-01-01T00:00:00Z".to_string(),
            training_info: Some(TrainingInfo {
                n_samples: 1000,
                n_features: 10,
                n_targets: None,
                training_time_seconds: Some(1.5),
                convergence_info: None,
            }),
            performance_metrics: None,
            feature_info: None,
        };

        registry.register_model("model_1", metadata.clone());

        assert_eq!(registry.list_models().len(), 1);
        assert!(registry.get_model("model_1").is_some());

        let retrieved_metadata = registry.get_model("model_1").unwrap();
        assert_eq!(retrieved_metadata.model_type, metadata.model_type);
        assert_eq!(
            retrieved_metadata.training_info.as_ref().unwrap().n_samples,
            1000
        );
    }

    #[test]
    fn test_version_compatibility() {
        assert!(ModelVersioning::is_compatible("1.0", "1.0"));
        assert!(ModelVersioning::is_compatible("1.0", "1.1"));
        assert!(ModelVersioning::is_compatible("1.1", "1.2"));
        assert!(!ModelVersioning::is_compatible("1.1", "1.0"));
        assert!(!ModelVersioning::is_compatible("2.0", "1.9"));
    }

    #[test]
    fn test_multi_output_serialization() {
        let model = SerializableMultiOutputRegression {
            coefficients: Some(SerializableMatrix {
                nrows: 3,
                ncols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }),
            intercept: Some(SerializableVector {
                data: vec![0.1, 0.2],
            }),
            target_correlations: None,
            rank_factors: None,
            chain_order: None,
            config: SerializableMultiOutputConfig {
                alpha: 1.0,
                l1_ratio: 0.5,
                strategy: "Joint".to_string(),
                max_iter: 1000,
                tolerance: 1e-4,
                rank: None,
                model_correlations: false,
                fit_intercept: true,
            },
            n_features: Some(3),
            n_targets: Some(2),
            training_loss: Some(0.1),
            model_type: "MultiOutputRegression".to_string(),
            version: "1.0".to_string(),
            timestamp: "2023-01-01T00:00:00Z".to_string(),
        };

        let json = model.to_json().unwrap();
        let deserialized: SerializableMultiOutputRegression =
            SerializableMultiOutputRegression::from_json(&json).unwrap();

        assert_eq!(model.n_features, deserialized.n_features);
        assert_eq!(model.n_targets, deserialized.n_targets);
        assert_eq!(model.config.alpha, deserialized.config.alpha);
    }

    #[test]
    fn test_metadata_creation() {
        let training_info = TrainingInfo {
            n_samples: 1000,
            n_features: 10,
            n_targets: None,
            training_time_seconds: Some(2.5),
            convergence_info: Some(ConvergenceInfo {
                converged: true,
                n_iterations: 100,
                final_loss: 0.001,
                tolerance_achieved: 1e-6,
            }),
        };

        let metadata =
            ModelSerializer::create_metadata("TestModel", Some(training_info), None, None);

        assert_eq!(metadata.model_type, "TestModel");
        assert_eq!(metadata.version, "1.0");
        assert!(metadata.training_info.is_some());
        assert_eq!(metadata.training_info.as_ref().unwrap().n_samples, 1000);
    }
} // End of serde_support module

#[cfg(feature = "serde")]
pub use serde_support::{
    ModelMetadata, ModelRegistry, ModelSerializer, ModelVersioning, PerformanceMetrics,
    SerializableConstrainedOptimization, SerializableLassoRegression, SerializableLinearRegression,
    SerializableMatrix, SerializableModel, SerializableMultiOutputRegression,
    SerializableRidgeRegression, SerializableVector, SerializationFormat, TrainingInfo,
};
