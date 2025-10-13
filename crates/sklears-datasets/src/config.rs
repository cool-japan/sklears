//! Configuration management for dataset generation
//!
//! This module provides YAML and JSON configuration support for dataset generators,
//! allowing users to define complex dataset generation pipelines declaratively.
//!
//! This module requires the `serde` feature to be enabled.

use thiserror::Error;

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML parsing error: {0}")]
    Yaml(String),
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Generation error: {0}")]
    Generation(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

pub type ConfigResult<T> = Result<T, ConfigError>;

#[cfg(feature = "serde")]
mod implementation {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;

    /// Dataset generation configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GenerationConfig {
        /// Configuration metadata
        pub metadata: ConfigMetadata,
        /// List of datasets to generate
        pub datasets: Vec<DatasetSpec>,
        /// Global generation settings
        pub global_settings: Option<GlobalSettings>,
        /// Export settings
        pub export: Option<ExportConfig>,
    }

    /// Configuration metadata
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConfigMetadata {
        /// Configuration name
        pub name: String,
        /// Configuration version
        pub version: String,
        /// Configuration description
        pub description: Option<String>,
        /// Author information
        pub author: Option<String>,
        /// Creation timestamp
        #[serde(skip_serializing_if = "Option::is_none")]
        pub created_at: Option<String>,
        /// Tags for categorization
        #[serde(default)]
        pub tags: Vec<String>,
    }

    /// Individual dataset specification
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum DatasetSpec {
        #[serde(rename = "classification")]
        Classification(ClassificationConfig),
        #[serde(rename = "regression")]
        Regression(RegressionConfig),
        #[serde(rename = "clustering")]
        Clustering(ClusteringConfig),
        #[serde(rename = "manifold")]
        Manifold(ManifoldConfig),
        #[serde(rename = "time_series")]
        TimeSeries(TimeSeriesConfig),
        #[serde(rename = "custom")]
        Custom(CustomDatasetConfig),
    }

    /// Classification dataset configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ClassificationConfig {
        /// Dataset name/identifier
        pub name: String,
        /// Number of samples
        pub n_samples: usize,
        /// Number of features
        pub n_features: usize,
        /// Number of classes
        pub n_classes: usize,
        /// Number of informative features
        pub n_informative: Option<usize>,
        /// Random state for reproducibility
        pub random_state: Option<u64>,
        /// Enable SIMD optimization
        pub use_simd: Option<bool>,
        /// Feature names
        pub feature_names: Option<Vec<String>>,
        /// Class names
        pub class_names: Option<Vec<String>>,
    }

    /// Regression dataset configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RegressionConfig {
        /// Dataset name/identifier
        pub name: String,
        /// Number of samples
        pub n_samples: usize,
        /// Number of features
        pub n_features: usize,
        /// Number of informative features
        pub n_informative: Option<usize>,
        /// Noise level
        pub noise: Option<f64>,
        /// Random state
        pub random_state: Option<u64>,
        /// Enable SIMD optimization
        pub use_simd: Option<bool>,
        /// Feature names
        pub feature_names: Option<Vec<String>>,
        /// Target names
        pub target_names: Option<Vec<String>>,
    }

    /// Clustering dataset configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ClusteringConfig {
        /// Dataset name/identifier
        pub name: String,
        /// Number of samples
        pub n_samples: usize,
        /// Number of features
        pub n_features: usize,
        /// Number of centers
        pub centers: Option<usize>,
        /// Cluster standard deviation
        pub cluster_std: Option<f64>,
        /// Random state
        pub random_state: Option<u64>,
    }

    /// Manifold dataset configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ManifoldConfig {
        /// Dataset name/identifier
        pub name: String,
        /// Manifold type
        pub manifold_type: ManifoldType,
        /// Number of samples
        pub n_samples: usize,
        /// Noise level
        pub noise: Option<f64>,
        /// Random state
        pub random_state: Option<u64>,
        /// Manifold-specific parameters
        #[serde(flatten)]
        pub parameters: HashMap<String, serde_json::Value>,
    }

    /// Available manifold types
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ManifoldType {
        /// SwissRoll

        SwissRoll,
        /// SCurve

        SCurve,
        /// Custom

        Custom,
    }

    /// Time series dataset configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimeSeriesConfig {
        /// Dataset name/identifier
        pub name: String,
        /// Number of time steps
        pub n_timesteps: usize,
        /// Number of features/series
        pub n_features: usize,
        /// Noise component
        pub noise: Option<f64>,
        /// Random state
        pub random_state: Option<u64>,
    }

    /// Custom dataset configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CustomDatasetConfig {
        /// Dataset name/identifier
        pub name: String,
        /// Generator function name
        pub generator: String,
        /// Parameters for the generator
        pub parameters: HashMap<String, serde_json::Value>,
    }

    /// Global generation settings
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlobalSettings {
        /// Default random state
        pub default_random_state: Option<u64>,
        /// Use SIMD by default
        pub default_simd: Option<bool>,
        /// Number of parallel workers
        pub n_workers: Option<usize>,
        /// Validation settings
        pub validation: Option<ValidationSettings>,
    }

    /// Validation settings
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ValidationSettings {
        /// Enable automatic validation
        pub enabled: bool,
        /// Statistical tests to run
        pub tests: Vec<StatisticalTest>,
    }

    /// Statistical test types
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum StatisticalTest {
        /// Normality

        Normality,
        /// Quality

        Quality,
        /// Custom

        Custom(String),
    }

    /// Export configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ExportConfig {
        /// Output directory
        pub output_dir: String,
        /// Export formats
        pub formats: Vec<ExportFormat>,
        /// Include metadata
        pub include_metadata: Option<bool>,
    }

    /// Export format specification
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ExportFormat {
        /// Csv

        Csv,
        /// Json

        Json,
        /// Parquet

        Parquet,
    }

    /// Configuration parser and loader
    pub struct ConfigLoader;

    impl ConfigLoader {
        /// Load configuration from file
        pub fn load_from_file<P: AsRef<Path>>(path: P) -> ConfigResult<GenerationConfig> {
            let path = path.as_ref();
            let contents = fs::read_to_string(path)?;

            match path.extension().and_then(|s| s.to_str()) {
                Some("yaml") | Some("yml") => Self::load_from_yaml(&contents),
                Some("json") => Self::load_from_json(&contents),
                Some(ext) => Err(ConfigError::UnsupportedFormat(format!(
                    "Unsupported config format: {}. Supported formats: yaml, yml, json",
                    ext
                ))),
                None => Err(ConfigError::UnsupportedFormat(
                    "No file extension found. Supported formats: yaml, yml, json".to_string(),
                )),
            }
        }

        /// Load configuration from YAML string
        pub fn load_from_yaml(yaml_str: &str) -> ConfigResult<GenerationConfig> {
            serde_yaml::from_str(yaml_str).map_err(|e| ConfigError::Yaml(e.to_string()))
        }

        /// Load configuration from JSON string
        pub fn load_from_json(json_str: &str) -> ConfigResult<GenerationConfig> {
            Ok(serde_json::from_str(json_str)?)
        }

        /// Save configuration to file
        pub fn save_to_file<P: AsRef<Path>>(
            config: &GenerationConfig,
            path: P,
        ) -> ConfigResult<()> {
            let path = path.as_ref();

            let contents = match path.extension().and_then(|s| s.to_str()) {
                Some("yaml") | Some("yml") => Self::to_yaml(config)?,
                Some("json") => Self::to_json(config)?,
                Some(ext) => {
                    return Err(ConfigError::UnsupportedFormat(format!(
                        "Unsupported config format: {}",
                        ext
                    )))
                }
                None => {
                    return Err(ConfigError::UnsupportedFormat(
                        "No file extension found".to_string(),
                    ))
                }
            };

            fs::write(path, contents)?;
            Ok(())
        }

        /// Convert configuration to YAML string
        pub fn to_yaml(config: &GenerationConfig) -> ConfigResult<String> {
            serde_yaml::to_string(config).map_err(|e| ConfigError::Yaml(e.to_string()))
        }

        /// Convert configuration to JSON string
        pub fn to_json(config: &GenerationConfig) -> ConfigResult<String> {
            Ok(serde_json::to_string_pretty(config)?)
        }
    }

    /// Generate example configuration
    pub fn generate_example_config() -> GenerationConfig {
        GenerationConfig {
            metadata: ConfigMetadata {
                name: "Example Dataset Configuration".to_string(),
                version: "1.0.0".to_string(),
                description: Some("An example configuration".to_string()),
                author: Some("sklears-datasets".to_string()),
                created_at: Some("2024-01-01T00:00:00Z".to_string()),
                tags: vec!["example".to_string()],
            },
            datasets: vec![
                DatasetSpec::Classification(ClassificationConfig {
                    name: "iris_like".to_string(),
                    n_samples: 150,
                    n_features: 4,
                    n_classes: 3,
                    n_informative: Some(4),
                    random_state: Some(42),
                    use_simd: Some(true),
                    feature_names: Some(vec![
                        "sepal_length".to_string(),
                        "sepal_width".to_string(),
                        "petal_length".to_string(),
                        "petal_width".to_string(),
                    ]),
                    class_names: Some(vec![
                        "setosa".to_string(),
                        "versicolor".to_string(),
                        "virginica".to_string(),
                    ]),
                }),
                DatasetSpec::Regression(RegressionConfig {
                    name: "boston_like".to_string(),
                    n_samples: 500,
                    n_features: 13,
                    n_informative: Some(10),
                    noise: Some(10.0),
                    random_state: Some(42),
                    use_simd: Some(true),
                    feature_names: None,
                    target_names: Some(vec!["price".to_string()]),
                }),
            ],
            global_settings: Some(GlobalSettings {
                default_random_state: Some(42),
                default_simd: Some(true),
                n_workers: Some(4),
                validation: Some(ValidationSettings {
                    enabled: true,
                    tests: vec![StatisticalTest::Quality],
                }),
            }),
            export: Some(ExportConfig {
                output_dir: "./datasets".to_string(),
                formats: vec![ExportFormat::Csv, ExportFormat::Json],
                include_metadata: Some(true),
            }),
        }
    }

    #[allow(non_snake_case)]
#[cfg(test)]
    mod tests {
        use super::*;
        use tempfile::tempdir;

        #[test]
        fn test_example_config_generation() {
            let config = generate_example_config();
            assert_eq!(config.datasets.len(), 2);
        }

        #[test]
        fn test_config_serialization() {
            let config = generate_example_config();

            // Test YAML serialization
            let yaml_str = ConfigLoader::to_yaml(&config).unwrap();
            assert!(!yaml_str.is_empty());
            assert!(yaml_str.contains("iris_like"));

            // Test JSON serialization
            let json_str = ConfigLoader::to_json(&config).unwrap();
            assert!(!json_str.is_empty());
            assert!(json_str.contains("iris_like"));
        }

        #[test]
        fn test_config_deserialization() {
            let config = generate_example_config();

            // Test YAML round-trip
            let yaml_str = ConfigLoader::to_yaml(&config).unwrap();
            let config_from_yaml = ConfigLoader::load_from_yaml(&yaml_str).unwrap();
            assert_eq!(config.datasets.len(), config_from_yaml.datasets.len());

            // Test JSON round-trip
            let json_str = ConfigLoader::to_json(&config).unwrap();
            let config_from_json = ConfigLoader::load_from_json(&json_str).unwrap();
            assert_eq!(config.datasets.len(), config_from_json.datasets.len());
        }

        #[test]
        fn test_file_operations() {
            let config = generate_example_config();
            let dir = tempdir().unwrap();

            // Test YAML file operations
            let yaml_path = dir.path().join("config.yaml");
            ConfigLoader::save_to_file(&config, &yaml_path).unwrap();
            let loaded_config = ConfigLoader::load_from_file(&yaml_path).unwrap();
            assert_eq!(config.datasets.len(), loaded_config.datasets.len());

            // Test JSON file operations
            let json_path = dir.path().join("config.json");
            ConfigLoader::save_to_file(&config, &json_path).unwrap();
            let loaded_config = ConfigLoader::load_from_file(&json_path).unwrap();
            assert_eq!(config.datasets.len(), loaded_config.datasets.len());
        }
    }
}

#[cfg(feature = "serde")]
pub use implementation::*;

#[cfg(not(feature = "serde"))]
mod stubs {
    use super::*;

    pub struct ConfigLoader;
    pub struct GenerationConfig;
    pub struct ConfigMetadata;
    pub struct DatasetSpec;
    pub struct ClassificationConfig;
    pub struct RegressionConfig;
    pub struct ClusteringConfig;
    pub struct ManifoldConfig;
    pub struct TimeSeriesConfig;
    pub struct CustomDatasetConfig;
    pub struct GlobalSettings;
    pub struct ValidationSettings;
    pub struct StatisticalTest;
    pub struct ExportConfig;
    pub struct ExportFormat;
    pub struct ManifoldType;

    impl ConfigLoader {
        pub fn load_from_file<P: AsRef<std::path::Path>>(
            _path: P,
        ) -> ConfigResult<GenerationConfig> {
            Err(ConfigError::UnsupportedFormat(
                "Configuration management requires the 'serde' feature to be enabled".to_string(),
            ))
        }

        pub fn load_from_yaml(_yaml_str: &str) -> ConfigResult<GenerationConfig> {
            Err(ConfigError::UnsupportedFormat(
                "YAML support requires the 'serde' feature to be enabled".to_string(),
            ))
        }

        pub fn load_from_json(_json_str: &str) -> ConfigResult<GenerationConfig> {
            Err(ConfigError::UnsupportedFormat(
                "JSON support requires the 'serde' feature to be enabled".to_string(),
            ))
        }
    }

    pub fn generate_example_config() -> ConfigResult<GenerationConfig> {
        Err(ConfigError::UnsupportedFormat(
            "Configuration generation requires the 'serde' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(not(feature = "serde"))]
pub use stubs::*;
