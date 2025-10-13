//! Configuration Management for Neural Networks
//!
//! This module provides comprehensive configuration support for neural networks,
//! allowing users to define architectures, training parameters, and model settings
//! through YAML and JSON files.

use crate::activation::Activation;
use crate::weight_init::InitStrategy;
use crate::NeuralResult;
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration format for loading/saving
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfigFormat {
    JSON,
    YAML,
}

impl ConfigFormat {
    /// Determine format from file extension
    pub fn from_path<P: AsRef<Path>>(path: P) -> ConfigFormat {
        let path = path.as_ref();
        match path.extension().and_then(|s| s.to_str()) {
            Some("yaml") | Some("yml") => ConfigFormat::YAML,
            Some("json") => ConfigFormat::JSON,
            _ => ConfigFormat::JSON, // default to JSON
        }
    }
}

/// Complete neural network configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct NeuralNetworkConfig<T: FloatBounds> {
    /// Model architecture configuration
    pub model: ModelConfig<T>,
    /// Training configuration
    pub training: TrainingConfig<T>,
    /// Optimization configuration
    pub optimizer: OptimizerConfig<T>,
    /// Data processing configuration
    pub data: DataConfig,
    /// Evaluation configuration
    pub evaluation: EvaluationConfig,
    /// Miscellaneous settings
    pub misc: MiscConfig,
}

/// Model architecture configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct ModelConfig<T: FloatBounds> {
    /// Model type (mlp, transformer, rnn, etc.)
    pub model_type: String,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Hidden layer sizes for MLP
    pub hidden_layers: Vec<usize>,
    /// Activation function
    pub activation: String,
    /// Dropout rate
    pub dropout: Option<T>,
    /// Batch normalization
    pub batch_norm: bool,
    /// Layer normalization
    pub layer_norm: bool,
    /// Weight initialization strategy
    pub weight_init: String,
    /// Architecture-specific parameters
    pub arch_params: HashMap<String, String>,
}

/// Training configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct TrainingConfig<T: FloatBounds> {
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: T,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule<T>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    /// Validation split
    pub validation_split: Option<T>,
    /// Random seed
    pub random_seed: Option<u64>,
    /// Mixed precision training
    pub mixed_precision: bool,
    /// Gradient clipping
    pub gradient_clipping: Option<T>,
}

/// Learning rate schedule configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct LearningRateSchedule<T: FloatBounds> {
    /// Schedule type
    pub schedule_type: String,
    /// Step size for step decay
    pub step_size: Option<usize>,
    /// Decay factor
    pub decay_factor: Option<T>,
    /// Minimum learning rate
    pub min_lr: Option<T>,
    /// Maximum learning rate for cyclical schedules
    pub max_lr: Option<T>,
    /// Warmup steps
    pub warmup_steps: Option<usize>,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct EarlyStoppingConfig<T: FloatBounds> {
    /// Metric to monitor
    pub monitor: String,
    /// Patience (number of epochs)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: T,
    /// Mode (min or max)
    pub mode: String,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: FloatBounds + serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct OptimizerConfig<T: FloatBounds> {
    /// Optimizer type
    pub optimizer_type: String,
    /// Momentum for SGD/RMSprop
    pub momentum: Option<T>,
    /// Beta1 for Adam-like optimizers
    pub beta1: Option<T>,
    /// Beta2 for Adam-like optimizers
    pub beta2: Option<T>,
    /// Weight decay
    pub weight_decay: Option<T>,
    /// Epsilon for numerical stability
    pub epsilon: Option<T>,
    /// Nesterov momentum
    pub nesterov: bool,
}

/// Data processing configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Data normalization
    pub normalization: Option<String>,
    /// Data augmentation
    pub augmentation: Vec<DataAugmentationConfig>,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Number of data loading workers
    pub num_workers: usize,
    /// Pin memory for GPU training
    pub pin_memory: bool,
}

/// Data augmentation configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DataAugmentationConfig {
    /// Augmentation type
    pub aug_type: String,
    /// Probability of applying
    pub probability: f64,
    /// Parameters for the augmentation
    pub params: HashMap<String, String>,
}

/// Evaluation configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Metrics to compute
    pub metrics: Vec<String>,
    /// Whether to compute confusion matrix
    pub confusion_matrix: bool,
    /// Whether to compute classification report
    pub classification_report: bool,
    /// Test batch size
    pub test_batch_size: Option<usize>,
}

/// Miscellaneous configuration
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MiscConfig {
    /// Whether to log training progress
    pub verbose: bool,
    /// Logging frequency (every N epochs)
    pub log_frequency: usize,
    /// Checkpoint saving frequency
    pub checkpoint_frequency: Option<usize>,
    /// Directory for saving checkpoints
    pub checkpoint_dir: Option<String>,
    /// Device preference (cpu, gpu)
    pub device: String,
    /// Whether to use deterministic algorithms
    pub deterministic: bool,
}

impl<T: FloatBounds> Default for NeuralNetworkConfig<T> {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            optimizer: OptimizerConfig::default(),
            data: DataConfig::default(),
            evaluation: EvaluationConfig::default(),
            misc: MiscConfig::default(),
        }
    }
}

impl<T: FloatBounds> Default for ModelConfig<T> {
    fn default() -> Self {
        Self {
            model_type: "mlp".to_string(),
            input_dim: 784,
            output_dim: 10,
            hidden_layers: vec![128, 64],
            activation: "relu".to_string(),
            dropout: Some(T::from(0.2).unwrap()),
            batch_norm: false,
            layer_norm: false,
            weight_init: "xavier_uniform".to_string(),
            arch_params: HashMap::new(),
        }
    }
}

impl<T: FloatBounds> Default for TrainingConfig<T> {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: T::from(0.001).unwrap(),
            lr_schedule: LearningRateSchedule::default(),
            early_stopping: None,
            validation_split: Some(T::from(0.2).unwrap()),
            random_seed: Some(42),
            mixed_precision: false,
            gradient_clipping: None,
        }
    }
}

impl<T: FloatBounds> Default for LearningRateSchedule<T> {
    fn default() -> Self {
        Self {
            schedule_type: "constant".to_string(),
            step_size: None,
            decay_factor: None,
            min_lr: None,
            max_lr: None,
            warmup_steps: None,
        }
    }
}

impl<T: FloatBounds> Default for OptimizerConfig<T> {
    fn default() -> Self {
        Self {
            optimizer_type: "adam".to_string(),
            momentum: None,
            beta1: Some(T::from(0.9).unwrap()),
            beta2: Some(T::from(0.999).unwrap()),
            weight_decay: Some(T::from(1e-4).unwrap()),
            epsilon: Some(T::from(1e-8).unwrap()),
            nesterov: false,
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            normalization: Some("standard".to_string()),
            augmentation: Vec::new(),
            shuffle: true,
            num_workers: 1,
            pin_memory: false,
        }
    }
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            metrics: vec!["accuracy".to_string(), "loss".to_string()],
            confusion_matrix: false,
            classification_report: false,
            test_batch_size: None,
        }
    }
}

impl Default for MiscConfig {
    fn default() -> Self {
        Self {
            verbose: true,
            log_frequency: 10,
            checkpoint_frequency: None,
            checkpoint_dir: None,
            device: "cpu".to_string(),
            deterministic: false,
        }
    }
}

/// Configuration manager for loading and saving configurations
pub struct ConfigManager;

impl ConfigManager {
    /// Load configuration from file
    #[cfg(feature = "serde")]
    pub fn load_config<T: FloatBounds + for<'de> serde::Deserialize<'de>>(
        path: &str,
    ) -> NeuralResult<NeuralNetworkConfig<T>> {
        let format = ConfigFormat::from_path(path);
        let content = std::fs::read_to_string(path).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to read config file: {}", e))
        })?;

        match format {
            ConfigFormat::JSON => serde_json::from_str(&content)
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to parse JSON: {}", e))),
            ConfigFormat::YAML => serde_yaml::from_str(&content)
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to parse YAML: {}", e))),
        }
    }

    /// Save configuration to file
    #[cfg(feature = "serde")]
    pub fn save_config<T: FloatBounds + serde::Serialize>(
        config: &NeuralNetworkConfig<T>,
        path: &str,
    ) -> NeuralResult<()> {
        let format = ConfigFormat::from_path(path);

        let content = match format {
            ConfigFormat::JSON => serde_json::to_string_pretty(config).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to serialize to JSON: {}", e))
            })?,
            ConfigFormat::YAML => serde_yaml::to_string(config).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to serialize to YAML: {}", e))
            })?,
        };

        std::fs::write(path, content).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to write config file: {}", e))
        })?;

        Ok(())
    }

    /// Create a template configuration file
    #[cfg(feature = "serde")]
    pub fn create_template<T: FloatBounds + serde::Serialize + Default>(
        path: &str,
        model_type: &str,
    ) -> NeuralResult<()> {
        let mut config = NeuralNetworkConfig::<T>::default();
        config.model.model_type = model_type.to_string();

        // Customize defaults based on model type
        match model_type {
            "transformer" => {
                config
                    .model
                    .arch_params
                    .insert("num_heads".to_string(), "8".to_string());
                config
                    .model
                    .arch_params
                    .insert("d_model".to_string(), "512".to_string());
                config
                    .model
                    .arch_params
                    .insert("num_layers".to_string(), "6".to_string());
                config.training.learning_rate = T::from(1e-4).unwrap();
            }
            "rnn" | "lstm" | "gru" => {
                config
                    .model
                    .arch_params
                    .insert("sequence_length".to_string(), "50".to_string());
                config
                    .model
                    .arch_params
                    .insert("bidirectional".to_string(), "false".to_string());
            }
            "cnn" => {
                config
                    .model
                    .arch_params
                    .insert("kernel_size".to_string(), "3".to_string());
                config
                    .model
                    .arch_params
                    .insert("stride".to_string(), "1".to_string());
                config
                    .model
                    .arch_params
                    .insert("padding".to_string(), "1".to_string());
            }
            _ => {} // MLP defaults are already set
        }

        Self::save_config(&config, path)
    }

    /// Validate configuration
    pub fn validate_config<T: FloatBounds>(config: &NeuralNetworkConfig<T>) -> NeuralResult<()> {
        // Validate model configuration
        if config.model.input_dim == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "input_dim".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if config.model.output_dim == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "output_dim".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        // Validate training configuration
        if config.training.epochs == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "epochs".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if config.training.batch_size == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "batch_size".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if config.training.learning_rate <= T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        // Validate optimizer configuration
        if let Some(beta1) = config.optimizer.beta1 {
            if beta1 < T::zero() || beta1 >= T::one() {
                return Err(SklearsError::InvalidParameter {
                    name: "beta1".to_string(),
                    reason: "must be in range [0, 1)".to_string(),
                });
            }
        }

        if let Some(beta2) = config.optimizer.beta2 {
            if beta2 < T::zero() || beta2 >= T::one() {
                return Err(SklearsError::InvalidParameter {
                    name: "beta2".to_string(),
                    reason: "must be in range [0, 1)".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Convert string representations to enums/types
    pub fn parse_activation(activation_str: &str) -> NeuralResult<Activation> {
        match activation_str.to_lowercase().as_str() {
            "relu" => Ok(Activation::Relu),
            "sigmoid" => Ok(Activation::Logistic),
            "tanh" => Ok(Activation::Tanh),
            "identity" | "linear" => Ok(Activation::Identity),
            "elu" => Ok(Activation::Elu),
            "leakyrelu" => Ok(Activation::LeakyRelu),
            "swish" | "silu" => Ok(Activation::Swish),
            "gelu" => Ok(Activation::Gelu),
            "mish" => Ok(Activation::Mish),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown activation function: {}",
                activation_str
            ))),
        }
    }

    /// Convert string representations to weight initialization strategies
    pub fn parse_weight_init(init_str: &str) -> NeuralResult<InitStrategy> {
        match init_str.to_lowercase().as_str() {
            "zeros" => Ok(InitStrategy::Zeros),
            "uniform" => Ok(InitStrategy::Uniform {
                low: -0.1,
                high: 0.1,
            }),
            "normal" => Ok(InitStrategy::Normal {
                mean: 0.0,
                std: 0.1,
            }),
            "xavier_uniform" | "glorot_uniform" => Ok(InitStrategy::XavierUniform),
            "xavier_normal" | "glorot_normal" => Ok(InitStrategy::XavierNormal),
            "he_uniform" => Ok(InitStrategy::HeUniform),
            "he_normal" => Ok(InitStrategy::HeNormal),
            "lecun_uniform" => Ok(InitStrategy::LeCunUniform),
            "lecun_normal" => Ok(InitStrategy::LeCunNormal),
            "orthogonal" => Ok(InitStrategy::Orthogonal { gain: 1.0 }),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown weight initialization: {}",
                init_str
            ))),
        }
    }
}

/// Create example configuration files
pub fn create_example_configs() -> NeuralResult<()> {
    // Create output directory
    std::fs::create_dir_all("configs/examples")
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to create directory: {}", e)))?;

    #[cfg(feature = "serde")]
    {
        // MLP configuration
        ConfigManager::create_template::<f32>("configs/examples/mlp_config.yaml", "mlp")?;
        ConfigManager::create_template::<f32>("configs/examples/mlp_config.json", "mlp")?;

        // Transformer configuration
        ConfigManager::create_template::<f32>(
            "configs/examples/transformer_config.yaml",
            "transformer",
        )?;

        // RNN configuration
        ConfigManager::create_template::<f32>("configs/examples/lstm_config.yaml", "lstm")?;

        // CNN configuration
        ConfigManager::create_template::<f32>("configs/examples/cnn_config.yaml", "cnn")?;
    }

    println!("Example configuration files created in configs/examples/");
    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_format_from_path() {
        assert_eq!(ConfigFormat::from_path("config.yaml"), ConfigFormat::YAML);
        assert_eq!(ConfigFormat::from_path("config.yml"), ConfigFormat::YAML);
        assert_eq!(ConfigFormat::from_path("config.json"), ConfigFormat::JSON);
        assert_eq!(ConfigFormat::from_path("config.txt"), ConfigFormat::JSON); // default
    }

    #[test]
    fn test_default_configs() {
        let config = NeuralNetworkConfig::<f32>::default();
        assert_eq!(config.model.model_type, "mlp");
        assert_eq!(config.training.epochs, 100);
        assert_eq!(config.optimizer.optimizer_type, "adam");
    }

    #[test]
    fn test_config_validation() {
        let mut config = NeuralNetworkConfig::<f32>::default();
        assert!(ConfigManager::validate_config(&config).is_ok());

        // Test invalid input_dim
        config.model.input_dim = 0;
        assert!(ConfigManager::validate_config(&config).is_err());
    }

    #[test]
    fn test_activation_parsing() {
        assert!(matches!(
            ConfigManager::parse_activation("relu"),
            Ok(Activation::Relu)
        ));
        assert!(matches!(
            ConfigManager::parse_activation("ReLU"),
            Ok(Activation::Relu)
        ));
        assert!(ConfigManager::parse_activation("unknown").is_err());
    }

    #[test]
    fn test_weight_init_parsing() {
        assert!(matches!(
            ConfigManager::parse_weight_init("xavier_uniform"),
            Ok(InitStrategy::XavierUniform)
        ));
        assert!(ConfigManager::parse_weight_init("unknown").is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_config_serialization() {
        let config = NeuralNetworkConfig::<f32>::default();

        // Test JSON serialization
        let json_str = serde_json::to_string(&config).unwrap();
        let parsed: NeuralNetworkConfig<f32> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.model.model_type, config.model.model_type);

        // Test YAML serialization
        let yaml_str = serde_yaml::to_string(&config).unwrap();
        let parsed: NeuralNetworkConfig<f32> = serde_yaml::from_str(&yaml_str).unwrap();
        assert_eq!(parsed.model.model_type, config.model.model_type);
    }
}
