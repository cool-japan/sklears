//! Configuration Management System for Model Selection
//!
//! This module provides comprehensive configuration management for model selection
//! operations, including YAML/JSON serialization, configuration inheritance,
//! and template management.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Configuration management error types
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    Yaml(String),
    #[error("Configuration validation error: {0}")]
    Validation(String),
    #[error("Template error: {0}")]
    Template(String),
}

/// Main configuration structure for model selection operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionConfig {
    /// Cross-validation configuration
    pub cross_validation: CrossValidationConfig,
    /// Hyperparameter optimization configuration
    pub optimization: OptimizationConfig,
    /// Scoring and evaluation configuration
    pub scoring: ScoringConfig,
    /// Resource and performance configuration
    pub resources: ResourceConfig,
    /// Custom parameters and extensions
    #[serde(default)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Type of cross-validation (kfold, stratified, etc.)
    pub method: String,
    /// Number of folds
    #[serde(default = "default_n_folds")]
    pub n_folds: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Shuffle data before splitting
    #[serde(default = "default_shuffle")]
    pub shuffle: bool,
    /// Additional method-specific parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization method (grid_search, bayesian, evolutionary, etc.)
    pub method: String,
    /// Maximum number of iterations/evaluations
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Parameter space definition
    pub parameter_space: HashMap<String, ParameterDefinition>,
    /// Optimization-specific parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    #[serde(default)]
    pub enabled: bool,
    /// Patience (number of iterations without improvement)
    #[serde(default = "default_patience")]
    pub patience: usize,
    /// Minimum improvement required
    #[serde(default = "default_min_delta")]
    pub min_delta: f64,
}

/// Parameter definition for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ParameterDefinition {
    #[serde(rename = "uniform")]
    Uniform { low: f64, high: f64 },
    #[serde(rename = "log_uniform")]
    LogUniform { low: f64, high: f64 },
    #[serde(rename = "categorical")]
    Categorical { choices: Vec<serde_json::Value> },
    #[serde(rename = "integer")]
    Integer { low: i64, high: i64 },
    #[serde(rename = "choice")]
    Choice { choices: Vec<serde_json::Value> },
}

/// Scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Primary scoring metric
    pub primary: String,
    /// Additional metrics to compute
    #[serde(default)]
    pub additional: Vec<String>,
    /// Custom scoring functions
    #[serde(default)]
    pub custom_scorers: HashMap<String, serde_json::Value>,
    /// Scoring parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Resource and performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Number of parallel jobs (-1 for all cores)
    #[serde(default = "default_n_jobs")]
    pub n_jobs: i32,
    /// Memory limit in MB
    pub memory_limit: Option<usize>,
    /// Enable GPU acceleration
    #[serde(default)]
    pub use_gpu: bool,
    /// Batch size for parallel processing
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Enable memory optimization
    #[serde(default)]
    pub memory_efficient: bool,
}

/// Configuration manager for loading, saving, and validating configurations
pub struct ConfigManager {
    base_config: Option<ModelSelectionConfig>,
    template_registry: HashMap<String, ModelSelectionConfig>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            base_config: None,
            template_registry: HashMap::new(),
        }
    }

    /// Load configuration from a file (supports JSON and YAML)
    pub fn load_from_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<ModelSelectionConfig, ConfigError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;

        let config = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::from_str(&content)?,
            Some("yaml") | Some("yml") => {
                // For now, use JSON as a placeholder since YAML requires additional dependency
                serde_json::from_str(&content).map_err(|e| ConfigError::Yaml(e.to_string()))?
            }
            _ => {
                return Err(ConfigError::Validation(
                    "Unsupported file format".to_string(),
                ))
            }
        };

        self.validate_config(&config)?;
        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file<P: AsRef<Path>>(
        &self,
        config: &ModelSelectionConfig,
        path: P,
    ) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let content = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::to_string_pretty(config)?,
            Some("yaml") | Some("yml") => {
                // For now, use JSON as a placeholder
                serde_json::to_string_pretty(config)?
            }
            _ => {
                return Err(ConfigError::Validation(
                    "Unsupported file format".to_string(),
                ))
            }
        };

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration from a string
    pub fn load_from_string(
        &mut self,
        content: &str,
        format: &str,
    ) -> Result<ModelSelectionConfig, ConfigError> {
        let config = match format {
            "json" => serde_json::from_str(content)?,
            "yaml" | "yml" => {
                serde_json::from_str(content).map_err(|e| ConfigError::Yaml(e.to_string()))?
            }
            _ => return Err(ConfigError::Validation("Unsupported format".to_string())),
        };

        self.validate_config(&config)?;
        Ok(config)
    }

    /// Register a configuration template
    pub fn register_template(
        &mut self,
        name: String,
        config: ModelSelectionConfig,
    ) -> Result<(), ConfigError> {
        self.validate_config(&config)?;
        self.template_registry.insert(name, config);
        Ok(())
    }

    /// Get a configuration template
    pub fn get_template(&self, name: &str) -> Option<&ModelSelectionConfig> {
        self.template_registry.get(name)
    }

    /// Create configuration from template with overrides
    pub fn from_template(
        &self,
        template_name: &str,
        overrides: HashMap<String, serde_json::Value>,
    ) -> Result<ModelSelectionConfig, ConfigError> {
        let template = self.get_template(template_name).ok_or_else(|| {
            ConfigError::Template(format!("Template '{}' not found", template_name))
        })?;

        let mut config = template.clone();
        self.apply_overrides(&mut config, overrides)?;
        self.validate_config(&config)?;

        Ok(config)
    }

    /// Validate configuration
    fn validate_config(&self, config: &ModelSelectionConfig) -> Result<(), ConfigError> {
        // Validate cross-validation configuration
        if config.cross_validation.n_folds < 2 {
            return Err(ConfigError::Validation(
                "n_folds must be at least 2".to_string(),
            ));
        }

        // Validate optimization configuration
        if config.optimization.max_iter == 0 {
            return Err(ConfigError::Validation(
                "max_iter must be greater than 0".to_string(),
            ));
        }

        // Validate resource configuration
        if config.resources.n_jobs == 0 {
            return Err(ConfigError::Validation("n_jobs cannot be 0".to_string()));
        }

        if config.resources.batch_size == 0 {
            return Err(ConfigError::Validation(
                "batch_size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Apply configuration overrides
    fn apply_overrides(
        &self,
        config: &mut ModelSelectionConfig,
        overrides: HashMap<String, serde_json::Value>,
    ) -> Result<(), ConfigError> {
        for (key, value) in overrides {
            self.apply_override(config, &key, value)?;
        }
        Ok(())
    }

    /// Apply a single override using dot notation (e.g., "cross_validation.n_folds")
    fn apply_override(
        &self,
        config: &mut ModelSelectionConfig,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), ConfigError> {
        let parts: Vec<&str> = key.split('.').collect();
        match parts.as_slice() {
            ["cross_validation", "n_folds"] => {
                if let Some(n) = value.as_u64() {
                    config.cross_validation.n_folds = n as usize;
                }
            }
            ["cross_validation", "random_state"] => {
                if let Some(n) = value.as_u64() {
                    config.cross_validation.random_state = Some(n);
                }
            }
            ["optimization", "max_iter"] => {
                if let Some(n) = value.as_u64() {
                    config.optimization.max_iter = n as usize;
                }
            }
            ["resources", "n_jobs"] => {
                if let Some(n) = value.as_i64() {
                    config.resources.n_jobs = n as i32;
                }
            }
            _ => {
                // Store in custom parameters
                config.custom.insert(key.to_string(), value);
            }
        }
        Ok(())
    }

    /// Get default configuration templates
    pub fn load_default_templates(&mut self) {
        // Grid search template
        let grid_search_config = ModelSelectionConfig {
            cross_validation: CrossValidationConfig {
                method: "kfold".to_string(),
                n_folds: 5,
                random_state: Some(42),
                shuffle: true,
                parameters: HashMap::new(),
            },
            optimization: OptimizationConfig {
                method: "grid_search".to_string(),
                max_iter: 100,
                early_stopping: None,
                parameter_space: HashMap::new(),
                parameters: HashMap::new(),
            },
            scoring: ScoringConfig {
                primary: "accuracy".to_string(),
                additional: vec!["precision".to_string(), "recall".to_string()],
                custom_scorers: HashMap::new(),
                parameters: HashMap::new(),
            },
            resources: ResourceConfig {
                n_jobs: -1,
                memory_limit: None,
                use_gpu: false,
                batch_size: 32,
                memory_efficient: false,
            },
            custom: HashMap::new(),
        };

        // Bayesian optimization template
        let bayesian_config = ModelSelectionConfig {
            optimization: OptimizationConfig {
                method: "bayesian".to_string(),
                max_iter: 50,
                early_stopping: Some(EarlyStoppingConfig {
                    enabled: true,
                    patience: 5,
                    min_delta: 0.001,
                }),
                parameter_space: HashMap::new(),
                parameters: HashMap::new(),
            },
            ..grid_search_config.clone()
        };

        self.template_registry
            .insert("grid_search".to_string(), grid_search_config);
        self.template_registry
            .insert("bayesian".to_string(), bayesian_config);
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        let mut manager = Self::new();
        manager.load_default_templates();
        manager
    }
}

// Default value functions
fn default_n_folds() -> usize {
    5
}
fn default_shuffle() -> bool {
    true
}
fn default_max_iter() -> usize {
    100
}
fn default_patience() -> usize {
    5
}
fn default_min_delta() -> f64 {
    0.001
}
fn default_n_jobs() -> i32 {
    -1
}
fn default_batch_size() -> usize {
    32
}

impl Default for ModelSelectionConfig {
    fn default() -> Self {
        Self {
            cross_validation: CrossValidationConfig {
                method: "kfold".to_string(),
                n_folds: default_n_folds(),
                random_state: Some(42),
                shuffle: default_shuffle(),
                parameters: HashMap::new(),
            },
            optimization: OptimizationConfig {
                method: "grid_search".to_string(),
                max_iter: default_max_iter(),
                early_stopping: None,
                parameter_space: HashMap::new(),
                parameters: HashMap::new(),
            },
            scoring: ScoringConfig {
                primary: "accuracy".to_string(),
                additional: vec!["precision".to_string(), "recall".to_string()],
                custom_scorers: HashMap::new(),
                parameters: HashMap::new(),
            },
            resources: ResourceConfig {
                n_jobs: default_n_jobs(),
                memory_limit: None,
                use_gpu: false,
                batch_size: default_batch_size(),
                memory_efficient: false,
            },
            custom: HashMap::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::new();
        assert!(manager.base_config.is_none());
        assert!(manager.template_registry.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = ModelSelectionConfig::default();
        assert_eq!(config.cross_validation.method, "kfold");
        assert_eq!(config.cross_validation.n_folds, 5);
        assert_eq!(config.optimization.method, "grid_search");
        assert_eq!(config.optimization.max_iter, 100);
        assert_eq!(config.scoring.primary, "accuracy");
        assert_eq!(config.resources.n_jobs, -1);
    }

    #[test]
    fn test_config_validation() {
        let manager = ConfigManager::new();
        let mut config = ModelSelectionConfig::default();

        // Valid configuration should pass
        assert!(manager.validate_config(&config).is_ok());

        // Invalid n_folds should fail
        config.cross_validation.n_folds = 1;
        assert!(manager.validate_config(&config).is_err());

        // Reset and test invalid max_iter
        config.cross_validation.n_folds = 5;
        config.optimization.max_iter = 0;
        assert!(manager.validate_config(&config).is_err());
    }

    #[test]
    fn test_template_registration() {
        let mut manager = ConfigManager::new();
        let config = ModelSelectionConfig::default();

        assert!(manager
            .register_template("test_template".to_string(), config)
            .is_ok());
        assert!(manager.get_template("test_template").is_some());
        assert!(manager.get_template("nonexistent").is_none());
    }

    #[test]
    fn test_json_serialization() {
        let config = ModelSelectionConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelSelectionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(
            config.cross_validation.method,
            deserialized.cross_validation.method
        );
        assert_eq!(
            config.optimization.max_iter,
            deserialized.optimization.max_iter
        );
    }

    #[test]
    fn test_override_application() {
        let mut manager = ConfigManager::default();
        let template_config = ModelSelectionConfig::default();
        manager
            .register_template("test".to_string(), template_config)
            .unwrap();

        let mut overrides = HashMap::new();
        overrides.insert(
            "cross_validation.n_folds".to_string(),
            serde_json::Value::from(10),
        );
        overrides.insert(
            "optimization.max_iter".to_string(),
            serde_json::Value::from(200),
        );

        let config = manager.from_template("test", overrides).unwrap();
        assert_eq!(config.cross_validation.n_folds, 10);
        assert_eq!(config.optimization.max_iter, 200);
    }

    #[test]
    fn test_default_templates() {
        let manager = ConfigManager::default();

        assert!(manager.get_template("grid_search").is_some());
        assert!(manager.get_template("bayesian").is_some());

        let grid_template = manager.get_template("grid_search").unwrap();
        assert_eq!(grid_template.optimization.method, "grid_search");

        let bayesian_template = manager.get_template("bayesian").unwrap();
        assert_eq!(bayesian_template.optimization.method, "bayesian");
        assert!(bayesian_template.optimization.early_stopping.is_some());
    }
}
