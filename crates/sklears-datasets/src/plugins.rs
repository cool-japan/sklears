//! Plugin architecture for custom dataset generators
//!
//! This module provides a dynamic plugin system that allows users to register
//! and load custom dataset generators at runtime. It supports both static
//! registration and dynamic library loading.

use crate::traits::{
    Dataset, DatasetGenerator, DatasetTraitError, DatasetTraitResult, GeneratorConfig,
    InMemoryDataset,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Plugin system errors
#[derive(Error, Debug)]
pub enum PluginError {
    #[error("Generator not found: {0}")]
    GeneratorNotFound(String),
    #[error("Plugin already registered: {0}")]
    AlreadyRegistered(String),
    #[error("Plugin registration failed: {0}")]
    RegistrationFailed(String),
    #[error("Plugin validation failed: {0}")]
    ValidationFailed(String),
    #[error("Dynamic loading error: {0}")]
    DynamicLoading(String),
    #[error("API version mismatch: expected {expected}, got {actual}")]
    ApiVersionMismatch { expected: String, actual: String },
    #[error("Plugin dependency missing: {0}")]
    DependencyMissing(String),
}

pub type PluginResult<T> = Result<T, PluginError>;

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub api_version: String,
    pub dependencies: Vec<String>,
    pub capabilities: Vec<String>,
    pub tags: Vec<String>,
}

impl Default for PluginMetadata {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            version: "0.1.0".to_string(),
            description: "No description".to_string(),
            author: "Unknown".to_string(),
            api_version: "1.0.0".to_string(),
            dependencies: Vec::new(),
            capabilities: Vec::new(),
            tags: Vec::new(),
        }
    }
}

/// Plugin interface that all generators must implement
pub trait PluginGenerator: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Generate a dataset with the given configuration
    fn generate(&self, config: GeneratorConfig) -> DatasetTraitResult<InMemoryDataset>;

    /// Validate the configuration
    fn validate_config(&self, config: &GeneratorConfig) -> DatasetTraitResult<()> {
        let _ = config;
        Ok(())
    }

    /// Get parameter schema for this generator
    fn parameter_schema(&self) -> HashMap<String, ParameterInfo> {
        HashMap::new()
    }

    /// Check if this generator can handle the given configuration
    fn can_handle(&self, config: &GeneratorConfig) -> bool {
        let _ = config;
        true
    }

    /// Initialize the plugin (called once after loading)
    fn initialize(&mut self) -> PluginResult<()> {
        Ok(())
    }

    /// Cleanup the plugin (called before unloading)
    fn cleanup(&mut self) -> PluginResult<()> {
        Ok(())
    }
}

/// Parameter information for generator configuration
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub description: String,
    pub parameter_type: ParameterType,
    pub required: bool,
    pub default_value: Option<String>,
    pub constraints: Vec<ParameterConstraint>,
}

/// Parameter types supported by the plugin system
#[derive(Debug, Clone)]
pub enum ParameterType {
    /// Integer

    Integer { min: Option<i64>, max: Option<i64> },
    /// Float

    Float { min: Option<f64>, max: Option<f64> },
    /// String

    String { pattern: Option<String> },
    /// Boolean

    Boolean,
    /// IntegerArray

    IntegerArray,
    /// FloatArray

    FloatArray,
    /// Enum

    Enum { values: Vec<String> },
}

/// Parameter constraints
#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Range

    Range { min: f64, max: f64 },
    /// Length

    Length { min: usize, max: usize },
    /// Pattern

    Pattern(String),
    /// Custom

    Custom(String),
}

/// Plugin registry for managing generators
pub struct PluginRegistry {

    generators: Arc<RwLock<HashMap<String, Box<dyn PluginGenerator>>>>,

    metadata_cache: Arc<RwLock<HashMap<String, PluginMetadata>>>,
    hooks: Arc<RwLock<Vec<Box<dyn PluginHook>>>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            generators: Arc::new(RwLock::new(HashMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            hooks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a plugin generator
    pub fn register<G>(&self, generator: G) -> PluginResult<()>
    where
        G: PluginGenerator + 'static,
    {
        let metadata = generator.metadata();
        let name = metadata.name.clone();

        // Validate API version
        if metadata.api_version != "1.0.0" {
            return Err(PluginError::ApiVersionMismatch {
                expected: "1.0.0".to_string(),
                actual: metadata.api_version,
            });
        }

        // Check if already registered
        {
            let generators = self.generators.read().unwrap();
            if generators.contains_key(&name) {
                return Err(PluginError::AlreadyRegistered(name));
            }
        }

        // Validate dependencies
        self.validate_dependencies(&metadata.dependencies)?;

        // Run registration hooks
        self.run_registration_hooks(&metadata)?;

        // Register the generator
        {
            let mut generators = self.generators.write().unwrap();
            let mut metadata_cache = self.metadata_cache.write().unwrap();

            generators.insert(name.clone(), Box::new(generator));
            metadata_cache.insert(name.clone(), metadata);
        }

        Ok(())
    }

    /// Unregister a plugin generator
    pub fn unregister(&self, name: &str) -> PluginResult<()> {
        let mut generators = self.generators.write().unwrap();
        let mut metadata_cache = self.metadata_cache.write().unwrap();

        if let Some(mut generator) = generators.remove(name) {
            generator.cleanup()?;
            metadata_cache.remove(name);
            Ok(())
        } else {
            Err(PluginError::GeneratorNotFound(name.to_string()))
        }
    }

    /// Get a generator by name
    pub fn get(&self, name: &str) -> Option<Box<dyn PluginGenerator>> {
        let generators = self.generators.read().unwrap();
        // Note: This is a simplified version. In a real implementation,
        // you'd need a way to clone or share the generator safely
        None // Placeholder - would need trait object cloning or Arc<Mutex<>>
    }

    /// Check if a generator is registered
    pub fn has_generator(&self, name: &str) -> bool {
        let generators = self.generators.read().unwrap();
        generators.contains_key(name)
    }

    /// List all registered generators
    pub fn list_generators(&self) -> Vec<String> {
        let generators = self.generators.read().unwrap();
        generators.keys().cloned().collect()
    }

    /// Get metadata for a generator
    pub fn get_metadata(&self, name: &str) -> Option<PluginMetadata> {
        let metadata_cache = self.metadata_cache.read().unwrap();
        metadata_cache.get(name).cloned()
    }

    /// List all metadata
    pub fn list_metadata(&self) -> Vec<PluginMetadata> {
        let metadata_cache = self.metadata_cache.read().unwrap();
        metadata_cache.values().cloned().collect()
    }

    /// Generate a dataset using a named generator
    pub fn generate(
        &self,
        name: &str,
        config: GeneratorConfig,
    ) -> DatasetTraitResult<InMemoryDataset> {
        let generators = self.generators.read().unwrap();
        if let Some(generator) = generators.get(name) {
            generator.validate_config(&config)?;
            generator.generate(config)
        } else {
            Err(DatasetTraitError::Configuration(format!(
                "Generator not found: {}",
                name
            )))
        }
    }

    /// Find generators by capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<String> {
        let metadata_cache = self.metadata_cache.read().unwrap();
        metadata_cache
            .iter()
            .filter(|(_, meta)| meta.capabilities.contains(&capability.to_string()))
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Find generators by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<String> {
        let metadata_cache = self.metadata_cache.read().unwrap();
        metadata_cache
            .iter()
            .filter(|(_, meta)| meta.tags.contains(&tag.to_string()))
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Validate dependencies
    fn validate_dependencies(&self, dependencies: &[String]) -> PluginResult<()> {
        let generators = self.generators.read().unwrap();
        for dep in dependencies {
            if !generators.contains_key(dep) {
                return Err(PluginError::DependencyMissing(dep.clone()));
            }
        }
        Ok(())
    }

    /// Run registration hooks
    fn run_registration_hooks(&self, metadata: &PluginMetadata) -> PluginResult<()> {
        let hooks = self.hooks.read().unwrap();
        for hook in hooks.iter() {
            hook.on_registration(metadata)?;
        }
        Ok(())
    }

    /// Add a plugin hook
    pub fn add_hook<H>(&self, hook: H)
    where
        H: PluginHook + 'static,
    {
        let mut hooks = self.hooks.write().unwrap();
        hooks.push(Box::new(hook));
    }

    /// Clear all hooks
    pub fn clear_hooks(&self) {
        let mut hooks = self.hooks.write().unwrap();
        hooks.clear();
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Hook interface for plugin lifecycle events
pub trait PluginHook: Send + Sync {
    /// Called when a plugin is registered
    fn on_registration(&self, metadata: &PluginMetadata) -> PluginResult<()>;

    /// Called when a plugin is unregistered
    fn on_unregistration(&self, name: &str) -> PluginResult<()> {
        let _ = name;
        Ok(())
    }

    /// Called before dataset generation
    fn on_generation_start(
        &self,
        generator_name: &str,
        config: &GeneratorConfig,
    ) -> PluginResult<()> {
        let _ = (generator_name, config);
        Ok(())
    }

    /// Called after dataset generation
    fn on_generation_complete(
        &self,
        generator_name: &str,
        dataset: &InMemoryDataset,
    ) -> PluginResult<()> {
        let _ = (generator_name, dataset);
        Ok(())
    }
}

/// Example plugin implementations

/// Simple custom generator example
pub struct CustomLinearGenerator;

impl PluginGenerator for CustomLinearGenerator {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "custom_linear".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates linear datasets with custom patterns".to_string(),
            author: "Example Author".to_string(),
            api_version: "1.0.0".to_string(),
            dependencies: vec![],
            capabilities: vec!["regression".to_string(), "linear".to_string()],
            tags: vec!["custom".to_string(), "linear".to_string()],
        }
    }

    fn generate(&self, config: GeneratorConfig) -> DatasetTraitResult<InMemoryDataset> {
        use scirs2_core::ndarray::{Array1, Array2};
        use scirs2_core::random::Random;

        let mut rng = match config.random_state {
            Some(seed) => Random::new_with_seed(seed),
            None => Random::new(),
        };

        // Get slope parameter
        let slope = config
            .get_parameter("slope")
            .and_then(|v| match v {
                crate::traits::ConfigValue::Float(s) => Some(*s),
                _ => None,
            })
            .unwrap_or(1.0);

        // Generate features
        let mut features = Array2::<f64>::zeros((config.n_samples, config.n_features));
        for mut row in features.rows_mut() {
            for val in row.iter_mut() {
                *val = rng.gen_range(-10.0..10.0);
            }
        }

        // Generate targets with linear relationship
        let targets: Array1<f64> = features
            .rows()
            .into_iter()
            .map(|row| row.sum() * slope + rng.gen_normal(0.0, 0.1))
            .collect();

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("generator".to_string(), "custom_linear".to_string());
        metadata.insert("slope".to_string(), slope.to_string());

        Ok(crate::traits::InMemoryDataset::with_metadata(
            features,
            Some(targets),
            metadata,
        ))
    }

    fn parameter_schema(&self) -> HashMap<String, ParameterInfo> {
        let mut schema = HashMap::new();
        schema.insert(
            "slope".to_string(),
            ParameterInfo {
                name: "slope".to_string(),
                description: "Linear slope coefficient".to_string(),
                parameter_type: ParameterType::Float {
                    min: Some(-10.0),
                    max: Some(10.0),
                },
                required: false,
                default_value: Some("1.0".to_string()),
                constraints: vec![ParameterConstraint::Range {
                    min: -10.0,
                    max: 10.0,
                }],
            },
        );
        schema
    }

    fn validate_config(&self, config: &GeneratorConfig) -> DatasetTraitResult<()> {
        if config.n_samples == 0 || config.n_features == 0 {
            return Err(DatasetTraitError::Configuration(
                "n_samples and n_features must be > 0".to_string(),
            ));
        }

        // Validate slope parameter if present
        if let Some(slope_val) = config.get_parameter("slope") {
            match slope_val {
                crate::traits::ConfigValue::Float(slope) => {
                    if !(-10.0..=10.0).contains(slope) {
                        return Err(DatasetTraitError::Configuration(
                            "slope must be between -10.0 and 10.0".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(DatasetTraitError::Configuration(
                        "slope parameter must be a float".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Logging hook example
pub struct LoggingHook;

impl PluginHook for LoggingHook {
    fn on_registration(&self, metadata: &PluginMetadata) -> PluginResult<()> {
        println!("Plugin registered: {} v{}", metadata.name, metadata.version);
        Ok(())
    }

    fn on_unregistration(&self, name: &str) -> PluginResult<()> {
        println!("Plugin unregistered: {}", name);
        Ok(())
    }

    fn on_generation_start(
        &self,
        generator_name: &str,
        config: &GeneratorConfig,
    ) -> PluginResult<()> {
        println!(
            "Starting generation with {}: {} samples x {} features",
            generator_name, config.n_samples, config.n_features
        );
        Ok(())
    }

    fn on_generation_complete(
        &self,
        generator_name: &str,
        dataset: &InMemoryDataset,
    ) -> PluginResult<()> {
        println!(
            "Completed generation with {}: {} samples generated",
            generator_name,
            dataset.n_samples()
        );
        Ok(())
    }
}

/// Plugin discovery and management utilities
pub struct PluginManager {
    registry: PluginRegistry,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            registry: PluginRegistry::new(),
        }
    }

    /// Get the plugin registry
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    /// Load built-in plugins
    pub fn load_builtin_plugins(&self) -> PluginResult<()> {
        // Register custom linear generator
        self.registry.register(CustomLinearGenerator)?;

        // Add logging hook
        self.registry.add_hook(LoggingHook);

        Ok(())
    }

    /// Auto-discover and load plugins from a directory
    pub fn discover_plugins(&self, _plugin_dir: &str) -> PluginResult<Vec<String>> {
        // This would implement dynamic loading in a real scenario
        // For now, return empty list as this requires dynamic library loading
        Ok(vec![])
    }

    /// Validate all registered plugins
    pub fn validate_all(&self) -> PluginResult<Vec<String>> {
        let mut failed = Vec::new();
        let generators = self.registry.list_generators();

        for generator_name in generators {
            if let Some(metadata) = self.registry.get_metadata(&generator_name) {
                // Basic validation - check dependencies
                if let Err(_) = self.registry.validate_dependencies(&metadata.dependencies) {
                    failed.push(generator_name);
                }
            }
        }

        if failed.is_empty() {
            Ok(vec![])
        } else {
            Err(PluginError::ValidationFailed(format!(
                "Failed plugins: {:?}",
                failed
            )))
        }
    }

    /// Create a test configuration for a generator
    pub fn create_test_config(&self, generator_name: &str) -> Option<GeneratorConfig> {
        if let Some(metadata) = self.registry.get_metadata(generator_name) {
            let mut config = GeneratorConfig::new(100, 4);
            config = config.with_random_state(42);

            // Add default parameters based on schema
            // This is a simplified implementation
            if metadata.capabilities.contains(&"regression".to_string()) {
                config.set_parameter("noise".to_string(), 0.1);
            }

            Some(config)
        } else {
            None
        }
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ConfigValue;

    #[test]
    fn test_plugin_registration() {
        let registry = PluginRegistry::new();

        // Register custom generator
        assert!(registry.register(CustomLinearGenerator).is_ok());

        // Check if registered
        assert!(registry.has_generator("custom_linear"));
        assert_eq!(registry.list_generators(), vec!["custom_linear"]);

        // Check metadata
        let metadata = registry.get_metadata("custom_linear").unwrap();
        assert_eq!(metadata.name, "custom_linear");
        assert_eq!(metadata.version, "1.0.0");
    }

    #[test]
    fn test_custom_generator() {
        let generator = CustomLinearGenerator;

        // Test metadata
        let metadata = generator.metadata();
        assert_eq!(metadata.name, "custom_linear");
        assert!(metadata.capabilities.contains(&"regression".to_string()));

        // Test parameter schema
        let schema = generator.parameter_schema();
        assert!(schema.contains_key("slope"));

        // Test generation
        let mut config = GeneratorConfig::new(50, 3);
        config.set_parameter("slope".to_string(), 2.0);
        config = config.with_random_state(42);

        assert!(generator.validate_config(&config).is_ok());

        let dataset = generator.generate(config).unwrap();
        assert_eq!(dataset.n_samples(), 50);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.has_targets());
    }

    #[test]
    fn test_plugin_hooks() {
        let registry = PluginRegistry::new();
        registry.add_hook(LoggingHook);

        // Register a generator (should trigger hook)
        assert!(registry.register(CustomLinearGenerator).is_ok());
    }

    #[test]
    fn test_plugin_manager() {
        let manager = PluginManager::new();

        // Load built-in plugins
        assert!(manager.load_builtin_plugins().is_ok());

        // Validate all plugins
        assert!(manager.validate_all().is_ok());

        // Test configuration creation
        let config = manager.create_test_config("custom_linear");
        assert!(config.is_some());

        let config = config.unwrap();
        assert_eq!(config.n_samples, 100);
        assert_eq!(config.n_features, 4);
    }

    #[test]
    fn test_capability_and_tag_search() {
        let registry = PluginRegistry::new();
        registry.register(CustomLinearGenerator).unwrap();

        // Find by capability
        let regression_generators = registry.find_by_capability("regression");
        assert!(regression_generators.contains(&"custom_linear".to_string()));

        // Find by tag
        let custom_generators = registry.find_by_tag("custom");
        assert!(custom_generators.contains(&"custom_linear".to_string()));
    }

    #[test]
    fn test_parameter_validation() {
        let generator = CustomLinearGenerator;

        // Valid config
        let mut valid_config = GeneratorConfig::new(100, 5);
        valid_config.set_parameter("slope".to_string(), 2.0);
        assert!(generator.validate_config(&valid_config).is_ok());

        // Invalid slope (out of range)
        let mut invalid_config = GeneratorConfig::new(100, 5);
        invalid_config.set_parameter("slope".to_string(), 15.0);
        assert!(generator.validate_config(&invalid_config).is_err());

        // Invalid dimensions
        let invalid_dims = GeneratorConfig::new(0, 5);
        assert!(generator.validate_config(&invalid_dims).is_err());
    }
}
