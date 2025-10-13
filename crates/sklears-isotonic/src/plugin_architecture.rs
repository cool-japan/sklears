//! Plugin architecture for custom constraints and optimization methods
//!
//! This module provides a flexible plugin system for dynamically loading and registering
//! custom constraint modules, optimization modules, and other components for isotonic regression.

use crate::modular_framework::{
    ConstraintModule, OptimizationModule, PostprocessingModule, PreprocessingModule,
};
use scirs2_core::ndarray::Array1;
use sklears_core::{prelude::SklearsError, types::Float};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// Plugin registry for managing dynamically loaded modules
pub struct PluginRegistry {
    /// Registered constraint modules
    constraint_modules: Arc<RwLock<HashMap<String, Box<dyn ConstraintModule>>>>,
    /// Registered optimization modules
    optimization_modules: Arc<RwLock<HashMap<String, Box<dyn OptimizationModule>>>>,
    /// Registered preprocessing modules
    preprocessing_modules: Arc<RwLock<HashMap<String, Box<dyn PreprocessingModule>>>>,
    /// Registered postprocessing modules
    postprocessing_modules: Arc<RwLock<HashMap<String, Box<dyn PostprocessingModule>>>>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            constraint_modules: Arc::new(RwLock::new(HashMap::new())),
            optimization_modules: Arc::new(RwLock::new(HashMap::new())),
            preprocessing_modules: Arc::new(RwLock::new(HashMap::new())),
            postprocessing_modules: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a constraint module
    pub fn register_constraint_module(
        &mut self,
        name: String,
        module: Box<dyn ConstraintModule>,
    ) -> Result<(), SklearsError> {
        let mut modules = self.constraint_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for constraint modules".to_string(),
            )
        })?;
        modules.insert(name, module);
        Ok(())
    }

    /// Register an optimization module
    pub fn register_optimization_module(
        &mut self,
        name: String,
        module: Box<dyn OptimizationModule>,
    ) -> Result<(), SklearsError> {
        let mut modules = self.optimization_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for optimization modules".to_string(),
            )
        })?;
        modules.insert(name, module);
        Ok(())
    }

    /// Register a preprocessing module
    pub fn register_preprocessing_module(
        &mut self,
        name: String,
        module: Box<dyn PreprocessingModule>,
    ) -> Result<(), SklearsError> {
        let mut modules = self.preprocessing_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for preprocessing modules".to_string(),
            )
        })?;
        modules.insert(name, module);
        Ok(())
    }

    /// Register a postprocessing module
    pub fn register_postprocessing_module(
        &mut self,
        name: String,
        module: Box<dyn PostprocessingModule>,
    ) -> Result<(), SklearsError> {
        let mut modules = self.postprocessing_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for postprocessing modules".to_string(),
            )
        })?;
        modules.insert(name, module);
        Ok(())
    }

    /// Get a constraint module by name
    pub fn get_constraint_module(
        &self,
        name: &str,
    ) -> Result<Option<Box<dyn ConstraintModule>>, SklearsError> {
        let modules = self.constraint_modules.read().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire read lock for constraint modules".to_string(),
            )
        })?;

        if let Some(module) = modules.get(name) {
            // Clone the module (assuming it implements Clone or we need to handle this differently)
            // For now, we'll return an error indicating the module exists but can't be cloned
            return Err(SklearsError::InvalidInput(format!(
                "Module '{}' exists but cannot be cloned",
                name
            )));
        }
        Ok(None)
    }

    /// Get names of all registered constraint modules
    pub fn list_constraint_modules(&self) -> Result<Vec<String>, SklearsError> {
        let modules = self.constraint_modules.read().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire read lock for constraint modules".to_string(),
            )
        })?;
        Ok(modules.keys().cloned().collect())
    }

    /// Get names of all registered optimization modules
    pub fn list_optimization_modules(&self) -> Result<Vec<String>, SklearsError> {
        let modules = self.optimization_modules.read().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire read lock for optimization modules".to_string(),
            )
        })?;
        Ok(modules.keys().cloned().collect())
    }

    /// Get names of all registered preprocessing modules
    pub fn list_preprocessing_modules(&self) -> Result<Vec<String>, SklearsError> {
        let modules = self.preprocessing_modules.read().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire read lock for preprocessing modules".to_string(),
            )
        })?;
        Ok(modules.keys().cloned().collect())
    }

    /// Get names of all registered postprocessing modules
    pub fn list_postprocessing_modules(&self) -> Result<Vec<String>, SklearsError> {
        let modules = self.postprocessing_modules.read().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire read lock for postprocessing modules".to_string(),
            )
        })?;
        Ok(modules.keys().cloned().collect())
    }

    /// Clear all registered modules
    pub fn clear(&mut self) -> Result<(), SklearsError> {
        let mut constraint_modules = self.constraint_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for constraint modules".to_string(),
            )
        })?;
        let mut optimization_modules = self.optimization_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for optimization modules".to_string(),
            )
        })?;
        let mut preprocessing_modules = self.preprocessing_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for preprocessing modules".to_string(),
            )
        })?;
        let mut postprocessing_modules = self.postprocessing_modules.write().map_err(|_| {
            SklearsError::InvalidInput(
                "Failed to acquire write lock for postprocessing modules".to_string(),
            )
        })?;

        constraint_modules.clear();
        optimization_modules.clear();
        preprocessing_modules.clear();
        postprocessing_modules.clear();

        Ok(())
    }
}

/// Plugin manager for handling plugin lifecycles
pub struct PluginManager {
    /// Plugin registry
    registry: PluginRegistry,
    /// Plugin metadata
    metadata: HashMap<String, PluginMetadata>,
}

/// Metadata for a plugin
#[derive(Debug, Clone)]
/// PluginMetadata
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin dependencies
    pub dependencies: Vec<String>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            registry: PluginRegistry::new(),
            metadata: HashMap::new(),
        }
    }

    /// Register a plugin with metadata
    pub fn register_plugin_with_metadata(
        &mut self,
        metadata: PluginMetadata,
        constraint_modules: Vec<(String, Box<dyn ConstraintModule>)>,
        optimization_modules: Vec<(String, Box<dyn OptimizationModule>)>,
        preprocessing_modules: Vec<(String, Box<dyn PreprocessingModule>)>,
        postprocessing_modules: Vec<(String, Box<dyn PostprocessingModule>)>,
    ) -> Result<(), SklearsError> {
        // Register constraint modules
        for (name, module) in constraint_modules {
            self.registry.register_constraint_module(name, module)?;
        }

        // Register optimization modules
        for (name, module) in optimization_modules {
            self.registry.register_optimization_module(name, module)?;
        }

        // Register preprocessing modules
        for (name, module) in preprocessing_modules {
            self.registry.register_preprocessing_module(name, module)?;
        }

        // Register postprocessing modules
        for (name, module) in postprocessing_modules {
            self.registry.register_postprocessing_module(name, module)?;
        }

        // Store metadata
        self.metadata.insert(metadata.name.clone(), metadata);

        Ok(())
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, name: &str) -> Option<&PluginMetadata> {
        self.metadata.get(name)
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        self.metadata.values().collect()
    }

    /// Get the plugin registry
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    /// Get a mutable reference to the plugin registry
    pub fn registry_mut(&mut self) -> &mut PluginRegistry {
        &mut self.registry
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin trait for defining plugin interfaces
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Initialize the plugin
    fn initialize(&mut self) -> Result<(), SklearsError>;

    /// Shutdown the plugin
    fn shutdown(&mut self) -> Result<(), SklearsError>;

    /// Get constraint modules provided by this plugin
    fn constraint_modules(&self) -> Vec<(String, Box<dyn ConstraintModule>)> {
        Vec::new()
    }

    /// Get optimization modules provided by this plugin
    fn optimization_modules(&self) -> Vec<(String, Box<dyn OptimizationModule>)> {
        Vec::new()
    }

    /// Get preprocessing modules provided by this plugin
    fn preprocessing_modules(&self) -> Vec<(String, Box<dyn PreprocessingModule>)> {
        Vec::new()
    }

    /// Get postprocessing modules provided by this plugin
    fn postprocessing_modules(&self) -> Vec<(String, Box<dyn PostprocessingModule>)> {
        Vec::new()
    }
}

/// Example custom constraint plugin
#[derive(Debug)]
/// CustomBoundsConstraint
pub struct CustomBoundsConstraint {
    min_value: Float,
    max_value: Float,
}

impl CustomBoundsConstraint {
    /// Create a new custom bounds constraint
    pub fn new(min_value: Float, max_value: Float) -> Self {
        Self {
            min_value,
            max_value,
        }
    }
}

impl ConstraintModule for CustomBoundsConstraint {
    fn name(&self) -> &'static str {
        "CustomBoundsConstraint"
    }

    fn apply_constraint(&self, values: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        Ok(values.mapv(|x| x.max(self.min_value).min(self.max_value)))
    }

    fn check_constraint(&self, values: &Array1<Float>) -> bool {
        values
            .iter()
            .all(|&x| x >= self.min_value && x <= self.max_value)
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("min_value".to_string(), self.min_value.to_string());
        params.insert("max_value".to_string(), self.max_value.to_string());
        params
    }
}

/// Example plugin implementation
#[derive(Debug)]
/// ExamplePlugin
pub struct ExamplePlugin {
    initialized: bool,
}

impl ExamplePlugin {
    /// Create a new example plugin
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl Default for ExamplePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for ExamplePlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "ExamplePlugin".to_string(),
            version: "1.0.0".to_string(),
            description: "An example plugin for demonstration".to_string(),
            author: "Sklears Team".to_string(),
            dependencies: vec![],
        }
    }

    fn initialize(&mut self) -> Result<(), SklearsError> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), SklearsError> {
        self.initialized = false;
        Ok(())
    }

    fn constraint_modules(&self) -> Vec<(String, Box<dyn ConstraintModule>)> {
        vec![(
            "custom_bounds".to_string(),
            Box::new(CustomBoundsConstraint::new(0.0, 1.0)),
        )]
    }
}

/// Convenience function to create a plugin manager with default plugins
pub fn create_plugin_manager_with_defaults() -> Result<PluginManager, SklearsError> {
    let mut manager = PluginManager::new();

    // Register example plugin
    let mut example_plugin = ExamplePlugin::new();
    example_plugin.initialize()?;

    let metadata = example_plugin.metadata();
    let constraint_modules = example_plugin.constraint_modules();
    let optimization_modules = example_plugin.optimization_modules();
    let preprocessing_modules = example_plugin.preprocessing_modules();
    let postprocessing_modules = example_plugin.postprocessing_modules();

    manager.register_plugin_with_metadata(
        metadata,
        constraint_modules,
        optimization_modules,
        preprocessing_modules,
        postprocessing_modules,
    )?;

    Ok(manager)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_plugin_registry_creation() {
        let registry = PluginRegistry::new();
        assert!(registry.list_constraint_modules().unwrap().is_empty());
        assert!(registry.list_optimization_modules().unwrap().is_empty());
    }

    #[test]
    fn test_custom_bounds_constraint() {
        let constraint = CustomBoundsConstraint::new(0.0, 1.0);
        let values = array![0.5, 1.5, -0.5, 0.8];
        let constrained = constraint.apply_constraint(&values).unwrap();
        let expected = array![0.5, 1.0, 0.0, 0.8];

        for (actual, expected) in constrained.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_custom_bounds_constraint_check() {
        let constraint = CustomBoundsConstraint::new(0.0, 1.0);
        let valid_values = array![0.0, 0.5, 1.0];
        let invalid_values = array![0.0, 1.5, 1.0];

        assert!(constraint.check_constraint(&valid_values));
        assert!(!constraint.check_constraint(&invalid_values));
    }

    #[test]
    fn test_example_plugin() {
        let mut plugin = ExamplePlugin::new();
        assert!(plugin.initialize().is_ok());

        let metadata = plugin.metadata();
        assert_eq!(metadata.name, "ExamplePlugin");
        assert_eq!(metadata.version, "1.0.0");

        let constraint_modules = plugin.constraint_modules();
        assert_eq!(constraint_modules.len(), 1);
        assert_eq!(constraint_modules[0].0, "custom_bounds");

        assert!(plugin.shutdown().is_ok());
    }

    #[test]
    fn test_plugin_manager() {
        let mut manager = PluginManager::new();
        let plugins = manager.list_plugins();
        assert!(plugins.is_empty());

        // Test with example plugin
        let mut example_plugin = ExamplePlugin::new();
        example_plugin.initialize().unwrap();

        let metadata = example_plugin.metadata();
        let constraint_modules = example_plugin.constraint_modules();

        manager
            .register_plugin_with_metadata(
                metadata.clone(),
                constraint_modules,
                vec![],
                vec![],
                vec![],
            )
            .unwrap();

        let plugins = manager.list_plugins();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, metadata.name);

        let plugin_metadata = manager.get_plugin_metadata("ExamplePlugin");
        assert!(plugin_metadata.is_some());
    }

    #[test]
    fn test_create_plugin_manager_with_defaults() {
        let manager = create_plugin_manager_with_defaults();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        let plugins = manager.list_plugins();
        assert!(!plugins.is_empty());

        let constraint_modules = manager.registry().list_constraint_modules().unwrap();
        assert!(constraint_modules.contains(&"custom_bounds".to_string()));
    }

    #[test]
    fn test_plugin_registry_clear() {
        let mut registry = PluginRegistry::new();
        let constraint = Box::new(CustomBoundsConstraint::new(0.0, 1.0));

        registry
            .register_constraint_module("test".to_string(), constraint)
            .unwrap();
        assert!(!registry.list_constraint_modules().unwrap().is_empty());

        registry.clear().unwrap();
        assert!(registry.list_constraint_modules().unwrap().is_empty());
    }

    #[test]
    fn test_custom_bounds_constraint_parameters() {
        let constraint = CustomBoundsConstraint::new(0.5, 2.0);
        let params = constraint.get_parameters();

        assert_eq!(params.get("min_value"), Some(&"0.5".to_string()));
        assert_eq!(params.get("max_value"), Some(&"2".to_string()));
    }
}
