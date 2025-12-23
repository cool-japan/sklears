//! Plugin architecture for custom kernel approximations
//!
//! This module provides a flexible plugin system for registering and using
//! custom kernel approximation methods. It allows runtime discovery and
//! instantiation of kernel approximation plugins.

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Transform};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Errors that can occur in the plugin system
#[derive(Error, Debug)]
/// PluginError
pub enum PluginError {
    #[error("Plugin not found: {name}")]
    PluginNotFound { name: String },
    #[error("Plugin already registered: {name}")]
    PluginAlreadyRegistered { name: String },
    #[error("Invalid plugin configuration: {message}")]
    InvalidConfiguration { message: String },
    #[error("Plugin initialization failed: {message}")]
    InitializationFailed { message: String },
    #[error("Type casting error for plugin: {name}")]
    TypeCastError { name: String },
}

/// Metadata about a plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Supported kernel types
    pub supported_kernels: Vec<String>,
    /// Required parameters
    pub required_parameters: Vec<String>,
    /// Optional parameters
    pub optional_parameters: Vec<String>,
}

/// Configuration for a plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
/// PluginConfig
#[derive(Default)]
pub struct PluginConfig {
    /// Parameters for the plugin
    pub parameters: HashMap<String, serde_json::Value>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

/// Trait for kernel approximation plugins
pub trait KernelApproximationPlugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Create a new instance of the plugin with given configuration
    fn create(
        &self,
        config: PluginConfig,
    ) -> std::result::Result<Box<dyn KernelApproximationInstance>, PluginError>;

    /// Validate configuration
    fn validate_config(&self, config: &PluginConfig) -> std::result::Result<(), PluginError>;

    /// Get default configuration
    fn default_config(&self) -> PluginConfig;
}

/// Instance of a kernel approximation plugin
pub trait KernelApproximationInstance: Send + Sync {
    /// Fit the approximation to data
    fn fit(&mut self, x: &Array2<f64>, y: &()) -> std::result::Result<(), PluginError>;

    /// Transform data using the fitted approximation
    fn transform(&self, x: &Array2<f64>) -> std::result::Result<Array2<f64>, PluginError>;

    /// Check if the instance is fitted
    fn is_fitted(&self) -> bool;

    /// Get the number of output features
    fn n_output_features(&self) -> Option<usize>;

    /// Clone the instance
    fn clone_instance(&self) -> Box<dyn KernelApproximationInstance>;

    /// Get instance as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Plugin factory for creating instances
pub struct PluginFactory {
    plugins: Arc<RwLock<HashMap<String, Box<dyn KernelApproximationPlugin>>>>,
}

impl Default for PluginFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginFactory {
    /// Create a new plugin factory
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a plugin
    pub fn register_plugin(
        &self,
        plugin: Box<dyn KernelApproximationPlugin>,
    ) -> std::result::Result<(), PluginError> {
        let metadata = plugin.metadata();
        let mut plugins = self.plugins.write().unwrap();

        if plugins.contains_key(&metadata.name) {
            return Err(PluginError::PluginAlreadyRegistered {
                name: metadata.name,
            });
        }

        plugins.insert(metadata.name.clone(), plugin);
        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister_plugin(&self, name: &str) -> std::result::Result<(), PluginError> {
        let mut plugins = self.plugins.write().unwrap();
        plugins
            .remove(name)
            .ok_or_else(|| PluginError::PluginNotFound {
                name: name.to_string(),
            })?;
        Ok(())
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(
        &self,
        name: &str,
    ) -> std::result::Result<PluginMetadata, PluginError> {
        let plugins = self.plugins.read().unwrap();
        let plugin = plugins
            .get(name)
            .ok_or_else(|| PluginError::PluginNotFound {
                name: name.to_string(),
            })?;
        Ok(plugin.metadata())
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        let plugins = self.plugins.read().unwrap();
        plugins.values().map(|p| p.metadata()).collect()
    }

    /// Create an instance of a plugin
    pub fn create_instance(
        &self,
        name: &str,
        config: PluginConfig,
    ) -> std::result::Result<Box<dyn KernelApproximationInstance>, PluginError> {
        let plugins = self.plugins.read().unwrap();
        let plugin = plugins
            .get(name)
            .ok_or_else(|| PluginError::PluginNotFound {
                name: name.to_string(),
            })?;

        plugin.validate_config(&config)?;
        plugin.create(config)
    }

    /// Get default configuration for a plugin
    pub fn get_default_config(&self, name: &str) -> std::result::Result<PluginConfig, PluginError> {
        let plugins = self.plugins.read().unwrap();
        let plugin = plugins
            .get(name)
            .ok_or_else(|| PluginError::PluginNotFound {
                name: name.to_string(),
            })?;
        Ok(plugin.default_config())
    }
}

/// Wrapper to make plugin instances compatible with sklears traits
pub struct PluginWrapper {
    instance: Box<dyn KernelApproximationInstance>,
    metadata: PluginMetadata,
}

impl PluginWrapper {
    /// Create a new plugin wrapper
    pub fn new(instance: Box<dyn KernelApproximationInstance>, metadata: PluginMetadata) -> Self {
        Self { instance, metadata }
    }

    /// Get plugin metadata
    pub fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    /// Get the underlying instance
    pub fn instance(&self) -> &dyn KernelApproximationInstance {
        self.instance.as_ref()
    }

    /// Get mutable access to the underlying instance
    pub fn instance_mut(&mut self) -> &mut dyn KernelApproximationInstance {
        self.instance.as_mut()
    }
}

impl Clone for PluginWrapper {
    fn clone(&self) -> Self {
        Self {
            instance: self.instance.clone_instance(),
            metadata: self.metadata.clone(),
        }
    }
}

impl Fit<Array2<f64>, ()> for PluginWrapper {
    type Fitted = FittedPluginWrapper;

    fn fit(mut self, x: &Array2<f64>, y: &()) -> Result<Self::Fitted, SklearsError> {
        self.instance
            .fit(x, y)
            .map_err(|e| SklearsError::InvalidInput(format!("{}", e)))?;
        Ok(FittedPluginWrapper {
            instance: self.instance,
            metadata: self.metadata,
        })
    }
}

/// Fitted plugin wrapper
pub struct FittedPluginWrapper {
    instance: Box<dyn KernelApproximationInstance>,
    metadata: PluginMetadata,
}

impl FittedPluginWrapper {
    /// Get plugin metadata
    pub fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    /// Get the underlying instance
    pub fn instance(&self) -> &dyn KernelApproximationInstance {
        self.instance.as_ref()
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> Option<usize> {
        self.instance.n_output_features()
    }
}

impl Clone for FittedPluginWrapper {
    fn clone(&self) -> Self {
        Self {
            instance: self.instance.clone_instance(),
            metadata: self.metadata.clone(),
        }
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedPluginWrapper {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        self.instance
            .transform(x)
            .map_err(|e| SklearsError::InvalidInput(format!("{}", e)))
    }
}

/// Global plugin registry
static GLOBAL_FACTORY: std::sync::LazyLock<PluginFactory> =
    std::sync::LazyLock::new(PluginFactory::new);

/// Register a plugin globally
pub fn register_global_plugin(
    plugin: Box<dyn KernelApproximationPlugin>,
) -> std::result::Result<(), PluginError> {
    GLOBAL_FACTORY.register_plugin(plugin)
}

/// Create an instance from the global registry
pub fn create_global_plugin_instance(
    name: &str,
    config: PluginConfig,
) -> std::result::Result<PluginWrapper, PluginError> {
    let instance = GLOBAL_FACTORY.create_instance(name, config)?;
    let metadata = GLOBAL_FACTORY.get_plugin_metadata(name)?;
    Ok(PluginWrapper::new(instance, metadata))
}

/// List all globally registered plugins
pub fn list_global_plugins() -> Vec<PluginMetadata> {
    GLOBAL_FACTORY.list_plugins()
}

/// Example plugin implementing a simple linear kernel approximation
pub struct LinearKernelPlugin;

impl KernelApproximationPlugin for LinearKernelPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "linear_kernel".to_string(),
            version: "1.0.0".to_string(),
            description: "Simple linear kernel approximation plugin".to_string(),
            author: "sklears".to_string(),
            supported_kernels: vec!["linear".to_string()],
            required_parameters: vec!["n_components".to_string()],
            optional_parameters: vec!["normalize".to_string()],
        }
    }

    fn create(
        &self,
        config: PluginConfig,
    ) -> std::result::Result<Box<dyn KernelApproximationInstance>, PluginError> {
        let n_components = config
            .parameters
            .get("n_components")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| PluginError::InvalidConfiguration {
                message: "n_components parameter required".to_string(),
            })? as usize;

        let normalize = config
            .parameters
            .get("normalize")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Box::new(LinearKernelInstance {
            n_components,
            normalize,
            projection_matrix: None,
        }))
    }

    fn validate_config(&self, config: &PluginConfig) -> std::result::Result<(), PluginError> {
        if !config.parameters.contains_key("n_components") {
            return Err(PluginError::InvalidConfiguration {
                message: "n_components parameter is required".to_string(),
            });
        }

        if let Some(n_comp) = config
            .parameters
            .get("n_components")
            .and_then(|v| v.as_u64())
        {
            if n_comp == 0 {
                return Err(PluginError::InvalidConfiguration {
                    message: "n_components must be greater than 0".to_string(),
                });
            }
        } else {
            return Err(PluginError::InvalidConfiguration {
                message: "n_components must be a positive integer".to_string(),
            });
        }

        Ok(())
    }

    fn default_config(&self) -> PluginConfig {
        let mut config = PluginConfig::default();
        config.parameters.insert(
            "n_components".to_string(),
            serde_json::Value::Number(100.into()),
        );
        config
            .parameters
            .insert("normalize".to_string(), serde_json::Value::Bool(false));
        config
    }
}

/// Linear kernel instance
pub struct LinearKernelInstance {
    n_components: usize,
    normalize: bool,
    projection_matrix: Option<Array2<f64>>,
}

impl KernelApproximationInstance for LinearKernelInstance {
    fn fit(&mut self, x: &Array2<f64>, _y: &()) -> std::result::Result<(), PluginError> {
        use scirs2_core::random::thread_rng;
        use scirs2_core::random::StandardNormal;

        let (_, n_features) = x.dim();
        let mut rng = thread_rng();

        // Create random projection matrix
        let mut proj_matrix = Array2::zeros((n_features, self.n_components));
        for elem in proj_matrix.iter_mut() {
            *elem = rng.sample(StandardNormal);
        }

        if self.normalize {
            // Normalize columns to unit length
            for j in 0..self.n_components {
                let mut col = proj_matrix.column_mut(j);
                let norm = col.mapv(|x: f64| x * x).sum().sqrt();
                if norm > 1e-8 {
                    col /= norm;
                }
            }
        }

        self.projection_matrix = Some(proj_matrix);
        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> std::result::Result<Array2<f64>, PluginError> {
        let proj_matrix =
            self.projection_matrix
                .as_ref()
                .ok_or_else(|| PluginError::InitializationFailed {
                    message: "Plugin not fitted".to_string(),
                })?;

        Ok(x.dot(proj_matrix))
    }

    fn is_fitted(&self) -> bool {
        self.projection_matrix.is_some()
    }

    fn n_output_features(&self) -> Option<usize> {
        if self.is_fitted() {
            Some(self.n_components)
        } else {
            None
        }
    }

    fn clone_instance(&self) -> Box<dyn KernelApproximationInstance> {
        Box::new(LinearKernelInstance {
            n_components: self.n_components,
            normalize: self.normalize,
            projection_matrix: self.projection_matrix.clone(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_plugin_registration() {
        let factory = PluginFactory::new();
        let plugin = Box::new(LinearKernelPlugin);

        assert!(factory.register_plugin(plugin).is_ok());

        let plugins = factory.list_plugins();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "linear_kernel");
    }

    #[test]
    fn test_plugin_instance_creation() {
        let factory = PluginFactory::new();
        let plugin = Box::new(LinearKernelPlugin);
        factory.register_plugin(plugin).unwrap();

        let mut config = PluginConfig::default();
        config.parameters.insert(
            "n_components".to_string(),
            serde_json::Value::Number(50.into()),
        );

        let instance = factory.create_instance("linear_kernel", config);
        assert!(instance.is_ok());
    }

    #[test]
    fn test_plugin_wrapper_fit_transform() {
        let factory = PluginFactory::new();
        let plugin = Box::new(LinearKernelPlugin);
        factory.register_plugin(plugin).unwrap();

        let mut config = PluginConfig::default();
        config.parameters.insert(
            "n_components".to_string(),
            serde_json::Value::Number(30.into()),
        );

        let instance = factory.create_instance("linear_kernel", config).unwrap();
        let metadata = factory.get_plugin_metadata("linear_kernel").unwrap();
        let wrapper = PluginWrapper::new(instance, metadata);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = wrapper.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[3, 30]);
    }

    #[test]
    fn test_global_plugin_registry() {
        let plugin = Box::new(LinearKernelPlugin);
        assert!(register_global_plugin(plugin).is_ok());

        let plugins = list_global_plugins();
        assert!(!plugins.is_empty());
    }

    #[test]
    fn test_invalid_configuration() {
        let factory = PluginFactory::new();
        let plugin = Box::new(LinearKernelPlugin);
        factory.register_plugin(plugin).unwrap();

        let config = PluginConfig::default(); // Missing n_components
        let result = factory.create_instance("linear_kernel", config);
        assert!(result.is_err());
    }
}
