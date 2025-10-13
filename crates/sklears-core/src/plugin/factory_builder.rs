//! Plugin Factory and Configuration Builder
//!
//! This module provides factory patterns and builders for creating plugin instances
//! and configurations. It enables flexible plugin creation with configurable
//! parameters and runtime settings.

use super::core_traits::Plugin;
use super::types_config::{LogLevel, PluginConfig, PluginMetadata, PluginParameter};
use crate::error::Result;

/// Factory for creating plugin instances
///
/// The PluginFactory trait provides a standardized interface for creating
/// plugin instances with specific configurations. This enables dynamic
/// plugin creation and configuration management.
///
/// # Examples
///
/// ```rust,no_run
/// use sklears_core::plugin::{PluginFactory, Plugin, PluginConfig, PluginMetadata};
/// use sklears_core::error::Result;
///
/// struct LinearRegressionFactory;
///
/// impl PluginFactory for LinearRegressionFactory {
///     fn create_plugin(&self, config: &PluginConfig) -> Result<Box<dyn Plugin>> {
///         // Create and configure the plugin based on the provided config
///         // Box::new(LinearRegressionPlugin::new(config))
///         todo!("Implement plugin creation")
///     }
///
///     fn metadata(&self) -> PluginMetadata {
///         PluginMetadata {
///             name: "LinearRegression".to_string(),
///             version: "1.0.0".to_string(),
///             description: "Linear regression algorithm".to_string(),
///             ..Default::default()
///         }
///     }
///
///     fn validate_config(&self, config: &PluginConfig) -> Result<()> {
///         // Validate configuration parameters
///         Ok(())
///     }
/// }
/// ```
pub trait PluginFactory: Send + Sync {
    /// Create a new plugin instance
    ///
    /// This method creates a new plugin instance configured according to
    /// the provided configuration. The factory should validate the configuration
    /// and return an error if the plugin cannot be created with the given settings.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the plugin instance
    ///
    /// # Returns
    ///
    /// A boxed plugin instance, or an error if creation fails.
    fn create_plugin(&self, config: &PluginConfig) -> Result<Box<dyn Plugin>>;

    /// Get plugin metadata
    ///
    /// Returns metadata describing the plugins that this factory can create.
    /// This includes information about capabilities, supported types, and
    /// configuration requirements.
    ///
    /// # Returns
    ///
    /// Metadata for plugins created by this factory.
    fn metadata(&self) -> PluginMetadata;

    /// Validate configuration
    ///
    /// Validates that the provided configuration is suitable for creating
    /// a plugin instance. This should check parameter types, ranges, and
    /// any dependencies or requirements.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to validate
    ///
    /// # Returns
    ///
    /// Ok(()) if the configuration is valid, or an error describing
    /// what is invalid.
    fn validate_config(&self, config: &PluginConfig) -> Result<()>;

    /// Get default configuration
    ///
    /// Returns a default configuration that can be used to create a plugin
    /// instance with sensible defaults. This provides a starting point for
    /// configuration customization.
    ///
    /// # Returns
    ///
    /// A default plugin configuration.
    fn default_config(&self) -> PluginConfig {
        PluginConfig::default()
    }

    /// Get configuration schema
    ///
    /// Returns information about the configuration parameters that this
    /// factory accepts, including their types, ranges, and descriptions.
    /// This can be used for automatic UI generation or documentation.
    ///
    /// # Returns
    ///
    /// A map of parameter names to their descriptions and constraints.
    fn config_schema(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

/// Builder for creating plugin configurations
///
/// The PluginConfigBuilder provides a fluent interface for constructing
/// plugin configurations with various parameters and runtime settings.
/// It ensures type safety and provides convenient methods for common
/// configuration patterns.
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::{PluginConfigBuilder, PluginParameter, LogLevel};
///
/// let config = PluginConfigBuilder::new()
///     .with_parameter("learning_rate", PluginParameter::Float(0.01))
///     .with_parameter("max_iterations", PluginParameter::Int(1000))
///     .with_parameter("use_bias", PluginParameter::Bool(true))
///     .with_threads(4)
///     .with_memory_limit(1024 * 1024 * 1024) // 1GB
///     .with_timeout(30000) // 30 seconds
///     .with_gpu(true)
///     .with_log_level(LogLevel::Info)
///     .with_setting("backend", "cuda")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct PluginConfigBuilder {
    /// The configuration being built
    config: PluginConfig,
}

impl PluginConfigBuilder {
    /// Create a new config builder
    ///
    /// Initializes a new builder with default configuration values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginConfigBuilder;
    ///
    /// let builder = PluginConfigBuilder::new();
    /// let config = builder.build();
    /// ```
    pub fn new() -> Self {
        Self {
            config: PluginConfig::default(),
        }
    }

    /// Add a parameter to the configuration
    ///
    /// Adds a named parameter with the specified value to the configuration.
    /// This method can be chained to add multiple parameters.
    ///
    /// # Arguments
    ///
    /// * `key` - The parameter name
    /// * `value` - The parameter value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::{PluginConfigBuilder, PluginParameter};
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_parameter("learning_rate", PluginParameter::Float(0.01))
    ///     .with_parameter("regularization", PluginParameter::Float(0.001))
    ///     .build();
    /// ```
    pub fn with_parameter(mut self, key: &str, value: PluginParameter) -> Self {
        self.config.parameters.insert(key.to_string(), value);
        self
    }

    /// Add multiple parameters at once
    ///
    /// Convenience method for adding multiple parameters from a map.
    ///
    /// # Arguments
    ///
    /// * `params` - Map of parameter names to values
    pub fn with_parameters(
        mut self,
        params: std::collections::HashMap<String, PluginParameter>,
    ) -> Self {
        self.config.parameters.extend(params);
        self
    }

    /// Set the number of threads to use
    ///
    /// Configures the number of threads that the plugin should use for
    /// parallel processing. If not set, the plugin will use its default
    /// threading behavior.
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginConfigBuilder;
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_threads(8)
    ///     .build();
    /// ```
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.config.runtime_settings.num_threads = Some(threads);
        self
    }

    /// Set memory limit in bytes
    ///
    /// Configures the maximum amount of memory that the plugin should use.
    /// This can help prevent out-of-memory errors in resource-constrained
    /// environments.
    ///
    /// # Arguments
    ///
    /// * `limit` - Memory limit in bytes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginConfigBuilder;
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_memory_limit(2 * 1024 * 1024 * 1024) // 2GB
    ///     .build();
    /// ```
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.config.runtime_settings.memory_limit = Some(limit);
        self
    }

    /// Set timeout in milliseconds
    ///
    /// Configures the maximum time that plugin operations should take
    /// before timing out. This can help prevent hanging operations.
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - Timeout in milliseconds
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginConfigBuilder;
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_timeout(60000) // 1 minute
    ///     .build();
    /// ```
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.runtime_settings.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable or disable GPU acceleration
    ///
    /// Configures whether the plugin should use GPU acceleration if available.
    ///
    /// # Arguments
    ///
    /// * `use_gpu` - Whether to use GPU acceleration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginConfigBuilder;
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_gpu(true)
    ///     .build();
    /// ```
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.config.runtime_settings.use_gpu = use_gpu;
        self
    }

    /// Set logging level
    ///
    /// Configures the verbosity of logging output from the plugin.
    ///
    /// # Arguments
    ///
    /// * `level` - The logging level to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::{PluginConfigBuilder, LogLevel};
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_log_level(LogLevel::Debug)
    ///     .build();
    /// ```
    pub fn with_log_level(mut self, level: LogLevel) -> Self {
        self.config.runtime_settings.log_level = level;
        self
    }

    /// Add a plugin-specific setting
    ///
    /// Adds a custom setting that is specific to the plugin being configured.
    /// These settings are typically used for plugin-specific configuration
    /// that doesn't fit into the standard parameter system.
    ///
    /// # Arguments
    ///
    /// * `key` - The setting name
    /// * `value` - The setting value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginConfigBuilder;
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_setting("backend", "tensorflow")
    ///     .with_setting("device", "/gpu:0")
    ///     .build();
    /// ```
    pub fn with_setting(mut self, key: &str, value: &str) -> Self {
        self.config
            .plugin_settings
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Add multiple settings at once
    ///
    /// Convenience method for adding multiple plugin-specific settings.
    ///
    /// # Arguments
    ///
    /// * `settings` - Map of setting names to values
    pub fn with_settings(mut self, settings: std::collections::HashMap<String, String>) -> Self {
        self.config.plugin_settings.extend(settings);
        self
    }

    /// Build the final configuration
    ///
    /// Consumes the builder and returns the constructed configuration.
    ///
    /// # Returns
    ///
    /// The configured PluginConfig instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::{PluginConfigBuilder, PluginParameter};
    ///
    /// let config = PluginConfigBuilder::new()
    ///     .with_parameter("learning_rate", PluginParameter::Float(0.01))
    ///     .with_threads(4)
    ///     .build();
    /// ```
    pub fn build(self) -> PluginConfig {
        self.config
    }

    /// Get a reference to the current configuration
    ///
    /// Returns a reference to the configuration being built without
    /// consuming the builder. This can be useful for inspecting the
    /// current state during construction.
    ///
    /// # Returns
    ///
    /// A reference to the current configuration.
    pub fn config(&self) -> &PluginConfig {
        &self.config
    }

    /// Validate the current configuration
    ///
    /// Validates the current configuration state without building it.
    /// This can be useful for checking configuration validity during
    /// the building process.
    ///
    /// # Returns
    ///
    /// Ok(()) if the configuration is valid, or an error describing
    /// what is invalid.
    pub fn validate(&self) -> Result<()> {
        // Basic validation - can be extended
        if self.config.runtime_settings.num_threads == Some(0) {
            return Err(crate::error::SklearsError::InvalidOperation(
                "Number of threads cannot be zero".to_string(),
            ));
        }

        if let Some(timeout) = self.config.runtime_settings.timeout_ms {
            if timeout == 0 {
                return Err(crate::error::SklearsError::InvalidOperation(
                    "Timeout cannot be zero".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Clone the builder
    ///
    /// Creates a copy of the current builder state, allowing for
    /// branching configuration construction.
    ///
    /// # Returns
    ///
    /// A cloned builder with the same configuration state.
    pub fn clone_builder(&self) -> Self {
        self.clone()
    }
}

impl Default for PluginConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common plugin configurations
impl PluginConfigBuilder {
    /// Create a configuration optimized for CPU-intensive tasks
    ///
    /// Sets up a configuration with appropriate threading and resource
    /// settings for CPU-bound algorithms.
    pub fn cpu_optimized() -> Self {
        Self::new()
            .with_threads(num_cpus::get())
            .with_gpu(false)
            .with_log_level(LogLevel::Info)
    }

    /// Create a configuration optimized for GPU acceleration
    ///
    /// Sets up a configuration with GPU acceleration enabled and
    /// appropriate resource settings.
    pub fn gpu_optimized() -> Self {
        Self::new()
            .with_gpu(true)
            .with_threads(2) // Fewer CPU threads when using GPU
            .with_log_level(LogLevel::Info)
    }

    /// Create a configuration for development/debugging
    ///
    /// Sets up a configuration with verbose logging and longer timeouts
    /// suitable for development environments.
    pub fn development() -> Self {
        Self::new()
            .with_log_level(LogLevel::Debug)
            .with_timeout(300000) // 5 minutes
            .with_threads(1) // Single-threaded for easier debugging
    }

    /// Create a configuration for production environments
    ///
    /// Sets up a configuration with optimized settings for production use,
    /// including appropriate resource limits and logging levels.
    pub fn production() -> Self {
        Self::new()
            .with_log_level(LogLevel::Warn)
            .with_timeout(30000) // 30 seconds
            .with_memory_limit(4 * 1024 * 1024 * 1024) // 4GB
            .with_threads(num_cpus::get())
    }
}
