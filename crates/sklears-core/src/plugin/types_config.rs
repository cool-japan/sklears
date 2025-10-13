//! Plugin Types and Configuration
//!
//! This module defines the core types, enumerations, and configuration structures
//! used throughout the plugin system. It provides type-safe parameter handling,
//! metadata definitions, and runtime configuration options.

use crate::error::{Result, SklearsError};
use std::any::TypeId;
use std::collections::HashMap;

/// Metadata describing a plugin
///
/// This structure contains comprehensive information about a plugin including
/// its identity, capabilities, dependencies, and requirements. It's used by
/// the plugin system for discovery, validation, and compatibility checking.
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::{PluginMetadata, PluginCategory, PluginCapability};
/// use std::any::TypeId;
///
/// let metadata = PluginMetadata {
///     name: "LinearRegression".to_string(),
///     version: "1.0.0".to_string(),
///     description: "Linear regression algorithm".to_string(),
///     author: "SKLears Team".to_string(),
///     category: PluginCategory::Algorithm,
///     supported_types: vec![TypeId::of::<f64>()],
///     dependencies: vec!["ndarray".to_string()],
///     capabilities: vec![PluginCapability::Parallel],
///     min_sdk_version: "0.1.0".to_string(),
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct PluginMetadata {
    /// Plugin name - should be unique within a category
    pub name: String,
    /// Plugin version following semantic versioning
    pub version: String,
    /// Human-readable description of the plugin's functionality
    pub description: String,
    /// Plugin author or organization
    pub author: String,
    /// Plugin category for organization and discovery
    pub category: PluginCategory,
    /// List of supported input data types
    #[serde(skip)]
    pub supported_types: Vec<TypeId>,
    /// Required external dependencies
    pub dependencies: Vec<String>,
    /// Plugin capabilities and features
    pub capabilities: Vec<PluginCapability>,
    /// Minimum SDK version required for this plugin
    pub min_sdk_version: String,
}

impl Default for PluginMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "1.0.0".to_string(),
            description: String::new(),
            author: String::new(),
            category: PluginCategory::Algorithm,
            supported_types: Vec::new(),
            dependencies: Vec::new(),
            capabilities: Vec::new(),
            min_sdk_version: "0.1.0".to_string(),
        }
    }
}

/// Plugin categories for organization
///
/// Categories help organize plugins in the registry and enable
/// category-based discovery and filtering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub enum PluginCategory {
    /// Machine learning algorithms (classifiers, regressors, clustering)
    Algorithm,
    /// Data transformers (scalers, encoders, feature selectors)
    Transformer,
    /// Data loaders and processors (I/O, parsing, validation)
    DataProcessor,
    /// Metrics and evaluators (accuracy, loss functions, validators)
    Evaluator,
    /// Visualization tools (plotters, dashboards, reporters)
    Visualizer,
    /// Custom category with user-defined name
    Custom(String),
}

/// Plugin capabilities
///
/// Capabilities describe what features and optimizations a plugin supports.
/// This enables the plugin system to make intelligent decisions about
/// plugin selection and execution.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum PluginCapability {
    /// Supports parallel processing across multiple threads
    Parallel,
    /// Supports streaming data processing
    Streaming,
    /// Supports GPU acceleration (CUDA, OpenCL, etc.)
    GpuAccelerated,
    /// Supports online/incremental learning
    OnlineLearning,
    /// Supports sparse data formats efficiently
    SparseData,
    /// Can handle missing values gracefully
    MissingValues,
    /// Supports categorical data natively
    CategoricalData,
    /// Provides interpretability features (feature importance, explanations)
    Interpretable,
    /// Custom capability with user-defined name
    Custom(String),
}

/// Configuration for plugins
///
/// This structure contains all the configuration information needed
/// to initialize and run a plugin, including parameters, runtime settings,
/// and plugin-specific configurations.
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::{PluginConfig, PluginParameter, RuntimeSettings, LogLevel};
/// use std::collections::HashMap;
///
/// let mut config = PluginConfig::default();
/// config.parameters.insert(
///     "learning_rate".to_string(),
///     PluginParameter::Float(0.01)
/// );
/// config.runtime_settings.use_gpu = true;
/// config.runtime_settings.log_level = LogLevel::Debug;
/// ```
#[derive(Debug, Clone, Default)]
pub struct PluginConfig {
    /// Algorithm-specific parameters
    pub parameters: HashMap<String, PluginParameter>,
    /// Runtime execution settings
    pub runtime_settings: RuntimeSettings,
    /// Plugin-specific configuration settings
    pub plugin_settings: HashMap<String, String>,
}

/// Runtime settings for plugin execution
///
/// These settings control how plugins are executed, including resource
/// limits, performance optimizations, and logging configuration.
#[derive(Debug, Clone)]
pub struct RuntimeSettings {
    /// Number of threads to use (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Memory limit in bytes (None = no limit)
    pub memory_limit: Option<usize>,
    /// Timeout for operations in milliseconds (None = no timeout)
    pub timeout_ms: Option<u64>,
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
    /// Logging level for plugin operations
    pub log_level: LogLevel,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            num_threads: None,
            memory_limit: None,
            timeout_ms: None,
            use_gpu: false,
            log_level: LogLevel::Info,
        }
    }
}

/// Logging levels for plugins
///
/// Defines the verbosity of logging output from plugin operations.
/// Higher levels include all lower level messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogLevel {
    /// Only log errors
    Error,
    /// Log warnings and errors
    Warn,
    /// Log info, warnings, and errors (default)
    Info,
    /// Log debug info and all higher levels
    Debug,
    /// Log everything including trace information
    Trace,
}

/// Plugin parameter types
///
/// This enum provides type-safe parameter handling for plugin configuration.
/// It supports common parameter types and nested structures for complex
/// configurations.
///
/// # Examples
///
/// ```rust
/// use sklears_core::plugin::PluginParameter;
/// use std::collections::HashMap;
///
/// // Simple parameters
/// let learning_rate = PluginParameter::Float(0.01);
/// let max_iterations = PluginParameter::Int(1000);
/// let use_bias = PluginParameter::Bool(true);
/// let algorithm_name = PluginParameter::String("adam".to_string());
///
/// // Array parameters
/// let layer_sizes = PluginParameter::IntArray(vec![100, 50, 10]);
/// let dropout_rates = PluginParameter::FloatArray(vec![0.2, 0.3, 0.5]);
///
/// // Nested parameters
/// let mut optimizer_config = HashMap::new();
/// optimizer_config.insert("type".to_string(), PluginParameter::String("adam".to_string()));
/// optimizer_config.insert("beta1".to_string(), PluginParameter::Float(0.9));
/// optimizer_config.insert("beta2".to_string(), PluginParameter::Float(0.999));
/// let optimizer = PluginParameter::Object(optimizer_config);
/// ```
#[derive(Debug, Clone)]
pub enum PluginParameter {
    /// Integer parameter
    Int(i64),
    /// Floating-point parameter
    Float(f64),
    /// String parameter
    String(String),
    /// Boolean parameter
    Bool(bool),
    /// Array of integers
    IntArray(Vec<i64>),
    /// Array of floats
    FloatArray(Vec<f64>),
    /// Array of strings
    StringArray(Vec<String>),
    /// Nested parameters for complex configurations
    Object(HashMap<String, PluginParameter>),
}

impl PluginParameter {
    /// Try to extract an integer value
    ///
    /// # Returns
    ///
    /// The integer value if the parameter is an Int, otherwise an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginParameter;
    ///
    /// let param = PluginParameter::Int(42);
    /// assert_eq!(param.as_int().unwrap(), 42);
    ///
    /// let param = PluginParameter::String("not a number".to_string());
    /// assert!(param.as_int().is_err());
    /// ```
    pub fn as_int(&self) -> Result<i64> {
        match self {
            PluginParameter::Int(v) => Ok(*v),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not an integer".to_string(),
            )),
        }
    }

    /// Try to extract a float value
    ///
    /// This method also supports converting integers to floats.
    ///
    /// # Returns
    ///
    /// The float value if the parameter is numeric, otherwise an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginParameter;
    ///
    /// let param = PluginParameter::Float(3.14);
    /// assert_eq!(param.as_float().unwrap(), 3.14);
    ///
    /// let param = PluginParameter::Int(42);
    /// assert_eq!(param.as_float().unwrap(), 42.0);
    ///
    /// let param = PluginParameter::Bool(true);
    /// assert!(param.as_float().is_err());
    /// ```
    pub fn as_float(&self) -> Result<f64> {
        match self {
            PluginParameter::Float(v) => Ok(*v),
            PluginParameter::Int(v) => Ok(*v as f64),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not numeric".to_string(),
            )),
        }
    }

    /// Try to extract a string value
    ///
    /// # Returns
    ///
    /// A reference to the string value if the parameter is a String, otherwise an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginParameter;
    ///
    /// let param = PluginParameter::String("hello".to_string());
    /// assert_eq!(param.as_string().unwrap(), "hello");
    ///
    /// let param = PluginParameter::Int(42);
    /// assert!(param.as_string().is_err());
    /// ```
    pub fn as_string(&self) -> Result<&str> {
        match self {
            PluginParameter::String(v) => Ok(v),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not a string".to_string(),
            )),
        }
    }

    /// Try to extract a boolean value
    ///
    /// # Returns
    ///
    /// The boolean value if the parameter is a Bool, otherwise an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginParameter;
    ///
    /// let param = PluginParameter::Bool(true);
    /// assert_eq!(param.as_bool().unwrap(), true);
    ///
    /// let param = PluginParameter::String("true".to_string());
    /// assert!(param.as_bool().is_err());
    /// ```
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            PluginParameter::Bool(v) => Ok(*v),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not a boolean".to_string(),
            )),
        }
    }

    /// Try to extract an integer array
    ///
    /// # Returns
    ///
    /// A reference to the integer array if the parameter is an IntArray, otherwise an error.
    pub fn as_int_array(&self) -> Result<&Vec<i64>> {
        match self {
            PluginParameter::IntArray(v) => Ok(v),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not an integer array".to_string(),
            )),
        }
    }

    /// Try to extract a float array
    ///
    /// # Returns
    ///
    /// A reference to the float array if the parameter is a FloatArray, otherwise an error.
    pub fn as_float_array(&self) -> Result<&Vec<f64>> {
        match self {
            PluginParameter::FloatArray(v) => Ok(v),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not a float array".to_string(),
            )),
        }
    }

    /// Try to extract a string array
    ///
    /// # Returns
    ///
    /// A reference to the string array if the parameter is a StringArray, otherwise an error.
    pub fn as_string_array(&self) -> Result<&Vec<String>> {
        match self {
            PluginParameter::StringArray(v) => Ok(v),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not a string array".to_string(),
            )),
        }
    }

    /// Try to extract a nested object
    ///
    /// # Returns
    ///
    /// A reference to the nested parameter map if the parameter is an Object, otherwise an error.
    pub fn as_object(&self) -> Result<&HashMap<String, PluginParameter>> {
        match self {
            PluginParameter::Object(v) => Ok(v),
            _ => Err(SklearsError::InvalidOperation(
                "Parameter is not an object".to_string(),
            )),
        }
    }

    /// Check if the parameter is of a specific type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sklears_core::plugin::PluginParameter;
    ///
    /// let param = PluginParameter::Float(3.14);
    /// assert!(param.is_float());
    /// assert!(!param.is_int());
    /// ```
    pub fn is_int(&self) -> bool {
        matches!(self, PluginParameter::Int(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, PluginParameter::Float(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, PluginParameter::String(_))
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, PluginParameter::Bool(_))
    }

    pub fn is_int_array(&self) -> bool {
        matches!(self, PluginParameter::IntArray(_))
    }

    pub fn is_float_array(&self) -> bool {
        matches!(self, PluginParameter::FloatArray(_))
    }

    pub fn is_string_array(&self) -> bool {
        matches!(self, PluginParameter::StringArray(_))
    }

    pub fn is_object(&self) -> bool {
        matches!(self, PluginParameter::Object(_))
    }

    /// Get the type name of the parameter
    ///
    /// # Returns
    ///
    /// A string describing the type of the parameter.
    pub fn type_name(&self) -> &'static str {
        match self {
            PluginParameter::Int(_) => "int",
            PluginParameter::Float(_) => "float",
            PluginParameter::String(_) => "string",
            PluginParameter::Bool(_) => "bool",
            PluginParameter::IntArray(_) => "int_array",
            PluginParameter::FloatArray(_) => "float_array",
            PluginParameter::StringArray(_) => "string_array",
            PluginParameter::Object(_) => "object",
        }
    }
}

/// Convenience functions for creating plugin parameters
impl PluginParameter {
    /// Create an integer parameter
    pub fn int(value: i64) -> Self {
        PluginParameter::Int(value)
    }

    /// Create a float parameter
    pub fn float(value: f64) -> Self {
        PluginParameter::Float(value)
    }

    /// Create a string parameter
    pub fn string(value: impl Into<String>) -> Self {
        PluginParameter::String(value.into())
    }

    /// Create a boolean parameter
    pub fn bool(value: bool) -> Self {
        PluginParameter::Bool(value)
    }

    /// Create an integer array parameter
    pub fn int_array(value: Vec<i64>) -> Self {
        PluginParameter::IntArray(value)
    }

    /// Create a float array parameter
    pub fn float_array(value: Vec<f64>) -> Self {
        PluginParameter::FloatArray(value)
    }

    /// Create a string array parameter
    pub fn string_array(value: Vec<String>) -> Self {
        PluginParameter::StringArray(value)
    }

    /// Create an object parameter
    pub fn object(value: HashMap<String, PluginParameter>) -> Self {
        PluginParameter::Object(value)
    }
}
