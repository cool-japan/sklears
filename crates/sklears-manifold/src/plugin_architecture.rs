//! Plugin architecture for custom manifold learning methods
//!
//! This module provides a framework for creating custom manifold learning algorithms
//! that integrate seamlessly with the existing sklears-manifold ecosystem.

use scirs2_core::ndarray::{Array2, ArrayView2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Registry for custom manifold learning plugins
static PLUGIN_REGISTRY: once_cell::sync::Lazy<RwLock<PluginRegistry>> =
    once_cell::sync::Lazy::new(|| RwLock::new(PluginRegistry::new()));

/// Trait for custom manifold learning plugins
pub trait ManifoldPlugin: Send + Sync + Debug {
    /// Get the name of the plugin
    fn name(&self) -> &str;

    /// Get the version of the plugin
    fn version(&self) -> &str;

    /// Get a description of the plugin
    fn description(&self) -> &str;

    /// Get the author(s) of the plugin
    fn author(&self) -> &str;

    /// Create a new instance of the plugin with default parameters
    fn create_default(&self) -> Box<dyn CustomManifoldLearner>;

    /// Create a new instance of the plugin with custom parameters
    fn create_with_params(
        &self,
        params: &PluginParameters,
    ) -> SklResult<Box<dyn CustomManifoldLearner>>;

    /// Get the default parameters for this plugin
    fn default_parameters(&self) -> PluginParameters;

    /// Validate parameters for this plugin
    fn validate_parameters(&self, params: &PluginParameters) -> SklResult<()>;

    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata {
        /// PluginMetadata
        PluginMetadata {
            name: self.name().to_string(),
            version: self.version().to_string(),
            description: self.description().to_string(),
            author: self.author().to_string(),
            supported_features: self.supported_features(),
            parameter_schema: self.parameter_schema(),
        }
    }

    /// Get supported features of this plugin
    fn supported_features(&self) -> Vec<PluginFeature> {
        vec![PluginFeature::DimensionalityReduction]
    }

    /// Get parameter schema for validation and documentation
    fn parameter_schema(&self) -> Vec<ParameterDefinition>;
}

/// Trait for custom manifold learning implementations
pub trait CustomManifoldLearner: Send + Sync + Debug {
    /// Set a parameter value
    fn set_parameter(&mut self, name: &str, value: ParameterValue) -> SklResult<()>;

    /// Get a parameter value
    fn get_parameter(&self, name: &str) -> Option<ParameterValue>;

    /// Get all parameters
    fn get_all_parameters(&self) -> HashMap<String, ParameterValue>;

    /// Fit the model to data
    fn fit(&mut self, x: &ArrayView2<Float>) -> SklResult<()>;

    /// Transform data using the fitted model
    fn transform(&self, x: &ArrayView2<Float>) -> SklResult<Array2<Float>>;

    /// Fit and transform data in one step
    fn fit_transform(&mut self, x: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Check if the model is fitted
    fn is_fitted(&self) -> bool;

    /// Get model metadata
    fn get_metadata(&self) -> CustomModelMetadata;

    /// Clone the learner
    fn clone_learner(&self) -> Box<dyn CustomManifoldLearner>;
}

/// Plugin registry for managing custom manifold learning plugins
#[derive(Debug)]
pub struct PluginRegistry {
    plugins: HashMap<String, Arc<dyn ManifoldPlugin>>,
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
            plugins: HashMap::new(),
        }
    }

    /// Register a new plugin
    pub fn register_plugin(&mut self, plugin: Arc<dyn ManifoldPlugin>) -> SklResult<()> {
        let name = plugin.name().to_string();

        if self.plugins.contains_key(&name) {
            return Err(SklearsError::InvalidInput(format!(
                "Plugin '{}' is already registered",
                name
            )));
        }

        self.plugins.insert(name, plugin);
        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister_plugin(&mut self, name: &str) -> SklResult<()> {
        if self.plugins.remove(name).is_none() {
            return Err(SklearsError::InvalidInput(format!(
                "Plugin '{}' is not registered",
                name
            )));
        }
        Ok(())
    }

    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Arc<dyn ManifoldPlugin>> {
        self.plugins.get(name).cloned()
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }

    /// Get metadata for all plugins
    pub fn get_all_metadata(&self) -> Vec<PluginMetadata> {
        self.plugins
            .values()
            .map(|plugin| plugin.metadata())
            .collect()
    }

    /// Create a new instance from a plugin
    pub fn create_instance(
        &self,
        name: &str,
        params: Option<&PluginParameters>,
    ) -> SklResult<Box<dyn CustomManifoldLearner>> {
        let plugin = self
            .get_plugin(name)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Plugin '{}' not found", name)))?;

        match params {
            Some(params) => plugin.create_with_params(params),
            None => Ok(plugin.create_default()),
        }
    }
}

/// Global functions for plugin management
impl PluginRegistry {
    /// Get the global plugin registry
    pub fn global() -> &'static RwLock<PluginRegistry> {
        &PLUGIN_REGISTRY
    }
}

/// Plugin parameters container
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PluginParameters {
    parameters: HashMap<String, ParameterValue>,
}

impl PluginParameters {
    /// Create new empty parameters
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    /// Set a parameter
    pub fn set<T: Into<ParameterValue>>(&mut self, name: &str, value: T) -> &mut Self {
        self.parameters.insert(name.to_string(), value.into());
        self
    }

    /// Get a parameter
    pub fn get(&self, name: &str) -> Option<&ParameterValue> {
        self.parameters.get(name)
    }

    /// Check if parameter exists
    pub fn contains(&self, name: &str) -> bool {
        self.parameters.contains_key(name)
    }

    /// Get all parameters
    pub fn all(&self) -> &HashMap<String, ParameterValue> {
        &self.parameters
    }

    /// Merge with another parameter set
    pub fn merge(&mut self, other: &PluginParameters) {
        for (key, value) in &other.parameters {
            self.parameters.insert(key.clone(), value.clone());
        }
    }
}

impl Default for PluginParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameter value types
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParameterValue {
    /// Int
    Int(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
    /// Bool
    Bool(bool),
    /// IntArray
    IntArray(Vec<i64>),
    /// FloatArray
    FloatArray(Vec<f64>),
    /// StringArray
    StringArray(Vec<String>),
}

impl From<i64> for ParameterValue {
    fn from(value: i64) -> Self {
        ParameterValue::Int(value)
    }
}

impl From<i32> for ParameterValue {
    fn from(value: i32) -> Self {
        ParameterValue::Int(value as i64)
    }
}

impl From<usize> for ParameterValue {
    fn from(value: usize) -> Self {
        ParameterValue::Int(value as i64)
    }
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        ParameterValue::Float(value)
    }
}

impl From<f32> for ParameterValue {
    fn from(value: f32) -> Self {
        ParameterValue::Float(value as f64)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        ParameterValue::String(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        ParameterValue::String(value.to_string())
    }
}

impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        ParameterValue::Bool(value)
    }
}

impl From<Vec<i64>> for ParameterValue {
    fn from(value: Vec<i64>) -> Self {
        ParameterValue::IntArray(value)
    }
}

impl From<Vec<f64>> for ParameterValue {
    fn from(value: Vec<f64>) -> Self {
        ParameterValue::FloatArray(value)
    }
}

impl From<Vec<String>> for ParameterValue {
    fn from(value: Vec<String>) -> Self {
        ParameterValue::StringArray(value)
    }
}

/// Plugin feature capabilities
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PluginFeature {
    /// DimensionalityReduction
    DimensionalityReduction,
    /// Clustering
    Clustering,
    /// Classification
    Classification,
    /// Regression
    Regression,
    /// Visualization
    Visualization,
    /// OutOfSample
    OutOfSample,
    /// IncrementalLearning
    IncrementalLearning,
    /// Parallelization
    Parallelization,
    /// GPU
    GPU,
}

/// Parameter definition for schema
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParameterDefinition {
    /// name
    pub name: String,
    /// param_type
    pub param_type: ParameterType,
    /// description
    pub description: String,
    /// default_value
    pub default_value: Option<ParameterValue>,
    /// required
    pub required: bool,
    /// constraints
    pub constraints: Option<ParameterConstraints>,
}

/// Parameter type specification
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParameterType {
    /// Int
    Int,
    /// Float
    Float,
    /// String
    String,
    /// Bool
    Bool,
    /// IntArray
    IntArray,
    /// FloatArray
    FloatArray,
    /// StringArray
    StringArray,
}

/// Parameter constraints for validation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParameterConstraints {
    /// min_value
    pub min_value: Option<f64>,
    /// max_value
    pub max_value: Option<f64>,
    /// allowed_values
    pub allowed_values: Option<Vec<String>>,
    /// min_length
    pub min_length: Option<usize>,
    /// max_length
    pub max_length: Option<usize>,
}

/// Plugin metadata
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PluginMetadata {
    /// name
    pub name: String,
    /// version
    pub version: String,
    /// description
    pub description: String,
    /// author
    pub author: String,
    /// supported_features
    pub supported_features: Vec<PluginFeature>,
    /// parameter_schema
    pub parameter_schema: Vec<ParameterDefinition>,
}

/// Custom model metadata
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CustomModelMetadata {
    /// plugin_name
    pub plugin_name: String,
    /// plugin_version
    pub plugin_version: String,
    /// is_fitted
    pub is_fitted: bool,
    /// n_samples
    pub n_samples: Option<usize>,
    /// n_features
    pub n_features: Option<usize>,
    /// n_components
    pub n_components: Option<usize>,
    /// training_time
    pub training_time: Option<f64>,
    /// parameters
    pub parameters: HashMap<String, ParameterValue>,
}

/// Wrapper for custom manifold learners to integrate with sklearn-style API
#[derive(Debug)]
pub struct CustomManifoldWrapper {
    learner: Box<dyn CustomManifoldLearner>,
    plugin_name: String,
}

impl CustomManifoldWrapper {
    /// Create a new wrapper for a custom manifold learner
    pub fn new(plugin_name: &str, params: Option<&PluginParameters>) -> SklResult<Self> {
        let registry = PLUGIN_REGISTRY.read().unwrap();
        let learner = registry.create_instance(plugin_name, params)?;

        Ok(Self {
            learner,
            plugin_name: plugin_name.to_string(),
        })
    }

    /// Get the underlying learner
    pub fn learner(&self) -> &dyn CustomManifoldLearner {
        self.learner.as_ref()
    }

    /// Get mutable access to the underlying learner
    pub fn learner_mut(&mut self) -> &mut dyn CustomManifoldLearner {
        self.learner.as_mut()
    }

    /// Get the plugin name
    pub fn plugin_name(&self) -> &str {
        &self.plugin_name
    }
}

impl Clone for CustomManifoldWrapper {
    fn clone(&self) -> Self {
        Self {
            learner: self.learner.clone_learner(),
            plugin_name: self.plugin_name.clone(),
        }
    }
}

/// Implementation of sklearn-style traits for custom manifold wrapper
impl Estimator for CustomManifoldWrapper {
    type Config = PluginParameters;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        // Return empty config for now - could be improved
        static EMPTY_CONFIG: once_cell::sync::Lazy<PluginParameters> =
            once_cell::sync::Lazy::new(PluginParameters::new);
        &EMPTY_CONFIG
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for CustomManifoldWrapper {
    type Fitted = CustomManifoldWrapper;

    fn fit(mut self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        self.learner.fit(x)?;
        Ok(self)
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for CustomManifoldWrapper {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.learner.transform(x)
    }
}

/// Utility functions for plugin management
pub mod utils {
    use super::*;

    /// Register a plugin globally
    pub fn register_plugin(plugin: Arc<dyn ManifoldPlugin>) -> SklResult<()> {
        PLUGIN_REGISTRY.write().unwrap().register_plugin(plugin)
    }

    /// Unregister a plugin globally
    pub fn unregister_plugin(name: &str) -> SklResult<()> {
        PLUGIN_REGISTRY.write().unwrap().unregister_plugin(name)
    }

    /// List all registered plugins
    pub fn list_plugins() -> Vec<String> {
        PLUGIN_REGISTRY.read().unwrap().list_plugins()
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(name: &str) -> Option<PluginMetadata> {
        /// PLUGIN_REGISTRY
        PLUGIN_REGISTRY
            .read()
            .unwrap()
            .get_plugin(name)
            .map(|p| p.metadata())
    }

    /// Get all plugin metadata
    pub fn get_all_plugin_metadata() -> Vec<PluginMetadata> {
        PLUGIN_REGISTRY.read().unwrap().get_all_metadata()
    }

    /// Create a new instance of a plugin
    pub fn create_plugin_instance(
        name: &str,
        params: Option<&PluginParameters>,
    ) -> SklResult<CustomManifoldWrapper> {
        CustomManifoldWrapper::new(name, params)
    }

    /// Validate parameters against plugin schema
    pub fn validate_parameters(plugin_name: &str, params: &PluginParameters) -> SklResult<()> {
        let registry = PLUGIN_REGISTRY.read().unwrap();
        let plugin = registry.get_plugin(plugin_name).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Plugin '{}' not found", plugin_name))
        })?;
        plugin.validate_parameters(params)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, ArrayView2};
    use scirs2_core::{random::Rng, thread_rng};

    /// Example plugin implementation for testing
    #[derive(Debug)]
    struct ExamplePlugin;

    impl ManifoldPlugin for ExamplePlugin {
        fn name(&self) -> &str {
            "ExamplePlugin"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn description(&self) -> &str {
            "An example plugin for testing"
        }
        fn author(&self) -> &str {
            "Test Author"
        }

        fn create_default(&self) -> Box<dyn CustomManifoldLearner> {
            Box::new(ExampleLearner::default())
        }

        fn create_with_params(
            &self,
            params: &PluginParameters,
        ) -> SklResult<Box<dyn CustomManifoldLearner>> {
            let mut learner = ExampleLearner::default();

            if let Some(ParameterValue::Int(n_components)) = params.get("n_components") {
                learner.set_parameter("n_components", ParameterValue::Int(*n_components))?;
            }

            Ok(Box::new(learner))
        }

        fn default_parameters(&self) -> PluginParameters {
            let mut params = PluginParameters::new();
            params.set("n_components", 2i64);
            params
        }

        fn validate_parameters(&self, params: &PluginParameters) -> SklResult<()> {
            if let Some(ParameterValue::Int(n_components)) = params.get("n_components") {
                if *n_components <= 0 {
                    return Err(SklearsError::InvalidInput(
                        "n_components must be positive".to_string(),
                    ));
                }
            }
            Ok(())
        }

        fn parameter_schema(&self) -> Vec<ParameterDefinition> {
            vec![ParameterDefinition {
                name: "n_components".to_string(),
                param_type: ParameterType::Int,
                description: "Number of components".to_string(),
                default_value: Some(ParameterValue::Int(2)),
                required: false,
                constraints: Some(ParameterConstraints {
                    min_value: Some(1.0),
                    max_value: None,
                    allowed_values: None,
                    min_length: None,
                    max_length: None,
                }),
            }]
        }
    }

    /// Example learner implementation for testing
    #[derive(Debug, Clone)]
    struct ExampleLearner {
        n_components: usize,
        fitted: bool,
        embedding: Option<Array2<Float>>,
    }

    impl Default for ExampleLearner {
        fn default() -> Self {
            Self {
                n_components: 2,
                fitted: false,
                embedding: None,
            }
        }
    }

    impl CustomManifoldLearner for ExampleLearner {
        fn set_parameter(&mut self, name: &str, value: ParameterValue) -> SklResult<()> {
            match name {
                "n_components" => {
                    if let ParameterValue::Int(val) = value {
                        self.n_components = val as usize;
                        Ok(())
                    } else {
                        Err(SklearsError::InvalidInput(
                            "n_components must be an integer".to_string(),
                        ))
                    }
                }
                _ => Err(SklearsError::InvalidInput(format!(
                    "Unknown parameter: {}",
                    name
                ))),
            }
        }

        fn get_parameter(&self, name: &str) -> Option<ParameterValue> {
            match name {
                "n_components" => Some(ParameterValue::Int(self.n_components as i64)),
                _ => None,
            }
        }

        fn get_all_parameters(&self) -> HashMap<String, ParameterValue> {
            let mut params = HashMap::new();
            params.insert(
                "n_components".to_string(),
                ParameterValue::Int(self.n_components as i64),
            );
            params
        }

        fn fit(&mut self, x: &ArrayView2<Float>) -> SklResult<()> {
            let (n_samples, _) = x.dim();

            // Simple example: just create random embedding
            let mut rng = thread_rng();
            let mut embedding = Array2::zeros((n_samples, self.n_components));
            for elem in embedding.iter_mut() {
                *elem = rng.gen::<f64>();
            }

            self.embedding = Some(embedding);
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            if !self.fitted {
                return Err(SklearsError::InvalidInput(
                    "Model is not fitted".to_string(),
                ));
            }

            // For this example, just return the stored embedding
            // In a real implementation, this would transform new data
            self.embedding
                .clone()
                .ok_or_else(|| SklearsError::InvalidInput("No embedding available".to_string()))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn get_metadata(&self) -> CustomModelMetadata {
            /// CustomModelMetadata
            CustomModelMetadata {
                plugin_name: "ExamplePlugin".to_string(),
                plugin_version: "1.0.0".to_string(),
                is_fitted: self.fitted,
                n_samples: self.embedding.as_ref().map(|e| e.nrows()),
                n_features: None,
                n_components: Some(self.n_components),
                training_time: None,
                parameters: self.get_all_parameters(),
            }
        }

        fn clone_learner(&self) -> Box<dyn CustomManifoldLearner> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_plugin_registration() {
        let plugin = Arc::new(ExamplePlugin);
        let result = utils::register_plugin(plugin);
        assert!(result.is_ok());

        let plugins = utils::list_plugins();
        assert!(plugins.contains(&"ExamplePlugin".to_string()));

        // Clean up
        utils::unregister_plugin("ExamplePlugin").unwrap();
    }

    #[test]
    fn test_plugin_instance_creation() {
        let plugin = Arc::new(ExamplePlugin);
        utils::register_plugin(plugin).unwrap();

        let wrapper = utils::create_plugin_instance("ExamplePlugin", None);
        assert!(wrapper.is_ok());

        // Clean up
        utils::unregister_plugin("ExamplePlugin").unwrap();
    }

    #[test]
    fn test_parameter_validation() {
        let plugin = Arc::new(ExamplePlugin);
        utils::register_plugin(plugin).unwrap();

        let mut params = PluginParameters::new();
        params.set("n_components", 5i64);

        let result = utils::validate_parameters("ExamplePlugin", &params);
        assert!(result.is_ok());

        // Test invalid parameters
        params.set("n_components", -1i64);
        let result = utils::validate_parameters("ExamplePlugin", &params);
        assert!(result.is_err());

        // Clean up
        utils::unregister_plugin("ExamplePlugin").unwrap();
    }
}
