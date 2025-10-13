//! Plugin Architecture for Custom Explanation Methods
//!
//! This module provides a comprehensive plugin system for registering and managing
//! custom explanation methods, allowing users to extend the library with their own
//! interpretability algorithms.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Trait for plugin explanation methods
pub trait ExplanationPlugin: Debug + Send + Sync {
    /// Plugin identifier
    fn plugin_id(&self) -> &str;

    /// Plugin name
    fn plugin_name(&self) -> &str;

    /// Plugin version
    fn plugin_version(&self) -> &str;

    /// Plugin description
    fn plugin_description(&self) -> &str;

    /// Plugin author
    fn plugin_author(&self) -> &str;

    /// Supported input types
    fn supported_input_types(&self) -> Vec<InputType>;

    /// Supported output types
    fn supported_output_types(&self) -> Vec<OutputType>;

    /// Plugin capabilities
    fn capabilities(&self) -> PluginCapabilities;

    /// Initialize the plugin
    fn initialize(&mut self, config: &PluginConfig) -> SklResult<()>;

    /// Execute the explanation method
    fn execute(&self, input: &PluginInput) -> SklResult<PluginOutput>;

    /// Validate input before execution
    fn validate_input(&self, input: &PluginInput) -> SklResult<()>;

    /// Cleanup resources
    fn cleanup(&mut self) -> SklResult<()>;

    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            id: self.plugin_id().to_string(),
            name: self.plugin_name().to_string(),
            version: self.plugin_version().to_string(),
            description: self.plugin_description().to_string(),
            author: self.plugin_author().to_string(),
            supported_inputs: self.supported_input_types(),
            supported_outputs: self.supported_output_types(),
            capabilities: self.capabilities(),
            created_at: chrono::Utc::now(),
        }
    }
}

/// Plugin input types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputType {
    /// Tabular data (features x samples)
    Tabular,
    /// Time series data
    TimeSeries,
    /// Image data
    Image,
    /// Text data
    Text,
    /// Graph data
    Graph,
    /// Custom data type
    Custom(u32),
}

/// Plugin output types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    /// Feature importance scores
    FeatureImportance,
    /// Local explanations
    LocalExplanation,
    /// Global explanations
    GlobalExplanation,
    /// Counterfactual explanations
    CounterfactualExplanation,
    /// Visualization data
    VisualizationData,
    /// Custom output type
    Custom(u32),
}

/// Plugin capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    /// Supports local explanations
    pub local_explanations: bool,
    /// Supports global explanations
    pub global_explanations: bool,
    /// Supports counterfactual explanations
    pub counterfactual_explanations: bool,
    /// Supports uncertainty quantification
    pub uncertainty_quantification: bool,
    /// Supports model-agnostic explanations
    pub model_agnostic: bool,
    /// Supports parallel processing
    pub parallel_processing: bool,
    /// Supports real-time explanations
    pub real_time: bool,
    /// Supports streaming data
    pub streaming: bool,
    /// Maximum dataset size (number of samples)
    pub max_dataset_size: Option<usize>,
    /// Maximum number of features
    pub max_features: Option<usize>,
    /// Estimated memory usage in bytes
    pub estimated_memory_usage: Option<usize>,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            local_explanations: false,
            global_explanations: false,
            counterfactual_explanations: false,
            uncertainty_quantification: false,
            model_agnostic: true,
            parallel_processing: false,
            real_time: false,
            streaming: false,
            max_dataset_size: None,
            max_features: None,
            estimated_memory_usage: None,
        }
    }
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin-specific parameters
    pub parameters: HashMap<String, PluginParameter>,
    /// Maximum execution time in seconds
    pub max_execution_time: Option<u64>,
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Logging level
    pub log_level: LogLevel,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            max_execution_time: Some(300),          // 5 minutes
            memory_limit: Some(1024 * 1024 * 1024), // 1GB
            num_threads: Some(1),
            random_seed: None,
            log_level: LogLevel::Info,
        }
    }
}

/// Plugin parameter values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginParameter {
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
    /// Boolean
    Boolean(bool),
    /// IntegerArray
    IntegerArray(Vec<i64>),
    /// FloatArray
    FloatArray(Vec<f64>),
    /// StringArray
    StringArray(Vec<String>),
    /// BooleanArray
    BooleanArray(Vec<bool>),
}

/// Logging levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogLevel {
    /// Error
    Error,
    /// Warn
    Warn,
    /// Info
    Info,
    /// Debug
    Debug,
    /// Trace
    Trace,
}

/// Plugin input data
#[derive(Debug, Clone)]
pub struct PluginInput {
    /// Input data
    pub data: PluginData,
    /// Model predictions (if available)
    pub predictions: Option<Array1<Float>>,
    /// Target values (if available)
    pub targets: Option<Array1<Float>>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Sample weights
    pub sample_weights: Option<Array1<Float>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Plugin data types
#[derive(Debug, Clone)]
pub enum PluginData {
    /// Tabular data (samples x features)
    Tabular(Array2<Float>),
    /// Time series data
    TimeSeries(Array2<Float>),
    /// Image data (height x width x channels)
    Image(Array2<Float>),
    /// Text data (as string)
    Text(String),
    /// Graph data (adjacency matrix)
    Graph(Array2<Float>),
    /// Custom data (serialized)
    Custom(Vec<u8>),
}

/// Plugin output data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginOutput {
    /// Output type
    pub output_type: OutputType,
    /// Output data
    pub data: PluginOutputData,
    /// Execution metadata
    pub metadata: ExecutionMetadata,
    /// Confidence scores (if available)
    pub confidence: Option<Array1<Float>>,
    /// Uncertainty estimates (if available)
    pub uncertainty: Option<Array1<Float>>,
}

/// Plugin output data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginOutputData {
    /// Feature importance scores
    FeatureImportance {
        scores: Vec<Float>,

        feature_names: Vec<String>,

        std_errors: Option<Vec<Float>>,
    },
    /// Local explanation
    LocalExplanation {
        instance_id: usize,

        feature_contributions: Vec<Float>,
        feature_names: Vec<String>,
        base_value: Float,
    },
    /// Global explanation
    GlobalExplanation {
        feature_effects: Vec<Float>,
        feature_names: Vec<String>,
        interaction_effects: Option<Array2<Float>>,
    },
    /// Counterfactual explanation
    CounterfactualExplanation {
        counterfactual_instance: Array1<Float>,
        feature_changes: Vec<(usize, Float, Float)>, // (feature_idx, original, new)
        distance: Float,
        feasibility_score: Float,
    },
    /// Visualization data
    VisualizationData {
        plot_type: String,
        data: serde_json::Value,
        config: HashMap<String, String>,
    },
    /// Custom output
    Custom(serde_json::Value),
}

/// Execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of iterations (if applicable)
    pub iterations: Option<usize>,
    /// Convergence status
    pub converged: Option<bool>,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin ID
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Supported input types
    pub supported_inputs: Vec<InputType>,
    /// Supported output types
    pub supported_outputs: Vec<OutputType>,
    /// Plugin capabilities
    pub capabilities: PluginCapabilities,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Plugin registry for managing plugins
#[derive(Debug, Default)]
pub struct PluginRegistry {
    plugins: Arc<RwLock<HashMap<String, Arc<dyn ExplanationPlugin>>>>,
    plugin_configs: Arc<RwLock<HashMap<String, PluginConfig>>>,
    plugin_metadata: Arc<RwLock<HashMap<String, PluginMetadata>>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new plugin
    pub fn register_plugin<P: ExplanationPlugin + 'static>(
        &self,
        mut plugin: P,
        config: Option<PluginConfig>,
    ) -> SklResult<()> {
        let plugin_id = plugin.plugin_id().to_string();
        let config = config.unwrap_or_default();

        // Initialize the plugin
        plugin.initialize(&config)?;

        // Get metadata
        let metadata = plugin.metadata();

        // Store plugin, config, and metadata
        {
            let mut plugins = self.plugins.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire plugins lock".to_string())
            })?;
            plugins.insert(plugin_id.clone(), Arc::new(plugin));
        }

        {
            let mut configs = self.plugin_configs.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire configs lock".to_string())
            })?;
            configs.insert(plugin_id.clone(), config);
        }

        {
            let mut metadata_store = self.plugin_metadata.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire metadata lock".to_string())
            })?;
            metadata_store.insert(plugin_id, metadata);
        }

        Ok(())
    }

    /// Get a plugin by ID
    pub fn get_plugin(&self, plugin_id: &str) -> Option<Arc<dyn ExplanationPlugin>> {
        self.plugins.read().ok()?.get(plugin_id).cloned()
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, plugin_id: &str) -> Option<PluginConfig> {
        self.plugin_configs.read().ok()?.get(plugin_id).cloned()
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, plugin_id: &str) -> Option<PluginMetadata> {
        self.plugin_metadata.read().ok()?.get(plugin_id).cloned()
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins
            .read()
            .ok()
            .map(|plugins| plugins.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// List plugins by input type
    pub fn list_plugins_by_input_type(&self, input_type: InputType) -> Vec<String> {
        self.plugin_metadata
            .read()
            .ok()
            .map(|metadata| {
                metadata
                    .iter()
                    .filter(|(_, meta)| meta.supported_inputs.contains(&input_type))
                    .map(|(id, _)| id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List plugins by output type
    pub fn list_plugins_by_output_type(&self, output_type: OutputType) -> Vec<String> {
        self.plugin_metadata
            .read()
            .ok()
            .map(|metadata| {
                metadata
                    .iter()
                    .filter(|(_, meta)| meta.supported_outputs.contains(&output_type))
                    .map(|(id, _)| id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List plugins by capability
    pub fn list_plugins_by_capability(&self, capability: PluginCapabilityFilter) -> Vec<String> {
        self.plugin_metadata
            .read()
            .ok()
            .map(|metadata| {
                metadata
                    .iter()
                    .filter(|(_, meta)| capability.matches(&meta.capabilities))
                    .map(|(id, _)| id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Execute a plugin
    pub fn execute_plugin(&self, plugin_id: &str, input: &PluginInput) -> SklResult<PluginOutput> {
        let plugin = self.get_plugin(plugin_id).ok_or_else(|| {
            crate::SklearsError::InvalidInput(format!("Plugin '{}' not found", plugin_id))
        })?;

        // Validate input
        plugin.validate_input(input)?;

        // Execute plugin
        let start_time = std::time::Instant::now();
        let result = plugin.execute(input);
        let execution_time = start_time.elapsed().as_millis() as u64;

        // Add timing information to result
        match result {
            Ok(mut output) => {
                output.metadata.execution_time_ms = execution_time;
                Ok(output)
            }
            Err(e) => Err(e),
        }
    }

    /// Unregister a plugin
    pub fn unregister_plugin(&self, plugin_id: &str) -> SklResult<()> {
        {
            let mut plugins = self.plugins.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire plugins lock".to_string())
            })?;
            plugins.remove(plugin_id);
        }

        {
            let mut configs = self.plugin_configs.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire configs lock".to_string())
            })?;
            configs.remove(plugin_id);
        }

        {
            let mut metadata_store = self.plugin_metadata.write().map_err(|_| {
                crate::SklearsError::InvalidInput("Failed to acquire metadata lock".to_string())
            })?;
            metadata_store.remove(plugin_id);
        }

        Ok(())
    }

    /// Get plugin statistics
    pub fn get_statistics(&self) -> PluginRegistryStatistics {
        let plugins = self.plugins.read().ok();
        let metadata = self.plugin_metadata.read().ok();

        let total_plugins = plugins.as_ref().map(|p| p.len()).unwrap_or(0);

        let plugins_by_type = metadata
            .as_ref()
            .map(|meta| {
                let mut input_types = HashMap::new();
                let mut output_types = HashMap::new();

                for (_, plugin_meta) in meta.iter() {
                    for input_type in &plugin_meta.supported_inputs {
                        *input_types.entry(*input_type).or_insert(0) += 1;
                    }
                    for output_type in &plugin_meta.supported_outputs {
                        *output_types.entry(*output_type).or_insert(0) += 1;
                    }
                }

                (input_types, output_types)
            })
            .unwrap_or_default();

        PluginRegistryStatistics {
            total_plugins,
            plugins_by_input_type: plugins_by_type.0,
            plugins_by_output_type: plugins_by_type.1,
            registry_created_at: chrono::Utc::now(),
        }
    }
}

/// Plugin capability filter
#[derive(Debug, Clone)]
pub struct PluginCapabilityFilter {
    /// Requires local explanations
    pub local_explanations: Option<bool>,
    /// Requires global explanations
    pub global_explanations: Option<bool>,
    /// Requires counterfactual explanations
    pub counterfactual_explanations: Option<bool>,
    /// Requires uncertainty quantification
    pub uncertainty_quantification: Option<bool>,
    /// Requires model-agnostic support
    pub model_agnostic: Option<bool>,
    /// Requires parallel processing
    pub parallel_processing: Option<bool>,
    /// Requires real-time support
    pub real_time: Option<bool>,
    /// Requires streaming support
    pub streaming: Option<bool>,
    /// Maximum dataset size constraint
    pub max_dataset_size: Option<usize>,
    /// Maximum features constraint
    pub max_features: Option<usize>,
}

impl PluginCapabilityFilter {
    /// Create a new capability filter
    pub fn new() -> Self {
        Self {
            local_explanations: None,
            global_explanations: None,
            counterfactual_explanations: None,
            uncertainty_quantification: None,
            model_agnostic: None,
            parallel_processing: None,
            real_time: None,
            streaming: None,
            max_dataset_size: None,
            max_features: None,
        }
    }

    /// Check if capabilities match the filter
    pub fn matches(&self, capabilities: &PluginCapabilities) -> bool {
        if let Some(required) = self.local_explanations {
            if capabilities.local_explanations != required {
                return false;
            }
        }

        if let Some(required) = self.global_explanations {
            if capabilities.global_explanations != required {
                return false;
            }
        }

        if let Some(required) = self.counterfactual_explanations {
            if capabilities.counterfactual_explanations != required {
                return false;
            }
        }

        if let Some(required) = self.uncertainty_quantification {
            if capabilities.uncertainty_quantification != required {
                return false;
            }
        }

        if let Some(required) = self.model_agnostic {
            if capabilities.model_agnostic != required {
                return false;
            }
        }

        if let Some(required) = self.parallel_processing {
            if capabilities.parallel_processing != required {
                return false;
            }
        }

        if let Some(required) = self.real_time {
            if capabilities.real_time != required {
                return false;
            }
        }

        if let Some(required) = self.streaming {
            if capabilities.streaming != required {
                return false;
            }
        }

        if let Some(max_size) = self.max_dataset_size {
            if let Some(cap_size) = capabilities.max_dataset_size {
                if cap_size < max_size {
                    return false;
                }
            } else {
                return false;
            }
        }

        if let Some(max_features) = self.max_features {
            if let Some(cap_features) = capabilities.max_features {
                if cap_features < max_features {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

impl Default for PluginCapabilityFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistryStatistics {
    /// Total number of plugins
    pub total_plugins: usize,
    /// Number of plugins by input type
    pub plugins_by_input_type: HashMap<InputType, usize>,
    /// Number of plugins by output type
    pub plugins_by_output_type: HashMap<OutputType, usize>,
    /// Registry creation timestamp
    pub registry_created_at: chrono::DateTime<chrono::Utc>,
}

/// Plugin manager for orchestrating multiple plugins
#[derive(Debug)]
pub struct PluginManager {
    registry: PluginRegistry,
    execution_history: Arc<RwLock<Vec<PluginExecution>>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            registry: PluginRegistry::new(),
            execution_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get the plugin registry
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    /// Execute a plugin with history tracking
    pub fn execute_with_history(
        &self,
        plugin_id: &str,
        input: &PluginInput,
    ) -> SklResult<PluginOutput> {
        let start_time = std::time::Instant::now();
        let result = self.registry.execute_plugin(plugin_id, input);
        let execution_time = start_time.elapsed();

        // Record execution
        let execution = PluginExecution {
            plugin_id: plugin_id.to_string(),
            success: result.is_ok(),
            execution_time_ms: execution_time.as_millis() as u64,
            timestamp: chrono::Utc::now(),
            error_message: result.as_ref().err().map(|e| e.to_string()),
        };

        if let Ok(mut history) = self.execution_history.write() {
            history.push(execution);
        }

        result
    }

    /// Get execution history
    pub fn get_execution_history(&self) -> Vec<PluginExecution> {
        self.execution_history
            .read()
            .ok()
            .map(|history| history.clone())
            .unwrap_or_default()
    }

    /// Get execution statistics
    pub fn get_execution_statistics(&self) -> ExecutionStatistics {
        let history = self.get_execution_history();

        let total_executions = history.len();
        let successful_executions = history.iter().filter(|e| e.success).count();
        let failed_executions = total_executions - successful_executions;

        let average_execution_time = if total_executions > 0 {
            history.iter().map(|e| e.execution_time_ms).sum::<u64>() / total_executions as u64
        } else {
            0
        };

        let plugin_usage = {
            let mut usage = HashMap::new();
            for execution in &history {
                *usage.entry(execution.plugin_id.clone()).or_insert(0) += 1;
            }
            usage
        };

        ExecutionStatistics {
            total_executions,
            successful_executions,
            failed_executions,
            average_execution_time_ms: average_execution_time,
            plugin_usage,
        }
    }

    /// Clear execution history
    pub fn clear_execution_history(&self) {
        if let Ok(mut history) = self.execution_history.write() {
            history.clear();
        }
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginExecution {
    /// Plugin ID
    pub plugin_id: String,
    /// Whether execution was successful
    pub success: bool,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    /// Total number of executions
    pub total_executions: usize,
    /// Number of successful executions
    pub successful_executions: usize,
    /// Number of failed executions
    pub failed_executions: usize,
    /// Average execution time in milliseconds
    pub average_execution_time_ms: u64,
    /// Plugin usage counts
    pub plugin_usage: HashMap<String, usize>,
}

/// Example plugin implementation
#[derive(Debug)]
pub struct ExampleCustomPlugin {
    id: String,
    name: String,
    version: String,
    description: String,
    author: String,
    initialized: bool,
}

impl ExampleCustomPlugin {
    /// Create a new example plugin
    pub fn new() -> Self {
        Self {
            id: "example_custom_plugin".to_string(),
            name: "Example Custom Plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "An example plugin for demonstration purposes".to_string(),
            author: "Sklears Team".to_string(),
            initialized: false,
        }
    }
}

impl ExplanationPlugin for ExampleCustomPlugin {
    fn plugin_id(&self) -> &str {
        &self.id
    }

    fn plugin_name(&self) -> &str {
        &self.name
    }

    fn plugin_version(&self) -> &str {
        &self.version
    }

    fn plugin_description(&self) -> &str {
        &self.description
    }

    fn plugin_author(&self) -> &str {
        &self.author
    }

    fn supported_input_types(&self) -> Vec<InputType> {
        vec![InputType::Tabular, InputType::TimeSeries]
    }

    fn supported_output_types(&self) -> Vec<OutputType> {
        vec![OutputType::FeatureImportance, OutputType::LocalExplanation]
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            local_explanations: true,
            global_explanations: true,
            counterfactual_explanations: false,
            uncertainty_quantification: false,
            model_agnostic: true,
            parallel_processing: false,
            real_time: true,
            streaming: false,
            max_dataset_size: Some(10000),
            max_features: Some(1000),
            estimated_memory_usage: Some(1024 * 1024), // 1MB
        }
    }

    fn initialize(&mut self, _config: &PluginConfig) -> SklResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn execute(&self, input: &PluginInput) -> SklResult<PluginOutput> {
        if !self.initialized {
            return Err(crate::SklearsError::InvalidInput(
                "Plugin not initialized".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        // Example implementation: compute simple feature importance
        let feature_importance = match &input.data {
            PluginData::Tabular(data) => {
                let n_features = data.ncols();
                let importance_scores: Vec<Float> = (0..n_features)
                    .map(|i| {
                        let column = data.column(i);
                        column.var(0.0) // Use variance as importance
                    })
                    .collect();

                let feature_names = input
                    .feature_names
                    .clone()
                    .unwrap_or_else(|| (0..n_features).map(|i| format!("feature_{}", i)).collect());

                PluginOutputData::FeatureImportance {
                    scores: importance_scores,
                    feature_names,
                    std_errors: None,
                }
            }
            _ => {
                return Err(crate::SklearsError::InvalidInput(
                    "Unsupported input type for this plugin".to_string(),
                ));
            }
        };

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(PluginOutput {
            output_type: OutputType::FeatureImportance,
            data: feature_importance,
            metadata: ExecutionMetadata {
                execution_time_ms: execution_time,
                memory_usage_bytes: 0,
                iterations: None,
                converged: Some(true),
                warnings: Vec::new(),
                timestamp: chrono::Utc::now(),
            },
            confidence: None,
            uncertainty: None,
        })
    }

    fn validate_input(&self, input: &PluginInput) -> SklResult<()> {
        match &input.data {
            PluginData::Tabular(data) => {
                if data.nrows() == 0 || data.ncols() == 0 {
                    return Err(crate::SklearsError::InvalidInput(
                        "Input data cannot be empty".to_string(),
                    ));
                }

                if let Some(max_features) = self.capabilities().max_features {
                    if data.ncols() > max_features {
                        return Err(crate::SklearsError::InvalidInput(format!(
                            "Too many features: {} > {}",
                            data.ncols(),
                            max_features
                        )));
                    }
                }

                if let Some(max_samples) = self.capabilities().max_dataset_size {
                    if data.nrows() > max_samples {
                        return Err(crate::SklearsError::InvalidInput(format!(
                            "Too many samples: {} > {}",
                            data.nrows(),
                            max_samples
                        )));
                    }
                }

                Ok(())
            }
            _ => Err(crate::SklearsError::InvalidInput(
                "Unsupported input type".to_string(),
            )),
        }
    }

    fn cleanup(&mut self) -> SklResult<()> {
        self.initialized = false;
        Ok(())
    }
}

impl Default for ExampleCustomPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_plugin_registry() {
        let registry = PluginRegistry::new();

        // Register a plugin
        let plugin = ExampleCustomPlugin::new();
        let result = registry.register_plugin(plugin, None);
        assert!(result.is_ok());

        // Check plugin is registered
        let plugins = registry.list_plugins();
        assert!(plugins.contains(&"example_custom_plugin".to_string()));

        // Get plugin metadata
        let metadata = registry.get_plugin_metadata("example_custom_plugin");
        assert!(metadata.is_some());
        let metadata = metadata.unwrap();
        assert_eq!(metadata.name, "Example Custom Plugin");
        assert_eq!(metadata.version, "1.0.0");
    }

    #[test]
    fn test_plugin_execution() {
        let registry = PluginRegistry::new();

        // Register plugin
        let plugin = ExampleCustomPlugin::new();
        registry.register_plugin(plugin, None).unwrap();

        // Create input data
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as Float).collect()).unwrap();
        let input = PluginInput {
            data: PluginData::Tabular(data),
            predictions: None,
            targets: None,
            feature_names: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
            sample_weights: None,
            metadata: HashMap::new(),
        };

        // Execute plugin
        let result = registry.execute_plugin("example_custom_plugin", &input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.output_type, OutputType::FeatureImportance);

        match output.data {
            PluginOutputData::FeatureImportance {
                scores,
                feature_names,
                ..
            } => {
                assert_eq!(scores.len(), 3);
                assert_eq!(feature_names.len(), 3);
                assert_eq!(feature_names[0], "f1");
            }
            _ => panic!("Expected feature importance output"),
        }
    }

    #[test]
    fn test_plugin_capability_filter() {
        let capabilities = PluginCapabilities {
            local_explanations: true,
            global_explanations: true,
            counterfactual_explanations: false,
            uncertainty_quantification: false,
            model_agnostic: true,
            parallel_processing: false,
            real_time: true,
            streaming: false,
            max_dataset_size: Some(10000),
            max_features: Some(1000),
            estimated_memory_usage: Some(1024 * 1024),
        };

        let mut filter = PluginCapabilityFilter::new();
        filter.local_explanations = Some(true);
        filter.real_time = Some(true);
        filter.max_dataset_size = Some(5000);

        assert!(filter.matches(&capabilities));

        filter.max_dataset_size = Some(20000);
        assert!(!filter.matches(&capabilities));
    }

    #[test]
    fn test_plugin_manager() {
        let manager = PluginManager::new();

        // Register plugin
        let plugin = ExampleCustomPlugin::new();
        manager.registry().register_plugin(plugin, None).unwrap();

        // Create input data
        let data = Array2::from_shape_vec((5, 2), (0..10).map(|x| x as Float).collect()).unwrap();
        let input = PluginInput {
            data: PluginData::Tabular(data),
            predictions: None,
            targets: None,
            feature_names: None,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        // Execute with history
        let result = manager.execute_with_history("example_custom_plugin", &input);
        assert!(result.is_ok());

        // Check history
        let history = manager.get_execution_history();
        assert_eq!(history.len(), 1);
        assert!(history[0].success);

        // Get statistics
        let stats = manager.get_execution_statistics();
        assert_eq!(stats.total_executions, 1);
        assert_eq!(stats.successful_executions, 1);
        assert_eq!(stats.failed_executions, 0);
    }

    #[test]
    fn test_plugin_list_by_type() {
        let registry = PluginRegistry::new();

        // Register plugin
        let plugin = ExampleCustomPlugin::new();
        registry.register_plugin(plugin, None).unwrap();

        // List by input type
        let tabular_plugins = registry.list_plugins_by_input_type(InputType::Tabular);
        assert!(tabular_plugins.contains(&"example_custom_plugin".to_string()));

        let image_plugins = registry.list_plugins_by_input_type(InputType::Image);
        assert!(image_plugins.is_empty());

        // List by output type
        let importance_plugins =
            registry.list_plugins_by_output_type(OutputType::FeatureImportance);
        assert!(importance_plugins.contains(&"example_custom_plugin".to_string()));

        let counterfactual_plugins =
            registry.list_plugins_by_output_type(OutputType::CounterfactualExplanation);
        assert!(counterfactual_plugins.is_empty());
    }

    #[test]
    fn test_plugin_validation() {
        let plugin = ExampleCustomPlugin::new();

        // Test empty data validation
        let empty_data = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        let input = PluginInput {
            data: PluginData::Tabular(empty_data),
            predictions: None,
            targets: None,
            feature_names: None,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        let result = plugin.validate_input(&input);
        assert!(result.is_err());

        // Test valid data
        let valid_data =
            Array2::from_shape_vec((5, 2), (0..10).map(|x| x as Float).collect()).unwrap();
        let input = PluginInput {
            data: PluginData::Tabular(valid_data),
            predictions: None,
            targets: None,
            feature_names: None,
            sample_weights: None,
            metadata: HashMap::new(),
        };

        let result = plugin.validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_plugin_parameter_types() {
        let mut config = PluginConfig::default();

        config
            .parameters
            .insert("integer_param".to_string(), PluginParameter::Integer(42));
        config
            .parameters
            .insert("float_param".to_string(), PluginParameter::Float(3.14));
        config.parameters.insert(
            "string_param".to_string(),
            PluginParameter::String("test".to_string()),
        );
        config
            .parameters
            .insert("bool_param".to_string(), PluginParameter::Boolean(true));

        assert_eq!(config.parameters.len(), 4);

        match config.parameters.get("integer_param") {
            Some(PluginParameter::Integer(val)) => assert_eq!(*val, 42),
            _ => panic!("Expected integer parameter"),
        }

        match config.parameters.get("float_param") {
            Some(PluginParameter::Float(val)) => assert_eq!(*val, 3.14),
            _ => panic!("Expected float parameter"),
        }
    }
}
