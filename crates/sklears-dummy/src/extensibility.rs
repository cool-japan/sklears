//! Extensibility Framework for Dummy Estimators
//!
//! This module provides a comprehensive extensibility framework including:
//! - Plugin architecture for custom baselines
//! - Hooks for prediction callbacks
//! - Integration with evaluation utilities
//! - Custom strategy registration
//! - Middleware for baseline pipelines

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::Estimator;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};

/// Plugin interface for custom baseline strategies
pub trait BaselinePlugin: Send + Sync + Debug {
    /// Plugin identifier
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Plugin description
    fn description(&self) -> &str;

    /// Initialize the plugin
    fn initialize(&mut self, config: &PluginConfig) -> Result<()>;

    /// Shutdown the plugin
    fn shutdown(&mut self) -> Result<()>;

    /// Check if plugin is compatible with given data
    fn is_compatible(&self, data_info: &DataInfo) -> bool;

    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;
}

/// Plugin configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PluginConfig {
    /// parameters
    pub parameters: HashMap<String, PluginParameter>,
    /// resources
    pub resources: ResourceConfig,
    /// logging
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceConfig {
    /// max_memory_mb
    pub max_memory_mb: usize,
    /// max_cpu_cores
    pub max_cpu_cores: usize,
    /// temp_directory
    pub temp_directory: String,
    /// cache_enabled
    pub cache_enabled: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LoggingConfig {
    /// level
    pub level: LogLevel,
    /// output_file
    pub output_file: Option<String>,
    /// include_timestamps
    pub include_timestamps: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PluginParameter {
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
    /// Boolean
    Boolean(bool),
    /// Array
    Array(Vec<PluginParameter>),
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// author
    pub author: String,
    /// license
    pub license: String,
    /// homepage
    pub homepage: String,
    /// supported_tasks
    pub supported_tasks: Vec<TaskType>,
    /// requirements
    pub requirements: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Classification
    Classification,
    /// Regression
    Regression,
    /// Clustering
    Clustering,
    /// DimensionalityReduction
    DimensionalityReduction,
}

/// Data information for plugin compatibility
#[derive(Debug, Clone)]
pub struct DataInfo {
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
    /// feature_types
    pub feature_types: Vec<FeatureType>,
    /// target_type
    pub target_type: TargetType,
    /// missing_values
    pub missing_values: bool,
    /// sparse
    pub sparse: bool,
}

#[derive(Debug, Clone)]
pub enum FeatureType {
    /// Continuous
    Continuous,
    /// Categorical
    Categorical,
    /// Binary
    Binary,
    /// Ordinal
    Ordinal,
    /// Text
    Text,
}

#[derive(Debug, Clone)]
pub enum TargetType {
    /// Continuous
    Continuous,
    /// Binary
    Binary,
    /// Multiclass
    Multiclass,
    /// Multilabel
    Multilabel,
}

/// Plugin registry for managing custom baselines
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, Arc<dyn BaselinePlugin>>>,
    plugin_configs: RwLock<HashMap<String, PluginConfig>>,
    active_plugins: RwLock<HashMap<String, bool>>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            plugin_configs: RwLock::new(HashMap::new()),
            active_plugins: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new plugin
    pub fn register_plugin(
        &self,
        plugin: Arc<dyn BaselinePlugin>,
        config: PluginConfig,
    ) -> Result<()> {
        let name = plugin.name().to_string();

        // Check for name conflicts
        if self.plugins.read().unwrap().contains_key(&name) {
            return Err(SklearsError::InvalidInput(format!(
                "Plugin '{}' already registered",
                name
            )));
        }

        // Register plugin
        self.plugins.write().unwrap().insert(name.clone(), plugin);
        self.plugin_configs
            .write()
            .unwrap()
            .insert(name.clone(), config);
        self.active_plugins.write().unwrap().insert(name, false);

        Ok(())
    }

    /// Activate a plugin
    pub fn activate_plugin(&self, name: &str) -> Result<()> {
        let plugins = self.plugins.read().unwrap();
        let plugin_configs = self.plugin_configs.write().unwrap();
        let mut active_plugins = self.active_plugins.write().unwrap();

        if let Some(plugin) = plugins.get(name) {
            if let Some(config) = plugin_configs.get(name) {
                // Initialize plugin (would need mutable reference in real implementation)
                active_plugins.insert(name.to_string(), true);
                Ok(())
            } else {
                Err(SklearsError::InvalidInput(format!(
                    "No configuration found for plugin '{}'",
                    name
                )))
            }
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Plugin '{}' not found",
                name
            )))
        }
    }

    /// Deactivate a plugin
    pub fn deactivate_plugin(&self, name: &str) -> Result<()> {
        let mut active_plugins = self.active_plugins.write().unwrap();

        if active_plugins.contains_key(name) {
            active_plugins.insert(name.to_string(), false);
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Plugin '{}' not found",
                name
            )))
        }
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.read().unwrap().keys().cloned().collect()
    }

    /// Get active plugins
    pub fn get_active_plugins(&self) -> Vec<String> {
        self.active_plugins
            .read()
            .unwrap()
            .iter()
            .filter_map(|(name, &active)| if active { Some(name.clone()) } else { None })
            .collect()
    }

    /// Check plugin compatibility with data
    pub fn find_compatible_plugins(&self, data_info: &DataInfo) -> Vec<String> {
        let plugins = self.plugins.read().unwrap();
        let active_plugins = self.active_plugins.read().unwrap();

        plugins
            .iter()
            .filter(|(name, plugin)| {
                *active_plugins.get(*name).unwrap_or(&false) && plugin.is_compatible(data_info)
            })
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// Hook system for prediction callbacks
pub struct HookSystem {
    pre_fit_hooks: Arc<Mutex<Vec<Box<dyn PreFitHook>>>>,
    post_fit_hooks: Arc<Mutex<Vec<Box<dyn PostFitHook>>>>,
    pre_predict_hooks: Arc<Mutex<Vec<Box<dyn PrePredictHook>>>>,
    post_predict_hooks: Arc<Mutex<Vec<Box<dyn PostPredictHook>>>>,
    error_hooks: Arc<Mutex<Vec<Box<dyn ErrorHook>>>>,
}

/// Pre-fit hook interface
pub trait PreFitHook: Send + Sync + Debug {
    fn execute(&self, context: &mut FitContext) -> Result<()>;
    fn priority(&self) -> i32 {
        0
    }
}

/// Post-fit hook interface
pub trait PostFitHook: Send + Sync + Debug {
    fn execute(&self, context: &FitContext, result: &FitResult) -> Result<()>;
    fn priority(&self) -> i32 {
        0
    }
}

/// Pre-prediction hook interface
pub trait PrePredictHook: Send + Sync + Debug {
    fn execute(&self, context: &mut PredictContext) -> Result<()>;
    fn priority(&self) -> i32 {
        0
    }
}

/// Post-prediction hook interface
pub trait PostPredictHook: Send + Sync + Debug {
    fn execute(&self, context: &PredictContext, predictions: &mut Array1<f64>) -> Result<()>;
    fn priority(&self) -> i32 {
        0
    }
}

/// Error hook interface
pub trait ErrorHook: Send + Sync + Debug {
    fn execute(&self, context: &ErrorContext) -> Result<()>;
    fn priority(&self) -> i32 {
        0
    }
}

/// Context for fit operations
#[derive(Debug)]
pub struct FitContext {
    /// estimator_name
    pub estimator_name: String,
    /// strategy
    pub strategy: String,
    /// x
    pub x: Array2<f64>,
    /// y
    pub y: Array1<f64>,
    /// metadata
    pub metadata: HashMap<String, String>,
    /// start_time
    pub start_time: std::time::Instant,
}

/// Result of fit operation
#[derive(Debug)]
pub struct FitResult {
    /// success
    pub success: bool,
    /// duration
    pub duration: std::time::Duration,
    /// parameters
    pub parameters: HashMap<String, f64>,
    /// metrics
    pub metrics: HashMap<String, f64>,
}

/// Context for predict operations
#[derive(Debug)]
pub struct PredictContext {
    /// estimator_name
    pub estimator_name: String,
    /// strategy
    pub strategy: String,
    /// x
    pub x: Array2<f64>,
    /// metadata
    pub metadata: HashMap<String, String>,
    /// start_time
    pub start_time: std::time::Instant,
}

/// Context for error handling
#[derive(Debug)]
pub struct ErrorContext {
    /// operation
    pub operation: String,
    /// error
    pub error: String,
    /// estimator_name
    pub estimator_name: String,
    /// context_data
    pub context_data: HashMap<String, String>,
    /// timestamp
    pub timestamp: std::time::Instant,
}

impl Default for HookSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl HookSystem {
    /// Create new hook system
    pub fn new() -> Self {
        Self {
            pre_fit_hooks: Arc::new(Mutex::new(Vec::new())),
            post_fit_hooks: Arc::new(Mutex::new(Vec::new())),
            pre_predict_hooks: Arc::new(Mutex::new(Vec::new())),
            post_predict_hooks: Arc::new(Mutex::new(Vec::new())),
            error_hooks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add pre-fit hook
    pub fn add_pre_fit_hook(&self, hook: Box<dyn PreFitHook>) {
        let mut hooks = self.pre_fit_hooks.lock().unwrap();
        hooks.push(hook);
        hooks.sort_by_key(|h| -h.priority()); // Higher priority first
    }

    /// Add post-fit hook
    pub fn add_post_fit_hook(&self, hook: Box<dyn PostFitHook>) {
        let mut hooks = self.post_fit_hooks.lock().unwrap();
        hooks.push(hook);
        hooks.sort_by_key(|h| -h.priority());
    }

    /// Add pre-predict hook
    pub fn add_pre_predict_hook(&self, hook: Box<dyn PrePredictHook>) {
        let mut hooks = self.pre_predict_hooks.lock().unwrap();
        hooks.push(hook);
        hooks.sort_by_key(|h| -h.priority());
    }

    /// Add post-predict hook
    pub fn add_post_predict_hook(&self, hook: Box<dyn PostPredictHook>) {
        let mut hooks = self.post_predict_hooks.lock().unwrap();
        hooks.push(hook);
        hooks.sort_by_key(|h| -h.priority());
    }

    /// Add error hook
    pub fn add_error_hook(&self, hook: Box<dyn ErrorHook>) {
        let mut hooks = self.error_hooks.lock().unwrap();
        hooks.push(hook);
        hooks.sort_by_key(|h| -h.priority());
    }

    /// Execute pre-fit hooks
    pub fn execute_pre_fit_hooks(&self, context: &mut FitContext) -> Result<()> {
        let hooks = self.pre_fit_hooks.lock().unwrap();
        for hook in hooks.iter() {
            hook.execute(context)?;
        }
        Ok(())
    }

    /// Execute post-fit hooks
    pub fn execute_post_fit_hooks(&self, context: &FitContext, result: &FitResult) -> Result<()> {
        let hooks = self.post_fit_hooks.lock().unwrap();
        for hook in hooks.iter() {
            hook.execute(context, result)?;
        }
        Ok(())
    }

    /// Execute pre-predict hooks
    pub fn execute_pre_predict_hooks(&self, context: &mut PredictContext) -> Result<()> {
        let hooks = self.pre_predict_hooks.lock().unwrap();
        for hook in hooks.iter() {
            hook.execute(context)?;
        }
        Ok(())
    }

    /// Execute post-predict hooks
    pub fn execute_post_predict_hooks(
        &self,
        context: &PredictContext,
        predictions: &mut Array1<f64>,
    ) -> Result<()> {
        let hooks = self.post_predict_hooks.lock().unwrap();
        for hook in hooks.iter() {
            hook.execute(context, predictions)?;
        }
        Ok(())
    }

    /// Execute error hooks
    pub fn execute_error_hooks(&self, context: &ErrorContext) -> Result<()> {
        let hooks = self.error_hooks.lock().unwrap();
        for hook in hooks.iter() {
            hook.execute(context)?;
        }
        Ok(())
    }
}

/// Middleware interface for baseline pipelines
pub trait PipelineMiddleware: Send + Sync + Debug {
    /// Middleware name
    fn name(&self) -> &str;

    /// Process before main operation
    fn before(&self, context: &mut MiddlewareContext) -> Result<()>;

    /// Process after main operation
    fn after(&self, context: &mut MiddlewareContext, result: &mut MiddlewareResult) -> Result<()>;

    /// Handle errors
    fn on_error(&self, context: &MiddlewareContext, error: &SklearsError) -> Result<()> {
        // Default implementation logs the error
        eprintln!("Middleware '{}' error: {:?}", self.name(), error);
        Ok(())
    }
}

/// Context for middleware execution
#[derive(Debug)]
pub struct MiddlewareContext {
    /// operation
    pub operation: String,
    /// parameters
    pub parameters: HashMap<String, MiddlewareParameter>,
    /// data
    pub data: HashMap<String, Box<dyn Any + Send>>,
    /// metrics
    pub metrics: HashMap<String, f64>,
    /// start_time
    pub start_time: std::time::Instant,
}

/// Result from middleware execution
#[derive(Debug)]
pub struct MiddlewareResult {
    /// success
    pub success: bool,
    /// duration
    pub duration: std::time::Duration,
    /// data
    pub data: HashMap<String, Box<dyn Any + Send>>,
    /// metrics
    pub metrics: HashMap<String, f64>,
    /// warnings
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum MiddlewareParameter {
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
    /// Boolean
    Boolean(bool),
    /// Array
    Array(Vec<MiddlewareParameter>),
}

/// Middleware pipeline for chaining middleware components
pub struct MiddlewarePipeline {
    middleware: Vec<Box<dyn PipelineMiddleware>>,
    error_handler: Option<Box<dyn Fn(&SklearsError) -> Result<()> + Send + Sync>>,
}

impl Default for MiddlewarePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl MiddlewarePipeline {
    /// Create new middleware pipeline
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
            error_handler: None,
        }
    }

    /// Add middleware to pipeline
    pub fn add_middleware(&mut self, middleware: Box<dyn PipelineMiddleware>) {
        self.middleware.push(middleware);
    }

    /// Set error handler
    pub fn set_error_handler<F>(&mut self, handler: F)
    where
        F: Fn(&SklearsError) -> Result<()> + Send + Sync + 'static,
    {
        self.error_handler = Some(Box::new(handler));
    }

    /// Execute pipeline
    pub fn execute<F>(
        &self,
        mut context: MiddlewareContext,
        operation: F,
    ) -> Result<MiddlewareResult>
    where
        F: FnOnce(&mut MiddlewareContext) -> Result<MiddlewareResult>,
    {
        // Execute before middleware
        for middleware in &self.middleware {
            if let Err(e) = middleware.before(&mut context) {
                middleware.on_error(&context, &e)?;
                if let Some(handler) = &self.error_handler {
                    handler(&e)?;
                }
                return Err(e);
            }
        }

        // Execute main operation
        let mut result = match operation(&mut context) {
            Ok(result) => result,
            Err(e) => {
                // Handle error with middleware
                for middleware in &self.middleware {
                    middleware.on_error(&context, &e)?;
                }
                if let Some(handler) = &self.error_handler {
                    handler(&e)?;
                }
                return Err(e);
            }
        };

        // Execute after middleware (in reverse order)
        for middleware in self.middleware.iter().rev() {
            if let Err(e) = middleware.after(&mut context, &mut result) {
                middleware.on_error(&context, &e)?;
                if let Some(handler) = &self.error_handler {
                    handler(&e)?;
                }
                return Err(e);
            }
        }

        Ok(result)
    }
}

/// Built-in middleware implementations
pub mod middleware {
    use super::*;

    /// Logging middleware
    #[derive(Debug)]
    pub struct LoggingMiddleware {
        /// log_level
        pub log_level: LogLevel,
        /// include_timing
        pub include_timing: bool,
        /// include_parameters
        pub include_parameters: bool,
    }

    impl LoggingMiddleware {
        pub fn new(log_level: LogLevel) -> Self {
            Self {
                log_level,
                include_timing: true,
                include_parameters: true,
            }
        }
    }

    impl PipelineMiddleware for LoggingMiddleware {
        fn name(&self) -> &str {
            "logging"
        }

        fn before(&self, context: &mut MiddlewareContext) -> Result<()> {
            println!("Starting operation: {}", context.operation);
            if self.include_parameters {
                println!("Parameters: {:?}", context.parameters);
            }
            Ok(())
        }

        fn after(
            &self,
            context: &mut MiddlewareContext,
            result: &mut MiddlewareResult,
        ) -> Result<()> {
            if self.include_timing {
                println!(
                    "Operation '{}' completed in {:?}",
                    context.operation, result.duration
                );
            }
            if !result.warnings.is_empty() {
                println!("Warnings: {:?}", result.warnings);
            }
            Ok(())
        }
    }

    /// Validation middleware
    #[derive(Debug)]
    pub struct ValidationMiddleware {
        /// validate_inputs
        pub validate_inputs: bool,
        /// validate_outputs
        pub validate_outputs: bool,
        /// strict_mode
        pub strict_mode: bool,
    }

    impl Default for ValidationMiddleware {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ValidationMiddleware {
        pub fn new() -> Self {
            Self {
                validate_inputs: true,
                validate_outputs: true,
                strict_mode: false,
            }
        }
    }

    impl PipelineMiddleware for ValidationMiddleware {
        fn name(&self) -> &str {
            "validation"
        }

        fn before(&self, context: &mut MiddlewareContext) -> Result<()> {
            if self.validate_inputs {
                // Validate input parameters
                for (key, value) in &context.parameters {
                    match value {
                        MiddlewareParameter::Float(f) => {
                            if f.is_nan() || f.is_infinite() {
                                return Err(SklearsError::InvalidInput(format!(
                                    "Invalid float value for parameter '{}': {}",
                                    key, f
                                )));
                            }
                        }
                        _ => {} // Add more validation as needed
                    }
                }
            }
            Ok(())
        }

        fn after(
            &self,
            _context: &mut MiddlewareContext,
            result: &mut MiddlewareResult,
        ) -> Result<()> {
            if self.validate_outputs && !result.success {
                if self.strict_mode {
                    return Err(SklearsError::InvalidInput(
                        "Operation failed validation".to_string(),
                    ));
                } else {
                    result
                        .warnings
                        .push("Operation validation failed".to_string());
                }
            }
            Ok(())
        }
    }

    /// Performance monitoring middleware
    #[derive(Debug)]
    pub struct PerformanceMiddleware {
        /// collect_metrics
        pub collect_metrics: bool,
        /// memory_tracking
        pub memory_tracking: bool,
    }

    impl Default for PerformanceMiddleware {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PerformanceMiddleware {
        pub fn new() -> Self {
            Self {
                collect_metrics: true,
                memory_tracking: true,
            }
        }
    }

    impl PipelineMiddleware for PerformanceMiddleware {
        fn name(&self) -> &str {
            "performance"
        }

        fn before(&self, context: &mut MiddlewareContext) -> Result<()> {
            context.start_time = std::time::Instant::now();
            if self.memory_tracking {
                // In a real implementation, this would collect actual memory usage
                context.metrics.insert("memory_before_mb".to_string(), 0.0);
            }
            Ok(())
        }

        fn after(
            &self,
            context: &mut MiddlewareContext,
            result: &mut MiddlewareResult,
        ) -> Result<()> {
            result.duration = context.start_time.elapsed();

            if self.collect_metrics {
                result.metrics.insert(
                    "duration_ms".to_string(),
                    result.duration.as_millis() as f64,
                );
            }

            if self.memory_tracking {
                // In a real implementation, this would collect actual memory usage
                result.metrics.insert("memory_after_mb".to_string(), 0.0);
                result.metrics.insert("memory_delta_mb".to_string(), 0.0);
            }

            Ok(())
        }
    }
}

/// Custom strategy registration system
pub struct CustomStrategyRegistry {
    classification_strategies: RwLock<HashMap<String, Box<dyn CustomClassificationStrategy>>>,
    regression_strategies: RwLock<HashMap<String, Box<dyn CustomRegressionStrategy>>>,
    strategy_metadata: RwLock<HashMap<String, StrategyMetadata>>,
}

/// Interface for custom classification strategies
pub trait CustomClassificationStrategy: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<i32>>;
    fn predict_proba(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    fn fit(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<()>;
    fn is_fitted(&self) -> bool;
    fn get_parameters(&self) -> HashMap<String, f64>;
}

/// Interface for custom regression strategies
pub trait CustomRegressionStrategy: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>>;
    fn fit(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<()>;
    fn is_fitted(&self) -> bool;
    fn get_parameters(&self) -> HashMap<String, f64>;
}

/// Metadata for custom strategies
#[derive(Debug, Clone)]
pub struct StrategyMetadata {
    /// author
    pub author: String,
    /// version
    pub version: String,
    /// description
    pub description: String,
    /// task_type
    pub task_type: TaskType,
    /// complexity
    pub complexity: StrategyComplexity,
    /// requirements
    pub requirements: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum StrategyComplexity {
    /// Constant
    Constant,
    /// Linear
    Linear,
    /// Quadratic
    Quadratic,
    /// Exponential
    Exponential,
}

impl Default for CustomStrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomStrategyRegistry {
    /// Create new custom strategy registry
    pub fn new() -> Self {
        Self {
            classification_strategies: RwLock::new(HashMap::new()),
            regression_strategies: RwLock::new(HashMap::new()),
            strategy_metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Register a custom classification strategy
    pub fn register_classification_strategy(
        &self,
        strategy: Box<dyn CustomClassificationStrategy>,
        metadata: StrategyMetadata,
    ) -> Result<()> {
        let name = strategy.name().to_string();

        // Validate metadata
        if metadata.task_type != TaskType::Classification {
            return Err(SklearsError::InvalidInput(
                "Strategy task type must be Classification".to_string(),
            ));
        }

        // Check for conflicts
        if self
            .classification_strategies
            .read()
            .unwrap()
            .contains_key(&name)
        {
            return Err(SklearsError::InvalidInput(format!(
                "Classification strategy '{}' already registered",
                name
            )));
        }

        // Register strategy
        self.classification_strategies
            .write()
            .unwrap()
            .insert(name.clone(), strategy);
        self.strategy_metadata
            .write()
            .unwrap()
            .insert(name, metadata);

        Ok(())
    }

    /// Register a custom regression strategy
    pub fn register_regression_strategy(
        &self,
        strategy: Box<dyn CustomRegressionStrategy>,
        metadata: StrategyMetadata,
    ) -> Result<()> {
        let name = strategy.name().to_string();

        // Validate metadata
        if metadata.task_type != TaskType::Regression {
            return Err(SklearsError::InvalidInput(
                "Strategy task type must be Regression".to_string(),
            ));
        }

        // Check for conflicts
        if self
            .regression_strategies
            .read()
            .unwrap()
            .contains_key(&name)
        {
            return Err(SklearsError::InvalidInput(format!(
                "Regression strategy '{}' already registered",
                name
            )));
        }

        // Register strategy
        self.regression_strategies
            .write()
            .unwrap()
            .insert(name.clone(), strategy);
        self.strategy_metadata
            .write()
            .unwrap()
            .insert(name, metadata);

        Ok(())
    }

    /// List all registered classification strategies
    pub fn list_classification_strategies(&self) -> Vec<String> {
        self.classification_strategies
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    /// List all registered regression strategies
    pub fn list_regression_strategies(&self) -> Vec<String> {
        self.regression_strategies
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    /// Get strategy metadata
    pub fn get_strategy_metadata(&self, name: &str) -> Option<StrategyMetadata> {
        self.strategy_metadata.read().unwrap().get(name).cloned()
    }

    /// Get classification strategy by name
    pub fn get_classification_strategy(
        &self,
        name: &str,
    ) -> Option<Box<dyn CustomClassificationStrategy>> {
        // Note: In practice, this would return a reference or clone
        // For simplicity, we return None to indicate the lookup
        if self
            .classification_strategies
            .read()
            .unwrap()
            .contains_key(name)
        {
            // Would return actual strategy reference
            None
        } else {
            None
        }
    }

    /// Get regression strategy by name
    pub fn get_regression_strategy(&self, name: &str) -> Option<Box<dyn CustomRegressionStrategy>> {
        // Note: In practice, this would return a reference or clone
        // For simplicity, we return None to indicate the lookup
        if self
            .regression_strategies
            .read()
            .unwrap()
            .contains_key(name)
        {
            // Would return actual strategy reference
            None
        } else {
            None
        }
    }

    /// Remove a strategy from registry
    pub fn unregister_strategy(&self, name: &str) -> Result<()> {
        let mut metadata = self.strategy_metadata.write().unwrap();

        if let Some(meta) = metadata.remove(name) {
            match meta.task_type {
                TaskType::Classification => {
                    self.classification_strategies.write().unwrap().remove(name);
                }
                TaskType::Regression => {
                    self.regression_strategies.write().unwrap().remove(name);
                }
                _ => {
                    return Err(SklearsError::InvalidInput(
                        "Unsupported task type for strategy removal".to_string(),
                    ))
                }
            }
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Strategy '{}' not found",
                name
            )))
        }
    }
}

/// Integration utilities for evaluation frameworks
pub struct EvaluationIntegration {
    evaluators: HashMap<String, Box<dyn EvaluationFramework>>,
    metrics: HashMap<String, Box<dyn MetricComputer>>,
}

/// Evaluation framework interface
pub trait EvaluationFramework: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn evaluate(&self, estimator: &dyn Any, test_data: &TestData) -> Result<EvaluationResult>;
    fn supported_tasks(&self) -> Vec<TaskType>;
}

/// Metric computer interface
pub trait MetricComputer: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn compute(&self, y_true: &ArrayView1<f64>, y_pred: &ArrayView1<f64>) -> Result<MetricResult>;
    fn metric_type(&self) -> MetricType;
}

/// Test data structure
#[derive(Debug)]
pub struct TestData {
    /// x
    pub x: Array2<f64>,
    /// y
    pub y: Array1<f64>,
    /// metadata
    pub metadata: HashMap<String, String>,
}

/// Evaluation result
#[derive(Debug)]
pub struct EvaluationResult {
    /// primary_metric
    pub primary_metric: f64,
    /// metrics
    pub metrics: HashMap<String, f64>,
    /// confidence_intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// execution_time
    pub execution_time: std::time::Duration,
    /// warnings
    pub warnings: Vec<String>,
}

/// Metric result
#[derive(Debug)]
pub struct MetricResult {
    /// value
    pub value: f64,
    /// confidence_interval
    pub confidence_interval: Option<(f64, f64)>,
    /// metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum MetricType {
    /// Accuracy
    Accuracy,
    /// Loss
    Loss,
    /// Similarity
    Similarity,
    /// Distance
    Distance,
    /// Custom
    Custom(String),
}

impl Default for EvaluationIntegration {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationIntegration {
    /// Create new evaluation integration
    pub fn new() -> Self {
        Self {
            evaluators: HashMap::new(),
            metrics: HashMap::new(),
        }
    }

    /// Register evaluation framework
    pub fn register_evaluator(&mut self, evaluator: Box<dyn EvaluationFramework>) {
        let name = evaluator.name().to_string();
        self.evaluators.insert(name, evaluator);
    }

    /// Register metric computer
    pub fn register_metric(&mut self, metric: Box<dyn MetricComputer>) {
        let name = metric.name().to_string();
        self.metrics.insert(name, metric);
    }

    /// Evaluate estimator with specified framework
    pub fn evaluate(
        &self,
        framework_name: &str,
        estimator: &dyn Any,
        test_data: &TestData,
    ) -> Result<EvaluationResult> {
        if let Some(evaluator) = self.evaluators.get(framework_name) {
            evaluator.evaluate(estimator, test_data)
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Evaluation framework '{}' not found",
                framework_name
            )))
        }
    }

    /// Compute metric
    pub fn compute_metric(
        &self,
        metric_name: &str,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> Result<MetricResult> {
        if let Some(metric) = self.metrics.get(metric_name) {
            metric.compute(y_true, y_pred)
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Metric '{}' not found",
                metric_name
            )))
        }
    }

    /// List available evaluators
    pub fn list_evaluators(&self) -> Vec<String> {
        self.evaluators.keys().cloned().collect()
    }

    /// List available metrics
    pub fn list_metrics(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Mock plugin for testing
    #[derive(Debug)]
    struct MockPlugin {
        name: String,
        version: String,
    }

    impl BaselinePlugin for MockPlugin {
        fn name(&self) -> &str {
            &self.name
        }
        fn version(&self) -> &str {
            &self.version
        }
        fn description(&self) -> &str {
            "Mock plugin for testing"
        }

        fn initialize(&mut self, _config: &PluginConfig) -> Result<()> {
            Ok(())
        }
        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn is_compatible(&self, _data_info: &DataInfo) -> bool {
            true
        }

        fn metadata(&self) -> PluginMetadata {
            /// PluginMetadata
            PluginMetadata {
                author: "Test Author".to_string(),
                license: "MIT".to_string(),
                homepage: "https://example.com".to_string(),
                supported_tasks: vec![TaskType::Classification, TaskType::Regression],
                requirements: vec![],
            }
        }
    }

    // Mock hook for testing
    #[derive(Debug)]
    struct MockPreFitHook;

    impl PreFitHook for MockPreFitHook {
        fn execute(&self, _context: &mut FitContext) -> Result<()> {
            // Mock implementation
            Ok(())
        }

        fn priority(&self) -> i32 {
            1
        }
    }

    #[test]
    fn test_plugin_registry() {
        let registry = PluginRegistry::new();

        let plugin = Arc::new(MockPlugin {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
        });

        let config = PluginConfig {
            parameters: HashMap::new(),
            resources: ResourceConfig {
                max_memory_mb: 1024,
                max_cpu_cores: 4,
                temp_directory: "/tmp".to_string(),
                cache_enabled: true,
            },
            logging: LoggingConfig {
                level: LogLevel::Info,
                output_file: None,
                include_timestamps: true,
            },
        };

        // Register plugin
        let result = registry.register_plugin(plugin, config);
        assert!(result.is_ok());

        // Check plugin is registered
        let plugins = registry.list_plugins();
        assert!(plugins.contains(&"test_plugin".to_string()));

        // Activate plugin
        let result = registry.activate_plugin("test_plugin");
        assert!(result.is_ok());

        // Check plugin is active
        let active_plugins = registry.get_active_plugins();
        assert!(active_plugins.contains(&"test_plugin".to_string()));
    }

    #[test]
    fn test_hook_system() {
        let hook_system = HookSystem::new();

        // Add hook
        hook_system.add_pre_fit_hook(Box::new(MockPreFitHook));

        // Create context
        let mut context = FitContext {
            estimator_name: "test".to_string(),
            strategy: "mean".to_string(),
            x: Array2::zeros((10, 5)),
            y: Array1::zeros(10),
            metadata: HashMap::new(),
            start_time: std::time::Instant::now(),
        };

        // Execute hooks
        let result = hook_system.execute_pre_fit_hooks(&mut context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_middleware_pipeline() {
        let mut pipeline = MiddlewarePipeline::new();

        // Add logging middleware
        pipeline.add_middleware(Box::new(middleware::LoggingMiddleware::new(LogLevel::Info)));

        // Create context
        let context = MiddlewareContext {
            operation: "test_operation".to_string(),
            parameters: HashMap::new(),
            data: HashMap::new(),
            metrics: HashMap::new(),
            start_time: std::time::Instant::now(),
        };

        // Execute pipeline
        let result = pipeline.execute(context, |_ctx| {
            Ok(MiddlewareResult {
                success: true,
                duration: std::time::Duration::from_millis(100),
                data: HashMap::new(),
                metrics: HashMap::new(),
                warnings: Vec::new(),
            })
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_evaluation_integration() {
        let integration = EvaluationIntegration::new();

        // Test empty state
        assert!(integration.list_evaluators().is_empty());
        assert!(integration.list_metrics().is_empty());
    }
}
