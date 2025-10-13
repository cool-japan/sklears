//! Plugin Architecture for Calibration Extensibility
//!
//! This module implements a comprehensive plugin system for calibration methods,
//! including plugin discovery, dynamic loading, hooks for calibration callbacks,
//! custom metric registration, and middleware for calibration pipelines.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result},
    types::Float,
};
use std::sync::{Arc};
use std::any::{Any, TypeId};

use crate::{CalibrationEstimator};

/// Plugin trait for calibration methods
pub trait CalibrationPlugin: Send + Sync {
    /// Plugin name
    fn name(&self) -> &str;
    
    /// Plugin version
    fn version(&self) -> &str;
    
    /// Plugin description
    fn description(&self) -> &str;
    
    /// Create a new calibrator instance
    fn create_calibrator(&self) -> Result<Box<dyn CalibrationEstimator>>;
    
    /// Get plugin configuration schema
    fn config_schema(&self) -> HashMap<String, ParameterType>;
    
    /// Initialize plugin with configuration
    fn initialize(&mut self, config: &HashMap<String, ParameterValue>) -> Result<()>;
    
    /// Get plugin dependencies
    fn dependencies(&self) -> Vec<String> {
        Vec::new()
    }
    
    /// Plugin cleanup
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Parameter types for plugin configuration
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterType {
    Float { min: Option<Float>, max: Option<Float> },
    Integer { min: Option<i64>, max: Option<i64> },
    String { max_length: Option<usize> },
    Boolean,
    Array(Box<ParameterType>),
    Object(HashMap<String, ParameterType>),
}

/// Parameter values for plugin configuration
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Float(Float),
    Integer(i64),
    String(String),
    Boolean(bool),
    Array(Vec<ParameterValue>),
    Object(HashMap<String, ParameterValue>),
}

/// Plugin registry for managing calibration plugins
#[derive(Debug)]
pub struct PluginRegistry {
    /// Registered plugins
    plugins: RwLock<HashMap<String, Arc<Mutex<dyn CalibrationPlugin>>>>,
    /// Plugin metadata
    metadata: RwLock<HashMap<String, PluginMetadata>>,
    /// Plugin dependencies graph
    dependencies: RwLock<HashMap<String, Vec<String>>>,
    /// Active hooks
    hooks: RwLock<HashMap<HookType, Vec<Arc<dyn Hook>>>>,
    /// Custom metrics
    metrics: RwLock<HashMap<String, Arc<dyn CustomMetric>>>,
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub tags: Vec<String>,
    pub is_enabled: bool,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            dependencies: RwLock::new(HashMap::new()),
            hooks: RwLock::new(HashMap::new()),
            metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Register a plugin
    pub fn register_plugin(
        &self,
        plugin: Arc<Mutex<dyn CalibrationPlugin>>,
        metadata: PluginMetadata,
    ) -> Result<()> {
        let plugin_name = metadata.name.clone();
        
        // Check for name conflicts
        {
            let plugins = self.plugins.read().unwrap();
            if plugins.contains_key(&plugin_name) {
                return Err(SklearsError::InvalidInput(
                    format!("Plugin '{}' is already registered", plugin_name)
                ));
            }
        }

        // Register dependencies
        {
            let dependencies = {
                let plugin_guard = plugin.lock().unwrap();
                plugin_guard.dependencies()
            };
            
            self.dependencies.write().unwrap().insert(plugin_name.clone(), dependencies);
        }

        // Store plugin and metadata
        {
            self.plugins.write().unwrap().insert(plugin_name.clone(), plugin);
            self.metadata.write().unwrap().insert(plugin_name.clone(), metadata);
        }

        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister_plugin(&self, name: &str) -> Result<()> {
        // Check if plugin exists
        {
            let plugins = self.plugins.read().unwrap();
            if !plugins.contains_key(name) {
                return Err(SklearsError::InvalidInput(
                    format!("Plugin '{}' is not registered", name)
                ));
            }
        }

        // Cleanup plugin
        {
            let plugins = self.plugins.read().unwrap();
            if let Some(plugin) = plugins.get(name) {
                let mut plugin_guard = plugin.lock().unwrap();
                plugin_guard.cleanup()?;
            }
        }

        // Remove from registry
        {
            self.plugins.write().unwrap().remove(name);
            self.metadata.write().unwrap().remove(name);
            self.dependencies.write().unwrap().remove(name);
        }

        Ok(())
    }

    /// Get available plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        let metadata = self.metadata.read().unwrap();
        metadata.values().cloned().collect()
    }

    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Arc<Mutex<dyn CalibrationPlugin>>> {
        let plugins = self.plugins.read().unwrap();
        plugins.get(name).cloned()
    }

    /// Create calibrator from plugin
    pub fn create_calibrator(&self, plugin_name: &str) -> Result<Box<dyn CalibrationEstimator>> {
        let plugin = self.get_plugin(plugin_name)
            .ok_or_else(|| SklearsError::InvalidInput(
                format!("Plugin '{}' not found", plugin_name)
            ))?;

        let plugin_guard = plugin.lock().unwrap();
        plugin_guard.create_calibrator()
    }

    /// Enable plugin
    pub fn enable_plugin(&self, name: &str) -> Result<()> {
        let mut metadata = self.metadata.write().unwrap();
        if let Some(meta) = metadata.get_mut(name) {
            meta.is_enabled = true;
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(
                format!("Plugin '{}' not found", name)
            ))
        }
    }

    /// Disable plugin
    pub fn disable_plugin(&self, name: &str) -> Result<()> {
        let mut metadata = self.metadata.write().unwrap();
        if let Some(meta) = metadata.get_mut(name) {
            meta.is_enabled = false;
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(
                format!("Plugin '{}' not found", name)
            ))
        }
    }

    /// Resolve plugin dependencies
    pub fn resolve_dependencies(&self, plugin_name: &str) -> Result<Vec<String>> {
        let dependencies = self.dependencies.read().unwrap();
        let mut resolved = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        self.resolve_dependencies_recursive(plugin_name, &dependencies, &mut resolved, &mut visited)?;
        Ok(resolved)
    }

    fn resolve_dependencies_recursive(
        &self,
        plugin_name: &str,
        dependencies: &HashMap<String, Vec<String>>,
        resolved: &mut Vec<String>,
        visited: &mut std::collections::HashSet<String>,
    ) -> Result<()> {
        if visited.contains(plugin_name) {
            return Err(SklearsError::InvalidInput(
                format!("Circular dependency detected for plugin '{}'", plugin_name)
            ));
        }

        visited.insert(plugin_name.to_string());

        if let Some(deps) = dependencies.get(plugin_name) {
            for dep in deps {
                self.resolve_dependencies_recursive(dep, dependencies, resolved, visited)?;
                if !resolved.contains(dep) {
                    resolved.push(dep.clone());
                }
            }
        }

        if !resolved.contains(&plugin_name.to_string()) {
            resolved.push(plugin_name.to_string());
        }

        visited.remove(plugin_name);
        Ok(())
    }

    /// Register a hook
    pub fn register_hook(&self, hook_type: HookType, hook: Arc<dyn Hook>) {
        let mut hooks = self.hooks.write().unwrap();
        hooks.entry(hook_type).or_insert_with(Vec::new).push(hook);
    }

    /// Execute hooks
    pub fn execute_hooks(&self, hook_type: HookType, context: &HookContext) -> Result<()> {
        let hooks = self.hooks.read().unwrap();
        if let Some(hook_list) = hooks.get(&hook_type) {
            for hook in hook_list {
                hook.execute(context)?;
            }
        }
        Ok(())
    }

    /// Register custom metric
    pub fn register_metric(&self, name: String, metric: Arc<dyn CustomMetric>) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.insert(name, metric);
    }

    /// Get custom metric
    pub fn get_metric(&self, name: &str) -> Option<Arc<dyn CustomMetric>> {
        let metrics = self.metrics.read().unwrap();
        metrics.get(name).cloned()
    }

    /// List custom metrics
    pub fn list_metrics(&self) -> Vec<String> {
        let metrics = self.metrics.read().unwrap();
        metrics.keys().cloned().collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Hook types for calibration events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookType {
    /// Before calibration training
    PreFit,
    /// After calibration training
    PostFit,
    /// Before prediction
    PrePredict,
    /// After prediction
    PostPredict,
    /// Before calibration evaluation
    PreEvaluate,
    /// After calibration evaluation
    PostEvaluate,
    /// Custom hook type
    Custom(&'static str),
}

/// Hook context containing event data
#[derive(Debug)]
pub struct HookContext {
    /// Event type
    pub hook_type: HookType,
    /// Plugin name
    pub plugin_name: String,
    /// Calibrator instance
    pub calibrator: Option<*const dyn CalibrationEstimator>,
    /// Input probabilities
    pub probabilities: Option<Array1<Float>>,
    /// Target values
    pub targets: Option<Array1<i32>>,
    /// Predictions
    pub predictions: Option<Array1<Float>>,
    /// Evaluation metrics
    pub metrics: Option<HashMap<String, Float>>,
    /// Additional data
    pub data: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl HookContext {
    /// Create a new hook context
    pub fn new(hook_type: HookType, plugin_name: String) -> Self {
        Self {
            hook_type,
            plugin_name,
            calibrator: None,
            probabilities: None,
            targets: None,
            predictions: None,
            metrics: None,
            data: HashMap::new(),
        }
    }

    /// Add custom data
    pub fn with_data<T: Any + Send + Sync>(mut self, key: String, value: T) -> Self {
        self.data.insert(key, Box::new(value));
        self
    }

    /// Get custom data
    pub fn get_data<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.data.get(key)?.downcast_ref::<T>()
    }
}

/// Hook trait for calibration events
pub trait Hook: Send + Sync {
    /// Execute hook
    fn execute(&self, context: &HookContext) -> Result<()>;
    
    /// Hook name
    fn name(&self) -> &str;
    
    /// Hook description
    fn description(&self) -> &str;
}

/// Custom metric trait
pub trait CustomMetric: Send + Sync {
    /// Metric name
    fn name(&self) -> &str;
    
    /// Compute metric
    fn compute(&self, predictions: &Array1<Float>, targets: &Array1<i32>) -> Result<Float>;
    
    /// Metric description
    fn description(&self) -> &str;
    
    /// Whether higher values are better
    fn higher_is_better(&self) -> bool;
}

/// Middleware for calibration pipelines
pub trait CalibrationMiddleware: Send + Sync {
    /// Process before calibration
    fn pre_process(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<(Array1<Float>, Array1<i32>)>;
    
    /// Process after calibration
    fn post_process(
        &self,
        predictions: &Array1<Float>,
        original_probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>>;
    
    /// Middleware name
    fn name(&self) -> &str;
}

/// Calibration pipeline with middleware support
#[derive(Debug)]
pub struct CalibrationPipeline {
    /// Middleware stack
    middleware: Vec<Arc<dyn CalibrationMiddleware>>,
    /// Base calibrator
    calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Plugin registry
    registry: Arc<PluginRegistry>,
    /// Pipeline configuration
    config: PipelineConfig,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable hooks
    pub enable_hooks: bool,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Parallel processing
    pub parallel_processing: bool,
    /// Error handling strategy
    pub error_handling: ErrorHandling,
}

#[derive(Debug, Clone)]
pub enum ErrorHandling {
    /// Stop on first error
    StopOnError,
    /// Continue with warnings
    ContinueWithWarnings,
    /// Ignore errors
    IgnoreErrors,
}

impl CalibrationPipeline {
    /// Create a new calibration pipeline
    pub fn new(registry: Arc<PluginRegistry>) -> Self {
        Self {
            middleware: Vec::new(),
            calibrator: None,
            registry,
            config: PipelineConfig {
                enable_hooks: true,
                enable_metrics: true,
                parallel_processing: false,
                error_handling: ErrorHandling::StopOnError,
            },
        }
    }

    /// Add middleware to the pipeline
    pub fn add_middleware(&mut self, middleware: Arc<dyn CalibrationMiddleware>) {
        self.middleware.push(middleware);
    }

    /// Set the base calibrator
    pub fn set_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.calibrator = Some(calibrator);
    }

    /// Set calibrator from plugin
    pub fn set_calibrator_from_plugin(&mut self, plugin_name: &str) -> Result<()> {
        let calibrator = self.registry.create_calibrator(plugin_name)?;
        self.calibrator = Some(calibrator);
        Ok(())
    }

    /// Configure pipeline
    pub fn configure(&mut self, config: PipelineConfig) {
        self.config = config;
    }

    /// Fit the calibration pipeline
    pub fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        let mut processed_probs = probabilities.clone();
        let mut processed_targets = targets.clone();

        // Execute pre-fit hooks
        if self.config.enable_hooks {
            let context = HookContext::new(HookType::PreFit, "pipeline".to_string())
                .with_data("probabilities".to_string(), probabilities.clone())
                .with_data("targets".to_string(), targets.clone());
            self.registry.execute_hooks(HookType::PreFit, &context)?;
        }

        // Apply pre-processing middleware
        for middleware in &self.middleware {
            let (new_probs, new_targets) = match middleware.pre_process(&processed_probs, &processed_targets) {
                Ok(result) => result,
                Err(e) => match self.config.error_handling {
                    ErrorHandling::StopOnError => return Err(e),
                    ErrorHandling::ContinueWithWarnings => {
                        eprintln!("Warning in middleware {}: {}", middleware.name(), e);
                        (processed_probs, processed_targets)
                    }
                    ErrorHandling::IgnoreErrors => (processed_probs, processed_targets),
                }
            };
            processed_probs = new_probs;
            processed_targets = new_targets;
        }

        // Fit the calibrator
        if let Some(ref mut calibrator) = self.calibrator {
            calibrator.fit(&processed_probs, &processed_targets)?;
        } else {
            return Err(SklearsError::InvalidInput(
                "No calibrator set in pipeline".to_string()
            ));
        }

        // Execute post-fit hooks
        if self.config.enable_hooks {
            let context = HookContext::new(HookType::PostFit, "pipeline".to_string())
                .with_data("processed_probabilities".to_string(), processed_probs)
                .with_data("processed_targets".to_string(), processed_targets);
            self.registry.execute_hooks(HookType::PostFit, &context)?;
        }

        Ok(())
    }

    /// Predict using the calibration pipeline
    pub fn predict(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let mut processed_probs = probabilities.clone();

        // Execute pre-predict hooks
        if self.config.enable_hooks {
            let context = HookContext::new(HookType::PrePredict, "pipeline".to_string())
                .with_data("probabilities".to_string(), probabilities.clone());
            self.registry.execute_hooks(HookType::PrePredict, &context)?;
        }

        // Apply pre-processing middleware
        let dummy_targets = Array1::zeros(probabilities.len());
        for middleware in &self.middleware {
            let (new_probs, _) = match middleware.pre_process(&processed_probs, &dummy_targets) {
                Ok(result) => result,
                Err(e) => match self.config.error_handling {
                    ErrorHandling::StopOnError => return Err(e),
                    ErrorHandling::ContinueWithWarnings => {
                        eprintln!("Warning in middleware {}: {}", middleware.name(), e);
                        (processed_probs, dummy_targets)
                    }
                    ErrorHandling::IgnoreErrors => (processed_probs, dummy_targets),
                }
            };
            processed_probs = new_probs;
        }

        // Get predictions from calibrator
        let mut predictions = if let Some(ref calibrator) = self.calibrator {
            calibrator.predict_proba(&processed_probs)?
        } else {
            return Err(SklearsError::InvalidInput(
                "No calibrator set in pipeline".to_string()
            ));
        };

        // Apply post-processing middleware (in reverse order)
        for middleware in self.middleware.iter().rev() {
            predictions = match middleware.post_process(&predictions, probabilities) {
                Ok(result) => result,
                Err(e) => match self.config.error_handling {
                    ErrorHandling::StopOnError => return Err(e),
                    ErrorHandling::ContinueWithWarnings => {
                        eprintln!("Warning in middleware {}: {}", middleware.name(), e);
                        predictions
                    }
                    ErrorHandling::IgnoreErrors => predictions,
                }
            };
        }

        // Execute post-predict hooks
        if self.config.enable_hooks {
            let context = HookContext::new(HookType::PostPredict, "pipeline".to_string())
                .with_data("predictions".to_string(), predictions.clone());
            self.registry.execute_hooks(HookType::PostPredict, &context)?;
        }

        Ok(predictions)
    }

    /// Evaluate pipeline with custom metrics
    pub fn evaluate(&self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<HashMap<String, Float>> {
        let predictions = self.predict(probabilities)?;
        let mut results = HashMap::new();

        // Execute pre-evaluate hooks
        if self.config.enable_hooks {
            let context = HookContext::new(HookType::PreEvaluate, "pipeline".to_string())
                .with_data("predictions".to_string(), predictions.clone())
                .with_data("targets".to_string(), targets.clone());
            self.registry.execute_hooks(HookType::PreEvaluate, &context)?;
        }

        // Compute custom metrics
        if self.config.enable_metrics {
            let metrics = self.registry.list_metrics();
            for metric_name in metrics {
                if let Some(metric) = self.registry.get_metric(&metric_name) {
                    match metric.compute(&predictions, targets) {
                        Ok(value) => {
                            results.insert(metric_name, value);
                        }
                        Err(e) => match self.config.error_handling {
                            ErrorHandling::StopOnError => return Err(e),
                            ErrorHandling::ContinueWithWarnings => {
                                eprintln!("Warning computing metric {}: {}", metric_name, e);
                            }
                            ErrorHandling::IgnoreErrors => {}
                        }
                    }
                }
            }
        }

        // Execute post-evaluate hooks
        if self.config.enable_hooks {
            let context = HookContext::new(HookType::PostEvaluate, "pipeline".to_string())
                .with_data("metrics".to_string(), results.clone());
            self.registry.execute_hooks(HookType::PostEvaluate, &context)?;
        }

        Ok(results)
    }
}

/// Example plugin implementation
#[derive(Debug)]
pub struct ExamplePlugin {
    name: String,
    version: String,
    description: String,
    config: HashMap<String, ParameterValue>,
}

impl ExamplePlugin {
    pub fn new() -> Self {
        Self {
            name: "example_plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Example calibration plugin".to_string(),
            config: HashMap::new(),
        }
    }
}

impl CalibrationPlugin for ExamplePlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn create_calibrator(&self) -> Result<Box<dyn CalibrationEstimator>> {
        Ok(Box::new(crate::SigmoidCalibrator::new()))
    }

    fn config_schema(&self) -> HashMap<String, ParameterType> {
        let mut schema = HashMap::new();
        schema.insert("learning_rate".to_string(), ParameterType::Float { min: Some(0.0), max: Some(1.0) });
        schema.insert("max_iterations".to_string(), ParameterType::Integer { min: Some(1), max: Some(1000) });
        schema.insert("regularization".to_string(), ParameterType::Boolean);
        schema
    }

    fn initialize(&mut self, config: &HashMap<String, ParameterValue>) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }
}

impl Default for ExamplePlugin {
    fn default() -> Self {
        Self::new()
    }
}

/// Example hook implementation
#[derive(Debug)]
pub struct LoggingHook {
    name: String,
}

impl LoggingHook {
    pub fn new() -> Self {
        Self {
            name: "logging_hook".to_string(),
        }
    }
}

impl Hook for LoggingHook {
    fn execute(&self, context: &HookContext) -> Result<()> {
        println!("Hook {} executed for {:?} on plugin {}", 
                 self.name, context.hook_type, context.plugin_name);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Logs calibration events"
    }
}

impl Default for LoggingHook {
    fn default() -> Self {
        Self::new()
    }
}

/// Example custom metric implementation
#[derive(Debug)]
pub struct CustomBrierScore {
    name: String,
}

impl CustomBrierScore {
    pub fn new() -> Self {
        Self {
            name: "custom_brier_score".to_string(),
        }
    }
}

impl CustomMetric for CustomBrierScore {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute(&self, predictions: &Array1<Float>, targets: &Array1<i32>) -> Result<Float> {
        if predictions.len() != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let mut brier_score = 0.0;
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            brier_score += (pred - target as Float).powi(2);
        }
        brier_score /= predictions.len() as Float;

        Ok(brier_score)
    }

    fn description(&self) -> &str {
        "Custom implementation of Brier score"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

impl Default for CustomBrierScore {
    fn default() -> Self {
        Self::new()
    }
}

/// Example middleware implementation
#[derive(Debug)]
pub struct NormalizationMiddleware {
    name: String,
}

impl NormalizationMiddleware {
    pub fn new() -> Self {
        Self {
            name: "normalization_middleware".to_string(),
        }
    }
}

impl CalibrationMiddleware for NormalizationMiddleware {
    fn pre_process(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<(Array1<Float>, Array1<i32>)> {
        // Normalize probabilities to [0, 1] range
        let min_val = probabilities.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = probabilities.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        
        let normalized = if max_val > min_val {
            probabilities.map(|&x| (x - min_val) / (max_val - min_val))
        } else {
            probabilities.clone()
        };

        Ok((normalized, targets.clone()))
    }

    fn post_process(
        &self,
        predictions: &Array1<Float>,
        _original_probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Ensure predictions are in [0, 1] range
        let clipped = predictions.map(|&x| x.clamp(0.0, 1.0));
        Ok(clipped)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Default for NormalizationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_plugin_registry() {
        let registry = PluginRegistry::new();
        
        let plugin = Arc::new(Mutex::new(ExamplePlugin::new()));
        let metadata = PluginMetadata {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test plugin".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            tags: vec!["test".to_string()],
            is_enabled: true,
        };

        registry.register_plugin(plugin, metadata).unwrap();

        let plugins = registry.list_plugins();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "test_plugin");

        let calibrator = registry.create_calibrator("test_plugin").unwrap();
        assert!(calibrator.as_ref() as *const _ != std::ptr::null());
    }

    #[test]
    fn test_hooks() {
        let registry = PluginRegistry::new();
        let hook = Arc::new(LoggingHook::new());
        
        registry.register_hook(HookType::PreFit, hook);

        let context = HookContext::new(HookType::PreFit, "test_plugin".to_string());
        registry.execute_hooks(HookType::PreFit, &context).unwrap();
    }

    #[test]
    fn test_custom_metrics() {
        let registry = PluginRegistry::new();
        let metric = Arc::new(CustomBrierScore::new());
        
        registry.register_metric("brier_score".to_string(), metric);

        let predictions = Array1::from(vec![0.1, 0.7, 0.9]);
        let targets = Array1::from(vec![0, 1, 1]);

        let metric = registry.get_metric("brier_score").unwrap();
        let score = metric.compute(&predictions, &targets).unwrap();
        assert!(score >= 0.0);
    }

    #[test]
    fn test_calibration_pipeline() {
        let registry = Arc::new(PluginRegistry::new());
        let mut pipeline = CalibrationPipeline::new(registry.clone());

        // Add middleware
        let middleware = Arc::new(NormalizationMiddleware::new());
        pipeline.add_middleware(middleware);

        // Set calibrator
        let calibrator = Box::new(crate::SigmoidCalibrator::new());
        pipeline.set_calibrator(calibrator);

        // Test data
        let probabilities = Array1::from(vec![0.1, 0.3, 0.7, 0.9]);
        let targets = Array1::from(vec![0, 0, 1, 1]);

        // Fit and predict
        pipeline.fit(&probabilities, &targets).unwrap();
        let predictions = pipeline.predict(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_parameter_validation() {
        let plugin = ExamplePlugin::new();
        let schema = plugin.config_schema();

        assert!(schema.contains_key("learning_rate"));
        assert!(schema.contains_key("max_iterations"));
        assert!(schema.contains_key("regularization"));

        // Test parameter types
        match &schema["learning_rate"] {
            ParameterType::Float { min, max } => {
                assert_eq!(*min, Some(0.0));
                assert_eq!(*max, Some(1.0));
            }
            _ => panic!("Expected Float parameter type"),
        }
    }

    #[test]
    fn test_dependency_resolution() {
        let registry = PluginRegistry::new();

        // Setup dependencies: A -> B -> C
        {
            let mut deps = registry.dependencies.write().unwrap();
            deps.insert("A".to_string(), vec!["B".to_string()]);
            deps.insert("B".to_string(), vec!["C".to_string()]);
            deps.insert("C".to_string(), vec![]);
        }

        let resolved = registry.resolve_dependencies("A").unwrap();
        assert_eq!(resolved, vec!["C", "B", "A"]);
    }

    #[test]
    fn test_error_handling() {
        let registry = Arc::new(PluginRegistry::new());
        let mut pipeline = CalibrationPipeline::new(registry);

        // Configure error handling
        pipeline.configure(PipelineConfig {
            enable_hooks: false,
            enable_metrics: false,
            parallel_processing: false,
            error_handling: ErrorHandling::ContinueWithWarnings,
        });

        // This should work even with middleware that might fail
        let middleware = Arc::new(NormalizationMiddleware::new());
        pipeline.add_middleware(middleware);

        let calibrator = Box::new(crate::SigmoidCalibrator::new());
        pipeline.set_calibrator(calibrator);

        let probabilities = Array1::from(vec![0.1, 0.3, 0.7, 0.9]);
        let targets = Array1::from(vec![0, 0, 1, 1]);

        pipeline.fit(&probabilities, &targets).unwrap();
        let predictions = pipeline.predict(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
    }
}