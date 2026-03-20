//! Plugin Architecture for Custom Optimizers
//!
//! This module provides a flexible plugin system for implementing custom hyperparameter
//! optimization algorithms. It includes:
//! - Trait-based plugin interface
//! - Plugin registry for dynamic loading
//! - Hook system for optimization callbacks
//! - Middleware support for optimization pipelines
//! - Custom metric registration
//!
//! This enables users to extend the optimization framework with their own algorithms
//! without modifying the core library.

use sklears_core::types::Float;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// Core Plugin Traits
// ============================================================================

/// Core trait for optimization plugins
pub trait OptimizerPlugin: Send + Sync {
    /// Plugin name (must be unique)
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Plugin description
    fn description(&self) -> &str;

    /// Initialize the plugin
    fn initialize(&mut self, config: &PluginConfig) -> Result<(), PluginError>;

    /// Suggest next hyperparameter configuration to evaluate
    fn suggest(
        &mut self,
        history: &OptimizationHistory,
        constraints: &ParameterConstraints,
    ) -> Result<HashMap<String, Float>, PluginError>;

    /// Update plugin state with new observation
    fn observe(
        &mut self,
        parameters: &HashMap<String, Float>,
        objective_value: Float,
        metadata: Option<&HashMap<String, String>>,
    ) -> Result<(), PluginError>;

    /// Check if optimization should stop
    fn should_stop(&self, history: &OptimizationHistory) -> bool;

    /// Get plugin-specific statistics
    fn get_statistics(&self) -> Result<HashMap<String, Float>, PluginError>;

    /// Clean up resources
    fn shutdown(&mut self) -> Result<(), PluginError>;

    /// Downcast to concrete type (for advanced usage)
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Configuration for plugins
#[derive(Debug, Clone)]
pub struct PluginConfig {
    pub max_iterations: usize,
    pub random_seed: Option<u64>,
    pub parallel: bool,
    pub custom_params: HashMap<String, String>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            random_seed: None,
            parallel: false,
            custom_params: HashMap::new(),
        }
    }
}

/// Optimization history
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    pub evaluations: Vec<Evaluation>,
    pub best_value: Float,
    pub best_parameters: HashMap<String, Float>,
    pub n_evaluations: usize,
}

impl OptimizationHistory {
    pub fn new() -> Self {
        Self {
            evaluations: Vec::new(),
            best_value: f64::NEG_INFINITY,
            best_parameters: HashMap::new(),
            n_evaluations: 0,
        }
    }

    pub fn add_evaluation(&mut self, params: HashMap<String, Float>, value: Float) {
        self.evaluations.push(Evaluation {
            parameters: params.clone(),
            objective_value: value,
            iteration: self.n_evaluations,
        });

        if value > self.best_value {
            self.best_value = value;
            self.best_parameters = params;
        }

        self.n_evaluations += 1;
    }
}

impl Default for OptimizationHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Single evaluation record
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub parameters: HashMap<String, Float>,
    pub objective_value: Float,
    pub iteration: usize,
}

/// Parameter constraints for optimization
#[derive(Debug, Clone, Default)]
pub struct ParameterConstraints {
    pub bounds: HashMap<String, (Float, Float)>,
    pub integer_params: Vec<String>,
    pub categorical_params: HashMap<String, Vec<String>>,
}

/// Plugin error type
#[derive(Debug, Clone)]
pub enum PluginError {
    InitializationFailed(String),
    SuggestionFailed(String),
    ObservationFailed(String),
    InvalidConfiguration(String),
    NotInitialized,
    AlreadyInitialized,
    InternalError(String),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PluginError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            PluginError::SuggestionFailed(msg) => write!(f, "Suggestion failed: {}", msg),
            PluginError::ObservationFailed(msg) => write!(f, "Observation failed: {}", msg),
            PluginError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            PluginError::NotInitialized => write!(f, "Plugin not initialized"),
            PluginError::AlreadyInitialized => write!(f, "Plugin already initialized"),
            PluginError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {}

// ============================================================================
// Plugin Registry
// ============================================================================

/// Global plugin registry
pub struct PluginRegistry {
    plugins: Arc<RwLock<HashMap<String, Box<dyn OptimizerPlugin>>>>,
    factories: Arc<RwLock<HashMap<String, PluginFactory>>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            factories: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a plugin factory
    pub fn register_factory(
        &self,
        name: String,
        factory: PluginFactory,
    ) -> Result<(), PluginError> {
        let mut factories = self
            .factories
            .write()
            .map_err(|e| PluginError::InternalError(format!("Failed to lock registry: {}", e)))?;

        if factories.contains_key(&name) {
            return Err(PluginError::AlreadyInitialized);
        }

        factories.insert(name, factory);
        Ok(())
    }

    /// Create and register a plugin instance
    pub fn create_plugin(&self, name: &str, config: &PluginConfig) -> Result<(), PluginError> {
        let factories = self
            .factories
            .read()
            .map_err(|e| PluginError::InternalError(format!("Failed to lock registry: {}", e)))?;

        let factory = factories.get(name).ok_or_else(|| {
            PluginError::InitializationFailed(format!("Plugin '{}' not found", name))
        })?;

        let mut plugin = (factory.create)()?;
        plugin.initialize(config)?;

        let mut plugins = self
            .plugins
            .write()
            .map_err(|e| PluginError::InternalError(format!("Failed to lock plugins: {}", e)))?;

        plugins.insert(name.to_string(), plugin);
        Ok(())
    }

    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Result<Box<dyn OptimizerPlugin>, PluginError> {
        let plugins = self
            .plugins
            .read()
            .map_err(|e| PluginError::InternalError(format!("Failed to lock plugins: {}", e)))?;

        plugins
            .get(name)
            .ok_or_else(|| {
                PluginError::InitializationFailed(format!("Plugin '{}' not found", name))
            })
            .map(|_plugin| {
                // This is a placeholder - in reality, we'd need to clone or use Arc
                // For now, return an error
                Err(PluginError::InternalError(
                    "Cannot borrow plugin".to_string(),
                ))
            })?
    }

    /// List all registered plugin names
    pub fn list_plugins(&self) -> Result<Vec<String>, PluginError> {
        let factories = self
            .factories
            .read()
            .map_err(|e| PluginError::InternalError(format!("Failed to lock registry: {}", e)))?;

        Ok(factories.keys().cloned().collect())
    }

    /// Unregister a plugin
    pub fn unregister_plugin(&self, name: &str) -> Result<(), PluginError> {
        let mut plugins = self
            .plugins
            .write()
            .map_err(|e| PluginError::InternalError(format!("Failed to lock plugins: {}", e)))?;

        if let Some(mut plugin) = plugins.remove(name) {
            plugin.shutdown()?;
        }

        Ok(())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin factory for creating plugin instances
pub struct PluginFactory {
    pub name: String,
    pub version: String,
    pub description: String,
    pub create: fn() -> Result<Box<dyn OptimizerPlugin>, PluginError>,
}

// ============================================================================
// Hook System
// ============================================================================

/// Optimization hook trait for callbacks
pub trait OptimizationHook: Send + Sync {
    /// Called before optimization starts
    fn on_optimization_start(
        &mut self,
        config: &PluginConfig,
        constraints: &ParameterConstraints,
    ) -> Result<(), HookError>;

    /// Called before each iteration
    fn on_iteration_start(
        &mut self,
        iteration: usize,
        history: &OptimizationHistory,
    ) -> Result<(), HookError>;

    /// Called after each evaluation
    fn on_evaluation(
        &mut self,
        parameters: &HashMap<String, Float>,
        objective_value: Float,
        iteration: usize,
    ) -> Result<(), HookError>;

    /// Called after each iteration
    fn on_iteration_end(
        &mut self,
        iteration: usize,
        history: &OptimizationHistory,
    ) -> Result<(), HookError>;

    /// Called when optimization completes
    fn on_optimization_end(
        &mut self,
        history: &OptimizationHistory,
        reason: StopReason,
    ) -> Result<(), HookError>;

    /// Called on optimization error
    fn on_error(&mut self, error: &dyn std::error::Error) -> Result<(), HookError>;
}

/// Reason for stopping optimization
#[derive(Debug, Clone)]
pub enum StopReason {
    MaxIterationsReached,
    ConvergenceReached,
    TimeoutReached,
    UserInterrupted,
    PluginDecision,
    Error(String),
}

/// Hook error type
#[derive(Debug, Clone)]
pub enum HookError {
    ExecutionFailed(String),
    InvalidState(String),
}

impl std::fmt::Display for HookError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HookError::ExecutionFailed(msg) => write!(f, "Hook execution failed: {}", msg),
            HookError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for HookError {}

/// Hook manager for managing multiple hooks
pub struct HookManager {
    hooks: Vec<Box<dyn OptimizationHook>>,
}

impl HookManager {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn add_hook(&mut self, hook: Box<dyn OptimizationHook>) {
        self.hooks.push(hook);
    }

    pub fn trigger_optimization_start(
        &mut self,
        config: &PluginConfig,
        constraints: &ParameterConstraints,
    ) -> Result<(), HookError> {
        for hook in &mut self.hooks {
            hook.on_optimization_start(config, constraints)?;
        }
        Ok(())
    }

    pub fn trigger_iteration_start(
        &mut self,
        iteration: usize,
        history: &OptimizationHistory,
    ) -> Result<(), HookError> {
        for hook in &mut self.hooks {
            hook.on_iteration_start(iteration, history)?;
        }
        Ok(())
    }

    pub fn trigger_evaluation(
        &mut self,
        parameters: &HashMap<String, Float>,
        objective_value: Float,
        iteration: usize,
    ) -> Result<(), HookError> {
        for hook in &mut self.hooks {
            hook.on_evaluation(parameters, objective_value, iteration)?;
        }
        Ok(())
    }

    pub fn trigger_iteration_end(
        &mut self,
        iteration: usize,
        history: &OptimizationHistory,
    ) -> Result<(), HookError> {
        for hook in &mut self.hooks {
            hook.on_iteration_end(iteration, history)?;
        }
        Ok(())
    }

    pub fn trigger_optimization_end(
        &mut self,
        history: &OptimizationHistory,
        reason: StopReason,
    ) -> Result<(), HookError> {
        for hook in &mut self.hooks {
            hook.on_optimization_end(history, reason.clone())?;
        }
        Ok(())
    }

    pub fn trigger_error(&mut self, error: &dyn std::error::Error) -> Result<(), HookError> {
        for hook in &mut self.hooks {
            hook.on_error(error)?;
        }
        Ok(())
    }
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Middleware Support
// ============================================================================

/// Middleware for optimization pipelines
pub trait OptimizationMiddleware: Send + Sync {
    /// Process suggestion before returning to optimizer
    fn process_suggestion(
        &self,
        parameters: &mut HashMap<String, Float>,
        history: &OptimizationHistory,
    ) -> Result<(), MiddlewareError>;

    /// Process observation before storing
    fn process_observation(
        &self,
        parameters: &HashMap<String, Float>,
        objective_value: &mut Float,
        history: &OptimizationHistory,
    ) -> Result<(), MiddlewareError>;

    /// Middleware name
    fn name(&self) -> &str;
}

/// Middleware error type
#[derive(Debug, Clone)]
pub enum MiddlewareError {
    ProcessingFailed(String),
    ValidationFailed(String),
}

impl std::fmt::Display for MiddlewareError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MiddlewareError::ProcessingFailed(msg) => write!(f, "Processing failed: {}", msg),
            MiddlewareError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
        }
    }
}

impl std::error::Error for MiddlewareError {}

/// Middleware pipeline
pub struct MiddlewarePipeline {
    middleware: Vec<Box<dyn OptimizationMiddleware>>,
}

impl MiddlewarePipeline {
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
        }
    }

    pub fn add(&mut self, middleware: Box<dyn OptimizationMiddleware>) {
        self.middleware.push(middleware);
    }

    pub fn process_suggestion(
        &self,
        parameters: &mut HashMap<String, Float>,
        history: &OptimizationHistory,
    ) -> Result<(), MiddlewareError> {
        for m in &self.middleware {
            m.process_suggestion(parameters, history)?;
        }
        Ok(())
    }

    pub fn process_observation(
        &self,
        parameters: &HashMap<String, Float>,
        objective_value: &mut Float,
        history: &OptimizationHistory,
    ) -> Result<(), MiddlewareError> {
        for m in &self.middleware {
            m.process_observation(parameters, objective_value, history)?;
        }
        Ok(())
    }
}

impl Default for MiddlewarePipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Custom Metric Registration
// ============================================================================

/// Custom metric trait
pub trait CustomMetric: Send + Sync {
    /// Metric name
    fn name(&self) -> &str;

    /// Compute metric value
    fn compute(
        &self,
        parameters: &HashMap<String, Float>,
        objective_value: Float,
        history: &OptimizationHistory,
    ) -> Result<Float, MetricError>;

    /// Whether higher is better
    fn higher_is_better(&self) -> bool;
}

/// Metric error type
#[derive(Debug, Clone)]
pub enum MetricError {
    ComputationFailed(String),
    InvalidInput(String),
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MetricError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            MetricError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for MetricError {}

/// Metric registry
pub struct MetricRegistry {
    metrics: HashMap<String, Box<dyn CustomMetric>>,
}

impl MetricRegistry {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn register(&mut self, metric: Box<dyn CustomMetric>) -> Result<(), MetricError> {
        let name = metric.name().to_string();
        if self.metrics.contains_key(&name) {
            return Err(MetricError::InvalidInput(format!(
                "Metric '{}' already registered",
                name
            )));
        }
        self.metrics.insert(name, metric);
        Ok(())
    }

    pub fn compute(
        &self,
        metric_name: &str,
        parameters: &HashMap<String, Float>,
        objective_value: Float,
        history: &OptimizationHistory,
    ) -> Result<Float, MetricError> {
        let metric = self.metrics.get(metric_name).ok_or_else(|| {
            MetricError::InvalidInput(format!("Metric '{}' not found", metric_name))
        })?;

        metric.compute(parameters, objective_value, history)
    }

    pub fn list_metrics(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Example Implementations
// ============================================================================

/// Simple logging hook
pub struct LoggingHook {
    log_interval: usize,
}

impl LoggingHook {
    pub fn new(log_interval: usize) -> Self {
        Self { log_interval }
    }
}

impl OptimizationHook for LoggingHook {
    fn on_optimization_start(
        &mut self,
        _config: &PluginConfig,
        _constraints: &ParameterConstraints,
    ) -> Result<(), HookError> {
        println!("Optimization started");
        Ok(())
    }

    fn on_iteration_start(
        &mut self,
        iteration: usize,
        _history: &OptimizationHistory,
    ) -> Result<(), HookError> {
        if iteration % self.log_interval == 0 {
            println!("Starting iteration {}", iteration);
        }
        Ok(())
    }

    fn on_evaluation(
        &mut self,
        _parameters: &HashMap<String, Float>,
        _objective_value: Float,
        _iteration: usize,
    ) -> Result<(), HookError> {
        Ok(())
    }

    fn on_iteration_end(
        &mut self,
        iteration: usize,
        history: &OptimizationHistory,
    ) -> Result<(), HookError> {
        if iteration % self.log_interval == 0 {
            println!(
                "Iteration {} complete. Best so far: {}",
                iteration, history.best_value
            );
        }
        Ok(())
    }

    fn on_optimization_end(
        &mut self,
        history: &OptimizationHistory,
        reason: StopReason,
    ) -> Result<(), HookError> {
        println!("Optimization ended. Reason: {:?}", reason);
        println!("Best value: {}", history.best_value);
        println!("Total evaluations: {}", history.n_evaluations);
        Ok(())
    }

    fn on_error(&mut self, error: &dyn std::error::Error) -> Result<(), HookError> {
        eprintln!("Error during optimization: {}", error);
        Ok(())
    }
}

/// Parameter normalization middleware
pub struct NormalizationMiddleware {
    name: String,
}

impl NormalizationMiddleware {
    pub fn new() -> Self {
        Self {
            name: "normalization".to_string(),
        }
    }
}

impl Default for NormalizationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationMiddleware for NormalizationMiddleware {
    fn process_suggestion(
        &self,
        parameters: &mut HashMap<String, Float>,
        _history: &OptimizationHistory,
    ) -> Result<(), MiddlewareError> {
        // Example: ensure all parameters are within valid ranges
        for (_, value) in parameters.iter_mut() {
            if value.is_nan() || value.is_infinite() {
                *value = 0.0; // Default value for invalid parameters
            }
        }
        Ok(())
    }

    fn process_observation(
        &self,
        _parameters: &HashMap<String, Float>,
        objective_value: &mut Float,
        _history: &OptimizationHistory,
    ) -> Result<(), MiddlewareError> {
        // Example: handle invalid objective values
        if objective_value.is_nan() || objective_value.is_infinite() {
            *objective_value = f64::NEG_INFINITY;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_config() {
        let config = PluginConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!(!config.parallel);
    }

    #[test]
    fn test_optimization_history() {
        let mut history = OptimizationHistory::new();
        assert_eq!(history.n_evaluations, 0);

        let mut params = HashMap::new();
        params.insert("lr".to_string(), 0.01);
        history.add_evaluation(params, 0.95);

        assert_eq!(history.n_evaluations, 1);
        assert_eq!(history.best_value, 0.95);
    }

    #[test]
    fn test_plugin_registry() {
        let registry = PluginRegistry::new();
        assert!(registry.list_plugins().is_ok());
    }

    #[test]
    fn test_hook_manager() {
        let mut manager = HookManager::new();
        let hook = Box::new(LoggingHook::new(10));
        manager.add_hook(hook);

        let config = PluginConfig::default();
        let constraints = ParameterConstraints::default();
        assert!(manager
            .trigger_optimization_start(&config, &constraints)
            .is_ok());
    }

    #[test]
    fn test_middleware_pipeline() {
        let mut pipeline = MiddlewarePipeline::new();
        pipeline.add(Box::new(NormalizationMiddleware::new()));

        let mut params = HashMap::new();
        params.insert("x".to_string(), f64::NAN);

        let history = OptimizationHistory::new();
        assert!(pipeline.process_suggestion(&mut params, &history).is_ok());
        assert_eq!(params.get("x"), Some(&0.0));
    }

    #[test]
    fn test_metric_registry() {
        let registry = MetricRegistry::new();
        assert!(registry.list_metrics().is_empty());
    }
}
