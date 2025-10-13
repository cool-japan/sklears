//! Modular Plugin Architecture for Feature Selection
//!
//! This module provides a flexible plugin system that allows users to register
//! custom feature selection methods, metrics, and transformations. The architecture
//! uses trait objects, dynamic dispatch, and reflection for maximum extensibility.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Transform};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

type Result<T> = SklResult<T>;

/// Core trait for feature selection plugins
pub trait FeatureSelectionPlugin: Send + Sync {
    /// Get the plugin name
    fn name(&self) -> &str;

    /// Get the plugin version
    fn version(&self) -> &str;

    /// Get the plugin description
    fn description(&self) -> &str;

    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Fit the selector on training data
    fn fit(&mut self, X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()>;

    /// Transform data by selecting features
    fn transform(&self, X: ArrayView2<f64>) -> Result<Array2<f64>>;

    /// Get selected feature indices
    fn selected_features(&self) -> Result<Vec<usize>>;

    /// Get feature scores/importances
    fn feature_scores(&self) -> Result<Array1<f64>>;

    /// Check if the plugin is fitted
    fn is_fitted(&self) -> bool;

    /// Get plugin configuration as Any trait object
    fn as_any(&self) -> &dyn Any;

    /// Clone the plugin
    fn clone_plugin(&self) -> Box<dyn FeatureSelectionPlugin>;
}

/// Trait for custom scoring functions
pub trait ScoringFunction: Send + Sync {
    /// Get the scoring function name
    fn name(&self) -> &str;

    /// Compute score for a single feature
    fn score(&self, feature: ArrayView1<f64>, target: ArrayView1<f64>) -> Result<f64>;

    /// Compute scores for all features in parallel
    fn score_features(&self, X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>> {
        let mut scores = Array1::zeros(X.ncols());
        for (i, score) in scores.iter_mut().enumerate() {
            *score = self.score(X.column(i), y)?;
        }
        Ok(scores)
    }

    /// Clone the scoring function
    fn clone_scoring(&self) -> Box<dyn ScoringFunction>;
}

/// Trait for custom transformation functions
pub trait TransformationFunction: Send + Sync {
    /// Get the transformation function name
    fn name(&self) -> &str;

    /// Apply transformation to features
    fn transform(&self, X: ArrayView2<f64>) -> Result<Array2<f64>>;

    /// Get output feature count (if deterministic)
    fn output_features(&self, input_features: usize) -> Option<usize>;

    /// Clone the transformation function
    fn clone_transform(&self) -> Box<dyn TransformationFunction>;
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub author: String,
    pub license: String,
    pub categories: Vec<String>,
    pub tags: Vec<String>,
    pub min_samples: Option<usize>,
    pub max_features: Option<usize>,
    pub supports_sparse: bool,
    pub supports_multiclass: bool,
    pub supports_regression: bool,
    pub computational_complexity: ComputationalComplexity,
    pub memory_complexity: MemoryComplexity,
}

impl Default for PluginMetadata {
    fn default() -> Self {
        Self {
            author: String::new(),
            license: "MIT".to_string(),
            categories: Vec::new(),
            tags: Vec::new(),
            min_samples: None,
            max_features: None,
            supports_sparse: false,
            supports_multiclass: true,
            supports_regression: true,
            computational_complexity: ComputationalComplexity::default(),
            memory_complexity: MemoryComplexity::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ComputationalComplexity {
    /// Constant
    Constant,
    /// Linear
    Linear,
    /// Quadratic
    Quadratic,
    /// Cubic
    Cubic,
    /// Exponential
    Exponential,
    /// Custom
    Custom(String),
}

impl Default for ComputationalComplexity {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone)]
pub enum MemoryComplexity {
    /// Constant
    Constant,
    /// Linear
    Linear,
    /// Quadratic
    Quadratic,
    /// Custom
    Custom(String),
}

impl Default for MemoryComplexity {
    fn default() -> Self {
        Self::Linear
    }
}

/// Plugin registry for managing feature selection plugins
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, Box<dyn FeatureSelectionPlugin>>>,
    scoring_functions: RwLock<HashMap<String, Box<dyn ScoringFunction>>>,
    transformations: RwLock<HashMap<String, Box<dyn TransformationFunction>>>,
    middleware: RwLock<Vec<Box<dyn PluginMiddleware>>>,
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
            plugins: RwLock::new(HashMap::new()),
            scoring_functions: RwLock::new(HashMap::new()),
            transformations: RwLock::new(HashMap::new()),
            middleware: RwLock::new(Vec::new()),
        }
    }

    /// Register a feature selection plugin
    pub fn register_plugin(&self, plugin: Box<dyn FeatureSelectionPlugin>) -> Result<()> {
        let name = plugin.name().to_string();
        let mut plugins = self
            .plugins
            .write()
            .map_err(|_| SklearsError::FitError("Failed to acquire write lock".to_string()))?;

        if plugins.contains_key(&name) {
            return Err(SklearsError::InvalidInput(format!(
                "Plugin '{}' is already registered",
                name
            )));
        }

        plugins.insert(name, plugin);
        Ok(())
    }

    /// Register a custom scoring function
    pub fn register_scoring_function(&self, function: Box<dyn ScoringFunction>) -> Result<()> {
        let name = function.name().to_string();
        let mut functions = self
            .scoring_functions
            .write()
            .map_err(|_| SklearsError::FitError("Failed to acquire write lock".to_string()))?;

        functions.insert(name, function);
        Ok(())
    }

    /// Register a custom transformation function
    pub fn register_transformation(
        &self,
        transformation: Box<dyn TransformationFunction>,
    ) -> Result<()> {
        let name = transformation.name().to_string();
        let mut transformations = self
            .transformations
            .write()
            .map_err(|_| SklearsError::FitError("Failed to acquire write lock".to_string()))?;

        transformations.insert(name, transformation);
        Ok(())
    }

    /// Register middleware
    pub fn register_middleware(&self, middleware: Box<dyn PluginMiddleware>) -> Result<()> {
        let mut middleware_vec = self
            .middleware
            .write()
            .map_err(|_| SklearsError::FitError("Failed to acquire write lock".to_string()))?;

        middleware_vec.push(middleware);
        Ok(())
    }

    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Result<Box<dyn FeatureSelectionPlugin>> {
        let plugins = self
            .plugins
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;

        plugins
            .get(name)
            .map(|plugin| plugin.clone_plugin())
            .ok_or_else(|| SklearsError::InvalidInput(format!("Plugin '{}' not found", name)))
    }

    /// Get a scoring function by name
    pub fn get_scoring_function(&self, name: &str) -> Result<Box<dyn ScoringFunction>> {
        let functions = self
            .scoring_functions
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;

        functions
            .get(name)
            .map(|func| func.clone_scoring())
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!("Scoring function '{}' not found", name))
            })
    }

    /// Get a transformation by name
    pub fn get_transformation(&self, name: &str) -> Result<Box<dyn TransformationFunction>> {
        let transformations = self
            .transformations
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;

        transformations
            .get(name)
            .map(|transform| transform.clone_transform())
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!("Transformation '{}' not found", name))
            })
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Result<Vec<String>> {
        let plugins = self
            .plugins
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;

        Ok(plugins.keys().cloned().collect())
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, name: &str) -> Result<PluginMetadata> {
        let plugins = self
            .plugins
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;

        plugins
            .get(name)
            .map(|plugin| plugin.metadata())
            .ok_or_else(|| SklearsError::InvalidInput(format!("Plugin '{}' not found", name)))
    }

    /// Execute middleware before plugin operations
    pub fn execute_before_middleware(
        &self,
        plugin_name: &str,
        context: &PluginContext,
    ) -> Result<()> {
        let middleware = self
            .middleware
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;

        for mw in middleware.iter() {
            mw.before_execution(plugin_name, context)?;
        }

        Ok(())
    }

    /// Execute middleware after plugin operations
    pub fn execute_after_middleware(
        &self,
        plugin_name: &str,
        context: &PluginContext,
        result: &PluginResult,
    ) -> Result<()> {
        let middleware = self
            .middleware
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;

        for mw in middleware.iter() {
            mw.after_execution(plugin_name, context, result)?;
        }

        Ok(())
    }
}

/// Global plugin registry instance (commented out due to missing lazy_static dependency)
// lazy_static::lazy_static! {
//     pub static ref GLOBAL_REGISTRY: PluginRegistry = PluginRegistry::new();
// }

/// Plugin middleware trait for cross-cutting concerns
pub trait PluginMiddleware: Send + Sync {
    /// Execute before plugin operation
    fn before_execution(&self, plugin_name: &str, context: &PluginContext) -> Result<()>;

    /// Execute after plugin operation
    fn after_execution(
        &self,
        plugin_name: &str,
        context: &PluginContext,
        result: &PluginResult,
    ) -> Result<()>;
}

/// Context passed to middleware
#[derive(Debug, Clone)]
pub struct PluginContext {
    pub operation: String,
    pub data_shape: (usize, usize),
    pub parameters: HashMap<String, String>,
    pub start_time: std::time::Instant,
}

/// Result passed to middleware
#[derive(Debug, Clone)]
pub struct PluginResult {
    pub success: bool,
    pub execution_time: std::time::Duration,
    pub selected_features: Vec<usize>,
    pub error_message: Option<String>,
}

/// Composable plugin pipeline
pub struct PluginPipeline {
    steps: Vec<PipelineStep>,
    registry: Arc<PluginRegistry>,
}

#[derive(Clone)]
pub enum PipelineStep {
    /// Plugin
    Plugin {
        name: String,

        config: HashMap<String, String>,
    },
    /// Transformation
    Transformation {
        name: String,

        config: HashMap<String, String>,
    },
    /// Scoring
    Scoring {
        name: String,
        config: HashMap<String, String>,
    },
}

impl Default for PluginPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginPipeline {
    /// Create a new plugin pipeline
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            registry: Arc::new(PluginRegistry::new()),
        }
    }

    /// Create a pipeline with a custom registry
    pub fn with_registry(registry: Arc<PluginRegistry>) -> Self {
        Self {
            steps: Vec::new(),
            registry,
        }
    }

    /// Add a plugin step
    pub fn add_plugin(mut self, name: String, config: HashMap<String, String>) -> Self {
        self.steps.push(PipelineStep::Plugin { name, config });
        self
    }

    /// Add a transformation step
    pub fn add_transformation(mut self, name: String, config: HashMap<String, String>) -> Self {
        self.steps
            .push(PipelineStep::Transformation { name, config });
        self
    }

    /// Add a scoring step
    pub fn add_scoring(mut self, name: String, config: HashMap<String, String>) -> Self {
        self.steps.push(PipelineStep::Scoring { name, config });
        self
    }

    /// Execute the pipeline
    pub fn execute(&self, X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<PipelineResult> {
        let start_time = std::time::Instant::now();
        let mut current_X = X.to_owned();
        let mut step_results = Vec::new();

        for (step_index, step) in self.steps.iter().enumerate() {
            let step_start = std::time::Instant::now();

            match step {
                PipelineStep::Plugin { name, config } => {
                    let context = PluginContext {
                        operation: "plugin_execution".to_string(),
                        data_shape: (current_X.nrows(), current_X.ncols()),
                        parameters: config.clone(),
                        start_time: step_start,
                    };

                    self.registry.execute_before_middleware(name, &context)?;

                    let mut plugin = self.registry.get_plugin(name)?;
                    plugin.fit(current_X.view(), y.view())?;
                    current_X = plugin.transform(current_X.view())?;
                    let selected_features = plugin.selected_features()?;

                    let result = PluginResult {
                        success: true,
                        execution_time: step_start.elapsed(),
                        selected_features: selected_features.clone(),
                        error_message: None,
                    };

                    self.registry
                        .execute_after_middleware(name, &context, &result)?;

                    step_results.push(StepResult {
                        step_index,
                        step_type: "Plugin".to_string(),
                        step_name: name.clone(),
                        execution_time: step_start.elapsed(),
                        input_features: context.data_shape.1,
                        output_features: current_X.ncols(),
                        selected_features,
                    });
                }
                PipelineStep::Transformation { name, config } => {
                    let transformation = self.registry.get_transformation(name)?;
                    let input_features = current_X.ncols();
                    current_X = transformation.transform(current_X.view())?;

                    step_results.push(StepResult {
                        step_index,
                        step_type: "Transformation".to_string(),
                        step_name: name.clone(),
                        execution_time: step_start.elapsed(),
                        input_features,
                        output_features: current_X.ncols(),
                        selected_features: (0..current_X.ncols()).collect(),
                    });
                }
                PipelineStep::Scoring { name, config: _ } => {
                    let scoring_function = self.registry.get_scoring_function(name)?;
                    let _scores = scoring_function.score_features(current_X.view(), y.view())?;

                    step_results.push(StepResult {
                        step_index,
                        step_type: "Scoring".to_string(),
                        step_name: name.clone(),
                        execution_time: step_start.elapsed(),
                        input_features: current_X.ncols(),
                        output_features: current_X.ncols(),
                        selected_features: (0..current_X.ncols()).collect(),
                    });
                }
            }
        }

        Ok(PipelineResult {
            final_data: current_X.clone(),
            step_results,
            total_execution_time: start_time.elapsed(),
            original_features: X.ncols(),
            final_features: current_X.ncols(),
        })
    }
}

/// Result of pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub final_data: Array2<f64>,
    pub step_results: Vec<StepResult>,
    pub total_execution_time: std::time::Duration,
    pub original_features: usize,
    pub final_features: usize,
}

/// Result of individual pipeline step
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_index: usize,
    pub step_type: String,
    pub step_name: String,
    pub execution_time: std::time::Duration,
    pub input_features: usize,
    pub output_features: usize,
    pub selected_features: Vec<usize>,
}

/// Built-in plugins
pub mod builtin {
    use super::*;

    /// Variance threshold plugin
    #[derive(Debug, Clone)]
    pub struct VarianceThresholdPlugin {
        threshold: f64,
        feature_variances: Option<Array1<f64>>,
        selected_indices: Option<Vec<usize>>,
        fitted: bool,
    }

    impl VarianceThresholdPlugin {
        pub fn new(threshold: f64) -> Self {
            Self {
                threshold,
                feature_variances: None,
                selected_indices: None,
                fitted: false,
            }
        }
    }

    impl FeatureSelectionPlugin for VarianceThresholdPlugin {
        fn name(&self) -> &str {
            "variance_threshold"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn description(&self) -> &str {
            "Removes features with variance below threshold"
        }

        fn metadata(&self) -> PluginMetadata {
            PluginMetadata {
                author: "Sklears Team".to_string(),
                license: "MIT".to_string(),
                categories: vec!["filter".to_string(), "univariate".to_string()],
                tags: vec!["variance".to_string(), "threshold".to_string()],
                min_samples: None,
                max_features: None,
                supports_sparse: true,
                supports_multiclass: true,
                supports_regression: true,
                computational_complexity: ComputationalComplexity::Linear,
                memory_complexity: MemoryComplexity::Linear,
            }
        }

        fn fit(&mut self, X: ArrayView2<f64>, _y: ArrayView1<f64>) -> Result<()> {
            let mut variances = Array1::zeros(X.ncols());
            for (i, var) in variances.iter_mut().enumerate() {
                *var = X.column(i).var(1.0);
            }

            let selected_indices: Vec<usize> = variances
                .iter()
                .enumerate()
                .filter_map(|(i, &var)| if var > self.threshold { Some(i) } else { None })
                .collect();

            self.feature_variances = Some(variances);
            self.selected_indices = Some(selected_indices);
            self.fitted = true;

            Ok(())
        }

        fn transform(&self, X: ArrayView2<f64>) -> Result<Array2<f64>> {
            if !self.fitted {
                return Err(SklearsError::FitError("Plugin not fitted".to_string()));
            }

            let selected_indices = self.selected_indices.as_ref().unwrap();
            if selected_indices.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "No features selected".to_string(),
                ));
            }

            let mut result = Array2::zeros((X.nrows(), selected_indices.len()));
            for (new_col, &old_col) in selected_indices.iter().enumerate() {
                for row in 0..X.nrows() {
                    result[[row, new_col]] = X[[row, old_col]];
                }
            }

            Ok(result)
        }

        fn selected_features(&self) -> Result<Vec<usize>> {
            self.selected_indices
                .clone()
                .ok_or_else(|| SklearsError::FitError("Plugin not fitted".to_string()))
        }

        fn feature_scores(&self) -> Result<Array1<f64>> {
            self.feature_variances
                .clone()
                .ok_or_else(|| SklearsError::FitError("Plugin not fitted".to_string()))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_plugin(&self) -> Box<dyn FeatureSelectionPlugin> {
            Box::new(self.clone())
        }
    }

    /// Correlation-based scoring function
    #[derive(Debug, Clone)]
    pub struct CorrelationScoring;

    impl ScoringFunction for CorrelationScoring {
        fn name(&self) -> &str {
            "correlation"
        }

        fn score(&self, feature: ArrayView1<f64>, target: ArrayView1<f64>) -> Result<f64> {
            if feature.len() != target.len() {
                return Err(SklearsError::InvalidInput(
                    "Feature and target length mismatch".to_string(),
                ));
            }

            let correlation = crate::performance::SIMDStats::correlation_auto(feature, target);
            Ok(correlation.abs())
        }

        fn clone_scoring(&self) -> Box<dyn ScoringFunction> {
            Box::new(self.clone())
        }
    }

    /// Normalization transformation
    #[derive(Debug, Clone)]
    pub struct NormalizationTransform;

    impl TransformationFunction for NormalizationTransform {
        fn name(&self) -> &str {
            "normalization"
        }

        fn transform(&self, X: ArrayView2<f64>) -> Result<Array2<f64>> {
            let mut result = X.to_owned();

            for col in 0..result.ncols() {
                let column = result.column(col);
                let mean = column.mean().unwrap_or(0.0);
                let std = column.var(1.0).sqrt();

                if std > 1e-10 {
                    for row in 0..result.nrows() {
                        result[[row, col]] = (result[[row, col]] - mean) / std;
                    }
                }
            }

            Ok(result)
        }

        fn output_features(&self, input_features: usize) -> Option<usize> {
            Some(input_features)
        }

        fn clone_transform(&self) -> Box<dyn TransformationFunction> {
            Box::new(self.clone())
        }
    }
}

/// Logging middleware
#[derive(Debug, Clone)]
pub struct LoggingMiddleware {
    log_level: LogLevel,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    /// Debug
    Debug,
    /// Info
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
}

impl LoggingMiddleware {
    pub fn new(log_level: LogLevel) -> Self {
        Self { log_level }
    }
}

impl PluginMiddleware for LoggingMiddleware {
    fn before_execution(&self, plugin_name: &str, context: &PluginContext) -> Result<()> {
        match self.log_level {
            LogLevel::Debug | LogLevel::Info => {
                println!(
                    "Executing plugin '{}' with operation '{}'",
                    plugin_name, context.operation
                );
                println!("  Data shape: {:?}", context.data_shape);
            }
            _ => {}
        }
        Ok(())
    }

    fn after_execution(
        &self,
        plugin_name: &str,
        _context: &PluginContext,
        result: &PluginResult,
    ) -> Result<()> {
        match self.log_level {
            LogLevel::Debug | LogLevel::Info => {
                println!(
                    "Plugin '{}' completed in {:?}",
                    plugin_name, result.execution_time
                );
                println!("  Selected {} features", result.selected_features.len());
                if let Some(ref error) = result.error_message {
                    println!("  Error: {}", error);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Performance monitoring middleware
#[derive(Debug)]
pub struct PerformanceMiddleware {
    metrics: Arc<RwLock<HashMap<String, PerformanceMetrics>>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_executions: usize,
    pub total_time: std::time::Duration,
    pub average_time: std::time::Duration,
    pub min_time: std::time::Duration,
    pub max_time: std::time::Duration,
}

impl Default for PerformanceMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMiddleware {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn get_metrics(&self) -> Result<HashMap<String, PerformanceMetrics>> {
        let metrics = self
            .metrics
            .read()
            .map_err(|_| SklearsError::FitError("Failed to acquire read lock".to_string()))?;
        Ok(metrics.clone())
    }
}

impl PluginMiddleware for PerformanceMiddleware {
    fn before_execution(&self, _plugin_name: &str, _context: &PluginContext) -> Result<()> {
        // Nothing to do before execution
        Ok(())
    }

    fn after_execution(
        &self,
        plugin_name: &str,
        _context: &PluginContext,
        result: &PluginResult,
    ) -> Result<()> {
        let mut metrics = self
            .metrics
            .write()
            .map_err(|_| SklearsError::FitError("Failed to acquire write lock".to_string()))?;

        let entry = metrics
            .entry(plugin_name.to_string())
            .or_insert_with(|| PerformanceMetrics {
                total_executions: 0,
                total_time: std::time::Duration::from_secs(0),
                average_time: std::time::Duration::from_secs(0),
                min_time: std::time::Duration::from_secs(u64::MAX),
                max_time: std::time::Duration::from_secs(0),
            });

        entry.total_executions += 1;
        entry.total_time += result.execution_time;
        entry.average_time = entry.total_time / entry.total_executions as u32;
        entry.min_time = entry.min_time.min(result.execution_time);
        entry.max_time = entry.max_time.max(result.execution_time);

        Ok(())
    }
}

/// Helper macro for easy plugin registration
#[macro_export]
macro_rules! register_plugin {
    ($registry:expr, $plugin:expr) => {
        $registry.register_plugin(Box::new($plugin))?;
    };
}

/// Helper macro for creating plugin pipelines
#[macro_export]
macro_rules! plugin_pipeline {
    ($($step_type:ident($name:expr, $config:expr)),+ $(,)?) => {
        {
            let mut pipeline = PluginPipeline::new();
            $(
                pipeline = match stringify!($step_type) {
                    "plugin" => pipeline.add_plugin($name.to_string(), $config),
                    "transform" => pipeline.add_transformation($name.to_string(), $config),
                    "scoring" => pipeline.add_scoring($name.to_string(), $config),
                    _ => panic!("Unknown step type: {}", stringify!($step_type)),
                };
            )+
            pipeline
        }
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::builtin::*;
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_plugin_registry() -> Result<()> {
        let registry = PluginRegistry::new();

        // Register a plugin
        let plugin = VarianceThresholdPlugin::new(0.1);
        registry.register_plugin(Box::new(plugin))?;

        // Register a scoring function
        let scoring = CorrelationScoring;
        registry.register_scoring_function(Box::new(scoring))?;

        // Register a transformation
        let transform = NormalizationTransform;
        registry.register_transformation(Box::new(transform))?;

        // Test retrieval
        let retrieved_plugin = registry.get_plugin("variance_threshold")?;
        assert_eq!(retrieved_plugin.name(), "variance_threshold");

        let retrieved_scoring = registry.get_scoring_function("correlation")?;
        assert_eq!(retrieved_scoring.name(), "correlation");

        let retrieved_transform = registry.get_transformation("normalization")?;
        assert_eq!(retrieved_transform.name(), "normalization");

        Ok(())
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_plugin_execution() -> Result<()> {
        let X = array![
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 0.0],
            [3.0, 6.0, 0.0],
            [4.0, 8.0, 0.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let mut plugin = VarianceThresholdPlugin::new(0.1);
        plugin.fit(X.view(), y.view())?;

        let selected_features = plugin.selected_features()?;
        assert!(selected_features.len() <= 3);

        let transformed = plugin.transform(X.view())?;
        assert_eq!(transformed.ncols(), selected_features.len());

        Ok(())
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_plugin_pipeline() -> Result<()> {
        let registry = Arc::new(PluginRegistry::new());

        // Register plugins
        registry.register_plugin(Box::new(VarianceThresholdPlugin::new(0.1)))?;
        registry.register_transformation(Box::new(NormalizationTransform))?;
        registry.register_scoring_function(Box::new(CorrelationScoring))?;

        let pipeline = PluginPipeline::with_registry(registry)
            .add_transformation("normalization".to_string(), HashMap::new())
            .add_plugin("variance_threshold".to_string(), HashMap::new());

        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let result = pipeline.execute(X.view(), y.view())?;
        assert!(result.final_features <= 3);
        assert_eq!(result.step_results.len(), 2);

        Ok(())
    }

    #[test]
    fn test_middleware() -> Result<()> {
        let registry = PluginRegistry::new();

        // Register middleware
        let logging_middleware = LoggingMiddleware::new(LogLevel::Info);
        registry.register_middleware(Box::new(logging_middleware))?;

        let performance_middleware = PerformanceMiddleware::new();
        registry.register_middleware(Box::new(performance_middleware))?;

        // Register and execute plugin
        registry.register_plugin(Box::new(VarianceThresholdPlugin::new(0.1)))?;

        let context = PluginContext {
            operation: "test".to_string(),
            data_shape: (100, 10),
            parameters: HashMap::new(),
            start_time: std::time::Instant::now(),
        };

        registry.execute_before_middleware("variance_threshold", &context)?;

        let result = PluginResult {
            success: true,
            execution_time: std::time::Duration::from_millis(10),
            selected_features: vec![0, 1, 2],
            error_message: None,
        };

        registry.execute_after_middleware("variance_threshold", &context, &result)?;

        Ok(())
    }

    #[test]
    fn test_macro_pipeline() -> Result<()> {
        let pipeline = plugin_pipeline! {
            transform("normalization", HashMap::new()),
            plugin("variance_threshold", HashMap::new()),
            scoring("correlation", HashMap::new()),
        };

        // Pipeline should have 3 steps
        assert_eq!(pipeline.steps.len(), 3);

        Ok(())
    }
}
