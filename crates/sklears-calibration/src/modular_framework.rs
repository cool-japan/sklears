//! Enhanced modular framework for calibration
//!
//! This module provides an enhanced modular design for calibration methods,
//! including composable strategies, pluggable modules, and flexible pipelines.

use crate::{
    metrics::{brier_score_decomposition, expected_calibration_error, CalibrationMetricsConfig},
    CalibrationEstimator,
};
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::any::Any;
use std::collections::HashMap;

/// Trait for pluggable calibration modules
pub trait CalibrationModule: Send + Sync {
    /// Get module name
    fn name(&self) -> &str;

    /// Get module version
    fn version(&self) -> &str;

    /// Get module description
    fn description(&self) -> &str;

    /// Create a calibrator instance from this module
    fn create_calibrator(
        &self,
        config: &HashMap<String, String>,
    ) -> Result<Box<dyn CalibrationEstimator>>;

    /// Get module configuration schema
    fn config_schema(&self) -> HashMap<String, ConfigParameter>;

    /// Validate configuration
    fn validate_config(&self, config: &HashMap<String, String>) -> Result<()>;

    /// Get module as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Configuration parameter description
#[derive(Debug, Clone)]
pub struct ConfigParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default: Option<String>,
    /// Description
    pub description: String,
    /// Whether parameter is required
    pub required: bool,
    /// Valid range or options
    pub constraints: Option<ParameterConstraints>,
}

/// Parameter types
#[derive(Debug, Clone)]
pub enum ParameterType {
    /// Float
    Float,
    /// Integer
    Integer,
    /// String
    String,
    /// Boolean
    Boolean,
    /// Array
    Array(Box<ParameterType>),
    /// Object
    Object,
}

/// Parameter constraints
#[derive(Debug, Clone)]
pub enum ParameterConstraints {
    /// Numeric range
    Range { min: Float, max: Float },
    /// String options
    Options(Vec<String>),
    /// Array length
    ArrayLength { min: usize, max: Option<usize> },
    /// Regular expression pattern
    Pattern(String),
}

/// Registry for calibration modules
pub struct CalibrationRegistry {
    /// Registered modules
    modules: HashMap<String, Box<dyn CalibrationModule>>,
    /// Module aliases
    aliases: HashMap<String, String>,
    /// Module dependencies
    dependencies: HashMap<String, Vec<String>>,
}

impl CalibrationRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            aliases: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Register a module
    pub fn register_module(&mut self, module: Box<dyn CalibrationModule>) -> Result<()> {
        let name = module.name().to_string();

        if self.modules.contains_key(&name) {
            return Err(SklearsError::InvalidInput(format!(
                "Module '{}' is already registered",
                name
            )));
        }

        self.modules.insert(name, module);
        Ok(())
    }

    /// Register an alias for a module
    pub fn register_alias(&mut self, alias: String, module_name: String) -> Result<()> {
        if !self.modules.contains_key(&module_name) {
            return Err(SklearsError::InvalidInput(format!(
                "Module '{}' not found",
                module_name
            )));
        }

        self.aliases.insert(alias, module_name);
        Ok(())
    }

    /// Get module by name or alias
    pub fn get_module(&self, name: &str) -> Option<&dyn CalibrationModule> {
        // Try direct name first
        if let Some(module) = self.modules.get(name) {
            return Some(module.as_ref());
        }

        // Try alias
        if let Some(real_name) = self.aliases.get(name) {
            return self.modules.get(real_name).map(|m| m.as_ref());
        }

        None
    }

    /// List all registered modules
    pub fn list_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    /// Get module info
    pub fn get_module_info(&self, name: &str) -> Option<ModuleInfo> {
        self.get_module(name).map(|module| ModuleInfo {
            name: module.name().to_string(),
            version: module.version().to_string(),
            description: module.description().to_string(),
            config_schema: module.config_schema(),
        })
    }

    /// Create calibrator from module
    pub fn create_calibrator(
        &self,
        module_name: &str,
        config: &HashMap<String, String>,
    ) -> Result<Box<dyn CalibrationEstimator>> {
        let module = self.get_module(module_name).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Module '{}' not found", module_name))
        })?;

        module.validate_config(config)?;
        module.create_calibrator(config)
    }
}

impl Default for CalibrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Module information
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub config_schema: HashMap<String, ConfigParameter>,
}

/// Composable calibration strategy
#[derive(Debug, Clone)]
pub struct ComposableStrategy {
    /// Strategy name
    pub name: String,
    /// List of calibration steps
    pub steps: Vec<CalibrationStep>,
    /// Strategy configuration
    pub config: HashMap<String, String>,
}

/// Single calibration step in a strategy
#[derive(Debug, Clone)]
pub struct CalibrationStep {
    /// Step name
    pub name: String,
    /// Module name to use
    pub module: String,
    /// Step configuration
    pub config: HashMap<String, String>,
    /// Whether step is optional
    pub optional: bool,
    /// Conditions for step execution
    pub conditions: Vec<StepCondition>,
}

/// Condition for step execution
#[derive(Debug, Clone)]
pub enum StepCondition {
    PreviousStepSucceeded,
    MetricThreshold {
        metric: String,
        threshold: Float,
        operator: ComparisonOperator,
    },
    /// Data size condition
    DataSize {
        min_samples: Option<usize>,
        max_samples: Option<usize>,
    },
    /// Custom condition
    Custom {
        condition: String,
    },
}

/// Comparison operators for conditions
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    /// GreaterThan
    GreaterThan,
    /// LessThan
    LessThan,
    /// GreaterThanOrEqual
    GreaterThanOrEqual,
    /// LessThanOrEqual
    LessThanOrEqual,
    /// Equal
    Equal,
    /// NotEqual
    NotEqual,
}

/// Extensible evaluation metrics framework
pub struct MetricsFramework {
    /// Registered metrics
    metrics: HashMap<String, Box<dyn CalibrationMetric>>,
    /// Metric aliases
    aliases: HashMap<String, String>,
}

/// Trait for calibration metrics
pub trait CalibrationMetric: Send + Sync {
    /// Metric name
    fn name(&self) -> &str;

    /// Metric description
    fn description(&self) -> &str;

    /// Compute metric
    fn compute(
        &self,
        y_true: &Array1<i32>,
        y_prob: &Array1<Float>,
        config: &CalibrationMetricsConfig,
    ) -> Result<Float>;

    /// Whether higher values are better
    fn higher_is_better(&self) -> bool;

    /// Metric range (if known)
    fn range(&self) -> Option<(Float, Float)>;

    /// Get metric as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

impl MetricsFramework {
    /// Create new metrics framework
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    /// Register a metric
    pub fn register_metric(&mut self, metric: Box<dyn CalibrationMetric>) {
        let name = metric.name().to_string();
        self.metrics.insert(name, metric);
    }

    /// Register metric alias
    pub fn register_alias(&mut self, alias: String, metric_name: String) {
        self.aliases.insert(alias, metric_name);
    }

    /// Compute metric by name
    pub fn compute_metric(
        &self,
        name: &str,
        y_true: &Array1<i32>,
        y_prob: &Array1<Float>,
        config: &CalibrationMetricsConfig,
    ) -> Result<Float> {
        let metric_name = self.aliases.get(name).map_or(name, |v| v.as_str());

        let metric = self
            .metrics
            .get(metric_name)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Metric '{}' not found", name)))?;

        metric.compute(y_true, y_prob, config)
    }

    /// List available metrics
    pub fn list_metrics(&self) -> Vec<&str> {
        self.metrics.keys().map(|s| s.as_str()).collect()
    }

    /// Get metric info
    pub fn get_metric_info(&self, name: &str) -> Option<MetricInfo> {
        let metric_name = self.aliases.get(name).map_or(name, |v| v.as_str());

        self.metrics.get(metric_name).map(|metric| MetricInfo {
            name: metric.name().to_string(),
            description: metric.description().to_string(),
            higher_is_better: metric.higher_is_better(),
            range: metric.range(),
        })
    }
}

impl Default for MetricsFramework {
    fn default() -> Self {
        Self::new()
    }
}

/// Metric information
#[derive(Debug, Clone)]
pub struct MetricInfo {
    pub name: String,
    pub description: String,
    pub higher_is_better: bool,
    pub range: Option<(Float, Float)>,
}

/// Flexible calibration pipeline
pub struct CalibrationPipeline {
    /// Pipeline name
    pub name: String,
    /// Registry for modules
    registry: CalibrationRegistry,
    /// Metrics framework
    metrics: MetricsFramework,
    /// Pipeline steps
    steps: Vec<PipelineStep>,
    /// Pipeline configuration
    config: HashMap<String, String>,
    /// Execution history
    execution_history: Vec<StepResult>,
}

/// Pipeline step
#[derive(Debug, Clone)]
pub struct PipelineStep {
    /// Step ID
    pub id: String,
    /// Step type
    pub step_type: PipelineStepType,
    /// Step configuration
    pub config: HashMap<String, String>,
    /// Dependencies (previous step IDs)
    pub dependencies: Vec<String>,
}

/// Pipeline step types
#[derive(Debug, Clone)]
pub enum PipelineStepType {
    /// Data preprocessing
    Preprocessing { module: String },
    /// Calibration
    Calibration { module: String },
    /// Validation
    Validation { metrics: Vec<String> },
    /// Ensemble
    Ensemble { strategy: String },
    /// Post-processing
    PostProcessing { module: String },
    /// Custom step
    Custom { handler: String },
}

/// Step execution result
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step ID
    pub step_id: String,
    /// Whether step succeeded
    pub success: bool,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Output metrics
    pub metrics: HashMap<String, Float>,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CalibrationPipeline {
    /// Create new pipeline
    pub fn new(name: String) -> Self {
        Self {
            name,
            registry: CalibrationRegistry::new(),
            metrics: MetricsFramework::new(),
            steps: Vec::new(),
            config: HashMap::new(),
            execution_history: Vec::new(),
        }
    }

    /// Add step to pipeline
    pub fn add_step(&mut self, step: PipelineStep) -> Result<()> {
        // Validate dependencies
        for dep in &step.dependencies {
            if !self.steps.iter().any(|s| s.id == *dep) {
                return Err(SklearsError::InvalidInput(format!(
                    "Dependency '{}' not found",
                    dep
                )));
            }
        }

        self.steps.push(step);
        Ok(())
    }

    /// Execute pipeline
    pub fn execute(
        &mut self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<Array1<Float>> {
        self.execution_history.clear();
        let mut current_probabilities = probabilities.clone();

        for step in &self.steps {
            let start_time = std::time::Instant::now();

            let result = match &step.step_type {
                PipelineStepType::Calibration { module } => self.execute_calibration_step(
                    module,
                    &step.config,
                    &current_probabilities,
                    targets,
                ),
                PipelineStepType::Validation { metrics } => {
                    self.execute_validation_step(metrics, &current_probabilities, targets)?;
                    Ok(current_probabilities.clone())
                }
                _ => {
                    // Other step types would be implemented here
                    Ok(current_probabilities.clone())
                }
            };

            let execution_time = start_time.elapsed().as_millis() as u64;

            match result {
                Ok(new_probabilities) => {
                    current_probabilities = new_probabilities;

                    self.execution_history.push(StepResult {
                        step_id: step.id.clone(),
                        success: true,
                        execution_time_ms: execution_time,
                        metrics: HashMap::new(), // Would be populated in real implementation
                        error: None,
                        metadata: HashMap::new(),
                    });
                }
                Err(e) => {
                    self.execution_history.push(StepResult {
                        step_id: step.id.clone(),
                        success: false,
                        execution_time_ms: execution_time,
                        metrics: HashMap::new(),
                        error: Some(e.to_string()),
                        metadata: HashMap::new(),
                    });

                    return Err(e);
                }
            }
        }

        Ok(current_probabilities)
    }

    /// Execute calibration step
    fn execute_calibration_step(
        &self,
        module_name: &str,
        config: &HashMap<String, String>,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<Array1<Float>> {
        let mut calibrator = self.registry.create_calibrator(module_name, config)?;
        calibrator.fit(probabilities, targets)?;
        calibrator.predict_proba(probabilities)
    }

    /// Execute validation step
    fn execute_validation_step(
        &self,
        metric_names: &[String],
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<()> {
        let config = CalibrationMetricsConfig::default();

        for metric_name in metric_names {
            let _metric_value =
                self.metrics
                    .compute_metric(metric_name, targets, probabilities, &config)?;
            // In real implementation, would store/log metric values
        }

        Ok(())
    }

    /// Get execution summary
    pub fn get_execution_summary(&self) -> ExecutionSummary {
        let total_time: u64 = self
            .execution_history
            .iter()
            .map(|r| r.execution_time_ms)
            .sum();
        let successful_steps = self.execution_history.iter().filter(|r| r.success).count();
        let failed_steps = self.execution_history.iter().filter(|r| !r.success).count();

        // ExecutionSummary
        ExecutionSummary {
            total_steps: self.execution_history.len(),
            successful_steps,
            failed_steps,
            total_execution_time_ms: total_time,
            steps: self.execution_history.clone(),
        }
    }

    /// Get registry reference
    pub fn registry(&self) -> &CalibrationRegistry {
        &self.registry
    }

    /// Get mutable registry reference
    pub fn registry_mut(&mut self) -> &mut CalibrationRegistry {
        &mut self.registry
    }

    /// Get metrics framework reference
    pub fn metrics(&self) -> &MetricsFramework {
        &self.metrics
    }

    /// Get mutable metrics framework reference
    pub fn metrics_mut(&mut self) -> &mut MetricsFramework {
        &mut self.metrics
    }
}

/// Pipeline execution summary
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub failed_steps: usize,
    pub total_execution_time_ms: u64,
    pub steps: Vec<StepResult>,
}

/// Built-in ECE metric implementation
#[derive(Debug)]
pub struct ECEMetric;

impl CalibrationMetric for ECEMetric {
    fn name(&self) -> &str {
        "expected_calibration_error"
    }

    fn description(&self) -> &str {
        "Expected Calibration Error - measures calibration quality"
    }

    fn compute(
        &self,
        y_true: &Array1<i32>,
        y_prob: &Array1<Float>,
        config: &CalibrationMetricsConfig,
    ) -> Result<Float> {
        expected_calibration_error(y_true, y_prob, config)
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn range(&self) -> Option<(Float, Float)> {
        Some((0.0, 1.0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Built-in Brier Score metric implementation
#[derive(Debug)]
pub struct BrierScoreMetric;

impl CalibrationMetric for BrierScoreMetric {
    fn name(&self) -> &str {
        "brier_score"
    }

    fn description(&self) -> &str {
        "Brier Score - measures prediction accuracy"
    }

    fn compute(
        &self,
        y_true: &Array1<i32>,
        y_prob: &Array1<Float>,
        config: &CalibrationMetricsConfig,
    ) -> Result<Float> {
        let decomp = brier_score_decomposition(y_true, y_prob, config)?;
        Ok(decomp.brier_score)
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn range(&self) -> Option<(Float, Float)> {
        Some((0.0, 1.0))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Example sigmoid calibration module
#[derive(Debug)]
pub struct SigmoidModule;

impl CalibrationModule for SigmoidModule {
    fn name(&self) -> &str {
        "sigmoid"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Sigmoid (Platt) calibration using logistic regression"
    }

    fn create_calibrator(
        &self,
        _config: &HashMap<String, String>,
    ) -> Result<Box<dyn CalibrationEstimator>> {
        Ok(Box::new(crate::SigmoidCalibrator::new()))
    }

    fn config_schema(&self) -> HashMap<String, ConfigParameter> {
        HashMap::new() // Sigmoid calibration has no configuration parameters
    }

    fn validate_config(&self, _config: &HashMap<String, String>) -> Result<()> {
        Ok(()) // No validation needed for empty config
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Create default registry with built-in modules
pub fn create_default_registry() -> CalibrationRegistry {
    let mut registry = CalibrationRegistry::new();

    // Register built-in modules
    registry.register_module(Box::new(SigmoidModule)).unwrap();

    // Register aliases
    registry
        .register_alias("platt".to_string(), "sigmoid".to_string())
        .unwrap();

    registry
}

/// Create default metrics framework
pub fn create_default_metrics() -> MetricsFramework {
    let mut metrics = MetricsFramework::new();

    // Register built-in metrics
    metrics.register_metric(Box::new(ECEMetric));
    metrics.register_metric(Box::new(BrierScoreMetric));

    // Register aliases
    metrics.register_alias("ece".to_string(), "expected_calibration_error".to_string());
    metrics.register_alias("brier".to_string(), "brier_score".to_string());

    metrics
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let targets = array![0, 0, 1, 1, 1];
        (probabilities, targets)
    }

    #[test]
    fn test_calibration_registry() {
        let mut registry = CalibrationRegistry::new();

        // Register module
        registry.register_module(Box::new(SigmoidModule)).unwrap();

        // Test module lookup
        assert!(registry.get_module("sigmoid").is_some());
        assert!(registry.get_module("nonexistent").is_none());

        // Test alias
        registry
            .register_alias("platt".to_string(), "sigmoid".to_string())
            .unwrap();
        assert!(registry.get_module("platt").is_some());

        // Test module info
        let info = registry.get_module_info("sigmoid").unwrap();
        assert_eq!(info.name, "sigmoid");
        assert_eq!(info.version, "1.0.0");
    }

    #[test]
    fn test_metrics_framework() {
        let mut metrics = MetricsFramework::new();

        // Register metric
        metrics.register_metric(Box::new(ECEMetric));

        // Test metric computation
        let (probabilities, targets) = create_test_data();
        let config = CalibrationMetricsConfig::default();
        let ece = metrics
            .compute_metric(
                "expected_calibration_error",
                &targets,
                &probabilities,
                &config,
            )
            .unwrap();

        assert!(ece >= 0.0 && ece <= 1.0);

        // Test alias
        metrics.register_alias("ece".to_string(), "expected_calibration_error".to_string());
        let ece_alias = metrics
            .compute_metric("ece", &targets, &probabilities, &config)
            .unwrap();
        assert_eq!(ece, ece_alias);
    }

    #[test]
    fn test_calibration_pipeline() {
        let mut pipeline = CalibrationPipeline::new("test_pipeline".to_string());

        // Register modules and metrics
        pipeline
            .registry_mut()
            .register_module(Box::new(SigmoidModule))
            .unwrap();
        pipeline.metrics_mut().register_metric(Box::new(ECEMetric));

        // Add calibration step
        let cal_step = PipelineStep {
            id: "calibration".to_string(),
            step_type: PipelineStepType::Calibration {
                module: "sigmoid".to_string(),
            },
            config: HashMap::new(),
            dependencies: Vec::new(),
        };
        pipeline.add_step(cal_step).unwrap();

        // Add validation step
        let val_step = PipelineStep {
            id: "validation".to_string(),
            step_type: PipelineStepType::Validation {
                metrics: vec!["expected_calibration_error".to_string()],
            },
            config: HashMap::new(),
            dependencies: vec!["calibration".to_string()],
        };
        pipeline.add_step(val_step).unwrap();

        // Execute pipeline
        let (probabilities, targets) = create_test_data();
        let result = pipeline.execute(&probabilities, &targets).unwrap();

        assert_eq!(result.len(), probabilities.len());
        assert!(result.iter().all(|&p| p >= 0.0 && p <= 1.0));

        // Check execution summary
        let summary = pipeline.get_execution_summary();
        assert_eq!(summary.total_steps, 2);
        assert_eq!(summary.successful_steps, 2);
        assert_eq!(summary.failed_steps, 0);
    }

    #[test]
    fn test_default_registry() {
        let registry = create_default_registry();

        assert!(registry.get_module("sigmoid").is_some());
        assert!(registry.get_module("platt").is_some()); // Alias

        let modules = registry.list_modules();
        assert!(!modules.is_empty());
    }

    #[test]
    fn test_default_metrics() {
        let metrics = create_default_metrics();

        let (probabilities, targets) = create_test_data();
        let config = CalibrationMetricsConfig::default();

        // Test ECE
        let ece = metrics
            .compute_metric("ece", &targets, &probabilities, &config)
            .unwrap();
        assert!(ece >= 0.0);

        // Test Brier score
        let brier = metrics
            .compute_metric("brier", &targets, &probabilities, &config)
            .unwrap();
        assert!(brier >= 0.0 && brier <= 1.0);

        let metrics_list = metrics.list_metrics();
        assert!(metrics_list.len() >= 2);
    }

    #[test]
    fn test_config_parameter() {
        let param = ConfigParameter {
            name: "learning_rate".to_string(),
            param_type: ParameterType::Float,
            default: Some("0.01".to_string()),
            description: "Learning rate for optimization".to_string(),
            required: false,
            constraints: Some(ParameterConstraints::Range { min: 0.0, max: 1.0 }),
        };

        assert_eq!(param.name, "learning_rate");
        assert!(!param.required);
        assert!(param.default.is_some());
    }
}
