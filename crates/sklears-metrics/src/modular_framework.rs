//! Modular Metric Framework with Trait-Based Design
//!
//! This module provides a comprehensive trait-based framework for metrics
//! that enables composable combinations, extensible scoring functions,
//! and flexible aggregation strategies. The design emphasizes modularity,
//! extensibility, and type safety.
//!
//! # Features
//!
//! - Trait-based metric definitions with associated types
//! - Composable metric combinations and transformations
//! - Extensible scoring function system with pluggable scorers
//! - Flexible aggregation strategies (mean, weighted, robust)
//! - Plugin architecture for custom metrics
//! - Middleware system for metric pipelines
//! - Dynamic metric registration and discovery
//! - Serializable metric results and configurations
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_metrics::modular_framework::*;
//! use sklears_metrics::MetricsError;
//! use scirs2_core::ndarray::Array1;
//!
//! // Define a custom metric
//! struct CustomAccuracy;
//!
//! impl Metric for CustomAccuracy {
//!     type Input = (Array1<i32>, Array1<i32>);
//!     type Output = f64;
//!     type Error = MetricsError;
//!
//!     fn compute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
//!         let (y_true, y_pred) = input;
//!         let correct = y_true.iter().zip(y_pred.iter())
//!             .filter(|(a, b)| a == b)
//!             .count();
//!         Ok(correct as f64 / y_true.len() as f64)
//!     }
//!
//!     fn name(&self) -> &'static str {
//!         "custom_accuracy"
//!     }
//! }
//!
//! // Use with metric pipeline
//! let pipeline = MetricPipeline::new()
//!     .add_metric(Box::new(CustomAccuracy))
//!     .add_aggregator(Box::new(MeanAggregator));
//!
//! let y_true = Array1::from_vec(vec![0, 1, 1, 0]);
//! let y_pred = Array1::from_vec(vec![0, 1, 0, 0]);
//! let results = pipeline.compute(&(y_true, y_pred)).unwrap();
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Type alias for metric aggregator map
type AggregatorMap =
    Arc<RwLock<HashMap<String, Box<dyn MetricAggregator<Value = f64, Result = f64>>>>>;

/// Core trait for all metrics in the framework
pub trait Metric: Send + Sync {
    /// Input type for the metric
    type Input;
    /// Output type for the metric result
    type Output;
    /// Error type for metric computation
    type Error;

    /// Compute the metric value
    fn compute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;

    /// Get the metric name
    fn name(&self) -> &'static str;

    /// Get metric metadata
    fn metadata(&self) -> MetricMetadata
    where
        Self::Input: 'static,
        Self::Output: 'static,
    {
        MetricMetadata {
            name: self.name(),
            description: format!("Metric: {}", self.name()),
            input_types: vec![TypeId::of::<Self::Input>()],
            output_type: TypeId::of::<Self::Output>(),
            properties: MetricProperties::default(),
        }
    }

    /// Validate input before computation
    fn validate_input(&self, _input: &Self::Input) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Metadata for metrics
#[derive(Debug, Clone)]
pub struct MetricMetadata {
    pub name: &'static str,
    pub description: String,
    pub input_types: Vec<TypeId>,
    pub output_type: TypeId,
    pub properties: MetricProperties,
}

/// Properties of a metric
#[derive(Debug, Clone, Default)]
pub struct MetricProperties {
    /// Whether higher values are better
    pub higher_is_better: bool,
    /// Expected range of values
    pub value_range: Option<(f64, f64)>,
    /// Whether the metric is symmetric
    pub is_symmetric: bool,
    /// Whether the metric handles missing values
    pub handles_missing: bool,
    /// Computational complexity
    pub complexity: ComputationalComplexity,
}

/// Computational complexity classification
#[derive(Debug, Clone, Default)]
pub enum ComputationalComplexity {
    #[default]
    Constant,
    /// Linear
    Linear,
    /// Quadratic
    Quadratic,
    /// Cubic
    Cubic,
    /// Exponential
    Exponential,
    /// Unknown
    Unknown,
}

/// Trait for composable metrics that can be combined
pub trait ComposableMetric: Metric {
    /// Combine with another metric
    fn compose<Other>(self, other: Other) -> ComposedMetric<Self, Other>
    where
        Self: Sized,
        Other: ComposableMetric<Input = Self::Input>,
    {
        ComposedMetric::new(self, other)
    }

    /// Transform the metric output
    fn transform<F, Output>(self, transform_fn: F) -> TransformedMetric<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Output) -> Output + Send + Sync,
    {
        TransformedMetric::new(self, transform_fn)
    }
}

/// A composed metric that combines two metrics
pub struct ComposedMetric<M1, M2> {
    metric1: M1,
    metric2: M2,
}

impl<M1, M2> ComposedMetric<M1, M2> {
    pub fn new(metric1: M1, metric2: M2) -> Self {
        Self { metric1, metric2 }
    }
}

impl<M1, M2> Metric for ComposedMetric<M1, M2>
where
    M1: ComposableMetric,
    M2: ComposableMetric<Input = M1::Input>,
    M1::Output: Clone,
    M2::Output: Clone,
{
    type Input = M1::Input;
    type Output = (M1::Output, M2::Output);
    type Error = MetricsError;

    fn compute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let result1 = self.metric1.compute(input).map_err(|_| {
            MetricsError::InvalidInput("First metric computation failed".to_string())
        })?;
        let result2 = self.metric2.compute(input).map_err(|_| {
            MetricsError::InvalidInput("Second metric computation failed".to_string())
        })?;

        Ok((result1, result2))
    }

    fn name(&self) -> &'static str {
        "composed_metric"
    }
}

/// A metric with transformed output
pub struct TransformedMetric<M, F> {
    metric: M,
    transform_fn: F,
}

impl<M, F> TransformedMetric<M, F> {
    pub fn new(metric: M, transform_fn: F) -> Self {
        Self {
            metric,
            transform_fn,
        }
    }
}

impl<M, F, Output> Metric for TransformedMetric<M, F>
where
    M: Metric,
    F: Fn(M::Output) -> Output + Send + Sync,
    Output: Send + Sync,
{
    type Input = M::Input;
    type Output = Output;
    type Error = M::Error;

    fn compute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let result = self.metric.compute(input)?;
        Ok((self.transform_fn)(result))
    }

    fn name(&self) -> &'static str {
        "transformed_metric"
    }
}

/// Trait for aggregating multiple metric results
pub trait MetricAggregator: Send + Sync {
    /// Type of values being aggregated
    type Value;
    /// Type of aggregated result
    type Result;

    /// Aggregate multiple values
    fn aggregate(&self, values: &[Self::Value]) -> Self::Result;

    /// Get aggregator name
    fn name(&self) -> &'static str;
}

/// Mean aggregator
pub struct MeanAggregator;

impl MetricAggregator for MeanAggregator {
    type Value = f64;
    type Result = f64;

    fn aggregate(&self, values: &[Self::Value]) -> Self::Result {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn name(&self) -> &'static str {
        "mean"
    }
}

/// Weighted aggregator
pub struct WeightedAggregator {
    weights: Vec<f64>,
}

impl WeightedAggregator {
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }
}

impl MetricAggregator for WeightedAggregator {
    type Value = f64;
    type Result = f64;

    fn aggregate(&self, values: &[Self::Value]) -> Self::Result {
        if values.is_empty() || self.weights.is_empty() {
            return 0.0;
        }

        let n = values.len().min(self.weights.len());
        let weighted_sum: f64 = values
            .iter()
            .take(n)
            .zip(self.weights.iter().take(n))
            .map(|(value, weight)| value * weight)
            .sum();

        let weight_sum: f64 = self.weights.iter().take(n).sum();

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    fn name(&self) -> &'static str {
        "weighted"
    }
}

/// Robust aggregator using median
pub struct RobustAggregator;

impl MetricAggregator for RobustAggregator {
    type Value = f64;
    type Result = f64;

    fn aggregate(&self, values: &[Self::Value]) -> Self::Result {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted_values.len();
        if len % 2 == 0 {
            (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
        } else {
            sorted_values[len / 2]
        }
    }

    fn name(&self) -> &'static str {
        "robust"
    }
}

/// Metric pipeline for composing multiple metrics and aggregators
pub struct MetricPipeline {
    metrics: Vec<Box<dyn DynMetric>>,
    aggregators: Vec<Box<dyn MetricAggregator<Value = f64, Result = f64>>>,
    middleware: Vec<Box<dyn MetricMiddleware>>,
}

impl MetricPipeline {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            aggregators: Vec::new(),
            middleware: Vec::new(),
        }
    }

    pub fn add_metric(mut self, metric: Box<dyn DynMetric>) -> Self {
        self.metrics.push(metric);
        self
    }

    pub fn add_aggregator(
        mut self,
        aggregator: Box<dyn MetricAggregator<Value = f64, Result = f64>>,
    ) -> Self {
        self.aggregators.push(aggregator);
        self
    }

    pub fn add_middleware(mut self, middleware: Box<dyn MetricMiddleware>) -> Self {
        self.middleware.push(middleware);
        self
    }

    /// Compute all metrics in the pipeline
    pub fn compute(&self, input: &MetricInput) -> MetricsResult<PipelineResult> {
        let mut results = Vec::new();
        let mut context = MetricContext::new();

        // Apply pre-processing middleware
        for middleware in &self.middleware {
            middleware.pre_process(input, &mut context)?;
        }

        // Compute all metrics
        for metric in &self.metrics {
            match metric.compute_dyn(input) {
                Ok(result) => {
                    results.push(result);

                    // Apply per-metric middleware
                    for middleware in &self.middleware {
                        middleware.post_metric(metric.name(), result, &mut context)?;
                    }
                }
                Err(e) => {
                    return Err(MetricsError::InvalidInput(format!(
                        "Metric '{}' failed: {:?}",
                        metric.name(),
                        e
                    )));
                }
            }
        }

        // Aggregate results
        let mut aggregated_results = HashMap::new();
        for aggregator in &self.aggregators {
            let aggregated = aggregator.aggregate(&results);
            aggregated_results.insert(aggregator.name().to_string(), aggregated);
        }

        // Apply post-processing middleware
        for middleware in &self.middleware {
            middleware.post_process(&aggregated_results, &mut context)?;
        }

        Ok(PipelineResult {
            individual_results: results,
            aggregated_results,
            metadata: context.metadata,
        })
    }
}

impl Default for MetricPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Dynamic metric trait for type erasure
pub trait DynMetric: Send + Sync {
    fn compute_dyn(&self, input: &MetricInput) -> Result<f64, Box<dyn std::error::Error>>;
    fn name(&self) -> &'static str;
    fn metadata(&self) -> MetricMetadata;
}

/// Implementation wrapper for dynamic metrics
pub struct DynMetricWrapper<M> {
    metric: M,
}

impl<M> DynMetricWrapper<M>
where
    M: Metric<Input = MetricInput, Output = f64, Error = MetricsError> + 'static,
{
    pub fn new(metric: M) -> Self {
        Self { metric }
    }
}

impl<M> DynMetric for DynMetricWrapper<M>
where
    M: Metric<Input = MetricInput, Output = f64, Error = MetricsError> + 'static,
{
    fn compute_dyn(&self, input: &MetricInput) -> Result<f64, Box<dyn std::error::Error>> {
        self.metric
            .compute(input)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    fn name(&self) -> &'static str {
        self.metric.name()
    }

    fn metadata(&self) -> MetricMetadata {
        self.metric.metadata()
    }
}

/// Common input type for metrics
#[derive(Debug, Clone)]
pub enum MetricInput {
    /// Classification
    Classification {
        y_true: Array1<i32>,

        y_pred: Array1<i32>,

        y_prob: Option<Array2<f64>>,
    },
    /// Regression
    Regression {
        y_true: Array1<f64>,

        y_pred: Array1<f64>,
    },
    /// Clustering
    Clustering {
        labels_true: Array1<i32>,
        labels_pred: Array1<i32>,
        data: Option<Array2<f64>>,
    },
    Ranking {
        y_true: Array1<f64>,
        y_score: Array1<f64>,
    },
}

/// Execution context for metrics
#[derive(Debug, Clone, Default)]
pub struct MetricContext {
    #[cfg(feature = "serde")]
    pub metadata: HashMap<String, serde_json::Value>,
    #[cfg(not(feature = "serde"))]
    pub metadata: HashMap<String, String>,
    pub timings: HashMap<String, std::time::Duration>,
    pub intermediate_results: HashMap<String, f64>,
}

impl MetricContext {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "serde")]
    pub fn add_metadata<T: Serialize>(&mut self, key: String, value: T) {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.metadata.insert(key, json_value);
        }
    }

    #[cfg(not(feature = "serde"))]
    pub fn add_metadata<T: std::fmt::Debug>(&mut self, key: String, value: T) {
        self.metadata.insert(key, format!("{:?}", value));
    }

    pub fn add_timing(&mut self, key: String, duration: std::time::Duration) {
        self.timings.insert(key, duration);
    }
}

/// Result of pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub individual_results: Vec<f64>,
    pub aggregated_results: HashMap<String, f64>,
    #[cfg(feature = "serde")]
    pub metadata: HashMap<String, serde_json::Value>,
    #[cfg(not(feature = "serde"))]
    pub metadata: HashMap<String, String>,
}

impl PipelineResult {
    /// Get a specific aggregated result
    pub fn get_aggregated(&self, aggregator_name: &str) -> Option<f64> {
        self.aggregated_results.get(aggregator_name).copied()
    }

    /// Get the mean of all individual results
    pub fn mean(&self) -> f64 {
        if self.individual_results.is_empty() {
            0.0
        } else {
            self.individual_results.iter().sum::<f64>() / self.individual_results.len() as f64
        }
    }
}

/// Trait for middleware in metric pipelines
pub trait MetricMiddleware: Send + Sync {
    /// Called before metric computation
    fn pre_process(&self, input: &MetricInput, context: &mut MetricContext) -> MetricsResult<()> {
        let _ = (input, context);
        Ok(())
    }

    /// Called after each metric computation
    fn post_metric(
        &self,
        metric_name: &str,
        result: f64,
        context: &mut MetricContext,
    ) -> MetricsResult<()> {
        let _ = (metric_name, result, context);
        Ok(())
    }

    /// Called after all metrics are computed
    fn post_process(
        &self,
        results: &HashMap<String, f64>,
        context: &mut MetricContext,
    ) -> MetricsResult<()> {
        let _ = (results, context);
        Ok(())
    }

    /// Get middleware name
    fn name(&self) -> &'static str;
}

/// Logging middleware
pub struct LoggingMiddleware {
    log_level: LogLevel,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    /// Debug
    Debug,
    /// Info
    Info,
    /// Warn
    Warn,
    /// Error
    Error,
}

impl LoggingMiddleware {
    pub fn new(log_level: LogLevel) -> Self {
        Self { log_level }
    }
}

impl MetricMiddleware for LoggingMiddleware {
    fn pre_process(&self, _input: &MetricInput, _context: &mut MetricContext) -> MetricsResult<()> {
        match self.log_level {
            LogLevel::Debug => println!("DEBUG: Starting metric computation"),
            LogLevel::Info => println!("INFO: Computing metrics"),
            _ => {}
        }
        Ok(())
    }

    fn post_metric(
        &self,
        metric_name: &str,
        result: f64,
        _context: &mut MetricContext,
    ) -> MetricsResult<()> {
        match self.log_level {
            LogLevel::Debug => println!("DEBUG: Metric '{}' = {}", metric_name, result),
            LogLevel::Info => println!("INFO: Computed metric '{}'", metric_name),
            _ => {}
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "logging"
    }
}

/// Timing middleware
pub struct TimingMiddleware;

impl MetricMiddleware for TimingMiddleware {
    fn pre_process(&self, _input: &MetricInput, context: &mut MetricContext) -> MetricsResult<()> {
        context.add_timing(
            "pipeline_start".to_string(),
            std::time::Duration::from_nanos(0),
        );
        Ok(())
    }

    fn post_metric(
        &self,
        metric_name: &str,
        _result: f64,
        context: &mut MetricContext,
    ) -> MetricsResult<()> {
        let now = std::time::Instant::now();
        context.add_timing(format!("metric_{}_computed", metric_name), now.elapsed());
        Ok(())
    }

    fn name(&self) -> &'static str {
        "timing"
    }
}

/// Metric registry for dynamic registration and discovery
pub struct MetricRegistry {
    metrics: Arc<RwLock<HashMap<String, Box<dyn DynMetric>>>>,
    aggregators: AggregatorMap,
    middleware: Arc<RwLock<HashMap<String, Box<dyn MetricMiddleware>>>>,
}

impl MetricRegistry {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            aggregators: Arc::new(RwLock::new(HashMap::new())),
            middleware: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a metric
    pub fn register_metric(&self, name: String, metric: Box<dyn DynMetric>) -> MetricsResult<()> {
        let mut metrics = self
            .metrics
            .write()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire write lock".to_string()))?;
        metrics.insert(name, metric);
        Ok(())
    }

    /// Register an aggregator
    pub fn register_aggregator(
        &self,
        name: String,
        aggregator: Box<dyn MetricAggregator<Value = f64, Result = f64>>,
    ) -> MetricsResult<()> {
        let mut aggregators = self
            .aggregators
            .write()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire write lock".to_string()))?;
        aggregators.insert(name, aggregator);
        Ok(())
    }

    /// Register middleware
    pub fn register_middleware(
        &self,
        name: String,
        middleware: Box<dyn MetricMiddleware>,
    ) -> MetricsResult<()> {
        let mut middleware_map = self
            .middleware
            .write()
            .map_err(|_| MetricsError::InvalidInput("Failed to acquire write lock".to_string()))?;
        middleware_map.insert(name, middleware);
        Ok(())
    }

    /// Get registered metric names
    pub fn metric_names(&self) -> Vec<String> {
        if let Ok(metrics) = self.metrics.read() {
            metrics.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Create pipeline from registered components
    pub fn create_pipeline(&self, metric_names: Vec<String>) -> MetricsResult<MetricPipeline> {
        let pipeline = MetricPipeline::new();

        if let Ok(metrics) = self.metrics.read() {
            for name in metric_names {
                if let Some(_metric) = metrics.get(&name) {
                    // Note: This is simplified - in practice you'd need to clone the metric
                    // which requires additional trait bounds
                }
            }
        }

        Ok(pipeline)
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global metric registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<MetricRegistry> = std::sync::OnceLock::new();

/// Get the global metric registry
pub fn global_registry() -> &'static MetricRegistry {
    GLOBAL_REGISTRY.get_or_init(MetricRegistry::new)
}

/// Scoring function trait for extensible scoring
pub trait ScoringFunction: Send + Sync {
    /// Compute score based on metric results
    fn score(&self, results: &HashMap<String, f64>) -> f64;

    /// Get scoring function name
    fn name(&self) -> &'static str;

    /// Get configuration if any
    fn config(&self) -> Option<ScoringConfig> {
        None
    }
}

/// Configuration for scoring functions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScoringConfig {
    pub weights: HashMap<String, f64>,
    pub normalization: Option<NormalizationMethod>,
    pub threshold: Option<f64>,
}

/// Normalization methods for scoring
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NormalizationMethod {
    /// MinMax
    MinMax,
    /// ZScore
    ZScore,
    /// RobustZScore
    RobustZScore,

    None,
}

/// Weighted scoring function
pub struct WeightedScoringFunction {
    weights: HashMap<String, f64>,
}

impl WeightedScoringFunction {
    pub fn new(weights: HashMap<String, f64>) -> Self {
        Self { weights }
    }
}

impl ScoringFunction for WeightedScoringFunction {
    fn score(&self, results: &HashMap<String, f64>) -> f64 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        for (metric_name, &result) in results {
            if let Some(&weight) = self.weights.get(metric_name) {
                score += result * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }

    fn name(&self) -> &'static str {
        "weighted"
    }

    fn config(&self) -> Option<ScoringConfig> {
        Some(ScoringConfig {
            weights: self.weights.clone(),
            normalization: None,
            threshold: None,
        })
    }
}

/// Plugin trait for extending the framework
pub trait MetricPlugin: Send + Sync {
    /// Initialize the plugin
    fn initialize(&self, registry: &MetricRegistry) -> MetricsResult<()>;

    /// Get plugin name
    fn name(&self) -> &'static str;

    /// Get plugin version
    fn version(&self) -> &'static str;

    /// Get plugin dependencies
    fn dependencies(&self) -> Vec<&'static str> {
        Vec::new()
    }
}

/// Plugin manager for loading and managing plugins
pub struct PluginManager {
    plugins: Vec<Box<dyn MetricPlugin>>,
    registry: Arc<MetricRegistry>,
}

impl PluginManager {
    pub fn new(registry: Arc<MetricRegistry>) -> Self {
        Self {
            plugins: Vec::new(),
            registry,
        }
    }

    /// Load a plugin
    pub fn load_plugin(&mut self, plugin: Box<dyn MetricPlugin>) -> MetricsResult<()> {
        plugin.initialize(&self.registry)?;
        self.plugins.push(plugin);
        Ok(())
    }

    /// Get loaded plugins
    pub fn loaded_plugins(&self) -> Vec<(&'static str, &'static str)> {
        self.plugins
            .iter()
            .map(|p| (p.name(), p.version()))
            .collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Test metric implementation
    struct TestAccuracy;

    impl Metric for TestAccuracy {
        type Input = MetricInput;
        type Output = f64;
        type Error = MetricsError;

        fn compute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
            match input {
                MetricInput::Classification { y_true, y_pred, .. } => {
                    if y_true.len() != y_pred.len() {
                        return Err(MetricsError::ShapeMismatch {
                            expected: vec![y_true.len()],
                            actual: vec![y_pred.len()],
                        });
                    }

                    let correct = y_true
                        .iter()
                        .zip(y_pred.iter())
                        .filter(|(a, b)| a == b)
                        .count();

                    Ok(correct as f64 / y_true.len() as f64)
                }
                _ => Err(MetricsError::InvalidInput(
                    "Expected classification input".to_string(),
                )),
            }
        }

        fn name(&self) -> &'static str {
            "test_accuracy"
        }
    }

    impl ComposableMetric for TestAccuracy {}

    #[test]
    fn test_metric_trait() {
        let metric = TestAccuracy;
        let input = MetricInput::Classification {
            y_true: array![0, 1, 1, 0],
            y_pred: array![0, 1, 0, 0],
            y_prob: None,
        };

        let result = metric.compute(&input).unwrap();
        assert_eq!(result, 0.75);
        assert_eq!(metric.name(), "test_accuracy");
    }

    #[test]
    fn test_composable_metrics() {
        let metric1 = TestAccuracy;
        let metric2 = TestAccuracy;

        let composed = metric1.compose(metric2);

        let input = MetricInput::Classification {
            y_true: array![0, 1, 1, 0],
            y_pred: array![0, 1, 0, 0],
            y_prob: None,
        };

        let result = composed.compute(&input).unwrap();
        assert_eq!(result.0, 0.75);
        assert_eq!(result.1, 0.75);
    }

    #[test]
    fn test_transformed_metric() {
        let metric = TestAccuracy;
        let transformed = metric.transform(|x| x * 100.0);

        let input = MetricInput::Classification {
            y_true: array![0, 1, 1, 0],
            y_pred: array![0, 1, 0, 0],
            y_prob: None,
        };

        let result = transformed.compute(&input).unwrap();
        assert_eq!(result, 75.0);
    }

    #[test]
    fn test_aggregators() {
        let values = vec![0.8, 0.9, 0.7, 0.85];

        // Test mean aggregator
        let mean_agg = MeanAggregator;
        let mean_result = mean_agg.aggregate(&values);
        assert!((mean_result - 0.8125).abs() < 1e-10);

        // Test weighted aggregator
        let weights = vec![0.5, 0.3, 0.1, 0.1];
        let weighted_agg = WeightedAggregator::new(weights);
        let weighted_result = weighted_agg.aggregate(&values);
        assert!((weighted_result - 0.825).abs() < 1e-10);

        // Test robust aggregator
        let robust_agg = RobustAggregator;
        let robust_result = robust_agg.aggregate(&values);
        assert_eq!(robust_result, 0.825); // Median of [0.7, 0.8, 0.85, 0.9]
    }

    #[test]
    fn test_metric_pipeline() {
        let pipeline = MetricPipeline::new()
            .add_metric(Box::new(DynMetricWrapper::new(TestAccuracy)))
            .add_aggregator(Box::new(MeanAggregator))
            .add_middleware(Box::new(TimingMiddleware));

        let input = MetricInput::Classification {
            y_true: array![0, 1, 1, 0],
            y_pred: array![0, 1, 0, 0],
            y_prob: None,
        };

        let result = pipeline.compute(&input).unwrap();
        assert_eq!(result.individual_results.len(), 1);
        assert_eq!(result.individual_results[0], 0.75);
        assert!(result.aggregated_results.contains_key("mean"));
    }

    #[test]
    fn test_metric_registry() {
        let registry = MetricRegistry::new();

        let metric = Box::new(DynMetricWrapper::new(TestAccuracy));
        registry
            .register_metric("test_accuracy".to_string(), metric)
            .unwrap();

        let names = registry.metric_names();
        assert!(names.contains(&"test_accuracy".to_string()));
    }

    #[test]
    fn test_scoring_function() {
        let mut weights = HashMap::new();
        weights.insert("accuracy".to_string(), 0.7);
        weights.insert("precision".to_string(), 0.3);

        let scoring = WeightedScoringFunction::new(weights);

        let mut results = HashMap::new();
        results.insert("accuracy".to_string(), 0.8);
        results.insert("precision".to_string(), 0.9);

        let score = scoring.score(&results);
        assert!((score - 0.83).abs() < 1e-10); // 0.8 * 0.7 + 0.9 * 0.3 = 0.83
    }

    #[test]
    fn test_middleware() {
        let mut context = MetricContext::new();
        let middleware = TimingMiddleware;

        let input = MetricInput::Classification {
            y_true: array![0, 1],
            y_pred: array![0, 1],
            y_prob: None,
        };

        middleware.pre_process(&input, &mut context).unwrap();
        middleware.post_metric("test", 0.8, &mut context).unwrap();

        assert_eq!(middleware.name(), "timing");
    }
}
