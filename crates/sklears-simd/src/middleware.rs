//! Middleware system for operation pipelines
//!
//! This module provides a flexible middleware system for composing and chaining
//! SIMD operations through configurable pipelines.

use crate::traits::SimdError;

#[cfg(feature = "no-std")]
extern crate alloc;

#[cfg(feature = "no-std")]
use alloc::{
    boxed::Box,
    collections::BTreeMap as HashMap,
    format,
    string::{String, ToString},
    vec::Vec,
};
#[cfg(feature = "no-std")]
use core::any::Any;
#[cfg(not(feature = "no-std"))]
use std::{any::Any, collections::HashMap, string::ToString};

// Note: format is already imported above for no-std pattern

#[cfg(feature = "no-std")]
use alloc::sync::Arc;
#[cfg(not(feature = "no-std"))]
use std::sync::Arc;

/// Result type for middleware operations
pub type MiddlewareResult<T> = Result<T, SimdError>;

/// Context object that passes through the pipeline
pub struct PipelineContext {
    /// Input data for the pipeline
    pub data: Vec<f32>,
    /// Metadata that can be passed between middleware
    pub metadata: HashMap<String, String>,
    /// Arbitrary context data
    pub context: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl PipelineContext {
    /// Create a new pipeline context with input data
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
            context: HashMap::new(),
        }
    }

    /// Set metadata value
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Set context data
    pub fn set_context<T: Any + Send + Sync>(&mut self, key: String, value: T) {
        self.context.insert(key, Box::new(value));
    }

    /// Get context data
    pub fn get_context<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.context.get(key).and_then(|v| v.downcast_ref::<T>())
    }
}

/// Trait for middleware components
pub trait Middleware: Send + Sync {
    /// Process the pipeline context
    fn process(&self, context: &mut PipelineContext) -> MiddlewareResult<()>;

    /// Get the middleware name
    fn name(&self) -> &str;

    /// Check if this middleware should be executed based on context
    fn should_execute(&self, context: &PipelineContext) -> bool {
        let _ = context; // Suppress unused parameter warning
        true
    }
}

/// Pipeline executor that runs middleware in sequence
pub struct Pipeline {
    /// List of middleware in execution order
    middleware: Vec<Arc<dyn Middleware>>,
    /// Pipeline name
    name: String,
    /// Whether to stop on first error
    fail_fast: bool,
}

impl Pipeline {
    /// Create a new pipeline
    pub fn new(name: String) -> Self {
        Self {
            middleware: Vec::new(),
            name,
            fail_fast: true,
        }
    }

    /// Add middleware to the pipeline
    pub fn add_middleware<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middleware.push(Arc::new(middleware));
        self
    }

    /// Set fail-fast behavior
    pub fn fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Execute the pipeline
    pub fn execute(&self, mut context: PipelineContext) -> MiddlewareResult<PipelineContext> {
        context.set_metadata("pipeline_name".to_string(), self.name.clone());

        for middleware in &self.middleware {
            if middleware.should_execute(&context) {
                if let Err(e) = middleware.process(&mut context) {
                    if self.fail_fast {
                        return Err(e);
                    }
                    // Log error but continue if not fail-fast
                    context.set_metadata(
                        format!("error_{}", middleware.name()),
                        format!("Error: {}", e),
                    );
                }
            }
        }

        Ok(context)
    }

    /// Get pipeline name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get middleware count
    pub fn middleware_count(&self) -> usize {
        self.middleware.len()
    }
}

/// Builder for creating pipelines
pub struct PipelineBuilder {
    name: String,
    middleware: Vec<Arc<dyn Middleware>>,
    fail_fast: bool,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(name: String) -> Self {
        Self {
            name,
            middleware: Vec::new(),
            fail_fast: true,
        }
    }

    /// Add middleware to the pipeline
    pub fn add<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middleware.push(Arc::new(middleware));
        self
    }

    /// Set fail-fast behavior
    pub fn fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline {
        Pipeline {
            middleware: self.middleware,
            name: self.name,
            fail_fast: self.fail_fast,
        }
    }
}

/// Common middleware implementations
/// Normalization middleware
#[derive(Debug, Clone)]
pub struct NormalizationMiddleware {
    /// Normalization type
    norm_type: NormType,
}

#[derive(Debug, Clone)]
pub enum NormType {
    L1,
    L2,
    MinMax,
}

impl NormalizationMiddleware {
    pub fn new(norm_type: NormType) -> Self {
        Self { norm_type }
    }
}

impl Middleware for NormalizationMiddleware {
    fn process(&self, context: &mut PipelineContext) -> MiddlewareResult<()> {
        let data = &mut context.data;

        match self.norm_type {
            NormType::L1 => {
                let sum: f32 = data.iter().map(|x| x.abs()).sum();
                if sum != 0.0 {
                    data.iter_mut().for_each(|x| *x /= sum);
                }
            }
            NormType::L2 => {
                let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm != 0.0 {
                    data.iter_mut().for_each(|x| *x /= norm);
                }
            }
            NormType::MinMax => {
                let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let range = max_val - min_val;
                if range != 0.0 {
                    data.iter_mut().for_each(|x| *x = (*x - min_val) / range);
                }
            }
        }

        context.set_metadata("normalized".to_string(), format!("{:?}", self.norm_type));
        Ok(())
    }

    fn name(&self) -> &str {
        "normalization"
    }
}

/// Filtering middleware
#[derive(Debug, Clone)]
pub struct FilteringMiddleware {
    /// Minimum value threshold
    min_threshold: f32,
    /// Maximum value threshold
    max_threshold: f32,
}

impl FilteringMiddleware {
    pub fn new(min_threshold: f32, max_threshold: f32) -> Self {
        Self {
            min_threshold,
            max_threshold,
        }
    }
}

impl Middleware for FilteringMiddleware {
    fn process(&self, context: &mut PipelineContext) -> MiddlewareResult<()> {
        let original_len = context.data.len();
        context
            .data
            .retain(|&x| x >= self.min_threshold && x <= self.max_threshold);

        let filtered_count = original_len - context.data.len();
        context.set_metadata("filtered_count".to_string(), filtered_count.to_string());

        Ok(())
    }

    fn name(&self) -> &str {
        "filtering"
    }
}

/// Transformation middleware
#[derive(Debug, Clone)]
pub struct TransformationMiddleware {
    /// Transformation function
    transform_type: TransformType,
}

#[derive(Debug, Clone)]
pub enum TransformType {
    Log,
    Exp,
    Sqrt,
    Square,
    Abs,
}

impl TransformationMiddleware {
    pub fn new(transform_type: TransformType) -> Self {
        Self { transform_type }
    }
}

impl Middleware for TransformationMiddleware {
    fn process(&self, context: &mut PipelineContext) -> MiddlewareResult<()> {
        let data = &mut context.data;

        match self.transform_type {
            TransformType::Log => {
                data.iter_mut().for_each(|x| *x = x.max(f32::EPSILON).ln());
            }
            TransformType::Exp => {
                data.iter_mut().for_each(|x| *x = x.exp());
            }
            TransformType::Sqrt => {
                data.iter_mut().for_each(|x| *x = x.max(0.0).sqrt());
            }
            TransformType::Square => {
                data.iter_mut().for_each(|x| *x = *x * *x);
            }
            TransformType::Abs => {
                data.iter_mut().for_each(|x| *x = x.abs());
            }
        }

        context.set_metadata(
            "transformed".to_string(),
            format!("{:?}", self.transform_type),
        );
        Ok(())
    }

    fn name(&self) -> &str {
        "transformation"
    }
}

/// Aggregation middleware
#[derive(Debug, Clone)]
pub struct AggregationMiddleware {
    /// Aggregation function
    agg_type: AggregationType,
}

#[derive(Debug, Clone)]
pub enum AggregationType {
    Sum,
    Mean,
    Max,
    Min,
    StdDev,
}

impl AggregationMiddleware {
    pub fn new(agg_type: AggregationType) -> Self {
        Self { agg_type }
    }
}

impl Middleware for AggregationMiddleware {
    fn process(&self, context: &mut PipelineContext) -> MiddlewareResult<()> {
        let data = &context.data;

        if data.is_empty() {
            return Err(SimdError::InvalidInput(
                "Empty data for aggregation".to_string(),
            ));
        }

        let result = match self.agg_type {
            AggregationType::Sum => data.iter().sum::<f32>(),
            AggregationType::Mean => data.iter().sum::<f32>() / data.len() as f32,
            AggregationType::Max => data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            AggregationType::Min => data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            AggregationType::StdDev => {
                let mean = data.iter().sum::<f32>() / data.len() as f32;
                let variance =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
                variance.sqrt()
            }
        };

        context.set_metadata("aggregation_result".to_string(), result.to_string());
        context.set_metadata(
            "aggregation_type".to_string(),
            format!("{:?}", self.agg_type),
        );

        Ok(())
    }

    fn name(&self) -> &str {
        "aggregation"
    }
}

/// Conditional middleware that executes based on context
pub struct ConditionalMiddleware {
    /// Condition function
    condition: Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>,
    /// Wrapped middleware
    middleware: Arc<dyn Middleware>,
}

impl ConditionalMiddleware {
    pub fn new<F, M>(condition: F, middleware: M) -> Self
    where
        F: Fn(&PipelineContext) -> bool + Send + Sync + 'static,
        M: Middleware + 'static,
    {
        Self {
            condition: Box::new(condition),
            middleware: Arc::new(middleware),
        }
    }
}

impl Middleware for ConditionalMiddleware {
    fn process(&self, context: &mut PipelineContext) -> MiddlewareResult<()> {
        if (self.condition)(context) {
            self.middleware.process(context)
        } else {
            Ok(())
        }
    }

    fn name(&self) -> &str {
        "conditional"
    }

    fn should_execute(&self, context: &PipelineContext) -> bool {
        (self.condition)(context)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_context_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let context = PipelineContext::new(data.clone());

        assert_eq!(context.data, data);
        assert!(context.metadata.is_empty());
        assert!(context.context.is_empty());
    }

    #[test]
    fn test_pipeline_context_metadata() {
        let mut context = PipelineContext::new(vec![1.0, 2.0, 3.0]);

        context.set_metadata("test_key".to_string(), "test_value".to_string());
        assert_eq!(
            context.get_metadata("test_key"),
            Some(&"test_value".to_string())
        );
        assert_eq!(context.get_metadata("nonexistent"), None);
    }

    #[test]
    fn test_pipeline_context_context_data() {
        let mut context = PipelineContext::new(vec![1.0, 2.0, 3.0]);

        context.set_context("test_int".to_string(), 42i32);
        assert_eq!(context.get_context::<i32>("test_int"), Some(&42i32));
        assert_eq!(context.get_context::<f32>("test_int"), None);
    }

    #[test]
    fn test_normalization_middleware_l2() {
        let mut context = PipelineContext::new(vec![3.0, 4.0, 0.0]);
        let middleware = NormalizationMiddleware::new(NormType::L2);

        middleware.process(&mut context).unwrap();

        // L2 norm of [3, 4, 0] is 5, so normalized should be [0.6, 0.8, 0.0]
        assert!((context.data[0] - 0.6).abs() < 1e-6);
        assert!((context.data[1] - 0.8).abs() < 1e-6);
        assert!((context.data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_filtering_middleware() {
        let mut context = PipelineContext::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let middleware = FilteringMiddleware::new(2.0, 4.0);

        middleware.process(&mut context).unwrap();

        assert_eq!(context.data, vec![2.0, 3.0, 4.0]);
        assert_eq!(
            context.get_metadata("filtered_count"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn test_transformation_middleware_sqrt() {
        let mut context = PipelineContext::new(vec![1.0, 4.0, 9.0, 16.0]);
        let middleware = TransformationMiddleware::new(TransformType::Sqrt);

        middleware.process(&mut context).unwrap();

        assert_eq!(context.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_aggregation_middleware_mean() {
        let mut context = PipelineContext::new(vec![1.0, 2.0, 3.0, 4.0]);
        let middleware = AggregationMiddleware::new(AggregationType::Mean);

        middleware.process(&mut context).unwrap();

        assert_eq!(
            context.get_metadata("aggregation_result"),
            Some(&"2.5".to_string())
        );
        assert_eq!(
            context.get_metadata("aggregation_type"),
            Some(&"Mean".to_string())
        );
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new("test_pipeline".to_string())
            .add(NormalizationMiddleware::new(NormType::L2))
            .add(FilteringMiddleware::new(0.1, 0.9))
            .fail_fast(false)
            .build();

        assert_eq!(pipeline.name(), "test_pipeline");
        assert_eq!(pipeline.middleware_count(), 2);
    }

    #[test]
    fn test_pipeline_execution() {
        let pipeline = Pipeline::new("test_pipeline".to_string())
            .add_middleware(NormalizationMiddleware::new(NormType::L2))
            .add_middleware(TransformationMiddleware::new(TransformType::Square));

        let context = PipelineContext::new(vec![3.0, 4.0, 0.0]);
        let result = pipeline.execute(context).unwrap();

        // After L2 normalization: [0.6, 0.8, 0.0]
        // After squaring: [0.36, 0.64, 0.0]
        assert!((result.data[0] - 0.36).abs() < 1e-6);
        assert!((result.data[1] - 0.64).abs() < 1e-6);
        assert!((result.data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_conditional_middleware() {
        let condition = |context: &PipelineContext| context.data.len() > 2;
        let middleware =
            ConditionalMiddleware::new(condition, NormalizationMiddleware::new(NormType::L2));

        // Test with data length > 2 (should execute)
        let mut context = PipelineContext::new(vec![3.0, 4.0, 0.0]);
        middleware.process(&mut context).unwrap();
        assert!((context.data[0] - 0.6).abs() < 1e-6);

        // Test with data length <= 2 (should not execute)
        let mut context = PipelineContext::new(vec![3.0, 4.0]);
        let original_data = context.data.clone();
        middleware.process(&mut context).unwrap();
        assert_eq!(context.data, original_data); // Should be unchanged
    }

    #[test]
    fn test_empty_data_handling() {
        let mut context = PipelineContext::new(vec![]);
        let middleware = AggregationMiddleware::new(AggregationType::Mean);

        let result = middleware.process(&mut context);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_metadata() {
        let pipeline = Pipeline::new("test_pipeline".to_string())
            .add_middleware(NormalizationMiddleware::new(NormType::L2));

        let context = PipelineContext::new(vec![1.0, 2.0, 3.0]);
        let result = pipeline.execute(context).unwrap();

        assert_eq!(
            result.get_metadata("pipeline_name"),
            Some(&"test_pipeline".to_string())
        );
        assert_eq!(result.get_metadata("normalized"), Some(&"L2".to_string()));
    }
}
