//! Middleware system for kernel approximation pipelines
//!
//! This module provides a flexible middleware architecture for composing
//! kernel approximation transformations with hooks, callbacks, and monitoring.
//!
//! # Examples
//!
//! ```rust
//! use sklears_kernel_approximation::middleware::{Pipeline, PipelineBuilder};
//! // Create a pipeline with multiple transformations
//! // let pipeline = PipelineBuilder::new()
//! //     .add_transform(rbf_sampler)
//! //     .add_hook(logging_hook)
//! //     .add_middleware(normalization_middleware)
//! //     .build();
//! ```

use scirs2_core::ndarray::Array2;
use sklears_core::error::SklearsError;
use std::any::Any;
use std::sync::Arc;
use std::time::Instant;

/// Hook that can be called at various stages of the pipeline
pub trait Hook: Send + Sync {
    /// Called before fit
    fn before_fit(
        &mut self,
        x: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let _ = (x, context);
        Ok(())
    }

    /// Called after fit
    fn after_fit(
        &mut self,
        x: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let _ = (x, context);
        Ok(())
    }

    /// Called before transform
    fn before_transform(
        &mut self,
        x: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let _ = (x, context);
        Ok(())
    }

    /// Called after transform
    fn after_transform(
        &mut self,
        x: &Array2<f64>,
        output: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let _ = (x, output, context);
        Ok(())
    }

    /// Called on error
    fn on_error(&mut self, error: &SklearsError, context: &mut HookContext) {
        let _ = (error, context);
    }

    /// Get hook name
    fn name(&self) -> &str {
        "Hook"
    }
}

/// Context passed to hooks containing metadata
#[derive(Debug, Clone, Default)]
pub struct HookContext {
    /// Stage name (e.g., "fit", "transform")
    pub stage: String,
    /// Transform index in pipeline
    pub transform_index: usize,
    /// Transform name
    pub transform_name: String,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl HookContext {
    /// Create a new hook context
    pub fn new(stage: &str, transform_index: usize, transform_name: &str) -> Self {
        Self {
            stage: stage.to_string(),
            transform_index,
            transform_name: transform_name.to_string(),
            elapsed_ms: 0.0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata entry
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Middleware that wraps transformations
pub trait Middleware: Send + Sync {
    /// Process before fit
    fn process_before_fit(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        Ok(x.clone())
    }

    /// Process after fit
    fn process_after_fit(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        Ok(x.clone())
    }

    /// Process before transform
    fn process_before_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        Ok(x.clone())
    }

    /// Process after transform
    fn process_after_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        Ok(x.clone())
    }

    /// Get middleware name
    fn name(&self) -> &str {
        "Middleware"
    }
}

/// Transform stage in the pipeline
pub trait PipelineStage: Send + Sync {
    /// Fit the stage
    fn fit(&mut self, x: &Array2<f64>) -> Result<(), SklearsError>;

    /// Transform using the fitted stage
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError>;

    /// Check if stage is fitted
    fn is_fitted(&self) -> bool;

    /// Get stage name
    fn name(&self) -> &str;

    /// Clone the stage
    fn clone_stage(&self) -> Box<dyn PipelineStage>;

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Logging hook that records timing and shapes
pub struct LoggingHook {
    logs: Vec<String>,
}

impl LoggingHook {
    /// Create a new logging hook
    pub fn new() -> Self {
        Self { logs: Vec::new() }
    }

    /// Get all logs
    pub fn logs(&self) -> &[String] {
        &self.logs
    }
}

impl Default for LoggingHook {
    fn default() -> Self {
        Self::new()
    }
}

impl Hook for LoggingHook {
    fn before_fit(
        &mut self,
        x: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let log = format!(
            "[{}] Before fit - transform: {}, shape: {:?}",
            context.stage,
            context.transform_name,
            x.dim()
        );
        self.logs.push(log);
        Ok(())
    }

    fn after_fit(
        &mut self,
        _x: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let log = format!(
            "[{}] After fit - transform: {}, time: {:.2}ms",
            context.stage, context.transform_name, context.elapsed_ms
        );
        self.logs.push(log);
        Ok(())
    }

    fn before_transform(
        &mut self,
        x: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let log = format!(
            "[{}] Before transform - transform: {}, shape: {:?}",
            context.stage,
            context.transform_name,
            x.dim()
        );
        self.logs.push(log);
        Ok(())
    }

    fn after_transform(
        &mut self,
        _x: &Array2<f64>,
        output: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        let log = format!(
            "[{}] After transform - transform: {}, output shape: {:?}, time: {:.2}ms",
            context.stage,
            context.transform_name,
            output.dim(),
            context.elapsed_ms
        );
        self.logs.push(log);
        Ok(())
    }

    fn name(&self) -> &str {
        "LoggingHook"
    }
}

/// Normalization middleware
pub struct NormalizationMiddleware {
    mean: Option<Array2<f64>>,
    std: Option<Array2<f64>>,
}

impl NormalizationMiddleware {
    /// Create a new normalization middleware
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }
}

impl Default for NormalizationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for NormalizationMiddleware {
    fn process_before_fit(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        // Calculate mean and std
        let mean = x
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| SklearsError::InvalidInput("Cannot compute mean".to_string()))?;
        let std = x.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

        // Normalize
        let mut normalized = x.clone();
        for (i, mut col) in normalized
            .axis_iter_mut(scirs2_core::ndarray::Axis(1))
            .enumerate()
        {
            let std_val = std[i].max(1e-8); // Avoid division by zero
            for elem in col.iter_mut() {
                *elem = (*elem - mean[i]) / std_val;
            }
        }

        Ok(normalized)
    }

    fn process_before_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        self.process_before_fit(x)
    }

    fn name(&self) -> &str {
        "NormalizationMiddleware"
    }
}

/// Pipeline for composing multiple kernel approximations
pub struct Pipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    hooks: Vec<Box<dyn Hook>>,
    middleware: Vec<Arc<dyn Middleware>>,
    name: String,
    is_fitted: bool,
}

impl Pipeline {
    /// Create a new pipeline
    pub fn new(name: String) -> Self {
        Self {
            stages: Vec::new(),
            hooks: Vec::new(),
            middleware: Vec::new(),
            name,
            is_fitted: false,
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
    }

    /// Add a hook
    pub fn add_hook(&mut self, hook: Box<dyn Hook>) {
        self.hooks.push(hook);
    }

    /// Add middleware
    pub fn add_middleware(&mut self, middleware: Arc<dyn Middleware>) {
        self.middleware.push(middleware);
    }

    /// Fit the pipeline
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), SklearsError> {
        let mut current_data = x.clone();

        // Apply middleware before fit
        for mw in &self.middleware {
            current_data = mw.process_before_fit(&current_data)?;
        }

        // Fit each stage
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            let start = Instant::now();
            let mut context = HookContext::new("fit", idx, stage.name());

            // Call before fit hooks
            for hook in &mut self.hooks {
                hook.before_fit(&current_data, &mut context)?;
            }

            // Fit the stage
            match stage.fit(&current_data) {
                Ok(_) => {
                    context.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                    // Call after fit hooks
                    for hook in &mut self.hooks {
                        hook.after_fit(&current_data, &mut context)?;
                    }

                    // Transform for next stage
                    current_data = stage.transform(&current_data)?;
                }
                Err(e) => {
                    for hook in &mut self.hooks {
                        hook.on_error(&e, &mut context);
                    }
                    return Err(e);
                }
            }
        }

        // Apply middleware after fit
        for mw in &self.middleware {
            current_data = mw.process_after_fit(&current_data)?;
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Transform using the fitted pipeline
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "Pipeline must be fitted before transform".to_string(),
            });
        }

        let mut current_data = x.clone();

        // Apply middleware before transform
        for mw in &self.middleware {
            current_data = mw.process_before_transform(&current_data)?;
        }

        // Transform through each stage
        for stage in self.stages.iter() {
            // Note: Hook calls during transform would require interior mutability
            // since transform takes &self. For now, hooks are only called during fit.
            // To enable transform hooks, consider using Arc<Mutex<dyn Hook>> instead.

            // Transform
            match stage.transform(&current_data) {
                Ok(output) => {
                    current_data = output;
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        // Apply middleware after transform
        for mw in &self.middleware {
            current_data = mw.process_after_transform(&current_data)?;
        }

        Ok(current_data)
    }

    /// Get pipeline name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if pipeline is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Get number of stages
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

/// Builder for creating pipelines
pub struct PipelineBuilder {
    pipeline: Pipeline,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(name: &str) -> Self {
        Self {
            pipeline: Pipeline::new(name.to_string()),
        }
    }

    /// Add a stage
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.pipeline.add_stage(stage);
        self
    }

    /// Add a hook
    pub fn add_hook(mut self, hook: Box<dyn Hook>) -> Self {
        self.pipeline.add_hook(hook);
        self
    }

    /// Add middleware
    pub fn add_middleware(mut self, middleware: Arc<dyn Middleware>) -> Self {
        self.pipeline.add_middleware(middleware);
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline {
        self.pipeline
    }
}

/// Validation hook that checks for NaN and Inf
pub struct ValidationHook;

impl Hook for ValidationHook {
    fn after_transform(
        &mut self,
        _x: &Array2<f64>,
        output: &Array2<f64>,
        _context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        for &val in output.iter() {
            if val.is_nan() || val.is_infinite() {
                return Err(SklearsError::InvalidInput(
                    "Output contains NaN or Inf values".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "ValidationHook"
    }
}

/// Performance monitoring hook
pub struct PerformanceHook {
    timings: Vec<(String, f64)>,
}

impl PerformanceHook {
    /// Create a new performance hook
    pub fn new() -> Self {
        Self {
            timings: Vec::new(),
        }
    }

    /// Get all timings
    pub fn timings(&self) -> &[(String, f64)] {
        &self.timings
    }

    /// Get total time
    pub fn total_time(&self) -> f64 {
        self.timings.iter().map(|(_, t)| t).sum()
    }
}

impl Default for PerformanceHook {
    fn default() -> Self {
        Self::new()
    }
}

impl Hook for PerformanceHook {
    fn after_transform(
        &mut self,
        _x: &Array2<f64>,
        _output: &Array2<f64>,
        context: &mut HookContext,
    ) -> Result<(), SklearsError> {
        self.timings
            .push((context.transform_name.clone(), context.elapsed_ms));
        Ok(())
    }

    fn name(&self) -> &str {
        "PerformanceHook"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    struct DummyStage {
        name: String,
        fitted: bool,
    }

    impl DummyStage {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                fitted: false,
            }
        }
    }

    impl PipelineStage for DummyStage {
        fn fit(&mut self, _x: &Array2<f64>) -> Result<(), SklearsError> {
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
            if !self.fitted {
                return Err(SklearsError::NotFitted {
                    operation: "Stage not fitted".to_string(),
                });
            }
            Ok(x.mapv(|v| v * 2.0))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn clone_stage(&self) -> Box<dyn PipelineStage> {
            Box::new(DummyStage {
                name: self.name.clone(),
                fitted: self.fitted,
            })
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_pipeline_basic() {
        let mut pipeline = Pipeline::new("test".to_string());
        pipeline.add_stage(Box::new(DummyStage::new("stage1")));

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        pipeline.fit(&x).unwrap();

        let result = pipeline.transform(&x).unwrap();
        assert_eq!(result[[0, 0]], 2.0);
        assert_eq!(result[[1, 1]], 8.0);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Box::new(DummyStage::new("stage1")))
            .add_hook(Box::new(LoggingHook::new()))
            .build();

        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_logging_hook() {
        let mut hook = LoggingHook::new();
        let x = array![[1.0, 2.0]];
        let mut context = HookContext::new("fit", 0, "test_stage");

        hook.before_fit(&x, &mut context).unwrap();
        assert_eq!(hook.logs().len(), 1);
    }

    #[test]
    fn test_validation_hook() {
        let mut hook = ValidationHook;
        let x = array![[1.0, 2.0]];
        let output = array![[1.0, 2.0]];
        let mut context = HookContext::new("transform", 0, "test");

        assert!(hook.after_transform(&x, &output, &mut context).is_ok());

        let invalid_output = array![[f64::NAN, 2.0]];
        assert!(hook
            .after_transform(&x, &invalid_output, &mut context)
            .is_err());
    }

    #[test]
    fn test_performance_hook() {
        let mut hook = PerformanceHook::new();
        let x = array![[1.0, 2.0]];
        let output = array![[2.0, 4.0]];
        let mut context = HookContext::new("transform", 0, "test");
        context.elapsed_ms = 10.0;

        hook.after_transform(&x, &output, &mut context).unwrap();
        assert_eq!(hook.timings().len(), 1);
        assert_eq!(hook.total_time(), 10.0);
    }

    #[test]
    fn test_hook_context() {
        let mut context = HookContext::new("fit", 0, "test_stage");
        context.add_metadata("key".to_string(), "value".to_string());

        assert_eq!(context.stage, "fit");
        assert_eq!(context.transform_index, 0);
        assert!(context.metadata.contains_key("key"));
    }
}
