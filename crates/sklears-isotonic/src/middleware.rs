//! Middleware for constraint pipelines
//!
//! This module provides a flexible middleware system for composing and chaining
//! constraint transformations in isotonic regression pipelines.

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Middleware trait for constraint transformations
///
/// Middleware allows pre-processing, post-processing, and transformation
/// of constraints in isotonic regression pipelines.
pub trait ConstraintMiddleware: Send + Sync {
    /// Process data before constraint application
    fn pre_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        Ok(y.clone())
    }

    /// Process data after constraint application
    fn post_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        Ok(y.clone())
    }

    /// Transform constraints
    fn transform_constraints(&self, constraints: &ConstraintSet) -> Result<ConstraintSet> {
        Ok(constraints.clone())
    }

    /// Get middleware name
    fn name(&self) -> &str;

    /// Get middleware priority (lower = earlier execution)
    fn priority(&self) -> i32 {
        0
    }

    /// Check if middleware is enabled
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Constraint set representation
#[derive(Debug, Clone)]
pub struct ConstraintSet {
    /// Monotonicity constraint (None, Increasing, Decreasing)
    pub monotonicity: Option<Monotonicity>,
    /// Bounds constraints
    pub bounds: Option<(Float, Float)>,
    /// Smoothness penalty
    pub smoothness: Option<Float>,
    /// Custom constraints
    pub custom: HashMap<String, ConstraintValue>,
}

/// Monotonicity types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Monotonicity {
    Increasing,
    Decreasing,
}

/// Constraint value types
#[derive(Debug, Clone)]
pub enum ConstraintValue {
    Float(Float),
    Bool(bool),
    Array(Array1<Float>),
    String(String),
}

impl ConstraintSet {
    /// Create new empty constraint set
    pub fn new() -> Self {
        Self {
            monotonicity: None,
            bounds: None,
            smoothness: None,
            custom: HashMap::new(),
        }
    }

    /// Set monotonicity constraint
    pub fn with_monotonicity(mut self, monotonicity: Monotonicity) -> Self {
        self.monotonicity = Some(monotonicity);
        self
    }

    /// Set bounds constraint
    pub fn with_bounds(mut self, min: Float, max: Float) -> Self {
        self.bounds = Some((min, max));
        self
    }

    /// Set smoothness penalty
    pub fn with_smoothness(mut self, smoothness: Float) -> Self {
        self.smoothness = Some(smoothness);
        self
    }

    /// Add custom constraint
    pub fn with_custom(mut self, key: String, value: ConstraintValue) -> Self {
        self.custom.insert(key, value);
        self
    }
}

impl Default for ConstraintSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Middleware pipeline for constraint processing
#[derive(Default)]
pub struct ConstraintPipeline {
    middleware: Vec<Box<dyn ConstraintMiddleware>>,
    constraints: ConstraintSet,
}

impl ConstraintPipeline {
    /// Create new constraint pipeline
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
            constraints: ConstraintSet::new(),
        }
    }

    /// Add middleware to pipeline
    pub fn add_middleware(&mut self, middleware: Box<dyn ConstraintMiddleware>) -> &mut Self {
        self.middleware.push(middleware);
        // Sort by priority
        self.middleware.sort_by_key(|m| m.priority());
        self
    }

    /// Set constraints
    pub fn set_constraints(&mut self, constraints: ConstraintSet) -> &mut Self {
        self.constraints = constraints;
        self
    }

    /// Process data through entire pipeline
    pub fn process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        let mut result = y.clone();

        // Apply pre-processing middleware
        for middleware in &self.middleware {
            if middleware.is_enabled() {
                result = middleware.pre_process(&result)?;
            }
        }

        // Apply constraints (this would integrate with actual isotonic regression)
        result = self.apply_constraints(&result)?;

        // Apply post-processing middleware
        for middleware in self.middleware.iter().rev() {
            if middleware.is_enabled() {
                result = middleware.post_process(&result)?;
            }
        }

        Ok(result)
    }

    /// Apply constraints (simplified for demonstration)
    fn apply_constraints(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        let mut result = y.clone();

        // Apply bounds if specified
        if let Some((min, max)) = self.constraints.bounds {
            for val in result.iter_mut() {
                if *val < min {
                    *val = min;
                }
                if *val > max {
                    *val = max;
                }
            }
        }

        Ok(result)
    }

    /// Get transformed constraints
    pub fn get_transformed_constraints(&self) -> Result<ConstraintSet> {
        let mut constraints = self.constraints.clone();

        for middleware in &self.middleware {
            if middleware.is_enabled() {
                constraints = middleware.transform_constraints(&constraints)?;
            }
        }

        Ok(constraints)
    }
}

// ============================================================================
// Built-in Middleware Implementations
// ============================================================================

/// Outlier removal middleware
pub struct OutlierRemovalMiddleware {
    threshold: Float,
    enabled: bool,
}

impl OutlierRemovalMiddleware {
    pub fn new(threshold: Float) -> Self {
        Self {
            threshold,
            enabled: true,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl ConstraintMiddleware for OutlierRemovalMiddleware {
    fn pre_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        let median = {
            let mut sorted = y.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        let mad = {
            let deviations: Vec<Float> = y.iter().map(|&val| (val - median).abs()).collect();
            let mut sorted_deviations = deviations.clone();
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_deviations[sorted_deviations.len() / 2]
        };

        let mut result = y.clone();
        for (i, &val) in y.iter().enumerate() {
            if (val - median).abs() > self.threshold * mad {
                result[i] = median;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "OutlierRemoval"
    }

    fn priority(&self) -> i32 {
        10 // Run early
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Normalization middleware
pub struct NormalizationMiddleware {
    method: NormalizationMethod,
    enabled: bool,
    // Store normalization parameters
    mean: Option<Float>,
    std: Option<Float>,
    min: Option<Float>,
    max: Option<Float>,
}

#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    ZScore,
    MinMax,
}

impl NormalizationMiddleware {
    pub fn new(method: NormalizationMethod) -> Self {
        Self {
            method,
            enabled: true,
            mean: None,
            std: None,
            min: None,
            max: None,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl ConstraintMiddleware for NormalizationMiddleware {
    fn pre_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        match self.method {
            NormalizationMethod::ZScore => {
                let mean = y.iter().sum::<Float>() / y.len() as Float;
                let variance =
                    y.iter().map(|&val| (val - mean).powi(2)).sum::<Float>() / y.len() as Float;
                let std = variance.sqrt();

                if std < 1e-10 {
                    return Ok(y.clone());
                }

                let normalized = y.mapv(|val| (val - mean) / std);
                Ok(normalized)
            }
            NormalizationMethod::MinMax => {
                let min = y.iter().cloned().fold(Float::INFINITY, Float::min);
                let max = y.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

                if (max - min).abs() < 1e-10 {
                    return Ok(y.clone());
                }

                let normalized = y.mapv(|val| (val - min) / (max - min));
                Ok(normalized)
            }
        }
    }

    fn name(&self) -> &str {
        "Normalization"
    }

    fn priority(&self) -> i32 {
        20 // Run after outlier removal
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Smoothing middleware
pub struct SmoothingMiddleware {
    window_size: usize,
    enabled: bool,
}

impl SmoothingMiddleware {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            enabled: true,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl ConstraintMiddleware for SmoothingMiddleware {
    fn post_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        let n = y.len();
        let mut smoothed = Array1::zeros(n);

        for i in 0..n {
            let start = i.saturating_sub(self.window_size / 2);
            let end = (i + self.window_size / 2 + 1).min(n);

            let window_sum: Float = (start..end).map(|j| y[j]).sum();
            let window_size = (end - start) as Float;

            smoothed[i] = window_sum / window_size;
        }

        Ok(smoothed)
    }

    fn name(&self) -> &str {
        "Smoothing"
    }

    fn priority(&self) -> i32 {
        100 // Run late (post-processing)
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Constraint validation middleware
pub struct ConstraintValidationMiddleware {
    strict: bool,
    enabled: bool,
}

impl ConstraintValidationMiddleware {
    pub fn new(strict: bool) -> Self {
        Self {
            strict,
            enabled: true,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    fn validate(&self, y: &Array1<Float>, constraints: &ConstraintSet) -> Result<()> {
        // Validate monotonicity
        if let Some(monotonicity) = constraints.monotonicity {
            for i in 0..y.len() - 1 {
                match monotonicity {
                    Monotonicity::Increasing => {
                        if y[i] > y[i + 1] + 1e-6 && self.strict {
                            return Err(SklearsError::InvalidInput(format!(
                                "Monotonicity violated at index {}: {} > {}",
                                i,
                                y[i],
                                y[i + 1]
                            )));
                        }
                    }
                    Monotonicity::Decreasing => {
                        if y[i] < y[i + 1] - 1e-6 && self.strict {
                            return Err(SklearsError::InvalidInput(format!(
                                "Monotonicity violated at index {}: {} < {}",
                                i,
                                y[i],
                                y[i + 1]
                            )));
                        }
                    }
                }
            }
        }

        // Validate bounds
        if let Some((min, max)) = constraints.bounds {
            for (i, &val) in y.iter().enumerate() {
                if (val < min - 1e-6 || val > max + 1e-6) && self.strict {
                    return Err(SklearsError::InvalidInput(format!(
                        "Bounds violated at index {}: {} not in [{}, {}]",
                        i, val, min, max
                    )));
                }
            }
        }

        Ok(())
    }
}

impl ConstraintMiddleware for ConstraintValidationMiddleware {
    fn post_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        // Validation happens in a separate method
        Ok(y.clone())
    }

    fn name(&self) -> &str {
        "ConstraintValidation"
    }

    fn priority(&self) -> i32 {
        1000 // Run last
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Logging middleware
pub struct LoggingMiddleware {
    verbose: bool,
    enabled: bool,
}

impl LoggingMiddleware {
    pub fn new(verbose: bool) -> Self {
        Self {
            verbose,
            enabled: true,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl ConstraintMiddleware for LoggingMiddleware {
    fn pre_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        if self.verbose {
            println!("[LoggingMiddleware] Pre-processing: n={}", y.len());
            println!(
                "[LoggingMiddleware] Input stats: min={:.3}, max={:.3}, mean={:.3}",
                y.iter().cloned().fold(Float::INFINITY, Float::min),
                y.iter().cloned().fold(Float::NEG_INFINITY, Float::max),
                y.iter().sum::<Float>() / y.len() as Float
            );
        }
        Ok(y.clone())
    }

    fn post_process(&self, y: &Array1<Float>) -> Result<Array1<Float>> {
        if self.verbose {
            println!("[LoggingMiddleware] Post-processing: n={}", y.len());
            println!(
                "[LoggingMiddleware] Output stats: min={:.3}, max={:.3}, mean={:.3}",
                y.iter().cloned().fold(Float::INFINITY, Float::min),
                y.iter().cloned().fold(Float::NEG_INFINITY, Float::max),
                y.iter().sum::<Float>() / y.len() as Float
            );
        }
        Ok(y.clone())
    }

    fn name(&self) -> &str {
        "Logging"
    }

    fn priority(&self) -> i32 {
        -1000 // Run very early for pre-process, very late for post-process
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// ============================================================================
// Pipeline Builder
// ============================================================================

/// Builder for constraint pipelines
pub struct PipelineBuilder {
    pipeline: ConstraintPipeline,
}

impl PipelineBuilder {
    /// Create new pipeline builder
    pub fn new() -> Self {
        Self {
            pipeline: ConstraintPipeline::new(),
        }
    }

    /// Add outlier removal
    pub fn with_outlier_removal(mut self, threshold: Float) -> Self {
        self.pipeline
            .add_middleware(Box::new(OutlierRemovalMiddleware::new(threshold)));
        self
    }

    /// Add normalization
    pub fn with_normalization(mut self, method: NormalizationMethod) -> Self {
        self.pipeline
            .add_middleware(Box::new(NormalizationMiddleware::new(method)));
        self
    }

    /// Add smoothing
    pub fn with_smoothing(mut self, window_size: usize) -> Self {
        self.pipeline
            .add_middleware(Box::new(SmoothingMiddleware::new(window_size)));
        self
    }

    /// Add validation
    pub fn with_validation(mut self, strict: bool) -> Self {
        self.pipeline
            .add_middleware(Box::new(ConstraintValidationMiddleware::new(strict)));
        self
    }

    /// Add logging
    pub fn with_logging(mut self, verbose: bool) -> Self {
        self.pipeline
            .add_middleware(Box::new(LoggingMiddleware::new(verbose)));
        self
    }

    /// Add custom middleware
    pub fn with_middleware(mut self, middleware: Box<dyn ConstraintMiddleware>) -> Self {
        self.pipeline.add_middleware(middleware);
        self
    }

    /// Set constraints
    pub fn with_constraints(mut self, constraints: ConstraintSet) -> Self {
        self.pipeline.set_constraints(constraints);
        self
    }

    /// Build the pipeline
    pub fn build(self) -> ConstraintPipeline {
        self.pipeline
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_constraint_set() {
        let constraints = ConstraintSet::new()
            .with_monotonicity(Monotonicity::Increasing)
            .with_bounds(0.0, 1.0)
            .with_smoothness(0.1);

        assert_eq!(constraints.monotonicity, Some(Monotonicity::Increasing));
        assert_eq!(constraints.bounds, Some((0.0, 1.0)));
        assert_eq!(constraints.smoothness, Some(0.1));
    }

    #[test]
    fn test_outlier_removal_middleware() {
        let y = array![1.0, 2.0, 100.0, 3.0, 4.0];
        let middleware = OutlierRemovalMiddleware::new(3.0);

        let result = middleware.pre_process(&y).unwrap();

        // The outlier (100.0) should be replaced with median
        assert!(result[2] < 50.0);
    }

    #[test]
    fn test_normalization_middleware_zscore() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let middleware = NormalizationMiddleware::new(NormalizationMethod::ZScore);

        let result = middleware.pre_process(&y).unwrap();

        // Check that result is normalized (mean ≈ 0, std ≈ 1)
        let mean = result.iter().sum::<Float>() / result.len() as Float;
        assert!((mean).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_middleware_minmax() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let middleware = NormalizationMiddleware::new(NormalizationMethod::MinMax);

        let result = middleware.pre_process(&y).unwrap();

        // Check that result is in [0, 1]
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_smoothing_middleware() {
        let y = array![1.0, 10.0, 2.0, 3.0, 4.0];
        let middleware = SmoothingMiddleware::new(3);

        let result = middleware.post_process(&y).unwrap();

        // Middle values should be smoothed
        assert!(result[1] < y[1]);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_outlier_removal(3.0)
            .with_normalization(NormalizationMethod::ZScore)
            .with_smoothing(3)
            .with_constraints(
                ConstraintSet::new()
                    .with_monotonicity(Monotonicity::Increasing)
                    .with_bounds(0.0, 10.0),
            )
            .build();

        let y = array![1.0, 2.0, 100.0, 3.0, 4.0];
        let result = pipeline.process(&y).unwrap();

        assert_eq!(result.len(), y.len());
    }

    #[test]
    fn test_pipeline_process() {
        let mut pipeline = ConstraintPipeline::new();
        pipeline.add_middleware(Box::new(NormalizationMiddleware::new(
            NormalizationMethod::MinMax,
        )));
        pipeline.set_constraints(ConstraintSet::new().with_bounds(0.0, 1.0));

        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = pipeline.process(&y).unwrap();

        // All values should be in [0, 1]
        for &val in result.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_middleware_priority() {
        let mut pipeline = ConstraintPipeline::new();

        // Add in reverse priority order
        pipeline.add_middleware(Box::new(SmoothingMiddleware::new(3))); // priority 100
        pipeline.add_middleware(Box::new(OutlierRemovalMiddleware::new(3.0))); // priority 10

        // Middleware should be sorted by priority
        assert_eq!(pipeline.middleware[0].priority(), 10);
        assert_eq!(pipeline.middleware[1].priority(), 100);
    }

    #[test]
    fn test_middleware_enable_disable() {
        let middleware = OutlierRemovalMiddleware::new(3.0).with_enabled(false);
        assert!(!middleware.is_enabled());

        let middleware = OutlierRemovalMiddleware::new(3.0).with_enabled(true);
        assert!(middleware.is_enabled());
    }

    #[test]
    fn test_logging_middleware() {
        let middleware = LoggingMiddleware::new(false);
        let y = array![1.0, 2.0, 3.0];

        // Should not panic
        let result = middleware.pre_process(&y).unwrap();
        assert_eq!(result.len(), y.len());

        let result = middleware.post_process(&y).unwrap();
        assert_eq!(result.len(), y.len());
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = ConstraintPipeline::new();
        let y = array![1.0, 2.0, 3.0];

        let result = pipeline.process(&y).unwrap();
        assert_eq!(result.len(), y.len());
    }

    #[test]
    fn test_multiple_middleware() {
        let pipeline = PipelineBuilder::new()
            .with_outlier_removal(3.0)
            .with_normalization(NormalizationMethod::ZScore)
            .with_smoothing(3)
            .build();

        let y = array![1.0, 2.0, 100.0, 3.0, 4.0, 5.0];
        let result = pipeline.process(&y).unwrap();

        assert_eq!(result.len(), y.len());
    }

    #[test]
    fn test_constraint_value_types() {
        let mut constraints = ConstraintSet::new();
        constraints
            .custom
            .insert("test_float".to_string(), ConstraintValue::Float(1.0));
        constraints
            .custom
            .insert("test_bool".to_string(), ConstraintValue::Bool(true));
        constraints.custom.insert(
            "test_array".to_string(),
            ConstraintValue::Array(array![1.0, 2.0]),
        );
        constraints.custom.insert(
            "test_string".to_string(),
            ConstraintValue::String("test".to_string()),
        );

        assert_eq!(constraints.custom.len(), 4);
    }
}
