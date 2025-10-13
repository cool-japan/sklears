//! Middleware system for manifold learning pipelines
//!
//! This module provides a flexible middleware system for composing manifold learning
//! algorithms, preprocessing steps, and post-processing operations in a pipeline.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, SVD, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Pipeline context that passes through middleware layers
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Data flowing through the pipeline
    pub data: Array2<f64>,
    /// Metadata accumulated during pipeline execution
    pub metadata: PipelineMetadata,
    /// Parameters that can be modified by middleware
    pub parameters: HashMap<String, ParameterValue>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

/// Metadata accumulated during pipeline execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PipelineMetadata {
    /// Original data shape
    pub original_shape: (usize, usize),
    /// Current data shape
    pub current_shape: (usize, usize),
    /// Transformations applied
    pub transformations: Vec<TransformationInfo>,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Warnings and notes
    pub warnings: Vec<String>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Information about a transformation step
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TransformationInfo {
    /// Name of the transformation
    pub name: String,
    /// Parameters used
    pub parameters: HashMap<String, String>,
    /// Execution time
    pub execution_time: f64,
    /// Input shape
    pub input_shape: (usize, usize),
    /// Output shape
    pub output_shape: (usize, usize),
    /// Quality metrics for this step
    pub metrics: HashMap<String, f64>,
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Total execution time
    pub total_time: Duration,
    /// Memory usage statistics
    pub memory_usage: MemoryStats,
    /// Step-by-step timing
    pub step_times: Vec<Duration>,
    /// Performance counters
    pub counters: HashMap<String, u64>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_memory: u64,
    /// Current memory usage (bytes)
    pub current_memory: u64,
    /// Memory allocations
    pub allocations: u64,
    /// Memory deallocations
    pub deallocations: u64,
}

/// Parameter value types for pipeline configuration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParameterValue {
    /// Int
    Int(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
    /// Bool
    Bool(bool),
    /// IntArray
    IntArray(Vec<i64>),
    /// FloatArray
    FloatArray(Vec<f64>),
}

/// Trait for pipeline middleware components
pub trait PipelineMiddleware: Send + Sync + Debug {
    /// Process the pipeline context
    fn process(&self, context: PipelineContext) -> SklResult<PipelineContext>;

    /// Get middleware name
    fn name(&self) -> &str;

    /// Get middleware description
    fn description(&self) -> &str {
        "No description provided"
    }

    /// Check if middleware can be applied to the current context
    fn can_apply(&self, context: &PipelineContext) -> bool {
        true
    }

    /// Get middleware configuration
    fn configuration(&self) -> MiddlewareConfig {
        MiddlewareConfig::default()
    }
}

/// Middleware configuration
#[derive(Debug, Clone, Default)]
pub struct MiddlewareConfig {
    /// Whether this middleware is optional
    pub optional: bool,
    /// Whether to continue on errors
    pub continue_on_error: bool,
    /// Timeout for execution
    pub timeout: Option<Duration>,
    /// Priority (higher values execute first)
    pub priority: i32,
}

/// Manifold learning pipeline
pub struct ManifoldPipeline {
    middleware: Vec<Arc<dyn PipelineMiddleware>>,
    config: PipelineConfig,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Whether to enable detailed logging
    pub verbose: bool,
    /// Whether to collect performance metrics
    pub collect_metrics: bool,
    /// Whether to enable parallel execution
    pub parallel: bool,
    /// Maximum execution time for the entire pipeline
    pub max_execution_time: Option<Duration>,
    /// Whether to stop on first error
    pub fail_fast: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            collect_metrics: true,
            parallel: false,
            max_execution_time: None,
            fail_fast: true,
        }
    }
}

impl Default for ManifoldPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldPipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    /// Add middleware to the pipeline
    pub fn add_middleware(mut self, middleware: Arc<dyn PipelineMiddleware>) -> Self {
        self.middleware.push(middleware);
        self
    }

    /// Set pipeline configuration
    pub fn config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Execute the pipeline
    pub fn execute(&self, data: &ArrayView2<'_, Float>) -> SklResult<PipelineResult> {
        let start_time = Instant::now();

        // Initialize context
        let data_f64 = data.mapv(|x| x);
        let original_shape = data_f64.dim();

        let mut context = PipelineContext {
            data: data_f64,
            metadata: PipelineMetadata {
                original_shape,
                current_shape: original_shape,
                transformations: Vec::new(),
                quality_metrics: HashMap::new(),
                warnings: Vec::new(),
                custom: HashMap::new(),
            },
            parameters: HashMap::new(),
            stats: ExecutionStats {
                total_time: Duration::from_secs(0),
                memory_usage: MemoryStats {
                    peak_memory: 0,
                    current_memory: 0,
                    allocations: 0,
                    deallocations: 0,
                },
                step_times: Vec::new(),
                counters: HashMap::new(),
            },
        };

        // Sort middleware by priority
        let mut sorted_middleware = self.middleware.clone();
        sorted_middleware
            .sort_by(|a, b| b.configuration().priority.cmp(&a.configuration().priority));

        // Execute middleware in order
        for middleware in &sorted_middleware {
            if !middleware.can_apply(&context) {
                if self.config.verbose {
                    println!(
                        "Skipping middleware '{}' - not applicable",
                        middleware.name()
                    );
                }
                continue;
            }

            let step_start = Instant::now();

            let context_for_middleware = context.clone();
            let step_result = self.execute_middleware(middleware.as_ref(), context_for_middleware);

            match step_result {
                Ok(new_context) => {
                    context = new_context;
                    let step_time = step_start.elapsed();
                    context.stats.step_times.push(step_time);

                    if self.config.verbose {
                        println!(
                            "Completed middleware '{}' in {:?}",
                            middleware.name(),
                            step_time
                        );
                    }
                }
                Err(e) => {
                    let middleware_config = middleware.configuration();

                    if middleware_config.continue_on_error && !self.config.fail_fast {
                        context.metadata.warnings.push(format!(
                            "Middleware '{}' failed: {}",
                            middleware.name(),
                            e
                        ));

                        if self.config.verbose {
                            println!(
                                "Warning: Middleware '{}' failed but continuing: {}",
                                middleware.name(),
                                e
                            );
                        }
                    } else {
                        return Err(SklearsError::InvalidInput(format!(
                            "Pipeline failed at middleware '{}': {}",
                            middleware.name(),
                            e
                        )));
                    }
                }
            }

            // Check timeout
            if let Some(max_time) = self.config.max_execution_time {
                if start_time.elapsed() > max_time {
                    return Err(SklearsError::InvalidInput(
                        "Pipeline execution timed out".to_string(),
                    ));
                }
            }
        }

        // Finalize context
        context.stats.total_time = start_time.elapsed();
        context.metadata.current_shape = context.data.dim();

        Ok(PipelineResult {
            data: context.data,
            metadata: context.metadata,
            execution_stats: context.stats,
        })
    }

    /// Execute a single middleware component
    fn execute_middleware(
        &self,
        middleware: &dyn PipelineMiddleware,
        context: PipelineContext,
    ) -> SklResult<PipelineContext> {
        let start_time = Instant::now();
        let config = middleware.configuration();

        // Apply timeout if specified
        let result = if let Some(timeout) = config.timeout {
            // Simplified timeout handling - in a real implementation,
            // you might want to use async/await or spawn a thread
            let result = middleware.process(context);

            if start_time.elapsed() > timeout {
                return Err(SklearsError::InvalidInput(format!(
                    "Middleware '{}' timed out after {:?}",
                    middleware.name(),
                    timeout
                )));
            }

            result
        } else {
            middleware.process(context)
        };

        result
    }

    /// Get pipeline summary
    pub fn summary(&self) -> PipelineSummary {
        /// PipelineSummary
        PipelineSummary {
            middleware_count: self.middleware.len(),
            middleware_names: self
                .middleware
                .iter()
                .map(|m| m.name().to_string())
                .collect(),
            config: self.config.clone(),
        }
    }
}

/// Result of pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Final transformed data
    pub data: Array2<f64>,
    /// Pipeline metadata
    pub metadata: PipelineMetadata,
    /// Execution statistics
    pub execution_stats: ExecutionStats,
}

/// Pipeline summary information
#[derive(Debug, Clone)]
pub struct PipelineSummary {
    /// Number of middleware components
    pub middleware_count: usize,
    /// Names of middleware components
    pub middleware_names: Vec<String>,
    /// Pipeline configuration
    pub config: PipelineConfig,
}

// Built-in middleware implementations

/// Data validation middleware
#[derive(Debug)]
pub struct DataValidationMiddleware {
    min_samples: usize,
    max_samples: usize,
    min_features: usize,
    max_features: usize,
    check_finite: bool,
    check_duplicates: bool,
}

impl Default for DataValidationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl DataValidationMiddleware {
    pub fn new() -> Self {
        Self {
            min_samples: 1,
            max_samples: usize::MAX,
            min_features: 1,
            max_features: usize::MAX,
            check_finite: true,
            check_duplicates: false,
        }
    }

    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    pub fn max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    pub fn min_features(mut self, min_features: usize) -> Self {
        self.min_features = min_features;
        self
    }

    pub fn max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn check_finite(mut self, check_finite: bool) -> Self {
        self.check_finite = check_finite;
        self
    }

    pub fn check_duplicates(mut self, check_duplicates: bool) -> Self {
        self.check_duplicates = check_duplicates;
        self
    }
}

impl PipelineMiddleware for DataValidationMiddleware {
    fn process(&self, mut context: PipelineContext) -> SklResult<PipelineContext> {
        let (n_samples, n_features) = context.data.dim();

        // Check sample count
        if n_samples < self.min_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Too few samples: {} < {}",
                n_samples, self.min_samples
            )));
        }
        if n_samples > self.max_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Too many samples: {} > {}",
                n_samples, self.max_samples
            )));
        }

        // Check feature count
        if n_features < self.min_features {
            return Err(SklearsError::InvalidInput(format!(
                "Too few features: {} < {}",
                n_features, self.min_features
            )));
        }
        if n_features > self.max_features {
            return Err(SklearsError::InvalidInput(format!(
                "Too many features: {} > {}",
                n_features, self.max_features
            )));
        }

        // Check for finite values
        if self.check_finite && !context.data.iter().all(|&x| x.is_finite()) {
            return Err(SklearsError::InvalidInput(
                "Data contains non-finite values (NaN or infinity)".to_string(),
            ));
        }

        // Check for duplicate rows
        if self.check_duplicates {
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    let row_i = context.data.row(i);
                    let row_j = context.data.row(j);

                    if row_i
                        .iter()
                        .zip(row_j.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-15)
                    {
                        context
                            .metadata
                            .warnings
                            .push(format!("Duplicate rows found: {} and {}", i, j));
                    }
                }
            }
        }

        // Add validation info to metadata
        context
            .metadata
            .quality_metrics
            .insert("n_samples".to_string(), n_samples as f64);
        context
            .metadata
            .quality_metrics
            .insert("n_features".to_string(), n_features as f64);

        Ok(context)
    }

    fn name(&self) -> &str {
        "DataValidation"
    }

    fn description(&self) -> &str {
        "Validates input data for common issues"
    }
}

/// Data standardization middleware
#[derive(Debug)]
pub struct StandardizationMiddleware {
    with_mean: bool,
    with_std: bool,
}

impl Default for StandardizationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardizationMiddleware {
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }

    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }
}

impl PipelineMiddleware for StandardizationMiddleware {
    fn process(&self, mut context: PipelineContext) -> SklResult<PipelineContext> {
        let (n_samples, n_features) = context.data.dim();

        if self.with_mean || self.with_std {
            let means = if self.with_mean {
                context
                    .data
                    .mean_axis(scirs2_core::ndarray::Axis(0))
                    .unwrap()
            } else {
                Array1::zeros(n_features)
            };

            let stds = if self.with_std {
                context.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0)
            } else {
                Array1::ones(n_features)
            };

            // Standardize data
            for i in 0..n_samples {
                for j in 0..n_features {
                    if stds[j] > 1e-15 {
                        context.data[[i, j]] = (context.data[[i, j]] - means[j]) / stds[j];
                    } else {
                        context.data[[i, j]] -= means[j];
                    }
                }
            }

            // Store standardization parameters
            context
                .metadata
                .custom
                .insert("means".to_string(), format!("{:?}", means));
            context
                .metadata
                .custom
                .insert("stds".to_string(), format!("{:?}", stds));
        }

        // Add transformation info
        context.metadata.transformations.push(TransformationInfo {
            name: "Standardization".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("with_mean".to_string(), self.with_mean.to_string());
                params.insert("with_std".to_string(), self.with_std.to_string());
                params
            },
            execution_time: 0.0, // Would be filled by pipeline
            input_shape: (n_samples, n_features),
            output_shape: (n_samples, n_features),
            metrics: HashMap::new(),
        });

        Ok(context)
    }

    fn name(&self) -> &str {
        "Standardization"
    }

    fn description(&self) -> &str {
        "Standardizes data by removing mean and scaling to unit variance"
    }
}

/// Quality assessment middleware
#[derive(Debug)]
pub struct QualityAssessmentMiddleware {
    compute_intrinsic_dim: bool,
    compute_clustering_metrics: bool,
}

impl Default for QualityAssessmentMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityAssessmentMiddleware {
    pub fn new() -> Self {
        Self {
            compute_intrinsic_dim: true,
            compute_clustering_metrics: false,
        }
    }

    pub fn compute_intrinsic_dim(mut self, compute: bool) -> Self {
        self.compute_intrinsic_dim = compute;
        self
    }

    pub fn compute_clustering_metrics(mut self, compute: bool) -> Self {
        self.compute_clustering_metrics = compute;
        self
    }
}

impl PipelineMiddleware for QualityAssessmentMiddleware {
    fn process(&self, mut context: PipelineContext) -> SklResult<PipelineContext> {
        // Estimate intrinsic dimensionality
        if self.compute_intrinsic_dim {
            let intrinsic_dim = estimate_intrinsic_dimensionality(&context.data);
            context
                .metadata
                .quality_metrics
                .insert("estimated_intrinsic_dim".to_string(), intrinsic_dim);
        }

        // Compute data statistics
        let variance = context
            .data
            .var_axis(scirs2_core::ndarray::Axis(0), 0.0)
            .sum();
        context
            .metadata
            .quality_metrics
            .insert("total_variance".to_string(), variance);

        let condition_number = estimate_condition_number(&context.data);
        context
            .metadata
            .quality_metrics
            .insert("condition_number".to_string(), condition_number);

        Ok(context)
    }

    fn name(&self) -> &str {
        "QualityAssessment"
    }

    fn description(&self) -> &str {
        "Assesses data quality and computes various metrics"
    }
}

// Helper functions

/// Estimate intrinsic dimensionality using PCA
fn estimate_intrinsic_dimensionality(data: &Array2<f64>) -> f64 {
    // Simple estimation using explained variance ratio
    let (n_samples, n_features) = data.dim();
    if n_samples == 0 || n_features == 0 {
        return 0.0;
    }

    // Center the data
    let mean = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
    let centered = data - &mean;

    // Compute covariance matrix
    let cov_matrix = centered.t().dot(&centered) / (n_samples - 1) as f64;

    if let Ok((eigenvalues, _)) = cov_matrix.eigh(UPLO::Lower) {
        let total_var: f64 = eigenvalues.iter().filter(|&&x| x > 0.0).sum();
        let mut cumsum = 0.0;
        let mut count = 0;

        // Count dimensions needed to explain 95% of variance
        for &val in eigenvalues.iter().rev() {
            if val > 0.0 {
                cumsum += val;
                count += 1;
                if cumsum / total_var >= 0.95 {
                    break;
                }
            }
        }

        count as f64
    } else {
        data.ncols() as f64
    }
}

/// Estimate condition number
fn estimate_condition_number(data: &Array2<f64>) -> f64 {
    if let Ok((_, singular_values, _)) = data.svd(false, false) {
        if let (Some(&max_sv), Some(&min_sv)) = (
            singular_values
                .iter()
                .filter(|&&x| x > 0.0)
                .max_by(|a, b| a.partial_cmp(b).unwrap()),
            singular_values
                .iter()
                .filter(|&&x| x > 0.0)
                .min_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            max_sv / min_sv
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_pipeline_execution() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64).collect()).unwrap();

        let pipeline = ManifoldPipeline::new()
            .add_middleware(Arc::new(DataValidationMiddleware::new().min_samples(5)))
            .add_middleware(Arc::new(StandardizationMiddleware::new()))
            .add_middleware(Arc::new(QualityAssessmentMiddleware::new()));

        let result = pipeline.execute(&data.view()).unwrap();

        assert_eq!(result.data.dim(), (10, 3));
        assert!(!result.metadata.transformations.is_empty());
        assert!(!result.metadata.quality_metrics.is_empty());
    }

    #[test]
    fn test_data_validation_middleware() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let context = PipelineContext {
            data,
            metadata: PipelineMetadata {
                original_shape: (2, 2),
                current_shape: (2, 2),
                transformations: Vec::new(),
                quality_metrics: HashMap::new(),
                warnings: Vec::new(),
                custom: HashMap::new(),
            },
            parameters: HashMap::new(),
            stats: ExecutionStats {
                total_time: Duration::from_secs(0),
                memory_usage: MemoryStats {
                    peak_memory: 0,
                    current_memory: 0,
                    allocations: 0,
                    deallocations: 0,
                },
                step_times: Vec::new(),
                counters: HashMap::new(),
            },
        };

        let middleware = DataValidationMiddleware::new()
            .min_samples(1)
            .max_samples(10);
        let result = middleware.process(context).unwrap();

        assert_eq!(result.metadata.quality_metrics.get("n_samples"), Some(&2.0));
        assert_eq!(
            result.metadata.quality_metrics.get("n_features"),
            Some(&2.0)
        );
    }

    #[test]
    fn test_standardization_middleware() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let context = PipelineContext {
            data,
            metadata: PipelineMetadata {
                original_shape: (4, 2),
                current_shape: (4, 2),
                transformations: Vec::new(),
                quality_metrics: HashMap::new(),
                warnings: Vec::new(),
                custom: HashMap::new(),
            },
            parameters: HashMap::new(),
            stats: ExecutionStats {
                total_time: Duration::from_secs(0),
                memory_usage: MemoryStats {
                    peak_memory: 0,
                    current_memory: 0,
                    allocations: 0,
                    deallocations: 0,
                },
                step_times: Vec::new(),
                counters: HashMap::new(),
            },
        };

        let middleware = StandardizationMiddleware::new();
        let result = middleware.process(context).unwrap();

        // Check that data is approximately standardized
        let means = result
            .data
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .unwrap();
        let stds = result.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

        for &mean in means.iter() {
            assert!((mean.abs()) < 1e-10);
        }
        for &std in stds.iter() {
            assert!((std - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pipeline_summary() {
        let pipeline = ManifoldPipeline::new()
            .add_middleware(Arc::new(DataValidationMiddleware::new()))
            .add_middleware(Arc::new(StandardizationMiddleware::new()));

        let summary = pipeline.summary();

        assert_eq!(summary.middleware_count, 2);
        assert!(summary
            .middleware_names
            .contains(&"DataValidation".to_string()));
        assert!(summary
            .middleware_names
            .contains(&"Standardization".to_string()));
    }
}
