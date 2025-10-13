//! Fluent API and advanced builder patterns for pipeline construction
//!
//! This module provides a fluent, chainable API for building complex machine learning
//! pipelines with type safety, method chaining, and configuration presets.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use sklears_core::{
    error::Result as SklResult,
    prelude::Fit,
    traits::{Estimator, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{
    ParallelConfig, ParallelExecutionStrategy, Pipeline, PipelinePredictor, PipelineStep,
    SimdConfig,
};

/// Helper function to create high-performance SIMD configuration
fn create_high_performance_simd_config() -> SimdConfig {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        /// SimdConfig
        SimdConfig {
            use_avx2: true,
            use_avx512: is_x86_feature_detected!("avx512f"),
            use_fma: true,
            vector_width: if is_x86_feature_detected!("avx512f") {
                16
            } else {
                8
            },
            alignment: 64,
            simd_threshold: 32,
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        /// SimdConfig
        SimdConfig {
            use_avx2: false,
            use_avx512: false,
            use_fma: false,
            vector_width: 4,
            alignment: 64,
            simd_threshold: 32,
        }
    }
}

/// Fluent pipeline builder with advanced chaining capabilities
#[derive(Debug)]
pub struct FluentPipelineBuilder<State = BuilderEmpty> {
    /// Builder state
    state: PhantomData<State>,
    /// Pipeline steps
    steps: Vec<(String, Box<dyn PipelineStep>)>,
    /// Final estimator
    estimator: Option<Box<dyn PipelinePredictor>>,
    /// Configuration options
    config: PipelineConfiguration,
    /// Validation rules
    validators: Vec<ValidationRule>,
    /// Presets applied
    presets: Vec<String>,
}

/// Builder state types for type safety
#[derive(Debug)]
pub struct BuilderEmpty;

#[derive(Debug)]
pub struct BuilderWithSteps;

#[derive(Debug)]
pub struct BuilderWithEstimator;

#[derive(Debug)]
pub struct BuilderComplete;

/// Pipeline configuration
#[derive(Debug, Clone, Default)]
pub struct PipelineConfiguration {
    /// Parallel execution config
    pub parallel: Option<ParallelConfig>,
    /// SIMD optimization config
    pub simd: Option<SimdConfig>,
    /// Execution strategy
    pub execution_strategy: Option<ParallelExecutionStrategy>,
    /// Memory optimization settings
    pub memory_config: MemoryConfiguration,
    /// Caching settings
    pub caching: CachingConfiguration,
    /// Validation settings
    pub validation: ValidationConfiguration,
    /// Debug settings
    pub debug: DebugConfiguration,
}

/// Memory configuration
#[derive(Debug, Clone)]
pub struct MemoryConfiguration {
    /// Use memory-efficient operations
    pub efficient_ops: bool,
    /// Chunk size for large datasets
    pub chunk_size: Option<usize>,
    /// Memory limit (MB)
    pub memory_limit_mb: Option<usize>,
    /// Garbage collection frequency
    pub gc_frequency: Option<usize>,
}

/// Caching configuration
#[derive(Debug, Clone)]
pub struct CachingConfiguration {
    /// Enable intermediate result caching
    pub enabled: bool,
    /// Cache directory
    pub cache_dir: Option<String>,
    /// Maximum cache size (MB)
    pub max_size_mb: Option<usize>,
    /// Cache TTL (seconds)
    pub ttl_sec: Option<usize>,
    /// Cache strategy
    pub strategy: CacheStrategy,
}

/// Cache strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TimeExpire,
    /// Size-based eviction
    SizeBased,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfiguration {
    /// Enable input validation
    pub validate_input: bool,
    /// Enable output validation
    pub validate_output: bool,
    /// Enable pipeline structure validation
    pub validate_structure: bool,
    /// Validation level
    pub level: ValidationLevel,
}

/// Validation level
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationLevel {
    /// No validation
    None,
    /// Basic validation
    Basic,
    /// Comprehensive validation
    Comprehensive,
    /// Strict validation with type checking
    Strict,
}

/// Debug configuration
#[derive(Debug, Clone)]
pub struct DebugConfiguration {
    /// Enable debug mode
    pub enabled: bool,
    /// Log level
    pub log_level: LogLevel,
    /// Profiling enabled
    pub profiling: bool,
    /// Trace execution
    pub tracing: bool,
}

/// Log level
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Validation rule for pipeline construction
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Validation function
    pub validator: Box<dyn Fn(&FluentPipelineBuilder<BuilderComplete>) -> SklResult<()>>,
}

impl std::fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("validator", &"<function>")
            .finish()
    }
}

/// Configuration preset for common pipeline patterns
#[derive(Debug, Clone)]
pub struct ConfigurationPreset {
    /// Preset name
    pub name: String,
    /// Description
    pub description: String,
    /// Configuration
    pub config: PipelineConfiguration,
    /// Default steps
    pub default_steps: Vec<PresetStep>,
}

/// Preset step definition
#[derive(Debug, Clone)]
pub struct PresetStep {
    /// Step name
    pub name: String,
    /// Step type
    pub step_type: String,
    /// Parameters
    pub parameters: HashMap<String, PresetParameter>,
}

/// Preset parameter
#[derive(Debug, Clone)]
pub enum PresetParameter {
    /// Float
    Float(f64),
    /// Int
    Int(i64),
    /// Bool
    Bool(bool),
    /// String
    String(String),
    /// Array
    Array(Vec<PresetParameter>),
}

impl Default for MemoryConfiguration {
    fn default() -> Self {
        Self {
            efficient_ops: true,
            chunk_size: Some(10000),
            memory_limit_mb: None,
            gc_frequency: Some(100),
        }
    }
}

impl Default for CachingConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            cache_dir: None,
            max_size_mb: Some(1024),
            ttl_sec: Some(3600),
            strategy: CacheStrategy::LRU,
        }
    }
}

impl Default for ValidationConfiguration {
    fn default() -> Self {
        Self {
            validate_input: true,
            validate_output: true,
            validate_structure: true,
            level: ValidationLevel::Basic,
        }
    }
}

impl Default for DebugConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            log_level: LogLevel::Info,
            profiling: false,
            tracing: false,
        }
    }
}

impl FluentPipelineBuilder<BuilderEmpty> {
    /// Create a new fluent pipeline builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: PhantomData,
            steps: Vec::new(),
            estimator: None,
            config: PipelineConfiguration::default(),
            validators: Vec::new(),
            presets: Vec::new(),
        }
    }

    /// Create builder with a configuration preset
    #[must_use]
    pub fn with_preset(preset: ConfigurationPreset) -> Self {
        let mut builder = Self::new();
        builder.config = preset.config;
        builder.presets.push(preset.name);
        builder
    }

    /// Apply a common preset for data science workflows
    #[must_use]
    pub fn data_science_preset() -> Self {
        let config = PipelineConfiguration {
            parallel: Some(ParallelConfig::default()),
            simd: Some(SimdConfig::default()),
            execution_strategy: Some(ParallelExecutionStrategy::DataParallel { chunk_size: 5000 }),
            memory_config: MemoryConfiguration {
                efficient_ops: true,
                chunk_size: Some(5000),
                memory_limit_mb: Some(2048),
                gc_frequency: Some(50),
            },
            caching: CachingConfiguration {
                enabled: true,
                cache_dir: Some("/tmp/sklearn_cache".to_string()),
                max_size_mb: Some(512),
                ttl_sec: Some(1800),
                strategy: CacheStrategy::LRU,
            },
            validation: ValidationConfiguration {
                validate_input: true,
                validate_output: true,
                validate_structure: true,
                level: ValidationLevel::Comprehensive,
            },
            debug: DebugConfiguration {
                enabled: true,
                log_level: LogLevel::Info,
                profiling: false,
                tracing: false,
            },
        };

        let mut builder = Self::new();
        builder.config = config;
        builder.presets.push("data_science".to_string());
        builder
    }

    /// Apply a preset for high-performance computing
    #[must_use]
    pub fn high_performance_preset() -> Self {
        let config = PipelineConfiguration {
            parallel: Some(ParallelConfig {
                num_workers: num_cpus::get() * 2,
                work_stealing: true,
                ..ParallelConfig::default()
            }),
            simd: Some(create_high_performance_simd_config()),
            execution_strategy: Some(ParallelExecutionStrategy::FullParallel),
            memory_config: MemoryConfiguration {
                efficient_ops: true,
                chunk_size: Some(100000),
                memory_limit_mb: Some(8192),
                gc_frequency: Some(200),
            },
            caching: CachingConfiguration {
                enabled: true,
                cache_dir: Some("/tmp/hpc_cache".to_string()),
                max_size_mb: Some(2048),
                ttl_sec: Some(7200),
                strategy: CacheStrategy::SizeBased,
            },
            validation: ValidationConfiguration {
                validate_input: false,
                validate_output: false,
                validate_structure: false,
                level: ValidationLevel::None,
            },
            debug: DebugConfiguration {
                enabled: false,
                log_level: LogLevel::Error,
                profiling: true,
                tracing: false,
            },
        };

        let mut builder = Self::new();
        builder.config = config;
        builder.presets.push("high_performance".to_string());
        builder
    }

    /// Apply a preset for development and debugging
    #[must_use]
    pub fn development_preset() -> Self {
        let config = PipelineConfiguration {
            parallel: None,
            simd: None,
            execution_strategy: None,
            memory_config: MemoryConfiguration {
                efficient_ops: false,
                chunk_size: Some(1000),
                memory_limit_mb: Some(512),
                gc_frequency: Some(10),
            },
            caching: CachingConfiguration {
                enabled: false,
                ..CachingConfiguration::default()
            },
            validation: ValidationConfiguration {
                validate_input: true,
                validate_output: true,
                validate_structure: true,
                level: ValidationLevel::Strict,
            },
            debug: DebugConfiguration {
                enabled: true,
                log_level: LogLevel::Debug,
                profiling: true,
                tracing: true,
            },
        };

        let mut builder = Self::new();
        builder.config = config;
        builder.presets.push("development".to_string());
        builder
    }
}

impl<State> FluentPipelineBuilder<State> {
    /// Configure parallel execution
    #[must_use]
    pub fn parallel(mut self, config: ParallelConfig) -> Self {
        self.config.parallel = Some(config);
        self
    }

    /// Configure SIMD optimizations
    #[must_use]
    pub fn simd(mut self, config: SimdConfig) -> Self {
        self.config.simd = Some(config);
        self
    }

    /// Set execution strategy
    #[must_use]
    pub fn execution_strategy(mut self, strategy: ParallelExecutionStrategy) -> Self {
        self.config.execution_strategy = Some(strategy);
        self
    }

    /// Configure memory settings
    #[must_use]
    pub fn memory(mut self, config: MemoryConfiguration) -> Self {
        self.config.memory_config = config;
        self
    }

    /// Configure caching
    #[must_use]
    pub fn caching(mut self, config: CachingConfiguration) -> Self {
        self.config.caching = config;
        self
    }

    /// Configure validation
    #[must_use]
    pub fn validation(mut self, config: ValidationConfiguration) -> Self {
        self.config.validation = config;
        self
    }

    /// Configure debug settings
    #[must_use]
    pub fn debug(mut self, config: DebugConfiguration) -> Self {
        self.config.debug = config;
        self
    }

    /// Add a validation rule
    #[must_use]
    pub fn with_validation_rule(mut self, rule: ValidationRule) -> Self {
        self.validators.push(rule);
        self
    }

    /// Enable memory optimization
    #[must_use]
    pub fn memory_optimized(mut self) -> Self {
        self.config.memory_config.efficient_ops = true;
        self.config.memory_config.chunk_size = Some(50000);
        self
    }

    /// Enable high performance mode
    #[must_use]
    pub fn high_performance(mut self) -> Self {
        self.config.parallel = Some(ParallelConfig::default());
        self.config.simd = Some(SimdConfig::default());
        self.config.execution_strategy = Some(ParallelExecutionStrategy::FullParallel);
        self
    }

    /// Enable development mode (with debugging and validation)
    #[must_use]
    pub fn development_mode(mut self) -> Self {
        self.config.debug.enabled = true;
        self.config.debug.log_level = LogLevel::Debug;
        self.config.validation.level = ValidationLevel::Strict;
        self
    }
}

impl FluentPipelineBuilder<BuilderEmpty> {
    /// Add the first step to the pipeline
    pub fn step<S: Into<String>>(
        mut self,
        name: S,
        step: Box<dyn PipelineStep>,
    ) -> FluentPipelineBuilder<BuilderWithSteps> {
        self.steps.push((name.into(), step));
        /// FluentPipelineBuilder
        FluentPipelineBuilder {
            state: PhantomData,
            steps: self.steps,
            estimator: self.estimator,
            config: self.config,
            validators: self.validators,
            presets: self.presets,
        }
    }

    /// Start with a preprocessing chain
    #[must_use]
    pub fn preprocessing(self) -> PreprocessingChain {
        /// PreprocessingChain
        PreprocessingChain {
            builder: self,
            preprocessing_steps: Vec::new(),
        }
    }

    /// Start with a feature engineering chain
    #[must_use]
    pub fn feature_engineering(self) -> FeatureEngineeringChain {
        /// FeatureEngineeringChain
        FeatureEngineeringChain {
            builder: self,
            feature_steps: Vec::new(),
        }
    }
}

impl FluentPipelineBuilder<BuilderWithSteps> {
    /// Add another step to the pipeline
    pub fn step<S: Into<String>>(mut self, name: S, step: Box<dyn PipelineStep>) -> Self {
        self.steps.push((name.into(), step));
        self
    }

    /// Add the final estimator
    #[must_use]
    pub fn estimator(
        mut self,
        estimator: Box<dyn PipelinePredictor>,
    ) -> FluentPipelineBuilder<BuilderWithEstimator> {
        self.estimator = Some(estimator);
        /// FluentPipelineBuilder
        FluentPipelineBuilder {
            state: PhantomData,
            steps: self.steps,
            estimator: self.estimator,
            config: self.config,
            validators: self.validators,
            presets: self.presets,
        }
    }

    /// Create a feature union at this point
    pub fn feature_union<F>(mut self, union_fn: F) -> Self
    where
        F: FnOnce(FeatureUnionBuilder) -> FeatureUnionBuilder,
    {
        let union_builder = FeatureUnionBuilder::new();
        let union_builder = union_fn(union_builder);
        let feature_union = union_builder.build();

        self.steps
            .push(("feature_union".to_string(), Box::new(feature_union)));
        self
    }

    /// Add conditional execution
    pub fn when<F>(mut self, condition: F, then_step: Box<dyn PipelineStep>) -> Self
    where
        F: Fn(&ArrayView2<Float>) -> bool + 'static,
    {
        // In a real implementation, this would wrap the step in a conditional wrapper
        // For now, we'll just add the step directly
        self.steps.push(("conditional".to_string(), then_step));
        self
    }
}

impl FluentPipelineBuilder<BuilderWithEstimator> {
    /// Finalize the pipeline configuration
    #[must_use]
    pub fn finalize(self) -> FluentPipelineBuilder<BuilderComplete> {
        /// FluentPipelineBuilder
        FluentPipelineBuilder {
            state: PhantomData,
            steps: self.steps,
            estimator: self.estimator,
            config: self.config,
            validators: self.validators,
            presets: self.presets,
        }
    }
}

impl FluentPipelineBuilder<BuilderComplete> {
    /// Build the final pipeline
    pub fn build(self) -> SklResult<Pipeline<Untrained>> {
        // Run validation rules
        for validator in &self.validators {
            (validator.validator)(&self)?;
        }

        // Create pipeline builder
        let mut pipeline_builder = Pipeline::builder();

        // Add steps
        for (name, step) in self.steps {
            pipeline_builder = pipeline_builder.step(&name, step);
        }

        // Add estimator if present
        if let Some(estimator) = self.estimator {
            pipeline_builder = pipeline_builder.estimator(estimator);
        }

        // Build and return
        Ok(pipeline_builder.build())
    }

    /// Build and immediately fit the pipeline
    pub fn build_and_fit(
        self,
        x: &ArrayView2<Float>,
        y: &Option<&ArrayView1<Float>>,
    ) -> SklResult<crate::Pipeline<crate::pipeline::PipelineTrained>> {
        let pipeline = self.build()?;
        pipeline.fit(x, y)
    }
}

/// Preprocessing chain builder
#[derive(Debug)]
pub struct PreprocessingChain {
    builder: FluentPipelineBuilder<BuilderEmpty>,
    preprocessing_steps: Vec<(String, Box<dyn PipelineStep>)>,
}

impl PreprocessingChain {
    /// Add a standard scaler
    #[must_use]
    pub fn standard_scaler(mut self) -> Self {
        // In a real implementation, this would create an actual StandardScaler
        self.preprocessing_steps.push((
            "standard_scaler".to_string(),
            Box::new(crate::MockTransformer::new()),
        ));
        self
    }

    /// Add a min-max scaler
    #[must_use]
    pub fn min_max_scaler(mut self, feature_range: (f64, f64)) -> Self {
        self.preprocessing_steps.push((
            "min_max_scaler".to_string(),
            Box::new(crate::MockTransformer::new()),
        ));
        self
    }

    /// Add robust scaling
    #[must_use]
    pub fn robust_scaler(mut self) -> Self {
        self.preprocessing_steps.push((
            "robust_scaler".to_string(),
            Box::new(crate::MockTransformer::new()),
        ));
        self
    }

    /// Add missing value imputation
    #[must_use]
    pub fn impute_missing(mut self, strategy: ImputationStrategy) -> Self {
        self.preprocessing_steps.push((
            "imputer".to_string(),
            Box::new(crate::MockTransformer::new()),
        ));
        self
    }

    /// Finish preprocessing and return to main builder
    #[must_use]
    pub fn done(mut self) -> FluentPipelineBuilder<BuilderWithSteps> {
        for (name, step) in self.preprocessing_steps {
            self.builder.steps.push((name, step));
        }

        /// FluentPipelineBuilder
        FluentPipelineBuilder {
            state: PhantomData,
            steps: self.builder.steps,
            estimator: self.builder.estimator,
            config: self.builder.config,
            validators: self.builder.validators,
            presets: self.builder.presets,
        }
    }
}

/// Feature engineering chain builder
#[derive(Debug)]
pub struct FeatureEngineeringChain {
    builder: FluentPipelineBuilder<BuilderEmpty>,
    feature_steps: Vec<(String, Box<dyn PipelineStep>)>,
}

impl FeatureEngineeringChain {
    /// Add polynomial features
    #[must_use]
    pub fn polynomial_features(mut self, degree: usize, include_bias: bool) -> Self {
        self.feature_steps.push((
            "polynomial_features".to_string(),
            Box::new(crate::MockTransformer::new()),
        ));
        self
    }

    /// Add feature selection
    #[must_use]
    pub fn feature_selection(mut self, k_best: usize) -> Self {
        self.feature_steps.push((
            "feature_selection".to_string(),
            Box::new(crate::MockTransformer::new()),
        ));
        self
    }

    /// Add PCA
    #[must_use]
    pub fn pca(mut self, n_components: Option<usize>) -> Self {
        self.feature_steps
            .push(("pca".to_string(), Box::new(crate::MockTransformer::new())));
        self
    }

    /// Add text vectorization
    #[must_use]
    pub fn text_vectorizer(mut self, max_features: Option<usize>) -> Self {
        self.feature_steps.push((
            "text_vectorizer".to_string(),
            Box::new(crate::MockTransformer::new()),
        ));
        self
    }

    /// Finish feature engineering and return to main builder
    #[must_use]
    pub fn done(mut self) -> FluentPipelineBuilder<BuilderWithSteps> {
        for (name, step) in self.feature_steps {
            self.builder.steps.push((name, step));
        }

        /// FluentPipelineBuilder
        FluentPipelineBuilder {
            state: PhantomData,
            steps: self.builder.steps,
            estimator: self.builder.estimator,
            config: self.builder.config,
            validators: self.builder.validators,
            presets: self.builder.presets,
        }
    }
}

/// Feature union builder for parallel feature extraction
#[derive(Debug)]
pub struct FeatureUnionBuilder {
    transformers: Vec<(String, Box<dyn PipelineStep>)>,
    weights: Option<HashMap<String, f64>>,
    n_jobs: Option<i32>,
}

impl FeatureUnionBuilder {
    /// Create a new feature union builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
            weights: None,
            n_jobs: None,
        }
    }

    /// Add a transformer to the union
    pub fn add_transformer<S: Into<String>>(
        mut self,
        name: S,
        transformer: Box<dyn PipelineStep>,
    ) -> Self {
        self.transformers.push((name.into(), transformer));
        self
    }

    /// Set transformer weights
    #[must_use]
    pub fn weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set number of parallel jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Build the feature union
    #[must_use]
    pub fn build(self) -> crate::MockTransformer {
        // In a real implementation, this would create an actual FeatureUnion
        crate::MockTransformer::new()
    }
}

/// Imputation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ImputationStrategy {
    /// Mean
    Mean,
    /// Median
    Median,
    /// MostFrequent
    MostFrequent,
    /// Constant
    Constant(f64),
    /// Forward
    Forward,
    /// Backward
    Backward,
}

/// Quick access functions for common patterns
pub struct PipelinePresets;

impl PipelinePresets {
    /// Create a basic classification pipeline
    #[must_use]
    pub fn classification() -> FluentPipelineBuilder<BuilderEmpty> {
        FluentPipelineBuilder::data_science_preset()
    }

    /// Create a basic regression pipeline  
    #[must_use]
    pub fn regression() -> FluentPipelineBuilder<BuilderEmpty> {
        FluentPipelineBuilder::data_science_preset()
    }

    /// Create a text processing pipeline
    #[must_use]
    pub fn text_processing() -> FluentPipelineBuilder<BuilderEmpty> {
        FluentPipelineBuilder::new()
            .memory_optimized()
            .validation(ValidationConfiguration {
                validate_input: true,
                validate_output: true,
                validate_structure: true,
                level: ValidationLevel::Basic,
            })
    }

    /// Create an image processing pipeline
    #[must_use]
    pub fn image_processing() -> FluentPipelineBuilder<BuilderEmpty> {
        FluentPipelineBuilder::high_performance_preset().memory(MemoryConfiguration {
            efficient_ops: true,
            chunk_size: Some(1000),
            memory_limit_mb: Some(4096),
            gc_frequency: Some(50),
        })
    }

    /// Create a time series pipeline
    #[must_use]
    pub fn time_series() -> FluentPipelineBuilder<BuilderEmpty> {
        FluentPipelineBuilder::new().caching(CachingConfiguration {
            enabled: true,
            cache_dir: Some("/tmp/ts_cache".to_string()),
            max_size_mb: Some(512),
            ttl_sec: Some(1800),
            strategy: CacheStrategy::TimeExpire,
        })
    }
}

impl Default for FluentPipelineBuilder<BuilderEmpty> {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FeatureUnionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluent_builder_creation() {
        let builder = FluentPipelineBuilder::new();
        assert!(builder.steps.is_empty());
        assert!(builder.estimator.is_none());
    }

    #[test]
    fn test_preset_application() {
        let builder = FluentPipelineBuilder::data_science_preset();
        assert!(builder.config.parallel.is_some());
        assert!(builder.config.simd.is_some());
        assert!(builder.presets.contains(&"data_science".to_string()));
    }

    #[test]
    fn test_high_performance_preset() {
        let builder = FluentPipelineBuilder::high_performance_preset();
        assert!(builder.config.parallel.is_some());
        assert!(builder.config.simd.is_some());
        assert_eq!(builder.config.validation.level, ValidationLevel::None);
        assert!(builder.presets.contains(&"high_performance".to_string()));
    }

    #[test]
    fn test_development_preset() {
        let builder = FluentPipelineBuilder::development_preset();
        assert!(builder.config.debug.enabled);
        assert_eq!(builder.config.debug.log_level, LogLevel::Debug);
        assert_eq!(builder.config.validation.level, ValidationLevel::Strict);
    }

    #[test]
    fn test_method_chaining() {
        let builder = FluentPipelineBuilder::new()
            .memory_optimized()
            .high_performance()
            .development_mode();

        assert!(builder.config.memory_config.efficient_ops);
        assert!(builder.config.parallel.is_some());
        assert!(builder.config.debug.enabled);
    }

    #[test]
    fn test_preprocessing_chain() {
        let chain = FluentPipelineBuilder::new()
            .preprocessing()
            .standard_scaler()
            .min_max_scaler((0.0, 1.0))
            .impute_missing(ImputationStrategy::Mean);

        assert_eq!(chain.preprocessing_steps.len(), 3);
    }

    #[test]
    fn test_feature_engineering_chain() {
        let chain = FluentPipelineBuilder::new()
            .feature_engineering()
            .polynomial_features(2, true)
            .feature_selection(100)
            .pca(Some(50));

        assert_eq!(chain.feature_steps.len(), 3);
    }

    #[test]
    fn test_feature_union_builder() {
        let union_builder = FeatureUnionBuilder::new()
            .add_transformer("scaler", Box::new(crate::MockTransformer::new()))
            .add_transformer("pca", Box::new(crate::MockTransformer::new()))
            .n_jobs(2);

        assert_eq!(union_builder.transformers.len(), 2);
        assert_eq!(union_builder.n_jobs, Some(2));
    }

    #[test]
    fn test_configuration_defaults() {
        let config = PipelineConfiguration::default();
        assert!(config.memory_config.efficient_ops);
        assert_eq!(config.caching.strategy, CacheStrategy::LRU);
        assert_eq!(config.validation.level, ValidationLevel::Basic);
        assert_eq!(config.debug.log_level, LogLevel::Info);
    }

    #[test]
    fn test_pipeline_presets() {
        let classification = PipelinePresets::classification();
        let regression = PipelinePresets::regression();
        let text = PipelinePresets::text_processing();
        let image = PipelinePresets::image_processing();
        let ts = PipelinePresets::time_series();

        // All presets should create valid builders
        assert!(classification.steps.is_empty());
        assert!(regression.steps.is_empty());
        assert!(text.steps.is_empty());
        assert!(image.steps.is_empty());
        assert!(ts.steps.is_empty());
    }

    #[test]
    fn test_memory_configuration() {
        let memory_config = MemoryConfiguration {
            efficient_ops: true,
            chunk_size: Some(5000),
            memory_limit_mb: Some(1024),
            gc_frequency: Some(100),
        };

        assert!(memory_config.efficient_ops);
        assert_eq!(memory_config.chunk_size, Some(5000));
        assert_eq!(memory_config.memory_limit_mb, Some(1024));
    }

    #[test]
    fn test_caching_configuration() {
        let cache_config = CachingConfiguration {
            enabled: true,
            cache_dir: Some("/tmp/test".to_string()),
            max_size_mb: Some(512),
            ttl_sec: Some(1800),
            strategy: CacheStrategy::LFU,
        };

        assert!(cache_config.enabled);
        assert_eq!(cache_config.strategy, CacheStrategy::LFU);
        assert_eq!(cache_config.max_size_mb, Some(512));
    }
}
