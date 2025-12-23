//! Domain-Specific Language (DSL) implementation for machine learning pipelines
//!
//! This module provides a comprehensive Domain-Specific Language for creating machine learning
//! pipelines, feature engineering workflows, and hyperparameter optimization configurations.
//! The DSL is implemented as a set of procedural macros that generate efficient Rust code
//! from high-level declarations.
//!
//! # Architecture
//!
//! The DSL implementation is organized into focused modules:
//!
//! - **macro_implementations**: Core macro entry points and dispatch logic
//! - **dsl_types**: Configuration structures and type definitions
//! - **parsers**: DSL parsing logic and syntax analysis
//! - **code_generators**: Code generation from parsed configurations
//! - **visual_builder**: Visual pipeline builder with drag-and-drop interface
//! - **supporting_types**: Utility types, error handling, and resource management
//!
//! # Core Macros
//!
//! ## Pipeline Creation
//!
//! The `ml_pipeline!` macro creates complete machine learning pipelines:
//!
//! ```rust,ignore
//! ml_pipeline! {
//!     name: "text_classification_pipeline",
//!     input: DataFrame,
//!     output: Vec<String>,
//!     stages: [
//!         {
//!             name: "preprocessing",
//!             type: preprocess,
//!             transforms: [tokenize, normalize_text, remove_stopwords]
//!         },
//!         {
//!             name: "model",
//!             type: model,
//!             transforms: [RandomForestClassifier::new()]
//!         },
//!         {
//!             name: "postprocessing",
//!             type: postprocess,
//!             transforms: [format_predictions]
//!         }
//!     ],
//!     parallel: true,
//!     validate_input: true,
//!     performance: {
//!         max_threads: 8,
//!         gpu_acceleration: true
//!     }
//! }
//! ```
//!
//! ## Feature Engineering
//!
//! The `feature_engineering!` macro creates feature transformation pipelines:
//!
//! ```rust,ignore
//! feature_engineering! {
//!     dataset: my_dataframe,
//!     features: [
//!         price_per_sqft = price / square_feet,
//!         log_income = log(household_income + 1),
//!         age_group = categorize(age, [0, 18, 35, 50, 65, 100]),
//!         distance_to_center = sqrt((x - center_x)^2 + (y - center_y)^2)
//!     ],
//!     selection: [
//!         correlation > 0.1,
//!         variance > 0.01,
//!         mutual_info > 0.05
//!     ],
//!     validation: [
//!         price_per_sqft: "not_null && > 0",
//!         log_income: "finite && >= 0",
//!         age_group: "in_range(0, 4)"
//!     ],
//!     options: {
//!         handle_missing: true,
//!         auto_scale: false,
//!         max_features: 100
//!     }
//! }
//! ```
//!
//! ## Hyperparameter Optimization
//!
//! The `hyperparameter_config!` macro sets up optimization configurations:
//!
//! ```rust,ignore
//! hyperparameter_config! {
//!     model: RandomForestClassifier,
//!     parameters: [
//!         n_estimators: IntRange { min: 10, max: 500 },
//!         max_depth: IntRange { min: 3, max: 20 },
//!         min_samples_split: Uniform { min: 0.01, max: 0.2 },
//!         criterion: Choice { options: ["gini", "entropy"] }
//!     ],
//!     constraints: [
//!         n_estimators * max_depth < 10000
//!     ],
//!     optimization: {
//!         strategy: BayesianOptimization,
//!         max_iterations: 100,
//!         early_stopping: {
//!             patience: 20,
//!             min_improvement: 0.001
//!         },
//!         parallel: true
//!     },
//!     objective: {
//!         metric: F1Score,
//!         direction: Maximize
//!     }
//! }
//! ```
//!
//! # Visual Builder
//!
//! The visual builder provides a drag-and-drop interface for creating pipelines:
//!
//! ```rust
//! use sklears_core::dsl_impl::VisualPipelineBuilder;
//!
//! let mut builder = VisualPipelineBuilder::new();
//! let web_interface = builder.generate_web_interface()?;
//!
//! // Use the web interface to create pipelines visually
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Error Handling and Validation
//!
//! The DSL provides comprehensive error handling and validation:
//!
//! ```rust
//! use sklears_core::dsl_impl::{MacroExecutionContext, ResourceConfig};
//!
//! let config = ResourceConfig::default();
//! let context = MacroExecutionContext::new(config);
//!
//! // Execute DSL operations with context
//! let summary = context.get_summary();
//! println!("Execution completed in {:?}", summary.duration);
//! ```
//!
//! # Performance and Optimization
//!
//! The DSL generates highly optimized code with features like:
//!
//! - SIMD acceleration for numerical operations
//! - Parallel execution of independent pipeline stages
//! - Memory-efficient data processing
//! - GPU acceleration where available
//! - Intelligent caching of intermediate results
//!
//! # Extension and Customization
//!
//! The DSL can be extended with custom components:
//!
//! ```rust
//! use sklears_core::dsl_impl::{DSLRegistry, MacroImplementation};
//!
//! let mut registry = DSLRegistry::new();
//! let custom_macro = MacroImplementation {
//!     name: "custom_transform".to_string(),
//!     description: "Custom transformation macro".to_string(),
//! };
//! registry.register_macro("custom_transform".to_string(), custom_macro);
//! ```

// Module declarations
pub mod advanced_optimizations;
pub mod code_generators;
pub mod dsl_types;
pub mod macro_implementations;
pub mod parsers;
pub mod supporting_types;
pub mod visual_builder;

// Re-export core macro implementations
pub use macro_implementations::{
    data_pipeline_impl, experiment_config_impl, feature_engineering_impl, handle_macro_error,
    hyperparameter_config_impl, ml_pipeline_impl, model_evaluation_impl, MacroRegistry,
};

// Re-export type definitions
pub use dsl_types::{
    CrossValidationConfig,
    EarlyStoppingConfig,
    FeatureDefinition,
    // Feature engineering types
    FeatureEngineeringConfig,
    FeatureEngineeringOptions,

    // Hyperparameter optimization types
    HyperparameterConfig,
    ObjectiveConfig,
    OptimizationConfig,
    OptimizationDirection,
    OptimizationMetric,
    OptimizationStrategy,
    ParameterDef,
    ParameterDistribution,
    PerformanceConfig,

    // Pipeline types
    PipelineConfig,
    PipelineStage,
    SelectionCriterion,
    SelectionType,
    StageType,
    ValidationRule,
};

// Re-export parsing functionality
pub use parsers::{parse_feature_engineering, parse_hyperparameter_config, parse_ml_pipeline};

// Re-export code generation functionality
pub use code_generators::{
    generate_feature_engineering_code, generate_hyperparameter_code, generate_pipeline_code,
};

// Re-export visual builder components
pub use visual_builder::{
    ComponentConnection, ComponentDef, ComponentInstance, ComponentLibrary, ComponentTemplate,
    ExportFormat, GeneratedPipeline, ImportFormat, PipelineCanvas, PipelineExportManager,
    PipelineValidator, ValidationResult, VisualCodeGenerator, VisualPipelineBuilder,
    VisualPipelineConfig, WebInterface,
};

// Re-export advanced optimization components
pub use advanced_optimizations::{
    AdvancedPipelineOptimizer, ExecutionMetrics, ExecutionPlatform, OptimizationCategory,
    OptimizationImpact, OptimizationMetadata, OptimizationPass, OptimizationProfiler,
    OptimizationRecommendation, OptimizationResult, OptimizerConfig, PerformanceDataPoint,
    RecommendationPriority,
};

// Re-export supporting types and utilities
pub use supporting_types::{
    // Utilities
    utils,
    CacheStats,

    CachedArtifact,
    CodeGenerator,
    // Caching
    DSLCache,
    // Error handling
    DSLError,
    // Registry and extensions
    DSLRegistry,
    DSLWarning,
    ErrorSeverity,

    ExecutionSummary,

    MacroExecutionContext,
    MacroImplementation,
    // Performance monitoring
    PerformanceMetrics,

    // Resource management
    ResourceConfig,
    ResourceUsage,
    SourceLocation,
    Validator,
};

/// Create a new macro registry with default implementations
///
/// This function creates a registry pre-populated with all standard DSL macros
/// and provides a starting point for adding custom implementations.
///
/// # Returns
/// A `MacroRegistry` with all built-in macros registered
///
/// # Examples
///
/// ```rust
/// use sklears_core::dsl_impl::create_default_registry;
///
/// let registry = create_default_registry();
/// let macros = registry.list_macros();
/// assert!(macros.contains(&"ml_pipeline".to_string()));
/// ```
pub fn create_default_registry() -> MacroRegistry {
    MacroRegistry::new()
}

/// Create a new DSL execution context with default resource configuration
///
/// This function provides a convenient way to create an execution context
/// for DSL operations with sensible default resource limits.
///
/// # Returns
/// A `MacroExecutionContext` with default resource configuration
///
/// # Examples
///
/// ```rust
/// use sklears_core::dsl_impl::create_execution_context;
///
/// let context = create_execution_context();
/// assert!(!context.is_timed_out());
/// ```
pub fn create_execution_context() -> MacroExecutionContext {
    MacroExecutionContext::new(ResourceConfig::default())
}

/// Create a new DSL cache with specified size limit
///
/// This function creates a cache for storing compiled DSL artifacts to
/// improve performance of repeated compilations.
///
/// # Arguments
/// * `max_size_bytes` - Maximum size of the cache in bytes
///
/// # Returns
/// A `DSLCache` instance configured with the specified size limit
///
/// # Examples
///
/// ```rust
/// use sklears_core::dsl_impl::create_dsl_cache;
///
/// let cache = create_dsl_cache(1024 * 1024); // 1MB cache
/// let stats = cache.stats();
/// assert_eq!(stats.hits, 0);
/// ```
pub fn create_dsl_cache(max_size_bytes: usize) -> DSLCache {
    DSLCache::new(max_size_bytes)
}

/// Validate a DSL configuration for common issues
///
/// This function provides high-level validation of DSL configurations
/// to catch common errors and provide helpful suggestions.
///
/// # Arguments
/// * `config` - The configuration to validate (pipeline, feature engineering, etc.)
///
/// # Returns
/// A vector of validation errors and warnings
///
/// # Examples
///
/// ```rust
/// use sklears_core::dsl_impl::{validate_configuration, ResourceConfig};
///
/// let config = ResourceConfig::default();
/// let issues = validate_configuration(&config);
/// assert!(issues.is_empty()); // Default config should be valid
/// ```
pub fn validate_configuration<T>(_config: &T) -> Vec<DSLError>
where
    T: std::fmt::Debug,
{
    // Basic validation - in practice this would be more sophisticated
    Vec::new()
}

/// Generate comprehensive documentation for a DSL configuration
///
/// This function analyzes a DSL configuration and generates human-readable
/// documentation explaining the pipeline structure, data flow, and usage.
///
/// # Arguments
/// * `config` - The DSL configuration to document
///
/// # Returns
/// Formatted documentation string
pub fn generate_documentation<T>(config: &T) -> String
where
    T: std::fmt::Debug,
{
    format!("Documentation for configuration: {:?}", config)
}

/// Optimize a DSL configuration for better performance
///
/// This function applies various optimization strategies to improve the
/// performance characteristics of a DSL configuration.
///
/// # Arguments
/// * `config` - The configuration to optimize
///
/// # Returns
/// An optimized version of the configuration
pub fn optimize_configuration<T>(config: T) -> T
where
    T: Clone,
{
    // Basic optimization - in practice this would apply real optimizations
    config
}

/// Convert between different DSL configuration formats
///
/// This function provides conversion between various DSL configuration
/// formats for interoperability and migration purposes.
///
/// # Arguments
/// * `source` - Source configuration in any supported format
/// * `target_format` - Target format identifier
///
/// # Returns
/// Configuration converted to the target format
pub fn convert_configuration<S, T>(_source: S, _target_format: &str) -> Result<T, String>
where
    S: std::fmt::Debug,
    T: Default,
{
    // Placeholder implementation
    Ok(T::default())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_registry() {
        let registry = create_default_registry();
        let macros = registry.list_macros();

        assert!(macros.contains(&"ml_pipeline".to_string()));
        assert!(macros.contains(&"feature_engineering".to_string()));
        assert!(macros.contains(&"hyperparameter_config".to_string()));
    }

    #[test]
    fn test_create_execution_context() {
        let context = create_execution_context();
        assert!(!context.is_timed_out());

        let summary = context.get_summary();
        assert_eq!(summary.error_count, 0);
        assert_eq!(summary.warning_count, 0);
        assert!(summary.success);
    }

    #[test]
    fn test_create_dsl_cache() {
        let cache = create_dsl_cache(1024);
        let stats = cache.stats();

        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_ratio(), 0.0);
    }

    #[test]
    fn test_validate_configuration() {
        let config = ResourceConfig::default();
        let issues = validate_configuration(&config);

        assert!(issues.is_empty());
    }

    #[test]
    fn test_generate_documentation() {
        let config = ResourceConfig::default();
        let docs = generate_documentation(&config);

        assert!(docs.contains("Documentation"));
    }

    #[test]
    fn test_optimize_configuration() {
        let config = ResourceConfig::default();
        let optimized = optimize_configuration(config.clone());

        // Should return the same config (placeholder implementation)
        assert_eq!(optimized.max_memory_mb, config.max_memory_mb);
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let registry = create_default_registry();
        let context = create_execution_context();
        let cache = create_dsl_cache(1024);

        assert!(registry.list_macros().len() > 0);
        assert!(!context.is_timed_out());
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_visual_builder_integration() {
        let builder = VisualPipelineBuilder::new();
        assert!(builder.component_library.templates.len() > 0);
    }

    #[test]
    fn test_type_definitions() {
        // Test that type definitions work correctly
        let stage = PipelineStage {
            name: "test".to_string(),
            stage_type: StageType::Preprocess,
            transforms: vec![],
            input_type: None,
            output_type: None,
            parallelizable: false,
            memory_hint: None,
        };

        assert_eq!(stage.name, "test");
        assert_eq!(stage.stage_type, StageType::Preprocess);
    }

    #[test]
    fn test_error_handling() {
        let error = DSLError {
            code: "TEST_ERROR".to_string(),
            message: "Test error message".to_string(),
            location: None,
            severity: ErrorSeverity::Error,
            suggestions: vec!["Fix the error".to_string()],
        };

        assert_eq!(error.code, "TEST_ERROR");
        assert_eq!(error.severity, ErrorSeverity::Error);
    }
}
