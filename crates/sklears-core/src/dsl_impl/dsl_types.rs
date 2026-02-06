//! Type definitions for DSL configuration structures
//!
//! This module contains all the data structures used to represent parsed DSL
//! configurations for machine learning pipelines, feature engineering, and
//! hyperparameter optimization. These types serve as the intermediate representation
//! between the parsed DSL syntax and the generated code.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pipeline configuration structure
///
/// Represents a complete machine learning pipeline configuration parsed from
/// the ml_pipeline! macro. Contains all stages, types, and execution options.
#[derive(Clone)]
pub struct PipelineConfig {
    /// Name of the pipeline for identification and code generation
    pub name: String,
    /// Ordered list of processing stages in the pipeline
    pub stages: Vec<PipelineStage>,
    /// Type of input data the pipeline expects
    pub input_type: syn::Type,
    /// Type of output data the pipeline produces
    pub output_type: syn::Type,
    /// Whether to execute stages in parallel when possible
    pub parallel: bool,
    /// Whether to validate input data before processing
    pub validate_input: bool,
    /// Whether to cache intermediate transformations
    pub cache_transforms: bool,
    /// Additional metadata for the pipeline
    pub metadata: HashMap<String, String>,
    /// Performance configuration options
    pub performance: PerformanceConfig,
}

/// Individual pipeline stage definition
///
/// Represents a single stage in the ML pipeline with its transformations
/// and configuration options.
#[derive(Clone)]
pub struct PipelineStage {
    /// Name of the stage for identification
    pub name: String,
    /// Type/category of the pipeline stage
    pub stage_type: StageType,
    /// List of transformations to apply in this stage
    pub transforms: Vec<syn::Expr>,
    /// Input type for this stage (if different from pipeline input)
    pub input_type: Option<syn::Type>,
    /// Output type for this stage (if different from next stage input)
    pub output_type: Option<syn::Type>,
    /// Whether this stage can be parallelized
    pub parallelizable: bool,
    /// Memory requirements for this stage
    pub memory_hint: Option<usize>,
}

/// Type of pipeline stage
///
/// Categorizes pipeline stages to enable stage-specific optimizations
/// and validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageType {
    /// Data preprocessing and cleaning stage
    Preprocess,
    /// Feature engineering and transformation stage
    FeatureEngineering,
    /// Model training or inference stage
    Model,
    /// Post-processing and output formatting stage
    Postprocess,
    /// Custom user-defined stage type
    Custom(String),
}

/// Performance configuration for pipelines
///
/// Configures performance-related options for pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceConfig {
    /// Maximum number of parallel threads to use
    #[serde(default)]
    pub max_threads: Option<usize>,
    /// Maximum memory usage in bytes
    #[serde(default)]
    pub max_memory_bytes: Option<usize>,
    /// Whether to enable GPU acceleration
    #[serde(default)]
    pub gpu_acceleration: bool,
    /// Batch size for processing large datasets
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Timeout for individual stages in seconds
    #[serde(default)]
    pub stage_timeout_seconds: Option<u64>,
}

/// Feature engineering configuration
///
/// Represents the configuration for the feature_engineering! macro,
/// including feature definitions, selection criteria, and validation rules.
#[derive(Clone)]
pub struct FeatureEngineeringConfig {
    /// Source dataset expression
    pub dataset: syn::Expr,
    /// List of feature definitions to create
    pub features: Vec<FeatureDefinition>,
    /// Feature selection criteria
    pub selection: Vec<SelectionCriterion>,
    /// Validation rules for generated features
    pub validation: Vec<ValidationRule>,
    /// Feature engineering options
    pub options: FeatureEngineeringOptions,
}

/// Individual feature definition
///
/// Defines how to compute a new feature from existing data.
#[derive(Clone)]
pub struct FeatureDefinition {
    /// Name of the new feature
    pub name: String,
    /// Expression to compute the feature value
    pub expression: syn::Expr,
    /// Data type of the feature (optional, inferred if not specified)
    pub data_type: Option<syn::Type>,
    /// Description of the feature for documentation
    pub description: Option<String>,
    /// Whether this feature is required or optional
    pub required: bool,
}

/// Feature selection criterion
///
/// Defines criteria for automatically selecting features based on
/// statistical properties or domain-specific rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriterion {
    /// Type of selection criterion
    pub criterion_type: SelectionType,
    /// Threshold value for the criterion
    pub threshold: f64,
    /// Whether to apply this criterion (can be disabled)
    pub enabled: bool,
}

/// Types of feature selection criteria
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionType {
    /// Correlation with target variable
    Correlation,
    /// Variance of the feature values
    Variance,
    /// Mutual information with target
    MutualInformation,
    /// Chi-squared test statistic
    ChiSquared,
    /// F-statistic for ANOVA
    FStatistic,
    /// Custom selection function
    Custom(String),
}

/// Validation rule for features
///
/// Defines validation constraints that features must satisfy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Name of the feature to validate
    pub feature: String,
    /// Validation expression (e.g., "not_null && > 0")
    pub rule: String,
    /// Error message to display if validation fails
    pub error_message: Option<String>,
    /// Whether validation failure should stop processing
    pub strict: bool,
}

/// Feature engineering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringOptions {
    /// Whether to automatically handle missing values
    pub handle_missing: bool,
    /// Whether to automatically scale numerical features
    pub auto_scale: bool,
    /// Whether to automatically encode categorical features
    pub auto_encode: bool,
    /// Maximum number of features to generate
    pub max_features: Option<usize>,
}

impl Default for FeatureEngineeringOptions {
    fn default() -> Self {
        Self {
            handle_missing: true,
            auto_scale: false,
            auto_encode: false,
            max_features: None,
        }
    }
}

/// Hyperparameter configuration
///
/// Represents the configuration for hyperparameter optimization from
/// the hyperparameter_config! macro.
#[derive(Clone)]
pub struct HyperparameterConfig {
    /// Model type to optimize hyperparameters for
    pub model: syn::Type,
    /// List of hyperparameters to optimize
    pub parameters: Vec<ParameterDef>,
    /// Constraints on parameter combinations
    pub constraints: Vec<syn::Expr>,
    /// Optimization strategy and settings
    pub optimization: OptimizationConfig,
    /// Objective function configuration
    pub objective: ObjectiveConfig,
}

/// Parameter definition for hyperparameter optimization
///
/// Defines a single hyperparameter, its search space, and properties.
#[derive(Clone)]
pub struct ParameterDef {
    /// Name of the hyperparameter
    pub name: String,
    /// Distribution type and range for the parameter
    pub distribution: ParameterDistribution,
    /// Default value (optional)
    pub default: Option<syn::Expr>,
    /// Description of the parameter
    pub description: Option<String>,
    /// Whether this parameter is required
    pub required: bool,
}

/// Parameter distribution types for hyperparameter search
///
/// Defines different types of distributions for sampling parameter values
/// during hyperparameter optimization.
#[derive(Clone)]
pub enum ParameterDistribution {
    /// Uniform distribution over a range [min, max]
    Uniform { min: syn::Expr, max: syn::Expr },
    /// Log-uniform distribution over a range [min, max]
    LogUniform { min: syn::Expr, max: syn::Expr },
    /// Discrete choice from a list of values
    Choice { options: Vec<syn::Expr> },
    /// Integer range [min, max]
    IntRange { min: i64, max: i64 },
    /// Normal distribution with mean and standard deviation
    Normal { mean: f64, std: f64 },
    /// Custom distribution function
    Custom { function: String },
}

/// Optimization configuration for hyperparameter tuning
///
/// Configures the optimization strategy and stopping criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization strategy to use
    pub strategy: OptimizationStrategy,
    /// Maximum number of optimization trials
    pub max_iterations: usize,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Cross-validation configuration
    pub cv_config: Option<CrossValidationConfig>,
    /// Parallelization settings
    pub parallel: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Optimization strategies for hyperparameter tuning
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    TPE,
    GeneticAlgorithm,
    SuccessiveHalving,
    Hyperband,
    Custom(String),
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Number of trials without improvement before stopping
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Maximum time to spend optimizing (in seconds)
    pub max_time_seconds: Option<u64>,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of cross-validation folds
    pub n_folds: usize,
    /// Whether to stratify the folds
    pub stratified: bool,
    /// Random seed for fold generation
    pub random_seed: Option<u64>,
}

/// Objective function configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveConfig {
    /// Metric to optimize
    pub metric: OptimizationMetric,
    /// Direction of optimization (minimize or maximize)
    pub direction: OptimizationDirection,
    /// Additional metrics to track
    pub additional_metrics: Vec<OptimizationMetric>,
}

/// Optimization metrics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationMetric {
    /// Accuracy score
    Accuracy,
    /// Precision score
    Precision,
    /// Recall score
    Recall,
    /// F1 score
    F1Score,
    /// Area under ROC curve
    AucRoc,
    /// Mean squared error
    MeanSquaredError,
    /// Mean absolute error
    MeanAbsoluteError,
    /// R-squared score
    RSquared,
    /// Custom metric function
    Custom(String),
}

/// Optimization direction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationDirection {
    /// Minimize the objective function
    Minimize,
    /// Maximize the objective function
    Maximize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::RandomSearch,
            max_iterations: 100,
            early_stopping: None,
            cv_config: None,
            parallel: false,
            random_seed: None,
        }
    }
}

impl Default for ObjectiveConfig {
    fn default() -> Self {
        Self {
            metric: OptimizationMetric::Accuracy,
            direction: OptimizationDirection::Maximize,
            additional_metrics: Vec::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_type_equality() {
        assert_eq!(StageType::Preprocess, StageType::Preprocess);
        assert_eq!(
            StageType::Custom("test".to_string()),
            StageType::Custom("test".to_string())
        );
        assert_ne!(StageType::Preprocess, StageType::Model);
    }

    #[test]
    fn test_optimization_strategy_equality() {
        assert_eq!(
            OptimizationStrategy::RandomSearch,
            OptimizationStrategy::RandomSearch
        );
        assert_eq!(
            OptimizationStrategy::Custom("test".to_string()),
            OptimizationStrategy::Custom("test".to_string())
        );
        assert_ne!(
            OptimizationStrategy::RandomSearch,
            OptimizationStrategy::GridSearch
        );
    }

    #[test]
    fn test_default_performance_config() {
        let config = PerformanceConfig::default();
        assert_eq!(config.max_threads, None);
        assert!(!config.gpu_acceleration);
    }

    #[test]
    fn test_default_optimization_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.strategy, OptimizationStrategy::RandomSearch);
        assert_eq!(config.max_iterations, 100);
        assert!(!config.parallel);
    }
}
