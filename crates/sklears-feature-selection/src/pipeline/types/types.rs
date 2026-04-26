//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Duration;

use super::types_2::{
    AcquisitionFunction, ImportanceGetter, ImputerConfig, PipelineConfiguration,
    PowerTransformerConfig, QuantileTransformerConfig, ScoringMetric, TrainedStep,
};

#[derive(Debug, Clone)]
/// FeatureOrigin
pub enum FeatureOrigin {
    /// Original
    Original(usize),
    /// Engineered
    Engineered {
        /// source_features
        source_features: Vec<usize>,
        /// operation
        operation: String,
    },
    /// Transformed
    Transformed {
        /// source_feature
        source_feature: usize,
        /// transformation
        transformation: String,
    },
}
#[derive(Debug, Clone)]
/// OutlierParams
pub struct OutlierParams {
    /// decision_function
    pub decision_function: Array1<f64>,
    /// threshold
    pub threshold: f64,
}
/// Information about a trained pipeline
#[derive(Debug, Clone)]
pub struct PipelineInfo {
    /// n_preprocessing_steps
    pub n_preprocessing_steps: usize,
    /// n_feature_engineering_steps
    pub n_feature_engineering_steps: usize,
    /// n_selection_methods
    pub n_selection_methods: usize,
    /// has_dimensionality_reduction
    pub has_dimensionality_reduction: bool,
    /// has_model_selection
    pub has_model_selection: bool,
    /// config
    pub config: PipelineConfiguration,
}
#[derive(Debug, Clone)]
/// LoggingLevel
pub enum LoggingLevel {
    /// None
    None,
    /// Error
    Error,
    /// Warning
    Warning,
    /// Info
    Info,
    /// Debug
    Debug,
    /// Trace
    Trace,
}
/// Feature mapping for tracking feature transformations
#[derive(Debug, Clone)]
pub struct FeatureMapping {
    /// original_features
    pub original_features: usize,
    /// final_features
    pub final_features: usize,
    /// feature_names
    pub feature_names: Vec<String>,
    /// feature_origins
    pub feature_origins: Vec<FeatureOrigin>,
    /// transformation_history
    pub transformation_history: Vec<TransformationStep>,
}
#[derive(Debug, Clone)]
/// CorrelationMethod
pub enum CorrelationMethod {
    /// Pearson
    Pearson,
    /// Spearman
    Spearman,
    /// Kendall
    Kendall,
}
#[derive(Debug, Clone)]
/// TransformationStep
pub struct TransformationStep {
    /// step_name
    pub step_name: String,
    /// input_features
    pub input_features: usize,
    /// output_features
    pub output_features: usize,
    /// transformation_type
    pub transformation_type: TransformationType,
}
#[derive(Debug, Clone)]
/// PowerParams
pub struct PowerParams {
    /// lambdas
    pub lambdas: Array1<f64>,
}
#[derive(Debug, Clone)]
/// PrefetchStrategy
pub enum PrefetchStrategy {
    /// None
    None,
    /// Sequential
    Sequential,
    /// Random
    Random,
    /// Adaptive
    Adaptive,
}
#[derive(Debug, Clone)]
/// OutlierMethod
pub enum OutlierMethod {
    /// IsolationForest
    IsolationForest,
    /// LocalOutlierFactor
    LocalOutlierFactor,
    /// OneClassSVM
    OneClassSVM,
    /// EllipticEnvelope
    EllipticEnvelope,
}
#[derive(Debug, Clone)]
/// TransformationType
pub enum TransformationType {
    /// OneToOne
    OneToOne,
    /// OneToMany
    OneToMany,
    /// ManyToOne
    ManyToOne,
    /// ManyToMany
    ManyToMany,
}
#[derive(Debug, Clone)]
/// WindowStatistic
pub enum WindowStatistic {
    /// Mean
    Mean,
    /// Std
    Std,
    /// Min
    Min,
    /// Max
    Max,
    /// Median
    Median,
    /// Skewness
    Skewness,
    /// Kurtosis
    Kurtosis,
}
/// Type-safe selection count specification
#[derive(Debug, Clone)]
pub enum SelectionCount {
    /// K
    K(usize),
    /// Percentile
    Percentile(f64),
    /// FDR
    FDR(f64),
    /// FPR
    FPR(f64),
    /// FWER
    FWER(f64),
}
#[derive(Debug, Clone)]
/// OutlierConfig
pub struct OutlierConfig {
    /// method
    pub method: OutlierMethod,
    /// threshold
    pub threshold: f64,
    /// contamination
    pub contamination: f64,
}
#[derive(Debug, Clone)]
/// PowerMethod
pub enum PowerMethod {
    /// YeoJohnson
    YeoJohnson,
    /// BoxCox
    BoxCox,
}
#[derive(Debug, Clone)]
/// ScalerParams
pub struct ScalerParams {
    /// mean
    pub mean: Array1<f64>,
    /// scale
    pub scale: Array1<f64>,
}
/// Optimization configuration for performance tuning
#[derive(Debug, Clone)]
pub struct OptimizationConfiguration {
    /// use_simd
    pub use_simd: bool,
    /// chunk_size
    pub chunk_size: usize,
    /// thread_pool_size
    pub thread_pool_size: Option<usize>,
    /// memory_pool_size
    pub memory_pool_size: usize,
    /// cache_size
    pub cache_size: usize,
    /// prefetch_strategy
    pub prefetch_strategy: PrefetchStrategy,
    /// vectorization_threshold
    pub vectorization_threshold: usize,
}
#[derive(Debug, Clone)]
/// RobustScalerParams
pub struct RobustScalerParams {
    /// center
    pub center: Array1<f64>,
    /// scale
    pub scale: Array1<f64>,
}
#[derive(Debug, Clone)]
/// ErrorHandling
pub enum ErrorHandling {
    /// Strict
    Strict,
    /// Graceful
    Graceful,
    /// Logging
    Logging,
}
#[derive(Debug, Clone)]
/// MissingValueIndicator
pub enum MissingValueIndicator {
    /// NaN
    NaN,
    /// Value
    Value(f64),
}
/// Individual preprocessing step in the pipeline
#[derive(Debug, Clone)]
pub enum PreprocessingStep {
    /// StandardScaler
    StandardScaler {
        /// config
        config: StandardScalerConfig,
        /// trained_params
        trained_params: Option<ScalerParams>,
    },
    /// RobustScaler
    RobustScaler {
        /// config
        config: RobustScalerConfig,
        /// trained_params
        trained_params: Option<RobustScalerParams>,
    },
    /// MinMaxScaler
    MinMaxScaler {
        /// config
        config: MinMaxScalerConfig,
        /// trained_params
        trained_params: Option<MinMaxScalerParams>,
    },
    /// QuantileTransformer
    QuantileTransformer {
        /// config
        config: QuantileTransformerConfig,
        /// trained_params
        trained_params: Option<QuantileParams>,
    },
    /// PowerTransformer
    PowerTransformer {
        /// config
        config: PowerTransformerConfig,
        /// trained_params
        trained_params: Option<PowerParams>,
    },
    /// MissingValueImputer
    MissingValueImputer {
        /// config
        config: ImputerConfig,
        /// trained_params
        trained_params: Option<ImputerParams>,
    },
    /// OutlierRemover
    OutlierRemover {
        /// config
        config: OutlierConfig,
        /// trained_params
        trained_params: Option<OutlierParams>,
    },
}
#[derive(Debug, Clone)]
/// RobustScalerConfig
pub struct RobustScalerConfig {
    /// with_centering
    pub with_centering: bool,
    /// with_scaling
    pub with_scaling: bool,
    /// quantile_range
    pub quantile_range: (f64, f64),
}
#[derive(Debug, Clone)]
/// MinMaxScalerConfig
pub struct MinMaxScalerConfig {
    /// feature_range
    pub feature_range: (f64, f64),
    /// clip
    pub clip: bool,
}
#[derive(Debug, Clone)]
/// StandardScalerConfig
pub struct StandardScalerConfig {
    /// with_mean
    pub with_mean: bool,
    /// with_std
    pub with_std: bool,
}
#[derive(Debug, Clone)]
/// MinMaxScalerParams
pub struct MinMaxScalerParams {
    /// min
    pub min: Array1<f64>,
    /// scale
    pub scale: Array1<f64>,
}
#[derive(Debug, Clone)]
/// UnivariateMethod
pub enum UnivariateMethod {
    /// Chi2
    Chi2,
    /// ANOVA
    ANOVA,
    /// MutualInfo
    MutualInfo,
    /// Correlation
    Correlation,
}
#[derive(Debug, Clone)]
/// RFEEstimator
pub enum RFEEstimator {
    /// SVM
    SVM,
    /// RandomForest
    RandomForest,
    /// LinearRegression
    LinearRegression,
    /// LogisticRegression
    LogisticRegression,
}
/// Pipeline metadata for tracking execution and performance
#[derive(Debug, Clone)]
pub struct PipelineMetadata {
    /// total_training_time
    pub total_training_time: Duration,
    /// total_transform_time
    pub total_transform_time: Duration,
    /// memory_usage_peak
    pub memory_usage_peak: usize,
    /// feature_reduction_ratio
    pub feature_reduction_ratio: f64,
    /// performance_metrics
    pub performance_metrics: HashMap<String, f64>,
    /// validation_results
    pub validation_results: Option<ValidationResults>,
}
#[derive(Debug, Clone)]
/// TreeEstimatorType
pub enum TreeEstimatorType {
    /// RandomForest
    RandomForest,
    /// ExtraTrees
    ExtraTrees,
    /// GradientBoosting
    GradientBoosting,
    /// AdaBoost
    AdaBoost,
}
#[derive(Debug, Clone)]
/// ValidationStrategy
pub enum ValidationStrategy {
    /// None
    None,
    /// Basic
    Basic,
    /// Comprehensive
    Comprehensive,
    /// Statistical
    Statistical,
}
/// Supporting enums and structs for configuration
#[derive(Debug, Clone)]
pub enum MemoryOptimization {
    /// None
    None,
    /// Conservative
    Conservative,
    /// Aggressive
    Aggressive,
}
#[derive(Debug, Clone)]
/// QuantileParams
pub struct QuantileParams {
    /// quantiles
    pub quantiles: Array2<f64>,
    /// references
    pub references: Array1<f64>,
}
#[derive(Debug, Clone)]
/// Distribution
pub enum Distribution {
    /// Uniform
    Uniform,
    /// Normal
    Normal,
}
/// Model selection step for choosing optimal features for specific models
#[derive(Debug, Clone)]
pub enum ModelSelectionStep {
    /// CrossValidationSelection
    CrossValidationSelection {
        /// estimator
        estimator: ModelEstimator,
        /// cv_folds
        cv_folds: usize,
        /// scoring
        scoring: ScoringMetric,
        /// feature_scores
        feature_scores: Option<Array1<f64>>,
    },
    /// ForwardSelection
    ForwardSelection {
        /// estimator
        estimator: ModelEstimator,
        /// max_features
        max_features: usize,
        /// scoring
        scoring: ScoringMetric,
        /// selected_features
        selected_features: Option<Vec<usize>>,
    },
    /// BackwardElimination
    BackwardElimination {
        /// estimator
        estimator: ModelEstimator,
        /// min_features
        min_features: usize,
        /// scoring
        scoring: ScoringMetric,
        /// remaining_features
        remaining_features: Option<Vec<usize>>,
    },
    /// StepwiseSelection
    StepwiseSelection {
        /// estimator
        estimator: ModelEstimator,
        /// direction
        direction: StepwiseDirection,
        /// p_enter
        p_enter: f64,
        /// p_remove
        p_remove: f64,
        /// selected_features
        selected_features: Option<Vec<usize>>,
    },
    /// BayesianOptimization
    BayesianOptimization {
        /// estimator
        estimator: ModelEstimator,
        /// acquisition_function
        acquisition_function: AcquisitionFunction,
        /// n_calls
        n_calls: usize,
        /// optimal_features
        optimal_features: Option<Vec<usize>>,
    },
}
#[derive(Debug, Clone)]
/// ICAAlgorithm
pub enum ICAAlgorithm {
    /// Parallel
    Parallel,
    /// Deflation
    Deflation,
}
#[derive(Debug, Clone)]
/// ImputerParams
pub struct ImputerParams {
    /// statistics
    pub statistics: Array1<f64>,
}
#[derive(Debug, Clone)]
/// BinningStrategy
pub enum BinningStrategy {
    /// Uniform
    Uniform,
    /// Quantile
    Quantile,
    /// KMeans
    KMeans,
}
/// Type-safe selection threshold specification
#[derive(Debug, Clone)]
pub enum SelectionThreshold {
    /// Mean
    Mean,
    /// Median
    Median,
    /// Absolute
    Absolute(f64),
    /// Percentile
    Percentile(f64),
    /// Auto
    Auto,
}
#[derive(Debug, Clone)]
/// ValidationResults
pub struct ValidationResults {
    /// cross_validation_scores
    pub cross_validation_scores: Vec<f64>,
    /// stability_scores
    pub stability_scores: Vec<f64>,
    /// robustness_scores
    pub robustness_scores: Vec<f64>,
    /// statistical_significance
    pub statistical_significance: bool,
}
/// Selection method configuration with type safety
#[derive(Debug, Clone)]
pub enum SelectionMethod {
    /// UnivariateFilter
    UnivariateFilter {
        /// method
        method: UnivariateMethod,
        /// k
        k: SelectionCount,
        /// score_func
        score_func: UnivariateScoreFunction,
    },
    /// RecursiveFeatureElimination
    RecursiveFeatureElimination {
        /// estimator
        estimator: RFEEstimator,
        /// n_features
        n_features: SelectionCount,
        /// step
        step: f64,
        /// importance_getter
        importance_getter: ImportanceGetter,
    },
    /// SelectFromModel
    SelectFromModel {
        /// estimator
        estimator: ModelEstimator,
        /// threshold
        threshold: SelectionThreshold,
        /// prefit
        prefit: bool,
        /// max_features
        max_features: Option<usize>,
    },
    /// VarianceThreshold
    VarianceThreshold {
        /// threshold
        threshold: f64,
        /// feature_variance
        feature_variance: Option<Array1<f64>>,
    },
    /// CorrelationFilter
    CorrelationFilter {
        /// threshold
        threshold: f64,
        /// method
        method: CorrelationMethod,
        /// correlation_matrix
        correlation_matrix: Option<Array2<f64>>,
    },
    /// MutualInformation
    MutualInformation {
        /// k
        k: SelectionCount,
        /// discrete_features
        discrete_features: Vec<bool>,
        /// random_state
        random_state: Option<u64>,
    },
    /// LASSO
    LASSO {
        /// alpha
        alpha: f64,
        /// max_iter
        max_iter: usize,
        /// tol
        tol: f64,
        /// coefficients
        coefficients: Option<Array1<f64>>,
    },
    /// ElasticNet
    ElasticNet {
        /// alpha
        alpha: f64,
        /// l1_ratio
        l1_ratio: f64,
        /// max_iter
        max_iter: usize,
        /// tol
        tol: f64,
        /// coefficients
        coefficients: Option<Array1<f64>>,
    },
    /// TreeBased
    TreeBased {
        /// estimator_type
        estimator_type: TreeEstimatorType,
        /// n_estimators
        n_estimators: usize,
        /// max_depth
        max_depth: Option<usize>,
        /// feature_importances
        feature_importances: Option<Array1<f64>>,
    },
    /// GeneticAlgorithm
    GeneticAlgorithm {
        /// population_size
        population_size: usize,
        /// n_generations
        n_generations: usize,
        /// mutation_rate
        mutation_rate: f64,
        /// crossover_rate
        crossover_rate: f64,
        /// best_individuals
        best_individuals: Option<Vec<Vec<bool>>>,
    },
    /// ParticleSwarmOptimization
    ParticleSwarmOptimization {
        /// n_particles
        n_particles: usize,
        /// n_iterations
        n_iterations: usize,
        /// inertia
        inertia: f64,
        /// cognitive
        cognitive: f64,
        /// social
        social: f64,
        /// best_positions
        best_positions: Option<Vec<Vec<f64>>>,
    },
    /// SimulatedAnnealing
    SimulatedAnnealing {
        /// initial_temp
        initial_temp: f64,
        /// cooling_rate
        cooling_rate: f64,
        /// min_temp
        min_temp: f64,
        /// max_iter
        max_iter: usize,
        /// current_solution
        current_solution: Option<Vec<bool>>,
    },
}
#[derive(Debug, Clone)]
/// UnivariateScoreFunction
pub enum UnivariateScoreFunction {
    /// Chi2
    Chi2,
    /// FClassif
    FClassif,
    /// FRegression
    FRegression,
    /// MutualInfoClassif
    MutualInfoClassif,
    /// MutualInfoRegression
    MutualInfoRegression,
}
#[derive(Debug, Clone)]
/// SVDSolver
pub enum SVDSolver {
    /// Auto
    Auto,
    /// Full
    Full,
    /// Arpack
    Arpack,
    /// Randomized
    Randomized,
}
#[derive(Debug, Clone)]
/// SVDAlgorithm
pub enum SVDAlgorithm {
    /// Randomized
    Randomized,
    /// Arpack
    Arpack,
}
#[derive(Debug, Clone)]
/// StepwiseDirection
pub enum StepwiseDirection {
    /// Forward
    Forward,
    /// Backward
    Backward,
    /// Both
    Both,
}
#[derive(Debug, Clone)]
/// ImputationStrategy
pub enum ImputationStrategy {
    /// Mean
    Mean,
    /// Median
    Median,
    /// Mode
    Mode,
    /// Constant
    Constant,
    /// KNN
    KNN,
    /// Iterative
    Iterative,
}
#[derive(Debug, Clone)]
/// ModelEstimator
pub enum ModelEstimator {
    /// LinearRegression
    LinearRegression,
    /// LogisticRegression
    LogisticRegression,
    /// RandomForest
    RandomForest,
    /// SVM
    SVM,
    /// XGBoost
    XGBoost,
    /// LightGBM
    LightGBM,
}
#[derive(Debug)]
/// Trained
pub struct Trained {
    _trained_steps: Vec<TrainedStep>,
    _feature_mapping: FeatureMapping,
    _pipeline_metadata: PipelineMetadata,
}
