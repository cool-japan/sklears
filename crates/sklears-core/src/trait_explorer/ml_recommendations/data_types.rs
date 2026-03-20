//! Core data types and configuration for ML-based trait recommendations
//!
//! This module contains the fundamental data structures, configuration types,
//! and core traits used throughout the ML recommendation system.

use crate::api_reference_generator::{MethodInfo, TraitInfo};
use crate::error::{Result, SklearsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// SciRS2 Compliance: Use scirs2_autograd for ndarray functionality
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, Ix1};

/// Configuration for ML recommendation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLRecommendationConfig {
    /// Maximum number of recommendations to return
    pub max_recommendations: usize,
    /// Minimum confidence threshold for recommendations
    pub min_confidence_threshold: f64,
    /// Enable collaborative filtering
    pub enable_collaborative_filtering: bool,
    /// Enable neural network embeddings
    pub enable_neural_embeddings: bool,
    /// Enable GPU acceleration
    pub enable_gpu_acceleration: bool,
    /// Enable SIMD optimization
    pub enable_simd_optimization: bool,
    /// Feature extraction dimensions
    pub feature_dimensions: usize,
    /// Neural network hidden layers
    pub neural_hidden_layers: Vec<usize>,
    /// Learning rate for neural networks
    pub learning_rate: f64,
    /// Number of epochs for training
    pub training_epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Regularization strength
    pub regularization_strength: f64,
    /// Dropout rate for neural networks
    pub dropout_rate: f64,
    /// Enable ensemble methods
    pub enable_ensemble_methods: bool,
    /// Model cache size
    pub model_cache_size: usize,
    /// Enable explainable AI
    pub enable_explainable_ai: bool,
}

impl Default for MLRecommendationConfig {
    fn default() -> Self {
        Self {
            max_recommendations: 10,
            min_confidence_threshold: 0.5,
            enable_collaborative_filtering: true,
            enable_neural_embeddings: true,
            enable_gpu_acceleration: false,
            enable_simd_optimization: false,
            feature_dimensions: 256,
            neural_hidden_layers: vec![512, 256, 128],
            learning_rate: 0.001,
            training_epochs: 100,
            batch_size: 32,
            regularization_strength: 0.01,
            dropout_rate: 0.2,
            enable_ensemble_methods: true,
            model_cache_size: 1000,
            enable_explainable_ai: true,
        }
    }
}

/// Context information about a trait for recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitContext {
    /// Name of the trait
    pub trait_name: String,
    /// Human-readable description
    pub description: String,
    /// Complexity score (0.0 to 1.0)
    pub complexity_score: f64,
    /// Usage frequency in codebases
    pub usage_frequency: u64,
    /// Performance impact score (0.0 to 1.0)
    pub performance_impact: f64,
    /// Learning curve difficulty (0.0 to 1.0)
    pub learning_curve_difficulty: f64,
    /// Whether the trait is experimental
    pub is_experimental: bool,
    /// Community adoption rate (0.0 to 1.0)
    pub community_adoption_rate: f64,
    /// Associated keywords/tags
    pub keywords: Vec<String>,
    /// Related crates
    pub related_crates: Vec<String>,
    /// Rust version requirements
    pub rust_version_requirement: Option<String>,
    /// Feature flags required
    pub feature_flags: Vec<String>,
    /// Documentation quality score (0.0 to 1.0)
    pub documentation_quality: f64,
    /// Stability level
    pub stability_level: StabilityLevel,
    /// API surface complexity
    pub api_complexity: ApiComplexity,
    /// Memory usage characteristics
    pub memory_characteristics: MemoryCharacteristics,
    /// Concurrency safety
    pub concurrency_safety: ConcurrencySafety,
}

/// Stability level of a trait
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StabilityLevel {
    Experimental,
    Unstable,
    Stable,
    Deprecated,
}

/// API complexity characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiComplexity {
    /// Number of methods in the trait
    pub method_count: usize,
    /// Number of associated types
    pub associated_type_count: usize,
    /// Number of generic parameters
    pub generic_parameter_count: usize,
    /// Has default implementations
    pub has_default_implementations: bool,
    /// Has blanket implementations
    pub has_blanket_implementations: bool,
    /// Requires unsafe code
    pub requires_unsafe: bool,
}

/// Memory usage characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCharacteristics {
    /// Typical memory overhead
    pub memory_overhead: MemoryOverhead,
    /// Allocation pattern
    pub allocation_pattern: AllocationPattern,
    /// Cache efficiency
    pub cache_efficiency: CacheEfficiency,
}

/// Memory overhead levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOverhead {
    None,
    Low,
    Medium,
    High,
}

/// Allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPattern {
    StackOnly,
    HeapOptional,
    HeapRequired,
    ZeroCopy,
}

/// Cache efficiency levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEfficiency {
    Excellent,
    Good,
    Average,
    Poor,
}

/// Concurrency safety characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencySafety {
    /// Thread safety level
    pub thread_safety: ThreadSafety,
    /// Requires synchronization
    pub requires_synchronization: bool,
    /// Lock-free implementation available
    pub lock_free_available: bool,
    /// Send + Sync bounds
    pub send_sync_bounds: SendSyncBounds,
}

/// Thread safety levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadSafety {
    NotThreadSafe,
    ThreadLocal,
    ThreadSafe,
    LockFree,
}

/// Send + Sync bounds characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendSyncBounds {
    /// Implements Send
    pub implements_send: bool,
    /// Implements Sync
    pub implements_sync: bool,
    /// Conditional bounds
    pub conditional_bounds: Vec<String>,
}

/// A trait recommendation with confidence and reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitRecommendation {
    /// Recommended trait combination
    pub trait_combination: Vec<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f64,
    /// Human-readable reasoning for the recommendation
    pub reasoning: String,
    /// Supporting evidence
    pub evidence: RecommendationEvidence,
    /// Estimated learning effort
    pub learning_effort: LearningEffort,
    /// Implementation examples
    pub implementation_examples: Vec<ImplementationExample>,
    /// Potential pitfalls
    pub potential_pitfalls: Vec<String>,
    /// Best practices
    pub best_practices: Vec<String>,
    /// Alternative approaches
    pub alternatives: Vec<AlternativeApproach>,
}

/// Evidence supporting a recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationEvidence {
    /// Statistical evidence from usage patterns
    pub statistical_evidence: StatisticalEvidence,
    /// Semantic similarity scores
    pub semantic_similarity: f64,
    /// Community feedback scores
    pub community_feedback: f64,
    /// Expert system reasoning
    pub expert_reasoning: Vec<String>,
    /// Performance benchmarks
    pub performance_benchmarks: Option<PerformanceBenchmark>,
}

/// Statistical evidence from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalEvidence {
    /// Co-occurrence frequency
    pub co_occurrence_frequency: f64,
    /// Success rate in similar contexts
    pub success_rate: f64,
    /// Sample size for statistics
    pub sample_size: usize,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Learning effort estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEffort {
    /// Estimated time to learn (in hours)
    pub estimated_hours: f64,
    /// Difficulty level
    pub difficulty_level: DifficultyLevel,
    /// Prerequisites
    pub prerequisites: Vec<String>,
    /// Recommended learning path
    pub learning_path: Vec<LearningStep>,
}

/// Difficulty levels for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// A step in the learning path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStep {
    /// Step description
    pub description: String,
    /// Estimated time for this step
    pub estimated_time: Duration,
    /// Resources for learning
    pub resources: Vec<LearningResource>,
}

/// Learning resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResource {
    /// Resource title
    pub title: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// URL or reference
    pub url: Option<String>,
    /// Quality rating (0.0 to 1.0)
    pub quality_rating: f64,
}

/// Types of learning resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    Documentation,
    Tutorial,
    Example,
    Video,
    Book,
    BlogPost,
    StackOverflow,
    GitHub,
}

/// Implementation example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationExample {
    /// Example title
    pub title: String,
    /// Example code
    pub code: String,
    /// Explanation
    pub explanation: String,
    /// Complexity level
    pub complexity: DifficultyLevel,
    /// Use case category
    pub use_case: String,
}

/// Alternative approach to implementing functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeApproach {
    /// Approach description
    pub description: String,
    /// Trade-offs compared to recommended approach
    pub trade_offs: Vec<TradeOff>,
    /// When to consider this alternative
    pub when_to_use: Vec<String>,
    /// Example implementation
    pub example: Option<ImplementationExample>,
}

/// Trade-off between different approaches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOff {
    /// Aspect being traded off
    pub aspect: String,
    /// Benefit description
    pub benefit: String,
    /// Cost description
    pub cost: String,
    /// Impact severity
    pub severity: ImpactSeverity,
}

/// Severity of impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactSeverity {
    Negligible,
    Minor,
    Moderate,
    Major,
    Critical,
}

/// Performance benchmark data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    /// Benchmark name
    pub name: String,
    /// Execution time metrics
    pub execution_time: ExecutionTimeMetrics,
    /// Memory usage metrics
    pub memory_usage: MemoryUsageMetrics,
    /// Throughput metrics
    pub throughput_metrics: Option<ThroughputMetrics>,
    /// Comparison with alternatives
    pub comparison: Vec<PerformanceComparison>,
}

/// Execution time performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTimeMetrics {
    /// Mean execution time
    pub mean_time: Duration,
    /// Standard deviation
    pub std_deviation: Duration,
    /// Median time
    pub median_time: Duration,
    /// 95th percentile
    pub percentile_95: Duration,
    /// 99th percentile
    pub percentile_99: Duration,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageMetrics {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Average memory usage
    pub average_memory: usize,
    /// Memory allocation count
    pub allocation_count: usize,
    /// Memory fragmentation score
    pub fragmentation_score: f64,
}

/// Throughput performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub operations_per_second: f64,
    /// Items processed per second
    pub items_per_second: f64,
    /// Bytes processed per second
    pub bytes_per_second: f64,
}

/// Performance comparison with alternatives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// Alternative implementation name
    pub alternative_name: String,
    /// Performance ratio (recommended / alternative)
    pub performance_ratio: f64,
    /// Memory ratio (recommended / alternative)
    pub memory_ratio: f64,
    /// Context where this comparison applies
    pub applicable_context: String,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    /// Text processing configuration
    pub text_processing: TextProcessingConfig,
    /// Embedding configuration
    pub embedding_config: EmbeddingConfig,
    /// Feature selection configuration
    pub feature_selection: FeatureSelectionConfig,
    /// Normalization configuration
    pub normalization: NormalizationConfig,
}

/// Text processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingConfig {
    /// Enable stemming
    pub enable_stemming: bool,
    /// Enable lemmatization
    pub enable_lemmatization: bool,
    /// Remove stop words
    pub remove_stop_words: bool,
    /// Minimum word length
    pub min_word_length: usize,
    /// Maximum word length
    pub max_word_length: usize,
    /// N-gram ranges
    pub ngram_ranges: Vec<(usize, usize)>,
    /// Term frequency weighting
    pub tf_weighting: TfWeighting,
    /// IDF weighting
    pub idf_weighting: bool,
}

/// Term frequency weighting schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TfWeighting {
    Raw,
    Log,
    DoubleNormalization,
    Binary,
}

/// Embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimensions
    pub dimensions: usize,
    /// Pre-trained model path
    pub pretrained_model: Option<String>,
    /// Fine-tuning configuration
    pub fine_tuning: Option<FineTuningConfig>,
    /// Context window size
    pub context_window: usize,
    /// Embedding pooling strategy
    pub pooling_strategy: PoolingStrategy,
}

/// Fine-tuning configuration for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Learning rate for fine-tuning
    pub learning_rate: f64,
    /// Number of fine-tuning epochs
    pub epochs: usize,
    /// Batch size for fine-tuning
    pub batch_size: usize,
    /// Frozen layers (count from bottom)
    pub frozen_layers: usize,
}

/// Pooling strategies for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingStrategy {
    Mean,
    Max,
    Sum,
    AttentionWeighted,
    LastToken,
    FirstToken,
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    /// Selection method
    pub selection_method: FeatureSelectionMethod,
    /// Number of features to select
    pub num_features: Option<usize>,
    /// Selection threshold
    pub selection_threshold: Option<f64>,
    /// Cross-validation folds for selection
    pub cv_folds: usize,
}

/// Feature selection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    VarianceThreshold,
    UnivariateSelection,
    RecursiveFeatureElimination,
    LassoRegularization,
    MutualInformation,
    Chi2Test,
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Normalization method
    pub method: NormalizationMethod,
    /// Apply to each feature independently
    pub per_feature: bool,
    /// Clipping bounds
    pub clipping_bounds: Option<(f64, f64)>,
}

/// Normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    StandardScaling,
    MinMaxScaling,
    RobustScaling,
    UnitVector,
    Quantile,
    PowerTransform,
}

/// Model training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingConfig {
    /// Training algorithm configuration
    pub algorithm_config: AlgorithmConfig,
    /// Validation configuration
    pub validation_config: ValidationConfig,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Hyperparameter optimization
    pub hyperparameter_optimization: Option<HyperparameterOptimizationConfig>,
}

/// Algorithm-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    /// Collaborative filtering configuration
    pub collaborative_filtering: Option<CollaborativeFilteringConfig>,
    /// Neural network configuration
    pub neural_network: Option<NeuralNetworkConfig>,
    /// Clustering configuration
    pub clustering: Option<ClusteringConfig>,
    /// Ensemble configuration
    pub ensemble: Option<EnsembleConfig>,
}

/// Collaborative filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeFilteringConfig {
    /// Number of factors for matrix factorization
    pub num_factors: usize,
    /// Regularization parameters
    pub regularization: (f64, f64),
    /// Number of iterations
    pub num_iterations: usize,
    /// Implicit feedback weight
    pub implicit_weight: f64,
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    /// Layer configurations
    pub layers: Vec<LayerConfig>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Loss function
    pub loss_function: LossFunction,
    /// Optimizer configuration
    pub optimizer: OptimizerConfig,
}

/// Neural network layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer type
    pub layer_type: LayerType,
    /// Number of units/neurons
    pub units: usize,
    /// Dropout rate
    pub dropout: Option<f64>,
    /// Batch normalization
    pub batch_normalization: bool,
}

/// Neural network layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Dense,
    Embedding,
    Recurrent,
    Attention,
    Convolutional,
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
    Softmax,
    Swish,
    GELU,
}

/// Loss functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    MeanSquaredError,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    Hinge,
    Huber,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    /// Learning rate
    pub learning_rate: f64,
    /// Learning rate schedule
    pub lr_schedule: Option<LearningRateSchedule>,
    /// Gradient clipping
    pub gradient_clipping: Option<f64>,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD { momentum: f64 },
    Adam { beta1: f64, beta2: f64, epsilon: f64 },
    AdamW { weight_decay: f64 },
    RMSprop { alpha: f64, epsilon: f64 },
}

/// Learning rate schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    StepDecay { step_size: usize, gamma: f64 },
    ExponentialDecay { gamma: f64 },
    CosineAnnealing { t_max: usize },
    ReduceOnPlateau { factor: f64, patience: usize },
}

/// Clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Clustering algorithm
    pub algorithm: ClusteringAlgorithm,
    /// Number of clusters
    pub num_clusters: Option<usize>,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Initialization method
    pub initialization: InitializationMethod,
}

/// Clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    KMeans,
    DBScan { eps: f64, min_samples: usize },
    HierarchicalClustering,
    GaussianMixture,
    SpectralClustering,
}

/// Distance metrics for clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Jaccard,
    Hamming,
}

/// Initialization methods for clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationMethod {
    Random,
    KMeansPlusPlus,
    Forgy,
    RandomPartition,
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Ensemble method
    pub ensemble_method: EnsembleMethod,
    /// Number of base estimators
    pub num_estimators: usize,
    /// Voting strategy
    pub voting_strategy: VotingStrategy,
    /// Bootstrap sampling
    pub bootstrap: bool,
}

/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    Bagging,
    Boosting,
    Stacking,
    VotingClassifier,
    RandomForest,
}

/// Voting strategies for ensembles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    Hard,
    Soft,
    Weighted { weights: Vec<f64> },
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validation method
    pub validation_method: ValidationMethod,
    /// Test split ratio
    pub test_split: f64,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Stratification
    pub stratify: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Validation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMethod {
    HoldOut,
    CrossValidation,
    TimeSeriesSplit,
    StratifiedKFold,
    LeaveOneOut,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Metric to monitor
    pub monitor_metric: String,
    /// Patience (epochs to wait)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
    /// Restore best weights
    pub restore_best_weights: bool,
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterOptimizationConfig {
    /// Optimization method
    pub optimization_method: OptimizationMethod,
    /// Number of trials
    pub num_trials: usize,
    /// Timeout per trial
    pub timeout_per_trial: Option<Duration>,
    /// Parameter space
    pub parameter_space: HashMap<String, ParameterRange>,
}

/// Hyperparameter optimization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMethod {
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
}

/// Parameter range for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    Continuous { min: f64, max: f64 },
    Integer { min: i64, max: i64 },
    Categorical { values: Vec<String> },
    Boolean,
}