use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive alerting engine providing advanced rule processing, notification delivery,
/// escalation management, and alert correlation for enterprise monitoring systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingImplementationConfig {
    /// Rule engine configuration
    pub rule_engine: RuleEngineConfig,
    /// Notification engine configuration
    pub notification_engine: NotificationEngineConfig,
    /// Escalation engine configuration
    pub escalation_engine: EscalationEngineConfig,
    /// Alert correlation configuration
    pub alert_correlation: AlertCorrelationConfig,
}

/// Rule engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleEngineConfig {
    /// Engine type
    pub engine_type: RuleEngineType,
    /// Rule compilation configuration
    pub rule_compilation: RuleCompilationConfig,
    /// Rule execution configuration
    pub rule_execution: RuleExecutionConfig,
    /// Rule optimization configuration
    pub rule_optimization: RuleOptimizationConfig,
}

/// Rule engine types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleEngineType {
    /// Simple rule engine
    Simple,
    /// Expert system
    Expert,
    /// Forward chaining
    Forward,
    /// Backward chaining
    Backward,
    /// Hybrid approach
    Hybrid,
}

/// Rule compilation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCompilationConfig {
    /// Compilation strategy
    pub compilation_strategy: CompilationStrategy,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable caching
    pub caching_enabled: bool,
}

/// Compilation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompilationStrategy {
    /// Immediate compilation
    Immediate,
    /// Lazy compilation
    Lazy,
    /// Just-in-time compilation
    JIT,
    /// Ahead-of-time compilation
    AOT,
}

/// Optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Advanced optimization
    Advanced,
    /// Aggressive optimization
    Aggressive,
}

/// Rule execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleExecutionConfig {
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Parallelization configuration
    pub parallelization: ParallelizationConfig,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Execution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Pipeline execution
    Pipeline,
    /// Event-driven execution
    EventDriven,
}

/// Parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConfig {
    /// Enable parallelization
    pub enabled: bool,
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Enable work stealing
    pub work_stealing: bool,
    /// Enable load balancing
    pub load_balancing: bool,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory in megabytes
    pub max_memory_mb: usize,
    /// Maximum CPU percentage
    pub max_cpu_percent: u8,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Maximum rules per execution
    pub max_rules_per_execution: usize,
}

/// Rule optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleOptimizationConfig {
    /// Enable optimization
    pub enabled: bool,
    /// Optimization techniques
    pub optimization_techniques: Vec<OptimizationTechnique>,
    /// Enable profiling
    pub profiling_enabled: bool,
    /// Enable adaptive optimization
    pub adaptive_optimization: bool,
}

/// Optimization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTechnique {
    /// Index hints
    IndexHints,
    /// Query rewriting
    QueryRewriting,
    /// Join optimization
    JoinOptimization,
    /// Predicate pushdown
    PredicatePushdown,
    /// Column pruning
    ColumnPruning,
    /// Vectorization
    Vectorization,
    /// Caching
    Caching,
    /// Memoization
    Memoization,
}

/// Notification engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationEngineConfig {
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Delivery configuration
    pub delivery: DeliveryConfig,
    /// Template engine
    pub template_engine: TemplateEngineConfig,
    /// Rate limiting
    pub rate_limiting: RateLimitingConfig,
}

/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel name
    pub name: String,
    /// Channel type
    pub channel_type: ChannelType,
    /// Configuration
    pub config: serde_json::Value,
    /// Enabled status
    pub enabled: bool,
}

/// Channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    /// Email notifications
    Email,
    /// SMS notifications
    SMS,
    /// Slack notifications
    Slack,
    /// Webhook notifications
    Webhook,
    /// Push notifications
    Push,
    /// Custom channel
    Custom(String),
}

/// Delivery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryConfig {
    /// Retry configuration
    pub retry: RetryConfiguration,
    /// Dead letter queue
    pub dead_letter_queue: DeadLetterQueueConfig,
    /// Delivery guarantees
    pub guarantees: DeliveryGuarantees,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfiguration {
    /// Maximum attempts
    pub max_attempts: u32,
    /// Initial delay
    pub initial_delay: Duration,
    /// Maximum delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

/// Dead letter queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterQueueConfig {
    /// Enable dead letter queue
    pub enabled: bool,
    /// Queue name
    pub queue_name: String,
    /// Retention period
    pub retention_period: Duration,
    /// Maximum queue size
    pub max_size: usize,
}

/// Delivery guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryGuarantees {
    /// At most once delivery
    AtMostOnce,
    /// At least once delivery
    AtLeastOnce,
    /// Exactly once delivery
    ExactlyOnce,
}

/// Template engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateEngineConfig {
    /// Engine type
    pub engine_type: TemplateEngineType,
    /// Variable resolution
    pub variable_resolution: VariableResolutionConfig,
    /// Security configuration
    pub security: TemplateSecurityConfig,
}

/// Template engine types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateEngineType {
    /// Handlebars template engine
    Handlebars,
    /// Jinja2 template engine
    Jinja2,
    /// Mustache template engine
    Mustache,
    /// Custom template engine
    Custom(String),
}

/// Variable resolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableResolutionConfig {
    /// Resolution strategy
    pub strategy: ResolutionStrategy,
    /// Caching enabled
    pub caching_enabled: bool,
    /// Default values
    pub default_values: HashMap<String, serde_json::Value>,
}

/// Resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Static resolution
    Static,
    /// Dynamic resolution
    Dynamic,
    /// Lazy resolution
    Lazy,
    /// Cached resolution
    Cached,
}

/// Template security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSecurityConfig {
    /// Sandbox enabled
    pub sandbox_enabled: bool,
    /// Allowed functions
    pub allowed_functions: Vec<String>,
    /// Forbidden patterns
    pub forbidden_patterns: Vec<String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Rate limiting enabled
    pub enabled: bool,
    /// Rate limit per minute
    pub rate_per_minute: u32,
    /// Burst allowance
    pub burst_allowance: u32,
    /// Adaptive limiting
    pub adaptive_limiting: AdaptiveLimitingConfig,
}

/// Adaptive limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLimitingConfig {
    /// Enabled status
    pub enabled: bool,
    /// Load threshold
    pub load_threshold: f64,
    /// Adjustment factor
    pub adjustment_factor: f64,
}

/// Escalation engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationEngineConfig {
    /// Escalation strategies
    pub strategies: Vec<EscalationStrategy>,
    /// Timing configuration
    pub timing: TimingConfiguration,
    /// Notification configuration
    pub notifications: EscalationNotificationConfig,
}

/// Escalation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationStrategy {
    /// Strategy name
    pub name: String,
    /// Trigger conditions
    pub trigger_conditions: Vec<TriggerCondition>,
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
}

/// Trigger condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerCondition {
    /// Condition type
    pub condition_type: String,
    /// Parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level number
    pub level: u32,
    /// Delay before escalation
    pub delay: Duration,
    /// Recipients
    pub recipients: Vec<String>,
    /// Actions
    pub actions: Vec<String>,
}

/// Timing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConfiguration {
    /// Base delay
    pub base_delay: Duration,
    /// Maximum delay
    pub max_delay: Duration,
    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Linear backoff
    Linear,
    /// Exponential backoff
    Exponential,
    /// Fixed backoff
    Fixed,
    /// Custom backoff
    Custom(String),
}

/// Escalation notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationNotificationConfig {
    /// Notification templates
    pub templates: HashMap<String, String>,
    /// Channel preferences
    pub channel_preferences: HashMap<String, Vec<String>>,
}

/// Alert correlation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCorrelationConfig {
    /// Correlation engines
    pub engines: Vec<CorrelationEngine>,
    /// Noise reduction
    pub noise_reduction: NoiseReductionConfig,
    /// Alert throttling
    pub throttling: AlertThrottlingConfig,
}

/// Correlation engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEngine {
    /// Engine name
    pub name: String,
    /// Engine type
    pub engine_type: CorrelationEngineType,
    /// Rules
    pub rules: Vec<CorrelationRule>,
}

/// Correlation engine types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationEngineType {
    /// Rule-based correlation
    RuleBased,
    /// Statistical correlation
    Statistical,
    /// Machine learning correlation
    MachineLearning,
    /// Temporal correlation
    Temporal,
}

/// Correlation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRule {
    /// Rule name
    pub name: String,
    /// Patterns
    pub patterns: Vec<String>,
    /// Time window
    pub time_window: Duration,
    /// Threshold
    pub threshold: f64,
    /// Actions
    pub actions: Vec<String>,
}

/// Noise reduction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseReductionConfig {
    /// Deduplication
    pub deduplication: AlertDeduplicationConfig,
    /// Filtering
    pub filtering: AlertFilteringConfig,
    /// ML filtering
    pub ml_filtering: MLFilteringConfig,
}

/// Alert deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertDeduplicationConfig {
    /// Enabled status
    pub enabled: bool,
    /// Deduplication window
    pub window: Duration,
    /// Key fields
    pub key_fields: Vec<String>,
}

/// Alert filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertFilteringConfig {
    /// Filter rules
    pub rules: Vec<FilterRule>,
    /// Default action
    pub default_action: FilterAction,
}

/// Filter rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRule {
    /// Rule name
    pub name: String,
    /// Conditions
    pub conditions: Vec<FilterCondition>,
    /// Action
    pub action: FilterAction,
}

/// Filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Field name
    pub field: String,
    /// Operator
    pub operator: String,
    /// Value
    pub value: serde_json::Value,
}

/// Filter action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterAction {
    /// Allow the alert
    Allow,
    /// Block the alert
    Block,
    /// Modify the alert
    Modify,
    /// Route the alert
    Route(String),
}

/// Machine learning filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLFilteringConfig {
    /// Enabled status
    pub enabled: bool,
    /// Model configuration
    pub model: MLModelConfig,
    /// Training data
    pub training_data: TrainingDataConfig,
    /// Feature engineering
    pub feature_engineering: FeatureEngineeringConfig,
}

/// Machine learning model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelConfig {
    /// Model type
    pub model_type: MLModelType,
    /// Parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Training frequency
    pub training_frequency: Duration,
}

/// Machine learning model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLModelType {
    /// Random Forest
    RandomForest,
    /// Support Vector Machine
    SVM,
    /// Neural Network
    NeuralNetwork,
    /// Gradient Boosting
    GradientBoosting,
    /// Custom model
    Custom(String),
}

/// Training data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataConfig {
    /// Data sources
    pub sources: Vec<String>,
    /// Labeling strategy
    pub labeling_strategy: LabelingStrategy,
    /// Data preprocessing
    pub preprocessing: DataPreprocessingConfig,
}

/// Labeling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LabelingStrategy {
    /// Manual labeling
    Manual,
    /// Semi-automatic labeling
    SemiAutomatic,
    /// Automatic labeling
    Automatic,
    /// Rule-based labeling
    RuleBased,
}

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreprocessingConfig {
    /// Normalization enabled
    pub normalization: bool,
    /// Feature selection
    pub feature_selection: FeatureSelectionConfig,
    /// Feature scaling
    pub feature_scaling: FeatureScalingConfig,
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    /// Selection method
    pub method: FeatureSelectionMethod,
    /// Number of features
    pub num_features: Option<usize>,
    /// Threshold
    pub threshold: Option<f64>,
}

/// Feature selection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    /// Variance threshold
    VarianceThreshold,
    /// Select K best
    SelectKBest,
    /// Recursive feature elimination
    RFE,
    /// Custom method
    Custom(String),
}

/// Feature scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScalingConfig {
    /// Scaling method
    pub method: FeatureScalingMethod,
    /// Parameters
    pub parameters: HashMap<String, f64>,
}

/// Feature scaling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureScalingMethod {
    /// Standard scaling
    Standard,
    /// Min-max scaling
    MinMax,
    /// Robust scaling
    Robust,
    /// Quantile uniform
    QuantileUniform,
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Feature extraction
    pub extraction: Vec<FeatureExtractionMethod>,
    /// Feature transformation
    pub transformation: Vec<FeatureTransformationMethod>,
    /// Feature combination
    pub combination: Vec<FeatureCombinationMethod>,
}

/// Feature extraction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureExtractionMethod {
    /// Text features
    TextFeatures,
    /// Time features
    TimeFeatures,
    /// Statistical features
    StatisticalFeatures,
    /// Custom extraction
    Custom(String),
}

/// Feature transformation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureTransformationMethod {
    /// Polynomial features
    Polynomial,
    /// Log transformation
    Log,
    /// Square root transformation
    Sqrt,
    /// Custom transformation
    Custom(String),
}

/// Feature combination methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureCombinationMethod {
    /// Product features
    Product,
    /// Ratio features
    Ratio,
    /// Difference features
    Difference,
    /// Custom combination
    Custom(String),
}

/// Alert throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThrottlingConfig {
    /// Throttling enabled
    pub enabled: bool,
    /// Global throttling
    pub global_throttling: GlobalThrottlingConfig,
    /// Per-source throttling
    pub per_source_throttling: PerSourceThrottlingConfig,
    /// Adaptive throttling
    pub adaptive_throttling: AdaptiveThrottlingConfig,
}

/// Global throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalThrottlingConfig {
    /// Maximum alerts per minute
    pub max_alerts_per_minute: u32,
    /// Burst allowance
    pub burst_allowance: u32,
}

/// Per-source throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerSourceThrottlingConfig {
    /// Maximum alerts per source per minute
    pub max_alerts_per_source_per_minute: u32,
    /// Source identification
    pub source_identification: SourceIdentificationMethod,
}

/// Source identification methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceIdentificationMethod {
    /// By hostname
    Hostname,
    /// By IP address
    IPAddress,
    /// By service name
    ServiceName,
    /// Custom identification
    Custom(String),
}

/// Adaptive throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThrottlingConfig {
    /// Enabled status
    pub enabled: bool,
    /// Load threshold
    pub load_threshold: f64,
    /// Adjustment algorithm
    pub adjustment_algorithm: ThrottlingAdjustmentAlgorithm,
}

/// Throttling adjustment algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThrottlingAdjustmentAlgorithm {
    /// Linear adjustment
    Linear,
    /// Exponential adjustment
    Exponential,
    /// PID controller
    PID,
    /// Custom algorithm
    Custom(String),
}

impl Default for AlertingImplementationConfig {
    fn default() -> Self {
        Self {
            rule_engine: RuleEngineConfig::default(),
            notification_engine: NotificationEngineConfig::default(),
            escalation_engine: EscalationEngineConfig::default(),
            alert_correlation: AlertCorrelationConfig::default(),
        }
    }
}

impl Default for RuleEngineConfig {
    fn default() -> Self {
        Self {
            engine_type: RuleEngineType::Simple,
            rule_compilation: RuleCompilationConfig {
                compilation_strategy: CompilationStrategy::Lazy,
                optimization_level: OptimizationLevel::Basic,
                caching_enabled: true,
            },
            rule_execution: RuleExecutionConfig {
                execution_strategy: ExecutionStrategy::Sequential,
                parallelization: ParallelizationConfig {
                    enabled: false,
                    thread_pool_size: 4,
                    work_stealing: false,
                    load_balancing: false,
                },
                resource_limits: ResourceLimits {
                    max_memory_mb: 1024,
                    max_cpu_percent: 80,
                    max_execution_time: Duration::from_secs(30),
                    max_rules_per_execution: 1000,
                },
            },
            rule_optimization: RuleOptimizationConfig {
                enabled: true,
                optimization_techniques: vec![OptimizationTechnique::Caching],
                profiling_enabled: false,
                adaptive_optimization: false,
            },
        }
    }
}

impl Default for NotificationEngineConfig {
    fn default() -> Self {
        Self {
            channels: vec![
                NotificationChannel {
                    name: "email".to_string(),
                    channel_type: ChannelType::Email,
                    config: serde_json::json!({}),
                    enabled: true,
                }
            ],
            delivery: DeliveryConfig {
                retry: RetryConfiguration {
                    max_attempts: 3,
                    initial_delay: Duration::from_secs(1),
                    max_delay: Duration::from_secs(300),
                    backoff_multiplier: 2.0,
                },
                dead_letter_queue: DeadLetterQueueConfig {
                    enabled: true,
                    queue_name: "dlq".to_string(),
                    retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
                    max_size: 10000,
                },
                guarantees: DeliveryGuarantees::AtLeastOnce,
            },
            template_engine: TemplateEngineConfig {
                engine_type: TemplateEngineType::Handlebars,
                variable_resolution: VariableResolutionConfig {
                    strategy: ResolutionStrategy::Dynamic,
                    caching_enabled: true,
                    default_values: HashMap::new(),
                },
                security: TemplateSecurityConfig {
                    sandbox_enabled: true,
                    allowed_functions: vec!["format".to_string(), "date".to_string()],
                    forbidden_patterns: vec!["eval".to_string(), "exec".to_string()],
                },
            },
            rate_limiting: RateLimitingConfig {
                enabled: true,
                rate_per_minute: 100,
                burst_allowance: 10,
                adaptive_limiting: AdaptiveLimitingConfig {
                    enabled: false,
                    load_threshold: 0.8,
                    adjustment_factor: 0.5,
                },
            },
        }
    }
}

impl Default for EscalationEngineConfig {
    fn default() -> Self {
        Self {
            strategies: Vec::new(),
            timing: TimingConfiguration {
                base_delay: Duration::from_secs(300), // 5 minutes
                max_delay: Duration::from_secs(3600), // 1 hour
                backoff_strategy: BackoffStrategy::Exponential,
            },
            notifications: EscalationNotificationConfig {
                templates: HashMap::new(),
                channel_preferences: HashMap::new(),
            },
        }
    }
}

impl Default for AlertCorrelationConfig {
    fn default() -> Self {
        Self {
            engines: Vec::new(),
            noise_reduction: NoiseReductionConfig {
                deduplication: AlertDeduplicationConfig {
                    enabled: true,
                    window: Duration::from_secs(300), // 5 minutes
                    key_fields: vec!["source".to_string(), "type".to_string()],
                },
                filtering: AlertFilteringConfig {
                    rules: Vec::new(),
                    default_action: FilterAction::Allow,
                },
                ml_filtering: MLFilteringConfig {
                    enabled: false,
                    model: MLModelConfig {
                        model_type: MLModelType::RandomForest,
                        parameters: HashMap::new(),
                        training_frequency: Duration::from_secs(24 * 3600), // Daily
                    },
                    training_data: TrainingDataConfig {
                        sources: Vec::new(),
                        labeling_strategy: LabelingStrategy::Manual,
                        preprocessing: DataPreprocessingConfig {
                            normalization: true,
                            feature_selection: FeatureSelectionConfig {
                                method: FeatureSelectionMethod::SelectKBest,
                                num_features: Some(10),
                                threshold: None,
                            },
                            feature_scaling: FeatureScalingConfig {
                                method: FeatureScalingMethod::Standard,
                                parameters: HashMap::new(),
                            },
                        },
                    },
                    feature_engineering: FeatureEngineeringConfig {
                        extraction: vec![FeatureExtractionMethod::StatisticalFeatures],
                        transformation: Vec::new(),
                        combination: Vec::new(),
                    },
                },
            },
            throttling: AlertThrottlingConfig {
                enabled: true,
                global_throttling: GlobalThrottlingConfig {
                    max_alerts_per_minute: 1000,
                    burst_allowance: 100,
                },
                per_source_throttling: PerSourceThrottlingConfig {
                    max_alerts_per_source_per_minute: 50,
                    source_identification: SourceIdentificationMethod::Hostname,
                },
                adaptive_throttling: AdaptiveThrottlingConfig {
                    enabled: false,
                    load_threshold: 0.8,
                    adjustment_algorithm: ThrottlingAdjustmentAlgorithm::Linear,
                },
            },
        }
    }
}