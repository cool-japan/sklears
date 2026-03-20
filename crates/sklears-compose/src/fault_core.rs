//! Core Fault Tolerance Traits and Definitions
//!
//! This module provides the fundamental traits, types, and configurations that form
//! the foundation of the fault tolerance system. It defines the core abstractions
//! for fault detection, recovery, and resilience management.

use serde::{Deserialize, Serialize};
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Core fault tolerance manager trait for pluggable fault handling implementations
///
/// Provides a flexible interface for different fault tolerance strategies
/// that can detect failures, implement recovery mechanisms, and ensure system resilience.
pub trait FaultToleranceManager: Send + Sync {
    /// Initialize fault tolerance for a specific execution session
    ///
    /// # Arguments
    /// * `session_id` - Unique identifier for the execution session
    /// * `config` - Fault tolerance configuration
    ///
    /// # Returns
    /// A fault tolerance session handle for management and control
    fn initialize_fault_tolerance(
        &mut self,
        session_id: String,
        config: FaultToleranceConfig,
    ) -> SklResult<FaultToleranceSession>;

    /// Register a component for fault tolerance monitoring
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `component` - Component to monitor for faults
    ///
    /// # Returns
    /// Component handle for fault tolerance management
    fn register_component(
        &mut self,
        session_id: String,
        component: FaultToleranceComponent,
    ) -> SklResult<ComponentHandle>;

    /// Report a fault occurrence
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    /// * `fault` - Fault report with details
    ///
    /// # Returns
    /// Fault response indicating action taken
    fn report_fault(&mut self, session_id: String, fault: FaultReport) -> SklResult<FaultResponse>;

    /// Get the current status of a fault tolerance session
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    ///
    /// # Returns
    /// Current session status
    fn get_session_status(&self, session_id: String) -> SklResult<FaultToleranceSessionStatus>;

    /// Shutdown fault tolerance for a session
    ///
    /// # Arguments
    /// * `session_id` - Session identifier
    ///
    /// # Returns
    /// Shutdown result with final statistics
    fn shutdown_fault_tolerance(&mut self, session_id: String) -> SklResult<FaultToleranceReport>;
}

/// Fault tolerance session handle
///
/// Provides management and control for an active fault tolerance session
/// including session lifecycle, component management, and recovery coordination.
#[derive(Debug, Clone)]
pub struct FaultToleranceSession {
    /// Unique session identifier
    pub session_id: String,
    /// Session start time
    pub start_time: SystemTime,
    /// Fault tolerance configuration
    pub config: FaultToleranceConfig,
    /// Registered components
    pub components: Vec<ComponentHandle>,
    /// Current session status
    pub status: FaultToleranceSessionStatus,
    /// Active circuit breakers
    pub circuit_breakers: Vec<CircuitBreakerHandle>,
    /// Recovery history
    pub recovery_history: Vec<RecoveryHistoryEntry>,
    /// Session metadata
    pub metadata: FaultToleranceMetadata,
}

/// Fault tolerance session status
#[derive(Debug, Clone, PartialEq)]
pub enum FaultToleranceSessionStatus {
    /// Session initializing
    Initializing,
    /// Session active and monitoring
    Active,
    /// Session degraded (some components failing)
    Degraded { failed_components: usize },
    /// Session in recovery mode
    Recovery,
    /// Session suspended
    Suspended,
    /// Session shutting down
    ShuttingDown,
    /// Session terminated
    Terminated,
}

/// Comprehensive fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable fault tolerance globally
    pub enabled: bool,
    /// Fault detection sensitivity (0.0 to 1.0)
    pub sensitivity: f64,
    /// Maximum concurrent recovery operations
    pub max_concurrent_recoveries: usize,
    /// Global timeout for operations
    pub global_timeout: Duration,
    /// Recovery strategy configuration
    pub recovery_config: RecoveryConfig,
    /// Circuit breaker configuration
    pub circuit_breaker_config: CircuitBreakerConfig,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Bulkhead configuration
    pub bulkhead_config: BulkheadConfig,
    /// Health check configuration
    pub health_check_config: HealthCheckConfig,
    /// Performance monitoring configuration
    pub performance_config: PerformanceConfig,
    /// Advanced features configuration
    pub advanced_config: AdvancedConfig,
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Enable automatic recovery
    pub automatic_recovery: bool,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Maximum recovery attempts
    pub max_recovery_attempts: usize,
    /// Recovery strategies priority order
    pub strategy_priority: Vec<RecoveryStrategyType>,
    /// Recovery validation settings
    pub validation: RecoveryValidationConfig,
    /// Recovery escalation settings
    pub escalation: RecoveryEscalationConfig,
}

/// Recovery strategy type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecoveryStrategyType {
    /// Restart
    Restart,
    /// Failover
    Failover,
    /// Scale
    Scale,
    /// Reset
    Reset,
    /// Rollback
    Rollback,
    /// Manual
    Manual,
    /// Custom
    Custom(String),
}

/// Recovery validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryValidationConfig {
    /// Enable recovery validation
    pub enabled: bool,
    /// Validation timeout
    pub timeout: Duration,
    /// Validation criteria
    pub criteria: Vec<ValidationCriterion>,
    /// Validation depth
    pub depth: ValidationDepth,
}

/// Validation criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCriterion {
    /// Criterion name
    pub name: String,
    /// Criterion type
    pub criterion_type: CriterionType,
    /// Expected value
    pub expected_value: String,
    /// Tolerance
    pub tolerance: f64,
    /// Weight in overall validation
    pub weight: f64,
}

/// Criterion type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CriterionType {
    /// HealthCheck
    HealthCheck,
    /// PerformanceMetric
    PerformanceMetric,
    /// ResourceUtilization
    ResourceUtilization,
    /// BusinessMetric
    BusinessMetric,
    /// Custom
    Custom(String),
}

/// Validation depth enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationDepth {
    /// Shallow
    Shallow,
    /// Medium
    Medium,
    /// Deep
    Deep,
    /// Comprehensive
    Comprehensive,
}

/// Recovery escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryEscalationConfig {
    /// Enable escalation
    pub enabled: bool,
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    /// Escalation timeout
    pub timeout: Duration,
    /// Notification settings
    pub notifications: NotificationConfig,
}

/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level identifier
    pub level: u32,
    /// Level name
    pub name: String,
    /// Recovery strategies available at this level
    pub strategies: Vec<RecoveryStrategyType>,
    /// Timeout before escalating to next level
    pub timeout: Duration,
    /// Approval required
    pub requires_approval: bool,
    /// Notification channels
    pub notification_channels: Vec<String>,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable notifications
    pub enabled: bool,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Notification templates
    pub templates: HashMap<String, NotificationTemplate>,
    /// Rate limiting
    pub rate_limit: RateLimitConfig,
}

/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel name
    pub name: String,
    /// Channel type
    pub channel_type: ChannelType,
    /// Channel configuration
    pub config: HashMap<String, String>,
    /// Priority threshold
    pub priority_threshold: Priority,
}

/// Channel type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChannelType {
    /// Email
    Email,
    /// Slack
    Slack,
    /// PagerDuty
    PagerDuty,
    /// Webhook
    Webhook,
    /// SMS
    SMS,
    /// Custom
    Custom(String),
}

/// Priority enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Priority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

/// Notification template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTemplate {
    /// Template name
    pub name: String,
    /// Subject template
    pub subject: String,
    /// Body template
    pub body: String,
    /// Template variables
    pub variables: Vec<String>,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum notifications per time window
    pub max_per_window: usize,
    /// Time window duration
    pub window_duration: Duration,
    /// Burst allowance
    pub burst_allowance: usize,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breakers
    pub enabled: bool,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Success threshold for recovery
    pub success_threshold: usize,
    /// Timeout for half-open state
    pub timeout: Duration,
    /// Maximum calls in half-open state
    pub half_open_max_calls: usize,
    /// Circuit breaker policies
    pub policies: Vec<CircuitBreakerPolicy>,
    /// Failure detection configuration
    pub failure_detection: FailureDetectionConfig,
    /// Analytics configuration
    pub analytics: crate::circuit_breaker::analytics_engine::AnalyticsConfig,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
            policies: Vec::new(),
            failure_detection: FailureDetectionConfig::default(),
            analytics: crate::circuit_breaker::analytics_engine::AnalyticsConfig::default(),
        }
    }
}

/// Circuit breaker policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerPolicy {
    /// Policy name
    pub name: String,
    /// Component patterns this policy applies to
    pub component_patterns: Vec<String>,
    /// Failure detection configuration
    pub failure_detection: FailureDetectionConfig,
    /// Recovery configuration
    pub recovery_configuration: CircuitRecoveryConfig,
}

/// Failure detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetectionConfig {
    /// Failure patterns
    pub patterns: Vec<FailurePattern>,
    /// Sliding window size
    pub window_size: usize,
    /// Minimum requests threshold
    pub min_requests: usize,
    /// Statistical analysis configuration
    pub statistics: StatisticalConfig,
}

/// Failure pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    /// Pattern name
    pub name: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern expression
    pub expression: String,
    /// Pattern weight
    pub weight: f64,
}

/// Pattern type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    /// ErrorRate
    ErrorRate,
    /// ResponseTime
    ResponseTime,
    /// ResourceUtilization
    ResourceUtilization,
    /// Custom
    Custom(String),
}

/// Statistical configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    /// Statistical method
    pub method: StatisticalMethod,
    /// Confidence level
    pub confidence_level: f64,
    /// Outlier detection
    pub outlier_detection: bool,
    /// Trend analysis
    pub trend_analysis: bool,
}

/// Statistical method enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StatisticalMethod {
    /// Simple
    Simple,
    /// ExponentialMovingAverage
    ExponentialMovingAverage,
    /// WeightedAverage
    WeightedAverage,
    /// Percentile
    Percentile(f64),
    /// StandardDeviation
    StandardDeviation,
    /// Custom
    Custom(String),
}

/// Circuit recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitRecoveryConfig {
    /// Recovery strategy
    pub strategy: CircuitRecoveryStrategy,
    /// Progressive recovery settings
    pub progressive: ProgressiveRecoveryConfig,
    /// Health check configuration
    pub health_check: CircuitHealthCheckConfig,
}

/// Circuit recovery strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CircuitRecoveryStrategy {
    /// Immediate
    Immediate,
    /// Progressive
    Progressive,
    /// HealthCheckBased
    HealthCheckBased,
    /// TimeBased
    TimeBased,
    /// Custom
    Custom(String),
}

/// Progressive recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveRecoveryConfig {
    /// Initial percentage of requests to allow
    pub initial_percentage: f64,
    /// Increment percentage on successful checks
    pub increment_percentage: f64,
    /// Maximum percentage
    pub max_percentage: f64,
    /// Check interval
    pub check_interval: Duration,
}

/// Circuit health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitHealthCheckConfig {
    /// Health check endpoint
    pub endpoint: String,
    /// Health check interval
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Expected response
    pub expected_response: String,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Enable retry mechanism
    pub enabled: bool,
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay
    pub max_delay: Duration,
    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
    /// Retry conditions
    pub retry_conditions: Vec<RetryCondition>,
    /// Jitter configuration
    pub jitter: JitterConfig,
}

/// Backoff strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackoffStrategy {
    /// Fixed
    Fixed,
    /// Linear
    Linear,
    /// Exponential
    Exponential { multiplier: f64 },
    /// Fibonacci
    Fibonacci,
    /// Custom
    Custom(String),
}

/// Retry condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryCondition {
    /// Condition name
    pub name: String,
    /// Error patterns to retry on
    pub error_patterns: Vec<String>,
    /// HTTP status codes to retry on
    pub status_codes: Vec<u16>,
    /// Custom condition function
    pub custom_condition: Option<String>,
}

/// Jitter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterConfig {
    /// Enable jitter
    pub enabled: bool,
    /// Jitter type
    pub jitter_type: JitterType,
    /// Jitter amount (0.0 to 1.0)
    pub amount: f64,
}

/// Jitter type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JitterType {
    None,
    /// Full
    Full,
    /// Equal
    Equal,
    /// Decorrelated
    Decorrelated,
}

/// Bulkhead configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkheadConfig {
    /// Enable bulkhead isolation
    pub enabled: bool,
    /// Default isolation settings
    pub default_isolation: IsolationSettings,
    /// Component-specific isolation settings
    pub component_isolation: HashMap<String, IsolationSettings>,
    /// Resource pool configuration
    pub resource_pools: Vec<ResourcePoolConfig>,
}

/// Isolation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationSettings {
    /// Maximum concurrent calls
    pub max_concurrent_calls: usize,
    /// Queue size
    pub queue_size: usize,
    /// Queue timeout
    pub queue_timeout: Duration,
    /// Isolation type
    pub isolation_type: IsolationType,
}

/// Isolation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IsolationType {
    /// ThreadPool
    ThreadPool,
    /// Semaphore
    Semaphore,
    /// Actor
    Actor,
    /// Custom
    Custom(String),
}

/// Resource pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePoolConfig {
    /// Pool name
    pub name: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Pool size
    pub size: usize,
    /// Maximum pool size
    pub max_size: Option<usize>,
    /// Pool timeout
    pub timeout: Duration,
    /// Pool management strategy
    pub management_strategy: PoolManagementStrategy,
}

/// Resource type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceType {
    /// Thread
    Thread,
    /// Connection
    Connection,
    /// Memory
    Memory,
    /// CPU
    CPU,
    /// Custom
    Custom(String),
}

/// Pool management strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PoolManagementStrategy {
    /// FIFO
    FIFO,
    /// LIFO
    LIFO,
    /// LeastRecentlyUsed
    LeastRecentlyUsed,
    /// MostRecentlyUsed
    MostRecentlyUsed,
    /// Custom
    Custom(String),
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,
    /// Default health check interval
    pub default_interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Health check configurations by component type
    pub component_configs: HashMap<ComponentType, ComponentHealthConfig>,
    /// Global health thresholds
    pub thresholds: HealthThresholds,
}

/// Component health configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealthConfig {
    /// Health check interval
    pub interval: Duration,
    /// Health check type
    pub check_type: HealthCheckType,
    /// Health check endpoint
    pub endpoint: String,
    /// Expected response
    pub expected_response: Option<String>,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Success threshold
    pub success_threshold: usize,
}

/// Health check type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthCheckType {
    /// Http
    Http {
        method: String,
        headers: HashMap<String, String>,
    },
    /// Tcp
    Tcp { host: String, port: u16 },
    /// Function
    Function { function_name: String },
    /// Resource
    Resource { resource_type: String },
    /// Custom
    Custom { check_name: String },
}

/// Health thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthThresholds {
    /// Critical health threshold
    pub critical: f64,
    /// Warning health threshold
    pub warning: f64,
    /// Good health threshold
    pub good: f64,
    /// Excellent health threshold
    pub excellent: f64,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
    /// Resource monitoring configuration
    pub resource_monitoring: ResourceMonitoringConfig,
    /// Performance optimization settings
    pub optimization: PerformanceOptimizationConfig,
}

/// Performance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Response time thresholds
    pub response_time: ThresholdConfig,
    /// Throughput thresholds
    pub throughput: ThresholdConfig,
    /// Error rate thresholds
    pub error_rate: ThresholdConfig,
    /// CPU utilization thresholds
    pub cpu_utilization: ThresholdConfig,
    /// Memory utilization thresholds
    pub memory_utilization: ThresholdConfig,
}

/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
    /// Measurement unit
    pub unit: String,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Monitor CPU usage
    pub monitor_cpu: bool,
    /// Monitor memory usage
    pub monitor_memory: bool,
    /// Monitor disk usage
    pub monitor_disk: bool,
    /// Monitor network usage
    pub monitor_network: bool,
    /// Custom resource monitors
    pub custom_monitors: Vec<CustomResourceMonitor>,
}

/// Custom resource monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomResourceMonitor {
    /// Monitor name
    pub name: String,
    /// Monitor type
    pub monitor_type: String,
    /// Configuration
    pub config: HashMap<String, String>,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimizationConfig {
    /// Enable automatic optimization
    pub auto_optimization: bool,
    /// Optimization strategies
    pub strategies: Vec<OptimizationStrategy>,
    /// Optimization interval
    pub optimization_interval: Duration,
    /// Learning configuration
    pub learning: LearningConfig,
}

/// Optimization strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationStrategy {
    /// LoadBalancing
    LoadBalancing,
    /// Caching
    Caching,
    /// ResourceScaling
    ResourceScaling,
    /// RequestBatching
    RequestBatching,
    /// Custom
    Custom(String),
}

/// Learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable machine learning
    pub enabled: bool,
    /// Learning algorithm
    pub algorithm: LearningAlgorithm,
    /// Training data retention
    pub data_retention: Duration,
    /// Model update interval
    pub update_interval: Duration,
}

/// Learning algorithm enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LearningAlgorithm {
    /// LinearRegression
    LinearRegression,
    /// DecisionTree
    DecisionTree,
    /// NeuralNetwork
    NeuralNetwork,
    /// ReinforcementLearning
    ReinforcementLearning,
    /// Custom
    Custom(String),
}

/// Advanced configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// Enable predictive analytics
    pub predictive_analytics: bool,
    /// Enable chaos engineering
    pub chaos_engineering: bool,
    /// Enable security monitoring
    pub security_monitoring: bool,
    /// Enable compliance reporting
    pub compliance_reporting: bool,
    /// Custom advanced features
    pub custom_features: HashMap<String, bool>,
}

/// Fault tolerance component
#[derive(Debug, Clone)]
pub struct FaultToleranceComponent {
    /// Component type
    pub component_type: ComponentType,
    /// Component identifier
    pub component_id: String,
    /// Component name
    pub name: String,
    /// Component description
    pub description: String,
    /// Health check configuration
    pub health_config: ComponentHealthConfig,
    /// Recovery configuration
    pub recovery_config: ComponentRecoveryConfig,
    /// Fault tolerance policies
    pub policies: Vec<FaultTolerancePolicy>,
    /// Component metadata
    pub metadata: ComponentMetadata,
    /// Component dependencies
    pub dependencies: Vec<String>,
    /// Component tags
    pub tags: Vec<String>,
}

/// Component type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash, Eq)]
pub enum ComponentType {
    /// Execution engine core
    ExecutionEngine,
    /// Task scheduler
    TaskScheduler,
    /// Resource manager
    ResourceManager,
    /// Monitoring system
    MonitoringSystem,
    /// Storage system
    StorageSystem,
    /// Network interface
    NetworkInterface,
    /// Database connection
    Database,
    /// Message queue
    MessageQueue,
    /// Cache system
    Cache,
    /// External service
    ExternalService { service_name: String },
    /// Custom component
    Custom { type_name: String },
}

/// Component recovery configuration
#[derive(Debug, Clone)]
pub struct ComponentRecoveryConfig {
    /// Enable automatic recovery
    pub automatic_recovery: bool,
    /// Recovery strategies
    pub strategies: Vec<RecoveryStrategy>,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Maximum recovery attempts
    pub max_recovery_attempts: usize,
    /// Recovery cooldown period
    pub cooldown_period: Duration,
    /// Recovery validation
    pub validation: RecoveryValidationConfig,
}

/// Recovery strategy
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Restart component
    Restart {
        restart_delay: Duration,
        cleanup_before_restart: bool,
    },
    /// Replace with backup
    Failover {
        backup_component: String,
        failover_delay: Duration,
    },
    /// Scale horizontally
    Scale {
        scale_factor: f64,
        scale_timeout: Duration,
    },
    /// Reset to known good state
    Reset {
        checkpoint: String,
        reset_timeout: Duration,
    },
    /// Manual intervention required
    Manual {
        notification_channels: Vec<String>,
        instructions: String,
    },
    /// Custom recovery strategy
    Custom {
        strategy_name: String,
        parameters: HashMap<String, String>,
    },
}

/// Fault tolerance policy
#[derive(Debug, Clone)]
pub enum FaultTolerancePolicy {
    /// Retry policy for transient failures
    RetryPolicy {
        max_attempts: usize,
        backoff_strategy: BackoffStrategy,
        retry_conditions: Vec<RetryCondition>,
    },
    /// Circuit breaker policy
    CircuitBreakerPolicy {
        failure_threshold: usize,
        recovery_timeout: Duration,
        half_open_max_calls: usize,
    },
    /// Bulkhead policy for resource isolation
    BulkheadPolicy {
        max_concurrent_calls: usize,
        queue_size: usize,
        timeout: Duration,
    },
    /// Fallback policy
    FallbackPolicy {
        fallback_action: FallbackAction,
        fallback_conditions: Vec<FallbackCondition>,
    },
    /// Timeout policy
    TimeoutPolicy {
        timeout: Duration,
        timeout_action: TimeoutAction,
    },
    /// Rate limiting policy
    RateLimitPolicy {
        max_requests: usize,
        time_window: Duration,
        rate_limit_action: RateLimitAction,
    },
    /// Custom policy
    CustomPolicy {
        policy_name: String,
        parameters: HashMap<String, String>,
    },
}

/// Fallback action enumeration
#[derive(Debug, Clone)]
pub enum FallbackAction {
    /// Return default value
    DefaultValue { value: String },
    /// Call alternative service
    AlternativeService { service_name: String },
    /// Use cached response
    CachedResponse { cache_key: String },
    /// Queue for later processing
    QueueRequest { queue_name: String },
    /// Manual action required
    Manual { action: ManualAction },
    /// Custom fallback
    Custom {
        action_name: String,
        parameters: HashMap<String, String>,
    },
}

/// Fallback condition
#[derive(Debug, Clone)]
pub struct FallbackCondition {
    /// Condition name
    pub name: String,
    /// Trigger patterns
    pub patterns: Vec<String>,
    /// Condition priority
    pub priority: i32,
}

/// Timeout action enumeration
#[derive(Debug, Clone)]
pub enum TimeoutAction {
    /// Cancel operation
    Cancel,
    /// Return partial result
    PartialResult,
    /// Retry with extended timeout
    ExtendTimeout { extension: Duration },
    /// Fallback to alternative
    Fallback { action: FallbackAction },
    /// Custom timeout action
    Custom { action_name: String },
}

/// Rate limit action enumeration
#[derive(Debug, Clone)]
pub enum RateLimitAction {
    /// Reject request
    Reject,
    /// Queue request
    Queue { max_queue_size: usize },
    /// Delay request
    Delay { delay: Duration },
    /// Throttle request
    Throttle { factor: f64 },
    /// Custom action
    Custom { action_name: String },
}

/// Manual action
#[derive(Debug, Clone)]
pub struct ManualAction {
    /// Action identifier
    pub action_id: String,
    /// Action description
    pub description: String,
    /// Required skills/roles
    pub required_skills: Vec<String>,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Priority level
    pub priority: i32,
    /// Instructions
    pub instructions: String,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Component metadata
#[derive(Debug, Clone)]
pub struct ComponentMetadata {
    /// Component version
    pub version: String,
    /// Component owner
    pub owner: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last updated timestamp
    pub updated_at: SystemTime,
    /// Environment
    pub environment: String,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Component handle for fault tolerance management
#[derive(Debug, Clone)]
pub struct ComponentHandle {
    /// Component unique identifier
    pub id: String,
    /// Component type
    pub component_type: ComponentType,
    /// Health check configuration
    pub health_config: ComponentHealthConfig,
    /// Recovery configuration
    pub recovery_config: ComponentRecoveryConfig,
    /// Active policies
    pub policies: Vec<FaultTolerancePolicy>,
    /// Component metadata
    pub metadata: ComponentMetadata,
    /// Registration timestamp
    pub registered_at: SystemTime,
    /// Last health check timestamp
    pub last_health_check: SystemTime,
}

/// Component health status
#[derive(Debug, Clone)]
pub enum ComponentHealth {
    /// Component healthy and operational
    Healthy {
        uptime: Duration,
        performance_score: f64,
    },
    /// Component degraded but functional
    Degraded { reason: String, impact_level: f64 },
    /// Component unhealthy but recoverable
    Unhealthy {
        error_count: usize,
        last_error: String,
    },
    /// Component failed and needs recovery
    Failed {
        failure_reason: String,
        failure_time: SystemTime,
    },
    /// Component unknown status
    Unknown { last_check: SystemTime },
}

/// Circuit breaker handle
#[derive(Debug, Clone)]
pub struct CircuitBreakerHandle {
    /// Circuit breaker identifier
    pub id: String,
    /// Circuit breaker name
    pub name: String,
    /// Current state
    pub state: CircuitBreakerState,
    /// Configuration
    pub config: CircuitBreakerConfig,
    /// Statistics
    pub stats: CircuitBreakerStats,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last state change timestamp
    pub last_state_change: SystemTime,
}

/// Circuit breaker state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CircuitBreakerState {
    /// Closed
    Closed,
    /// Open
    Open,
    /// HalfOpen
    HalfOpen,
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Consecutive failures
    pub consecutive_failures: u64,
    /// State changes
    pub state_changes: u64,
    /// Last failure time
    pub last_failure_time: Option<SystemTime>,
    /// Last success time
    pub last_success_time: Option<SystemTime>,
    /// Half-open state requests (for monitoring half-open behavior)
    pub half_open_requests: u64,
    /// Half-open state successful requests
    pub half_open_successes: u64,
}

/// Recovery history entry
#[derive(Debug, Clone)]
pub struct RecoveryHistoryEntry {
    /// Entry identifier
    pub id: String,
    /// Component identifier
    pub component_id: String,
    /// Recovery strategy used
    pub strategy: RecoveryStrategyType,
    /// Recovery start time
    pub start_time: SystemTime,
    /// Recovery end time
    pub end_time: Option<SystemTime>,
    /// Recovery result
    pub result: RecoveryResult,
    /// Recovery details
    pub details: String,
    /// Recovery metadata
    pub metadata: HashMap<String, String>,
}

/// Recovery result enumeration
#[derive(Debug, Clone)]
pub enum RecoveryResult {
    /// Success
    Success,
    /// Failure
    Failure { reason: String },
    /// Partial
    Partial { details: String },
    /// InProgress
    InProgress,
    /// Cancelled
    Cancelled,
}

/// Fault tolerance metadata
#[derive(Debug, Clone, Default)]
pub struct FaultToleranceMetadata {
    /// Session tags
    pub tags: Vec<String>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Configuration overrides
    pub config_overrides: HashMap<String, String>,
}

/// Fault report for reporting failures
#[derive(Debug, Clone)]
pub struct FaultReport {
    /// Fault identifier
    pub fault_id: String,
    /// Component identifier
    pub component_id: String,
    /// Fault type
    pub fault_type: FaultType,
    /// Fault severity
    pub severity: FaultSeverity,
    /// Fault timestamp
    pub timestamp: SystemTime,
    /// Fault description
    pub description: String,
    /// Error details
    pub error_details: ErrorDetails,
    /// Context information
    pub context: FaultContext,
}

/// Fault type enumeration
#[derive(Debug, Clone)]
pub enum FaultType {
    /// Timeout
    Timeout,
    /// ConnectionFailure
    ConnectionFailure,
    /// ServiceUnavailable
    ServiceUnavailable,
    /// ResourceExhaustion
    ResourceExhaustion,
    /// ConfigurationError
    ConfigurationError,
    /// SecurityViolation
    SecurityViolation,
    /// DataCorruption
    DataCorruption,
    /// PerformanceDegradation
    PerformanceDegradation,
    /// Custom
    Custom(String),
}

/// Fault severity enumeration
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum FaultSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

/// Error details
#[derive(Debug, Clone)]
pub struct ErrorDetails {
    /// Error code
    pub error_code: String,
    /// Error message
    pub error_message: String,
    /// Stack trace
    pub stack_trace: Option<String>,
    /// Additional error data
    pub additional_data: HashMap<String, String>,
}

/// Fault context
#[derive(Debug, Clone)]
pub struct FaultContext {
    /// Request identifier
    pub request_id: Option<String>,
    /// User identifier
    pub user_id: Option<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Environment context
    pub environment: HashMap<String, String>,
    /// System state
    pub system_state: HashMap<String, String>,
}

/// Fault response enumeration
#[derive(Debug, Clone)]
pub enum FaultResponse {
    /// Acknowledged
    Acknowledged,
    /// RecoveryInitiated
    RecoveryInitiated { strategy: RecoveryStrategyType },
    /// EscalationRequired
    EscalationRequired { level: u32 },
    /// ManualInterventionRequired
    ManualInterventionRequired,
    /// Ignored
    Ignored { reason: String },
}

/// Fault tolerance report
#[derive(Debug, Clone, Default)]
pub struct FaultToleranceReport {
    /// Session identifier
    pub session_id: String,
    /// Report type
    pub report_type: ReportType,
    /// Overall resilience score (0.0 to 1.0)
    pub resilience_score: f64,
    /// System availability (0.0 to 1.0)
    pub availability: f64,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Health summary
    pub health_summary: HealthSummary,
    /// Recovery summary
    pub recovery_summary: RecoverySummary,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

/// Report type enumeration
#[derive(Debug, Clone, Default)]
pub enum ReportType {
    #[default]
    Summary,
    /// Detailed
    Detailed,
    /// Comprehensive
    Comprehensive,
    /// Custom
    Custom(String),
}

impl Default for HealthSummary {
    fn default() -> Self {
        Self {
            overall_health: 1.0,
            critical_issues: 0,
            component_count: 0,
            healthy_components: 0,
        }
    }
}

impl Default for RecoverySummary {
    fn default() -> Self {
        Self {
            total_attempts: 0,
            successful_recoveries: 0,
            failed_recoveries: 0,
            avg_recovery_time: Duration::from_secs(0),
        }
    }
}

/// Performance metrics summary
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Average response time
    pub avg_response_time: Duration,
    /// Peak response time
    pub peak_response_time: Duration,
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Resource utilization
    pub resource_utilization: f64,
}

/// Health summary
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Overall health score
    pub overall_health: f64,
    /// Number of critical issues
    pub critical_issues: usize,
    /// Total component count
    pub component_count: usize,
    /// Healthy component count
    pub healthy_components: usize,
}

/// Recovery summary
#[derive(Debug, Clone)]
pub struct RecoverySummary {
    /// Total recovery attempts
    pub total_attempts: usize,
    /// Successful recoveries
    pub successful_recoveries: usize,
    /// Failed recoveries
    pub failed_recoveries: usize,
    /// Average recovery time
    pub avg_recovery_time: Duration,
}

/// Recommendation for system improvement
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation priority
    pub priority: Priority,
    /// Recommendation description
    pub description: String,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
    /// Expected impact
    pub expected_impact: String,
}

/// Recommendation type enumeration
#[derive(Debug, Clone)]
pub enum RecommendationType {
    /// Configuration
    Configuration,
    /// Infrastructure
    Infrastructure,
    /// Architecture
    Architecture,
    /// Monitoring
    Monitoring,
    /// Process
    Process,
    /// Custom
    Custom(String),
}

// Default implementations
impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity: 0.7,
            max_concurrent_recoveries: 5,
            global_timeout: Duration::from_secs(30),
            recovery_config: RecoveryConfig {
                automatic_recovery: true,
                recovery_timeout: Duration::from_secs(60),
                max_recovery_attempts: 3,
                strategy_priority: vec![
                    RecoveryStrategyType::Restart,
                    RecoveryStrategyType::Failover,
                    RecoveryStrategyType::Scale,
                ],
                validation: RecoveryValidationConfig {
                    enabled: true,
                    timeout: Duration::from_secs(10),
                    criteria: Vec::new(),
                    depth: ValidationDepth::Medium,
                },
                escalation: RecoveryEscalationConfig {
                    enabled: false,
                    levels: Vec::new(),
                    timeout: Duration::from_secs(300),
                    notifications: NotificationConfig {
                        enabled: false,
                        channels: Vec::new(),
                        templates: HashMap::new(),
                        rate_limit: RateLimitConfig {
                            max_per_window: 10,
                            window_duration: Duration::from_secs(60),
                            burst_allowance: 2,
                        },
                    },
                },
            },
            circuit_breaker_config: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                success_threshold: 3,
                timeout: Duration::from_secs(60),
                half_open_max_calls: 3,
                policies: Vec::new(),
                failure_detection: FailureDetectionConfig::default(),
                analytics: crate::circuit_breaker::analytics_engine::AnalyticsConfig::default(),
            },
            retry_config: RetryConfig {
                enabled: true,
                max_attempts: 3,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(30),
                backoff_strategy: BackoffStrategy::Exponential { multiplier: 2.0 },
                retry_conditions: Vec::new(),
                jitter: JitterConfig {
                    enabled: true,
                    jitter_type: JitterType::Equal,
                    amount: 0.1,
                },
            },
            bulkhead_config: BulkheadConfig {
                enabled: true,
                default_isolation: IsolationSettings {
                    max_concurrent_calls: 10,
                    queue_size: 20,
                    queue_timeout: Duration::from_secs(5),
                    isolation_type: IsolationType::Semaphore,
                },
                component_isolation: HashMap::new(),
                resource_pools: Vec::new(),
            },
            health_check_config: HealthCheckConfig {
                enabled: true,
                default_interval: Duration::from_secs(30),
                timeout: Duration::from_secs(5),
                component_configs: HashMap::new(),
                thresholds: HealthThresholds {
                    critical: 0.3,
                    warning: 0.6,
                    good: 0.8,
                    excellent: 0.95,
                },
            },
            performance_config: PerformanceConfig {
                enabled: true,
                collection_interval: Duration::from_secs(10),
                thresholds: PerformanceThresholds {
                    response_time: ThresholdConfig {
                        warning: 1000.0,
                        critical: 5000.0,
                        unit: "ms".to_string(),
                    },
                    throughput: ThresholdConfig {
                        warning: 10.0,
                        critical: 1.0,
                        unit: "rps".to_string(),
                    },
                    error_rate: ThresholdConfig {
                        warning: 0.05,
                        critical: 0.1,
                        unit: "ratio".to_string(),
                    },
                    cpu_utilization: ThresholdConfig {
                        warning: 0.8,
                        critical: 0.95,
                        unit: "ratio".to_string(),
                    },
                    memory_utilization: ThresholdConfig {
                        warning: 0.85,
                        critical: 0.95,
                        unit: "ratio".to_string(),
                    },
                },
                resource_monitoring: ResourceMonitoringConfig {
                    monitor_cpu: true,
                    monitor_memory: true,
                    monitor_disk: false,
                    monitor_network: false,
                    custom_monitors: Vec::new(),
                },
                optimization: PerformanceOptimizationConfig {
                    auto_optimization: false,
                    strategies: vec![OptimizationStrategy::LoadBalancing],
                    optimization_interval: Duration::from_secs(300),
                    learning: LearningConfig {
                        enabled: false,
                        algorithm: LearningAlgorithm::LinearRegression,
                        data_retention: Duration::from_secs(86400),
                        update_interval: Duration::from_secs(3600),
                    },
                },
            },
            advanced_config: AdvancedConfig {
                predictive_analytics: false,
                chaos_engineering: false,
                security_monitoring: false,
                compliance_reporting: false,
                custom_features: HashMap::new(),
            },
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_strategy: BackoffStrategy::Exponential { multiplier: 2.0 },
            retry_conditions: vec![
                /// RetryCondition
                RetryCondition {
                    name: "NetworkError".to_string(),
                    error_patterns: vec!["connection".to_string(), "timeout".to_string()],
                    status_codes: vec![],
                    custom_condition: None,
                },
                /// RetryCondition
                RetryCondition {
                    name: "ServiceUnavailable".to_string(),
                    error_patterns: vec!["unavailable".to_string()],
                    status_codes: vec![503],
                    custom_condition: None,
                },
            ],
            jitter: JitterConfig {
                enabled: true,
                jitter_type: JitterType::Equal,
                amount: 0.1,
            },
        }
    }
}

impl Default for FailureDetectionConfig {
    fn default() -> Self {
        Self {
            patterns: vec![],
            window_size: 100,
            min_requests: 10,
            statistics: StatisticalConfig {
                method: StatisticalMethod::Simple,
                confidence_level: 0.95,
                outlier_detection: true,
                trend_analysis: false,
            },
        }
    }
}

// Note: Core types are already public and don't need re-export
