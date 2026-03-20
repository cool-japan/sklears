//! Core Retry System Types and Traits
//!
//! This module provides the foundational types, traits, and error handling
//! for the sophisticated retry management system with SIMD acceleration,
//! machine learning integration, and adaptive optimization capabilities.

use sklears_core::error::Result as SklResult;
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, SystemTime},
};

/// Core retry strategy trait with pluggable implementations
pub trait RetryStrategy: Send + Sync {
    /// Determine if retry should be attempted
    fn should_retry(&self, context: &RetryContext) -> bool;

    /// Calculate delay before next retry
    fn calculate_delay(&self, attempt: u32, context: &RetryContext) -> Duration;

    /// Update strategy state based on attempt result
    fn update_state(&mut self, result: &RetryResult, context: &RetryContext);

    /// Get strategy configuration
    fn configuration(&self) -> StrategyConfiguration;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy capabilities
    fn capabilities(&self) -> StrategyCapabilities;
}

/// Backoff algorithm trait for delay calculations
pub trait BackoffAlgorithm: Send + Sync {
    /// Calculate backoff delay
    fn calculate_delay(&self, attempt: u32, base_delay: Duration) -> Duration;

    /// Get algorithm parameters
    fn parameters(&self) -> BackoffParameters;

    /// Update parameters based on performance
    fn update_parameters(&mut self, performance_feedback: &PerformanceDataPoint);

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfiguration {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Maximum total duration
    pub max_duration: Duration,
    /// Base delay between attempts
    pub base_delay: Duration,
    /// Strategy-specific parameters
    pub parameters: HashMap<String, String>,
}

/// Strategy capabilities
#[derive(Debug, Clone)]
pub struct StrategyCapabilities {
    /// Supports adaptive behavior
    pub adaptive: bool,
    /// Supports circuit breaking
    pub circuit_breaking: bool,
    /// Supports rate limiting
    pub rate_limiting: bool,
    /// Supports SIMD acceleration
    pub simd_acceleration: bool,
    /// Performance characteristics
    pub performance: PerformanceCharacteristics,
}

/// Performance characteristics
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    /// CPU usage level
    pub cpu_usage: PerformanceLevel,
    /// Memory usage level
    pub memory_usage: PerformanceLevel,
    /// Latency level
    pub latency: PerformanceLevel,
    /// Throughput level
    pub throughput: PerformanceLevel,
}

/// Performance level enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceLevel {
    Low,
    Medium,
    High,
    Excellent,
}

/// Backoff parameters
#[derive(Debug, Clone)]
pub struct BackoffParameters {
    /// Multiplier for exponential backoff
    pub multiplier: f64,
    /// Maximum delay cap
    pub max_delay: Duration,
    /// Jitter configuration
    pub jitter: JitterConfig,
    /// Randomization seed
    pub random_seed: Option<u64>,
}

/// Jitter configuration
#[derive(Debug, Clone)]
pub struct JitterConfig {
    /// Jitter type
    pub jitter_type: JitterType,
    /// Jitter amount (0.0 to 1.0)
    pub amount: f64,
}

/// Jitter type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum JitterType {
    None,
    Full,
    Equal,
    Decorrelated,
}

/// Retry context containing attempt history and metadata
#[derive(Debug, Clone)]
pub struct RetryContext {
    /// Unique identifier for retry sequence
    pub id: String,
    /// Current attempt number
    pub current_attempt: u32,
    /// Total attempts made
    pub total_attempts: u32,
    /// Context creation time
    pub created_at: SystemTime,
    /// Last attempt time
    pub last_attempt_at: Option<SystemTime>,
    /// Attempt history
    pub attempts: Vec<RetryAttempt>,
    /// Context metadata
    pub metadata: HashMap<String, String>,
    /// Error history
    pub errors: Vec<RetryError>,
    /// Performance data
    pub performance_data: Vec<PerformanceDataPoint>,
}

/// Individual retry attempt information
#[derive(Debug, Clone)]
pub struct RetryAttempt {
    /// Attempt number
    pub attempt_number: u32,
    /// Attempt timestamp
    pub timestamp: SystemTime,
    /// Attempt duration
    pub duration: Duration,
    /// Attempt result
    pub result: AttemptResult,
    /// Error information (if failed)
    pub error: Option<RetryError>,
    /// Attempt metadata
    pub metadata: HashMap<String, String>,
}

/// Attempt result enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum AttemptResult {
    Success,
    Failure,
    Timeout,
    CircuitOpen,
    RateLimited,
}

/// Performance data point for analytics
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Success rate
    pub success_rate: f64,
    /// Average duration
    pub avg_duration: Duration,
    /// Retry count
    pub retry_count: u32,
    /// Error rate by type
    pub error_rates: HashMap<String, f64>,
    /// Resource utilization
    pub resource_usage: ResourceUsage,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network usage in bytes/sec
    pub network_usage: u64,
    /// Thread count
    pub thread_count: u32,
}

/// Retry result type
pub type RetryResult = Result<(), RetryError>;

/// Comprehensive retry error enumeration
#[derive(Debug, Clone)]
pub enum RetryError {
    /// Network-related errors
    Network {
        message: String,
        error_code: Option<i32>,
        retry_after: Option<Duration>,
    },
    /// Service-related errors
    Service {
        message: String,
        status_code: Option<u16>,
        service_name: String,
    },
    /// Timeout errors
    Timeout {
        elapsed: Duration,
        operation_timeout: Duration,
    },
    /// Resource exhaustion errors
    ResourceExhaustion {
        resource_type: String,
        current_usage: f64,
        limit: f64,
    },
    /// Authentication/authorization errors
    Auth {
        message: String,
        auth_type: String,
    },
    /// Configuration errors
    Configuration {
        parameter: String,
        message: String,
    },
    /// Rate limiting errors
    RateLimit {
        limit_type: String,
        reset_time: Option<SystemTime>,
    },
    /// Circuit breaker errors
    CircuitOpen {
        circuit_id: String,
        open_until: SystemTime,
    },
    /// Custom application errors
    Custom {
        message: String,
        error_code: String,
    },
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Strategy name to use
    pub strategy: String,
    /// Backoff algorithm name
    pub backoff_algorithm: String,
    /// Maximum attempts
    pub max_attempts: u32,
    /// Maximum total duration
    pub max_duration: Duration,
    /// Base delay
    pub base_delay: Duration,
    /// Timeout per attempt
    pub timeout: Duration,
    /// Retry conditions
    pub retry_conditions: Vec<RetryCondition>,
    /// Circuit breaker settings
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// Rate limiting settings
    pub rate_limiting: Option<RateLimitConfig>,
}

/// Retry condition for determining when to retry
#[derive(Debug, Clone)]
pub struct RetryCondition {
    pub condition_type: ConditionType,
    pub parameters: HashMap<String, String>,
}

/// Condition type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionType {
    ErrorType,
    StatusCode,
    Custom,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold
    pub failure_threshold: u32,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Success threshold for recovery
    pub success_threshold: u32,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per time window
    pub requests: u32,
    /// Time window duration
    pub window: Duration,
    /// Rate limiting algorithm
    pub algorithm: RateLimitAlgorithm,
}

/// Rate limiting algorithm enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitAlgorithm {
    FixedWindow,
    SlidingWindow,
    TokenBucket,
    LeakyBucket,
}

/// Default implementations
impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            strategy: "exponential".to_string(),
            backoff_algorithm: "exponential".to_string(),
            max_attempts: 3,
            max_duration: Duration::from_secs(60),
            base_delay: Duration::from_millis(100),
            timeout: Duration::from_secs(10),
            retry_conditions: vec![],
            circuit_breaker: None,
            rate_limiting: None,
        }
    }
}

impl Default for BackoffParameters {
    fn default() -> Self {
        Self {
            multiplier: 2.0,
            max_delay: Duration::from_secs(300),
            jitter: JitterConfig {
                jitter_type: JitterType::Full,
                amount: 0.1,
            },
            random_seed: None,
        }
    }
}

impl Default for StrategyConfiguration {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            max_duration: Duration::from_secs(60),
            base_delay: Duration::from_millis(100),
            parameters: HashMap::new(),
        }
    }
}

/// Priority enumeration for various system components
#[derive(Debug, Clone, PartialEq)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Action type enumeration for recommendations and policies
#[derive(Debug, Clone, PartialEq)]
pub enum ActionType {
    Retry,
    CircuitBreak,
    RateLimit,
    Escalate,
    Log,
    Alert,
    Custom(String),
}

/// Optimization objective for adaptive systems
#[derive(Debug, Clone)]
pub struct OptimizationObjective {
    /// Objective name
    pub name: String,
    /// Objective type
    pub objective_type: ObjectiveType,
    /// Target value
    pub target: f64,
    /// Objective weight
    pub weight: f64,
}

/// Objective type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveType {
    Maximize,
    Minimize,
    Target,
}

/// Cache configuration for various system components
#[derive(Debug, Clone)]
pub struct CacheConfiguration {
    /// Cache TTL
    pub ttl: Duration,
    /// Maximum cache size
    pub max_size: usize,
    /// Enable caching
    pub enabled: bool,
}

/// Cache statistics for monitoring
#[derive(Debug, Default)]
pub struct CacheStatistics {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Cache evictions
    pub evictions: u64,
    /// Cache size
    pub size: usize,
}

/// Feature selection method enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureSelectionMethod {
    Correlation,
    MutualInformation,
    ChiSquare,
    ANOVA,
    RecursiveFeatureElimination,
}

impl fmt::Display for RetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RetryError::Network { message, .. } => write!(f, "Network error: {}", message),
            RetryError::Service { message, .. } => write!(f, "Service error: {}", message),
            RetryError::Timeout { elapsed, operation_timeout } => {
                write!(f, "Timeout error: operation took {:?}, timeout was {:?}", elapsed, operation_timeout)
            }
            RetryError::ResourceExhaustion { resource_type, current_usage, limit } => {
                write!(f, "Resource exhaustion: {} usage {:.2}% exceeds limit {:.2}%", resource_type, current_usage * 100.0, limit * 100.0)
            }
            RetryError::Auth { message, .. } => write!(f, "Authentication error: {}", message),
            RetryError::Configuration { parameter, message } => write!(f, "Configuration error in {}: {}", parameter, message),
            RetryError::RateLimit { limit_type, .. } => write!(f, "Rate limit exceeded: {}", limit_type),
            RetryError::CircuitOpen { circuit_id, .. } => write!(f, "Circuit breaker open: {}", circuit_id),
            RetryError::Custom { message, .. } => write!(f, "Custom error: {}", message),
        }
    }
}

impl std::error::Error for RetryError {}