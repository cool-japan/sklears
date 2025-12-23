//! Pipeline Execution Middleware Framework
//!
//! This module provides a comprehensive middleware system for pipeline execution,
//! allowing for flexible interception, modification, and extension of pipeline
//! behavior including authentication, validation, transformation, caching,
//! monitoring, and custom processing logic.

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Middleware execution context containing request/response data and metadata
#[derive(Debug)]
pub struct MiddlewareContext {
    /// Unique request identifier
    pub request_id: String,
    /// Request timestamp
    pub timestamp: SystemTime,
    /// Request metadata
    pub metadata: HashMap<String, String>,
    /// User/session information
    pub user_info: Option<UserInfo>,
    /// Processing state
    pub state: ContextState,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Custom data storage
    pub custom_data: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

/// User information for authentication and authorization
#[derive(Debug, Clone)]
pub struct UserInfo {
    /// User identifier
    pub user_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// User permissions
    pub permissions: Vec<String>,
    /// Session token
    pub session_token: Option<String>,
    /// Authentication method
    pub auth_method: AuthenticationMethod,
}

/// Authentication methods
#[derive(Debug, Clone)]
pub enum AuthenticationMethod {
    None,
    /// ApiKey
    ApiKey {
        key: String,
    },
    /// BearerToken
    BearerToken {
        token: String,
    },
    /// BasicAuth
    BasicAuth {
        username: String,
        password: String,
    },
    /// OAuth
    OAuth {
        provider: String,
        token: String,
    },
    /// Certificate
    Certificate {
        cert_fingerprint: String,
    },
    /// Custom
    Custom {
        method: String,
    },
}

/// Context processing state
#[derive(Debug, Clone)]
pub enum ContextState {
    /// Initializing
    Initializing,
    /// Processing
    Processing,
    /// Completed
    Completed,
    /// Error
    Error { message: String },
    /// Cancelled
    Cancelled,
}

/// Execution metrics for monitoring and profiling
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Processing duration
    pub duration: Option<Duration>,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Throughput (operations/second)
    pub throughput: f64,
    /// Error count
    pub error_count: usize,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Pipeline middleware trait
pub trait PipelineMiddleware: Send + Sync {
    /// Middleware name
    fn name(&self) -> &str;

    /// Execute before pipeline processing
    fn before_process(
        &self,
        context: &mut MiddlewareContext,
        input: &Array2<Float>,
    ) -> SklResult<()>;

    /// Execute after pipeline processing
    fn after_process(
        &self,
        context: &mut MiddlewareContext,
        output: &Array2<Float>,
    ) -> SklResult<()>;

    /// Handle errors during pipeline execution
    fn on_error(
        &self,
        context: &mut MiddlewareContext,
        error: &SklearsError,
    ) -> SklResult<ErrorAction>;

    /// Middleware priority (lower numbers execute first)
    fn priority(&self) -> i32 {
        100
    }

    /// Whether middleware should be executed
    fn should_execute(&self, context: &MiddlewareContext) -> bool {
        true
    }
}

/// Action to take when an error occurs
#[derive(Debug, Clone)]
pub enum ErrorAction {
    /// Continue processing
    Continue,
    /// Retry processing
    Retry {
        max_attempts: usize,
        delay: Duration,
    },
    /// Abort processing
    Abort,
    /// Fallback to alternative processing
    Fallback { fallback_data: Array2<Float> },
}

/// Middleware chain for executing multiple middleware components
pub struct MiddlewareChain {
    /// Registered middleware components
    middlewares: Vec<Box<dyn PipelineMiddleware>>,
    /// Chain configuration
    config: MiddlewareChainConfig,
    /// Execution statistics
    stats: MiddlewareStats,
}

/// Middleware chain configuration
#[derive(Debug, Clone)]
pub struct MiddlewareChainConfig {
    /// Enable parallel execution where possible
    pub parallel_execution: bool,
    /// Maximum execution time per middleware
    pub timeout_per_middleware: Duration,
    /// Global timeout for entire chain
    pub global_timeout: Duration,
    /// Continue on middleware errors
    pub continue_on_error: bool,
    /// Enable detailed logging
    pub detailed_logging: bool,
}

/// Middleware execution statistics
#[derive(Debug, Clone)]
pub struct MiddlewareStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Per-middleware statistics
    pub middleware_stats: HashMap<String, MiddlewareMetrics>,
}

/// Individual middleware metrics
#[derive(Debug, Clone)]
pub struct MiddlewareMetrics {
    /// Execution count
    pub execution_count: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Error count
    pub error_count: u64,
    /// Success rate
    pub success_rate: f64,
}

/// Authentication middleware
pub struct AuthenticationMiddleware {
    /// Authentication providers
    providers: HashMap<String, Box<dyn AuthenticationProvider>>,
    /// Authentication configuration
    config: AuthenticationConfig,
}

/// Authentication provider trait
pub trait AuthenticationProvider: Send + Sync {
    /// Provider name
    fn name(&self) -> &str;

    /// Authenticate user credentials
    fn authenticate(&self, credentials: &AuthenticationCredentials) -> SklResult<UserInfo>;

    /// Validate existing session
    fn validate_session(&self, session_token: &str) -> SklResult<bool>;

    /// Refresh authentication token
    fn refresh_token(&self, refresh_token: &str) -> SklResult<String>;
}

/// Authentication credentials
#[derive(Debug, Clone)]
pub enum AuthenticationCredentials {
    /// ApiKey
    ApiKey { key: String },
    /// BearerToken
    BearerToken { token: String },
    /// BasicAuth
    BasicAuth { username: String, password: String },
    /// OAuth
    OAuth { provider: String, token: String },
    /// Certificate
    Certificate { certificate: Vec<u8> },
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthenticationConfig {
    /// Required authentication methods
    pub required_methods: Vec<String>,
    /// Allow anonymous access
    pub allow_anonymous: bool,
    /// Session timeout
    pub session_timeout: Duration,
    /// Token refresh threshold
    pub token_refresh_threshold: Duration,
    /// Maximum failed attempts
    pub max_failed_attempts: usize,
    /// Lockout duration
    pub lockout_duration: Duration,
}

/// Authorization middleware
pub struct AuthorizationMiddleware {
    /// Access control policies
    policies: Vec<AccessPolicy>,
    /// Role-based access control
    rbac: RoleBasedAccessControl,
    /// Authorization configuration
    config: AuthorizationConfig,
}

/// Access control policy
#[derive(Debug, Clone)]
pub struct AccessPolicy {
    /// Policy name
    pub name: String,
    /// Resource pattern
    pub resource_pattern: String,
    /// Required permissions
    pub required_permissions: Vec<String>,
    /// Allowed roles
    pub allowed_roles: Vec<String>,
    /// Conditions
    pub conditions: Vec<AccessCondition>,
    /// Policy effect
    pub effect: PolicyEffect,
}

/// Access condition
#[derive(Debug, Clone)]
pub enum AccessCondition {
    /// TimeWindow
    TimeWindow { start: String, end: String },
    /// IpRange
    IpRange { cidr: String },
    /// UserAttribute
    UserAttribute { attribute: String, value: String },
    /// ResourceAttribute
    ResourceAttribute { attribute: String, value: String },
    /// Custom
    Custom { condition: String },
}

/// Policy effect
#[derive(Debug, Clone)]
pub enum PolicyEffect {
    /// Allow
    Allow,
    /// Deny
    Deny,
    /// Conditional
    Conditional,
}

/// Role-based access control
#[derive(Debug, Clone)]
pub struct RoleBasedAccessControl {
    /// Role definitions
    pub roles: HashMap<String, Role>,
    /// Permission definitions
    pub permissions: HashMap<String, Permission>,
    /// Role hierarchy
    pub role_hierarchy: HashMap<String, Vec<String>>,
}

/// Role definition
#[derive(Debug, Clone)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Role description
    pub description: String,
    /// Assigned permissions
    pub permissions: Vec<String>,
    /// Role metadata
    pub metadata: HashMap<String, String>,
}

/// Permission definition
#[derive(Debug, Clone)]
pub struct Permission {
    /// Permission name
    pub name: String,
    /// Permission description
    pub description: String,
    /// Resource type
    pub resource_type: String,
    /// Allowed actions
    pub actions: Vec<String>,
}

/// Authorization configuration
#[derive(Debug, Clone)]
pub struct AuthorizationConfig {
    /// Default policy effect
    pub default_effect: PolicyEffect,
    /// Enable role inheritance
    pub enable_role_inheritance: bool,
    /// Cache authorization decisions
    pub cache_decisions: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
}

/// Validation middleware
pub struct ValidationMiddleware {
    /// Input validators
    input_validators: Vec<Box<dyn InputValidator>>,
    /// Output validators
    output_validators: Vec<Box<dyn OutputValidator>>,
    /// Validation configuration
    config: ValidationConfig,
}

/// Input validation trait
pub trait InputValidator: Send + Sync {
    /// Validator name
    fn name(&self) -> &str;

    /// Validate input data
    fn validate(
        &self,
        input: &Array2<Float>,
        context: &MiddlewareContext,
    ) -> SklResult<ValidationResult>;

    /// Validation severity
    fn severity(&self) -> ValidationSeverity;
}

/// Output validation trait
pub trait OutputValidator: Send + Sync {
    /// Validator name
    fn name(&self) -> &str;

    /// Validate output data
    fn validate(
        &self,
        output: &Array2<Float>,
        context: &MiddlewareContext,
    ) -> SklResult<ValidationResult>;

    /// Validation severity
    fn severity(&self) -> ValidationSeverity;
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation passed
    pub valid: bool,
    /// Validation messages
    pub messages: Vec<ValidationMessage>,
    /// Suggested corrections
    pub corrections: Vec<ValidationCorrection>,
}

/// Validation message
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    /// Message text
    pub message: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Field or location
    pub field: Option<String>,
    /// Error code
    pub code: Option<String>,
}

/// Validation severity levels
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    /// Info
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Validation correction suggestion
#[derive(Debug, Clone)]
pub struct ValidationCorrection {
    /// Description of the correction
    pub description: String,
    /// Corrected value
    pub corrected_value: Option<Array2<Float>>,
    /// Confidence in correction
    pub confidence: f64,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Fail on validation errors
    pub fail_on_error: bool,
    /// Apply corrections automatically
    pub auto_correct: bool,
    /// Validation timeout
    pub timeout: Duration,
    /// Maximum corrections per request
    pub max_corrections: usize,
}

/// Transformation middleware
pub struct TransformationMiddleware {
    /// Pre-processing transformations
    pre_transformations: Vec<Box<dyn DataTransformer>>,
    /// Post-processing transformations
    post_transformations: Vec<Box<dyn DataTransformer>>,
    /// Transformation configuration
    config: TransformationConfig,
}

/// Data transformation trait
pub trait DataTransformer: Send + Sync {
    /// Transformer name
    fn name(&self) -> &str;

    /// Transform data
    fn transform(
        &self,
        data: &Array2<Float>,
        context: &MiddlewareContext,
    ) -> SklResult<Array2<Float>>;

    /// Check if transformation should be applied
    fn should_transform(&self, data: &Array2<Float>, context: &MiddlewareContext) -> bool;

    /// Get transformation metadata
    fn get_metadata(&self) -> TransformationMetadata;
}

/// Transformation metadata
#[derive(Debug, Clone)]
pub struct TransformationMetadata {
    /// Transformation type
    pub transformation_type: String,
    /// Input requirements
    pub input_requirements: Vec<String>,
    /// Output characteristics
    pub output_characteristics: Vec<String>,
    /// Performance impact
    pub performance_impact: PerformanceImpact,
}

/// Performance impact assessment
#[derive(Debug, Clone)]
pub enum PerformanceImpact {
    /// Minimal
    Minimal,
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Extreme
    Extreme,
}

/// Transformation configuration
#[derive(Debug, Clone)]
pub struct TransformationConfig {
    /// Enable parallel transformations
    pub parallel_transformations: bool,
    /// Transformation timeout
    pub timeout: Duration,
    /// Cache transformed data
    pub cache_results: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
}

/// Caching middleware
pub struct CachingMiddleware {
    /// Cache storage
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: CacheStats,
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cached data
    pub data: Array2<Float>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last accessed timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Entry metadata
    pub metadata: HashMap<String, String>,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size (entries)
    pub max_size: usize,
    /// Entry TTL
    pub ttl: Duration,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable cache statistics
    pub enable_stats: bool,
    /// Cache key strategy
    pub key_strategy: CacheKeyStrategy,
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    /// LRU
    LRU, // Least Recently Used
    /// LFU
    LFU, // Least Frequently Used
    /// FIFO
    FIFO, // First In, First Out
    /// TTL
    TTL, // Time To Live
    /// Random
    Random,
}

/// Cache key strategy
#[derive(Debug, Clone)]
pub enum CacheKeyStrategy {
    /// HashInput
    HashInput,
    /// HashInputAndContext
    HashInputAndContext,
    /// Custom
    Custom { generator: String },
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Hit ratio
    pub hit_ratio: f64,
    /// Total size (bytes)
    pub total_size: u64,
    /// Number of entries
    pub entry_count: usize,
    /// Evictions
    pub evictions: u64,
}

/// Monitoring middleware
pub struct MonitoringMiddleware {
    /// Metrics collectors
    collectors: Vec<Box<dyn MetricsCollector>>,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Alert manager
    alert_manager: AlertManager,
}

/// Metrics collector trait
pub trait MetricsCollector: Send + Sync {
    /// Collector name
    fn name(&self) -> &str;

    /// Collect metrics
    fn collect(&self, context: &MiddlewareContext, data: &Array2<Float>) -> SklResult<Vec<Metric>>;

    /// Get supported metric types
    fn supported_metrics(&self) -> Vec<String>;
}

/// Metric definition
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Metric type
    pub metric_type: MetricType,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Labels
    pub labels: HashMap<String, String>,
}

/// Metric types
#[derive(Debug, Clone)]
pub enum MetricType {
    /// Counter
    Counter,
    /// Gauge
    Gauge,
    /// Histogram
    Histogram,
    /// Summary
    Summary,
    /// Timer
    Timer,
}

/// Alert manager
#[derive(Debug, Clone)]
pub struct AlertManager {
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Active alerts
    pub active_alerts: Vec<Alert>,
    /// Alert channels
    pub channels: Vec<AlertChannel>,
}

/// Alert rule
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric to monitor
    pub metric: String,
    /// Threshold condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Evaluation interval
    pub evaluation_interval: Duration,
}

/// Alert condition
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold
    Threshold { operator: String, value: f64 },
    /// Range
    Range { min: f64, max: f64 },
    /// Rate
    Rate {
        change_percent: f64,
        time_window: Duration,
    },
    /// Anomaly
    Anomaly { sensitivity: f64 },
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    /// Info
    Info,
    /// Warning
    Warning,
    /// Critical
    Critical,
    /// Emergency
    Emergency,
}

/// Active alert
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Rule that triggered the alert
    pub rule_name: String,
    /// Current value
    pub current_value: f64,
    /// Alert message
    pub message: String,
    /// Triggered at
    pub triggered_at: SystemTime,
    /// Status
    pub status: AlertStatus,
}

/// Alert status
#[derive(Debug, Clone)]
pub enum AlertStatus {
    /// Triggered
    Triggered,
    /// Acknowledged
    Acknowledged,
    /// Resolved
    Resolved,
    /// Suppressed
    Suppressed,
}

/// Alert channel
#[derive(Debug, Clone)]
pub enum AlertChannel {
    /// Email
    Email { addresses: Vec<String> },
    /// Webhook
    Webhook { url: String },
    /// Slack
    Slack {
        webhook_url: String,
        channel: String,
    },
    /// Console
    Console,
    /// Log
    Log { file_path: String },
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable real-time monitoring
    pub real_time: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Enable alerting
    pub enable_alerting: bool,
    /// Alert evaluation interval
    pub alert_evaluation_interval: Duration,
}

impl MiddlewareChain {
    /// Create a new middleware chain
    #[must_use]
    pub fn new(config: MiddlewareChainConfig) -> Self {
        Self {
            middlewares: Vec::new(),
            config,
            stats: MiddlewareStats {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_execution_time: Duration::from_millis(0),
                middleware_stats: HashMap::new(),
            },
        }
    }

    /// Add middleware to the chain
    pub fn add_middleware(&mut self, middleware: Box<dyn PipelineMiddleware>) {
        self.middlewares.push(middleware);
        self.middlewares.sort_by_key(|m| m.priority());
    }

    /// Execute the middleware chain
    pub fn execute(
        &mut self,
        context: &mut MiddlewareContext,
        input: &Array2<Float>,
        processor: &dyn Fn(&Array2<Float>) -> SklResult<Array2<Float>>,
    ) -> SklResult<Array2<Float>> {
        let start_time = Instant::now();
        self.stats.total_requests += 1;

        // Execute before_process hooks
        for middleware in &self.middlewares {
            if middleware.should_execute(context) {
                if let Err(e) = middleware.before_process(context, input) {
                    let action = middleware.on_error(context, &e)?;
                    match action {
                        ErrorAction::Continue => {}
                        ErrorAction::Abort => {
                            self.stats.failed_requests += 1;
                            return Err(e);
                        }
                        ErrorAction::Retry {
                            max_attempts,
                            delay,
                        } => {
                            // Implement retry logic
                            std::thread::sleep(delay);
                            return self.execute(context, input, processor);
                        }
                        ErrorAction::Fallback { fallback_data } => {
                            return Ok(fallback_data);
                        }
                    }
                }
            }
        }

        // Execute main processor
        let result = processor(input)?;

        // Execute after_process hooks
        for middleware in &self.middlewares {
            if middleware.should_execute(context) {
                if let Err(e) = middleware.after_process(context, &result) {
                    let action = middleware.on_error(context, &e)?;
                    match action {
                        ErrorAction::Continue => {}
                        ErrorAction::Abort => {
                            self.stats.failed_requests += 1;
                            return Err(e);
                        }
                        _ => {}
                    }
                }
            }
        }

        // Update statistics
        let execution_time = start_time.elapsed();
        self.stats.successful_requests += 1;
        self.update_execution_stats(execution_time);

        context.state = ContextState::Completed;
        context.metrics.end_time = Some(Instant::now());
        context.metrics.duration = Some(execution_time);

        Ok(result)
    }

    /// Get execution statistics
    #[must_use]
    pub fn get_stats(&self) -> &MiddlewareStats {
        &self.stats
    }

    /// Update execution statistics
    fn update_execution_stats(&mut self, execution_time: Duration) {
        let total_time = self.stats.average_execution_time.as_nanos() as f64
            * (self.stats.total_requests - 1) as f64;
        let new_avg_nanos =
            (total_time + execution_time.as_nanos() as f64) / self.stats.total_requests as f64;
        self.stats.average_execution_time = Duration::from_nanos(new_avg_nanos as u64);
    }
}

impl AuthenticationMiddleware {
    /// Create new authentication middleware
    #[must_use]
    pub fn new(config: AuthenticationConfig) -> Self {
        Self {
            providers: HashMap::new(),
            config,
        }
    }

    /// Add authentication provider
    pub fn add_provider(&mut self, provider: Box<dyn AuthenticationProvider>) {
        self.providers.insert(provider.name().to_string(), provider);
    }

    /// Authenticate request
    pub fn authenticate(&self, credentials: &AuthenticationCredentials) -> SklResult<UserInfo> {
        for provider in self.providers.values() {
            if let Ok(user_info) = provider.authenticate(credentials) {
                return Ok(user_info);
            }
        }
        Err(SklearsError::InvalidInput(
            "Authentication failed".to_string(),
        ))
    }
}

impl PipelineMiddleware for AuthenticationMiddleware {
    fn name(&self) -> &'static str {
        "authentication"
    }

    fn before_process(
        &self,
        context: &mut MiddlewareContext,
        _input: &Array2<Float>,
    ) -> SklResult<()> {
        if !self.config.allow_anonymous && context.user_info.is_none() {
            return Err(SklearsError::InvalidInput(
                "Authentication required".to_string(),
            ));
        }
        Ok(())
    }

    fn after_process(
        &self,
        _context: &mut MiddlewareContext,
        _output: &Array2<Float>,
    ) -> SklResult<()> {
        Ok(())
    }

    fn on_error(
        &self,
        _context: &mut MiddlewareContext,
        _error: &SklearsError,
    ) -> SklResult<ErrorAction> {
        Ok(ErrorAction::Abort)
    }

    fn priority(&self) -> i32 {
        10 // High priority - authenticate early
    }
}

impl AuthorizationMiddleware {
    /// Create new authorization middleware
    #[must_use]
    pub fn new(config: AuthorizationConfig) -> Self {
        Self {
            policies: Vec::new(),
            rbac: RoleBasedAccessControl {
                roles: HashMap::new(),
                permissions: HashMap::new(),
                role_hierarchy: HashMap::new(),
            },
            config,
        }
    }

    /// Add access policy
    pub fn add_policy(&mut self, policy: AccessPolicy) {
        self.policies.push(policy);
    }

    /// Check authorization
    pub fn authorize(&self, user_info: &UserInfo, resource: &str, action: &str) -> SklResult<bool> {
        for policy in &self.policies {
            if self.policy_matches(policy, resource)
                && self.check_permissions(policy, user_info, action)
            {
                return Ok(policy.effect == PolicyEffect::Allow);
            }
        }

        // Default to configured default effect
        Ok(matches!(self.config.default_effect, PolicyEffect::Allow))
    }

    /// Check if policy matches resource
    fn policy_matches(&self, policy: &AccessPolicy, resource: &str) -> bool {
        // Simplified pattern matching
        policy.resource_pattern == "*" || policy.resource_pattern == resource
    }

    /// Check user permissions against policy
    fn check_permissions(&self, policy: &AccessPolicy, user_info: &UserInfo, action: &str) -> bool {
        // Check role-based access
        for role in &user_info.roles {
            if policy.allowed_roles.contains(role) {
                return true;
            }
        }

        // Check permission-based access
        for permission in &user_info.permissions {
            if policy.required_permissions.contains(permission) {
                return true;
            }
        }

        false
    }
}

impl PipelineMiddleware for AuthorizationMiddleware {
    fn name(&self) -> &'static str {
        "authorization"
    }

    fn before_process(
        &self,
        context: &mut MiddlewareContext,
        _input: &Array2<Float>,
    ) -> SklResult<()> {
        if let Some(user_info) = &context.user_info {
            if !self.authorize(user_info, "pipeline", "execute")? {
                return Err(SklearsError::InvalidInput("Access denied".to_string()));
            }
        }
        Ok(())
    }

    fn after_process(
        &self,
        _context: &mut MiddlewareContext,
        _output: &Array2<Float>,
    ) -> SklResult<()> {
        Ok(())
    }

    fn on_error(
        &self,
        _context: &mut MiddlewareContext,
        _error: &SklearsError,
    ) -> SklResult<ErrorAction> {
        Ok(ErrorAction::Abort)
    }

    fn priority(&self) -> i32 {
        20 // Execute after authentication
    }
}

impl CachingMiddleware {
    /// Create new caching middleware
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            config,
            stats: CacheStats {
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
                total_size: 0,
                entry_count: 0,
                evictions: 0,
            },
        }
    }

    /// Generate cache key
    fn generate_cache_key(&self, input: &Array2<Float>, context: &MiddlewareContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        match &self.config.key_strategy {
            CacheKeyStrategy::HashInput => {
                let mut hasher = DefaultHasher::new();
                if let Some(slice) = input.as_slice() {
                    for &x in slice {
                        (x.to_bits()).hash(&mut hasher);
                    }
                }
                format!("{:x}", hasher.finish())
            }
            CacheKeyStrategy::HashInputAndContext => {
                let mut hasher = DefaultHasher::new();
                if let Some(slice) = input.as_slice() {
                    for &x in slice {
                        (x.to_bits()).hash(&mut hasher);
                    }
                }
                context.request_id.hash(&mut hasher);
                format!("{:x}", hasher.finish())
            }
            CacheKeyStrategy::Custom { .. } => {
                let mut hasher = DefaultHasher::new();
                if let Some(slice) = input.as_slice() {
                    for &x in slice {
                        (x.to_bits()).hash(&mut hasher);
                    }
                }
                format!("{:x}", hasher.finish())
            }
        }
    }

    /// Get cached data
    pub fn get(&mut self, key: &str) -> Option<Array2<Float>> {
        let result = {
            let mut cache = self.cache.lock().unwrap();

            if let Some(entry) = cache.get_mut(key) {
                // Check if entry is still valid
                if entry.created_at.elapsed().unwrap_or(Duration::MAX) <= self.config.ttl {
                    entry.last_accessed = SystemTime::now();
                    entry.access_count += 1;
                    Some((entry.data.clone(), true)) // (data, is_hit)
                } else {
                    // Entry expired, remove it
                    cache.remove(key);
                    Some((Array2::zeros((0, 0)), false)) // Dummy data, will indicate eviction
                }
            } else {
                None
            }
        };

        match result {
            Some((data, true)) => {
                self.stats.hits += 1;
                self.update_hit_ratio();
                Some(data)
            }
            Some((_, false)) => {
                self.stats.evictions += 1;
                self.stats.misses += 1;
                self.update_hit_ratio();
                None
            }
            None => {
                self.stats.misses += 1;
                self.update_hit_ratio();
                None
            }
        }
    }

    /// Put data in cache
    pub fn put(&mut self, key: String, data: Array2<Float>) {
        let max_size = self.config.max_size;
        let eviction_policy = self.config.eviction_policy.clone();

        let (evicted, final_size) = {
            let mut cache = self.cache.lock().unwrap();

            // Check if cache is full
            let mut evicted = false;
            if cache.len() >= max_size {
                // Perform eviction within the lock
                match eviction_policy {
                    EvictionPolicy::LRU => {
                        if let Some(lru_key) = cache
                            .iter()
                            .min_by_key(|(_, entry)| entry.last_accessed)
                            .map(|(key, _)| key.clone())
                        {
                            cache.remove(&lru_key);
                            evicted = true;
                        }
                    }
                    EvictionPolicy::LFU => {
                        if let Some(lfu_key) = cache
                            .iter()
                            .min_by_key(|(_, entry)| entry.access_count)
                            .map(|(key, _)| key.clone())
                        {
                            cache.remove(&lfu_key);
                            evicted = true;
                        }
                    }
                    _ => {
                        // Simple FIFO for other policies
                        if let Some(first_key) = cache.keys().next().cloned() {
                            cache.remove(&first_key);
                            evicted = true;
                        }
                    }
                }
            }

            let entry = CacheEntry {
                data,
                created_at: SystemTime::now(),
                last_accessed: SystemTime::now(),
                access_count: 1,
                metadata: HashMap::new(),
            };

            cache.insert(key, entry);
            (evicted, cache.len())
        };

        if evicted {
            self.stats.evictions += 1;
        }
        self.stats.entry_count = final_size;
    }

    /// Evict entries based on policy (internal method that doesn't borrow self)
    fn evict_entries_internal(&mut self, cache: &mut HashMap<String, CacheEntry>) {
        let eviction_policy = self.config.eviction_policy.clone();
        match eviction_policy {
            EvictionPolicy::LRU => {
                if let Some(lru_key) = cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_accessed)
                    .map(|(key, _)| key.clone())
                {
                    cache.remove(&lru_key);
                    self.stats.evictions += 1;
                }
            }
            EvictionPolicy::LFU => {
                if let Some(lfu_key) = cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(key, _)| key.clone())
                {
                    cache.remove(&lfu_key);
                    self.stats.evictions += 1;
                }
            }
            _ => {
                // Simple FIFO for other policies
                if let Some(first_key) = cache.keys().next().cloned() {
                    cache.remove(&first_key);
                    self.stats.evictions += 1;
                }
            }
        }
    }

    /// Update hit ratio
    fn update_hit_ratio(&mut self) {
        let total = self.stats.hits + self.stats.misses;
        if total > 0 {
            self.stats.hit_ratio = self.stats.hits as f64 / total as f64;
        }
    }
}

impl PipelineMiddleware for CachingMiddleware {
    fn name(&self) -> &'static str {
        "caching"
    }

    fn before_process(
        &self,
        context: &mut MiddlewareContext,
        input: &Array2<Float>,
    ) -> SklResult<()> {
        let cache_key = self.generate_cache_key(input, context);
        context.metadata.insert("cache_key".to_string(), cache_key);
        Ok(())
    }

    fn after_process(
        &self,
        context: &mut MiddlewareContext,
        output: &Array2<Float>,
    ) -> SklResult<()> {
        if let Some(cache_key) = context.metadata.get("cache_key") {
            self.cache.lock().unwrap().insert(
                cache_key.clone(),
                /// CacheEntry
                CacheEntry {
                    data: output.clone(),
                    created_at: SystemTime::now(),
                    last_accessed: SystemTime::now(),
                    access_count: 1,
                    metadata: HashMap::new(),
                },
            );
        }
        Ok(())
    }

    fn on_error(
        &self,
        _context: &mut MiddlewareContext,
        _error: &SklearsError,
    ) -> SklResult<ErrorAction> {
        Ok(ErrorAction::Continue)
    }

    fn priority(&self) -> i32 {
        50
    }
}

impl Default for MiddlewareChainConfig {
    fn default() -> Self {
        Self {
            parallel_execution: false,
            timeout_per_middleware: Duration::from_secs(30),
            global_timeout: Duration::from_secs(300),
            continue_on_error: false,
            detailed_logging: false,
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            required_methods: Vec::new(),
            allow_anonymous: true,
            session_timeout: Duration::from_secs(3600),
            token_refresh_threshold: Duration::from_secs(300),
            max_failed_attempts: 3,
            lockout_duration: Duration::from_secs(300),
        }
    }
}

impl Default for AuthorizationConfig {
    fn default() -> Self {
        Self {
            default_effect: PolicyEffect::Deny,
            enable_role_inheritance: true,
            cache_decisions: true,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            ttl: Duration::from_secs(3600),
            eviction_policy: EvictionPolicy::LRU,
            enable_stats: true,
            key_strategy: CacheKeyStrategy::HashInput,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            fail_on_error: true,
            auto_correct: false,
            timeout: Duration::from_secs(30),
            max_corrections: 10,
        }
    }
}

impl Default for TransformationConfig {
    fn default() -> Self {
        Self {
            parallel_transformations: false,
            timeout: Duration::from_secs(60),
            cache_results: false,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            real_time: true,
            collection_interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(86400),
            enable_alerting: true,
            alert_evaluation_interval: Duration::from_secs(60),
        }
    }
}

impl PartialEq for PolicyEffect {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (PolicyEffect::Allow, PolicyEffect::Allow)
                | (PolicyEffect::Deny, PolicyEffect::Deny)
                | (PolicyEffect::Conditional, PolicyEffect::Conditional)
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_context_creation() {
        let context = MiddlewareContext {
            request_id: "test-123".to_string(),
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
            user_info: None,
            state: ContextState::Initializing,
            metrics: ExecutionMetrics {
                start_time: Instant::now(),
                end_time: None,
                duration: None,
                memory_usage: 0,
                cpu_usage: 0.0,
                throughput: 0.0,
                error_count: 0,
                custom_metrics: HashMap::new(),
            },
            custom_data: HashMap::new(),
        };

        assert_eq!(context.request_id, "test-123");
        assert!(matches!(context.state, ContextState::Initializing));
    }

    #[test]
    fn test_middleware_chain_creation() {
        let config = MiddlewareChainConfig::default();
        let chain = MiddlewareChain::new(config);

        assert_eq!(chain.middlewares.len(), 0);
        assert_eq!(chain.stats.total_requests, 0);
    }

    #[test]
    fn test_authentication_middleware() {
        let config = AuthenticationConfig::default();
        let auth_middleware = AuthenticationMiddleware::new(config);

        assert_eq!(auth_middleware.name(), "authentication");
        assert_eq!(auth_middleware.priority(), 10);
    }

    #[test]
    fn test_caching_middleware() {
        let config = CacheConfig::default();
        let cache_middleware = CachingMiddleware::new(config);

        assert_eq!(cache_middleware.name(), "caching");
        assert_eq!(cache_middleware.stats.hit_ratio, 0.0);
    }

    #[test]
    fn test_cache_key_generation() {
        let config = CacheConfig::default();
        let cache_middleware = CachingMiddleware::new(config);

        let input = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let context = MiddlewareContext {
            request_id: "test".to_string(),
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
            user_info: None,
            state: ContextState::Processing,
            metrics: ExecutionMetrics {
                start_time: Instant::now(),
                end_time: None,
                duration: None,
                memory_usage: 0,
                cpu_usage: 0.0,
                throughput: 0.0,
                error_count: 0,
                custom_metrics: HashMap::new(),
            },
            custom_data: HashMap::new(),
        };

        let key = cache_middleware.generate_cache_key(&input, &context);
        assert!(!key.is_empty());
    }

    #[test]
    fn test_access_policy() {
        let policy = AccessPolicy {
            name: "test_policy".to_string(),
            resource_pattern: "/api/*".to_string(),
            required_permissions: vec!["read".to_string()],
            allowed_roles: vec!["user".to_string()],
            conditions: Vec::new(),
            effect: PolicyEffect::Allow,
        };

        assert_eq!(policy.name, "test_policy");
        assert_eq!(policy.effect, PolicyEffect::Allow);
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult {
            valid: true,
            messages: Vec::new(),
            corrections: Vec::new(),
        };

        assert!(result.valid);
        assert_eq!(result.messages.len(), 0);
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats {
            hits: 10,
            misses: 5,
            hit_ratio: 0.0,
            total_size: 1024,
            entry_count: 15,
            evictions: 2,
        };

        // Calculate hit ratio
        let total = stats.hits + stats.misses;
        stats.hit_ratio = stats.hits as f64 / total as f64;

        assert_eq!(stats.hit_ratio, 10.0 / 15.0);
    }

    #[test]
    fn test_user_info() {
        let user_info = UserInfo {
            user_id: "user123".to_string(),
            roles: vec!["admin".to_string(), "user".to_string()],
            permissions: vec!["read".to_string(), "write".to_string()],
            session_token: Some("token123".to_string()),
            auth_method: AuthenticationMethod::ApiKey {
                key: "api_key_123".to_string(),
            },
        };

        assert_eq!(user_info.user_id, "user123");
        assert_eq!(user_info.roles.len(), 2);
        assert_eq!(user_info.permissions.len(), 2);
    }

    #[test]
    fn test_metric_creation() {
        let metric = Metric {
            name: "response_time".to_string(),
            value: 150.5,
            metric_type: MetricType::Timer,
            timestamp: SystemTime::now(),
            labels: HashMap::from([
                ("service".to_string(), "api".to_string()),
                ("version".to_string(), "1.0".to_string()),
            ]),
        };

        assert_eq!(metric.name, "response_time");
        assert_eq!(metric.value, 150.5);
        assert!(matches!(metric.metric_type, MetricType::Timer));
    }
}
