//! Notification Core - Central notification channel management
//!
//! This module provides the core notification channel management functionality including
//! the main manager, basic configuration, and fundamental channel operations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime};
use std::fmt::{self, Display, Formatter};

use crate::distributed_optimization::core_types::{NodeId, OptimizationError};

/// Central notification channel manager
#[derive(Debug, Clone)]
pub struct NotificationChannelManager {
    /// Available notification channels
    pub notification_channels: Arc<RwLock<Vec<NotificationChannel>>>,
    /// Channel health monitor
    pub health_monitor: Arc<RwLock<ChannelHealthMonitor>>,
    /// Channel router for load balancing
    pub channel_router: Arc<RwLock<ChannelRouter>>,
    /// Connection pool manager
    pub connection_pool: Arc<RwLock<ConnectionPoolManager>>,
    /// Manager configuration
    pub config: ChannelManagerConfig,
    /// Channel metrics
    pub metrics: Arc<RwLock<ChannelMetrics>>,
    /// Authentication manager
    pub auth_manager: Arc<RwLock<AuthenticationManager>>,
    /// Channel lifecycle manager
    pub lifecycle_manager: Arc<RwLock<ChannelLifecycleManager>>,
    /// Channel error handler
    pub error_handler: Arc<RwLock<ChannelErrorHandler>>,
    /// Channel event dispatcher
    pub event_dispatcher: Arc<RwLock<ChannelEventDispatcher>>,
    /// Channel validation system
    pub validation_system: Arc<RwLock<ChannelValidationSystem>>,
    /// Channel state manager
    pub state_manager: Arc<RwLock<ChannelStateManager>>,
}

/// Configuration for the channel manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelManagerConfig {
    /// Maximum number of channels
    pub max_channels: usize,
    /// Default channel timeout
    pub default_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable automatic failover
    pub enable_automatic_failover: bool,
    /// Connection pool configuration
    pub connection_pool_config: ConnectionPoolConfig,
    /// Security settings
    pub security_config: ChannelSecurityConfig,
    /// Performance optimization
    pub performance_config: ChannelPerformanceConfig,
    /// Manager operation mode
    pub operation_mode: ManagerOperationMode,
    /// Channel discovery configuration
    pub discovery_config: ChannelDiscoveryConfig,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Error recovery configuration
    pub error_recovery_config: ErrorRecoveryConfig,
    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
    /// Debug configuration
    pub debug_config: DebugConfig,
}

/// Manager operation mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManagerOperationMode {
    /// Single-node operation
    Standalone,
    /// Clustered operation
    Clustered(ClusterConfig),
    /// Distributed operation
    Distributed(DistributedConfig),
    /// Hybrid operation
    Hybrid(HybridConfig),
    /// Custom operation mode
    Custom(String),
}

/// Cluster configuration for clustered operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster identifier
    pub cluster_id: String,
    /// Node identifier
    pub node_id: String,
    /// Cluster members
    pub cluster_members: Vec<ClusterMember>,
    /// Leader election configuration
    pub leader_election: LeaderElectionConfig,
    /// Consensus algorithm
    pub consensus_algorithm: ConsensusAlgorithm,
    /// Partition tolerance
    pub partition_tolerance: PartitionToleranceConfig,
}

/// Cluster member definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMember {
    /// Member node ID
    pub node_id: String,
    /// Member endpoint
    pub endpoint: String,
    /// Member role
    pub role: ClusterMemberRole,
    /// Member state
    pub state: ClusterMemberState,
    /// Member metadata
    pub metadata: HashMap<String, String>,
}

/// Cluster member role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMemberRole {
    /// Leader node
    Leader,
    /// Follower node
    Follower,
    /// Observer node
    Observer,
    /// Candidate node
    Candidate,
    /// Inactive node
    Inactive,
}

/// Cluster member state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMemberState {
    /// Active member
    Active,
    /// Joining member
    Joining,
    /// Leaving member
    Leaving,
    /// Failed member
    Failed,
    /// Suspended member
    Suspended,
}

/// Notification channel definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel identifier
    pub channel_id: String,
    /// Channel name
    pub name: String,
    /// Channel type
    pub channel_type: ChannelType,
    /// Channel configuration
    pub config: ChannelConfig,
    /// Whether channel is enabled
    pub enabled: bool,
    /// Channel priority
    pub priority: ChannelPriority,
    /// Channel metadata
    pub metadata: ChannelMetadata,
    /// Failover configuration
    pub failover_config: ChannelFailoverConfig,
    /// Health check configuration
    pub health_check: ChannelHealthCheck,
    /// Rate limiting configuration
    pub rate_limit: ChannelRateLimit,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
    /// Channel state
    pub state: ChannelState,
    /// Channel statistics
    pub statistics: ChannelStatistics,
    /// Channel dependencies
    pub dependencies: Vec<String>,
    /// Channel tags
    pub tags: Vec<String>,
}

/// Types of notification channels with extended support
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChannelType {
    /// Email channel
    Email,
    /// Slack channel
    Slack,
    /// Webhook channel
    Webhook,
    /// SMS channel
    SMS,
    /// PagerDuty channel
    PagerDuty,
    /// Microsoft Teams channel
    MicrosoftTeams,
    /// Discord channel
    Discord,
    /// Telegram channel
    Telegram,
    /// WhatsApp channel
    WhatsApp,
    /// Matrix channel
    Matrix,
    /// Mattermost channel
    Mattermost,
    /// RocketChat channel
    RocketChat,
    /// Kafka channel
    Kafka,
    /// RabbitMQ channel
    RabbitMQ,
    /// MQTT channel
    MQTT,
    /// WebSocket channel
    WebSocket,
    /// GraphQL channel
    GraphQL,
    /// gRPC channel
    GRPC,
    /// IRC Bot channel
    IrcBot,
    /// Jira Ticket channel
    JiraTicket,
    /// ServiceNow channel
    ServiceNow,
    /// Zendesk channel
    Zendesk,
    /// Custom channel
    Custom(String),
}

/// Channel priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
    /// Emergency priority
    Emergency,
    /// Custom priority
    Custom(u32),
}

/// Channel configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Endpoint URL or address
    pub endpoint: String,
    /// Authentication configuration
    pub auth_config: AuthConfig,
    /// Request timeout
    pub timeout: Duration,
    /// Connection configuration
    pub connection_config: ConnectionConfig,
    /// Protocol configuration
    pub protocol_config: ProtocolConfig,
    /// Custom headers
    pub custom_headers: HashMap<String, String>,
    /// Custom parameters
    pub custom_parameters: HashMap<String, String>,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
    /// Message formatting configuration
    pub message_format: MessageFormatConfig,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
}

/// Channel metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMetadata {
    /// Channel description
    pub description: String,
    /// Channel owner
    pub owner: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
    /// Channel version
    pub version: String,
    /// Channel tags
    pub tags: HashMap<String, String>,
    /// Custom metadata
    pub custom_metadata: HashMap<String, String>,
    /// Documentation links
    pub documentation: Vec<String>,
    /// Support contacts
    pub support_contacts: Vec<String>,
}

/// Channel state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelState {
    /// Current state
    pub current_state: ChannelStateType,
    /// State history
    pub state_history: VecDeque<ChannelStateEntry>,
    /// State transition rules
    pub transition_rules: HashMap<String, StateTransitionRule>,
    /// State machine configuration
    pub state_machine_config: StateMachineConfig,
}

/// Channel state types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelStateType {
    /// Channel is initializing
    Initializing,
    /// Channel is active and ready
    Active,
    /// Channel is temporarily inactive
    Inactive,
    /// Channel is in error state
    Error(ChannelError),
    /// Channel is in maintenance mode
    Maintenance,
    /// Channel is being shut down
    Shutdown,
    /// Channel is suspended
    Suspended,
    /// Channel is degraded but operational
    Degraded,
}

/// Channel state entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStateEntry {
    /// Previous state
    pub previous_state: ChannelStateType,
    /// Current state
    pub current_state: ChannelStateType,
    /// Transition timestamp
    pub timestamp: SystemTime,
    /// Transition reason
    pub reason: String,
    /// Transition metadata
    pub metadata: HashMap<String, String>,
}

/// Channel statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStatistics {
    /// Total messages sent
    pub total_messages: u64,
    /// Successful messages
    pub successful_messages: u64,
    /// Failed messages
    pub failed_messages: u64,
    /// Average response time
    pub average_response_time: Duration,
    /// Last activity timestamp
    pub last_activity: SystemTime,
    /// Error count by type
    pub error_counts: HashMap<String, u32>,
    /// Performance metrics
    pub performance_metrics: ChannelPerformanceMetrics,
    /// Usage statistics
    pub usage_statistics: ChannelUsageStatistics,
}

/// Channel performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelPerformanceMetrics {
    /// Throughput (messages per second)
    pub throughput: f64,
    /// Latency percentiles
    pub latency_percentiles: LatencyPercentiles,
    /// Success rate
    pub success_rate: f64,
    /// Availability percentage
    pub availability: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// 50th percentile (median)
    pub p50: Duration,
    /// 90th percentile
    pub p90: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile
    pub p999: Duration,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_usage: f64,
    /// Memory utilization in bytes
    pub memory_usage: usize,
    /// Network bandwidth utilization
    pub network_usage: NetworkUsage,
    /// Connection pool utilization
    pub connection_pool_usage: f64,
}

/// Network usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkUsage {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Network errors
    pub network_errors: u32,
}

/// Channel usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelUsageStatistics {
    /// Peak usage time
    pub peak_usage_time: SystemTime,
    /// Peak message rate
    pub peak_message_rate: f64,
    /// Usage patterns
    pub usage_patterns: Vec<UsagePattern>,
    /// User activity
    pub user_activity: HashMap<String, UserActivityStats>,
}

/// Usage pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// Pattern name
    pub pattern_name: String,
    /// Pattern frequency
    pub frequency: f64,
    /// Pattern duration
    pub duration: Duration,
    /// Pattern characteristics
    pub characteristics: HashMap<String, String>,
}

/// User activity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserActivityStats {
    /// Total messages from user
    pub message_count: u64,
    /// Last activity timestamp
    pub last_activity: SystemTime,
    /// Activity frequency
    pub activity_frequency: f64,
    /// Preferred channels
    pub preferred_channels: Vec<String>,
}

/// Channel error definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelError {
    /// Error code
    pub error_code: String,
    /// Error message
    pub error_message: String,
    /// Error type
    pub error_type: ChannelErrorType,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error timestamp
    pub timestamp: SystemTime,
    /// Error context
    pub context: HashMap<String, String>,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
}

/// Channel error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelErrorType {
    /// Connection error
    Connection,
    /// Authentication error
    Authentication,
    /// Authorization error
    Authorization,
    /// Configuration error
    Configuration,
    /// Timeout error
    Timeout,
    /// Rate limit error
    RateLimit,
    /// Protocol error
    Protocol,
    /// Serialization error
    Serialization,
    /// Network error
    Network,
    /// Service unavailable error
    ServiceUnavailable,
    /// Custom error
    Custom(String),
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
    /// Fatal severity
    Fatal,
}

/// Channel health monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelHealthMonitor {
    /// Health check configuration
    pub health_check_config: HealthCheckConfig,
    /// Health status by channel
    pub channel_health: HashMap<String, ChannelHealthStatus>,
    /// Health check history
    pub health_history: VecDeque<HealthCheckResult>,
    /// Alert configuration
    pub alert_config: HealthAlertConfig,
    /// Recovery actions
    pub recovery_actions: Vec<RecoveryAction>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub check_interval: Duration,
    /// Health check timeout
    pub check_timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    /// Health check endpoints
    pub health_endpoints: HashMap<String, String>,
}

/// Channel health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelHealthStatus {
    /// Overall health status
    pub status: HealthStatus,
    /// Last check timestamp
    pub last_check: SystemTime,
    /// Consecutive failures
    pub consecutive_failures: u32,
    /// Health score (0-100)
    pub health_score: f64,
    /// Health details
    pub health_details: HealthDetails,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy status
    Healthy,
    /// Degraded status
    Degraded,
    /// Unhealthy status
    Unhealthy,
    /// Unknown status
    Unknown,
    /// Maintenance status
    Maintenance,
}

/// Health details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDetails {
    /// Component health status
    pub component_health: HashMap<String, ComponentHealth>,
    /// Performance indicators
    pub performance_indicators: HashMap<String, f64>,
    /// Resource availability
    pub resource_availability: HashMap<String, bool>,
    /// Diagnostic information
    pub diagnostic_info: HashMap<String, String>,
}

/// Component health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub component_name: String,
    /// Component status
    pub status: HealthStatus,
    /// Status message
    pub status_message: String,
    /// Last check timestamp
    pub last_check: SystemTime,
}

/// Channel router for load balancing and routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRouter {
    /// Routing configuration
    pub routing_config: RoutingConfig,
    /// Route table
    pub route_table: HashMap<String, Route>,
    /// Load balancing strategy
    pub load_balancer: LoadBalancer,
    /// Circuit breaker
    pub circuit_breaker: CircuitBreaker,
    /// Request tracking
    pub request_tracking: RequestTracker,
}

/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Default route
    pub default_route: String,
    /// Routing rules
    pub routing_rules: Vec<RoutingRule>,
    /// Fallback strategy
    pub fallback_strategy: FallbackStrategy,
    /// Route priorities
    pub route_priorities: HashMap<String, u32>,
}

/// Routing rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule name
    pub rule_name: String,
    /// Rule condition
    pub condition: RoutingCondition,
    /// Target channel
    pub target_channel: String,
    /// Rule priority
    pub priority: u32,
    /// Rule enabled
    pub enabled: bool,
}

/// Routing condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Route by message type
    MessageType(String),
    /// Route by priority
    Priority(ChannelPriority),
    /// Route by tag
    Tag(String),
    /// Route by user
    User(String),
    /// Route by time
    Time(TimeCondition),
    /// Custom condition
    Custom(String),
}

/// Time-based routing condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeCondition {
    /// Time range
    pub time_range: TimeRange,
    /// Days of week
    pub days_of_week: Vec<DayOfWeek>,
    /// Timezone
    pub timezone: String,
}

/// Time range definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start_time: String,
    /// End time
    pub end_time: String,
}

/// Day of week enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DayOfWeek {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

// Placeholder structures for comprehensive type safety (simplified for brevity)

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectionPoolManager { pub manager: String }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChannelMetrics { pub metrics: String }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthenticationManager { pub manager: String }

// Additional placeholder structures continue in the same pattern...

impl Default for NotificationChannelManager {
    fn default() -> Self {
        Self {
            notification_channels: Arc::new(RwLock::new(Vec::new())),
            health_monitor: Arc::new(RwLock::new(ChannelHealthMonitor::default())),
            channel_router: Arc::new(RwLock::new(ChannelRouter::default())),
            connection_pool: Arc::new(RwLock::new(ConnectionPoolManager::default())),
            config: ChannelManagerConfig::default(),
            metrics: Arc::new(RwLock::new(ChannelMetrics::default())),
            auth_manager: Arc::new(RwLock::new(AuthenticationManager::default())),
            lifecycle_manager: Arc::new(RwLock::new(ChannelLifecycleManager::default())),
            error_handler: Arc::new(RwLock::new(ChannelErrorHandler::default())),
            event_dispatcher: Arc::new(RwLock::new(ChannelEventDispatcher::default())),
            validation_system: Arc::new(RwLock::new(ChannelValidationSystem::default())),
            state_manager: Arc::new(RwLock::new(ChannelStateManager::default())),
        }
    }
}

impl Default for ChannelManagerConfig {
    fn default() -> Self {
        Self {
            max_channels: 100,
            default_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(60),
            enable_automatic_failover: true,
            connection_pool_config: ConnectionPoolConfig::default(),
            security_config: ChannelSecurityConfig::default(),
            performance_config: ChannelPerformanceConfig::default(),
            operation_mode: ManagerOperationMode::Standalone,
            discovery_config: ChannelDiscoveryConfig::default(),
            load_balancing_strategy: LoadBalancingStrategy::default(),
            error_recovery_config: ErrorRecoveryConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            debug_config: DebugConfig::default(),
        }
    }
}

impl Display for NotificationChannel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Channel: {} ({:?}) - {} - Priority: {:?}",
            self.name, self.channel_type, self.channel_id, self.priority
        )
    }
}

impl Display for ChannelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChannelError: {} - {} ({:?})",
            self.error_code, self.error_message, self.error_type
        )
    }
}

// Implement Default for placeholder structs using macro
macro_rules! impl_default_for_notification_placeholders {
    ($($struct_name:ident),*) => {
        $(
            #[derive(Debug, Clone, Serialize, Deserialize, Default)]
            pub struct $struct_name { pub data: String }
        )*
    };
}

// Apply Default implementation to remaining placeholder structures
impl_default_for_notification_placeholders!(
    ChannelLifecycleManager, ChannelErrorHandler, ChannelEventDispatcher,
    ChannelValidationSystem, ChannelStateManager, DistributedConfig, HybridConfig,
    LeaderElectionConfig, ConsensusAlgorithm, PartitionToleranceConfig,
    ChannelDiscoveryConfig, LoadBalancingStrategy, ErrorRecoveryConfig,
    MonitoringConfig, DebugConfig, ConnectionPoolConfig, ChannelSecurityConfig,
    ChannelPerformanceConfig, ChannelFailoverConfig, ChannelHealthCheck,
    ChannelRateLimit, LoadBalancingConfig, StateTransitionRule, StateMachineConfig,
    AuthConfig, ConnectionConfig, ProtocolConfig, MessageFormatConfig,
    RetryConfig, CircuitBreakerConfig, HealthCheckResult, HealthAlertConfig,
    RecoveryAction, Route, LoadBalancer, CircuitBreaker, RequestTracker,
    FallbackStrategy
);

impl Default for ChannelHealthMonitor {
    fn default() -> Self {
        Self {
            health_check_config: HealthCheckConfig::default(),
            channel_health: HashMap::new(),
            health_history: VecDeque::new(),
            alert_config: HealthAlertConfig::default(),
            recovery_actions: Vec::new(),
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(60),
            check_timeout: Duration::from_secs(10),
            failure_threshold: 3,
            success_threshold: 2,
            health_endpoints: HashMap::new(),
        }
    }
}

impl Default for ChannelRouter {
    fn default() -> Self {
        Self {
            routing_config: RoutingConfig::default(),
            route_table: HashMap::new(),
            load_balancer: LoadBalancer::default(),
            circuit_breaker: CircuitBreaker::default(),
            request_tracking: RequestTracker::default(),
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_route: "default".to_string(),
            routing_rules: Vec::new(),
            fallback_strategy: FallbackStrategy::default(),
            route_priorities: HashMap::new(),
        }
    }
}