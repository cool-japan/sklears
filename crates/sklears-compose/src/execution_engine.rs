//! Core Execution Engine Configuration and Management
//!
//! This module provides the fundamental execution engine configuration types,
//! engine management functionality, and core engine behavior definitions.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Main configuration for the execution engine
///
/// This configuration defines the overall behavior and constraints of the
/// composable execution engine, including resource limits, performance goals,
/// fault tolerance settings, and monitoring configuration.
#[derive(Debug, Clone)]
pub struct ExecutionEngineConfig {
    /// Engine name for identification and logging
    pub name: String,

    /// Default execution strategy to use when none is specified
    pub default_strategy: String,

    /// Resource constraints and limits
    pub resource_constraints: ResourceConstraints,

    /// Performance goals and targets
    pub performance_goals: PerformanceGoals,

    /// Fault tolerance and resilience settings
    pub fault_tolerance: FaultToleranceConfig,

    /// Monitoring and observability configuration
    pub monitoring: MonitoringConfig,
}

impl Default for ExecutionEngineConfig {
    fn default() -> Self {
        Self {
            name: "default_execution_engine".to_string(),
            default_strategy: "sequential".to_string(),
            resource_constraints: ResourceConstraints::default(),
            performance_goals: PerformanceGoals::default(),
            fault_tolerance: FaultToleranceConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

/// Resource constraints and limitations for execution
///
/// Defines the maximum resources that the execution engine can utilize,
/// including CPU, memory, I/O, and concurrency limits.
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum CPU cores the engine can use
    pub max_cpu_cores: Option<usize>,

    /// Maximum memory usage in bytes
    pub max_memory: Option<u64>,

    /// Maximum execution time per individual task
    pub max_task_time: Option<Duration>,

    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: Option<usize>,

    /// I/O bandwidth limitations
    pub io_bandwidth: Option<IoLimits>,

    /// GPU resource constraints
    pub gpu_constraints: Option<GpuConstraints>,

    /// Network bandwidth constraints
    pub network_constraints: Option<NetworkConstraints>,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_cores: None, // Use all available cores
            max_memory: None,    // No memory limit
            max_task_time: Some(Duration::from_secs(3600)), // 1 hour default timeout
            max_concurrent_tasks: None, // No concurrency limit
            io_bandwidth: None,
            gpu_constraints: None,
            network_constraints: None,
        }
    }
}

/// I/O bandwidth limitations
#[derive(Debug, Clone)]
pub struct IoLimits {
    /// Maximum read bandwidth in bytes per second
    pub max_read_bps: u64,

    /// Maximum write bandwidth in bytes per second
    pub max_write_bps: u64,

    /// Maximum concurrent I/O operations
    pub max_concurrent_io: usize,

    /// Maximum open file handles
    pub max_open_files: Option<usize>,
}

/// GPU resource constraints
#[derive(Debug, Clone)]
pub struct GpuConstraints {
    /// Maximum GPU memory usage in bytes
    pub max_gpu_memory: Option<u64>,

    /// Maximum number of GPU devices to use
    pub max_gpu_devices: Option<usize>,

    /// Preferred GPU device IDs
    pub preferred_devices: Vec<usize>,

    /// GPU compute capability requirements
    pub min_compute_capability: Option<(u32, u32)>,
}

/// Network bandwidth constraints
#[derive(Debug, Clone)]
pub struct NetworkConstraints {
    pub max_bandwidth_bps: u64,

    pub max_connections: usize,

    pub timeout: Duration,
}

/// Performance goals and optimization targets
///
/// Defines the performance characteristics the engine should strive to achieve,
/// including throughput, latency, resource utilization, and energy efficiency goals.
#[derive(Debug, Clone)]
pub struct PerformanceGoals {
    /// Target task throughput in tasks per second
    pub target_throughput: Option<f64>,

    /// Target task latency in milliseconds
    pub target_latency: Option<f64>,

    /// Target resource utilization as a percentage (0.0 to 1.0)
    pub target_utilization: Option<f64>,

    /// Energy efficiency goals
    pub energy_efficiency: Option<EnergyEfficiencyGoal>,

    /// Quality of service requirements
    pub qos_requirements: Option<QoSRequirements>,

    /// Performance optimization preferences
    pub optimization_preferences: OptimizationPreferences,
}

impl Default for PerformanceGoals {
    fn default() -> Self {
        Self {
            target_throughput: None,
            target_latency: None,
            target_utilization: Some(0.8), // 80% target utilization
            energy_efficiency: None,
            qos_requirements: None,
            optimization_preferences: OptimizationPreferences::default(),
        }
    }
}

/// Energy efficiency optimization goals
#[derive(Debug, Clone)]
pub struct EnergyEfficiencyGoal {
    /// Target power consumption in watts
    pub target_power: f64,

    /// Performance per watt optimization goal
    pub performance_per_watt: f64,

    /// Thermal management preferences
    pub thermal_management: ThermalManagement,
}

/// Quality of Service requirements
#[derive(Debug, Clone)]
pub struct QoSRequirements {
    /// Minimum acceptable performance level
    pub min_performance: f64,

    /// Maximum acceptable latency
    pub max_latency: Duration,

    /// Service level agreement parameters
    pub sla_parameters: SlaParameters,
}

/// Service Level Agreement parameters
#[derive(Debug, Clone)]
pub struct SlaParameters {
    /// Uptime requirement as percentage
    pub uptime_requirement: f64,

    /// Error rate threshold
    pub error_rate_threshold: f64,

    /// Response time percentile requirements
    pub response_time_p99: Duration,
    pub response_time_p95: Duration,
    pub response_time_p50: Duration,
}

/// Thermal management preferences
#[derive(Debug, Clone)]
pub enum ThermalManagement {
    /// Passive cooling preference
    Passive,
    /// Active cooling with fan control
    Active { max_fan_speed: f64 },
    /// Thermal throttling when temperature exceeds threshold
    Throttling { threshold_celsius: f64 },
    /// Custom thermal management strategy
    Custom { strategy_name: String },
}

/// Performance optimization preferences
#[derive(Debug, Clone)]
pub struct OptimizationPreferences {
    /// Prefer throughput over latency
    pub favor_throughput: bool,

    /// Prefer latency over throughput
    pub favor_latency: bool,

    /// Prefer energy efficiency
    pub favor_energy_efficiency: bool,

    /// Prefer resource utilization
    pub favor_utilization: bool,
}

impl Default for OptimizationPreferences {
    fn default() -> Self {
        Self {
            favor_throughput: false,
            favor_latency: false,
            favor_energy_efficiency: false,
            favor_utilization: true, // Default to balanced utilization
        }
    }
}

/// Fault tolerance and resilience configuration
///
/// Defines how the engine should handle failures, including retry policies,
/// failover strategies, and health checking mechanisms.
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Enable automatic task retry on failure
    pub enable_retry: bool,

    /// Maximum number of retry attempts per task
    pub max_retries: usize,

    /// Retry backoff strategy
    pub backoff_strategy: BackoffStrategy,

    /// Enable failover to alternative execution strategies
    pub enable_failover: bool,

    /// List of failover target strategies
    pub failover_targets: Vec<String>,

    /// Health check configuration
    pub health_check: HealthCheckConfig,

    /// Circuit breaker configuration
    pub circuit_breaker: Option<CircuitBreakerConfig>,

    /// Graceful degradation settings
    pub graceful_degradation: GracefulDegradationConfig,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_retry: true,
            max_retries: 3,
            backoff_strategy: BackoffStrategy::Exponential {
                base: Duration::from_millis(100),
                multiplier: 2.0,
            },
            enable_failover: false,
            failover_targets: Vec::new(),
            health_check: HealthCheckConfig::default(),
            circuit_breaker: None,
            graceful_degradation: GracefulDegradationConfig::default(),
        }
    }
}

/// Retry backoff strategies
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retry attempts
    Fixed(Duration),

    /// Exponential backoff with configurable base and multiplier
    Exponential { base: Duration, multiplier: f64 },

    /// Linear backoff with configurable base and increment
    Linear { base: Duration, increment: Duration },

    /// Jittered exponential backoff to avoid thundering herd
    Jittered { base: Duration, multiplier: f64, max_jitter: Duration },

    /// Custom backoff strategy by name
    Custom { name: String },
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Interval between health checks
    pub interval: Duration,

    /// Timeout for individual health checks
    pub timeout: Duration,

    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: usize,

    /// Number of consecutive successes before marking healthy
    pub recovery_threshold: usize,

    /// Health check endpoints or methods
    pub endpoints: Vec<HealthCheckEndpoint>,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
            endpoints: Vec::new(),
        }
    }
}

/// Health check endpoint configuration
#[derive(Debug, Clone)]
pub struct HealthCheckEndpoint {
    /// Endpoint name or identifier
    pub name: String,

    /// Health check method
    pub method: HealthCheckMethod,

    /// Expected response criteria
    pub expected_response: HealthCheckExpectation,
}

/// Health check methods
#[derive(Debug, Clone)]
pub enum HealthCheckMethod {
    /// HTTP endpoint check
    Http { url: String, method: String },

    /// TCP connection check
    Tcp { host: String, port: u16 },

    /// Custom function check
    Custom { function_name: String },

    /// Memory usage check
    MemoryUsage { threshold: f64 },

    /// CPU usage check
    CpuUsage { threshold: f64 },
}

/// Health check expectation criteria
#[derive(Debug, Clone)]
pub enum HealthCheckExpectation {
    /// HTTP status code expectation
    HttpStatus(u16),

    /// Response contains specific text
    Contains(String),

    /// Numeric value within range
    NumericRange { min: f64, max: f64 },

    /// Boolean result
    Boolean(bool),

    /// Custom validation function
    Custom { validator_name: String },
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: usize,

    /// Timeout before attempting to close circuit
    pub timeout: Duration,

    /// Success threshold to close circuit
    pub success_threshold: usize,

    /// Window size for failure counting
    pub window_size: Duration,
}

/// Graceful degradation configuration
#[derive(Debug, Clone)]
pub struct GracefulDegradationConfig {
    /// Enable graceful degradation
    pub enabled: bool,

    /// Degradation levels and thresholds
    pub degradation_levels: Vec<DegradationLevel>,

    /// Recovery thresholds
    pub recovery_thresholds: HashMap<String, f64>,
}

impl Default for GracefulDegradationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            degradation_levels: Vec::new(),
            recovery_thresholds: HashMap::new(),
        }
    }
}

/// Degradation level configuration
#[derive(Debug, Clone)]
pub struct DegradationLevel {
    /// Level name
    pub name: String,

    /// Trigger conditions
    pub trigger_conditions: Vec<DegradationTrigger>,

    /// Actions to take at this level
    pub actions: Vec<DegradationAction>,
}

/// Degradation trigger conditions
#[derive(Debug, Clone)]
pub enum DegradationTrigger {
    /// CPU usage exceeds threshold
    CpuUsage(f64),

    /// Memory usage exceeds threshold
    MemoryUsage(f64),

    /// Error rate exceeds threshold
    ErrorRate(f64),

    /// Latency exceeds threshold
    Latency(Duration),

    /// Custom trigger condition
    Custom { condition_name: String },
}

/// Degradation actions
#[derive(Debug, Clone)]
pub enum DegradationAction {
    /// Reduce task concurrency
    ReduceConcurrency { factor: f64 },

    /// Switch to simpler execution strategy
    SwitchStrategy { strategy_name: String },

    /// Disable non-essential features
    DisableFeatures { features: Vec<String> },

    /// Custom degradation action
    Custom { action_name: String },
}

/// Monitoring and observability configuration
///
/// Controls what metrics are collected, how frequently, and where they are stored.
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable performance metrics collection
    pub enable_performance: bool,

    /// Enable resource utilization monitoring
    pub enable_resources: bool,

    /// Enable error and exception monitoring
    pub enable_errors: bool,

    /// Enable distributed tracing
    pub enable_tracing: bool,

    /// Metrics collection interval
    pub collection_interval: Duration,

    /// Metrics retention period
    pub retention_period: Duration,

    /// Export configuration for metrics
    pub export_config: Option<MetricsExportConfig>,

    /// Alerting configuration
    pub alerting: Option<AlertingConfig>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_performance: true,
            enable_resources: true,
            enable_errors: true,
            enable_tracing: false,
            collection_interval: Duration::from_secs(10),
            retention_period: Duration::from_secs(3600 * 24), // 24 hours
            export_config: None,
            alerting: None,
        }
    }
}

/// Metrics export configuration
#[derive(Debug, Clone)]
pub struct MetricsExportConfig {
    /// Export format (prometheus, influxdb, etc.)
    pub format: String,

    /// Export endpoint
    pub endpoint: String,

    /// Export interval
    pub interval: Duration,

    /// Authentication configuration
    pub auth: Option<AuthConfig>,
}

/// Authentication configuration for external services
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// Authentication method
    pub method: AuthMethod,

    /// Credentials or tokens
    pub credentials: HashMap<String, String>,
}

/// Authentication methods
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// API key authentication
    ApiKey,

    /// Bearer token authentication
    BearerToken,

    /// Basic authentication
    Basic,

    /// OAuth 2.0
    OAuth2,

    /// Custom authentication method
    Custom { method_name: String },
}

/// Alerting configuration
#[derive(Debug, Clone)]
pub struct AlertingConfig {
    /// Alert thresholds
    pub thresholds: HashMap<String, AlertThreshold>,

    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,

    /// Alert aggregation rules
    pub aggregation_rules: Vec<AlertAggregationRule>,
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThreshold {
    /// Metric name
    pub metric: String,

    /// Threshold value
    pub threshold: f64,

    /// Comparison operator
    pub operator: ComparisonOperator,

    /// Duration the condition must persist
    pub duration: Duration,
}

/// Comparison operators for alerting
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Notification channels for alerts
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    /// Email notification
    Email { addresses: Vec<String> },

    /// Slack notification
    Slack { webhook_url: String, channel: String },

    /// Custom webhook
    Webhook { url: String, method: String },

    /// Log file notification
    LogFile { path: String },
}

/// Alert aggregation rules
#[derive(Debug, Clone)]
pub struct AlertAggregationRule {
    /// Rule name
    pub name: String,

    /// Grouping criteria
    pub group_by: Vec<String>,

    /// Aggregation window
    pub window: Duration,

    /// Aggregation function
    pub function: AggregationFunction,
}

/// Aggregation functions for alerts
#[derive(Debug, Clone)]
pub enum AggregationFunction {
    Sum,
    Average,
    Maximum,
    Minimum,
    Count,
    Percentile(f64),
    Custom { function_name: String },
}

/// Execution context containing shared runtime state
///
/// Provides runtime environment and shared state that can be accessed
/// by tasks and strategies during execution.
#[derive(Debug)]
pub struct ExecutionContext {
    /// Unique context identifier
    pub id: String,

    /// Execution start timestamp
    pub start_time: SystemTime,

    /// Environment variables accessible to tasks
    pub environment: HashMap<String, String>,

    /// Shared state storage for inter-task communication
    pub shared_state: std::sync::Arc<std::sync::RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>>,

    /// Execution statistics and counters
    pub statistics: std::sync::Arc<std::sync::Mutex<ExecutionStatistics>>,
}

/// Execution statistics tracking
#[derive(Debug, Clone)]
pub struct ExecutionStatistics {
    /// Total number of tasks executed
    pub total_tasks: u64,

    /// Number of successfully completed tasks
    pub successful_tasks: u64,

    /// Number of failed tasks
    pub failed_tasks: u64,

    /// Cumulative execution time across all tasks
    pub total_execution_time: Duration,

    /// Average task execution duration
    pub average_task_duration: Duration,

    /// Current resource utilization snapshot
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization percentage (0.0 to 1.0)
    pub cpu_percent: f64,

    /// Memory utilization percentage (0.0 to 1.0)
    pub memory_percent: f64,

    /// I/O utilization percentage (0.0 to 1.0)
    pub io_percent: f64,

    /// Network utilization percentage (0.0 to 1.0)
    pub network_percent: f64,
}

impl Default for ExecutionStatistics {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            total_execution_time: Duration::new(0, 0),
            average_task_duration: Duration::new(0, 0),
            resource_utilization: ResourceUtilization {
                cpu_percent: 0.0,
                memory_percent: 0.0,
                io_percent: 0.0,
                network_percent: 0.0,
            },
        }
    }
}

impl ExecutionContext {
    /// Create a new execution context with unique identifier
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            start_time: SystemTime::now(),
            environment: std::env::vars().collect(),
            shared_state: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            statistics: std::sync::Arc::new(std::sync::Mutex::new(ExecutionStatistics::default())),
        }
    }

    /// Get a value from shared state
    pub fn get_shared<T: std::any::Any + Send + Sync + Clone>(&self, key: &str) -> Option<T> {
        let state = self.shared_state.read().ok()?;
        state.get(key)?.downcast_ref::<T>().cloned()
    }

    /// Set a value in shared state
    pub fn set_shared<T: std::any::Any + Send + Sync>(&self, key: String, value: T) -> SklResult<()> {
        let mut state = self.shared_state.write().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire shared state lock".to_string())
        })?;
        state.insert(key, Box::new(value));
        Ok(())
    }

    /// Update execution statistics
    pub fn update_statistics<F>(&self, updater: F) -> SklResult<()>
    where
        F: FnOnce(&mut ExecutionStatistics),
    {
        let mut stats = self.statistics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
        })?;
        updater(&mut *stats);
        Ok(())
    }
}

/// Main composable execution engine
///
/// The central orchestrator that coordinates execution strategies, resource management,
/// task scheduling, and performance monitoring for flexible pipeline runtime execution.
pub struct ComposableExecutionEngine {
    /// Engine configuration
    config: ExecutionEngineConfig,

    /// Registry of execution strategies
    strategies: HashMap<String, Box<dyn crate::execution_strategies::ExecutionStrategy>>,

    /// Resource manager for allocation and monitoring
    resource_manager: std::sync::Arc<crate::resource_management::ResourceManager>,

    /// Task scheduler for execution coordination
    scheduler: Box<dyn crate::task_scheduling::TaskScheduler>,

    /// Execution context with shared state
    context: ExecutionContext,

    /// Metrics collector for performance monitoring
    metrics: std::sync::Arc<std::sync::Mutex<crate::performance_monitoring::ExecutionMetrics>>,
}

impl ComposableExecutionEngine {
    /// Create a new execution engine with the specified configuration
    pub fn new(config: ExecutionEngineConfig) -> SklResult<Self> {
        let resource_manager = std::sync::Arc::new(
            crate::resource_management::ResourceManager::new(config.resource_constraints.clone())?
        );

        let scheduler = Box::new(
            crate::task_scheduling::DefaultTaskScheduler::new(
                crate::task_scheduling::SchedulerConfig::default()
            )
        );

        let context = ExecutionContext::new();

        let metrics = std::sync::Arc::new(std::sync::Mutex::new(
            crate::performance_monitoring::ExecutionMetrics::new()
        ));

        Ok(Self {
            config,
            strategies: HashMap::new(),
            resource_manager,
            scheduler,
            context,
            metrics,
        })
    }

    /// Register an execution strategy
    pub fn register_strategy(
        &mut self,
        strategy: Box<dyn crate::execution_strategies::ExecutionStrategy>
    ) -> SklResult<()> {
        let name = strategy.name().to_string();
        self.strategies.insert(name, strategy);
        Ok(())
    }

    /// Execute a single task using the specified or default strategy
    pub async fn execute_task(
        &self,
        task: crate::execution_types::ExecutionTask,
        strategy_name: Option<&str>,
    ) -> SklResult<crate::execution_types::TaskResult> {
        let strategy_name = strategy_name.unwrap_or(&self.config.default_strategy);

        let strategy = self.strategies.get(strategy_name).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Strategy '{}' not found", strategy_name))
        })?;

        // Check resource availability
        self.resource_manager.check_availability(&task.requirements)?;

        // Allocate resources
        let allocation = self.resource_manager.allocate_resources(&task)?;

        // Execute the task
        let result = strategy.execute_task(task, &mut self.context).await?;

        // Release resources
        self.resource_manager.release_resources(&allocation)?;

        // Update metrics
        self.update_metrics(&result)?;

        Ok(result)
    }

    /// Execute multiple tasks as a batch
    pub async fn execute_batch(
        &self,
        tasks: Vec<crate::execution_types::ExecutionTask>,
        strategy_name: Option<&str>,
    ) -> SklResult<Vec<crate::execution_types::TaskResult>> {
        let strategy_name = strategy_name.unwrap_or(&self.config.default_strategy);

        let strategy = self.strategies.get(strategy_name).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Strategy '{}' not found", strategy_name))
        })?;

        // Pre-validate resource availability
        for task in &tasks {
            self.resource_manager.check_availability(&task.requirements)?;
        }

        // Execute batch
        let results = strategy.execute_batch(tasks, &mut self.context).await?;

        // Update metrics for all results
        for result in &results {
            self.update_metrics(result)?;
        }

        Ok(results)
    }

    /// Get current engine metrics
    pub fn get_metrics(&self) -> SklResult<crate::performance_monitoring::ExecutionMetrics> {
        let metrics = self.metrics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire metrics lock".to_string())
        })?;
        Ok(metrics.clone())
    }

    /// Update engine configuration
    pub fn update_config(&mut self, config: ExecutionEngineConfig) -> SklResult<()> {
        self.config = config;
        Ok(())
    }

    /// Get list of registered strategies
    pub fn list_strategies(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }

    /// Get strategy information
    pub fn get_strategy_info(&self, name: &str) -> Option<(String, crate::execution_strategies::StrategyMetrics)> {
        self.strategies.get(name).map(|strategy| {
            (strategy.description().to_string(), strategy.get_metrics())
        })
    }

    /// Shutdown the engine gracefully
    pub async fn shutdown(&mut self) -> SklResult<()> {
        // Wait for running tasks to complete
        self.scheduler.shutdown_gracefully().await?;

        // Release all resources
        self.resource_manager.release_all_resources()?;

        Ok(())
    }

    /// Update internal metrics based on task results
    fn update_metrics(&self, result: &crate::execution_types::TaskResult) -> SklResult<()> {
        let mut metrics = self.metrics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire metrics lock".to_string())
        })?;

        metrics.update_with_result(result);
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_execution_engine_config() {
        let config = ExecutionEngineConfig::default();
        assert_eq!(config.name, "default_execution_engine");
        assert_eq!(config.default_strategy, "sequential");
        assert!(config.resource_constraints.max_cpu_cores.is_none());
        assert!(config.performance_goals.target_utilization.is_some());
    }

    #[test]
    fn test_resource_constraints() {
        let constraints = ResourceConstraints {
            max_cpu_cores: Some(8),
            max_memory: Some(16 * 1024 * 1024 * 1024), // 16GB
            max_task_time: Some(Duration::from_secs(600)), // 10 minutes
            max_concurrent_tasks: Some(100),
            io_bandwidth: Some(IoLimits {
                max_read_bps: 1_000_000_000, // 1 GB/s
                max_write_bps: 500_000_000,  // 500 MB/s
                max_concurrent_io: 50,
                max_open_files: Some(1000),
            }),
            gpu_constraints: None,
            network_constraints: None,
        };

        assert_eq!(constraints.max_cpu_cores, Some(8));
        assert_eq!(constraints.max_memory, Some(16 * 1024 * 1024 * 1024));
        assert!(constraints.io_bandwidth.is_some());
    }

    #[test]
    fn test_backoff_strategy() {
        let exponential = BackoffStrategy::Exponential {
            base: Duration::from_millis(100),
            multiplier: 2.0,
        };

        match exponential {
            BackoffStrategy::Exponential { base, multiplier } => {
                assert_eq!(base, Duration::from_millis(100));
                assert_eq!(multiplier, 2.0);
            }
            _ => panic!("Expected exponential backoff"),
        }
    }

    #[test]
    fn test_health_check_config() {
        let config = HealthCheckConfig {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
            endpoints: vec![HealthCheckEndpoint {
                name: "test_endpoint".to_string(),
                method: HealthCheckMethod::Http {
                    url: "http://localhost:8080/health".to_string(),
                    method: "GET".to_string(),
                },
                expected_response: HealthCheckExpectation::HttpStatus(200),
            }],
        };

        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.endpoints.len(), 1);
    }

    #[test]
    fn test_monitoring_config() {
        let config = MonitoringConfig::default();
        assert!(config.enable_performance);
        assert!(config.enable_resources);
        assert!(config.enable_errors);
        assert!(!config.enable_tracing);
        assert_eq!(config.collection_interval, Duration::from_secs(10));
    }

    #[test]
    fn test_execution_context() {
        let context = ExecutionContext::new();
        assert!(!context.id.is_empty());
        assert!(!context.environment.is_empty());
        assert!(context.start_time <= SystemTime::now());
    }

    #[test]
    fn test_execution_statistics_defaults() {
        let stats = ExecutionStatistics::default();
        assert_eq!(stats.total_tasks, 0);
        assert_eq!(stats.successful_tasks, 0);
        assert_eq!(stats.failed_tasks, 0);
        assert_eq!(stats.total_execution_time, Duration::new(0, 0));
        assert_eq!(stats.average_task_duration, Duration::new(0, 0));
    }

    #[test]
    fn test_shared_state() {
        let context = ExecutionContext::new();

        // Set a value
        let result = context.set_shared("test_key".to_string(), 42u32);
        assert!(result.is_ok());

        // Get the value back
        let value: Option<u32> = context.get_shared("test_key");
        assert_eq!(value, Some(42));

        // Try to get non-existent key
        let missing: Option<String> = context.get_shared("missing_key");
        assert_eq!(missing, None);
    }
}