//! Shared strategy trait, configuration, health, and metrics types.

use crate::execution_config::{
    FaultToleranceConfig, PerformanceGoals, ResourceConstraints,
};
use crate::task_definitions::{ExecutionTask, TaskRequirements, TaskResult};
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, SystemTime};


/// Error record for tracking
#[derive(Debug, Clone)]
pub struct ErrorRecord {
    /// Error timestamp
    pub timestamp: SystemTime,
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Task ID that caused the error
    pub task_id: String,
}
/// Error statistics
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    /// Error count by type
    pub error_counts: HashMap<String, u64>,
    /// Recent errors
    pub recent_errors: Vec<ErrorRecord>,
    /// Error rate over time
    pub error_rate_history: Vec<f64>,
}
/// Execution environments
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionEnvironment {
    /// Development
    Development,
    /// Testing
    Testing,
    /// Staging
    Staging,
    /// Production
    Production,
    /// Custom
    Custom(String),
}
/// Health issues
#[derive(Debug, Clone)]
pub struct HealthIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Issue timestamp
    pub timestamp: SystemTime,
    /// Suggested resolution
    pub resolution: Option<String>,
}
/// Health status levels
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Healthy
    Healthy,
    /// Warning
    Warning,
    /// Critical
    Critical,
    /// Unknown
    Unknown,
}
/// Issue severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}
/// Network I/O statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkIoStats {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
}
/// Performance data point for time series analysis
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Throughput at this point
    pub throughput: f64,
    /// Latency at this point
    pub latency: Duration,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}
/// Performance summary metrics
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Tasks completed
    pub tasks_completed: u64,
    /// Tasks failed
    pub tasks_failed: u64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Throughput (tasks per second)
    pub throughput: f64,
    /// Error rate
    pub error_rate: f64,
}
/// Resource usage statistics
#[derive(Debug, Clone, Default)]
pub struct ResourceStats {
    /// CPU usage history
    pub cpu_usage: Vec<f64>,
    /// Memory usage history
    pub memory_usage: Vec<u64>,
    /// GPU usage history
    pub gpu_usage: Option<Vec<f64>>,
    /// Network I/O statistics
    pub network_io: NetworkIoStats,
    /// Storage I/O statistics
    pub storage_io: StorageIoStats,
}
/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu: f64,
    /// Memory utilization percentage
    pub memory: f64,
    /// GPU utilization percentage
    pub gpu: Option<f64>,
    /// Network utilization percentage
    pub network: f64,
    /// Storage utilization percentage
    pub storage: f64,
}
/// Storage I/O statistics
#[derive(Debug, Clone, Default)]
pub struct StorageIoStats {
    /// Bytes read
    pub bytes_read: u64,
    /// Bytes written
    pub bytes_written: u64,
    /// Read operations
    pub read_ops: u64,
    /// Write operations
    pub write_ops: u64,
}
/// Strategy configuration settings
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Strategy name identifier
    pub name: String,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Task execution timeout
    pub timeout: Option<Duration>,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Performance goals
    pub performance_goals: PerformanceGoals,
    /// Fault tolerance configuration
    pub fault_tolerance: FaultToleranceConfig,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable detailed logging
    pub enable_logging: bool,
    /// Custom configuration parameters
    pub custom_params: HashMap<String, String>,
    /// Strategy priority
    pub priority: StrategyPriority,
    /// Execution environment
    pub environment: ExecutionEnvironment,
}
impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            name: "default_strategy".to_string(),
            max_concurrent_tasks: 10,
            timeout: Some(Duration::from_secs(300)),
            resource_constraints: ResourceConstraints::default(),
            performance_goals: PerformanceGoals::default(),
            fault_tolerance: FaultToleranceConfig::default(),
            enable_metrics: true,
            enable_logging: false,
            custom_params: HashMap::new(),
            priority: StrategyPriority::Normal,
            environment: ExecutionEnvironment::Development,
        }
    }
}
/// Strategy health status
#[derive(Debug, Clone)]
pub struct StrategyHealth {
    /// Overall health status
    pub status: HealthStatus,
    /// Last health check timestamp
    pub last_check: SystemTime,
    /// Health score (0.0 to 1.0)
    pub score: f64,
    /// Active issues
    pub issues: Vec<HealthIssue>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Performance metrics
    pub performance_summary: PerformanceSummary,
}
/// Comprehensive strategy metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Strategy uptime
    pub uptime: Duration,
    /// Total tasks processed
    pub total_tasks: u64,
    /// Successful tasks
    pub successful_tasks: u64,
    /// Failed tasks
    pub failed_tasks: u64,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Current throughput
    pub current_throughput: f64,
    /// Resource usage statistics
    pub resource_stats: ResourceStats,
    /// Performance metrics over time
    pub performance_history: Vec<PerformanceDataPoint>,
    /// Error statistics
    pub error_stats: ErrorStats,
}
impl Default for StrategyMetrics {
    fn default() -> Self {
        Self {
            uptime: Duration::from_secs(0),
            total_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            average_execution_time: Duration::from_millis(0),
            peak_throughput: 0.0,
            current_throughput: 0.0,
            resource_stats: ResourceStats::default(),
            performance_history: Vec::new(),
            error_stats: ErrorStats::default(),
        }
    }
}
/// Strategy priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum StrategyPriority {
    /// Low
    Low,
    /// Normal
    Normal,
    /// High
    High,
    /// Critical
    Critical,
}
/// Strategy execution state
#[derive(Debug, Clone, Default)]
pub struct StrategyState {
    /// Is strategy initialized?
    pub initialized: bool,
    /// Is strategy running?
    pub running: bool,
    /// Is strategy paused?
    pub paused: bool,
    /// Current execution context
    pub current_task: Option<String>,
    /// State metadata
    pub metadata: HashMap<String, String>,
}
/// Core execution strategy trait that all strategies must implement
pub trait ExecutionStrategy: Send + Sync + fmt::Debug {
    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy description
    fn description(&self) -> &str;

    /// Get strategy configuration
    fn config(&self) -> &StrategyConfig;

    /// Configure the strategy
    fn configure(
        &mut self,
        config: StrategyConfig,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>>;

    /// Initialize the strategy
    fn initialize(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>>;

    /// Execute a single task
    fn execute_task(
        &self,
        task: ExecutionTask,
    ) -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send + '_>>;

    /// Execute multiple tasks in batch
    fn execute_batch(
        &self,
        tasks: Vec<ExecutionTask>,
    ) -> Pin<Box<dyn Future<Output = SklResult<Vec<TaskResult>>> + Send + '_>>;

    /// Check if strategy can handle the given task
    fn can_handle(&self, task: &ExecutionTask) -> bool;

    /// Estimate execution time for a task
    fn estimate_execution_time(&self, task: &ExecutionTask) -> Option<Duration>;

    /// Get current strategy health status
    fn health_status(&self) -> StrategyHealth;

    /// Get strategy metrics
    fn metrics(&self) -> StrategyMetrics;

    /// Shutdown the strategy gracefully
    fn shutdown(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>>;

    /// Pause strategy execution
    fn pause(&mut self) -> SklResult<()>;

    /// Resume strategy execution
    fn resume(&mut self) -> SklResult<()>;

    /// Scale strategy resources
    fn scale(
        &mut self,
        scale_factor: f64,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>>;

    /// Get resource requirements for a task
    fn get_resource_requirements(&self, task: &ExecutionTask) -> TaskRequirements;

    /// Validate task compatibility
    fn validate_task(&self, task: &ExecutionTask) -> SklResult<()>;
}

