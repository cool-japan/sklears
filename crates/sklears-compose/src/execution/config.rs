//! Configuration structures for the execution engine
//!
//! This module provides all configuration-related structures used by the
//! composable execution engine.

use std::collections::HashMap;
use std::time::Duration;

/// Configuration for the execution engine
#[derive(Debug, Clone)]
pub struct ExecutionEngineConfig {
    /// Engine name
    pub name: String,
    /// Default execution strategy
    pub default_strategy: String,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Performance goals
    pub performance_goals: PerformanceGoals,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

/// Resource constraints for execution
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum CPU cores to use
    pub max_cpu_cores: Option<usize>,
    /// Maximum memory usage in bytes
    pub max_memory: Option<u64>,
    /// Maximum execution time per task
    pub max_task_time: Option<Duration>,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: Option<usize>,
    /// I/O bandwidth limits
    pub io_bandwidth: Option<IoLimits>,
}

/// I/O bandwidth limits
#[derive(Debug, Clone)]
pub struct IoLimits {
    pub max_read_bps: u64,
    pub max_write_bps: u64,
    pub max_concurrent_io: usize,
}

/// Performance goals for the execution engine
#[derive(Debug, Clone)]
pub struct PerformanceGoals {
    /// Target throughput (tasks/sec)
    pub target_throughput: f64,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Target resource utilization (0.0-1.0)
    pub target_utilization: f64,
    /// Energy efficiency goals
    pub energy_efficiency: Option<EnergyEfficiencyGoal>,
}

/// Energy efficiency goals
#[derive(Debug, Clone)]
pub struct EnergyEfficiencyGoal {
    /// Target power consumption in watts
    pub max_power_watts: f64,
    /// Energy budget per task in joules
    pub energy_budget_per_task: f64,
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Enable retry on failure
    pub enable_retry: bool,
    /// Maximum number of retries
    pub max_retries: usize,
    /// Backoff strategy for retries
    pub backoff_strategy: BackoffStrategy,
    /// Enable circuit breaker
    pub enable_circuit_breaker: bool,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
}

/// Backoff strategy for retry operations
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed { delay: Duration },
    /// Exponential backoff with base delay
    Exponential {
        base_delay: Duration,
        multiplier: f64,
    },
    /// Linear backoff with increment
    Linear {
        base_delay: Duration,
        increment: Duration,
    },
    /// Custom backoff function
    Custom { delays: Vec<Duration> },
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,
    /// Timeout for health checks
    pub timeout: Duration,
    /// Number of consecutive failures before marking as unhealthy
    pub failure_threshold: usize,
    /// Number of consecutive successes before marking as healthy
    pub success_threshold: usize,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Enable distributed tracing
    pub enable_tracing: bool,
    /// Logging level
    pub log_level: LogLevel,
    /// Custom metric tags
    pub metric_tags: HashMap<String, String>,
}

/// Logging levels
#[derive(Debug, Clone)]
pub enum LogLevel {
    /// Trace
    Trace,
    /// Debug
    Debug,
    /// Info
    Info,
    /// Warn
    Warn,
    /// Error
    Error,
}

/// Configuration for execution strategies
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Strategy name
    pub name: String,
    /// Strategy parameters
    pub parameters: HashMap<String, ParameterValue>,
    /// Resource allocation for this strategy
    pub resource_allocation: StrategyResourceAllocation,
    /// Performance tuning settings
    pub performance_tuning: PerformanceTuning,
}

/// Parameter values for strategy configuration
#[derive(Debug, Clone)]
pub enum ParameterValue {
    /// String
    String(String),
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// Boolean
    Boolean(bool),
    /// Duration
    Duration(Duration),
    /// List
    List(Vec<ParameterValue>),
}

/// Resource allocation configuration for strategies
#[derive(Debug, Clone)]
pub struct StrategyResourceAllocation {
    /// CPU core allocation
    pub cpu_cores: f64,
    /// Memory allocation in bytes
    pub memory_bytes: u64,
    /// Priority level (higher = more priority)
    pub priority: u32,
}

/// Performance tuning configuration
#[derive(Debug, Clone)]
pub struct PerformanceTuning {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable prefetching
    pub prefetching: PrefetchingStrategy,
    /// Caching strategy
    pub caching: CachingStrategy,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
}

/// Optimization levels
#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    None,
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Aggressive
    Aggressive,
}

/// Prefetching strategies
#[derive(Debug, Clone)]
pub enum PrefetchingStrategy {
    None,
    /// Sequential
    Sequential,
    /// Adaptive
    Adaptive,
    /// Predictive
    Predictive,
}

/// Caching strategies
#[derive(Debug, Clone)]
pub enum CachingStrategy {
    None,
    /// LRU
    LRU,
    /// LFU
    LFU,
    /// FIFO
    FIFO,
    /// Adaptive
    Adaptive,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    /// Enable load balancing
    pub enabled: bool,
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Rebalancing threshold
    pub rebalance_threshold: f64,
    /// Minimum load difference for rebalancing
    pub min_load_difference: f64,
}

/// Load balancing algorithms
#[derive(Debug, Clone)]
pub enum LoadBalancingAlgorithm {
    /// RoundRobin
    RoundRobin,
    /// LeastLoaded
    LeastLoaded,
    /// WeightedRoundRobin
    WeightedRoundRobin,
    /// ConsistentHashing
    ConsistentHashing,
    /// Random
    Random,
}

impl Default for ExecutionEngineConfig {
    fn default() -> Self {
        Self {
            name: "default-engine".to_string(),
            default_strategy: "default".to_string(),
            resource_constraints: ResourceConstraints::default(),
            performance_goals: PerformanceGoals::default(),
            fault_tolerance: FaultToleranceConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_cores: None,
            max_memory: None,
            max_task_time: Some(Duration::from_secs(300)), // 5 minutes
            max_concurrent_tasks: Some(10),
            io_bandwidth: None,
        }
    }
}

impl Default for PerformanceGoals {
    fn default() -> Self {
        Self {
            target_throughput: 10.0, // 10 tasks/sec
            max_latency: Duration::from_millis(100),
            target_utilization: 0.8, // 80%
            energy_efficiency: None,
        }
    }
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_retry: true,
            max_retries: 3,
            backoff_strategy: BackoffStrategy::Exponential {
                base_delay: Duration::from_millis(100),
                multiplier: 2.0,
            },
            enable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval: Duration::from_secs(10),
            enable_tracing: false,
            log_level: LogLevel::Info,
            metric_tags: HashMap::new(),
        }
    }
}
