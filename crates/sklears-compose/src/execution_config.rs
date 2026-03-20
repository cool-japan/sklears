//! Execution Engine Configuration System
//!
//! This module provides comprehensive configuration management for the composable execution engine,
//! including resource constraints, performance goals, fault tolerance settings, and monitoring
//! configurations. The configuration system is designed to be flexible, extensible, and
//! production-ready with validation and best practices built-in.
//!
//! # Configuration Architecture
//!
//! The configuration system follows a hierarchical structure:
//!
//! ```text
//! ExecutionEngineConfig
//! ├── ResourceConstraints       // CPU, memory, GPU, network limits
//! ├── PerformanceGoals          // Throughput, latency, utilization targets
//! ├── EnergyEfficiencyGoal      // Power consumption optimization
//! ├── FaultToleranceConfig      // Retry, failover, circuit breaker settings
//! ├── MonitoringConfig          // Metrics collection and observability
//! └── IoLimits                  // I/O bandwidth and operation limits
//! ```
//!
//! # Configuration Examples
//!
//! ## High-Performance Computing Configuration
//! ```rust,ignore
//! use sklears_compose::execution_config::*;
//!
//! let config = ExecutionEngineConfig {
//!     name: "hpc_cluster".to_string(),
//!     default_strategy: "distributed".to_string(),
//!     resource_constraints: ResourceConstraints {
//!         max_cpu_cores: Some(128),
//!         max_memory: Some(1024 * 1024 * 1024 * 1024), // 1TB
//!         max_gpu_devices: Some(8),
//!         max_concurrent_tasks: Some(1000),
//!         max_task_time: Some(Duration::from_hours(24)),
//!         memory_limit_per_task: Some(8 * 1024 * 1024 * 1024), // 8GB per task
//!         ..Default::default()
//!     },
//!     performance_goals: PerformanceGoals {
//!         target_throughput: Some(10000.0), // 10k tasks/sec
//!         target_latency: Some(1.0),         // 1ms avg latency
//!         target_utilization: Some(95.0),   // 95% resource utilization
//!         target_efficiency: Some(98.0),    // 98% efficiency
//!         ..Default::default()
//!     },
//!     energy_efficiency: EnergyEfficiencyGoal {
//!         target_power_consumption: Some(50000.0), // 50kW
//!         enable_power_capping: true,
//!         dynamic_frequency_scaling: true,
//!         energy_aware_scheduling: true,
//!         ..Default::default()
//!     },
//!     fault_tolerance: FaultToleranceConfig {
//!         enable_retry: true,
//!         max_retries: 5,
//!         backoff_strategy: BackoffStrategy::ExponentialWithJitter {
//!             base_delay: Duration::from_millis(100),
//!             max_delay: Duration::from_secs(60),
//!             jitter_factor: 0.1,
//!         },
//!         enable_circuit_breaker: true,
//!         circuit_breaker_threshold: 50,
//!         enable_failover: true,
//!         failover_targets: vec!["backup_cluster".to_string()],
//!         health_check: HealthCheckConfig {
//!             enabled: true,
//!             interval: Duration::from_secs(30),
//!             timeout: Duration::from_secs(5),
//!             failure_threshold: 3,
//!             recovery_threshold: 2,
//!         },
//!         ..Default::default()
//!     },
//!     monitoring: MonitoringConfig {
//!         collection_interval: Duration::from_secs(1),
//!         enable_detailed_metrics: true,
//!         enable_distributed_tracing: true,
//!         enable_profiling: true,
//!         export_prometheus: true,
//!         export_jaeger: true,
//!         custom_metrics: vec!["gpu_utilization", "memory_bandwidth"].iter().map(|s| s.to_string()).collect(),
//!         ..Default::default()
//!     },
//!     io_limits: IoLimits {
//!         max_read_bandwidth: Some(100 * 1024 * 1024 * 1024), // 100 GB/s
//!         max_write_bandwidth: Some(80 * 1024 * 1024 * 1024), // 80 GB/s
//!         max_iops: Some(1000000), // 1M IOPS
//!         max_concurrent_io: Some(10000),
//!         io_priority: IoPriority::High,
//!         ..Default::default()
//!     },
//! };
//! ```
//!
//! ## Edge Computing Configuration
//! ```rust,ignore
//! let edge_config = ExecutionEngineConfig {
//!     name: "edge_device".to_string(),
//!     default_strategy: "streaming".to_string(),
//!     resource_constraints: ResourceConstraints {
//!         max_cpu_cores: Some(4),
//!         max_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
//!         max_concurrent_tasks: Some(50),
//!         max_task_time: Some(Duration::from_secs(30)),
//!         memory_limit_per_task: Some(100 * 1024 * 1024), // 100MB per task
//!         ..Default::default()
//!     },
//!     energy_efficiency: EnergyEfficiencyGoal {
//!         target_power_consumption: Some(50.0), // 50W
//!         enable_power_capping: true,
//!         dynamic_frequency_scaling: true,
//!         thermal_throttling: true,
//!         ..Default::default()
//!     },
//!     performance_goals: PerformanceGoals {
//!         target_latency: Some(10.0), // 10ms for real-time processing
//!         target_efficiency: Some(85.0),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## Development Configuration
//! ```rust,ignore
//! let dev_config = ExecutionEngineConfig {
//!     name: "development".to_string(),
//!     default_strategy: "sequential".to_string(),
//!     resource_constraints: ResourceConstraints {
//!         max_cpu_cores: Some(2),
//!         max_memory: Some(4 * 1024 * 1024 * 1024), // 4GB
//!         max_concurrent_tasks: Some(10),
//!         ..Default::default()
//!     },
//!     monitoring: MonitoringConfig {
//!         collection_interval: Duration::from_secs(5),
//!         enable_detailed_metrics: true,
//!         enable_debug_logging: true,
//!         ..Default::default()
//!     },
//!     fault_tolerance: FaultToleranceConfig {
//!         enable_retry: true,
//!         max_retries: 2,
//!         backoff_strategy: BackoffStrategy::Linear {
//!             delay: Duration::from_millis(500),
//!         },
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```

use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::time::Duration;

/// Main configuration for the execution engine
#[derive(Debug, Clone)]
pub struct ExecutionEngineConfig {
    /// Engine name identifier
    pub name: String,
    /// Default execution strategy to use
    pub default_strategy: String,
    /// Resource allocation constraints
    pub resource_constraints: ResourceConstraints,
    /// Performance optimization goals
    pub performance_goals: PerformanceGoals,
    /// Energy efficiency optimization goals
    pub energy_efficiency: EnergyEfficiencyGoal,
    /// Fault tolerance and reliability settings
    pub fault_tolerance: FaultToleranceConfig,
    /// Monitoring and observability configuration
    pub monitoring: MonitoringConfig,
    /// I/O operation limits and constraints
    pub io_limits: IoLimits,
}

/// Resource allocation constraints and limits
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_cpu_cores: Option<usize>,
    pub max_memory: Option<u64>,
    pub max_gpu_devices: Option<usize>,
    pub max_concurrent_tasks: Option<usize>,
    pub max_task_time: Option<Duration>,
    pub memory_limit_per_task: Option<u64>,
    pub cpu_quota_per_task: Option<f64>,
    pub gpu_memory_per_task: Option<u64>,
    pub network_bandwidth_per_task: Option<u64>,
    pub max_queue_depth: Option<usize>,
    pub priority_levels: Option<u8>,
    pub isolation_mode: ResourceIsolationMode,
}

/// Resource isolation modes
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceIsolationMode {
    /// No isolation - shared resources
    None,
    /// Soft limits with best-effort isolation
    Soft,
    /// Hard limits with strict isolation
    Hard,
    /// Container-based isolation
    Container,
    /// Virtual machine isolation
    VirtualMachine,
}

/// Performance optimization goals and targets
#[derive(Debug, Clone)]
pub struct PerformanceGoals {
    /// Target throughput (tasks per second)
    pub target_throughput: Option<f64>,
    /// Target average latency in milliseconds
    pub target_latency: Option<f64>,
    /// Target resource utilization percentage (0-100)
    pub target_utilization: Option<f64>,
    /// Target system efficiency percentage (0-100)
    pub target_efficiency: Option<f64>,
    /// Target cache hit ratio (0-1)
    pub target_cache_hit_ratio: Option<f64>,
    /// Target memory bandwidth utilization
    pub target_memory_bandwidth: Option<f64>,
    /// Target CPU frequency scaling
    pub target_cpu_frequency: Option<f64>,
    /// Performance optimization mode
    pub optimization_mode: PerformanceMode,
    /// Quality of Service level
    pub qos_level: QualityOfService,
}

/// Performance optimization modes
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceMode {
    /// Optimize for maximum throughput
    Throughput,
    /// Optimize for minimum latency
    Latency,
    /// Balance between throughput and latency
    Balanced,
    /// Optimize for energy efficiency
    EnergyEfficient,
    /// Optimize for cost effectiveness
    CostOptimized,
    /// Custom optimization profile
    Custom(String),
}

/// Quality of Service levels
#[derive(Debug, Clone, PartialEq)]
pub enum QualityOfService {
    /// Best effort - no guarantees
    BestEffort,
    /// Guaranteed resources and performance
    Guaranteed,
    /// Burstable performance with baseline
    Burstable { baseline: f64 },
    /// Real-time guarantees
    RealTime { deadline: Duration },
}

/// Energy efficiency optimization goals
#[derive(Debug, Clone)]
pub struct EnergyEfficiencyGoal {
    /// Target power consumption in watts
    pub target_power_consumption: Option<f64>,
    /// Enable power capping
    pub enable_power_capping: bool,
    /// Enable dynamic voltage and frequency scaling
    pub dynamic_frequency_scaling: bool,
    /// Enable energy-aware task scheduling
    pub energy_aware_scheduling: bool,
    /// Enable thermal throttling
    pub thermal_throttling: bool,
    /// Target temperature threshold
    pub target_temperature: Option<f64>,
    /// Power efficiency mode
    pub power_mode: PowerMode,
    /// Carbon footprint optimization
    pub carbon_awareness: CarbonAwareness,
}

/// Power consumption modes
#[derive(Debug, Clone, PartialEq)]
pub enum PowerMode {
    /// Maximum performance regardless of power
    PerformanceFirst,
    /// Balance performance and power consumption
    Balanced,
    /// Minimize power consumption
    PowerSaver,
    /// Eco-friendly mode with renewable energy preference
    EcoMode,
    /// Custom power profile
    Custom {
        max_watts: f64,
        efficiency_target: f64,
    },
}

/// Carbon footprint awareness settings
#[derive(Debug, Clone, Default)]
pub struct CarbonAwareness {
    /// Enable carbon-aware scheduling
    pub enabled: bool,
    /// Prefer renewable energy sources
    pub prefer_renewable: bool,
    /// Carbon intensity threshold (gCO2/kWh)
    pub carbon_intensity_threshold: Option<f64>,
    /// Time-of-use optimization for green energy
    pub time_of_use_optimization: bool,
}

/// Fault tolerance and reliability configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    pub enable_retry: bool,
    pub max_retries: u32,
    pub backoff_strategy: BackoffStrategy,
    pub enable_circuit_breaker: bool,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_timeout: Duration,
    pub enable_failover: bool,
    pub failover_targets: Vec<String>,
    pub health_check: HealthCheckConfig,
    pub enable_graceful_degradation: bool,
    pub failure_detection: FailureDetectionStrategy,
    pub recovery_strategy: RecoveryStrategy,
}

/// Backoff strategies for retry mechanisms
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed { delay: Duration },
    /// Linear increase in delay
    Linear { delay: Duration },
    /// Exponential backoff
    Exponential {
        base_delay: Duration,
        max_delay: Duration,
    },
    /// Exponential backoff with random jitter
    ExponentialWithJitter {
        base_delay: Duration,
        max_delay: Duration,
        jitter_factor: f64,
    },
    /// Custom backoff function
    Custom(fn(u32) -> Duration),
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,
    /// Health check interval
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub recovery_threshold: u32,
    /// Health check endpoints
    pub endpoints: Vec<HealthCheckEndpoint>,
    /// Custom health check functions
    pub custom_checks: Vec<String>,
}

/// Health check endpoint configuration
#[derive(Debug, Clone)]
pub struct HealthCheckEndpoint {
    /// Endpoint name
    pub name: String,
    /// Endpoint URL or path
    pub url: String,
    /// HTTP method for the check
    pub method: String,
    /// Expected response code
    pub expected_status: u16,
    /// Request timeout
    pub timeout: Duration,
}

/// Failure detection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum FailureDetectionStrategy {
    /// Simple timeout-based detection
    Timeout,
    /// Heartbeat-based detection
    Heartbeat { interval: Duration },
    /// Performance degradation detection
    PerformanceDegradation { threshold: f64 },
    /// Resource exhaustion detection
    ResourceExhaustion,
    /// Multi-criteria detection
    MultiCriteria(Vec<FailureDetectionStrategy>),
}

/// Recovery strategies after failures
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Restart the failed component
    Restart,
    /// Failover to backup instance
    Failover,
    /// Graceful degradation with reduced functionality
    GracefulDegradation,
    /// Manual intervention required
    Manual,
    /// Custom recovery procedure
    Custom(String),
}

/// Monitoring and observability configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Enable detailed performance metrics
    pub enable_detailed_metrics: bool,
    /// Enable distributed tracing
    pub enable_distributed_tracing: bool,
    /// Enable execution profiling
    pub enable_profiling: bool,
    /// Enable debug logging
    pub enable_debug_logging: bool,
    /// Export metrics to Prometheus
    pub export_prometheus: bool,
    /// Export traces to Jaeger
    pub export_jaeger: bool,
    /// Export metrics to `InfluxDB`
    pub export_influxdb: bool,
    /// Custom metrics to collect
    pub custom_metrics: Vec<String>,
    /// Alerting configuration
    pub alerting: AlertingConfig,
    /// Sampling configuration for tracing
    pub sampling_config: SamplingConfig,
}

/// Alerting system configuration
#[derive(Debug, Clone)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert channels (email, slack, webhook, etc.)
    pub channels: Vec<AlertChannel>,
    /// Alert rules and thresholds
    pub rules: Vec<AlertRule>,
    /// Alert aggregation window
    pub aggregation_window: Duration,
    /// Rate limiting for alerts
    pub rate_limit: AlertRateLimit,
}

/// Alert channel configuration
#[derive(Debug, Clone)]
pub struct AlertChannel {
    /// Channel type (email, slack, webhook, etc.)
    pub channel_type: String,
    /// Channel configuration
    pub config: HashMap<String, String>,
    /// Channel priority
    pub priority: AlertPriority,
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric to monitor
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message template
    pub message_template: String,
}

/// Alert priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertPriority {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    /// Info
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Comparison operators for alert rules
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    /// Greater
    Greater,
    /// GreaterEqual
    GreaterEqual,
    /// Less
    Less,
    /// LessEqual
    LessEqual,
    /// Equal
    Equal,
    /// NotEqual
    NotEqual,
}

/// Alert rate limiting configuration
#[derive(Debug, Clone)]
pub struct AlertRateLimit {
    /// Maximum alerts per time window
    pub max_alerts: u32,
    /// Time window for rate limiting
    pub time_window: Duration,
    /// Burst allowance
    pub burst_allowance: u32,
}

/// Sampling configuration for tracing
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Adaptive sampling based on system load
    pub adaptive_sampling: bool,
    /// Minimum samples per second
    pub min_samples_per_second: u32,
    /// Maximum samples per second
    pub max_samples_per_second: u32,
}

/// I/O operation limits and constraints
#[derive(Debug, Clone)]
pub struct IoLimits {
    /// Maximum read bandwidth in bytes per second
    pub max_read_bandwidth: Option<u64>,
    /// Maximum write bandwidth in bytes per second
    pub max_write_bandwidth: Option<u64>,
    /// Maximum I/O operations per second
    pub max_iops: Option<u64>,
    /// Maximum concurrent I/O operations
    pub max_concurrent_io: Option<usize>,
    /// I/O priority level
    pub io_priority: IoPriority,
    /// I/O scheduler type
    pub io_scheduler: IoScheduler,
    /// Buffer size for I/O operations
    pub buffer_size: Option<usize>,
    /// Enable direct I/O
    pub direct_io: bool,
    /// Enable I/O compression
    pub compression: IoCompression,
}

/// I/O priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum IoPriority {
    /// Low
    Low,
    /// Normal
    Normal,
    /// High
    High,
    /// RealTime
    RealTime,
}

/// I/O scheduler types
#[derive(Debug, Clone, PartialEq)]
pub enum IoScheduler {
    /// Completely Fair Queuing
    CFQ,
    /// Deadline scheduler
    Deadline,
    /// NOOP scheduler
    Noop,
    /// Budget Fair Queueing
    BFQ,
    /// Multiqueue Block Layer
    MQ,
}

/// I/O compression settings
#[derive(Debug, Clone)]
pub struct IoCompression {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub level: u8,
    /// Compression threshold (minimum file size)
    pub threshold: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionAlgorithm {
    /// Gzip
    Gzip,
    /// Lz4
    Lz4,
    /// Zstd
    Zstd,
    /// Snappy
    Snappy,
    /// Brotli
    Brotli,
}

/// Default implementation for `ExecutionEngineConfig`
impl Default for ExecutionEngineConfig {
    fn default() -> Self {
        Self {
            name: "default_engine".to_string(),
            default_strategy: "sequential".to_string(),
            resource_constraints: ResourceConstraints::default(),
            performance_goals: PerformanceGoals::default(),
            energy_efficiency: EnergyEfficiencyGoal::default(),
            fault_tolerance: FaultToleranceConfig::default(),
            monitoring: MonitoringConfig::default(),
            io_limits: IoLimits::default(),
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_cores: None,
            max_memory: None,
            max_gpu_devices: None,
            max_concurrent_tasks: Some(100),
            max_task_time: Some(Duration::from_secs(3600)), // 1 hour
            memory_limit_per_task: None,
            cpu_quota_per_task: None,
            gpu_memory_per_task: None,
            network_bandwidth_per_task: None,
            max_queue_depth: Some(1000),
            priority_levels: Some(3),
            isolation_mode: ResourceIsolationMode::Soft,
        }
    }
}

impl Default for PerformanceGoals {
    fn default() -> Self {
        Self {
            target_throughput: None,
            target_latency: None,
            target_utilization: Some(80.0),    // 80% utilization
            target_efficiency: Some(90.0),     // 90% efficiency
            target_cache_hit_ratio: Some(0.8), // 80% cache hits
            target_memory_bandwidth: None,
            target_cpu_frequency: None,
            optimization_mode: PerformanceMode::Balanced,
            qos_level: QualityOfService::BestEffort,
        }
    }
}

impl Default for EnergyEfficiencyGoal {
    fn default() -> Self {
        Self {
            target_power_consumption: None,
            enable_power_capping: false,
            dynamic_frequency_scaling: true,
            energy_aware_scheduling: false,
            thermal_throttling: true,
            target_temperature: Some(75.0), // 75°C
            power_mode: PowerMode::Balanced,
            carbon_awareness: CarbonAwareness::default(),
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
                max_delay: Duration::from_secs(30),
            },
            enable_circuit_breaker: false,
            circuit_breaker_threshold: 10,
            circuit_breaker_timeout: Duration::from_secs(60),
            enable_failover: false,
            failover_targets: Vec::new(),
            health_check: HealthCheckConfig::default(),
            enable_graceful_degradation: true,
            failure_detection: FailureDetectionStrategy::Timeout,
            recovery_strategy: RecoveryStrategy::Restart,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
            endpoints: Vec::new(),
            custom_checks: Vec::new(),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(10),
            enable_detailed_metrics: false,
            enable_distributed_tracing: false,
            enable_profiling: false,
            enable_debug_logging: false,
            export_prometheus: false,
            export_jaeger: false,
            export_influxdb: false,
            custom_metrics: Vec::new(),
            alerting: AlertingConfig::default(),
            sampling_config: SamplingConfig::default(),
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            channels: Vec::new(),
            rules: Vec::new(),
            aggregation_window: Duration::from_secs(60),
            rate_limit: AlertRateLimit::default(),
        }
    }
}

impl Default for AlertRateLimit {
    fn default() -> Self {
        Self {
            max_alerts: 10,
            time_window: Duration::from_secs(300), // 5 minutes
            burst_allowance: 3,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 0.1, // 10% sampling
            adaptive_sampling: false,
            min_samples_per_second: 1,
            max_samples_per_second: 100,
        }
    }
}

impl Default for IoLimits {
    fn default() -> Self {
        Self {
            max_read_bandwidth: None,
            max_write_bandwidth: None,
            max_iops: None,
            max_concurrent_io: Some(100),
            io_priority: IoPriority::Normal,
            io_scheduler: IoScheduler::CFQ,
            buffer_size: Some(64 * 1024), // 64KB
            direct_io: false,
            compression: IoCompression::default(),
        }
    }
}

impl Default for IoCompression {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::Lz4,
            level: 6,
            threshold: 1024, // 1KB
        }
    }
}

/// Type alias for convenient default configuration
pub type DefaultExecutionEngineConfig = ExecutionEngineConfig;

/// Configuration validation
impl ExecutionEngineConfig {
    /// Validate the configuration for consistency and correctness
    pub fn validate(&self) -> SklResult<()> {
        // Validate resource constraints
        self.resource_constraints.validate()?;

        // Validate performance goals
        self.performance_goals.validate()?;

        // Validate fault tolerance settings
        self.fault_tolerance.validate()?;

        // Validate monitoring configuration
        self.monitoring.validate()?;

        // Validate I/O limits
        self.io_limits.validate()?;

        Ok(())
    }
}

impl ResourceConstraints {
    pub fn validate(&self) -> SklResult<()> {
        if let Some(cores) = self.max_cpu_cores {
            if cores == 0 {
                return Err(SklearsError::InvalidInput(
                    "CPU cores must be greater than 0".to_string(),
                ));
            }
        }

        if let Some(memory) = self.max_memory {
            if memory == 0 {
                return Err(SklearsError::InvalidInput(
                    "Memory must be greater than 0".to_string(),
                ));
            }
        }

        if let Some(tasks) = self.max_concurrent_tasks {
            if tasks == 0 {
                return Err(SklearsError::InvalidInput(
                    "Concurrent tasks must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl PerformanceGoals {
    pub fn validate(&self) -> SklResult<()> {
        if let Some(throughput) = self.target_throughput {
            if throughput <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Throughput must be positive".to_string(),
                ));
            }
        }

        if let Some(latency) = self.target_latency {
            if latency <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Latency must be positive".to_string(),
                ));
            }
        }

        if let Some(utilization) = self.target_utilization {
            if !(0.0..=100.0).contains(&utilization) {
                return Err(SklearsError::InvalidInput(
                    "Utilization must be between 0 and 100".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl FaultToleranceConfig {
    pub fn validate(&self) -> SklResult<()> {
        if self.enable_retry && self.max_retries == 0 {
            return Err(SklearsError::InvalidInput(
                "Max retries must be greater than 0 when retry is enabled".to_string(),
            ));
        }

        if self.enable_circuit_breaker && self.circuit_breaker_threshold == 0 {
            return Err(SklearsError::InvalidInput(
                "Circuit breaker threshold must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl MonitoringConfig {
    pub fn validate(&self) -> SklResult<()> {
        if self.collection_interval.as_millis() == 0 {
            return Err(SklearsError::InvalidInput(
                "Collection interval must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl IoLimits {
    pub fn validate(&self) -> SklResult<()> {
        if let Some(concurrent) = self.max_concurrent_io {
            if concurrent == 0 {
                return Err(SklearsError::InvalidInput(
                    "Concurrent I/O must be greater than 0".to_string(),
                ));
            }
        }

        if let Some(buffer_size) = self.buffer_size {
            if buffer_size == 0 {
                return Err(SklearsError::InvalidInput(
                    "Buffer size must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExecutionEngineConfig::default();
        assert_eq!(config.name, "default_engine");
        assert_eq!(config.default_strategy, "sequential");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_constraints_validation() {
        let mut constraints = ResourceConstraints::default();
        assert!(constraints.validate().is_ok());

        // Test invalid CPU cores
        constraints.max_cpu_cores = Some(0);
        assert!(constraints.validate().is_err());

        // Test invalid memory
        constraints.max_cpu_cores = Some(4);
        constraints.max_memory = Some(0);
        assert!(constraints.validate().is_err());
    }

    #[test]
    fn test_performance_goals_validation() {
        let mut goals = PerformanceGoals::default();
        assert!(goals.validate().is_ok());

        // Test invalid throughput
        goals.target_throughput = Some(-1.0);
        assert!(goals.validate().is_err());

        // Test invalid utilization
        goals.target_throughput = Some(100.0);
        goals.target_utilization = Some(150.0);
        assert!(goals.validate().is_err());
    }

    #[test]
    fn test_fault_tolerance_validation() {
        let mut config = FaultToleranceConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid retry configuration
        config.enable_retry = true;
        config.max_retries = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_backoff_strategy() {
        let fixed = BackoffStrategy::Fixed {
            delay: Duration::from_millis(100),
        };
        let linear = BackoffStrategy::Linear {
            delay: Duration::from_millis(100),
        };
        let exponential = BackoffStrategy::Exponential {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
        };

        // Test that all variants can be created
        assert!(matches!(fixed, BackoffStrategy::Fixed { .. }));
        assert!(matches!(linear, BackoffStrategy::Linear { .. }));
        assert!(matches!(exponential, BackoffStrategy::Exponential { .. }));
    }

    #[test]
    fn test_energy_efficiency_config() {
        let mut config = EnergyEfficiencyGoal::default();
        assert!(!config.enable_power_capping);
        assert!(config.dynamic_frequency_scaling);

        config.enable_power_capping = true;
        config.target_power_consumption = Some(100.0);
        assert_eq!(config.target_power_consumption, Some(100.0));
    }

    #[test]
    fn test_monitoring_config() {
        let config = MonitoringConfig::default();
        assert!(!config.enable_detailed_metrics);
        assert!(!config.enable_distributed_tracing);
        assert_eq!(config.collection_interval, Duration::from_secs(10));
    }

    #[test]
    fn test_io_limits() {
        let limits = IoLimits::default();
        assert_eq!(limits.io_priority, IoPriority::Normal);
        assert_eq!(limits.io_scheduler, IoScheduler::CFQ);
        assert_eq!(limits.buffer_size, Some(64 * 1024));
    }
}
