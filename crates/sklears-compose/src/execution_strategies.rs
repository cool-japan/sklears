//! Execution Strategies for Composable Execution Engine
//!
//! This module provides a comprehensive collection of execution strategies that can be
//! dynamically selected and configured based on workload characteristics, resource
//! availability, and performance requirements. Each strategy implements the
//! `ExecutionStrategy` trait and provides specialized optimization for different
//! execution patterns and environments.
//!
//! # Strategy Architecture
//!
//! The execution strategy system is built around pluggable strategy implementations:
//!
//! ```text
//! ExecutionStrategy (trait)
//! ├── SequentialExecutionStrategy    // Single-threaded, deterministic execution
//! ├── BatchExecutionStrategy         // High-throughput batch processing
//! ├── StreamingExecutionStrategy     // Real-time streaming and low-latency
//! ├── GpuExecutionStrategy          // GPU-accelerated computation
//! ├── DistributedExecutionStrategy   // Multi-node distributed execution
//! └── EventDrivenExecutionStrategy   // Reactive event-based execution
//! ```
//!
//! # Strategy Selection Guide
//!
//! ## `SequentialExecutionStrategy`
//! **Best for**: Development, debugging, deterministic workflows
//! - Single-threaded execution
//! - Predictable resource usage
//! - Easy debugging and profiling
//! - Deterministic results
//!
//! ## `BatchExecutionStrategy`
//! **Best for**: ETL pipelines, batch ML training, bulk data processing
//! - High-throughput processing
//! - Optimal resource utilization
//! - Batch size optimization
//! - Parallel task execution
//!
//! ## `StreamingExecutionStrategy`
//! **Best for**: Real-time inference, live data processing, low-latency requirements
//! - Continuous data processing
//! - Low-latency guarantees
//! - Backpressure handling
//! - Stream buffering
//!
//! ## `GpuExecutionStrategy`
//! **Best for**: Deep learning, matrix operations, parallel computations
//! - GPU acceleration
//! - CUDA/ROCm optimization
//! - Memory management
//! - Multi-GPU support
//!
//! ## `DistributedExecutionStrategy`
//! **Best for**: Large-scale processing, cluster computing, fault tolerance
//! - Multi-node execution
//! - Automatic load balancing
//! - Fault tolerance
//! - Dynamic scaling
//!
//! ## `EventDrivenExecutionStrategy`
//! **Best for**: Microservices, reactive systems, event processing
//! - Asynchronous execution
//! - Event-based triggers
//! - Resource efficiency
//! - Scalable architecture
//!
//! # Usage Examples
//!
//! ## Sequential Strategy for Development
//! ```rust,ignore
//! use sklears_compose::execution_strategies::*;
//!
//! let strategy = SequentialExecutionStrategy::builder()
//!     .enable_profiling(true)
//!     .enable_debugging(true)
//!     .checkpoint_interval(Duration::from_secs(60))
//!     .build();
//!
//! let config = StrategyConfig {
//!     max_concurrent_tasks: 1,
//!     timeout: Some(Duration::from_secs(3600)),
//!     enable_metrics: true,
//!     ..Default::default()
//! };
//!
//! strategy.configure(config).await?;
//! ```
//!
//! ## Batch Strategy for High-Throughput Processing
//! ```rust,ignore
//! let batch_strategy = BatchExecutionStrategy::builder()
//!     .batch_size(100)
//!     .max_batch_size(1000)
//!     .batch_timeout(Duration::from_secs(30))
//!     .parallel_batches(4)
//!     .enable_adaptive_batching(true)
//!     .build();
//!
//! let config = StrategyConfig {
//!     max_concurrent_tasks: 400, // 4 batches * 100 tasks
//!     resource_constraints: ResourceConstraints {
//!         max_cpu_cores: Some(16),
//!         max_memory: Some(64 * 1024 * 1024 * 1024), // 64GB
//!         ..Default::default()
//!     },
//!     performance_goals: PerformanceGoals {
//!         target_throughput: Some(10000.0), // 10k tasks/sec
//!         target_utilization: Some(90.0),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## Streaming Strategy for Real-Time Processing
//! ```rust,ignore
//! let streaming_strategy = StreamingExecutionStrategy::builder()
//!     .buffer_size(1000)
//!     .max_latency(Duration::from_millis(10))
//!     .backpressure_strategy(BackpressureStrategy::DropOldest)
//!     .enable_flow_control(true)
//!     .watermark_interval(Duration::from_millis(100))
//!     .build();
//!
//! let config = StrategyConfig {
//!     performance_goals: PerformanceGoals {
//!         target_latency: Some(5.0), // 5ms target
//!         target_throughput: Some(100000.0), // 100k/sec
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## GPU Strategy for Hardware Acceleration
//! ```rust,ignore
//! let gpu_strategy = GpuExecutionStrategy::builder()
//!     .devices(vec!["cuda:0".to_string(), "cuda:1".to_string()])
//!     .memory_pool_size(8 * 1024 * 1024 * 1024) // 8GB pool
//!     .enable_memory_optimization(true)
//!     .enable_mixed_precision(true)
//!     .compute_stream_count(4)
//!     .build();
//!
//! let config = StrategyConfig {
//!     resource_constraints: ResourceConstraints {
//!         gpu_devices: Some(2),
//!         gpu_memory: Some(16 * 1024 * 1024 * 1024), // 16GB total
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## Distributed Strategy for Cluster Computing
//! ```rust,ignore
//! let distributed_strategy = DistributedExecutionStrategy::builder()
//!     .nodes(vec![
//!         "worker1:8080".to_string(),
//!         "worker2:8080".to_string(),
//!         "worker3:8080".to_string(),
//!     ])
//!     .replication_factor(2)
//!     .enable_auto_scaling(true)
//!     .load_balancing_strategy(LoadBalancingStrategy::RoundRobin)
//!     .enable_fault_tolerance(true)
//!     .build();
//!
//! let config = StrategyConfig {
//!     fault_tolerance: FaultToleranceConfig {
//!         enable_retry: true,
//!         max_retries: 3,
//!         enable_failover: true,
//!         enable_circuit_breaker: true,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```

use crate::execution_config::{FaultToleranceConfig, PerformanceGoals, ResourceConstraints};
use crate::task_definitions::{
    ExecutionTask, TaskExecutionMetrics, TaskPerformanceMetrics, TaskPriority, TaskRequirements,
    TaskResourceUsage, TaskResult, TaskStatus,
};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

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

/// Sequential execution strategy for deterministic, single-threaded execution
#[derive(Debug)]
pub struct SequentialExecutionStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Current task queue
    task_queue: Arc<Mutex<VecDeque<ExecutionTask>>>,
    /// Execution metrics
    metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    state: Arc<RwLock<StrategyState>>,
    /// Profiling enabled
    enable_profiling: bool,
    /// Debugging enabled
    enable_debugging: bool,
    /// Checkpoint interval
    checkpoint_interval: Option<Duration>,
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

/// Batch execution strategy for high-throughput processing
#[derive(Debug)]
pub struct BatchExecutionStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Batch size
    batch_size: usize,
    /// Maximum batch size
    max_batch_size: usize,
    /// Batch processing timeout
    batch_timeout: Duration,
    /// Number of parallel batches
    parallel_batches: usize,
    /// Enable adaptive batching
    adaptive_batching: bool,
    /// Current batches
    active_batches: Arc<Mutex<Vec<Batch>>>,
    /// Execution metrics
    metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    state: Arc<RwLock<StrategyState>>,
}

/// Batch of tasks for processing
#[derive(Debug, Clone)]
pub struct Batch {
    /// Batch identifier
    pub id: String,
    /// Tasks in the batch
    pub tasks: Vec<ExecutionTask>,
    /// Batch creation time
    pub created_at: SystemTime,
    /// Batch status
    pub status: BatchStatus,
    /// Batch priority
    pub priority: TaskPriority,
}

/// Batch processing status
#[derive(Debug, Clone, PartialEq)]
pub enum BatchStatus {
    /// Created
    Created,
    /// Queued
    Queued,
    /// Processing
    Processing,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Streaming execution strategy for real-time processing
#[derive(Debug)]
pub struct StreamingExecutionStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Stream buffer size
    buffer_size: usize,
    /// Maximum acceptable latency
    max_latency: Duration,
    /// Backpressure handling strategy
    backpressure_strategy: BackpressureStrategy,
    /// Flow control enabled
    flow_control: bool,
    /// Watermark interval for event time processing
    watermark_interval: Duration,
    /// Active streams
    active_streams: Arc<Mutex<HashMap<String, Stream>>>,
    /// Execution metrics
    metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    state: Arc<RwLock<StrategyState>>,
}

/// Backpressure handling strategies
#[derive(Debug, Clone)]
pub enum BackpressureStrategy {
    /// Block until buffer space available
    Block,
    /// Drop oldest items in buffer
    DropOldest,
    /// Drop newest items
    DropNewest,
    /// Spill to disk
    SpillToDisk,
    /// Scale out resources
    ScaleOut,
}

/// Stream processing context
#[derive(Debug, Clone)]
pub struct Stream {
    /// Stream identifier
    pub id: String,
    /// Stream buffer
    pub buffer: VecDeque<ExecutionTask>,
    /// Stream metrics
    pub metrics: StreamMetrics,
    /// Stream state
    pub state: StreamState,
}

/// Stream-specific metrics
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    /// Items processed
    pub items_processed: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Average processing latency
    pub avg_latency: Duration,
    /// Throughput
    pub throughput: f64,
}

/// Stream processing state
#[derive(Debug, Clone, PartialEq)]
pub enum StreamState {
    /// Active
    Active,
    /// Paused
    Paused,
    /// Draining
    Draining,
    /// Stopped
    Stopped,
}

/// GPU execution strategy for hardware-accelerated computation
#[derive(Debug)]
pub struct GpuExecutionStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// GPU devices to use
    devices: Vec<String>,
    /// GPU memory pool size
    memory_pool_size: u64,
    /// Memory optimization enabled
    memory_optimization: bool,
    /// Mixed precision enabled
    mixed_precision: bool,
    /// Number of compute streams per device
    compute_streams: usize,
    /// GPU context manager
    gpu_context: Arc<Mutex<GpuContext>>,
    /// Execution metrics
    metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    state: Arc<RwLock<StrategyState>>,
}

/// GPU execution context
#[derive(Debug)]
pub struct GpuContext {
    pub devices: HashMap<String, GpuDevice>,
    pub memory_pools: HashMap<String, MemoryPool>,
    pub active_kernels: HashMap<String, GpuKernel>,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device ID
    pub id: String,
    /// Device name
    pub name: String,
    /// Compute capability
    pub compute_capability: String,
    /// Total memory
    pub total_memory: u64,
    /// Available memory
    pub available_memory: u64,
    /// Utilization percentage
    pub utilization: f64,
    /// Temperature
    pub temperature: f64,
}

/// GPU memory pool
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool size
    pub size: u64,
    /// Used memory
    pub used: u64,
    /// Free memory
    pub free: u64,
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

/// Memory allocation strategies
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// FirstFit
    FirstFit,
    /// BestFit
    BestFit,
    /// WorstFit
    WorstFit,
    /// Buddy
    Buddy,
    /// Slab
    Slab,
}

/// GPU kernel execution context
#[derive(Debug)]
pub struct GpuKernel {
    /// Kernel name
    pub name: String,
    /// Device ID
    pub device_id: String,
    /// Stream ID
    pub stream_id: String,
    /// Grid dimensions
    pub grid_dims: (u32, u32, u32),
    /// Block dimensions
    pub block_dims: (u32, u32, u32),
    /// Shared memory size
    pub shared_memory: u32,
}

/// Distributed execution strategy for cluster computing
#[derive(Debug)]
pub struct DistributedExecutionStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Cluster nodes
    nodes: Vec<String>,
    /// Replication factor for fault tolerance
    replication_factor: usize,
    /// Auto-scaling enabled
    auto_scaling: bool,
    /// Load balancing strategy
    load_balancing: LoadBalancingStrategy,
    /// Fault tolerance enabled
    fault_tolerance: bool,
    /// Cluster manager
    cluster_manager: Arc<Mutex<ClusterManager>>,
    /// Execution metrics
    metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    state: Arc<RwLock<StrategyState>>,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// RoundRobin
    RoundRobin,
    /// LeastConnections
    LeastConnections,
    /// WeightedRoundRobin
    WeightedRoundRobin,
    /// ResourceBased
    ResourceBased,
    /// Latency
    Latency,
    /// Custom
    Custom(String),
}

/// Cluster management context
#[derive(Debug)]
pub struct ClusterManager {
    /// Node information
    pub nodes: HashMap<String, ClusterNode>,
    /// Load balancer
    pub load_balancer: LoadBalancer,
    /// Service discovery
    pub service_discovery: ServiceDiscovery,
    /// Health monitoring
    pub health_monitor: HealthMonitor,
}

/// Cluster node information
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node ID
    pub id: String,
    /// Node address
    pub address: String,
    /// Node status
    pub status: NodeStatus,
    /// Available resources
    pub resources: AvailableResources,
    /// Current load
    pub load: NodeLoad,
    /// Health status
    pub health: HealthStatus,
}

/// Node status
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    /// Active
    Active,
    /// Inactive
    Inactive,
    /// Draining
    Draining,
    /// Failed
    Failed,
    /// Maintenance
    Maintenance,
}

/// Available resources on a node
#[derive(Debug, Clone)]
pub struct AvailableResources {
    /// CPU cores
    pub cpu_cores: usize,
    /// Memory in bytes
    pub memory: u64,
    /// GPU devices
    pub gpu_devices: Vec<String>,
    /// Storage space in bytes
    pub storage: u64,
    /// Network bandwidth in bytes/sec
    pub network_bandwidth: u64,
}

/// Node load metrics
#[derive(Debug, Clone)]
pub struct NodeLoad {
    /// CPU load percentage
    pub cpu_load: f64,
    /// Memory usage percentage
    pub memory_usage: f64,
    /// Active tasks count
    pub active_tasks: usize,
    /// Queue depth
    pub queue_depth: usize,
}

/// Load balancer component
#[derive(Debug)]
pub struct LoadBalancer {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Node weights
    pub node_weights: HashMap<String, f64>,
    /// Traffic distribution
    pub traffic_distribution: HashMap<String, u64>,
}

/// Service discovery component
#[derive(Debug)]
pub struct ServiceDiscovery {
    /// Service registry
    pub registry: HashMap<String, ServiceInfo>,
    /// Discovery strategy
    pub strategy: DiscoveryStrategy,
}

/// Service information
#[derive(Debug, Clone)]
pub struct ServiceInfo {
    /// Service name
    pub name: String,
    /// Service endpoints
    pub endpoints: Vec<String>,
    /// Service version
    pub version: String,
    /// Service health
    pub health: HealthStatus,
}

/// Service discovery strategies
#[derive(Debug, Clone)]
pub enum DiscoveryStrategy {
    /// Static
    Static,
    /// DNS
    DNS,
    /// Consul
    Consul,
    /// Etcd
    Etcd,
    /// Kubernetes
    Kubernetes,
    /// Custom
    Custom(String),
}

/// Health monitoring component
#[derive(Debug)]
pub struct HealthMonitor {
    /// Health checks
    pub checks: HashMap<String, HealthCheck>,
    /// Monitoring interval
    pub interval: Duration,
    /// Health thresholds
    pub thresholds: HealthThresholds,
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Check name
    pub name: String,
    /// Check type
    pub check_type: HealthCheckType,
    /// Check interval
    pub interval: Duration,
    /// Timeout
    pub timeout: Duration,
    /// Retry count
    pub retries: u32,
}

/// Health check types
#[derive(Debug, Clone)]
pub enum HealthCheckType {
    /// HttpGet
    HttpGet(String),
    /// TcpConnect
    TcpConnect(String),
    /// Command
    Command(String),
    /// Custom
    Custom(String),
}

/// Health thresholds
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// CPU usage warning threshold
    pub cpu_warning: f64,
    /// CPU usage critical threshold
    pub cpu_critical: f64,
    /// Memory usage warning threshold
    pub memory_warning: f64,
    /// Memory usage critical threshold
    pub memory_critical: f64,
    /// Response time warning threshold
    pub response_time_warning: Duration,
    /// Response time critical threshold
    pub response_time_critical: Duration,
}

/// Event-driven execution strategy for reactive systems
pub struct EventDrivenExecutionStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Event bus
    event_bus: Arc<Mutex<EventBus>>,
    /// Event handlers
    handlers: Arc<Mutex<HashMap<String, EventHandler>>>,
    /// Event queue
    event_queue: Arc<Mutex<VecDeque<Event>>>,
    /// Execution metrics
    metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    state: Arc<RwLock<StrategyState>>,
}

impl std::fmt::Debug for EventDrivenExecutionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventDrivenExecutionStrategy")
            .field("config", &self.config)
            .field("event_bus", &self.event_bus)
            .field(
                "handlers",
                &format!(
                    "<{} handlers>",
                    self.handlers.lock().map(|h| h.len()).unwrap_or(0)
                ),
            )
            .field("event_queue", &self.event_queue)
            .field("metrics", &self.metrics)
            .field("state", &self.state)
            .finish()
    }
}

/// Event bus for message routing
#[derive(Debug)]
pub struct EventBus {
    /// Subscriptions
    pub subscriptions: HashMap<String, Vec<String>>,
    /// Event history
    pub event_history: VecDeque<Event>,
    /// Bus configuration
    pub config: EventBusConfig,
}

/// Event bus configuration
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Maximum event history size
    pub max_history_size: usize,
    /// Event TTL
    pub event_ttl: Duration,
    /// Enable persistence
    pub persistence: bool,
    /// Delivery guarantees
    pub delivery_guarantees: DeliveryGuarantees,
}

/// Delivery guarantee levels
#[derive(Debug, Clone)]
pub enum DeliveryGuarantees {
    /// AtMostOnce
    AtMostOnce,
    /// AtLeastOnce
    AtLeastOnce,
    /// ExactlyOnce
    ExactlyOnce,
}

/// Event for reactive processing
#[derive(Debug, Clone)]
pub struct Event {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: String,
    /// Event data
    pub data: EventData,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event source
    pub source: String,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Event data types
#[derive(Debug, Clone)]
pub enum EventData {
    /// Task
    Task(ExecutionTask),
    /// Metric
    Metric(String, f64),
    /// Status
    Status(String, String),
    /// Custom
    Custom(HashMap<String, String>),
}

/// Event handler for processing events
pub type EventHandler =
    Arc<dyn Fn(Event) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send>> + Send + Sync>;

/// Strategy builder for creating strategies with configuration
pub struct StrategyBuilder {
    strategy_type: StrategyType,
    config: StrategyConfig,
}

/// Strategy types
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyType {
    /// Sequential
    Sequential,
    /// Batch
    Batch,
    /// Streaming
    Streaming,
    /// Gpu
    Gpu,
    /// Distributed
    Distributed,
    /// EventDriven
    EventDriven,
}

/// Strategy registry for managing multiple strategies
#[derive(Debug)]
pub struct StrategyRegistry {
    /// Registered strategies
    strategies: HashMap<String, Box<dyn ExecutionStrategy>>,
    /// Default strategy
    default_strategy: Option<String>,
    /// Strategy metadata
    metadata: HashMap<String, StrategyMetadata>,
}

/// Strategy metadata
#[derive(Debug, Clone)]
pub struct StrategyMetadata {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Strategy version
    pub version: String,
    /// Author
    pub author: String,
    /// Creation date
    pub created_at: SystemTime,
    /// Tags
    pub tags: Vec<String>,
}

/// Strategy factory for creating strategy instances
pub struct StrategyFactory;

// Implementation for SequentialExecutionStrategy
impl Default for SequentialExecutionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl SequentialExecutionStrategy {
    /// Create a new sequential execution strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::default(),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(Mutex::new(StrategyMetrics::default())),
            state: Arc::new(RwLock::new(StrategyState::default())),
            enable_profiling: false,
            enable_debugging: false,
            checkpoint_interval: None,
        }
    }

    /// Create a builder for sequential strategy
    #[must_use]
    pub fn builder() -> SequentialStrategyBuilder {
        SequentialStrategyBuilder::new()
    }
}

/// Builder for sequential execution strategy
pub struct SequentialStrategyBuilder {
    enable_profiling: bool,
    enable_debugging: bool,
    checkpoint_interval: Option<Duration>,
    config: StrategyConfig,
}

impl Default for SequentialStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SequentialStrategyBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            enable_profiling: false,
            enable_debugging: false,
            checkpoint_interval: None,
            config: StrategyConfig::default(),
        }
    }

    #[must_use]
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    #[must_use]
    pub fn enable_debugging(mut self, enable: bool) -> Self {
        self.enable_debugging = enable;
        self
    }

    #[must_use]
    pub fn checkpoint_interval(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    #[must_use]
    pub fn config(mut self, config: StrategyConfig) -> Self {
        self.config = config;
        self
    }

    #[must_use]
    pub fn build(self) -> SequentialExecutionStrategy {
        /// SequentialExecutionStrategy
        SequentialExecutionStrategy {
            config: self.config,
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(Mutex::new(StrategyMetrics::default())),
            state: Arc::new(RwLock::new(StrategyState::default())),
            enable_profiling: self.enable_profiling,
            enable_debugging: self.enable_debugging,
            checkpoint_interval: self.checkpoint_interval,
        }
    }
}

// Implementation for BatchExecutionStrategy
impl Default for BatchExecutionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchExecutionStrategy {
    /// Create a new batch execution strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::default(),
            batch_size: 10,
            max_batch_size: 100,
            batch_timeout: Duration::from_secs(30),
            parallel_batches: 1,
            adaptive_batching: false,
            active_batches: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(StrategyMetrics::default())),
            state: Arc::new(RwLock::new(StrategyState::default())),
        }
    }

    /// Create a builder for batch strategy
    #[must_use]
    pub fn builder() -> BatchStrategyBuilder {
        BatchStrategyBuilder::new()
    }
}

impl ExecutionStrategy for BatchExecutionStrategy {
    fn name(&self) -> &'static str {
        "batch"
    }

    fn description(&self) -> &'static str {
        "Batch execution strategy for high-throughput processing"
    }

    fn config(&self) -> &StrategyConfig {
        &self.config
    }

    fn configure(
        &mut self,
        config: StrategyConfig,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            self.config = config;
            Ok(())
        })
    }

    fn initialize(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            // Initialize batch processing
            Ok(())
        })
    }

    fn execute_task(
        &self,
        task: ExecutionTask,
    ) -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send + '_>> {
        Box::pin(async move {
            // Execute single task in batch context
            let result = TaskResult {
                task_id: task.metadata.id.clone(),
                status: TaskStatus::Completed,
                output: None,
                metrics: TaskExecutionMetrics::default(),
                resource_usage: TaskResourceUsage::default(),
                performance_metrics: TaskPerformanceMetrics::default(),
                error: None,
                logs: Vec::new(),
                artifacts: Vec::new(),
                execution_time: Some(Duration::from_millis(100)),
                metadata: std::collections::HashMap::new(),
            };
            Ok(result)
        })
    }

    fn execute_batch(
        &self,
        tasks: Vec<ExecutionTask>,
    ) -> Pin<Box<dyn Future<Output = SklResult<Vec<TaskResult>>> + Send + '_>> {
        Box::pin(async move {
            // Execute tasks in batch
            let results = tasks
                .into_iter()
                .map(|task| TaskResult {
                    task_id: task.metadata.id,
                    status: TaskStatus::Completed,
                    output: None,
                    metrics: TaskExecutionMetrics::default(),
                    resource_usage: TaskResourceUsage::default(),
                    performance_metrics: TaskPerformanceMetrics::default(),
                    error: None,
                    logs: Vec::new(),
                    artifacts: Vec::new(),
                    execution_time: Some(Duration::from_millis(100)),
                    metadata: std::collections::HashMap::new(),
                })
                .collect();
            Ok(results)
        })
    }

    fn can_handle(&self, _task: &ExecutionTask) -> bool {
        true // Batch strategy can handle any task
    }

    fn estimate_execution_time(&self, _task: &ExecutionTask) -> Option<Duration> {
        Some(Duration::from_millis(100))
    }

    fn health_status(&self) -> StrategyHealth {
        /// StrategyHealth
        StrategyHealth {
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            score: 1.0,
            issues: Vec::new(),
            resource_utilization: ResourceUtilization {
                cpu: 50.0,
                memory: 60.0,
                gpu: None,
                network: 10.0,
                storage: 20.0,
            },
            performance_summary: PerformanceSummary {
                tasks_completed: 0,
                tasks_failed: 0,
                avg_execution_time: Duration::from_millis(100),
                throughput: 10.0,
                error_rate: 0.0,
            },
        }
    }

    fn metrics(&self) -> StrategyMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn shutdown(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            // Shutdown batch processing
            Ok(())
        })
    }

    fn pause(&mut self) -> SklResult<()> {
        // Pause batch execution
        Ok(())
    }

    fn resume(&mut self) -> SklResult<()> {
        // Resume batch execution
        Ok(())
    }

    fn scale(
        &mut self,
        _scale_factor: f64,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            // Scale batch processing resources
            Ok(())
        })
    }

    fn get_resource_requirements(&self, _task: &ExecutionTask) -> TaskRequirements {
        TaskRequirements::default()
    }

    fn validate_task(&self, _task: &ExecutionTask) -> SklResult<()> {
        Ok(())
    }
}

/// Builder for batch execution strategy
pub struct BatchStrategyBuilder {
    batch_size: usize,
    max_batch_size: usize,
    batch_timeout: Duration,
    parallel_batches: usize,
    adaptive_batching: bool,
    config: StrategyConfig,
}

impl Default for BatchStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchStrategyBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            batch_size: 10,
            max_batch_size: 100,
            batch_timeout: Duration::from_secs(30),
            parallel_batches: 1,
            adaptive_batching: false,
            config: StrategyConfig::default(),
        }
    }

    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    #[must_use]
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    #[must_use]
    pub fn batch_timeout(mut self, timeout: Duration) -> Self {
        self.batch_timeout = timeout;
        self
    }

    #[must_use]
    pub fn parallel_batches(mut self, count: usize) -> Self {
        self.parallel_batches = count;
        self
    }

    #[must_use]
    pub fn enable_adaptive_batching(mut self, enable: bool) -> Self {
        self.adaptive_batching = enable;
        self
    }

    #[must_use]
    pub fn config(mut self, config: StrategyConfig) -> Self {
        self.config = config;
        self
    }

    #[must_use]
    pub fn build(self) -> BatchExecutionStrategy {
        /// BatchExecutionStrategy
        BatchExecutionStrategy {
            config: self.config,
            batch_size: self.batch_size,
            max_batch_size: self.max_batch_size,
            batch_timeout: self.batch_timeout,
            parallel_batches: self.parallel_batches,
            adaptive_batching: self.adaptive_batching,
            active_batches: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(StrategyMetrics::default())),
            state: Arc::new(RwLock::new(StrategyState::default())),
        }
    }
}

// Default implementations
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

// Placeholder implementations for ExecutionStrategy trait
// These would be fully implemented with actual execution logic

impl ExecutionStrategy for SequentialExecutionStrategy {
    fn name(&self) -> &'static str {
        "sequential"
    }

    fn description(&self) -> &'static str {
        "Sequential single-threaded execution strategy"
    }

    fn config(&self) -> &StrategyConfig {
        &self.config
    }

    fn configure(
        &mut self,
        config: StrategyConfig,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            self.config = config;
            Ok(())
        })
    }

    fn initialize(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            let mut state = self.state.write().unwrap();
            state.initialized = true;
            state.running = true;
            Ok(())
        })
    }

    fn execute_task(
        &self,
        task: ExecutionTask,
    ) -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send + '_>> {
        Box::pin(async move {
            // Placeholder implementation
            let start_time = SystemTime::now();

            // Simulate task execution
            tokio::time::sleep(Duration::from_millis(100)).await;

            let end_time = SystemTime::now();
            let duration = end_time.duration_since(start_time).unwrap_or_default();

            Ok(TaskResult {
                task_id: task.metadata.id.clone(),
                status: TaskStatus::Completed,
                output: None,
                metrics: crate::task_definitions::TaskExecutionMetrics {
                    start_time,
                    end_time: Some(end_time),
                    duration: Some(duration),
                    queue_wait_time: Duration::from_millis(0),
                    scheduling_time: Duration::from_millis(0),
                    setup_time: Duration::from_millis(0),
                    cleanup_time: Duration::from_millis(0),
                    retry_attempts: 0,
                    checkpoint_count: 0,
                    completion_percentage: 100.0,
                    efficiency_score: Some(0.95),
                },
                resource_usage: crate::task_definitions::TaskResourceUsage {
                    cpu_time: 0.1,                        // 100ms in seconds
                    memory_usage: 80 * 1024 * 1024,       // 80MB
                    peak_memory_usage: 100 * 1024 * 1024, // 100MB
                    disk_io_operations: 7,                // 5 reads + 2 writes
                    network_usage: 3072,                  // 1024 + 2048 bytes
                    gpu_usage: None,
                    gpu_memory_usage: None,
                },
                performance_metrics: crate::task_definitions::TaskPerformanceMetrics {
                    operations_per_second: 10.0,
                    throughput: 10.0, // 10 ops/sec
                    latency: duration,
                    error_rate: 0.0,
                    cache_hit_rate: Some(0.8),
                    efficiency_score: 0.95,
                },
                error: None,
                logs: Vec::new(),
                artifacts: Vec::new(),
                execution_time: Some(duration),
                metadata: std::collections::HashMap::new(),
            })
        })
    }

    fn execute_batch(
        &self,
        tasks: Vec<ExecutionTask>,
    ) -> Pin<Box<dyn Future<Output = SklResult<Vec<TaskResult>>> + Send + '_>> {
        Box::pin(async move {
            let mut results = Vec::new();
            for task in tasks {
                let result = self.execute_task(task).await?;
                results.push(result);
            }
            Ok(results)
        })
    }

    fn can_handle(&self, _task: &ExecutionTask) -> bool {
        true // Sequential strategy can handle any task
    }

    fn estimate_execution_time(&self, _task: &ExecutionTask) -> Option<Duration> {
        Some(Duration::from_millis(100)) // Simple estimate
    }

    fn health_status(&self) -> StrategyHealth {
        /// StrategyHealth
        StrategyHealth {
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            score: 1.0,
            issues: Vec::new(),
            resource_utilization: ResourceUtilization {
                cpu: 50.0,
                memory: 60.0,
                gpu: None,
                network: 20.0,
                storage: 30.0,
            },
            performance_summary: PerformanceSummary {
                tasks_completed: 100,
                tasks_failed: 0,
                avg_execution_time: Duration::from_millis(100),
                throughput: 10.0,
                error_rate: 0.0,
            },
        }
    }

    fn metrics(&self) -> StrategyMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn shutdown(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            let mut state = self.state.write().unwrap();
            state.running = false;
            state.initialized = false;
            Ok(())
        })
    }

    fn pause(&mut self) -> SklResult<()> {
        let mut state = self.state.write().unwrap();
        state.paused = true;
        Ok(())
    }

    fn resume(&mut self) -> SklResult<()> {
        let mut state = self.state.write().unwrap();
        state.paused = false;
        Ok(())
    }

    fn scale(
        &mut self,
        _scale_factor: f64,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            // Sequential strategy doesn't support scaling
            Err(SklearsError::InvalidOperation(
                "Sequential strategy does not support scaling".to_string(),
            ))
        })
    }

    fn get_resource_requirements(&self, task: &ExecutionTask) -> TaskRequirements {
        task.requirements.clone()
    }

    fn validate_task(&self, task: &ExecutionTask) -> SklResult<()> {
        if task.metadata.name.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Task name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// Similar placeholder implementations would be added for other strategies
// For brevity, I'm implementing just the sequential strategy fully here

impl StrategyFactory {
    /// Create a new strategy instance
    pub fn create_strategy(
        strategy_type: StrategyType,
        config: StrategyConfig,
    ) -> SklResult<Box<dyn ExecutionStrategy>> {
        match strategy_type {
            StrategyType::Sequential => {
                let mut strategy = SequentialExecutionStrategy::new();
                strategy.config = config;
                Ok(Box::new(strategy))
            }
            StrategyType::Batch => {
                let mut strategy = BatchExecutionStrategy::new();
                strategy.config = config;
                Ok(Box::new(strategy))
            }
            // Add other strategy types as needed
            _ => Err(SklearsError::NotImplemented(
                "Strategy type not implemented".to_string(),
            )),
        }
    }

    /// Get available strategy types
    #[must_use]
    pub fn available_strategies() -> Vec<StrategyType> {
        vec![
            StrategyType::Sequential,
            StrategyType::Batch,
            StrategyType::Streaming,
            StrategyType::Gpu,
            StrategyType::Distributed,
            StrategyType::EventDriven,
        ]
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyRegistry {
    /// Create a new strategy registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            default_strategy: None,
            metadata: HashMap::new(),
        }
    }

    /// Register a strategy
    pub fn register(
        &mut self,
        name: String,
        strategy: Box<dyn ExecutionStrategy>,
    ) -> SklResult<()> {
        self.strategies.insert(name.clone(), strategy);
        self.metadata.insert(
            name.clone(),
            /// StrategyMetadata
            StrategyMetadata {
                name: name.clone(),
                description: format!("Strategy: {name}"),
                version: "1.0.0".to_string(),
                author: "SkleaRS".to_string(),
                created_at: SystemTime::now(),
                tags: Vec::new(),
            },
        );
        Ok(())
    }

    /// Get a strategy by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Box<dyn ExecutionStrategy>> {
        self.strategies.get(name)
    }

    /// List all registered strategies
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }

    /// Set default strategy
    pub fn set_default(&mut self, name: String) -> SklResult<()> {
        if self.strategies.contains_key(&name) {
            self.default_strategy = Some(name);
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Strategy {name} not found"
            )))
        }
    }

    /// Get default strategy name
    #[must_use]
    pub fn get_default(&self) -> Option<&String> {
        self.default_strategy.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_config() {
        let config = StrategyConfig::default();
        assert_eq!(config.name, "default_strategy");
        assert_eq!(config.max_concurrent_tasks, 10);
        assert_eq!(config.priority, StrategyPriority::Normal);
    }

    #[test]
    fn test_sequential_strategy_creation() {
        let strategy = SequentialExecutionStrategy::new();
        assert_eq!(strategy.name(), "sequential");
        assert_eq!(
            strategy.description(),
            "Sequential single-threaded execution strategy"
        );
    }

    #[test]
    fn test_sequential_strategy_builder() {
        let strategy = SequentialExecutionStrategy::builder()
            .enable_profiling(true)
            .enable_debugging(true)
            .checkpoint_interval(Duration::from_secs(60))
            .build();

        assert!(strategy.enable_profiling);
        assert!(strategy.enable_debugging);
        assert_eq!(strategy.checkpoint_interval, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_batch_strategy_builder() {
        let strategy = BatchExecutionStrategy::builder()
            .batch_size(50)
            .max_batch_size(500)
            .parallel_batches(4)
            .enable_adaptive_batching(true)
            .build();

        assert_eq!(strategy.batch_size, 50);
        assert_eq!(strategy.max_batch_size, 500);
        assert_eq!(strategy.parallel_batches, 4);
        assert!(strategy.adaptive_batching);
    }

    #[test]
    fn test_strategy_factory() {
        let config = StrategyConfig::default();
        let result = StrategyFactory::create_strategy(StrategyType::Sequential, config);
        assert!(result.is_ok());

        let available = StrategyFactory::available_strategies();
        assert!(available.len() > 0);
        assert!(available.contains(&StrategyType::Sequential));
    }

    #[test]
    fn test_strategy_registry() {
        let mut registry = StrategyRegistry::new();
        let strategy = SequentialExecutionStrategy::new();

        let result = registry.register("seq".to_string(), Box::new(strategy));
        assert!(result.is_ok());

        assert!(registry.get("seq").is_some());
        assert_eq!(registry.list().len(), 1);

        let result = registry.set_default("seq".to_string());
        assert!(result.is_ok());
        assert_eq!(registry.get_default(), Some(&"seq".to_string()));
    }

    #[test]
    fn test_strategy_health() {
        let health = StrategyHealth {
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            score: 0.95,
            issues: Vec::new(),
            resource_utilization: ResourceUtilization {
                cpu: 50.0,
                memory: 60.0,
                gpu: None,
                network: 20.0,
                storage: 30.0,
            },
            performance_summary: PerformanceSummary {
                tasks_completed: 100,
                tasks_failed: 2,
                avg_execution_time: Duration::from_millis(150),
                throughput: 50.0,
                error_rate: 0.02,
            },
        };

        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.score, 0.95);
        assert_eq!(health.performance_summary.error_rate, 0.02);
    }

    #[tokio::test]
    async fn test_sequential_strategy_execution() {
        let strategy = SequentialExecutionStrategy::new();
        let task = crate::task_definitions::ExecutionTask::builder()
            .name("test_task")
            .task_type(crate::task_definitions::TaskType::Preprocess)
            .build();

        let result = strategy.execute_task(task).await;
        assert!(result.is_ok());

        let task_result = result.unwrap();
        assert_eq!(task_result.status, TaskStatus::Completed);
        assert!(task_result.metrics.duration.is_some());
    }
}
