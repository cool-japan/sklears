//! # Execution Core Module
//!
//! Core traits, types, and engine foundations for the composable execution framework.
//! This module defines the fundamental abstractions and interfaces used throughout
//! the execution system.

use sklears_core::error::Result as SklResult;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Core execution strategy trait that all strategies must implement
pub trait ExecutionStrategy: Send + Sync + fmt::Debug {
    /// Get the strategy name
    fn name(&self) -> &str;

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

    /// Check if strategy can handle the given task requirements
    fn can_handle(&self, requirements: &TaskRequirements) -> bool;

    /// Get strategy configuration
    fn config(&self) -> &StrategyConfig;

    /// Update strategy configuration
    fn update_config(&mut self, config: StrategyConfig) -> SklResult<()>;

    /// Get strategy health status
    fn health_check(&self) -> Health;

    /// Get strategy metrics
    fn get_metrics(&self) -> StrategyMetrics;
}

/// Core task scheduler trait
pub trait TaskScheduler: Send + Sync + fmt::Debug {
    /// Schedule a task for execution
    fn schedule_task(&mut self, task: ExecutionTask) -> SklResult<String>;

    /// Schedule multiple tasks
    fn schedule_batch(&mut self, tasks: Vec<ExecutionTask>) -> SklResult<Vec<String>>;

    /// Get next task to execute
    fn next_task(&mut self) -> Option<ExecutionTask>;

    /// Cancel a scheduled task
    fn cancel_task(&mut self, task_id: &str) -> SklResult<()>;

    /// Get queue status
    fn queue_status(&self) -> QueueStatus;

    /// Start the scheduler
    fn start(&mut self) -> SklResult<()>;

    /// Pause the scheduler
    fn pause(&mut self) -> SklResult<()>;

    /// Resume the scheduler
    fn resume(&mut self) -> SklResult<()>;

    /// Shutdown the scheduler
    fn shutdown(&mut self) -> SklResult<()>;

    /// Get scheduler health
    fn health_check(&self) -> Health;
}

/// Core resource manager trait
pub trait ResourceManager: Send + Sync + fmt::Debug {
    /// Check if resources are available for the given requirements
    fn check_availability(&self, requirements: &TaskRequirements) -> SklResult<bool>;

    /// Allocate resources for a task
    fn allocate_resources(&self, requirements: &TaskRequirements) -> SklResult<ResourceAllocation>;

    /// Release allocated resources
    fn release_resources(&self, allocation: ResourceAllocation) -> SklResult<()>;

    /// Get current resource usage summary
    fn get_usage_summary(&self) -> ResourceUsageSummary;

    /// Initialize the resource manager
    fn initialize(&self) -> SklResult<()>;

    /// Shutdown the resource manager
    fn shutdown(&self) -> SklResult<()>;

    /// Get resource manager health
    fn health_check(&self) -> Health;
}

/// Execution task definition
pub struct ExecutionTask {
    /// Unique task identifier
    pub id: String,
    /// Task metadata
    pub metadata: TaskMetadata,
    /// Task requirements
    pub requirements: TaskRequirements,
    /// Task constraints
    pub constraints: TaskConstraints,
    /// Task execution function
    pub execution_fn: Arc<dyn Fn() -> SklResult<Box<dyn Any + Send + Sync>> + Send + Sync>,
    /// Task creation timestamp
    pub created_at: SystemTime,
    /// Task priority
    pub priority: TaskPriority,
    /// Task status
    pub status: TaskStatus,
}

impl std::fmt::Debug for ExecutionTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionTask")
            .field("id", &self.id)
            .field("metadata", &self.metadata)
            .field("requirements", &self.requirements)
            .field("constraints", &self.constraints)
            .field("execution_fn", &"<function>")
            .field("created_at", &self.created_at)
            .field("priority", &self.priority)
            .field("status", &self.status)
            .finish()
    }
}

impl Clone for ExecutionTask {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            requirements: self.requirements.clone(),
            constraints: self.constraints.clone(),
            execution_fn: self.execution_fn.clone(), // Arc can be cloned
            created_at: self.created_at,
            priority: self.priority.clone(),
            status: self.status.clone(),
        }
    }
}

/// Task metadata containing descriptive information
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Task name
    pub name: String,
    /// Task type classification
    pub task_type: TaskType,
    /// Task description
    pub description: Option<String>,
    /// Task tags for categorization
    pub tags: Vec<String>,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

/// Task resource requirements
#[derive(Debug, Clone)]
pub struct TaskRequirements {
    /// Required CPU cores
    pub cpu_cores: Option<usize>,
    /// Required memory in bytes
    pub memory: Option<u64>,
    /// Required GPU memory in bytes
    pub gpu_memory: Option<u64>,
    /// Required GPU devices
    pub gpu_devices: Vec<String>,
    /// Required network bandwidth in bytes/sec
    pub network_bandwidth: Option<u64>,
    /// Required disk space in bytes
    pub disk_space: Option<u64>,
    /// Required execution environment
    pub execution_location: ExecutionLocation,
    /// Task affinity preferences
    pub affinity: TaskAffinity,
}

/// Task execution constraints
#[derive(Debug, Clone)]
pub struct TaskConstraints {
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Maximum memory usage
    pub max_memory_usage: Option<u64>,
    /// Maximum retries allowed
    pub max_retries: Option<usize>,
    /// Timeout for the task
    pub timeout: Option<Duration>,
    /// Dependencies on other tasks
    pub dependencies: Vec<String>,
    /// Exclusivity requirements
    pub exclusive_resources: Vec<String>,
}

/// Task execution result
#[derive(Debug)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: String,
    /// Execution status
    pub status: TaskStatus,
    /// Result data
    pub result: Option<Box<dyn Any + Send + Sync>>,
    /// Execution error if any
    pub error: Option<TaskError>,
    /// Execution metrics
    pub metrics: TaskExecutionMetrics,
    /// Task completion timestamp
    pub completed_at: SystemTime,
}

/// Task execution status
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
    /// Task is retrying after failure
    Retrying,
}

/// Task type classification
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Data preprocessing task
    Preprocess,
    /// Model training task
    Fit,
    /// Prediction/inference task
    Predict,
    /// Data transformation task
    Transform,
    /// Model evaluation task
    Evaluate,
    /// Feature extraction task
    FeatureExtraction,
    /// Data validation task
    Validate,
    /// Model optimization task
    Optimize,
    /// Custom task type
    Custom(String),
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TaskPriority {
    /// Lowest priority
    Low = 0,
    /// Normal priority
    #[default]
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

/// Task execution location preferences
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionLocation {
    /// Execute on any available node
    Any,
    /// Execute on local machine only
    Local,
    /// Execute on remote cluster
    Remote,
    /// Execute on specific node
    Specific(String),
    /// Execute on GPU
    Gpu,
    /// Execute in cloud
    Cloud,
}

/// Task affinity preferences
#[derive(Debug, Clone, Default)]
pub struct TaskAffinity {
    /// Preferred nodes
    pub preferred_nodes: Vec<String>,
    /// Avoid nodes
    pub avoid_nodes: Vec<String>,
    /// CPU affinity
    pub cpu_affinity: Vec<usize>,
    /// NUMA node preference
    pub numa_node: Option<usize>,
}

/// Task execution error
#[derive(Debug, Clone)]
pub struct TaskError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error category
    pub category: ErrorCategory,
    /// Retry information
    pub retry_info: Option<RetryInfo>,
    /// Stack trace if available
    pub stack_trace: Option<String>,
}

/// Error categories
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    /// Resource allocation error
    ResourceAllocation,
    /// Execution timeout
    Timeout,
    /// Invalid input or configuration
    InvalidInput,
    /// System/hardware error
    System,
    /// Network/communication error
    Network,
    /// User code error
    UserCode,
    /// Unknown error
    Unknown,
}

/// Retry information
#[derive(Debug, Clone)]
pub struct RetryInfo {
    /// Current retry attempt
    pub attempt: usize,
    /// Maximum retries allowed
    pub max_retries: usize,
    /// Next retry delay
    pub next_retry_delay: Duration,
    /// Retry strategy being used
    pub strategy: RetryStrategy,
}

/// Retry strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RetryStrategy {
    /// Fixed delay between retries
    Fixed(Duration),
    /// Exponential backoff
    Exponential { base: Duration, multiplier: f64 },
    /// Linear backoff
    Linear { base: Duration, increment: Duration },
    /// Custom strategy
    Custom(String),
}

/// Task execution metrics
#[derive(Debug, Clone)]
pub struct TaskExecutionMetrics {
    /// Task start time
    pub start_time: SystemTime,
    /// Task end time
    pub end_time: Option<SystemTime>,
    /// Total execution duration
    pub execution_duration: Option<Duration>,
    /// CPU time used
    pub cpu_time: Option<Duration>,
    /// Memory usage metrics
    pub memory_usage: TaskMemoryUsage,
    /// Resource usage metrics
    pub resource_usage: TaskResourceUsage,
    /// Performance metrics
    pub performance_metrics: TaskPerformanceMetrics,
}

/// Task memory usage metrics
#[derive(Debug, Clone)]
pub struct TaskMemoryUsage {
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// Average memory usage in bytes
    pub average_memory: u64,
    /// Memory allocations count
    pub allocations: u64,
    /// Memory deallocations count
    pub deallocations: u64,
}

/// Task resource usage metrics
#[derive(Debug, Clone)]
pub struct TaskResourceUsage {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// GPU utilization percentage
    pub gpu_utilization: Option<f64>,
    /// Network I/O bytes
    pub network_io: u64,
    /// Disk I/O bytes
    pub disk_io: u64,
}

/// Task performance metrics
#[derive(Debug, Clone)]
pub struct TaskPerformanceMetrics {
    /// Operations per second
    pub ops_per_second: Option<f64>,
    /// Data throughput in bytes/sec
    pub throughput: Option<f64>,
    /// Processing latency
    pub latency: Option<Duration>,
    /// Quality metrics (accuracy, precision, etc.)
    pub quality_metrics: HashMap<String, f64>,
}

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Strategy name
    pub name: String,
    /// Strategy parameters
    pub parameters: HashMap<String, String>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
    /// Enable debugging
    pub debug_enabled: bool,
}

/// Resource limits for strategies
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum CPU cores
    pub max_cpu_cores: Option<usize>,
    /// Maximum memory in bytes
    pub max_memory: Option<u64>,
    /// Maximum GPU memory in bytes
    pub max_gpu_memory: Option<u64>,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: Option<usize>,
}

/// Performance targets for strategies
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Target throughput (tasks/sec)
    pub target_throughput: Option<f64>,
    /// Target latency (milliseconds)
    pub target_latency: Option<f64>,
    /// Target resource utilization percentage
    pub target_utilization: Option<f64>,
}

/// Strategy execution metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Total tasks executed
    pub total_tasks: u64,
    /// Successful tasks
    pub successful_tasks: u64,
    /// Failed tasks
    pub failed_tasks: u64,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Current throughput
    pub current_throughput: f64,
    /// Resource utilization
    pub resource_utilization: f64,
}

/// Health status
#[derive(Debug, Clone, PartialEq)]
pub enum Health {
    /// Component is healthy
    Healthy,
    /// Component is unhealthy
    Unhealthy,
    /// Component is degraded
    Degraded,
    /// Component status is unknown
    Unknown,
}

/// Queue status information
#[derive(Debug, Clone)]
pub struct QueueStatus {
    /// Number of pending tasks
    pub pending_tasks: usize,
    /// Number of running tasks
    pub running_tasks: usize,
    /// Queue capacity
    pub queue_capacity: Option<usize>,
    /// Queue utilization percentage
    pub utilization: f64,
}

/// Resource allocation information
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocation identifier
    pub id: String,
    /// Allocated CPU cores
    pub cpu_cores: Vec<usize>,
    /// Allocated memory in bytes
    pub memory: u64,
    /// Allocated GPU devices
    pub gpu_devices: Vec<String>,
    /// Allocation timestamp
    pub allocated_at: SystemTime,
    /// Allocation timeout
    pub timeout: Option<Duration>,
}

/// Resource usage summary
#[derive(Debug, Clone)]
pub struct ResourceUsageSummary {
    /// Total CPU cores available
    pub total_cpu_cores: usize,
    /// Used CPU cores
    pub used_cpu_cores: usize,
    /// Total memory available in bytes
    pub total_memory: u64,
    /// Used memory in bytes
    pub used_memory: u64,
    /// Total GPU devices available
    pub total_gpu_devices: usize,
    /// Used GPU devices
    pub used_gpu_devices: usize,
    /// Current resource utilization percentage
    pub utilization_percentage: f64,
}

/// Default implementations
impl Default for TaskRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: None,
            memory: None,
            gpu_memory: None,
            gpu_devices: Vec::new(),
            network_bandwidth: None,
            disk_space: None,
            execution_location: ExecutionLocation::Any,
            affinity: TaskAffinity::default(),
        }
    }
}

impl Default for TaskConstraints {
    fn default() -> Self {
        Self {
            max_execution_time: None,
            max_memory_usage: None,
            max_retries: Some(3),
            timeout: Some(Duration::from_secs(300)), // 5 minutes default
            dependencies: Vec::new(),
            exclusive_resources: Vec::new(),
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            parameters: HashMap::new(),
            resource_limits: ResourceLimits::default(),
            performance_targets: PerformanceTargets::default(),
            debug_enabled: false,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_cores: None,
            max_memory: None,
            max_gpu_memory: None,
            max_concurrent_tasks: Some(10),
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_throughput: None,
            target_latency: None,
            target_utilization: Some(80.0), // 80% utilization target
        }
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "Pending"),
            TaskStatus::Running => write!(f, "Running"),
            TaskStatus::Completed => write!(f, "Completed"),
            TaskStatus::Failed => write!(f, "Failed"),
            TaskStatus::Cancelled => write!(f, "Cancelled"),
            TaskStatus::TimedOut => write!(f, "TimedOut"),
            TaskStatus::Retrying => write!(f, "Retrying"),
        }
    }
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskType::Preprocess => write!(f, "Preprocess"),
            TaskType::Fit => write!(f, "Fit"),
            TaskType::Predict => write!(f, "Predict"),
            TaskType::Transform => write!(f, "Transform"),
            TaskType::Evaluate => write!(f, "Evaluate"),
            TaskType::FeatureExtraction => write!(f, "FeatureExtraction"),
            TaskType::Validate => write!(f, "Validate"),
            TaskType::Optimize => write!(f, "Optimize"),
            TaskType::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

impl fmt::Display for Health {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Health::Healthy => write!(f, "Healthy"),
            Health::Unhealthy => write!(f, "Unhealthy"),
            Health::Degraded => write!(f, "Degraded"),
            Health::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Builder pattern for `ExecutionTask`
pub struct TaskBuilder {
    id: Option<String>,
    metadata: TaskMetadata,
    requirements: TaskRequirements,
    constraints: TaskConstraints,
    priority: TaskPriority,
}

impl TaskBuilder {
    /// Create a new task builder
    #[must_use]
    pub fn new(name: &str, task_type: TaskType) -> Self {
        Self {
            id: None,
            metadata: TaskMetadata {
                name: name.to_string(),
                task_type,
                description: None,
                tags: Vec::new(),
                custom_fields: HashMap::new(),
            },
            requirements: TaskRequirements::default(),
            constraints: TaskConstraints::default(),
            priority: TaskPriority::default(),
        }
    }

    /// Set task ID
    #[must_use]
    pub fn id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    /// Set task description
    #[must_use]
    pub fn description(mut self, description: &str) -> Self {
        self.metadata.description = Some(description.to_string());
        self
    }

    /// Add task tag
    #[must_use]
    pub fn tag(mut self, tag: &str) -> Self {
        self.metadata.tags.push(tag.to_string());
        self
    }

    /// Set CPU requirements
    #[must_use]
    pub fn cpu_cores(mut self, cores: usize) -> Self {
        self.requirements.cpu_cores = Some(cores);
        self
    }

    /// Set memory requirements in bytes
    #[must_use]
    pub fn memory(mut self, memory: u64) -> Self {
        self.requirements.memory = Some(memory);
        self
    }

    /// Set task priority
    #[must_use]
    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set maximum execution time
    #[must_use]
    pub fn max_execution_time(mut self, duration: Duration) -> Self {
        self.constraints.max_execution_time = Some(duration);
        self
    }

    /// Build the task (requires execution function to be provided separately)
    #[must_use]
    pub fn build_metadata(
        self,
    ) -> (
        TaskMetadata,
        TaskRequirements,
        TaskConstraints,
        TaskPriority,
    ) {
        (
            self.metadata,
            self.requirements,
            self.constraints,
            self.priority,
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_builder() {
        let (metadata, requirements, constraints, priority) =
            TaskBuilder::new("test_task", TaskType::Preprocess)
                .id("task_001")
                .description("Test task description")
                .tag("test")
                .cpu_cores(4)
                .memory(1024 * 1024 * 1024) // 1GB
                .priority(TaskPriority::High)
                .max_execution_time(Duration::from_secs(60))
                .build_metadata();

        assert_eq!(metadata.name, "test_task");
        assert_eq!(metadata.task_type, TaskType::Preprocess);
        assert_eq!(
            metadata.description,
            Some("Test task description".to_string())
        );
        assert_eq!(metadata.tags, vec!["test"]);
        assert_eq!(requirements.cpu_cores, Some(4));
        assert_eq!(requirements.memory, Some(1024 * 1024 * 1024));
        assert_eq!(priority, TaskPriority::High);
        assert_eq!(
            constraints.max_execution_time,
            Some(Duration::from_secs(60))
        );
    }

    #[test]
    fn test_task_status_display() {
        assert_eq!(format!("{}", TaskStatus::Pending), "Pending");
        assert_eq!(format!("{}", TaskStatus::Running), "Running");
        assert_eq!(format!("{}", TaskStatus::Completed), "Completed");
        assert_eq!(format!("{}", TaskStatus::Failed), "Failed");
    }

    #[test]
    fn test_health_status() {
        assert_eq!(Health::Healthy, Health::Healthy);
        assert_ne!(Health::Healthy, Health::Unhealthy);
        assert_eq!(format!("{}", Health::Healthy), "Healthy");
    }

    #[test]
    fn test_default_implementations() {
        let requirements = TaskRequirements::default();
        assert_eq!(requirements.execution_location, ExecutionLocation::Any);

        let constraints = TaskConstraints::default();
        assert_eq!(constraints.max_retries, Some(3));
        assert_eq!(constraints.timeout, Some(Duration::from_secs(300)));

        let priority = TaskPriority::default();
        assert_eq!(priority, TaskPriority::Normal);
    }
}
