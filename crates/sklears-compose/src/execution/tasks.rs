//! Task definitions and management for the composable execution engine
//!
//! This module provides the core task structures and related functionality
//! for the execution engine.

use std::any::Any;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use sklears_core::error::{Result as SklResult, SklearsError};

/// Core execution task structure
#[derive(Debug)]
pub struct ExecutionTask {
    /// Unique task identifier
    pub id: String,
    /// Type of task
    pub task_type: TaskType,
    /// Task metadata
    pub metadata: TaskMetadata,
    /// Resource requirements
    pub requirements: ResourceRequirements,
    /// Input data for the task
    pub input_data: Option<Box<dyn Any + Send + Sync>>,
    /// Task configuration
    pub configuration: TaskConfiguration,
}

/// Task types supported by the execution engine
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Computational task
    Computation,
    /// I/O operation task
    IoOperation,
    /// Network operation task
    NetworkOperation,
    /// Custom task type
    Custom,
}

/// Task metadata for tracking and management
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Human-readable task name
    pub name: String,
    /// Task description
    pub description: String,
    /// Task priority level
    pub priority: TaskPriority,
    /// Estimated execution duration
    pub estimated_duration: Option<Duration>,
    /// Task deadline
    pub deadline: Option<SystemTime>,
    /// Task dependencies (task IDs)
    pub dependencies: Vec<String>,
    /// Custom tags for categorization
    pub tags: Vec<String>,
    /// Task creation timestamp
    pub created_at: SystemTime,
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Resource requirements for task execution
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Required CPU cores
    pub cpu_cores: f64,
    /// Required memory in bytes
    pub memory_bytes: u64,
    /// Required disk space in bytes
    pub disk_bytes: u64,
    /// Required network bandwidth in bytes/sec
    pub network_bandwidth: u64,
    /// Required GPU memory in bytes
    pub gpu_memory_bytes: u64,
    /// Special resource requirements
    pub special_resources: Vec<String>,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_bytes: 100_000_000, // 100MB
            disk_bytes: 0,
            network_bandwidth: 0,
            gpu_memory_bytes: 0,
            special_resources: Vec::new(),
        }
    }
}

/// Task configuration parameters
#[derive(Debug, Clone)]
pub struct TaskConfiguration {
    /// Configuration parameters
    pub parameters: HashMap<String, ConfigValue>,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Working directory
    pub working_directory: Option<String>,
    /// Timeout configuration
    pub timeout: Option<Duration>,
}

impl Default for TaskConfiguration {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            environment: HashMap::new(),
            working_directory: None,
            timeout: Some(Duration::from_secs(3600)), // 1 hour default
        }
    }
}

/// Configuration value types
#[derive(Debug, Clone)]
pub enum ConfigValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Duration value
    Duration(Duration),
    /// List of values
    List(Vec<ConfigValue>),
    /// Nested configuration
    Object(HashMap<String, ConfigValue>),
}

/// Task execution result
#[derive(Debug)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: String,
    /// Execution status
    pub status: TaskStatus,
    /// Execution time
    pub execution_time: Duration,
    /// Resource usage during execution
    pub resource_usage: ResourceUsage,
    /// Output data
    pub output: Option<Box<dyn Any + Send + Sync>>,
    /// Error information if failed
    pub error: Option<TaskError>,
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
}

/// Resource usage during task execution
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Peak CPU usage percentage
    pub cpu_percent: f64,
    /// Peak memory usage in bytes
    pub memory_bytes: u64,
    /// Total I/O operations
    pub io_operations: u64,
    /// Total network bytes transferred
    pub network_bytes: u64,
    /// Peak GPU memory usage in bytes
    pub gpu_memory_bytes: u64,
    /// Execution duration
    pub execution_duration: Duration,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_bytes: 0,
            io_operations: 0,
            network_bytes: 0,
            gpu_memory_bytes: 0,
            execution_duration: Duration::ZERO,
        }
    }
}

/// Task error information
#[derive(Debug, Clone)]
pub struct TaskError {
    /// Error type/category
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Error code (if applicable)
    pub code: Option<i32>,
    /// Stack trace (if applicable)
    pub stack_trace: Option<String>,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
}

/// Task builder for convenient task creation
pub struct TaskBuilder {
    task: ExecutionTask,
}

impl TaskBuilder {
    /// Create a new task builder
    #[must_use]
    pub fn new(id: String, name: String) -> Self {
        Self {
            task: ExecutionTask {
                id,
                task_type: TaskType::Computation,
                metadata: TaskMetadata {
                    name,
                    description: String::new(),
                    priority: TaskPriority::Normal,
                    estimated_duration: None,
                    deadline: None,
                    dependencies: Vec::new(),
                    tags: Vec::new(),
                    created_at: SystemTime::now(),
                },
                requirements: ResourceRequirements::default(),
                input_data: None,
                configuration: TaskConfiguration::default(),
            },
        }
    }

    /// Set task type
    #[must_use]
    pub fn task_type(mut self, task_type: TaskType) -> Self {
        self.task.task_type = task_type;
        self
    }

    /// Set task description
    #[must_use]
    pub fn description(mut self, description: String) -> Self {
        self.task.metadata.description = description;
        self
    }

    /// Set task priority
    #[must_use]
    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.task.metadata.priority = priority;
        self
    }

    /// Set estimated duration
    #[must_use]
    pub fn estimated_duration(mut self, duration: Duration) -> Self {
        self.task.metadata.estimated_duration = Some(duration);
        self
    }

    /// Set task deadline
    #[must_use]
    pub fn deadline(mut self, deadline: SystemTime) -> Self {
        self.task.metadata.deadline = Some(deadline);
        self
    }

    /// Add dependency
    #[must_use]
    pub fn dependency(mut self, task_id: String) -> Self {
        self.task.metadata.dependencies.push(task_id);
        self
    }

    /// Add tag
    #[must_use]
    pub fn tag(mut self, tag: String) -> Self {
        self.task.metadata.tags.push(tag);
        self
    }

    /// Set CPU requirements
    #[must_use]
    pub fn cpu_cores(mut self, cores: f64) -> Self {
        self.task.requirements.cpu_cores = cores;
        self
    }

    /// Set memory requirements
    #[must_use]
    pub fn memory_bytes(mut self, bytes: u64) -> Self {
        self.task.requirements.memory_bytes = bytes;
        self
    }

    /// Set disk requirements
    #[must_use]
    pub fn disk_bytes(mut self, bytes: u64) -> Self {
        self.task.requirements.disk_bytes = bytes;
        self
    }

    /// Set network bandwidth requirements
    #[must_use]
    pub fn network_bandwidth(mut self, bandwidth: u64) -> Self {
        self.task.requirements.network_bandwidth = bandwidth;
        self
    }

    /// Set GPU memory requirements
    #[must_use]
    pub fn gpu_memory_bytes(mut self, bytes: u64) -> Self {
        self.task.requirements.gpu_memory_bytes = bytes;
        self
    }

    /// Add configuration parameter
    pub fn parameter<K: Into<String>>(mut self, key: K, value: ConfigValue) -> Self {
        self.task.configuration.parameters.insert(key.into(), value);
        self
    }

    /// Add environment variable
    pub fn environment<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.task
            .configuration
            .environment
            .insert(key.into(), value.into());
        self
    }

    /// Set working directory
    pub fn working_directory<P: Into<String>>(mut self, path: P) -> Self {
        self.task.configuration.working_directory = Some(path.into());
        self
    }

    /// Set task timeout
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.task.configuration.timeout = Some(timeout);
        self
    }

    /// Build the task
    #[must_use]
    pub fn build(self) -> ExecutionTask {
        self.task
    }
}

/// Task queue for managing pending tasks
#[derive(Debug)]
pub struct TaskQueue {
    /// Queued tasks
    tasks: Vec<ExecutionTask>,
    /// Maximum queue size
    max_size: Option<usize>,
}

impl TaskQueue {
    /// Create a new task queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            max_size: None,
        }
    }

    /// Create a new task queue with maximum size
    #[must_use]
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            tasks: Vec::new(),
            max_size: Some(max_size),
        }
    }

    /// Add a task to the queue
    pub fn enqueue(&mut self, task: ExecutionTask) -> SklResult<()> {
        if let Some(max_size) = self.max_size {
            if self.tasks.len() >= max_size {
                return Err(SklearsError::InvalidInput("Task queue is full".to_string()));
            }
        }

        self.tasks.push(task);
        Ok(())
    }

    /// Remove and return the next task from the queue
    pub fn dequeue(&mut self) -> Option<ExecutionTask> {
        if self.tasks.is_empty() {
            None
        } else {
            Some(self.tasks.remove(0))
        }
    }

    /// Peek at the next task without removing it
    #[must_use]
    pub fn peek(&self) -> Option<&ExecutionTask> {
        self.tasks.first()
    }

    /// Get the number of tasks in the queue
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if the queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Clear all tasks from the queue
    pub fn clear(&mut self) {
        self.tasks.clear();
    }

    /// Get all tasks in the queue
    #[must_use]
    pub fn tasks(&self) -> &[ExecutionTask] {
        &self.tasks
    }

    /// Sort tasks by priority
    pub fn sort_by_priority(&mut self) {
        self.tasks
            .sort_by(|a, b| b.metadata.priority.cmp(&a.metadata.priority));
    }

    /// Filter tasks by predicate
    pub fn filter<F>(&self, predicate: F) -> Vec<&ExecutionTask>
    where
        F: Fn(&ExecutionTask) -> bool,
    {
        self.tasks.iter().filter(|task| predicate(task)).collect()
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_builder() {
        let task = TaskBuilder::new("task1".to_string(), "Test Task".to_string())
            .task_type(TaskType::Computation)
            .priority(TaskPriority::High)
            .cpu_cores(2.0)
            .memory_bytes(1024 * 1024 * 1024)
            .parameter(
                "param1".to_string(),
                ConfigValue::String("value1".to_string()),
            )
            .build();

        assert_eq!(task.id, "task1");
        assert_eq!(task.metadata.name, "Test Task");
        assert_eq!(task.task_type, TaskType::Computation);
        assert_eq!(task.metadata.priority, TaskPriority::High);
        assert_eq!(task.requirements.cpu_cores, 2.0);
        assert_eq!(task.requirements.memory_bytes, 1024 * 1024 * 1024);
        assert!(task.configuration.parameters.contains_key("param1"));
    }

    #[test]
    fn test_task_queue() {
        let mut queue = TaskQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        let task = TaskBuilder::new("task1".to_string(), "Test Task".to_string()).build();
        queue.enqueue(task).unwrap();

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, "task1");
        assert!(queue.is_empty());
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_task_queue_capacity() {
        let mut queue = TaskQueue::with_capacity(1);
        let task1 = TaskBuilder::new("task1".to_string(), "Task 1".to_string()).build();
        let task2 = TaskBuilder::new("task2".to_string(), "Task 2".to_string()).build();

        assert!(queue.enqueue(task1).is_ok());
        assert!(queue.enqueue(task2).is_err());
    }

    #[test]
    fn test_task_queue_priority_sorting() {
        let mut queue = TaskQueue::new();

        let low_task = TaskBuilder::new("low".to_string(), "Low Task".to_string())
            .priority(TaskPriority::Low)
            .build();
        let high_task = TaskBuilder::new("high".to_string(), "High Task".to_string())
            .priority(TaskPriority::High)
            .build();
        let critical_task = TaskBuilder::new("critical".to_string(), "Critical Task".to_string())
            .priority(TaskPriority::Critical)
            .build();

        queue.enqueue(low_task).unwrap();
        queue.enqueue(high_task).unwrap();
        queue.enqueue(critical_task).unwrap();

        queue.sort_by_priority();

        assert_eq!(queue.tasks[0].id, "critical");
        assert_eq!(queue.tasks[1].id, "high");
        assert_eq!(queue.tasks[2].id, "low");
    }
}
