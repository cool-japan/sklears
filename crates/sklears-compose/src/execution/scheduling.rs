//! Task scheduling implementations for the composable execution engine
//!
//! This module provides comprehensive task scheduling capabilities including
//! priority-based scheduling, dependency resolution, and advanced queue management.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use sklears_core::error::{Result as SklResult, SklearsError};

use super::tasks::{ExecutionTask, TaskPriority};

/// Task scheduler trait for pluggable scheduling implementations
pub trait TaskScheduler: Send + Sync {
    /// Schedule a single task
    fn schedule_task(&mut self, task: ExecutionTask) -> SklResult<TaskHandle>;

    /// Schedule multiple tasks as a batch
    fn schedule_batch(&mut self, tasks: Vec<ExecutionTask>) -> SklResult<Vec<TaskHandle>>;

    /// Cancel a scheduled task
    fn cancel_task(&mut self, handle: TaskHandle) -> SklResult<()>;

    /// Get current scheduler status
    fn get_status(&self) -> SchedulerStatus;

    /// Update scheduler configuration
    fn update_config(&mut self, config: SchedulerConfig) -> SklResult<()>;

    /// Get next task to execute (if any)
    fn get_next_task(&mut self) -> Option<ExecutionTask>;

    /// Mark task as completed
    fn mark_completed(&mut self, task_id: &str) -> SklResult<()>;

    /// Mark task as failed
    fn mark_failed(&mut self, task_id: &str, error: String) -> SklResult<()>;

    /// Get scheduling metrics
    fn get_metrics(&self) -> SchedulingMetrics;
}

/// Task handle for tracking scheduled tasks
#[derive(Debug, Clone)]
pub struct TaskHandle {
    /// Task identifier
    pub task_id: String,
    /// Time when task was scheduled
    pub scheduled_at: SystemTime,
    /// Estimated task duration
    pub estimated_duration: Option<Duration>,
    /// Task priority
    pub priority: TaskPriority,
    /// Task dependencies
    pub dependencies: Vec<String>,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Scheduling algorithm to use
    pub algorithm: SchedulingAlgorithm,
    /// Queue management settings
    pub queue_management: QueueManagement,
    /// Priority handling configuration
    pub priority_handling: PriorityHandling,
    /// Dependency resolution settings
    pub dependency_resolution: DependencyResolution,
}

/// Scheduling algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingAlgorithm {
    FIFO,
    Priority,
    ShortestJobFirst,
    RoundRobin {
        quantum: Duration,
    },
    /// Completely Fair Scheduler
    CFS,
    /// Multilevel Queue
    MultilevelQueue,
    /// Weighted Fair Queuing
    WeightedFairQueuing,
    /// Earliest Deadline First
    EarliestDeadlineFirst,
}

/// Queue management configuration
#[derive(Debug, Clone)]
pub struct QueueManagement {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Strategy when queue is full
    pub overflow_strategy: QueueOverflowStrategy,
    /// Queue persistence mode
    pub persistence: QueuePersistence,
}

/// Queue overflow strategies
#[derive(Debug, Clone, PartialEq)]
pub enum QueueOverflowStrategy {
    /// Block new submissions
    Block,
    /// Drop oldest tasks
    DropOldest,
    /// Drop newest tasks
    DropNewest,
    /// Drop lowest priority tasks
    DropLowestPriority,
    /// Reject new submissions with error
    Reject,
}

/// Queue persistence modes
#[derive(Debug, Clone, PartialEq)]
pub enum QueuePersistence {
    /// In-memory only
    Memory,
    /// Persist to disk
    Disk { path: String },
    /// Persist to database
    Database { connection_string: String },
}

/// Priority handling configuration
#[derive(Debug, Clone)]
pub struct PriorityHandling {
    /// Available priority levels
    pub levels: Vec<TaskPriority>,
    /// Priority aging strategy
    pub aging_strategy: AgingStrategy,
    /// Enable starvation prevention
    pub starvation_prevention: bool,
}

/// Priority aging strategies
#[derive(Debug, Clone)]
pub enum AgingStrategy {
    /// No aging
    None,
    /// Linear aging
    Linear { increment_interval: Duration },
    /// Exponential aging
    Exponential {
        base_interval: Duration,
        multiplier: f64,
    },
    /// Custom aging function
    Custom { function_name: String },
}

/// Dependency resolution configuration
#[derive(Debug, Clone)]
pub struct DependencyResolution {
    /// Enable dependency tracking
    pub enable_tracking: bool,
    /// Enable cycle detection
    pub cycle_detection: bool,
    /// Enable deadlock prevention
    pub deadlock_prevention: bool,
    /// Timeout for dependency resolution
    pub resolution_timeout: Duration,
}

/// Scheduler status information
#[derive(Debug, Clone)]
pub struct SchedulerStatus {
    /// Number of tasks in queue
    pub queued_tasks: usize,
    /// Number of currently running tasks
    pub running_tasks: usize,
    /// Number of completed tasks
    pub completed_tasks: u64,
    /// Number of failed tasks
    pub failed_tasks: u64,
    /// Scheduler health status
    pub health: SchedulerHealth,
}

/// Scheduler health status
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerHealth {
    /// Scheduler is healthy
    Healthy,
    /// Scheduler is overloaded
    Overloaded,
    /// Scheduler has errors
    Degraded { reason: String },
    /// Scheduler is down
    Down { reason: String },
}

/// Scheduling metrics for monitoring
#[derive(Debug, Clone)]
pub struct SchedulingMetrics {
    /// Total tasks scheduled
    pub tasks_scheduled: u64,
    /// Average scheduling time
    pub avg_scheduling_time: Duration,
    /// Current queue length
    pub queue_length: usize,
    /// Scheduling efficiency (0.0-1.0)
    pub efficiency: f64,
    /// Tasks by priority level
    pub tasks_by_priority: HashMap<TaskPriority, u64>,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Throughput (tasks per second)
    pub throughput: f64,
    /// Last updated timestamp
    pub last_updated: SystemTime,
}

impl Default for SchedulingMetrics {
    fn default() -> Self {
        Self {
            tasks_scheduled: 0,
            avg_scheduling_time: Duration::ZERO,
            queue_length: 0,
            efficiency: 1.0,
            tasks_by_priority: HashMap::new(),
            avg_wait_time: Duration::ZERO,
            throughput: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

/// Default task scheduler implementation
pub struct DefaultTaskScheduler {
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Task queue
    queue: VecDeque<(ExecutionTask, TaskHandle)>,
    /// Running tasks
    running: HashMap<String, (ExecutionTask, TaskHandle)>,
    /// Completed tasks
    completed: u64,
    /// Failed tasks
    failed: u64,
    /// Scheduling metrics
    metrics: SchedulingMetrics,
    /// Start time for metrics calculation
    start_time: SystemTime,
}

impl DefaultTaskScheduler {
    /// Create a new default task scheduler
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            running: HashMap::new(),
            completed: 0,
            failed: 0,
            metrics: SchedulingMetrics::default(),
            start_time: SystemTime::now(),
        }
    }

    /// Sort queue based on scheduling algorithm
    fn sort_queue(&mut self) {
        match self.config.algorithm {
            SchedulingAlgorithm::Priority => {
                let mut tasks: Vec<_> = self.queue.drain(..).collect();
                tasks.sort_by(|(_, handle_a), (_, handle_b)| {
                    handle_b.priority.cmp(&handle_a.priority)
                });
                self.queue.extend(tasks);
            }
            SchedulingAlgorithm::ShortestJobFirst => {
                let mut tasks: Vec<_> = self.queue.drain(..).collect();
                tasks.sort_by(|(_, handle_a), (_, handle_b)| {
                    match (handle_a.estimated_duration, handle_b.estimated_duration) {
                        (Some(a), Some(b)) => a.cmp(&b),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                });
                self.queue.extend(tasks);
            }
            SchedulingAlgorithm::EarliestDeadlineFirst => {
                let mut tasks: Vec<_> = self.queue.drain(..).collect();
                tasks.sort_by(|(task_a, _), (task_b, _)| {
                    match (task_a.metadata.deadline, task_b.metadata.deadline) {
                        (Some(a), Some(b)) => a.cmp(&b),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                });
                self.queue.extend(tasks);
            }
            _ => {} // FIFO and others don't require sorting
        }
    }

    /// Check for dependency violations
    fn check_dependencies(&self, task: &ExecutionTask) -> bool {
        if !self.config.dependency_resolution.enable_tracking {
            return true;
        }

        for dependency in &task.metadata.dependencies {
            // Check if dependency is still running or queued
            if self.running.contains_key(dependency) {
                return false;
            }
            if self.queue.iter().any(|(t, _)| t.id == *dependency) {
                return false;
            }
        }
        true
    }

    /// Update metrics
    fn update_metrics(&mut self) {
        self.metrics.queue_length = self.queue.len();
        self.metrics.last_updated = SystemTime::now();

        if let Ok(elapsed) = self.start_time.elapsed() {
            let total_tasks =
                self.completed + self.failed + self.running.len() as u64 + self.queue.len() as u64;
            if elapsed.as_secs() > 0 {
                self.metrics.throughput = total_tasks as f64 / elapsed.as_secs_f64();
            }
        }

        // Calculate efficiency based on completed vs failed tasks
        let total_processed = self.completed + self.failed;
        if total_processed > 0 {
            self.metrics.efficiency = self.completed as f64 / total_processed as f64;
        }
    }
}

impl TaskScheduler for DefaultTaskScheduler {
    fn schedule_task(&mut self, task: ExecutionTask) -> SklResult<TaskHandle> {
        // Check queue capacity
        if self.queue.len() >= self.config.queue_management.max_queue_size {
            match self.config.queue_management.overflow_strategy {
                QueueOverflowStrategy::Block => {
                    return Err(SklearsError::InvalidInput(
                        "Queue is full and blocking new tasks".to_string(),
                    ));
                }
                QueueOverflowStrategy::Reject => {
                    return Err(SklearsError::InvalidInput(
                        "Queue is full, rejecting new task".to_string(),
                    ));
                }
                QueueOverflowStrategy::DropOldest => {
                    self.queue.pop_front();
                }
                QueueOverflowStrategy::DropNewest => {
                    self.queue.pop_back();
                }
                QueueOverflowStrategy::DropLowestPriority => {
                    // Find and remove lowest priority task
                    if let Some(min_idx) = self
                        .queue
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, handle))| &handle.priority)
                        .map(|(idx, _)| idx)
                    {
                        self.queue.remove(min_idx);
                    }
                }
            }
        }

        let handle = TaskHandle {
            task_id: task.id.clone(),
            scheduled_at: SystemTime::now(),
            estimated_duration: task.metadata.estimated_duration,
            priority: task.metadata.priority.clone(),
            dependencies: task.metadata.dependencies.clone(),
        };

        self.queue.push_back((task, handle.clone()));
        self.sort_queue();

        self.metrics.tasks_scheduled += 1;
        *self
            .metrics
            .tasks_by_priority
            .entry(handle.priority.clone())
            .or_insert(0) += 1;

        self.update_metrics();

        Ok(handle)
    }

    fn schedule_batch(&mut self, tasks: Vec<ExecutionTask>) -> SklResult<Vec<TaskHandle>> {
        let mut handles = Vec::new();
        for task in tasks {
            let handle = self.schedule_task(task)?;
            handles.push(handle);
        }
        Ok(handles)
    }

    fn cancel_task(&mut self, handle: TaskHandle) -> SklResult<()> {
        // Remove from queue if present
        self.queue.retain(|(_, h)| h.task_id != handle.task_id);

        // Remove from running if present
        self.running.remove(&handle.task_id);

        self.update_metrics();
        Ok(())
    }

    fn get_status(&self) -> SchedulerStatus {
        SchedulerStatus {
            queued_tasks: self.queue.len(),
            running_tasks: self.running.len(),
            completed_tasks: self.completed,
            failed_tasks: self.failed,
            health: if self.queue.len() > self.config.queue_management.max_queue_size / 2 {
                SchedulerHealth::Overloaded
            } else {
                SchedulerHealth::Healthy
            },
        }
    }

    fn update_config(&mut self, config: SchedulerConfig) -> SklResult<()> {
        self.config = config;
        self.sort_queue(); // Re-sort with new algorithm if changed
        Ok(())
    }

    fn get_next_task(&mut self) -> Option<ExecutionTask> {
        // Find the first task that meets dependency requirements
        let mut task_index = None;
        for (idx, (task, _)) in self.queue.iter().enumerate() {
            if self.check_dependencies(task) {
                task_index = Some(idx);
                break;
            }
        }

        if let Some(idx) = task_index {
            if let Some((task, handle)) = self.queue.remove(idx) {
                let task_id = task.id.clone();
                self.running.insert(task_id.clone(), (task, handle));
                self.update_metrics();
                // Return a new task from the running map or construct a minimal task representation
                return self.running.get(&task_id).map(|(t, _)| {
                    // Create a minimal task copy without the complex fields
                    ExecutionTask {
                        id: t.id.clone(),
                        task_type: t.task_type.clone(),
                        metadata: t.metadata.clone(),
                        requirements: t.requirements.clone(),
                        input_data: None, // Don't copy the input data
                        configuration: t.configuration.clone(),
                    }
                });
            }
        }

        None
    }

    fn mark_completed(&mut self, task_id: &str) -> SklResult<()> {
        if self.running.remove(task_id).is_some() {
            self.completed += 1;
            self.update_metrics();
        }
        Ok(())
    }

    fn mark_failed(&mut self, task_id: &str, _error: String) -> SklResult<()> {
        if self.running.remove(task_id).is_some() {
            self.failed += 1;
            self.update_metrics();
        }
        Ok(())
    }

    fn get_metrics(&self) -> SchedulingMetrics {
        self.metrics.clone()
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            algorithm: SchedulingAlgorithm::Priority,
            queue_management: QueueManagement {
                max_queue_size: 1000,
                overflow_strategy: QueueOverflowStrategy::Block,
                persistence: QueuePersistence::Memory,
            },
            priority_handling: PriorityHandling {
                levels: vec![
                    TaskPriority::Low,
                    TaskPriority::Normal,
                    TaskPriority::High,
                    TaskPriority::Critical,
                ],
                aging_strategy: AgingStrategy::Linear {
                    increment_interval: Duration::from_secs(60),
                },
                starvation_prevention: true,
            },
            dependency_resolution: DependencyResolution {
                enable_tracking: true,
                cycle_detection: true,
                deadlock_prevention: true,
                resolution_timeout: Duration::from_secs(30),
            },
        }
    }
}

/// Priority-based scheduler with advanced features
pub struct PriorityScheduler {
    config: SchedulerConfig,
    queues: HashMap<TaskPriority, VecDeque<(ExecutionTask, TaskHandle)>>,
    running: HashMap<String, (ExecutionTask, TaskHandle)>,
    completed: u64,
    failed: u64,
    metrics: SchedulingMetrics,
    start_time: SystemTime,
}

impl PriorityScheduler {
    /// Create a new priority-based scheduler
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        let mut queues = HashMap::new();
        for priority in &config.priority_handling.levels {
            queues.insert(priority.clone(), VecDeque::new());
        }

        Self {
            config,
            queues,
            running: HashMap::new(),
            completed: 0,
            failed: 0,
            metrics: SchedulingMetrics::default(),
            start_time: SystemTime::now(),
        }
    }
}

impl TaskScheduler for PriorityScheduler {
    fn schedule_task(&mut self, task: ExecutionTask) -> SklResult<TaskHandle> {
        let priority = task.metadata.priority.clone();
        let handle = TaskHandle {
            task_id: task.id.clone(),
            scheduled_at: SystemTime::now(),
            estimated_duration: task.metadata.estimated_duration,
            priority: priority.clone(),
            dependencies: task.metadata.dependencies.clone(),
        };

        if let Some(queue) = self.queues.get_mut(&priority) {
            queue.push_back((task, handle.clone()));
            self.metrics.tasks_scheduled += 1;
            *self.metrics.tasks_by_priority.entry(priority).or_insert(0) += 1;
        }

        Ok(handle)
    }

    fn schedule_batch(&mut self, tasks: Vec<ExecutionTask>) -> SklResult<Vec<TaskHandle>> {
        let mut handles = Vec::new();
        for task in tasks {
            let handle = self.schedule_task(task)?;
            handles.push(handle);
        }
        Ok(handles)
    }

    fn cancel_task(&mut self, handle: TaskHandle) -> SklResult<()> {
        // Remove from appropriate priority queue
        if let Some(queue) = self.queues.get_mut(&handle.priority) {
            queue.retain(|(_, h)| h.task_id != handle.task_id);
        }

        // Remove from running
        self.running.remove(&handle.task_id);

        Ok(())
    }

    fn get_status(&self) -> SchedulerStatus {
        let total_queued: usize = self
            .queues
            .values()
            .map(std::collections::VecDeque::len)
            .sum();

        SchedulerStatus {
            queued_tasks: total_queued,
            running_tasks: self.running.len(),
            completed_tasks: self.completed,
            failed_tasks: self.failed,
            health: if total_queued > self.config.queue_management.max_queue_size / 2 {
                SchedulerHealth::Overloaded
            } else {
                SchedulerHealth::Healthy
            },
        }
    }

    fn update_config(&mut self, config: SchedulerConfig) -> SklResult<()> {
        self.config = config;
        Ok(())
    }

    fn get_next_task(&mut self) -> Option<ExecutionTask> {
        // Get task from highest priority queue first
        for priority in &self.config.priority_handling.levels {
            if let Some(queue) = self.queues.get_mut(priority) {
                if let Some((task, handle)) = queue.pop_front() {
                    let task_id = task.id.clone();
                    let result_task = ExecutionTask {
                        id: task.id.clone(),
                        task_type: task.task_type.clone(),
                        metadata: task.metadata.clone(),
                        requirements: task.requirements.clone(),
                        input_data: None, // Don't copy the input data
                        configuration: task.configuration.clone(),
                    };
                    self.running.insert(task_id, (task, handle));
                    return Some(result_task);
                }
            }
        }
        None
    }

    fn mark_completed(&mut self, task_id: &str) -> SklResult<()> {
        if self.running.remove(task_id).is_some() {
            self.completed += 1;
        }
        Ok(())
    }

    fn mark_failed(&mut self, task_id: &str, _error: String) -> SklResult<()> {
        if self.running.remove(task_id).is_some() {
            self.failed += 1;
        }
        Ok(())
    }

    fn get_metrics(&self) -> SchedulingMetrics {
        self.metrics.clone()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::tasks::*;

    #[test]
    fn test_default_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = DefaultTaskScheduler::new(config);
        let status = scheduler.get_status();

        assert_eq!(status.queued_tasks, 0);
        assert_eq!(status.running_tasks, 0);
        assert_eq!(status.completed_tasks, 0);
        assert_eq!(status.failed_tasks, 0);
        assert_eq!(status.health, SchedulerHealth::Healthy);
    }

    #[test]
    fn test_priority_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);
        let status = scheduler.get_status();

        assert_eq!(status.queued_tasks, 0);
        assert_eq!(status.running_tasks, 0);
        assert_eq!(status.completed_tasks, 0);
        assert_eq!(status.failed_tasks, 0);
        assert_eq!(status.health, SchedulerHealth::Healthy);
    }

    #[test]
    fn test_task_scheduling() {
        let mut scheduler = DefaultTaskScheduler::new(SchedulerConfig::default());
        let task = create_test_task();

        let handle = scheduler.schedule_task(task).unwrap();
        assert!(!handle.task_id.is_empty());

        let status = scheduler.get_status();
        assert_eq!(status.queued_tasks, 1);
    }

    fn create_test_task() -> ExecutionTask {
        ExecutionTask {
            id: "test_task_1".to_string(),
            task_type: TaskType::Computation,
            metadata: TaskMetadata {
                name: "Test Task".to_string(),
                description: "A test task".to_string(),
                priority: TaskPriority::Normal,
                estimated_duration: Some(Duration::from_secs(10)),
                deadline: None,
                dependencies: Vec::new(),
                tags: Vec::new(),
                created_at: SystemTime::now(),
            },
            requirements: ResourceRequirements {
                cpu_cores: 1.0,
                memory_bytes: 1024 * 1024,
                disk_bytes: 0,
                network_bandwidth: 0,
                gpu_memory_bytes: 0,
                special_resources: Vec::new(),
            },
            input_data: None,
            configuration: TaskConfiguration::default(),
        }
    }
}
