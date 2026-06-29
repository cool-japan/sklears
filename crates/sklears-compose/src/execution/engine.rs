//! Core execution engine implementation
//!
//! This module provides the main `ComposableExecutionEngine` and its core functionality.

use sklears_core::error::{Result as SklResult, SklearsError};
use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::config::{ExecutionEngineConfig, ResourceConstraints, StrategyConfig};

/// Composable execution engine that can be configured with different strategies
#[allow(dead_code)]
pub struct ComposableExecutionEngine {
    /// Engine configuration
    config: ExecutionEngineConfig,
    /// Active execution strategies
    strategies: HashMap<String, Box<dyn ExecutionStrategy>>,
    /// Resource manager
    resource_manager: Arc<ResourceManager>,
    /// Task scheduler
    scheduler: Box<dyn TaskScheduler>,
    /// Execution context
    context: ExecutionContext,
    /// Metrics collector
    metrics: Arc<Mutex<ExecutionMetrics>>,
}

/// Execution context for maintaining state during execution
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Context ID
    pub id: String,
    /// Current execution phase
    pub phase: ExecutionPhase,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Start time
    pub start_time: Instant,
}

/// Current execution phase
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionPhase {
    /// Initializing
    Initializing,
    /// Planning
    Planning,
    /// Executing
    Executing,
    /// Monitoring
    Monitoring,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Current resource usage
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Active tasks count
    pub active_tasks: usize,
    /// I/O operations per second
    pub io_ops: u64,
}

/// Execution strategy trait for pluggable execution behaviors
pub trait ExecutionStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;

    /// Execute a task with this strategy
    fn execute(
        &self,
        task: &dyn ExecutableTask,
        context: &ExecutionContext,
    ) -> SklResult<ExecutionResult>;

    /// Check if strategy can handle the given task
    fn can_handle(&self, task: &dyn ExecutableTask) -> bool;

    /// Get strategy configuration
    fn config(&self) -> &StrategyConfig;

    /// Clone the strategy
    fn clone_strategy(&self) -> Box<dyn ExecutionStrategy>;
}

/// Task that can be executed by the engine
pub trait ExecutableTask: Send + Sync {
    /// Task identifier
    fn id(&self) -> &str;

    /// Task type
    fn task_type(&self) -> &str;

    /// Execute the task
    fn execute(&self) -> SklResult<TaskResult>;

    /// Estimate resource requirements
    fn resource_estimate(&self) -> ResourceEstimate;

    /// Get task dependencies
    fn dependencies(&self) -> Vec<String>;
}

/// Result of task execution
#[derive(Debug)]
pub struct TaskResult {
    /// Task ID
    pub task_id: String,
    /// Execution status
    pub status: TaskStatus,
    /// Result data
    pub data: Option<Box<dyn Any + Send>>,
    /// Execution duration
    pub duration: Duration,
    /// Resource usage during execution
    pub resource_usage: ResourceUsage,
}

/// Task execution status
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    /// Pending
    Pending,
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Resource estimation for a task
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Estimated CPU cores needed
    pub cpu_cores: f64,
    /// Estimated memory usage in bytes
    pub memory_bytes: u64,
    /// Estimated execution time
    pub execution_time: Duration,
    /// Estimated I/O operations
    pub io_operations: u64,
}

/// Result of execution strategy
#[derive(Debug)]
pub struct ExecutionResult {
    /// Strategy that executed the task
    pub strategy_name: String,
    /// Task result
    pub task_result: TaskResult,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Task scheduler trait for managing task execution order
pub trait TaskScheduler: Send + Sync {
    /// Add a task to the schedule
    fn schedule_task(&mut self, task: Box<dyn ExecutableTask>) -> SklResult<()>;

    /// Get the next task to execute
    fn next_task(&mut self) -> Option<Box<dyn ExecutableTask>>;

    /// Get current queue size
    fn queue_size(&self) -> usize;

    /// Set scheduler configuration
    fn set_config(&mut self, config: SchedulerConfig);
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    /// Priority weights
    pub priority_weights: HashMap<String, f64>,
    /// Resource-aware scheduling
    pub resource_aware: bool,
}

/// Available scheduling algorithms
#[derive(Debug, Clone)]
pub enum SchedulingAlgorithm {
    /// FIFO
    FIFO,
    /// Priority
    Priority,
    /// ShortestJobFirst
    ShortestJobFirst,
    /// ResourceAware
    ResourceAware,
    /// DeadlineAware
    DeadlineAware,
}

/// Resource manager for controlling resource usage
#[allow(dead_code)]
pub struct ResourceManager {
    /// Current resource allocations
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    /// Resource constraints
    constraints: ResourceConstraints,
    /// Resource monitor
    monitor: Arc<ResourceMonitor>,
}

/// Resource allocation for a specific consumer
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Consumer ID
    pub consumer_id: String,
    /// Allocated CPU cores
    pub cpu_cores: f64,
    /// Allocated memory in bytes
    pub memory_bytes: u64,
    /// Allocation timestamp
    pub allocated_at: Instant,
}

/// Resource monitor for tracking usage
#[allow(dead_code)]
pub struct ResourceMonitor {
    /// Resource usage history
    usage_history: Arc<RwLock<VecDeque<ResourceSnapshot>>>,
    /// Monitoring interval
    interval: Duration,
}

/// Snapshot of resource usage at a point in time
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Timestamp of snapshot
    pub timestamp: Instant,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Number of active tasks
    pub active_tasks: usize,
    /// I/O operations per second
    pub io_rate: f64,
}

/// Metrics collector for execution statistics
#[derive(Debug, Default, Clone)]
pub struct ExecutionMetrics {
    /// Total tasks executed
    pub tasks_executed: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average task duration
    pub average_task_duration: Duration,
    /// Failed tasks count
    pub failed_tasks: u64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization statistics
#[derive(Debug, Default, Clone)]
pub struct ResourceUtilization {
    /// Average CPU usage
    pub avg_cpu_usage: f64,
    /// Peak CPU usage
    pub peak_cpu_usage: f64,
    /// Average memory usage
    pub avg_memory_usage: u64,
    /// Peak memory usage
    pub peak_memory_usage: u64,
}

impl ComposableExecutionEngine {
    /// Create a new execution engine with the given configuration
    pub fn new(config: ExecutionEngineConfig) -> SklResult<Self> {
        let resource_manager = Arc::new(ResourceManager::new(config.resource_constraints.clone()));
        let scheduler = Box::new(DefaultTaskScheduler::new());
        let context = ExecutionContext {
            id: format!("ctx_{}", uuid::Uuid::new_v4()),
            phase: ExecutionPhase::Initializing,
            metadata: HashMap::new(),
            resource_usage: ResourceUsage {
                cpu_usage: 0.0,
                memory_usage: 0,
                active_tasks: 0,
                io_ops: 0,
            },
            start_time: Instant::now(),
        };

        Ok(Self {
            config,
            strategies: HashMap::new(),
            resource_manager,
            scheduler,
            context,
            metrics: Arc::new(Mutex::new(ExecutionMetrics::default())),
        })
    }

    /// Register a new execution strategy
    pub fn register_strategy(&mut self, strategy: Box<dyn ExecutionStrategy>) {
        let name = strategy.name().to_string();
        self.strategies.insert(name, strategy);
    }

    /// Execute a task using the best available strategy
    pub fn execute_task(&mut self, task: Box<dyn ExecutableTask>) -> SklResult<ExecutionResult> {
        // Find the best strategy for this task
        let strategy_name = self.select_strategy(&*task)?;

        if let Some(strategy) = self.strategies.get(&strategy_name) {
            // Update context
            self.context.phase = ExecutionPhase::Executing;

            // Execute the task
            let result = strategy.execute(&*task, &self.context)?;

            // Update metrics
            self.update_metrics(&result);

            // Update context
            self.context.phase = ExecutionPhase::Completed;

            Ok(result)
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Strategy '{strategy_name}' not found"
            )))
        }
    }

    /// Select the best strategy for a task
    fn select_strategy(&self, task: &dyn ExecutableTask) -> SklResult<String> {
        // Find strategies that can handle this task
        let capable_strategies: Vec<_> = self
            .strategies
            .iter()
            .filter(|(_, strategy)| strategy.can_handle(task))
            .collect();

        if capable_strategies.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No capable strategy found for task".to_string(),
            ));
        }

        // Select the strategy with the highest configured priority. When priorities
        // are equal, prefer the strategy whose estimated resource cost best fits the
        // task's declared resource estimate (lower estimated memory wins as a
        // tie-break so that lighter strategies are preferred for small tasks).
        let estimate = task.resource_estimate();
        let best = capable_strategies
            .into_iter()
            .max_by(|(_, a), (_, b)| {
                let a_priority = a.config().resource_allocation.priority;
                let b_priority = b.config().resource_allocation.priority;
                match a_priority.cmp(&b_priority) {
                    std::cmp::Ordering::Equal => {
                        // Tie-break: prefer the strategy whose memory allocation is
                        // closest to (but not less than) the task's own estimate.
                        let a_fit = a
                            .config()
                            .resource_allocation
                            .memory_bytes
                            .saturating_sub(estimate.memory_bytes);
                        let b_fit = b
                            .config()
                            .resource_allocation
                            .memory_bytes
                            .saturating_sub(estimate.memory_bytes);
                        // Smaller overshoot is better; reverse ordering picks minimum
                        b_fit.cmp(&a_fit)
                    }
                    other => other,
                }
            })
            .ok_or_else(|| {
                SklearsError::InvalidInput("No capable strategy found for task".to_string())
            })?;
        Ok(best.0.clone())
    }

    /// Update execution metrics
    fn update_metrics(&self, result: &ExecutionResult) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.tasks_executed += 1;
            metrics.total_execution_time += result.task_result.duration;

            if result.task_result.status == TaskStatus::Failed {
                metrics.failed_tasks += 1;
            }

            // Update average
            metrics.average_task_duration = Duration::from_nanos(
                metrics.total_execution_time.as_nanos() as u64 / metrics.tasks_executed,
            );
        }
    }

    /// Get current execution metrics
    #[must_use]
    pub fn metrics(&self) -> ExecutionMetrics {
        self.metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get engine configuration
    #[must_use]
    pub fn config(&self) -> &ExecutionEngineConfig {
        &self.config
    }
}

impl ResourceManager {
    /// Create a new resource manager
    #[must_use]
    pub fn new(constraints: ResourceConstraints) -> Self {
        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            constraints,
            monitor: Arc::new(ResourceMonitor::new(Duration::from_secs(1))),
        }
    }

    /// Allocate resources for a consumer
    pub fn allocate_resources(
        &self,
        consumer_id: String,
        request: ResourceRequest,
    ) -> SklResult<ResourceAllocation> {
        let allocation = ResourceAllocation {
            consumer_id: consumer_id.clone(),
            cpu_cores: request.cpu_cores,
            memory_bytes: request.memory_bytes,
            allocated_at: Instant::now(),
        };

        if let Ok(mut allocations) = self.allocations.write() {
            allocations.insert(consumer_id, allocation.clone());
        }

        Ok(allocation)
    }
}

/// Resource request for allocation
#[derive(Debug, Clone)]
pub struct ResourceRequest {
    /// Requested CPU cores
    pub cpu_cores: f64,
    /// Requested memory in bytes
    pub memory_bytes: u64,
    /// Request priority
    pub priority: u32,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        Self {
            usage_history: Arc::new(RwLock::new(VecDeque::new())),
            interval,
        }
    }
}

/// Default task scheduler implementation
pub struct DefaultTaskScheduler {
    /// Task queue
    queue: VecDeque<Box<dyn ExecutableTask>>,
    /// Scheduler configuration
    config: SchedulerConfig,
}

impl Default for DefaultTaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultTaskScheduler {
    /// Create a new default scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            config: SchedulerConfig {
                algorithm: SchedulingAlgorithm::FIFO,
                priority_weights: HashMap::new(),
                resource_aware: false,
            },
        }
    }
}

impl TaskScheduler for DefaultTaskScheduler {
    fn schedule_task(&mut self, task: Box<dyn ExecutableTask>) -> SklResult<()> {
        self.queue.push_back(task);
        Ok(())
    }

    fn next_task(&mut self) -> Option<Box<dyn ExecutableTask>> {
        self.queue.pop_front()
    }

    fn queue_size(&self) -> usize {
        self.queue.len()
    }

    fn set_config(&mut self, config: SchedulerConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::config::{
        CachingStrategy, ExecutionEngineConfig, LoadBalancingAlgorithm, LoadBalancingConfig,
        OptimizationLevel, PerformanceTuning, PrefetchingStrategy, StrategyConfig,
        StrategyResourceAllocation,
    };
    use std::collections::HashMap;
    use std::time::Duration;

    /// Minimal test task implementation.
    struct TestTask {
        id: String,
        memory_estimate: u64,
    }

    impl ExecutableTask for TestTask {
        fn id(&self) -> &str {
            &self.id
        }
        fn task_type(&self) -> &str {
            "test"
        }
        fn execute(&self) -> SklResult<TaskResult> {
            Ok(TaskResult {
                task_id: self.id.clone(),
                status: TaskStatus::Completed,
                data: None,
                duration: Duration::from_millis(1),
                resource_usage: ResourceUsage {
                    cpu_usage: 0.0,
                    memory_usage: 0,
                    active_tasks: 0,
                    io_ops: 0,
                },
            })
        }
        fn resource_estimate(&self) -> ResourceEstimate {
            ResourceEstimate {
                cpu_cores: 1.0,
                memory_bytes: self.memory_estimate,
                execution_time: Duration::from_millis(1),
                io_operations: 0,
            }
        }
        fn dependencies(&self) -> Vec<String> {
            vec![]
        }
    }

    /// Minimal test strategy that can handle any task.
    struct TestStrategy {
        name: String,
        priority: u32,
        memory_alloc: u64,
        cfg: StrategyConfig,
    }

    impl TestStrategy {
        fn new(name: &str, priority: u32, memory_alloc: u64) -> Self {
            Self {
                name: name.to_string(),
                priority,
                memory_alloc,
                cfg: StrategyConfig {
                    name: name.to_string(),
                    parameters: HashMap::new(),
                    resource_allocation: StrategyResourceAllocation {
                        cpu_cores: 1.0,
                        memory_bytes: memory_alloc,
                        priority,
                    },
                    performance_tuning: PerformanceTuning {
                        optimization_level: OptimizationLevel::None,
                        prefetching: PrefetchingStrategy::None,
                        caching: CachingStrategy::None,
                        load_balancing: LoadBalancingConfig {
                            enabled: false,
                            algorithm: LoadBalancingAlgorithm::RoundRobin,
                            rebalance_threshold: 0.2,
                            min_load_difference: 0.1,
                        },
                    },
                },
            }
        }
    }

    impl ExecutionStrategy for TestStrategy {
        fn name(&self) -> &str {
            &self.name
        }
        fn execute(
            &self,
            task: &dyn ExecutableTask,
            _ctx: &ExecutionContext,
        ) -> SklResult<ExecutionResult> {
            Ok(ExecutionResult {
                strategy_name: self.name.clone(),
                task_result: task.execute()?,
                metadata: HashMap::new(),
            })
        }
        fn can_handle(&self, _task: &dyn ExecutableTask) -> bool {
            true
        }
        fn config(&self) -> &StrategyConfig {
            &self.cfg
        }
        fn clone_strategy(&self) -> Box<dyn ExecutionStrategy> {
            Box::new(TestStrategy::new(
                &self.name,
                self.priority,
                self.memory_alloc,
            ))
        }
    }

    fn make_engine() -> ComposableExecutionEngine {
        ComposableExecutionEngine::new(ExecutionEngineConfig::default())
            .expect("engine should build")
    }

    #[test]
    fn test_select_strategy_prefers_higher_priority() {
        let mut engine = make_engine();
        engine.register_strategy(Box::new(TestStrategy::new("low", 1, 1024)));
        engine.register_strategy(Box::new(TestStrategy::new("high", 10, 2048)));

        let task = TestTask {
            id: "t1".to_string(),
            memory_estimate: 512,
        };
        let result = engine
            .execute_task(Box::new(task))
            .expect("execute should succeed");
        assert_eq!(
            result.strategy_name, "high",
            "Higher-priority strategy should be selected"
        );
    }

    #[test]
    fn test_select_strategy_tiebreak_by_memory_fit() {
        let mut engine = make_engine();
        // Both strategies have the same priority; prefer the one whose memory
        // allocation is closer to (but >= than) the task's estimate.
        engine.register_strategy(Box::new(TestStrategy::new("tight", 5, 600))); // overshoot: 88
        engine.register_strategy(Box::new(TestStrategy::new("loose", 5, 4096))); // overshoot: 3584

        let task = TestTask {
            id: "t2".to_string(),
            memory_estimate: 512,
        };
        let result = engine
            .execute_task(Box::new(task))
            .expect("execute should succeed");
        assert_eq!(
            result.strategy_name, "tight",
            "Strategy with smallest overshoot should win tie"
        );
    }

    #[test]
    fn test_no_capable_strategy_returns_error() {
        let mut engine = make_engine();
        // No strategies registered.
        let task = TestTask {
            id: "t3".to_string(),
            memory_estimate: 0,
        };
        let err = engine.execute_task(Box::new(task));
        assert!(
            err.is_err(),
            "Should return error when no strategies are registered"
        );
    }
}
