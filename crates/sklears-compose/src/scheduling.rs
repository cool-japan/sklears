//! Pipeline scheduling and execution utilities
//!
//! This module provides advanced scheduling capabilities including task dependency management,
//! resource allocation optimization, priority-based execution, and workflow monitoring.

use sklears_core::error::{Result as SklResult, SklearsError};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

use crate::distributed::{NodeId, ResourceRequirements, TaskId, TaskPriority};

/// Schedulable task representation
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Task identifier
    pub id: TaskId,
    /// Task name
    pub name: String,
    /// Pipeline component to execute
    pub component_type: ComponentType,
    /// Task dependencies
    pub dependencies: Vec<TaskId>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Task priority
    pub priority: TaskPriority,
    /// Estimated execution time
    pub estimated_duration: Duration,
    /// Submission time
    pub submitted_at: SystemTime,
    /// Deadline (optional)
    pub deadline: Option<SystemTime>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Component type for scheduling
#[derive(Debug, Clone)]
pub enum ComponentType {
    /// Transformer
    Transformer,
    /// Predictor
    Predictor,
    /// DataProcessor
    DataProcessor,
    /// CustomFunction
    CustomFunction,
}

/// Retry configuration for tasks
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Retry delay strategy
    pub delay_strategy: RetryDelayStrategy,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay between retries
    pub max_delay: Duration,
}

/// Retry delay strategies
#[derive(Debug, Clone)]
pub enum RetryDelayStrategy {
    /// Fixed delay between retries
    Fixed(Duration),
    /// Linear increase in delay
    Linear(Duration),
    /// Exponential backoff
    Exponential(Duration),
    /// Custom delay calculation
    Custom(fn(usize) -> Duration),
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            delay_strategy: RetryDelayStrategy::Exponential(Duration::from_millis(100)),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(60),
        }
    }
}

/// Task execution state
#[derive(Debug, Clone, PartialEq)]
pub enum TaskState {
    /// Task is waiting to be scheduled
    Pending,
    /// Task is ready to execute (dependencies satisfied)
    Ready,
    /// Task is currently running
    Running {
        started_at: SystemTime,
        node_id: Option<NodeId>,
    },
    /// Task completed successfully
    Completed {
        completed_at: SystemTime,
        execution_time: Duration,
    },
    /// Task failed
    Failed {
        failed_at: SystemTime,
        error: String,
        retry_count: usize,
    },
    /// Task was cancelled
    Cancelled { cancelled_at: SystemTime },
    /// Task is waiting for retry
    Retrying {
        next_retry_at: SystemTime,
        retry_count: usize,
    },
}

/// Task wrapper for priority queue
#[derive(Debug)]
struct PriorityTask {
    task: ScheduledTask,
    priority_score: i64,
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority score comes first (max-heap will pop highest first)
        self.priority_score.cmp(&other.priority_score)
    }
}

/// Scheduling strategy
#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    /// First-In-First-Out
    FIFO,
    /// Priority-based scheduling
    Priority,
    /// Shortest Job First
    ShortestJobFirst,
    /// Earliest Deadline First
    EarliestDeadlineFirst,
    /// Fair share scheduling
    FairShare {
        /// Time quantum for each task
        time_quantum: Duration,
    },
    /// Resource-aware scheduling
    ResourceAware,
    /// Custom scheduling function
    Custom {
        schedule_fn: fn(&[ScheduledTask], &ResourcePool) -> Option<TaskId>,
    },
}

/// Resource pool for scheduling
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// Available CPU cores
    pub available_cpu: u32,
    /// Available memory in MB
    pub available_memory: u64,
    /// Available disk space in MB
    pub available_disk: u64,
    /// Available GPU count
    pub available_gpu: u32,
    /// Resource utilization history
    pub utilization_history: Vec<ResourceUtilization>,
}

/// Resource utilization snapshot
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Timestamp
    pub timestamp: SystemTime,
    /// CPU utilization (0.0 - 1.0)
    pub cpu_usage: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_usage: f64,
    /// Disk utilization (0.0 - 1.0)
    pub disk_usage: f64,
    /// GPU utilization (0.0 - 1.0)
    pub gpu_usage: f64,
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self {
            available_cpu: 4,
            available_memory: 8192,
            available_disk: 100_000,
            available_gpu: 0,
            utilization_history: Vec::new(),
        }
    }
}

/// Pluggable scheduler trait for extensible scheduling strategies
pub trait PluggableScheduler: Send + Sync + std::fmt::Debug {
    /// Scheduler name/identifier
    fn name(&self) -> &str;

    /// Scheduler description
    fn description(&self) -> &str;

    /// Initialize the scheduler with configuration
    fn initialize(&mut self, config: &SchedulerConfig) -> SklResult<()>;

    /// Select the next task to execute from available tasks
    fn select_next_task(
        &self,
        available_tasks: &[ScheduledTask],
        resource_pool: &ResourcePool,
        current_time: SystemTime,
    ) -> Option<TaskId>;

    /// Calculate priority score for a task
    fn calculate_priority(&self, task: &ScheduledTask, context: &SchedulingContext) -> i64;

    /// Check if a task can be scheduled given current resources
    fn can_schedule_task(&self, task: &ScheduledTask, resource_pool: &ResourcePool) -> bool;

    /// Get scheduler-specific metrics
    fn get_metrics(&self) -> SchedulerMetrics;

    /// Handle task completion events
    fn on_task_completed(&mut self, task_id: &TaskId, execution_time: Duration) -> SklResult<()>;

    /// Handle task failure events
    fn on_task_failed(&mut self, task_id: &TaskId, error: &str) -> SklResult<()>;

    /// Cleanup scheduler resources
    fn cleanup(&mut self) -> SklResult<()>;
}

/// Scheduling context for decision making
#[derive(Debug, Clone, Default)]
pub struct SchedulingContext {
    /// Current system load
    pub system_load: SystemLoad,
    /// Historical task execution data
    pub execution_history: Vec<TaskExecutionHistory>,
    /// Current resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Time-based context
    pub temporal_context: TemporalContext,
    /// Custom context data
    pub custom_data: HashMap<String, String>,
}

/// System load information
#[derive(Debug, Clone)]
pub struct SystemLoad {
    /// Overall CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_utilization: f64,
    /// I/O wait percentage
    pub io_wait: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Load average (1, 5, 15 minutes)
    pub load_average: (f64, f64, f64),
}

impl Default for SystemLoad {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            io_wait: 0.0,
            network_utilization: 0.0,
            load_average: (0.0, 0.0, 0.0),
        }
    }
}

/// Task execution history
#[derive(Debug, Clone)]
pub struct TaskExecutionHistory {
    /// Task type/component
    pub task_type: ComponentType,
    /// Actual execution time
    pub execution_time: Duration,
    /// Resource usage during execution
    pub resource_usage: ResourceUsage,
    /// Success rate
    pub success_rate: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Resource usage during task execution
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Peak CPU usage
    pub peak_cpu: f64,
    /// Peak memory usage (MB)
    pub peak_memory: u64,
    /// I/O operations count
    pub io_operations: u64,
    /// Network bytes transferred
    pub network_bytes: u64,
}

/// Resource constraints for scheduling
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum CPU allocation per task
    pub max_cpu_per_task: f64,
    /// Maximum memory allocation per task (MB)
    pub max_memory_per_task: u64,
    /// Maximum concurrent I/O operations
    pub max_concurrent_io: u32,
    /// Network bandwidth limit (bytes/sec)
    pub network_bandwidth_limit: u64,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_per_task: 1.0,                // 1 CPU core per task
            max_memory_per_task: 1024,            // 1GB memory per task
            max_concurrent_io: 10,                // 10 concurrent I/O operations
            network_bandwidth_limit: 100_000_000, // 100 MB/s network bandwidth
        }
    }
}

/// Temporal context for time-aware scheduling
#[derive(Debug, Clone)]
pub struct TemporalContext {
    /// Current time
    pub current_time: SystemTime,
    /// Business hours definition
    pub business_hours: Option<BusinessHours>,
    /// Maintenance windows
    pub maintenance_windows: Vec<MaintenanceWindow>,
    /// Peak usage periods
    pub peak_periods: Vec<PeakPeriod>,
}

impl Default for TemporalContext {
    fn default() -> Self {
        Self {
            current_time: SystemTime::now(),
            business_hours: None,
            maintenance_windows: Vec::new(),
            peak_periods: Vec::new(),
        }
    }
}

/// Business hours definition
#[derive(Debug, Clone)]
pub struct BusinessHours {
    /// Start time (hour, minute)
    pub start: (u8, u8),
    /// End time (hour, minute)
    pub end: (u8, u8),
    /// Business days (0 = Sunday, 6 = Saturday)
    pub business_days: Vec<u8>,
    /// Timezone offset
    pub timezone_offset: i8,
}

/// Maintenance window definition
#[derive(Debug, Clone)]
pub struct MaintenanceWindow {
    /// Window name
    pub name: String,
    /// Start time
    pub start: SystemTime,
    /// End time
    pub end: SystemTime,
    /// Severity (affects scheduling decisions)
    pub severity: MaintenanceSeverity,
}

/// Maintenance severity levels
#[derive(Debug, Clone)]
pub enum MaintenanceSeverity {
    /// Normal maintenance - reduce scheduling
    Normal,
    /// Critical maintenance - halt non-essential scheduling
    Critical,
    /// Emergency maintenance - halt all scheduling
    Emergency,
}

/// Peak usage period
#[derive(Debug, Clone)]
pub struct PeakPeriod {
    /// Period name
    pub name: String,
    /// Start time (hour, minute)
    pub start: (u8, u8),
    /// End time (hour, minute)
    pub end: (u8, u8),
    /// Peak factor (multiplier for resource costs)
    pub peak_factor: f64,
}

/// Scheduler-specific metrics
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    /// Total tasks scheduled
    pub tasks_scheduled: u64,
    /// Average scheduling latency
    pub avg_scheduling_latency: Duration,
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
    /// Deadline miss rate
    pub deadline_miss_rate: f64,
    /// Fairness index (0.0 - 1.0)
    pub fairness_index: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Advanced scheduling strategies
#[derive(Debug, Clone)]
pub enum AdvancedSchedulingStrategy {
    /// Machine learning-based adaptive scheduling
    MLAdaptive {
        model_path: String,
        feature_extractors: Vec<String>,
    },
    /// Genetic algorithm optimization
    GeneticOptimization {
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
    },
    /// Multi-objective optimization (Pareto-optimal)
    MultiObjective {
        objectives: Vec<SchedulingObjective>,
        weights: Vec<f64>,
    },
    /// Reinforcement learning scheduler
    ReinforcementLearning {
        agent_type: String,
        learning_rate: f64,
        exploration_rate: f64,
    },
    /// Game theory-based scheduling
    GameTheory {
        strategy_type: GameTheoryStrategy,
        coalition_formation: bool,
    },
    /// Quantum-inspired optimization
    QuantumInspired {
        quantum_operators: Vec<String>,
        entanglement_depth: usize,
    },
}

/// Scheduling objectives for multi-objective optimization
#[derive(Debug, Clone)]
pub enum SchedulingObjective {
    /// Minimize makespan (total completion time)
    MinimizeMakespan,
    /// Minimize resource usage
    MinimizeResourceUsage,
    /// Maximize throughput
    MaximizeThroughput,
    /// Minimize energy consumption
    MinimizeEnergy,
    /// Maximize fairness
    MaximizeFairness,
    /// Minimize deadline violations
    MinimizeDeadlineViolations,
    /// Custom objective function
    Custom {
        name: String,
        objective_fn: fn(&[ScheduledTask], &ResourcePool) -> f64,
    },
}

/// Game theory strategies
#[derive(Debug, Clone)]
pub enum GameTheoryStrategy {
    /// Nash equilibrium-based
    NashEquilibrium,
    /// Stackelberg game
    Stackelberg,
    /// Cooperative game
    Cooperative,
    /// Auction-based mechanism
    Auction,
}

/// Multi-level feedback scheduler
pub struct MultiLevelFeedbackScheduler {
    name: String,
    queues: Vec<PriorityQueue>,
    time_quantum: Vec<Duration>,
    promotion_threshold: Vec<u32>,
    demotion_threshold: Vec<u32>,
    aging_factor: f64,
    metrics: SchedulerMetrics,
}

/// Priority queue for multi-level scheduler
#[derive(Debug)]
struct PriorityQueue {
    tasks: VecDeque<ScheduledTask>,
    priority_level: u8,
    time_slice: Duration,
}

/// Fair share scheduler with proportional allocation
pub struct FairShareScheduler {
    name: String,
    user_shares: HashMap<String, f64>,
    group_shares: HashMap<String, f64>,
    usage_history: HashMap<String, Vec<ResourceUsage>>,
    decay_factor: f64,
    metrics: SchedulerMetrics,
}

/// Deadline-aware earliest deadline first scheduler
pub struct DeadlineAwareScheduler {
    name: String,
    deadline_weight: f64,
    urgency_factor: f64,
    preemption_enabled: bool,
    grace_period: Duration,
    metrics: SchedulerMetrics,
}

/// Resource-aware scheduler with load balancing
pub struct ResourceAwareScheduler {
    name: String,
    resource_weights: HashMap<String, f64>,
    load_balancing_strategy: LoadBalancingStrategy,
    prediction_window: Duration,
    efficiency_threshold: f64,
    metrics: SchedulerMetrics,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round-robin allocation
    RoundRobin,
    /// Least loaded first
    LeastLoaded,
    /// Weighted round-robin
    WeightedRoundRobin { weights: HashMap<String, f64> },
    /// Random allocation
    Random,
    /// Consistent hashing
    ConsistentHashing { virtual_nodes: usize },
}

/// Machine learning adaptive scheduler
pub struct MLAdaptiveScheduler {
    name: String,
    model_type: MLModelType,
    feature_extractors: Vec<Box<dyn FeatureExtractor>>,
    training_data: Vec<SchedulingDecision>,
    prediction_accuracy: f64,
    retraining_threshold: usize,
    metrics: SchedulerMetrics,
}

/// ML model types for adaptive scheduling
#[derive(Debug, Clone)]
pub enum MLModelType {
    /// Decision tree
    DecisionTree,
    /// Random forest
    RandomForest { n_trees: usize },
    /// Neural network
    NeuralNetwork { layers: Vec<usize> },
    /// Support vector machine
    SVM { kernel: String },
    /// Reinforcement learning
    ReinforcementLearning { algorithm: String },
}

/// Feature extractor trait for ML scheduler
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from scheduling context
    fn extract_features(&self, context: &SchedulingContext) -> Vec<f64>;

    /// Get feature names
    fn feature_names(&self) -> Vec<String>;
}

/// Scheduling decision for ML training
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Input features
    pub features: Vec<f64>,
    /// Chosen task ID
    pub chosen_task: TaskId,
    /// Outcome metrics
    pub outcome: DecisionOutcome,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Outcome of a scheduling decision
#[derive(Debug, Clone)]
pub struct DecisionOutcome {
    /// Task completion time
    pub completion_time: Duration,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Deadline satisfaction
    pub deadline_met: bool,
    /// Overall satisfaction score
    pub satisfaction_score: f64,
}

/// Task scheduler implementation with pluggable strategies
#[derive(Debug)]
pub struct TaskScheduler {
    /// Primary scheduling strategy
    strategy: SchedulingStrategy,
    /// Pluggable scheduler instances
    pluggable_schedulers: HashMap<String, Box<dyn PluggableScheduler>>,
    /// Active scheduler name
    active_scheduler: Option<String>,
    /// Task queue (pending tasks)
    task_queue: Arc<Mutex<BinaryHeap<PriorityTask>>>,
    /// Task states
    task_states: Arc<RwLock<HashMap<TaskId, TaskState>>>,
    /// Resource pool
    resource_pool: Arc<RwLock<ResourcePool>>,
    /// Dependency graph
    dependency_graph: Arc<RwLock<HashMap<TaskId, HashSet<TaskId>>>>,
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Scheduling context
    context: Arc<RwLock<SchedulingContext>>,
    /// Condition variable for task notifications
    task_notification: Arc<Condvar>,
    /// Scheduler thread handle
    scheduler_thread: Option<JoinHandle<()>>,
    /// Running flag
    is_running: Arc<Mutex<bool>>,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Scheduling interval
    pub scheduling_interval: Duration,
    /// Resource monitoring interval
    pub monitoring_interval: Duration,
    /// Task timeout
    pub default_task_timeout: Duration,
    /// Dead task cleanup interval
    pub cleanup_interval: Duration,
    /// Maximum task history to keep
    pub max_task_history: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            scheduling_interval: Duration::from_millis(100),
            monitoring_interval: Duration::from_secs(1),
            default_task_timeout: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(300),
            max_task_history: 10000,
        }
    }
}

impl TaskScheduler {
    /// Create a new task scheduler
    #[must_use]
    pub fn new(strategy: SchedulingStrategy, config: SchedulerConfig) -> Self {
        Self {
            strategy,
            pluggable_schedulers: HashMap::new(),
            active_scheduler: None,
            task_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            task_states: Arc::new(RwLock::new(HashMap::new())),
            resource_pool: Arc::new(RwLock::new(ResourcePool::default())),
            dependency_graph: Arc::new(RwLock::new(HashMap::new())),
            config,
            context: Arc::new(RwLock::new(SchedulingContext::default())),
            task_notification: Arc::new(Condvar::new()),
            scheduler_thread: None,
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Submit a task for scheduling
    pub fn submit_task(&self, task: ScheduledTask) -> SklResult<()> {
        let task_id = task.id.clone();

        // Add to dependency graph
        {
            let mut graph = self.dependency_graph.write().unwrap();
            graph.insert(task_id.clone(), task.dependencies.iter().cloned().collect());
        }

        // Set initial state
        {
            let mut states = self.task_states.write().unwrap();
            states.insert(task_id, TaskState::Pending);
        }

        // Calculate priority score
        let priority_score = self.calculate_priority_score(&task);

        // Add to queue
        {
            let mut queue = self.task_queue.lock().unwrap();
            queue.push(PriorityTask {
                task,
                priority_score,
            });
        }

        // Notify scheduler
        self.task_notification.notify_one();

        Ok(())
    }

    /// Calculate priority score for a task
    fn calculate_priority_score(&self, task: &ScheduledTask) -> i64 {
        let mut score = match task.priority {
            TaskPriority::Low => 1,
            TaskPriority::Normal => 10,
            TaskPriority::High => 100,
            TaskPriority::Critical => 1000,
        };

        // Adjust for deadline
        if let Some(deadline) = task.deadline {
            let time_to_deadline = deadline
                .duration_since(SystemTime::now())
                .unwrap_or(Duration::ZERO)
                .as_secs() as i64;
            score += 1_000_000 / (time_to_deadline + 1); // Higher score for urgent tasks
        }

        // Adjust for submission time (older tasks get higher priority)
        let age = SystemTime::now()
            .duration_since(task.submitted_at)
            .unwrap_or(Duration::ZERO)
            .as_secs() as i64;
        score += age / 60; // Small bonus for waiting time

        score
    }

    /// Start the scheduler
    pub fn start(&mut self) -> SklResult<()> {
        {
            let mut running = self.is_running.lock().unwrap();
            *running = true;
        }

        let task_queue = Arc::clone(&self.task_queue);
        let task_states = Arc::clone(&self.task_states);
        let resource_pool = Arc::clone(&self.resource_pool);
        let dependency_graph = Arc::clone(&self.dependency_graph);
        let task_notification = Arc::clone(&self.task_notification);
        let is_running = Arc::clone(&self.is_running);
        let config = self.config.clone();
        let strategy = self.strategy.clone();

        let handle = thread::spawn(move || {
            Self::scheduler_loop(
                task_queue,
                task_states,
                resource_pool,
                dependency_graph,
                task_notification,
                is_running,
                config,
                strategy,
            );
        });

        self.scheduler_thread = Some(handle);
        Ok(())
    }

    /// Stop the scheduler
    pub fn stop(&mut self) -> SklResult<()> {
        {
            let mut running = self.is_running.lock().unwrap();
            *running = false;
        }

        self.task_notification.notify_all();

        if let Some(handle) = self.scheduler_thread.take() {
            handle.join().map_err(|_| SklearsError::InvalidData {
                reason: "Failed to join scheduler thread".to_string(),
            })?;
        }

        Ok(())
    }

    /// Main scheduler loop
    fn scheduler_loop(
        task_queue: Arc<Mutex<BinaryHeap<PriorityTask>>>,
        task_states: Arc<RwLock<HashMap<TaskId, TaskState>>>,
        resource_pool: Arc<RwLock<ResourcePool>>,
        dependency_graph: Arc<RwLock<HashMap<TaskId, HashSet<TaskId>>>>,
        task_notification: Arc<Condvar>,
        is_running: Arc<Mutex<bool>>,
        config: SchedulerConfig,
        strategy: SchedulingStrategy,
    ) {
        let mut lock = task_queue.lock().unwrap();

        while *is_running.lock().unwrap() {
            // Check for ready tasks
            let ready_tasks = Self::find_ready_tasks(&task_queue, &task_states, &dependency_graph);

            // Schedule ready tasks
            for task_id in ready_tasks {
                if Self::count_running_tasks(&task_states) >= config.max_concurrent_tasks {
                    break;
                }

                if Self::can_allocate_resources(&task_id, &task_states, &resource_pool) {
                    Self::start_task_execution(&task_id, &task_states, &resource_pool);
                }
            }

            // Clean up completed/failed tasks
            Self::cleanup_tasks(&task_states, &config);

            // Update resource monitoring
            Self::update_resource_monitoring(&resource_pool);

            // Wait for notification or timeout
            let _guard = task_notification
                .wait_timeout(lock, config.scheduling_interval)
                .unwrap();
            lock = _guard.0;
        }
    }

    /// Find tasks that are ready to execute
    fn find_ready_tasks(
        task_queue: &Arc<Mutex<BinaryHeap<PriorityTask>>>,
        task_states: &Arc<RwLock<HashMap<TaskId, TaskState>>>,
        dependency_graph: &Arc<RwLock<HashMap<TaskId, HashSet<TaskId>>>>,
    ) -> Vec<TaskId> {
        let mut ready_tasks = Vec::new();
        let states = task_states.read().unwrap();
        let graph = dependency_graph.read().unwrap();

        for (task_id, state) in states.iter() {
            if *state == TaskState::Pending {
                if let Some(dependencies) = graph.get(task_id) {
                    let all_deps_completed = dependencies.iter().all(|dep_id| {
                        if let Some(dep_state) = states.get(dep_id) {
                            matches!(dep_state, TaskState::Completed { .. })
                        } else {
                            false
                        }
                    });

                    if all_deps_completed {
                        ready_tasks.push(task_id.clone());
                    }
                }
            }
        }

        ready_tasks
    }

    /// Count currently running tasks
    fn count_running_tasks(task_states: &Arc<RwLock<HashMap<TaskId, TaskState>>>) -> usize {
        let states = task_states.read().unwrap();
        states
            .values()
            .filter(|state| matches!(state, TaskState::Running { .. }))
            .count()
    }

    /// Check if resources can be allocated for a task
    fn can_allocate_resources(
        task_id: &TaskId,
        task_states: &Arc<RwLock<HashMap<TaskId, TaskState>>>,
        resource_pool: &Arc<RwLock<ResourcePool>>,
    ) -> bool {
        // Simplified resource check
        let pool = resource_pool.read().unwrap();
        pool.available_cpu > 0 && pool.available_memory > 100
    }

    /// Start task execution
    fn start_task_execution(
        task_id: &TaskId,
        task_states: &Arc<RwLock<HashMap<TaskId, TaskState>>>,
        resource_pool: &Arc<RwLock<ResourcePool>>,
    ) {
        let mut states = task_states.write().unwrap();
        states.insert(
            task_id.clone(),
            TaskState::Running {
                started_at: SystemTime::now(),
                node_id: Some("local".to_string()),
            },
        );

        // Allocate resources (simplified)
        let mut pool = resource_pool.write().unwrap();
        pool.available_cpu = pool.available_cpu.saturating_sub(1);
        pool.available_memory = pool.available_memory.saturating_sub(100);
    }

    /// Clean up completed/failed tasks
    fn cleanup_tasks(
        task_states: &Arc<RwLock<HashMap<TaskId, TaskState>>>,
        config: &SchedulerConfig,
    ) {
        let mut states = task_states.write().unwrap();

        let cutoff_time = SystemTime::now() - config.cleanup_interval;
        let mut to_remove = Vec::new();

        for (task_id, state) in states.iter() {
            let should_remove = match state {
                TaskState::Completed { completed_at, .. } => *completed_at < cutoff_time,
                TaskState::Failed { failed_at, .. } => *failed_at < cutoff_time,
                TaskState::Cancelled { cancelled_at } => *cancelled_at < cutoff_time,
                _ => false,
            };

            if should_remove {
                to_remove.push(task_id.clone());
            }
        }

        // Keep only recent tasks
        if states.len() > config.max_task_history {
            let excess = states.len() - config.max_task_history;
            for _ in 0..excess {
                if let Some(oldest_id) = to_remove.first().cloned() {
                    to_remove.remove(0);
                    states.remove(&oldest_id);
                }
            }
        }

        for task_id in to_remove {
            states.remove(&task_id);
        }
    }

    /// Update resource monitoring
    fn update_resource_monitoring(resource_pool: &Arc<RwLock<ResourcePool>>) {
        let mut pool = resource_pool.write().unwrap();

        let utilization = ResourceUtilization {
            timestamp: SystemTime::now(),
            cpu_usage: 1.0 - (f64::from(pool.available_cpu) / 4.0), // Assuming 4 core system
            memory_usage: 1.0 - (pool.available_memory as f64 / 8192.0), // Assuming 8GB system
            disk_usage: 0.5,                                        // Simplified
            gpu_usage: 0.0,
        };

        pool.utilization_history.push(utilization);

        // Keep only recent history
        if pool.utilization_history.len() > 100 {
            pool.utilization_history.remove(0);
        }
    }

    /// Get task state
    #[must_use]
    pub fn get_task_state(&self, task_id: &TaskId) -> Option<TaskState> {
        let states = self.task_states.read().unwrap();
        states.get(task_id).cloned()
    }

    /// Get scheduler statistics
    #[must_use]
    pub fn get_statistics(&self) -> SchedulerStatistics {
        let states = self.task_states.read().unwrap();
        let queue = self.task_queue.lock().unwrap();
        let pool = self.resource_pool.read().unwrap();

        let pending_count = states
            .values()
            .filter(|s| matches!(s, TaskState::Pending))
            .count();
        let running_count = states
            .values()
            .filter(|s| matches!(s, TaskState::Running { .. }))
            .count();
        let completed_count = states
            .values()
            .filter(|s| matches!(s, TaskState::Completed { .. }))
            .count();
        let failed_count = states
            .values()
            .filter(|s| matches!(s, TaskState::Failed { .. }))
            .count();

        /// SchedulerStatistics
        SchedulerStatistics {
            total_tasks: states.len(),
            pending_tasks: pending_count,
            running_tasks: running_count,
            completed_tasks: completed_count,
            failed_tasks: failed_count,
            queued_tasks: queue.len(),
            resource_utilization: pool.utilization_history.last().cloned(),
        }
    }

    /// Cancel a task
    pub fn cancel_task(&self, task_id: &TaskId) -> SklResult<()> {
        let mut states = self.task_states.write().unwrap();

        if let Some(current_state) = states.get(task_id) {
            match current_state {
                TaskState::Pending | TaskState::Ready => {
                    states.insert(
                        task_id.clone(),
                        TaskState::Cancelled {
                            cancelled_at: SystemTime::now(),
                        },
                    );
                    Ok(())
                }
                TaskState::Running { .. } => {
                    // In a real implementation, this would signal the running task to stop
                    states.insert(
                        task_id.clone(),
                        TaskState::Cancelled {
                            cancelled_at: SystemTime::now(),
                        },
                    );
                    Ok(())
                }
                _ => Err(SklearsError::InvalidInput(format!(
                    "Cannot cancel task {task_id} in state {current_state:?}"
                ))),
            }
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Task {task_id} not found"
            )))
        }
    }

    /// List all tasks with their states
    #[must_use]
    pub fn list_tasks(&self) -> HashMap<TaskId, TaskState> {
        let states = self.task_states.read().unwrap();
        states.clone()
    }

    /// Get current resource utilization
    #[must_use]
    pub fn get_resource_utilization(&self) -> ResourceUtilization {
        let pool = self.resource_pool.read().unwrap();
        pool.utilization_history
            .last()
            .cloned()
            .unwrap_or_else(|| ResourceUtilization {
                timestamp: SystemTime::now(),
                cpu_usage: 0.0,
                memory_usage: 0.0,
                disk_usage: 0.0,
                gpu_usage: 0.0,
            })
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics {
    /// Total number of tasks
    pub total_tasks: usize,
    /// Number of pending tasks
    pub pending_tasks: usize,
    /// Number of running tasks
    pub running_tasks: usize,
    /// Number of completed tasks
    pub completed_tasks: usize,
    /// Number of failed tasks
    pub failed_tasks: usize,
    /// Number of queued tasks
    pub queued_tasks: usize,
    /// Current resource utilization
    pub resource_utilization: Option<ResourceUtilization>,
}

/// Workflow manager for complex task orchestration
#[derive(Debug)]
pub struct WorkflowManager {
    /// Task scheduler
    scheduler: TaskScheduler,
    /// Workflow definitions
    workflows: Arc<RwLock<HashMap<String, Workflow>>>,
    /// Workflow instances
    workflow_instances: Arc<RwLock<HashMap<String, WorkflowInstance>>>,
}

/// Workflow definition
#[derive(Debug, Clone)]
pub struct Workflow {
    /// Workflow identifier
    pub id: String,
    /// Workflow name
    pub name: String,
    /// Workflow tasks
    pub tasks: Vec<WorkflowTask>,
    /// Global workflow configuration
    pub config: WorkflowConfig,
}

/// Workflow task definition
#[derive(Debug, Clone)]
pub struct WorkflowTask {
    /// Task identifier within workflow
    pub id: String,
    /// Task template
    pub template: TaskTemplate,
    /// Task dependencies within workflow
    pub depends_on: Vec<String>,
    /// Task configuration overrides
    pub config_overrides: HashMap<String, String>,
}

/// Task template for workflows
#[derive(Debug, Clone)]
pub struct TaskTemplate {
    /// Template name
    pub name: String,
    /// Component type
    pub component_type: ComponentType,
    /// Default resource requirements
    pub default_resources: ResourceRequirements,
    /// Default configuration
    pub default_config: HashMap<String, String>,
}

/// Workflow configuration
#[derive(Debug, Clone)]
pub struct WorkflowConfig {
    /// Maximum parallel tasks
    pub max_parallelism: usize,
    /// Workflow timeout
    pub timeout: Duration,
    /// Failure handling strategy
    pub failure_strategy: WorkflowFailureStrategy,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Workflow failure handling strategies
#[derive(Debug, Clone)]
pub enum WorkflowFailureStrategy {
    /// Stop the entire workflow on any task failure
    StopOnFailure,
    /// Continue with other tasks, skip failed dependencies
    ContinueOnFailure,
    /// Retry failed tasks automatically
    RetryFailedTasks,
    /// Use fallback tasks for failed ones
    UseFallbackTasks,
}

/// Workflow instance (execution)
#[derive(Debug, Clone)]
pub struct WorkflowInstance {
    /// Instance identifier
    pub id: String,
    /// Workflow definition ID
    pub workflow_id: String,
    /// Instance state
    pub state: WorkflowState,
    /// Task instances
    pub task_instances: HashMap<String, TaskId>,
    /// Start time
    pub started_at: SystemTime,
    /// End time
    pub ended_at: Option<SystemTime>,
    /// Execution context
    pub context: HashMap<String, String>,
}

/// Workflow execution state
#[derive(Debug, Clone, PartialEq)]
pub enum WorkflowState {
    /// Workflow is starting
    Starting,
    /// Workflow is running
    Running,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed
    Failed { error: String },
    /// Workflow was cancelled
    Cancelled,
    /// Workflow is paused
    Paused,
}

impl WorkflowManager {
    /// Create a new workflow manager
    #[must_use]
    pub fn new(scheduler: TaskScheduler) -> Self {
        Self {
            scheduler,
            workflows: Arc::new(RwLock::new(HashMap::new())),
            workflow_instances: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a workflow definition
    pub fn register_workflow(&self, workflow: Workflow) -> SklResult<()> {
        let mut workflows = self.workflows.write().unwrap();
        workflows.insert(workflow.id.clone(), workflow);
        Ok(())
    }

    /// Start a workflow instance
    pub fn start_workflow(
        &self,
        workflow_id: &str,
        context: HashMap<String, String>,
    ) -> SklResult<String> {
        let workflows = self.workflows.read().unwrap();
        let workflow = workflows.get(workflow_id).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Workflow {workflow_id} not found"))
        })?;

        let instance_id = format!(
            "{}_{}",
            workflow_id,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        let instance = WorkflowInstance {
            id: instance_id.clone(),
            workflow_id: workflow_id.to_string(),
            state: WorkflowState::Starting,
            task_instances: HashMap::new(),
            started_at: SystemTime::now(),
            ended_at: None,
            context,
        };

        {
            let mut instances = self.workflow_instances.write().unwrap();
            instances.insert(instance_id.clone(), instance);
        }

        // Submit initial tasks (those with no dependencies)
        self.submit_ready_tasks(&instance_id, workflow)?;

        Ok(instance_id)
    }

    /// Submit tasks that are ready to execute
    fn submit_ready_tasks(&self, instance_id: &str, workflow: &Workflow) -> SklResult<()> {
        let ready_tasks: Vec<_> = workflow
            .tasks
            .iter()
            .filter(|task| task.depends_on.is_empty())
            .collect();

        for task in ready_tasks {
            let scheduled_task = self.create_scheduled_task(instance_id, task)?;
            self.scheduler.submit_task(scheduled_task)?;
        }

        Ok(())
    }

    /// Create a scheduled task from workflow task
    fn create_scheduled_task(
        &self,
        instance_id: &str,
        workflow_task: &WorkflowTask,
    ) -> SklResult<ScheduledTask> {
        let task_id = format!("{}_{}", instance_id, workflow_task.id);

        Ok(ScheduledTask {
            id: task_id,
            name: workflow_task.template.name.clone(),
            component_type: workflow_task.template.component_type.clone(),
            dependencies: workflow_task
                .depends_on
                .iter()
                .map(|dep| format!("{instance_id}_{dep}"))
                .collect(),
            resource_requirements: workflow_task.template.default_resources.clone(),
            priority: TaskPriority::Normal,
            estimated_duration: Duration::from_secs(60),
            submitted_at: SystemTime::now(),
            deadline: None,
            metadata: HashMap::new(),
            retry_config: RetryConfig::default(),
        })
    }

    /// Get workflow instance status
    #[must_use]
    pub fn get_workflow_status(&self, instance_id: &str) -> Option<WorkflowInstance> {
        let instances = self.workflow_instances.read().unwrap();
        instances.get(instance_id).cloned()
    }

    /// Cancel a workflow instance
    pub fn cancel_workflow(&self, instance_id: &str) -> SklResult<()> {
        let mut instances = self.workflow_instances.write().unwrap();

        if let Some(instance) = instances.get_mut(instance_id) {
            instance.state = WorkflowState::Cancelled;
            instance.ended_at = Some(SystemTime::now());

            // Cancel all associated tasks
            for task_id in instance.task_instances.values() {
                let _ = self.scheduler.cancel_task(task_id);
            }

            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Workflow instance {instance_id} not found"
            )))
        }
    }

    /// List all workflow instances
    #[must_use]
    pub fn list_workflow_instances(&self) -> HashMap<String, WorkflowInstance> {
        let instances = self.workflow_instances.read().unwrap();
        instances.clone()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduled_task_creation() {
        let task = ScheduledTask {
            id: "test_task".to_string(),
            name: "Test Task".to_string(),
            component_type: ComponentType::Transformer,
            dependencies: vec!["dep1".to_string()],
            resource_requirements: ResourceRequirements {
                cpu_cores: 1,
                memory_mb: 512,
                disk_mb: 100,
                gpu_required: false,
                estimated_duration: Duration::from_secs(60),
                priority: TaskPriority::Normal,
            },
            priority: TaskPriority::Normal,
            estimated_duration: Duration::from_secs(60),
            submitted_at: SystemTime::now(),
            deadline: None,
            metadata: HashMap::new(),
            retry_config: RetryConfig::default(),
        };

        assert_eq!(task.id, "test_task");
        assert_eq!(task.dependencies.len(), 1);
        assert_eq!(task.priority, TaskPriority::Normal);
    }

    #[test]
    fn test_task_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = TaskScheduler::new(SchedulingStrategy::FIFO, config);

        let stats = scheduler.get_statistics();
        assert_eq!(stats.total_tasks, 0);
        assert_eq!(stats.pending_tasks, 0);
    }

    #[test]
    fn test_priority_task_ordering() {
        let task1 = PriorityTask {
            task: ScheduledTask {
                id: "task1".to_string(),
                name: "Task 1".to_string(),
                component_type: ComponentType::Transformer,
                dependencies: Vec::new(),
                resource_requirements: ResourceRequirements {
                    cpu_cores: 1,
                    memory_mb: 512,
                    disk_mb: 100,
                    gpu_required: false,
                    estimated_duration: Duration::from_secs(60),
                    priority: TaskPriority::Normal,
                },
                priority: TaskPriority::Normal,
                estimated_duration: Duration::from_secs(60),
                submitted_at: SystemTime::now(),
                deadline: None,
                metadata: HashMap::new(),
                retry_config: RetryConfig::default(),
            },
            priority_score: 10,
        };

        let task2 = PriorityTask {
            task: ScheduledTask {
                id: "task2".to_string(),
                name: "Task 2".to_string(),
                component_type: ComponentType::Transformer,
                dependencies: Vec::new(),
                resource_requirements: ResourceRequirements {
                    cpu_cores: 1,
                    memory_mb: 512,
                    disk_mb: 100,
                    gpu_required: false,
                    estimated_duration: Duration::from_secs(60),
                    priority: TaskPriority::High,
                },
                priority: TaskPriority::High,
                estimated_duration: Duration::from_secs(60),
                submitted_at: SystemTime::now(),
                deadline: None,
                metadata: HashMap::new(),
                retry_config: RetryConfig::default(),
            },
            priority_score: 100,
        };

        assert!(task2 > task1); // Higher priority score should come first
    }

    #[test]
    fn test_workflow_creation() {
        let workflow = Workflow {
            id: "test_workflow".to_string(),
            name: "Test Workflow".to_string(),
            tasks: vec![WorkflowTask {
                id: "task1".to_string(),
                template: TaskTemplate {
                    name: "Task 1".to_string(),
                    component_type: ComponentType::Transformer,
                    default_resources: ResourceRequirements {
                        cpu_cores: 1,
                        memory_mb: 512,
                        disk_mb: 100,
                        gpu_required: false,
                        estimated_duration: Duration::from_secs(60),
                        priority: TaskPriority::Normal,
                    },
                    default_config: HashMap::new(),
                },
                depends_on: Vec::new(),
                config_overrides: HashMap::new(),
            }],
            config: WorkflowConfig {
                max_parallelism: 5,
                timeout: Duration::from_secs(3600),
                failure_strategy: WorkflowFailureStrategy::StopOnFailure,
                retry_config: RetryConfig::default(),
            },
        };

        assert_eq!(workflow.id, "test_workflow");
        assert_eq!(workflow.tasks.len(), 1);
        assert_eq!(workflow.config.max_parallelism, 5);
    }

    #[test]
    fn test_resource_utilization() {
        let utilization = ResourceUtilization {
            timestamp: SystemTime::now(),
            cpu_usage: 0.5,
            memory_usage: 0.7,
            disk_usage: 0.3,
            gpu_usage: 0.0,
        };

        assert_eq!(utilization.cpu_usage, 0.5);
        assert_eq!(utilization.memory_usage, 0.7);
    }
}
