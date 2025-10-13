//! Execution Engine for Benchmark Management
//!
//! This module provides comprehensive execution engine capabilities including
//! thread pool management, resource allocation, task scheduling, and system monitoring.

use super::config_types::*;
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, RwLock, Mutex, mpsc, Condvar};
use std::time::{Duration, SystemTime, Instant};
use std::thread::{self, JoinHandle};
use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

// ================================================================================================
// CORE EXECUTION ENGINE
// ================================================================================================

/// Central execution engine for benchmark orchestration
pub struct ExecutionEngine {
    execution_pools: HashMap<String, Arc<Mutex<ExecutionPool>>>,
    resource_manager: Arc<RwLock<ResourceManager>>,
    system_monitor: Arc<RwLock<SystemMonitor>>,
    task_scheduler: Arc<Mutex<TaskScheduler>>,
    execution_coordinator: Arc<RwLock<ExecutionCoordinator>>,
    engine_config: ExecutionEngineConfig,
    engine_stats: Arc<RwLock<ExecutionEngineStats>>,
}

impl ExecutionEngine {
    /// Create a new execution engine
    pub fn new(config: ExecutionEngineConfig) -> Result<Self, ExecutionEngineError> {
        let resource_manager = Arc::new(RwLock::new(ResourceManager::new()?));
        let system_monitor = Arc::new(RwLock::new(SystemMonitor::new()?));
        let task_scheduler = Arc::new(Mutex::new(TaskScheduler::new()));
        let execution_coordinator = Arc::new(RwLock::new(ExecutionCoordinator::new()));

        let mut engine = Self {
            execution_pools: HashMap::new(),
            resource_manager,
            system_monitor,
            task_scheduler,
            execution_coordinator,
            engine_config: config.clone(),
            engine_stats: Arc::new(RwLock::new(ExecutionEngineStats::new())),
        };

        // Initialize default execution pools
        engine.initialize_default_pools()?;

        Ok(engine)
    }

    /// Add a new execution pool
    pub fn add_execution_pool(&mut self, pool_name: &str, pool_config: ExecutionPoolConfig) -> Result<(), ExecutionEngineError> {
        let pool = ExecutionPool::new(pool_name, pool_config)?;
        self.execution_pools.insert(pool_name.to_string(), Arc::new(Mutex::new(pool)));
        Ok(())
    }

    /// Submit a task for execution
    pub fn submit_task(&self, task: BenchmarkTask) -> Result<TaskHandle, ExecutionEngineError> {
        // Check resource availability
        let resource_manager = self.resource_manager.read().unwrap();
        if !resource_manager.can_allocate_resources(&task.resource_requirements) {
            return Err(ExecutionEngineError::InsufficientResources(
                format!("Not enough resources for task {}", task.task_id)
            ));
        }
        drop(resource_manager);

        // Schedule the task
        let mut scheduler = self.task_scheduler.lock().unwrap();
        let task_handle = scheduler.schedule_task(task)?;

        // Try to execute immediately if possible
        self.try_execute_next_task()?;

        Ok(task_handle)
    }

    /// Get execution engine statistics
    pub fn get_statistics(&self) -> ExecutionEngineStats {
        self.engine_stats.read().unwrap().clone()
    }

    /// Get system information
    pub fn get_system_info(&self) -> SystemInfo {
        self.system_monitor.read().unwrap().get_system_info()
    }

    /// Shutdown the execution engine
    pub fn shutdown(&mut self) -> Result<(), ExecutionEngineError> {
        // Signal all pools to shutdown
        for (_, pool) in &self.execution_pools {
            pool.lock().unwrap().shutdown()?;
        }

        // Wait for all tasks to complete
        self.wait_for_completion(Duration::from_secs(30))?;

        Ok(())
    }

    /// Get available execution pools
    pub fn list_execution_pools(&self) -> Vec<String> {
        self.execution_pools.keys().cloned().collect()
    }

    /// Get pool statistics
    pub fn get_pool_statistics(&self, pool_name: &str) -> Option<ExecutionPoolStatistics> {
        self.execution_pools.get(pool_name).map(|pool| {
            pool.lock().unwrap().get_statistics()
        })
    }

    // Private helper methods
    fn initialize_default_pools(&mut self) -> Result<(), ExecutionEngineError> {
        // CPU-intensive pool
        let cpu_pool_config = ExecutionPoolConfig {
            max_workers: self.engine_config.default_cpu_workers,
            queue_capacity: 100,
            worker_timeout: Duration::from_secs(300),
            task_timeout: Duration::from_secs(3600),
            priority_levels: 5,
        };
        self.add_execution_pool("cpu_intensive", cpu_pool_config)?;

        // IO-intensive pool
        let io_pool_config = ExecutionPoolConfig {
            max_workers: self.engine_config.default_io_workers,
            queue_capacity: 200,
            worker_timeout: Duration::from_secs(60),
            task_timeout: Duration::from_secs(1800),
            priority_levels: 3,
        };
        self.add_execution_pool("io_intensive", io_pool_config)?;

        // GPU pool (if available)
        if self.system_monitor.read().unwrap().has_gpu() {
            let gpu_pool_config = ExecutionPoolConfig {
                max_workers: self.engine_config.default_gpu_workers,
                queue_capacity: 50,
                worker_timeout: Duration::from_secs(600),
                task_timeout: Duration::from_secs(7200),
                priority_levels: 3,
            };
            self.add_execution_pool("gpu_accelerated", gpu_pool_config)?;
        }

        Ok(())
    }

    fn try_execute_next_task(&self) -> Result<(), ExecutionEngineError> {
        let mut scheduler = self.task_scheduler.lock().unwrap();
        if let Some(task) = scheduler.get_next_ready_task() {
            let pool_name = self.determine_best_pool(&task)?;
            if let Some(pool) = self.execution_pools.get(&pool_name) {
                pool.lock().unwrap().submit_task(task)?;
            }
        }
        Ok(())
    }

    fn determine_best_pool(&self, task: &BenchmarkTask) -> Result<String, ExecutionEngineError> {
        match task.task_type {
            TaskType::CPUIntensive => Ok("cpu_intensive".to_string()),
            TaskType::IOIntensive => Ok("io_intensive".to_string()),
            TaskType::GPUAccelerated => {
                if self.execution_pools.contains_key("gpu_accelerated") {
                    Ok("gpu_accelerated".to_string())
                } else {
                    Ok("cpu_intensive".to_string())
                }
            },
            TaskType::Mixed => Ok("cpu_intensive".to_string()),
            TaskType::Custom(ref pool_name) => {
                if self.execution_pools.contains_key(pool_name) {
                    Ok(pool_name.clone())
                } else {
                    Ok("cpu_intensive".to_string())
                }
            }
        }
    }

    fn wait_for_completion(&self, timeout: Duration) -> Result<(), ExecutionEngineError> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            let all_idle = self.execution_pools.values()
                .all(|pool| pool.lock().unwrap().is_idle());

            if all_idle {
                return Ok(());
            }

            thread::sleep(Duration::from_millis(100));
        }

        Err(ExecutionEngineError::TimeoutError("Timeout waiting for completion".to_string()))
    }
}

/// Configuration for the execution engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEngineConfig {
    pub default_cpu_workers: usize,
    pub default_io_workers: usize,
    pub default_gpu_workers: usize,
    pub resource_monitoring_interval: Duration,
    pub task_retry_limit: u32,
    pub enable_resource_isolation: bool,
    pub enable_performance_monitoring: bool,
}

impl Default for ExecutionEngineConfig {
    fn default() -> Self {
        Self {
            default_cpu_workers: num_cpus::get(),
            default_io_workers: num_cpus::get() * 2,
            default_gpu_workers: 2,
            resource_monitoring_interval: Duration::from_secs(1),
            task_retry_limit: 3,
            enable_resource_isolation: false,
            enable_performance_monitoring: true,
        }
    }
}

/// Statistics for the execution engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEngineStats {
    pub total_tasks_submitted: u64,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
    pub average_task_duration: Duration,
    pub total_execution_time: Duration,
    pub resource_utilization: ResourceUtilization,
    pub pool_statistics: HashMap<String, ExecutionPoolStatistics>,
}

impl ExecutionEngineStats {
    pub fn new() -> Self {
        Self {
            total_tasks_submitted: 0,
            total_tasks_completed: 0,
            total_tasks_failed: 0,
            average_task_duration: Duration::from_secs(0),
            total_execution_time: Duration::from_secs(0),
            resource_utilization: ResourceUtilization::default(),
            pool_statistics: HashMap::new(),
        }
    }
}

// ================================================================================================
// EXECUTION POOLS
// ================================================================================================

/// Execution pool for parallel benchmark execution
pub struct ExecutionPool {
    pool_name: String,
    config: ExecutionPoolConfig,
    workers: Vec<Worker>,
    task_queue: Arc<Mutex<VecDeque<BenchmarkTask>>>,
    active_tasks: Arc<Mutex<HashMap<String, TaskExecution>>>,
    pool_stats: Arc<RwLock<ExecutionPoolStatistics>>,
    shutdown_signal: Arc<AtomicBool>,
    coordinator: Arc<(Mutex<bool>, Condvar)>,
}

impl ExecutionPool {
    /// Create a new execution pool
    pub fn new(pool_name: &str, config: ExecutionPoolConfig) -> Result<Self, ExecutionEngineError> {
        let task_queue = Arc::new(Mutex::new(VecDeque::with_capacity(config.queue_capacity)));
        let active_tasks = Arc::new(Mutex::new(HashMap::new()));
        let pool_stats = Arc::new(RwLock::new(ExecutionPoolStatistics::new(pool_name)));
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let coordinator = Arc::new((Mutex::new(false), Condvar::new()));

        let mut workers = Vec::with_capacity(config.max_workers);

        // Spawn worker threads
        for worker_id in 0..config.max_workers {
            let worker = Worker::new(
                worker_id,
                pool_name,
                task_queue.clone(),
                active_tasks.clone(),
                pool_stats.clone(),
                shutdown_signal.clone(),
                coordinator.clone(),
                config.clone(),
            )?;
            workers.push(worker);
        }

        Ok(Self {
            pool_name: pool_name.to_string(),
            config,
            workers,
            task_queue,
            active_tasks,
            pool_stats,
            shutdown_signal,
            coordinator,
        })
    }

    /// Submit a task to the pool
    pub fn submit_task(&mut self, task: BenchmarkTask) -> Result<(), ExecutionEngineError> {
        if self.shutdown_signal.load(Ordering::Relaxed) {
            return Err(ExecutionEngineError::PoolShutdown);
        }

        {
            let mut queue = self.task_queue.lock().unwrap();
            if queue.len() >= self.config.queue_capacity {
                return Err(ExecutionEngineError::QueueFull);
            }
            queue.push_back(task);
        }

        // Notify workers that a new task is available
        let (lock, cvar) = &*self.coordinator;
        let _started = lock.lock().unwrap();
        cvar.notify_one();

        Ok(())
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> ExecutionPoolStatistics {
        self.pool_stats.read().unwrap().clone()
    }

    /// Check if pool is idle
    pub fn is_idle(&self) -> bool {
        let queue = self.task_queue.lock().unwrap();
        let active = self.active_tasks.lock().unwrap();
        queue.is_empty() && active.is_empty()
    }

    /// Shutdown the pool
    pub fn shutdown(&mut self) -> Result<(), ExecutionEngineError> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wake up all workers
        let (lock, cvar) = &*self.coordinator;
        let _started = lock.lock().unwrap();
        cvar.notify_all();

        // Wait for workers to finish
        for worker in &mut self.workers {
            worker.join()?;
        }

        Ok(())
    }

    /// Get active task count
    pub fn active_task_count(&self) -> usize {
        self.active_tasks.lock().unwrap().len()
    }

    /// Get queued task count
    pub fn queued_task_count(&self) -> usize {
        self.task_queue.lock().unwrap().len()
    }
}

/// Configuration for execution pools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPoolConfig {
    pub max_workers: usize,
    pub queue_capacity: usize,
    pub worker_timeout: Duration,
    pub task_timeout: Duration,
    pub priority_levels: u32,
}

/// Statistics for execution pools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPoolStatistics {
    pub pool_name: String,
    pub max_workers: usize,
    pub active_workers: usize,
    pub queued_tasks: usize,
    pub total_completed: u64,
    pub total_failed: u64,
    pub average_execution_time: Duration,
    pub throughput: f64, // tasks per second
    pub utilization: f64, // percentage
}

impl ExecutionPoolStatistics {
    pub fn new(pool_name: &str) -> Self {
        Self {
            pool_name: pool_name.to_string(),
            max_workers: 0,
            active_workers: 0,
            queued_tasks: 0,
            total_completed: 0,
            total_failed: 0,
            average_execution_time: Duration::from_secs(0),
            throughput: 0.0,
            utilization: 0.0,
        }
    }
}

// ================================================================================================
// WORKER THREADS
// ================================================================================================

/// Worker thread for executing benchmark tasks
pub struct Worker {
    worker_id: usize,
    pool_name: String,
    join_handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Create a new worker
    pub fn new(
        worker_id: usize,
        pool_name: &str,
        task_queue: Arc<Mutex<VecDeque<BenchmarkTask>>>,
        active_tasks: Arc<Mutex<HashMap<String, TaskExecution>>>,
        pool_stats: Arc<RwLock<ExecutionPoolStatistics>>,
        shutdown_signal: Arc<AtomicBool>,
        coordinator: Arc<(Mutex<bool>, Condvar)>,
        config: ExecutionPoolConfig,
    ) -> Result<Self, ExecutionEngineError> {
        let pool_name_clone = pool_name.to_string();

        let join_handle = thread::Builder::new()
            .name(format!("worker-{}-{}", pool_name, worker_id))
            .spawn(move || {
                Self::worker_loop(
                    worker_id,
                    task_queue,
                    active_tasks,
                    pool_stats,
                    shutdown_signal,
                    coordinator,
                    config,
                );
            })
            .map_err(|e| ExecutionEngineError::ThreadSpawnError(e.to_string()))?;

        Ok(Self {
            worker_id,
            pool_name: pool_name_clone,
            join_handle: Some(join_handle),
        })
    }

    /// Join the worker thread
    pub fn join(&mut self) -> Result<(), ExecutionEngineError> {
        if let Some(handle) = self.join_handle.take() {
            handle.join()
                .map_err(|_| ExecutionEngineError::ThreadJoinError(
                    format!("Failed to join worker {}", self.worker_id)
                ))?;
        }
        Ok(())
    }

    // Worker thread main loop
    fn worker_loop(
        worker_id: usize,
        task_queue: Arc<Mutex<VecDeque<BenchmarkTask>>>,
        active_tasks: Arc<Mutex<HashMap<String, TaskExecution>>>,
        pool_stats: Arc<RwLock<ExecutionPoolStatistics>>,
        shutdown_signal: Arc<AtomicBool>,
        coordinator: Arc<(Mutex<bool>, Condvar)>,
        config: ExecutionPoolConfig,
    ) {
        loop {
            // Check for shutdown signal
            if shutdown_signal.load(Ordering::Relaxed) {
                break;
            }

            // Try to get a task from the queue
            let task = {
                let mut queue = task_queue.lock().unwrap();
                queue.pop_front()
            };

            match task {
                Some(task) => {
                    // Execute the task
                    Self::execute_task(
                        worker_id,
                        task,
                        &active_tasks,
                        &pool_stats,
                        &config,
                    );
                },
                None => {
                    // No tasks available, wait for signal or timeout
                    let (lock, cvar) = &*coordinator;
                    let result = cvar.wait_timeout(
                        lock.lock().unwrap(),
                        config.worker_timeout,
                    ).unwrap();

                    if result.1.timed_out() {
                        // Consider scaling down if idle for too long
                        continue;
                    }
                }
            }
        }
    }

    fn execute_task(
        worker_id: usize,
        task: BenchmarkTask,
        active_tasks: &Arc<Mutex<HashMap<String, TaskExecution>>>,
        pool_stats: &Arc<RwLock<ExecutionPoolStatistics>>,
        config: &ExecutionPoolConfig,
    ) {
        let start_time = Instant::now();
        let task_id = task.task_id.clone();

        // Add to active tasks
        {
            let mut active = active_tasks.lock().unwrap();
            active.insert(task_id.clone(), TaskExecution {
                task: task.clone(),
                worker_id,
                start_time,
                status: TaskExecutionStatus::Running,
            });
        }

        // Execute the actual benchmark task
        let result = Self::run_benchmark_task(&task, config);
        let execution_time = start_time.elapsed();

        // Remove from active tasks
        {
            let mut active = active_tasks.lock().unwrap();
            active.remove(&task_id);
        }

        // Update statistics
        {
            let mut stats = pool_stats.write().unwrap();
            match result {
                Ok(_) => {
                    stats.total_completed += 1;
                },
                Err(_) => {
                    stats.total_failed += 1;
                }
            }

            // Update average execution time
            let total_tasks = stats.total_completed + stats.total_failed;
            if total_tasks > 0 {
                let total_time_ms = stats.average_execution_time.as_millis() * (total_tasks - 1) as u128
                    + execution_time.as_millis();
                stats.average_execution_time = Duration::from_millis((total_time_ms / total_tasks as u128) as u64);
            }
        }
    }

    fn run_benchmark_task(task: &BenchmarkTask, config: &ExecutionPoolConfig) -> Result<TaskResult, ExecutionEngineError> {
        // This is where the actual benchmark execution would happen
        // For now, we'll simulate execution

        match &task.simulation_mode {
            Some(SimulationMode::FastSimulation(duration)) => {
                thread::sleep(*duration);
                Ok(TaskResult {
                    task_id: task.task_id.clone(),
                    success: true,
                    execution_time: *duration,
                    metrics: generate_mock_metrics(&task.task_id),
                    error_message: None,
                })
            },
            Some(SimulationMode::MemoryIntensive(memory_mb)) => {
                // Simulate memory-intensive task
                let _memory_buffer: Vec<u8> = vec![0; (*memory_mb * 1024 * 1024) as usize];
                thread::sleep(Duration::from_millis(100));
                Ok(TaskResult {
                    task_id: task.task_id.clone(),
                    success: true,
                    execution_time: Duration::from_millis(100),
                    metrics: generate_mock_metrics(&task.task_id),
                    error_message: None,
                })
            },
            Some(SimulationMode::CPUIntensive(iterations)) => {
                // Simulate CPU-intensive task
                let mut sum = 0u64;
                for i in 0..*iterations {
                    sum = sum.wrapping_add(i);
                }
                Ok(TaskResult {
                    task_id: task.task_id.clone(),
                    success: true,
                    execution_time: Duration::from_millis(10),
                    metrics: generate_mock_metrics(&task.task_id),
                    error_message: None,
                })
            },
            None => {
                // Real execution would happen here
                // For now, simulate with a short delay
                thread::sleep(Duration::from_millis(10));
                Ok(TaskResult {
                    task_id: task.task_id.clone(),
                    success: true,
                    execution_time: Duration::from_millis(10),
                    metrics: HashMap::new(),
                    error_message: None,
                })
            }
        }
    }
}

// ================================================================================================
// TASK MANAGEMENT
// ================================================================================================

/// Enhanced benchmark task with execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    pub task_id: String,
    pub benchmark_id: String,
    pub execution_id: String,
    pub parameters: HashMap<String, String>,
    pub priority: u32,
    pub estimated_duration: Option<Duration>,
    pub resource_requirements: ResourceRequirements,
    pub retry_count: u32,
    pub max_retries: u32,
    pub task_type: TaskType,
    pub dependencies: Vec<String>,
    pub timeout: Option<Duration>,
    pub simulation_mode: Option<SimulationMode>,
}

impl BenchmarkTask {
    /// Create a new benchmark task
    pub fn new(
        task_id: &str,
        benchmark_id: &str,
        execution_id: &str,
        parameters: HashMap<String, String>
    ) -> Self {
        Self {
            task_id: task_id.to_string(),
            benchmark_id: benchmark_id.to_string(),
            execution_id: execution_id.to_string(),
            parameters,
            priority: 0,
            estimated_duration: None,
            resource_requirements: ResourceRequirements::default(),
            retry_count: 0,
            max_retries: 3,
            task_type: TaskType::CPUIntensive,
            dependencies: Vec::new(),
            timeout: None,
            simulation_mode: None,
        }
    }

    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Check if task has dependencies
    pub fn has_dependencies(&self) -> bool {
        !self.dependencies.is_empty()
    }
}

/// Types of benchmark tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    CPUIntensive,
    IOIntensive,
    GPUAccelerated,
    Mixed,
    Custom(String),
}

/// Simulation modes for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationMode {
    FastSimulation(Duration),
    MemoryIntensive(u64), // MB
    CPUIntensive(u64),   // iterations
}

/// Task execution tracking
#[derive(Debug, Clone)]
pub struct TaskExecution {
    pub task: BenchmarkTask,
    pub worker_id: usize,
    pub start_time: Instant,
    pub status: TaskExecutionStatus,
}

/// Task execution status
#[derive(Debug, Clone)]
pub enum TaskExecutionStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
}

/// Task result after execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub success: bool,
    pub execution_time: Duration,
    pub metrics: HashMap<String, f64>,
    pub error_message: Option<String>,
}

/// Task handle for tracking submitted tasks
#[derive(Debug, Clone)]
pub struct TaskHandle {
    pub task_id: String,
    pub submission_time: SystemTime,
    pub status: Arc<RwLock<TaskExecutionStatus>>,
}

impl TaskHandle {
    pub fn new(task_id: String) -> Self {
        Self {
            task_id,
            submission_time: SystemTime::now(),
            status: Arc::new(RwLock::new(TaskExecutionStatus::Queued)),
        }
    }

    pub fn get_status(&self) -> TaskExecutionStatus {
        self.status.read().unwrap().clone()
    }

    pub fn is_complete(&self) -> bool {
        matches!(
            *self.status.read().unwrap(),
            TaskExecutionStatus::Completed | TaskExecutionStatus::Failed | TaskExecutionStatus::Cancelled
        )
    }
}

// ================================================================================================
// TASK SCHEDULER
// ================================================================================================

/// Advanced task scheduler with priority and dependency management
pub struct TaskScheduler {
    task_queue: BTreeMap<u32, VecDeque<BenchmarkTask>>, // Priority -> Tasks
    dependency_graph: HashMap<String, Vec<String>>,
    completed_tasks: std::collections::HashSet<String>,
    task_handles: HashMap<String, TaskHandle>,
    scheduler_stats: TaskSchedulerStats,
}

impl TaskScheduler {
    /// Create a new task scheduler
    pub fn new() -> Self {
        Self {
            task_queue: BTreeMap::new(),
            dependency_graph: HashMap::new(),
            completed_tasks: std::collections::HashSet::new(),
            task_handles: HashMap::new(),
            scheduler_stats: TaskSchedulerStats::new(),
        }
    }

    /// Schedule a task
    pub fn schedule_task(&mut self, task: BenchmarkTask) -> Result<TaskHandle, ExecutionEngineError> {
        let task_handle = TaskHandle::new(task.task_id.clone());

        // Add dependencies to graph
        if task.has_dependencies() {
            self.dependency_graph.insert(task.task_id.clone(), task.dependencies.clone());
        }

        // Add to priority queue
        self.task_queue
            .entry(task.priority)
            .or_insert_with(VecDeque::new)
            .push_back(task);

        self.task_handles.insert(task_handle.task_id.clone(), task_handle.clone());
        self.scheduler_stats.total_scheduled += 1;

        Ok(task_handle)
    }

    /// Get the next ready task (with all dependencies satisfied)
    pub fn get_next_ready_task(&mut self) -> Option<BenchmarkTask> {
        // Find highest priority task that's ready to execute
        for (_, queue) in self.task_queue.iter_mut().rev() {
            let mut i = 0;
            while i < queue.len() {
                let task = &queue[i];
                if self.are_dependencies_satisfied(&task.task_id) {
                    return queue.remove(i);
                }
                i += 1;
            }
        }
        None
    }

    /// Mark a task as completed
    pub fn mark_task_completed(&mut self, task_id: &str) {
        self.completed_tasks.insert(task_id.to_string());
        self.scheduler_stats.total_completed += 1;

        // Update task handle status
        if let Some(handle) = self.task_handles.get(task_id) {
            *handle.status.write().unwrap() = TaskExecutionStatus::Completed;
        }
    }

    /// Mark a task as failed
    pub fn mark_task_failed(&mut self, task_id: &str) {
        self.scheduler_stats.total_failed += 1;

        // Update task handle status
        if let Some(handle) = self.task_handles.get(task_id) {
            *handle.status.write().unwrap() = TaskExecutionStatus::Failed;
        }
    }

    /// Get scheduler statistics
    pub fn get_statistics(&self) -> &TaskSchedulerStats {
        &self.scheduler_stats
    }

    /// Get pending task count
    pub fn pending_task_count(&self) -> usize {
        self.task_queue.values().map(|q| q.len()).sum()
    }

    // Private helper methods
    fn are_dependencies_satisfied(&self, task_id: &str) -> bool {
        if let Some(dependencies) = self.dependency_graph.get(task_id) {
            dependencies.iter().all(|dep| self.completed_tasks.contains(dep))
        } else {
            true // No dependencies
        }
    }
}

/// Task scheduler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSchedulerStats {
    pub total_scheduled: u64,
    pub total_completed: u64,
    pub total_failed: u64,
    pub average_wait_time: Duration,
}

impl TaskSchedulerStats {
    pub fn new() -> Self {
        Self {
            total_scheduled: 0,
            total_completed: 0,
            total_failed: 0,
            average_wait_time: Duration::from_secs(0),
        }
    }
}

// ================================================================================================
// RESOURCE MANAGEMENT
// ================================================================================================

/// Resource manager for tracking and allocating system resources
pub struct ResourceManager {
    available_resources: SystemResources,
    allocated_resources: HashMap<String, ResourceAllocation>,
    resource_limits: ResourceLimits,
    allocation_strategy: AllocationStrategy,
    resource_monitor: ResourceMonitor,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Result<Self, ExecutionEngineError> {
        let available_resources = SystemResources::discover()?;
        let resource_limits = ResourceLimits::from_system(&available_resources);

        Ok(Self {
            available_resources,
            allocated_resources: HashMap::new(),
            resource_limits,
            allocation_strategy: AllocationStrategy::Conservative,
            resource_monitor: ResourceMonitor::new()?,
        })
    }

    /// Check if resources can be allocated for a task
    pub fn can_allocate_resources(&self, requirements: &ResourceRequirements) -> bool {
        let current_usage = self.calculate_current_usage();

        current_usage.cpu_cores + requirements.cpu_cores <= self.resource_limits.max_cpu_cores &&
        current_usage.memory_mb + requirements.memory_mb <= self.resource_limits.max_memory_mb &&
        current_usage.gpu_count + requirements.gpu_count <= self.resource_limits.max_gpu_count
    }

    /// Allocate resources for a task
    pub fn allocate_resources(&mut self, task_id: &str, requirements: ResourceRequirements) -> Result<ResourceAllocation, ExecutionEngineError> {
        if !self.can_allocate_resources(&requirements) {
            return Err(ExecutionEngineError::InsufficientResources(
                format!("Cannot allocate resources for task {}", task_id)
            ));
        }

        let allocation = ResourceAllocation {
            task_id: task_id.to_string(),
            allocated_resources: requirements,
            allocation_time: SystemTime::now(),
            status: AllocationStatus::Allocated,
        };

        self.allocated_resources.insert(task_id.to_string(), allocation.clone());
        Ok(allocation)
    }

    /// Release resources for a task
    pub fn release_resources(&mut self, task_id: &str) -> Result<(), ExecutionEngineError> {
        if let Some(mut allocation) = self.allocated_resources.remove(task_id) {
            allocation.status = AllocationStatus::Released;
            Ok(())
        } else {
            Err(ExecutionEngineError::ResourceError(
                format!("No allocation found for task {}", task_id)
            ))
        }
    }

    /// Get current resource utilization
    pub fn get_resource_utilization(&self) -> ResourceUtilization {
        let current_usage = self.calculate_current_usage();

        ResourceUtilization {
            cpu_utilization: (current_usage.cpu_cores as f64 / self.available_resources.cpu_cores as f64) * 100.0,
            memory_utilization: (current_usage.memory_mb as f64 / self.available_resources.memory_mb as f64) * 100.0,
            gpu_utilization: if self.available_resources.gpu_count > 0 {
                (current_usage.gpu_count as f64 / self.available_resources.gpu_count as f64) * 100.0
            } else {
                0.0
            },
            network_utilization: 0.0, // Would be monitored in real implementation
            storage_utilization: 0.0,  // Would be monitored in real implementation
        }
    }

    fn calculate_current_usage(&self) -> ResourceRequirements {
        self.allocated_resources.values()
            .filter(|alloc| matches!(alloc.status, AllocationStatus::Allocated))
            .fold(ResourceRequirements::default(), |acc, alloc| {
                ResourceRequirements {
                    cpu_cores: acc.cpu_cores + alloc.allocated_resources.cpu_cores,
                    memory_mb: acc.memory_mb + alloc.allocated_resources.memory_mb,
                    gpu_count: acc.gpu_count + alloc.allocated_resources.gpu_count,
                    disk_space_mb: acc.disk_space_mb + alloc.allocated_resources.disk_space_mb,
                    network_bandwidth_mbps: acc.network_bandwidth_mbps + alloc.allocated_resources.network_bandwidth_mbps,
                }
            })
    }
}

/// System resources available for allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub gpu_count: u32,
    pub storage_mb: u64,
    pub network_bandwidth_mbps: f64,
}

impl SystemResources {
    /// Discover system resources
    pub fn discover() -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            cpu_cores: num_cpus::get() as u32,
            memory_mb: Self::get_total_memory()?,
            gpu_count: Self::get_gpu_count()?,
            storage_mb: Self::get_available_storage()?,
            network_bandwidth_mbps: 1000.0, // Placeholder
        })
    }

    fn get_total_memory() -> Result<u64, ExecutionEngineError> {
        // In a real implementation, this would query actual system memory
        Ok(8 * 1024) // 8GB placeholder
    }

    fn get_gpu_count() -> Result<u32, ExecutionEngineError> {
        // In a real implementation, this would query GPU information
        Ok(0) // No GPU placeholder
    }

    fn get_available_storage() -> Result<u64, ExecutionEngineError> {
        // In a real implementation, this would query available disk space
        Ok(100 * 1024) // 100GB placeholder
    }
}

/// Resource limits for safe allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_cpu_cores: u32,
    pub max_memory_mb: u64,
    pub max_gpu_count: u32,
    pub max_storage_mb: u64,
    pub max_network_bandwidth_mbps: f64,
}

impl ResourceLimits {
    /// Create resource limits from system resources (with safety margins)
    pub fn from_system(system: &SystemResources) -> Self {
        Self {
            max_cpu_cores: (system.cpu_cores as f64 * 0.8) as u32, // 80% of cores
            max_memory_mb: (system.memory_mb as f64 * 0.7) as u64, // 70% of memory
            max_gpu_count: system.gpu_count,
            max_storage_mb: (system.storage_mb as f64 * 0.5) as u64, // 50% of storage
            max_network_bandwidth_mbps: system.network_bandwidth_mbps * 0.8, // 80% of bandwidth
        }
    }
}

/// Resource allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub task_id: String,
    pub allocated_resources: ResourceRequirements,
    pub allocation_time: SystemTime,
    pub status: AllocationStatus,
}

/// Resource allocation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStatus {
    Allocated,
    Released,
    Failed,
}

/// Resource allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    Conservative, // Reserve extra resources
    Aggressive,   // Use maximum available
    Balanced,     // Balance between performance and stability
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub network_utilization: f64,
    pub storage_utilization: f64,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: 0.0,
            network_utilization: 0.0,
            storage_utilization: 0.0,
        }
    }
}

// ================================================================================================
// SYSTEM MONITORING
// ================================================================================================

/// System monitor for real-time resource tracking
pub struct SystemMonitor {
    hardware_info: HardwareInfo,
    monitoring_config: MonitoringConfig,
    resource_history: VecDeque<ResourceSnapshot>,
    monitoring_thread: Option<JoinHandle<()>>,
    shutdown_signal: Arc<AtomicBool>,
}

impl SystemMonitor {
    /// Create a new system monitor
    pub fn new() -> Result<Self, ExecutionEngineError> {
        let hardware_info = HardwareInfo::detect()?;
        let monitoring_config = MonitoringConfig::default();

        Ok(Self {
            hardware_info,
            monitoring_config,
            resource_history: VecDeque::with_capacity(1000),
            monitoring_thread: None,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start resource monitoring
    pub fn start_monitoring(&mut self) -> Result<(), ExecutionEngineError> {
        let shutdown_signal = self.shutdown_signal.clone();
        let monitoring_interval = self.monitoring_config.sampling_interval;

        let handle = thread::spawn(move || {
            while !shutdown_signal.load(Ordering::Relaxed) {
                // Collect resource snapshot
                if let Ok(snapshot) = ResourceSnapshot::capture() {
                    // In a real implementation, this would be stored or processed
                }

                thread::sleep(monitoring_interval);
            }
        });

        self.monitoring_thread = Some(handle);
        Ok(())
    }

    /// Stop resource monitoring
    pub fn stop_monitoring(&mut self) -> Result<(), ExecutionEngineError> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        if let Some(handle) = self.monitoring_thread.take() {
            handle.join()
                .map_err(|_| ExecutionEngineError::ThreadJoinError("Failed to join monitoring thread".to_string()))?;
        }

        Ok(())
    }

    /// Get system information
    pub fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            hardware_info: self.hardware_info.clone(),
            resource_availability: self.get_current_availability(),
            monitoring_active: self.monitoring_thread.is_some(),
        }
    }

    /// Check if system has GPU
    pub fn has_gpu(&self) -> bool {
        !self.hardware_info.gpu_info.is_empty()
    }

    fn get_current_availability(&self) -> ResourceAvailability {
        // In a real implementation, this would calculate based on current usage
        ResourceAvailability {
            available_cpu_cores: self.hardware_info.cpu_info.cores,
            available_memory_mb: self.hardware_info.memory_info.total_memory / (1024 * 1024),
            available_gpu_count: self.hardware_info.gpu_info.len() as u32,
            available_storage_mb: 100 * 1024, // Placeholder
        }
    }
}

/// Hardware information detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_info: CPUInfo,
    pub memory_info: MemoryInfo,
    pub gpu_info: Vec<GPUInfo>,
    pub storage_info: Vec<StorageDeviceInfo>,
    pub motherboard_info: MotherboardInfo,
}

impl HardwareInfo {
    /// Detect hardware information
    pub fn detect() -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            cpu_info: CPUInfo::detect()?,
            memory_info: MemoryInfo::detect()?,
            gpu_info: GPUInfo::detect_all()?,
            storage_info: StorageDeviceInfo::detect_all()?,
            motherboard_info: MotherboardInfo::detect()?,
        })
    }
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub base_frequency: f64,
    pub max_frequency: f64,
    pub cache_sizes: Vec<u64>,
    pub features: Vec<String>,
}

impl CPUInfo {
    fn detect() -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            model: "Generic CPU".to_string(),
            cores: num_cpus::get() as u32,
            threads: num_cpus::get() as u32,
            base_frequency: 2.0,
            max_frequency: 3.0,
            cache_sizes: vec![32768, 262144, 8388608], // L1, L2, L3
            features: vec!["SSE".to_string(), "AVX".to_string()],
        })
    }
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_memory: u64,
    pub available_memory: u64,
    pub memory_type: String,
    pub memory_speed: u32,
    pub memory_channels: u32,
}

impl MemoryInfo {
    fn detect() -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            available_memory: 6 * 1024 * 1024 * 1024, // 6GB
            memory_type: "DDR4".to_string(),
            memory_speed: 3200,
            memory_channels: 2,
        })
    }
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub model: String,
    pub memory: u64,
    pub compute_capability: String,
    pub driver_version: String,
    pub power_limit: u32,
}

impl GPUInfo {
    fn detect_all() -> Result<Vec<Self>, ExecutionEngineError> {
        // In a real implementation, this would detect actual GPUs
        Ok(Vec::new()) // No GPUs detected
    }
}

/// Storage device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDeviceInfo {
    pub device_name: String,
    pub device_type: StorageType,
    pub total_capacity: u64,
    pub available_capacity: u64,
    pub read_speed: f64,
    pub write_speed: f64,
}

impl StorageDeviceInfo {
    fn detect_all() -> Result<Vec<Self>, ExecutionEngineError> {
        Ok(vec![Self {
            device_name: "System Drive".to_string(),
            device_type: StorageType::SSD,
            total_capacity: 500 * 1024 * 1024 * 1024, // 500GB
            available_capacity: 300 * 1024 * 1024 * 1024, // 300GB
            read_speed: 500.0,  // MB/s
            write_speed: 450.0, // MB/s
        }])
    }
}

/// Storage device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    HDD,
    SSD,
    NVMe,
    Network,
}

/// Motherboard information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotherboardInfo {
    pub manufacturer: String,
    pub model: String,
    pub bios_version: String,
    pub chipset: String,
}

impl MotherboardInfo {
    fn detect() -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            manufacturer: "Generic".to_string(),
            model: "Unknown".to_string(),
            bios_version: "1.0".to_string(),
            chipset: "Unknown".to_string(),
        })
    }
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub sampling_interval: Duration,
    pub history_size: usize,
    pub enable_detailed_monitoring: bool,
    pub alert_thresholds: AlertThresholds,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            sampling_interval: Duration::from_secs(1),
            history_size: 1000,
            enable_detailed_monitoring: false,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds for resource monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub disk_usage: f64,
    pub temperature: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage: 90.0,
            memory_usage: 85.0,
            gpu_usage: 90.0,
            disk_usage: 85.0,
            temperature: 80.0,
        }
    }
}

/// Resource snapshot for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub timestamp: SystemTime,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: Vec<f64>,
    pub disk_io: DiskIOStats,
    pub network_io: NetworkIOStats,
    pub temperature: f64,
}

impl ResourceSnapshot {
    /// Capture current resource snapshot
    pub fn capture() -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            timestamp: SystemTime::now(),
            cpu_usage: 0.0,     // Would be measured in real implementation
            memory_usage: 0.0,  // Would be measured in real implementation
            gpu_usage: Vec::new(), // Would be measured in real implementation
            disk_io: DiskIOStats::default(),
            network_io: NetworkIOStats::default(),
            temperature: 45.0,  // Placeholder
        })
    }
}

// ================================================================================================
// EXECUTION COORDINATION
// ================================================================================================

/// Execution coordinator for managing complex execution workflows
pub struct ExecutionCoordinator {
    active_workflows: HashMap<String, ExecutionWorkflow>,
    workflow_templates: HashMap<String, WorkflowTemplate>,
    coordinator_stats: ExecutionCoordinatorStats,
}

impl ExecutionCoordinator {
    /// Create a new execution coordinator
    pub fn new() -> Self {
        Self {
            active_workflows: HashMap::new(),
            workflow_templates: HashMap::new(),
            coordinator_stats: ExecutionCoordinatorStats::new(),
        }
    }

    /// Register a workflow template
    pub fn register_workflow_template(&mut self, template: WorkflowTemplate) {
        self.workflow_templates.insert(template.template_id.clone(), template);
    }

    /// Start a workflow execution
    pub fn start_workflow(&mut self, workflow_id: &str, template_id: &str, parameters: HashMap<String, String>) -> Result<(), ExecutionEngineError> {
        let template = self.workflow_templates.get(template_id)
            .ok_or_else(|| ExecutionEngineError::ConfigurationError(format!("Template {} not found", template_id)))?;

        let workflow = ExecutionWorkflow::from_template(workflow_id, template, parameters)?;
        self.active_workflows.insert(workflow_id.to_string(), workflow);

        Ok(())
    }

    /// Get workflow status
    pub fn get_workflow_status(&self, workflow_id: &str) -> Option<&ExecutionWorkflow> {
        self.active_workflows.get(workflow_id)
    }
}

/// Execution workflow for complex benchmark orchestration
#[derive(Debug, Clone)]
pub struct ExecutionWorkflow {
    pub workflow_id: String,
    pub template_id: String,
    pub status: WorkflowStatus,
    pub stages: Vec<WorkflowStage>,
    pub current_stage: usize,
    pub start_time: SystemTime,
    pub parameters: HashMap<String, String>,
}

impl ExecutionWorkflow {
    /// Create workflow from template
    pub fn from_template(workflow_id: &str, template: &WorkflowTemplate, parameters: HashMap<String, String>) -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            workflow_id: workflow_id.to_string(),
            template_id: template.template_id.clone(),
            status: WorkflowStatus::Pending,
            stages: template.stages.clone(),
            current_stage: 0,
            start_time: SystemTime::now(),
            parameters,
        })
    }
}

/// Workflow template for reusable execution patterns
#[derive(Debug, Clone)]
pub struct WorkflowTemplate {
    pub template_id: String,
    pub name: String,
    pub description: String,
    pub stages: Vec<WorkflowStage>,
    pub default_parameters: HashMap<String, String>,
}

/// Workflow stage
#[derive(Debug, Clone)]
pub struct WorkflowStage {
    pub stage_id: String,
    pub name: String,
    pub stage_type: StageType,
    pub dependencies: Vec<String>,
    pub timeout: Option<Duration>,
}

/// Types of workflow stages
#[derive(Debug, Clone)]
pub enum StageType {
    TaskExecution(Vec<String>), // Task IDs
    ResourcePreparation,
    Validation,
    Cleanup,
    Custom(String),
}

/// Workflow execution status
#[derive(Debug, Clone)]
pub enum WorkflowStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Execution coordinator statistics
#[derive(Debug, Clone)]
pub struct ExecutionCoordinatorStats {
    pub total_workflows: u64,
    pub completed_workflows: u64,
    pub failed_workflows: u64,
    pub average_workflow_duration: Duration,
}

impl ExecutionCoordinatorStats {
    pub fn new() -> Self {
        Self {
            total_workflows: 0,
            completed_workflows: 0,
            failed_workflows: 0,
            average_workflow_duration: Duration::from_secs(0),
        }
    }
}

// ================================================================================================
// SUPPORTING TYPES AND UTILITIES
// ================================================================================================

/// System information aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub hardware_info: HardwareInfo,
    pub resource_availability: ResourceAvailability,
    pub monitoring_active: bool,
}

/// Current resource availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    pub available_cpu_cores: u32,
    pub available_memory_mb: u64,
    pub available_gpu_count: u32,
    pub available_storage_mb: u64,
}

/// Resource monitor for real-time tracking
pub struct ResourceMonitor {
    monitoring_active: AtomicBool,
}

impl ResourceMonitor {
    pub fn new() -> Result<Self, ExecutionEngineError> {
        Ok(Self {
            monitoring_active: AtomicBool::new(false),
        })
    }
}

/// Execution engine errors
#[derive(Debug, thiserror::Error)]
pub enum ExecutionEngineError {
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
    #[error("Queue is full")]
    QueueFull,
    #[error("Pool is shutdown")]
    PoolShutdown,
    #[error("Thread spawn error: {0}")]
    ThreadSpawnError(String),
    #[error("Thread join error: {0}")]
    ThreadJoinError(String),
    #[error("Resource error: {0}")]
    ResourceError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    #[error("System error: {0}")]
    SystemError(String),
}

// Helper function for generating mock metrics
fn generate_mock_metrics(task_id: &str) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();
    metrics.insert("execution_time_ms".to_string(), 100.0);
    metrics.insert("memory_usage_mb".to_string(), 256.0);
    metrics.insert("cpu_utilization".to_string(), 75.0);
    metrics.insert(format!("task_{}_score", task_id), 0.95);
    metrics
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_engine_creation() {
        let config = ExecutionEngineConfig::default();
        let engine = ExecutionEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_task_creation() {
        let mut parameters = HashMap::new();
        parameters.insert("param1".to_string(), "value1".to_string());

        let task = BenchmarkTask::new("task1", "benchmark1", "exec1", parameters);
        assert_eq!(task.task_id, "task1");
        assert_eq!(task.benchmark_id, "benchmark1");
        assert_eq!(task.execution_id, "exec1");
        assert!(task.can_retry());
    }

    #[test]
    fn test_execution_pool_config() {
        let config = ExecutionPoolConfig {
            max_workers: 4,
            queue_capacity: 100,
            worker_timeout: Duration::from_secs(60),
            task_timeout: Duration::from_secs(300),
            priority_levels: 5,
        };

        assert_eq!(config.max_workers, 4);
        assert_eq!(config.queue_capacity, 100);
    }

    #[test]
    fn test_resource_requirements() {
        let requirements = ResourceRequirements {
            cpu_cores: 2,
            memory_mb: 1024,
            gpu_count: 1,
            disk_space_mb: 2048,
            network_bandwidth_mbps: 100.0,
        };

        assert_eq!(requirements.cpu_cores, 2);
        assert_eq!(requirements.memory_mb, 1024);
    }

    #[test]
    fn test_task_scheduler() {
        let mut scheduler = TaskScheduler::new();

        let task = BenchmarkTask::new("task1", "benchmark1", "exec1", HashMap::new());
        let handle = scheduler.schedule_task(task);
        assert!(handle.is_ok());

        assert_eq!(scheduler.pending_task_count(), 1);

        let next_task = scheduler.get_next_ready_task();
        assert!(next_task.is_some());
        assert_eq!(next_task.unwrap().task_id, "task1");
    }

    #[test]
    fn test_system_resources_discovery() {
        let resources = SystemResources::discover();
        assert!(resources.is_ok());

        let res = resources.unwrap();
        assert!(res.cpu_cores > 0);
        assert!(res.memory_mb > 0);
    }

    #[test]
    fn test_hardware_info_detection() {
        let hardware = HardwareInfo::detect();
        assert!(hardware.is_ok());

        let hw = hardware.unwrap();
        assert!(hw.cpu_info.cores > 0);
        assert!(hw.memory_info.total_memory > 0);
    }

    #[test]
    fn test_resource_manager() {
        let manager = ResourceManager::new();
        assert!(manager.is_ok());

        let mut rm = manager.unwrap();
        let requirements = ResourceRequirements {
            cpu_cores: 1,
            memory_mb: 512,
            gpu_count: 0,
            disk_space_mb: 1024,
            network_bandwidth_mbps: 10.0,
        };

        assert!(rm.can_allocate_resources(&requirements));

        let allocation = rm.allocate_resources("test_task", requirements);
        assert!(allocation.is_ok());
    }

    #[test]
    fn test_task_handle() {
        let handle = TaskHandle::new("test_task".to_string());
        assert_eq!(handle.task_id, "test_task");
        assert!(matches!(handle.get_status(), TaskExecutionStatus::Queued));
        assert!(!handle.is_complete());
    }

    #[test]
    fn test_task_simulation_modes() {
        let mut task = BenchmarkTask::new("task1", "benchmark1", "exec1", HashMap::new());

        task.simulation_mode = Some(SimulationMode::FastSimulation(Duration::from_millis(100)));
        assert!(task.simulation_mode.is_some());

        task.simulation_mode = Some(SimulationMode::MemoryIntensive(512));
        assert!(task.simulation_mode.is_some());

        task.simulation_mode = Some(SimulationMode::CPUIntensive(1000000));
        assert!(task.simulation_mode.is_some());
    }
}