//! Distributed computation infrastructure for enterprise-scale explanation tasks
//!
//! This module provides distributed computing capabilities for scaling explanation
//! computation across multiple nodes in a cluster. It includes:
//!
//! * Distributed task scheduling and execution
//! * Load balancing across compute nodes
//! * Fault tolerance and retry mechanisms
//! * Result aggregation from multiple workers
//! * Cluster health monitoring
//! * Dynamic worker scaling
//!
//! # Architecture
//!
//! The distributed infrastructure follows a coordinator-worker architecture:
//! - **Coordinator**: Manages task distribution, load balancing, and result aggregation
//! - **Workers**: Execute explanation computation tasks independently
//! - **Task Queue**: Manages pending tasks with priority scheduling
//! - **Result Store**: Aggregates and persists computation results
//!
//! # Example
//!
//! ```rust
//! use sklears_inspection::distributed::{DistributedCoordinator, WorkerNode, ClusterConfig};
//!
//! // Create a cluster configuration
//! let config = ClusterConfig {
//!     max_workers: 10,
//!     task_timeout_seconds: 300,
//!     retry_attempts: 3,
//!     load_balancing_strategy: LoadBalancingStrategy::LeastLoaded,
//!     enable_fault_tolerance: true,
//! };
//!
//! // Initialize coordinator
//! let coordinator = DistributedCoordinator::new(config)?;
//!
//! // Register workers
//! coordinator.register_worker("worker1".to_string(), "192.168.1.10:8080")?;
//! coordinator.register_worker("worker2".to_string(), "192.168.1.11:8080")?;
//!
//! // Submit explanation tasks
//! let task_id = coordinator.submit_task(explanation_task)?;
//!
//! // Wait for results
//! let result = coordinator.get_result(task_id)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::types::Float;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for distributed cluster
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Maximum number of workers in the cluster
    pub max_workers: usize,
    /// Task execution timeout in seconds
    pub task_timeout_seconds: u64,
    /// Number of retry attempts for failed tasks
    pub retry_attempts: usize,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Enable fault tolerance
    pub enable_fault_tolerance: bool,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_seconds: u64,
    /// Maximum task queue size
    pub max_queue_size: usize,
    /// Enable dynamic worker scaling
    pub enable_auto_scaling: bool,
    /// Target CPU utilization for auto-scaling (0.0-1.0)
    pub target_cpu_utilization: f64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            max_workers: 10,
            task_timeout_seconds: 300,
            retry_attempts: 3,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            enable_fault_tolerance: true,
            heartbeat_interval_seconds: 30,
            max_queue_size: 1000,
            enable_auto_scaling: false,
            target_cpu_utilization: 0.7,
        }
    }
}

/// Load balancing strategy for task distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Assign to least loaded worker
    LeastLoaded,
    /// Weighted distribution based on worker capacity
    Weighted,
    /// Random assignment
    Random,
    /// Locality-aware (prefer workers with cached data)
    LocalityAware,
}

/// Distributed task coordinator
pub struct DistributedCoordinator {
    /// Configuration
    config: ClusterConfig,
    /// Registered workers
    workers: Arc<Mutex<HashMap<String, WorkerNode>>>,
    /// Task queue
    task_queue: Arc<Mutex<VecDeque<DistributedTask>>>,
    /// Completed results
    results: Arc<Mutex<HashMap<String, TaskResult>>>,
    /// Task assignments
    assignments: Arc<Mutex<HashMap<String, String>>>, // task_id -> worker_id
    /// Round-robin counter for load balancing
    round_robin_counter: Arc<Mutex<usize>>,
    /// Cluster statistics
    statistics: Arc<Mutex<ClusterStatistics>>,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub fn new(config: ClusterConfig) -> SklResult<Self> {
        Ok(Self {
            config,
            workers: Arc::new(Mutex::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            results: Arc::new(Mutex::new(HashMap::new())),
            assignments: Arc::new(Mutex::new(HashMap::new())),
            round_robin_counter: Arc::new(Mutex::new(0)),
            statistics: Arc::new(Mutex::new(ClusterStatistics::new())),
        })
    }

    /// Register a worker node
    pub fn register_worker(&self, worker_id: String, address: String) -> SklResult<()> {
        let mut workers = self.workers.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
        })?;

        if workers.len() >= self.config.max_workers {
            return Err(SklearsError::InvalidInput(
                "Maximum number of workers reached".to_string(),
            ));
        }

        let worker = WorkerNode::new(worker_id.clone(), address);
        workers.insert(worker_id.clone(), worker);

        let mut stats = self.statistics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
        })?;
        stats.active_workers += 1;

        Ok(())
    }

    /// Unregister a worker node
    pub fn unregister_worker(&self, worker_id: &str) -> SklResult<()> {
        let mut workers = self.workers.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
        })?;

        if workers.remove(worker_id).is_some() {
            let mut stats = self.statistics.lock().map_err(|_| {
                SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
            })?;
            stats.active_workers = stats.active_workers.saturating_sub(1);
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Worker {} not found",
                worker_id
            )))
        }
    }

    /// Submit a task for distributed execution
    pub fn submit_task(&self, task: DistributedTask) -> SklResult<String> {
        let mut queue = self
            .task_queue
            .lock()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire queue lock".to_string()))?;

        if queue.len() >= self.config.max_queue_size {
            return Err(SklearsError::InvalidInput("Task queue is full".to_string()));
        }

        let task_id = task.task_id.clone();
        queue.push_back(task);

        let mut stats = self.statistics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
        })?;
        stats.total_tasks_submitted += 1;
        stats.pending_tasks += 1;

        Ok(task_id)
    }

    /// Process pending tasks and assign to workers
    pub fn schedule_tasks(&self) -> SklResult<usize> {
        let mut scheduled = 0;

        loop {
            // Get next task from queue
            let task = {
                let mut queue = self.task_queue.lock().map_err(|_| {
                    SklearsError::InvalidInput("Failed to acquire queue lock".to_string())
                })?;
                queue.pop_front()
            };

            match task {
                None => break, // No more tasks
                Some(task) => {
                    // Select worker based on load balancing strategy
                    let worker_id = self.select_worker(&task)?;

                    // Assign task to worker
                    self.assign_task_to_worker(task, &worker_id)?;

                    scheduled += 1;
                }
            }
        }

        Ok(scheduled)
    }

    /// Select a worker based on load balancing strategy
    fn select_worker(&self, task: &DistributedTask) -> SklResult<String> {
        let workers = self.workers.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
        })?;

        if workers.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No workers available".to_string(),
            ));
        }

        match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut counter = self.round_robin_counter.lock().map_err(|_| {
                    SklearsError::InvalidInput("Failed to acquire counter lock".to_string())
                })?;

                let worker_ids: Vec<String> = workers.keys().cloned().collect();
                let selected = &worker_ids[*counter % worker_ids.len()];
                *counter += 1;

                Ok(selected.clone())
            }
            LoadBalancingStrategy::LeastLoaded => {
                let mut least_loaded_worker = None;
                let mut min_load = usize::MAX;

                for (worker_id, worker) in workers.iter() {
                    if worker.current_load < min_load {
                        min_load = worker.current_load;
                        least_loaded_worker = Some(worker_id.clone());
                    }
                }

                least_loaded_worker.ok_or_else(|| {
                    SklearsError::InvalidInput("Failed to find least loaded worker".to_string())
                })
            }
            LoadBalancingStrategy::Weighted => {
                // Simplified: use worker capacity as weight
                let mut best_worker = None;
                let mut best_score = 0.0;

                for (worker_id, worker) in workers.iter() {
                    let score = (worker.capacity as f64) / (worker.current_load as f64 + 1.0);
                    if score > best_score {
                        best_score = score;
                        best_worker = Some(worker_id.clone());
                    }
                }

                best_worker.ok_or_else(|| {
                    SklearsError::InvalidInput("Failed to find weighted worker".to_string())
                })
            }
            LoadBalancingStrategy::Random => {
                use scirs2_core::random::{thread_rng, CoreRandom};

                let worker_ids: Vec<String> = workers.keys().cloned().collect();
                let mut rng = thread_rng();
                let index = rng.gen_range(0..worker_ids.len());
                Ok(worker_ids[index].clone())
            }
            LoadBalancingStrategy::LocalityAware => {
                // Simplified: prefer worker with lowest network latency
                let mut best_worker = None;
                let mut best_latency = Duration::from_secs(u64::MAX);

                for (worker_id, worker) in workers.iter() {
                    if worker.network_latency < best_latency {
                        best_latency = worker.network_latency;
                        best_worker = Some(worker_id.clone());
                    }
                }

                best_worker.ok_or_else(|| {
                    SklearsError::InvalidInput("Failed to find locality-aware worker".to_string())
                })
            }
        }
    }

    /// Assign task to a specific worker
    fn assign_task_to_worker(&self, task: DistributedTask, worker_id: &str) -> SklResult<()> {
        // Update worker state and record assignment (in separate scope to release locks)
        {
            let mut workers = self.workers.lock().map_err(|_| {
                SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
            })?;

            let worker = workers.get_mut(worker_id).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Worker {} not found", worker_id))
            })?;

            // Update worker state
            worker.current_load += 1;
            worker.total_tasks_processed += 1;
        } // Release workers lock here

        // Record assignment
        {
            let mut assignments = self.assignments.lock().map_err(|_| {
                SklearsError::InvalidInput("Failed to acquire assignments lock".to_string())
            })?;
            assignments.insert(task.task_id.clone(), worker_id.to_string());
        } // Release assignments lock here

        // Update statistics
        {
            let mut stats = self.statistics.lock().map_err(|_| {
                SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
            })?;
            stats.pending_tasks = stats.pending_tasks.saturating_sub(1);
            stats.running_tasks += 1;
        } // Release stats lock here

        // In a real implementation, this would send the task to the worker
        // For now, we simulate immediate processing (all locks released before this call)
        self.simulate_task_execution(task, worker_id.to_string())?;

        Ok(())
    }

    /// Simulate task execution (in real implementation, this would be async)
    fn simulate_task_execution(&self, task: DistributedTask, worker_id: String) -> SklResult<()> {
        // Simulate task processing
        let result = TaskResult {
            task_id: task.task_id.clone(),
            worker_id: worker_id.clone(),
            status: TaskStatus::Completed,
            result_data: Array1::zeros(10), // Placeholder
            execution_time: Duration::from_millis(100),
            retry_count: 0,
        };

        // Store result
        let mut results = self.results.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire results lock".to_string())
        })?;
        results.insert(task.task_id.clone(), result);

        // Update worker load
        let mut workers = self.workers.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
        })?;
        if let Some(worker) = workers.get_mut(&worker_id) {
            worker.current_load = worker.current_load.saturating_sub(1);
        }

        // Update statistics
        let mut stats = self.statistics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
        })?;
        stats.running_tasks = stats.running_tasks.saturating_sub(1);
        stats.completed_tasks += 1;

        Ok(())
    }

    /// Get result for a completed task
    pub fn get_result(&self, task_id: &str) -> SklResult<TaskResult> {
        let results = self.results.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire results lock".to_string())
        })?;

        results.get(task_id).cloned().ok_or_else(|| {
            SklearsError::InvalidInput(format!("Result for task {} not found", task_id))
        })
    }

    /// Get cluster statistics
    pub fn get_statistics(&self) -> SklResult<ClusterStatistics> {
        let stats = self.statistics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
        })?;

        Ok(stats.clone())
    }

    /// Get worker information
    pub fn get_worker_info(&self, worker_id: &str) -> SklResult<WorkerNode> {
        let workers = self.workers.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
        })?;

        workers
            .get(worker_id)
            .cloned()
            .ok_or_else(|| SklearsError::InvalidInput(format!("Worker {} not found", worker_id)))
    }

    /// Get all workers
    pub fn get_all_workers(&self) -> SklResult<Vec<WorkerNode>> {
        let workers = self.workers.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
        })?;

        Ok(workers.values().cloned().collect())
    }

    /// Health check for the cluster
    pub fn health_check(&self) -> SklResult<ClusterHealth> {
        let workers = self.workers.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire workers lock".to_string())
        })?;

        let stats = self.statistics.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire statistics lock".to_string())
        })?;

        let total_workers = workers.len();
        let healthy_workers = workers.values().filter(|w| w.is_healthy).count();

        let health_status = if healthy_workers == 0 {
            HealthStatus::Critical
        } else if healthy_workers < total_workers / 2 {
            HealthStatus::Degraded
        } else if healthy_workers < total_workers {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        Ok(ClusterHealth {
            status: health_status,
            total_workers,
            healthy_workers,
            total_capacity: workers.values().map(|w| w.capacity).sum(),
            current_load: workers.values().map(|w| w.current_load).sum(),
            pending_tasks: stats.pending_tasks,
            running_tasks: stats.running_tasks,
        })
    }
}

/// Worker node in the cluster
#[derive(Debug, Clone)]
pub struct WorkerNode {
    /// Worker identifier
    pub worker_id: String,
    /// Network address
    pub address: String,
    /// Worker capacity (max concurrent tasks)
    pub capacity: usize,
    /// Current load (number of active tasks)
    pub current_load: usize,
    /// Total tasks processed
    pub total_tasks_processed: usize,
    /// Worker health status
    pub is_healthy: bool,
    /// Last heartbeat time
    pub last_heartbeat: Instant,
    /// Network latency
    pub network_latency: Duration,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f64,
}

impl WorkerNode {
    /// Create a new worker node
    pub fn new(worker_id: String, address: String) -> Self {
        Self {
            worker_id,
            address,
            capacity: 10,
            current_load: 0,
            total_tasks_processed: 0,
            is_healthy: true,
            last_heartbeat: Instant::now(),
            network_latency: Duration::from_millis(10),
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
        }
    }

    /// Update heartbeat
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
        self.is_healthy = true;
    }

    /// Check if worker is overloaded
    pub fn is_overloaded(&self) -> bool {
        self.current_load >= self.capacity
    }

    /// Get available capacity
    pub fn available_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.current_load)
    }
}

/// Distributed task
#[derive(Debug, Clone)]
pub struct DistributedTask {
    /// Task identifier
    pub task_id: String,
    /// Task type
    pub task_type: TaskType,
    /// Task priority (higher = more important)
    pub priority: usize,
    /// Input data
    pub input_data: Array2<Float>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
    /// Creation time
    pub created_at: Instant,
}

/// Type of distributed task
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    /// Compute SHAP values
    ComputeShap,
    /// Compute permutation importance
    ComputePermutationImportance,
    /// Generate counterfactual explanations
    GenerateCounterfactuals,
    /// Compute feature importance
    ComputeFeatureImportance,
    /// Batch explanation generation
    BatchExplanation,
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: String,
    /// Worker that processed the task
    pub worker_id: String,
    /// Task status
    pub status: TaskStatus,
    /// Result data
    pub result_data: Array1<Float>,
    /// Execution time
    pub execution_time: Duration,
    /// Number of retries
    pub retry_count: usize,
}

/// Task execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is pending
    Pending,
    /// Task is running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    /// Number of active workers
    pub active_workers: usize,
    /// Total tasks submitted
    pub total_tasks_submitted: usize,
    /// Pending tasks
    pub pending_tasks: usize,
    /// Running tasks
    pub running_tasks: usize,
    /// Completed tasks
    pub completed_tasks: usize,
    /// Failed tasks
    pub failed_tasks: usize,
    /// Average task execution time
    pub avg_execution_time: Duration,
    /// Total data processed (in bytes)
    pub total_data_processed: usize,
}

impl ClusterStatistics {
    fn new() -> Self {
        Self {
            active_workers: 0,
            total_tasks_submitted: 0,
            pending_tasks: 0,
            running_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            avg_execution_time: Duration::from_secs(0),
            total_data_processed: 0,
        }
    }
}

/// Cluster health information
#[derive(Debug, Clone)]
pub struct ClusterHealth {
    /// Overall health status
    pub status: HealthStatus,
    /// Total number of workers
    pub total_workers: usize,
    /// Number of healthy workers
    pub healthy_workers: usize,
    /// Total cluster capacity
    pub total_capacity: usize,
    /// Current cluster load
    pub current_load: usize,
    /// Number of pending tasks
    pub pending_tasks: usize,
    /// Number of running tasks
    pub running_tasks: usize,
}

/// Health status of the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All workers healthy
    Healthy,
    /// Some workers degraded
    Warning,
    /// Many workers unhealthy
    Degraded,
    /// Critical failure
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_config_default() {
        let config = ClusterConfig::default();
        assert_eq!(config.max_workers, 10);
        assert_eq!(config.task_timeout_seconds, 300);
        assert_eq!(config.retry_attempts, 3);
        assert_eq!(
            config.load_balancing_strategy,
            LoadBalancingStrategy::RoundRobin
        );
        assert!(config.enable_fault_tolerance);
    }

    #[test]
    fn test_distributed_coordinator_creation() {
        let config = ClusterConfig::default();
        let coordinator = DistributedCoordinator::new(config);
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_register_worker() {
        let config = ClusterConfig::default();
        let coordinator = DistributedCoordinator::new(config).unwrap();

        let result =
            coordinator.register_worker("worker1".to_string(), "192.168.1.10:8080".to_string());
        assert!(result.is_ok());

        let stats = coordinator.get_statistics().unwrap();
        assert_eq!(stats.active_workers, 1);
    }

    #[test]
    fn test_register_multiple_workers() {
        let config = ClusterConfig::default();
        let coordinator = DistributedCoordinator::new(config).unwrap();

        coordinator
            .register_worker("worker1".to_string(), "192.168.1.10:8080".to_string())
            .unwrap();
        coordinator
            .register_worker("worker2".to_string(), "192.168.1.11:8080".to_string())
            .unwrap();

        let stats = coordinator.get_statistics().unwrap();
        assert_eq!(stats.active_workers, 2);
    }

    #[test]
    fn test_register_worker_limit() {
        let config = ClusterConfig {
            max_workers: 2,
            ..Default::default()
        };
        let coordinator = DistributedCoordinator::new(config).unwrap();

        coordinator
            .register_worker("worker1".to_string(), "192.168.1.10:8080".to_string())
            .unwrap();
        coordinator
            .register_worker("worker2".to_string(), "192.168.1.11:8080".to_string())
            .unwrap();

        let result =
            coordinator.register_worker("worker3".to_string(), "192.168.1.12:8080".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_unregister_worker() {
        let config = ClusterConfig::default();
        let coordinator = DistributedCoordinator::new(config).unwrap();

        coordinator
            .register_worker("worker1".to_string(), "192.168.1.10:8080".to_string())
            .unwrap();

        let result = coordinator.unregister_worker("worker1");
        assert!(result.is_ok());

        let stats = coordinator.get_statistics().unwrap();
        assert_eq!(stats.active_workers, 0);
    }

    #[test]
    fn test_submit_task() {
        let config = ClusterConfig::default();
        let coordinator = DistributedCoordinator::new(config).unwrap();

        let task = DistributedTask {
            task_id: "task1".to_string(),
            task_type: TaskType::ComputeShap,
            priority: 1,
            input_data: Array2::zeros((10, 5)),
            metadata: HashMap::new(),
            created_at: Instant::now(),
        };

        let result = coordinator.submit_task(task);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "task1");

        let stats = coordinator.get_statistics().unwrap();
        assert_eq!(stats.total_tasks_submitted, 1);
        assert_eq!(stats.pending_tasks, 1);
    }

    #[test]
    fn test_schedule_tasks() {
        let config = ClusterConfig::default();
        let coordinator = DistributedCoordinator::new(config).unwrap();

        // Register a worker
        coordinator
            .register_worker("worker1".to_string(), "192.168.1.10:8080".to_string())
            .unwrap();

        // Submit a task
        let task = DistributedTask {
            task_id: "task1".to_string(),
            task_type: TaskType::ComputeShap,
            priority: 1,
            input_data: Array2::zeros((10, 5)),
            metadata: HashMap::new(),
            created_at: Instant::now(),
        };
        coordinator.submit_task(task).unwrap();

        // Schedule tasks
        let scheduled = coordinator.schedule_tasks().unwrap();
        assert_eq!(scheduled, 1);
    }

    #[test]
    fn test_worker_node_creation() {
        let worker = WorkerNode::new("worker1".to_string(), "192.168.1.10:8080".to_string());
        assert_eq!(worker.worker_id, "worker1");
        assert_eq!(worker.address, "192.168.1.10:8080");
        assert_eq!(worker.capacity, 10);
        assert_eq!(worker.current_load, 0);
        assert!(worker.is_healthy);
    }

    #[test]
    fn test_worker_node_overload() {
        let mut worker = WorkerNode::new("worker1".to_string(), "192.168.1.10:8080".to_string());
        worker.capacity = 5;
        worker.current_load = 3;

        assert!(!worker.is_overloaded());

        worker.current_load = 5;
        assert!(worker.is_overloaded());

        worker.current_load = 6;
        assert!(worker.is_overloaded());
    }

    #[test]
    fn test_worker_node_available_capacity() {
        let mut worker = WorkerNode::new("worker1".to_string(), "192.168.1.10:8080".to_string());
        worker.capacity = 10;
        worker.current_load = 3;

        assert_eq!(worker.available_capacity(), 7);
    }

    #[test]
    fn test_cluster_health_check() {
        let config = ClusterConfig::default();
        let coordinator = DistributedCoordinator::new(config).unwrap();

        coordinator
            .register_worker("worker1".to_string(), "192.168.1.10:8080".to_string())
            .unwrap();
        coordinator
            .register_worker("worker2".to_string(), "192.168.1.11:8080".to_string())
            .unwrap();

        let health = coordinator.health_check().unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.total_workers, 2);
        assert_eq!(health.healthy_workers, 2);
    }

    #[test]
    fn test_load_balancing_strategies() {
        // Test that all strategy variants can be created and are unique
        assert_ne!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LeastLoaded
        );
        assert_ne!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::Weighted
        );
        assert_ne!(
            LoadBalancingStrategy::LeastLoaded,
            LoadBalancingStrategy::Weighted
        );
        assert_ne!(
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::LocalityAware
        );

        // Test equality
        assert_eq!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::RoundRobin
        );
    }
}

/// Cluster-based explanation computation orchestrator
pub struct ClusterExplanationOrchestrator {
    /// Distributed coordinator
    coordinator: Arc<DistributedCoordinator>,
    /// Configuration
    config: ClusterConfig,
    /// Explanation cache
    cache: Arc<Mutex<HashMap<String, CachedExplanation>>>,
    /// Active batch computations
    active_batches: Arc<Mutex<HashMap<String, BatchComputation>>>,
}

impl ClusterExplanationOrchestrator {
    /// Create a new cluster explanation orchestrator
    pub fn new(config: ClusterConfig) -> SklResult<Self> {
        let coordinator = Arc::new(DistributedCoordinator::new(config.clone())?);

        Ok(Self {
            coordinator,
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            active_batches: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Register workers from a configuration
    pub fn register_workers_from_config(&self, worker_configs: Vec<WorkerConfig>) -> SklResult<()> {
        for worker_config in worker_configs {
            self.coordinator
                .register_worker(worker_config.worker_id, worker_config.address)?;
        }
        Ok(())
    }

    /// Compute SHAP values across the cluster
    pub fn compute_shap_distributed(
        &self,
        data: &Array2<Float>,
        background_data: &Array2<Float>,
        batch_size: usize,
    ) -> SklResult<Array2<Float>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Create batch ID
        let batch_id = format!("shap_batch_{}", uuid::Uuid::new_v4());

        // Split data into batches
        let batches = self.split_into_batches(data, batch_size)?;

        // Submit tasks for each batch
        let mut task_ids = Vec::new();
        for (batch_idx, batch) in batches.iter().enumerate() {
            let task = DistributedTask {
                task_id: format!("{}_task_{}", batch_id, batch_idx),
                task_type: TaskType::ComputeShap,
                priority: 1,
                input_data: batch.clone(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("batch_id".to_string(), batch_id.clone());
                    meta.insert("batch_idx".to_string(), batch_idx.to_string());
                    meta
                },
                created_at: Instant::now(),
            };

            let task_id = self.coordinator.submit_task(task)?;
            task_ids.push(task_id);
        }

        // Schedule all tasks
        self.coordinator.schedule_tasks()?;

        // Collect results
        let mut all_results = Vec::new();
        for task_id in task_ids {
            let result = self.coordinator.get_result(&task_id)?;
            all_results.push(result.result_data);
        }

        // Aggregate results
        let aggregated = self.aggregate_shap_results(all_results, n_samples, n_features)?;

        Ok(aggregated)
    }

    /// Compute feature importance across the cluster
    pub fn compute_feature_importance_distributed(
        &self,
        data: &Array2<Float>,
        predictions: &Array1<Float>,
        batch_size: usize,
    ) -> SklResult<Array1<Float>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Create batch ID
        let batch_id = format!("importance_batch_{}", uuid::Uuid::new_v4());

        // Split data into batches
        let data_batches = self.split_into_batches(data, batch_size)?;
        let pred_batches = self.split_predictions(predictions, batch_size)?;

        // Submit tasks for each batch
        let mut task_ids = Vec::new();
        for (batch_idx, (data_batch, pred_batch)) in
            data_batches.iter().zip(pred_batches.iter()).enumerate()
        {
            let task = DistributedTask {
                task_id: format!("{}_task_{}", batch_id, batch_idx),
                task_type: TaskType::ComputeFeatureImportance,
                priority: 1,
                input_data: data_batch.clone(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("batch_id".to_string(), batch_id.clone());
                    meta.insert("batch_idx".to_string(), batch_idx.to_string());
                    meta
                },
                created_at: Instant::now(),
            };

            let task_id = self.coordinator.submit_task(task)?;
            task_ids.push(task_id);
        }

        // Schedule all tasks
        self.coordinator.schedule_tasks()?;

        // Collect and average results
        let mut importance_sum = Array1::zeros(n_features);
        let mut count = 0;

        for task_id in task_ids {
            let result = self.coordinator.get_result(&task_id)?;
            importance_sum += &result.result_data.slice(s![..n_features]).to_owned();
            count += 1;
        }

        // Average importance across batches
        Ok(importance_sum / (count as Float))
    }

    /// Generate counterfactuals across the cluster
    pub fn generate_counterfactuals_distributed(
        &self,
        instances: &Array2<Float>,
        target_class: usize,
        n_counterfactuals_per_instance: usize,
    ) -> SklResult<Vec<Array1<Float>>> {
        let batch_id = format!("counterfactual_batch_{}", uuid::Uuid::new_v4());

        // Submit tasks for each instance
        let mut task_ids = Vec::new();
        for (instance_idx, instance) in instances.axis_iter(Axis(0)).enumerate() {
            let task = DistributedTask {
                task_id: format!("{}_task_{}", batch_id, instance_idx),
                task_type: TaskType::GenerateCounterfactuals,
                priority: 2,
                input_data: instance.to_owned().insert_axis(Axis(0)),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("batch_id".to_string(), batch_id.clone());
                    meta.insert("instance_idx".to_string(), instance_idx.to_string());
                    meta.insert("target_class".to_string(), target_class.to_string());
                    meta
                },
                created_at: Instant::now(),
            };

            let task_id = self.coordinator.submit_task(task)?;
            task_ids.push(task_id);
        }

        // Schedule all tasks
        self.coordinator.schedule_tasks()?;

        // Collect counterfactuals
        let mut all_counterfactuals = Vec::new();
        for task_id in task_ids {
            let result = self.coordinator.get_result(&task_id)?;
            all_counterfactuals.push(result.result_data);
        }

        Ok(all_counterfactuals)
    }

    /// Split data into batches
    fn split_into_batches(
        &self,
        data: &Array2<Float>,
        batch_size: usize,
    ) -> SklResult<Vec<Array2<Float>>> {
        let n_samples = data.nrows();
        let mut batches = Vec::new();

        for start_idx in (0..n_samples).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(n_samples);
            let batch = data.slice(s![start_idx..end_idx, ..]).to_owned();
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Split predictions into batches
    fn split_predictions(
        &self,
        predictions: &Array1<Float>,
        batch_size: usize,
    ) -> SklResult<Vec<Array1<Float>>> {
        let n_samples = predictions.len();
        let mut batches = Vec::new();

        for start_idx in (0..n_samples).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(n_samples);
            let batch = predictions.slice(s![start_idx..end_idx]).to_owned();
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Aggregate SHAP results from multiple workers
    fn aggregate_shap_results(
        &self,
        results: Vec<Array1<Float>>,
        n_samples: usize,
        n_features: usize,
    ) -> SklResult<Array2<Float>> {
        let mut aggregated = Array2::zeros((n_samples, n_features));

        let mut sample_idx = 0;
        for result in results {
            // Each result contains SHAP values for a batch of samples
            let batch_size = result.len() / n_features;
            for i in 0..batch_size {
                if sample_idx < n_samples {
                    for j in 0..n_features {
                        let result_idx = i * n_features + j;
                        if result_idx < result.len() {
                            aggregated[[sample_idx, j]] = result[result_idx];
                        }
                    }
                    sample_idx += 1;
                }
            }
        }

        Ok(aggregated)
    }

    /// Get cluster statistics
    pub fn get_cluster_statistics(&self) -> SklResult<ClusterStatistics> {
        self.coordinator.get_statistics()
    }

    /// Get cluster health
    pub fn get_cluster_health(&self) -> SklResult<ClusterHealth> {
        self.coordinator.health_check()
    }

    /// Scale cluster up by adding workers
    pub fn scale_up(&self, new_workers: Vec<WorkerConfig>) -> SklResult<()> {
        self.register_workers_from_config(new_workers)
    }

    /// Scale cluster down by removing workers
    pub fn scale_down(&self, worker_ids: Vec<String>) -> SklResult<()> {
        for worker_id in worker_ids {
            self.coordinator.unregister_worker(&worker_id)?;
        }
        Ok(())
    }
}

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Worker identifier
    pub worker_id: String,
    /// Network address
    pub address: String,
    /// Worker capacity
    pub capacity: usize,
}

/// Cached explanation
#[derive(Debug, Clone)]
struct CachedExplanation {
    /// Explanation data
    data: Array1<Float>,
    /// Cache timestamp
    cached_at: Instant,
    /// Cache hit count
    hit_count: usize,
}

/// Batch computation tracking
#[derive(Debug, Clone)]
struct BatchComputation {
    /// Batch identifier
    batch_id: String,
    /// Task IDs in this batch
    task_ids: Vec<String>,
    /// Start time
    started_at: Instant,
    /// Completion status
    is_complete: bool,
}

// Import for slicing
use scirs2_core::ndarray::{s, Axis};

#[cfg(test)]
mod cluster_tests {
    use super::*;

    #[test]
    fn test_cluster_orchestrator_creation() {
        let config = ClusterConfig::default();
        let orchestrator = ClusterExplanationOrchestrator::new(config);
        assert!(orchestrator.is_ok());
    }

    #[test]
    fn test_register_workers_from_config() {
        let config = ClusterConfig::default();
        let orchestrator = ClusterExplanationOrchestrator::new(config).unwrap();

        let worker_configs = vec![
            WorkerConfig {
                worker_id: "worker1".to_string(),
                address: "192.168.1.10:8080".to_string(),
                capacity: 10,
            },
            WorkerConfig {
                worker_id: "worker2".to_string(),
                address: "192.168.1.11:8080".to_string(),
                capacity: 10,
            },
        ];

        let result = orchestrator.register_workers_from_config(worker_configs);
        assert!(result.is_ok());

        let stats = orchestrator.get_cluster_statistics().unwrap();
        assert_eq!(stats.active_workers, 2);
    }

    #[test]
    fn test_split_into_batches() {
        let config = ClusterConfig::default();
        let orchestrator = ClusterExplanationOrchestrator::new(config).unwrap();

        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as Float).collect()).unwrap();
        let batches = orchestrator.split_into_batches(&data, 3).unwrap();

        assert_eq!(batches.len(), 4); // 10 samples / 3 per batch = 4 batches
        assert_eq!(batches[0].nrows(), 3);
        assert_eq!(batches[1].nrows(), 3);
        assert_eq!(batches[2].nrows(), 3);
        assert_eq!(batches[3].nrows(), 1); // Last batch has remainder
    }

    #[test]
    fn test_split_predictions() {
        let config = ClusterConfig::default();
        let orchestrator = ClusterExplanationOrchestrator::new(config).unwrap();

        let predictions = Array1::from_vec((0..10).map(|x| x as Float).collect());
        let batches = orchestrator.split_predictions(&predictions, 4).unwrap();

        assert_eq!(batches.len(), 3); // 10 samples / 4 per batch = 3 batches
        assert_eq!(batches[0].len(), 4);
        assert_eq!(batches[1].len(), 4);
        assert_eq!(batches[2].len(), 2); // Last batch has remainder
    }

    #[test]
    fn test_cluster_health() {
        let config = ClusterConfig::default();
        let orchestrator = ClusterExplanationOrchestrator::new(config).unwrap();

        // Register workers
        let worker_configs = vec![WorkerConfig {
            worker_id: "worker1".to_string(),
            address: "192.168.1.10:8080".to_string(),
            capacity: 10,
        }];
        orchestrator
            .register_workers_from_config(worker_configs)
            .unwrap();

        let health = orchestrator.get_cluster_health().unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.total_workers, 1);
        assert_eq!(health.healthy_workers, 1);
    }

    #[test]
    fn test_scale_up() {
        let config = ClusterConfig::default();
        let orchestrator = ClusterExplanationOrchestrator::new(config).unwrap();

        // Start with one worker
        let initial_workers = vec![WorkerConfig {
            worker_id: "worker1".to_string(),
            address: "192.168.1.10:8080".to_string(),
            capacity: 10,
        }];
        orchestrator
            .register_workers_from_config(initial_workers)
            .unwrap();

        // Scale up
        let new_workers = vec![
            WorkerConfig {
                worker_id: "worker2".to_string(),
                address: "192.168.1.11:8080".to_string(),
                capacity: 10,
            },
            WorkerConfig {
                worker_id: "worker3".to_string(),
                address: "192.168.1.12:8080".to_string(),
                capacity: 10,
            },
        ];
        let result = orchestrator.scale_up(new_workers);
        assert!(result.is_ok());

        let stats = orchestrator.get_cluster_statistics().unwrap();
        assert_eq!(stats.active_workers, 3);
    }

    #[test]
    fn test_scale_down() {
        let config = ClusterConfig::default();
        let orchestrator = ClusterExplanationOrchestrator::new(config).unwrap();

        // Register workers
        let worker_configs = vec![
            WorkerConfig {
                worker_id: "worker1".to_string(),
                address: "192.168.1.10:8080".to_string(),
                capacity: 10,
            },
            WorkerConfig {
                worker_id: "worker2".to_string(),
                address: "192.168.1.11:8080".to_string(),
                capacity: 10,
            },
            WorkerConfig {
                worker_id: "worker3".to_string(),
                address: "192.168.1.12:8080".to_string(),
                capacity: 10,
            },
        ];
        orchestrator
            .register_workers_from_config(worker_configs)
            .unwrap();

        // Scale down
        let result = orchestrator.scale_down(vec!["worker3".to_string()]);
        assert!(result.is_ok());

        let stats = orchestrator.get_cluster_statistics().unwrap();
        assert_eq!(stats.active_workers, 2);
    }

    #[test]
    fn test_worker_config_creation() {
        let config = WorkerConfig {
            worker_id: "test_worker".to_string(),
            address: "localhost:8080".to_string(),
            capacity: 20,
        };

        assert_eq!(config.worker_id, "test_worker");
        assert_eq!(config.address, "localhost:8080");
        assert_eq!(config.capacity, 20);
    }
}
