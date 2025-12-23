//! Distributed pipeline execution components
//!
//! This module provides distributed execution capabilities including cluster management,
//! fault tolerance, load balancing, and MapReduce-style operations.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

use crate::{PipelinePredictor, PipelineStep};

/// Distributed node identifier
pub type NodeId = String;

/// Distributed task identifier
pub type TaskId = String;

/// Cluster node information
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node identifier
    pub id: NodeId,
    /// Node address
    pub address: SocketAddr,
    /// Node status
    pub status: NodeStatus,
    /// Available resources
    pub resources: NodeResources,
    /// Current load
    pub load: NodeLoad,
    /// Heartbeat timestamp
    pub last_heartbeat: SystemTime,
    /// Node metadata
    pub metadata: HashMap<String, String>,
}

/// Node status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    /// Node is healthy and available
    Healthy,
    /// Node is under heavy load but responsive
    Stressed,
    /// Node is temporarily unavailable
    Unavailable,
    /// Node has failed
    Failed,
    /// Node is shutting down
    ShuttingDown,
}

/// Node resource specification
#[derive(Debug, Clone)]
pub struct NodeResources {
    /// Available CPU cores
    pub cpu_cores: u32,
    /// Available memory in MB
    pub memory_mb: u64,
    /// Available disk space in MB
    pub disk_mb: u64,
    /// GPU availability
    pub gpu_count: u32,
    /// Network bandwidth in Mbps
    pub network_bandwidth: u32,
}

/// Current node load metrics
#[derive(Debug, Clone)]
pub struct NodeLoad {
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_utilization: f64,
    /// Disk utilization (0.0 - 1.0)
    pub disk_utilization: f64,
    /// Network utilization (0.0 - 1.0)
    pub network_utilization: f64,
    /// Active task count
    pub active_tasks: usize,
}

impl Default for NodeLoad {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            disk_utilization: 0.0,
            network_utilization: 0.0,
            active_tasks: 0,
        }
    }
}

/// Distributed task specification
#[derive(Debug)]
pub struct DistributedTask {
    /// Task identifier
    pub id: TaskId,
    /// Task name
    pub name: String,
    /// Pipeline component to execute
    pub component: Box<dyn PipelineStep>,
    /// Input data shards
    pub input_shards: Vec<DataShard>,
    /// Task dependencies
    pub dependencies: Vec<TaskId>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Task configuration
    pub config: TaskConfig,
    /// Task metadata
    pub metadata: HashMap<String, String>,
}

/// Data shard for distributed processing
#[derive(Debug, Clone)]
pub struct DataShard {
    /// Shard identifier
    pub id: String,
    /// Data content
    pub data: Array2<f64>,
    /// Target values (optional)
    pub targets: Option<Array1<f64>>,
    /// Shard metadata
    pub metadata: HashMap<String, String>,
    /// Source node
    pub source_node: Option<NodeId>,
}

/// Resource requirements for tasks
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Required CPU cores
    pub cpu_cores: u32,
    /// Required memory in MB
    pub memory_mb: u64,
    /// Required disk space in MB
    pub disk_mb: u64,
    /// GPU requirement
    pub gpu_required: bool,
    /// Estimated execution time
    pub estimated_duration: Duration,
    /// Priority level
    pub priority: TaskPriority,
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low
    Low,
    /// Normal
    Normal,
    /// High
    High,
    /// Critical
    Critical,
}

/// Task execution configuration
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Timeout duration
    pub timeout: Duration,
    /// Failure tolerance
    pub failure_tolerance: FailureTolerance,
    /// Checkpoint interval
    pub checkpoint_interval: Option<Duration>,
    /// Result persistence
    pub persist_results: bool,
}

/// Failure tolerance strategies
#[derive(Debug, Clone)]
pub enum FailureTolerance {
    /// Fail fast on any error
    FailFast,
    /// Retry on specific node
    RetryOnNode { max_retries: usize },
    /// Migrate to different node
    MigrateNode,
    /// Skip failed shard
    SkipFailed,
    /// Use fallback computation
    Fallback {
        fallback_fn: fn(&DataShard) -> SklResult<Array2<f64>>,
    },
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: TaskId,
    /// Execution status
    pub status: TaskStatus,
    /// Result data
    pub result: Option<Array2<f64>>,
    /// Error information
    pub error: Option<SklearsError>,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Executed on node
    pub node_id: NodeId,
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
    /// Retrying
    Retrying,
    /// Cancelled
    Cancelled,
}

/// Task execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Start time
    pub start_time: SystemTime,
    /// End time
    pub end_time: Option<SystemTime>,
    /// Execution duration
    pub duration: Option<Duration>,
    /// Resource usage
    pub resource_usage: NodeLoad,
    /// Data transfer metrics
    pub data_transfer: DataTransferMetrics,
}

/// Data transfer metrics
#[derive(Debug, Clone)]
pub struct DataTransferMetrics {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Transfer duration
    pub transfer_time: Duration,
    /// Network errors
    pub network_errors: usize,
}

/// Distributed cluster manager
#[derive(Debug)]
pub struct ClusterManager {
    /// Available cluster nodes
    nodes: Arc<RwLock<HashMap<NodeId, ClusterNode>>>,
    /// Active tasks
    active_tasks: Arc<Mutex<HashMap<TaskId, DistributedTask>>>,
    /// Task results
    task_results: Arc<Mutex<HashMap<TaskId, TaskResult>>>,
    /// Load balancer
    load_balancer: LoadBalancer,
    /// Fault detector
    fault_detector: FaultDetector,
    /// Cluster configuration
    config: ClusterConfig,
}

/// Cluster configuration
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Node failure timeout
    pub failure_timeout: Duration,
    /// Max concurrent tasks per node
    pub max_tasks_per_node: usize,
    /// Data replication factor
    pub replication_factor: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            failure_timeout: Duration::from_secs(30),
            max_tasks_per_node: 10,
            replication_factor: 2,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least loaded node
    LeastLoaded,
    /// Random assignment
    Random,
    /// Locality-aware (prefer nodes with data)
    LocalityAware,
    /// Custom balancing function
    Custom {
        balance_fn: fn(&[ClusterNode], &ResourceRequirements) -> Option<NodeId>,
    },
}

/// Load balancer component
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    round_robin_index: Mutex<usize>,
    node_assignments: Arc<Mutex<HashMap<TaskId, NodeId>>>,
}

impl LoadBalancer {
    /// Create a new load balancer
    #[must_use]
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            round_robin_index: Mutex::new(0),
            node_assignments: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Select a node for task execution
    pub fn select_node(
        &self,
        nodes: &[ClusterNode],
        requirements: &ResourceRequirements,
    ) -> SklResult<NodeId> {
        let available_nodes: Vec<_> = nodes
            .iter()
            .filter(|node| {
                node.status == NodeStatus::Healthy
                    && self.can_satisfy_requirements(node, requirements)
            })
            .collect();

        if available_nodes.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No available nodes satisfy requirements".to_string(),
            ));
        }

        match &self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut index = self.round_robin_index.lock().unwrap();
                let selected = &available_nodes[*index % available_nodes.len()];
                *index = (*index + 1) % available_nodes.len();
                Ok(selected.id.clone())
            }
            LoadBalancingStrategy::LeastLoaded => {
                let least_loaded = available_nodes
                    .iter()
                    .min_by_key(|node| {
                        (node.load.cpu_utilization * 100.0) as u32 + node.load.active_tasks as u32
                    })
                    .unwrap();
                Ok(least_loaded.id.clone())
            }
            LoadBalancingStrategy::Random => {
                use scirs2_core::random::thread_rng;
                let mut rng = thread_rng();
                let selected = &available_nodes[rng.gen_range(0..available_nodes.len())];
                Ok(selected.id.clone())
            }
            LoadBalancingStrategy::LocalityAware => {
                // Simplified: prefer first available node for now
                Ok(available_nodes[0].id.clone())
            }
            LoadBalancingStrategy::Custom { balance_fn } => {
                let nodes_vec: Vec<ClusterNode> = available_nodes.into_iter().cloned().collect();
                balance_fn(&nodes_vec, requirements).ok_or_else(|| {
                    SklearsError::InvalidInput("Custom balancer failed to select node".to_string())
                })
            }
        }
    }

    /// Check if node can satisfy resource requirements
    fn can_satisfy_requirements(
        &self,
        node: &ClusterNode,
        requirements: &ResourceRequirements,
    ) -> bool {
        node.resources.cpu_cores >= requirements.cpu_cores
            && node.resources.memory_mb >= requirements.memory_mb
            && node.resources.disk_mb >= requirements.disk_mb
            && (!requirements.gpu_required || node.resources.gpu_count > 0)
            && node.load.active_tasks < 10 // Max tasks per node
    }
}

/// Fault detection and recovery
#[derive(Debug)]
pub struct FaultDetector {
    /// Node failure history
    failure_history: Arc<Mutex<HashMap<NodeId, Vec<SystemTime>>>>,
    /// Recovery strategies
    recovery_strategies: HashMap<String, RecoveryStrategy>,
}

/// Recovery strategies for different failure types
#[derive(Debug)]
pub enum RecoveryStrategy {
    /// Restart task on same node
    RestartSameNode,
    /// Migrate task to different node
    MigrateTask,
    /// Replicate task on multiple nodes
    ReplicateTask { replicas: usize },
    /// Use cached results
    UseCachedResults,
    /// Skip failed task
    SkipTask,
}

impl Default for FaultDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FaultDetector {
    /// Create a new fault detector
    #[must_use]
    pub fn new() -> Self {
        let mut recovery_strategies = HashMap::new();
        recovery_strategies.insert("node_failure".to_string(), RecoveryStrategy::MigrateTask);
        recovery_strategies.insert(
            "task_failure".to_string(),
            RecoveryStrategy::RestartSameNode,
        );
        recovery_strategies.insert(
            "network_partition".to_string(),
            RecoveryStrategy::ReplicateTask { replicas: 2 },
        );

        Self {
            failure_history: Arc::new(Mutex::new(HashMap::new())),
            recovery_strategies,
        }
    }

    /// Detect if a node has failed
    #[must_use]
    pub fn detect_node_failure(&self, node: &ClusterNode, timeout: Duration) -> bool {
        node.last_heartbeat.elapsed().unwrap_or(Duration::MAX) > timeout
    }

    /// Record a failure event
    pub fn record_failure(&self, node_id: &NodeId) {
        let mut history = self.failure_history.lock().unwrap();
        history
            .entry(node_id.clone())
            .or_default()
            .push(SystemTime::now());
    }

    /// Get recovery strategy for failure type
    #[must_use]
    pub fn get_recovery_strategy(&self, failure_type: &str) -> Option<&RecoveryStrategy> {
        self.recovery_strategies.get(failure_type)
    }
}

impl ClusterManager {
    /// Create a new cluster manager
    #[must_use]
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            active_tasks: Arc::new(Mutex::new(HashMap::new())),
            task_results: Arc::new(Mutex::new(HashMap::new())),
            load_balancer: LoadBalancer::new(config.load_balancing.clone()),
            fault_detector: FaultDetector::new(),
            config,
        }
    }

    /// Add a node to the cluster
    pub fn add_node(&self, node: ClusterNode) -> SklResult<()> {
        let mut nodes = self.nodes.write().unwrap();
        nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&self, node_id: &NodeId) -> SklResult<()> {
        let mut nodes = self.nodes.write().unwrap();
        nodes.remove(node_id);
        Ok(())
    }

    /// Submit a distributed task
    pub fn submit_task(&self, task: DistributedTask) -> SklResult<TaskId> {
        let task_id = task.id.clone();

        // Select node for execution
        let nodes = self.nodes.read().unwrap();
        let available_nodes: Vec<ClusterNode> = nodes.values().cloned().collect();
        drop(nodes);

        let selected_node = self
            .load_balancer
            .select_node(&available_nodes, &task.resource_requirements)?;

        // Record task
        let mut active_tasks = self.active_tasks.lock().unwrap();
        active_tasks.insert(task_id.clone(), task);
        drop(active_tasks);

        // Execute task (simplified - in real implementation this would be async)
        self.execute_task_on_node(&task_id, &selected_node)?;

        Ok(task_id)
    }

    /// Execute a task on a specific node
    fn execute_task_on_node(&self, task_id: &TaskId, node_id: &NodeId) -> SklResult<()> {
        let active_tasks = self.active_tasks.lock().unwrap();
        let task = active_tasks
            .get(task_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Task {task_id} not found")))?;

        let start_time = SystemTime::now();
        let mut metrics = ExecutionMetrics {
            start_time,
            end_time: None,
            duration: None,
            resource_usage: NodeLoad::default(),
            data_transfer: DataTransferMetrics {
                bytes_sent: 0,
                bytes_received: 0,
                transfer_time: Duration::ZERO,
                network_errors: 0,
            },
        };

        // Simulate task execution
        let result = self.execute_pipeline_component(&task.component, &task.input_shards);

        let end_time = SystemTime::now();
        metrics.end_time = Some(end_time);
        metrics.duration = start_time.elapsed().ok();

        // Store result
        let (result_data, error_info) = match result {
            Ok(data) => (Some(data), None),
            Err(e) => (None, Some(e)),
        };

        let task_result = TaskResult {
            task_id: task_id.clone(),
            status: if result_data.is_some() {
                TaskStatus::Completed
            } else {
                TaskStatus::Failed
            },
            result: result_data,
            error: error_info,
            metrics,
            node_id: node_id.clone(),
        };

        let mut results = self.task_results.lock().unwrap();
        results.insert(task_id.clone(), task_result);

        Ok(())
    }

    /// Execute pipeline component on data shards
    fn execute_pipeline_component(
        &self,
        component: &Box<dyn PipelineStep>,
        shards: &[DataShard],
    ) -> SklResult<Array2<f64>> {
        let mut all_results = Vec::new();

        for shard in shards {
            let mapped_data = shard.data.view().mapv(|v| v as Float);
            let result = component.transform(&mapped_data.view())?;
            all_results.push(result);
        }

        // Concatenate results
        if all_results.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        let total_rows: usize = all_results
            .iter()
            .map(scirs2_core::ndarray::ArrayBase::nrows)
            .sum();
        let n_cols = all_results[0].ncols();

        let mut concatenated = Array2::zeros((total_rows, n_cols));
        let mut row_idx = 0;

        for result in all_results {
            let end_idx = row_idx + result.nrows();
            concatenated
                .slice_mut(s![row_idx..end_idx, ..])
                .assign(&result);
            row_idx = end_idx;
        }

        Ok(concatenated)
    }

    /// Get task result
    pub fn get_task_result(&self, task_id: &TaskId) -> Option<TaskResult> {
        let results = self.task_results.lock().unwrap();
        results.get(task_id).cloned()
    }

    /// Get cluster status
    pub fn cluster_status(&self) -> ClusterStatus {
        let nodes = self.nodes.read().unwrap();
        let active_tasks = self.active_tasks.lock().unwrap();
        let task_results = self.task_results.lock().unwrap();

        let healthy_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();
        let total_nodes = nodes.len();
        let pending_tasks = active_tasks.len();
        let completed_tasks = task_results
            .values()
            .filter(|r| r.status == TaskStatus::Completed)
            .count();
        let failed_tasks = task_results
            .values()
            .filter(|r| r.status == TaskStatus::Failed)
            .count();

        /// ClusterStatus
        ClusterStatus {
            total_nodes,
            healthy_nodes,
            pending_tasks,
            completed_tasks,
            failed_tasks,
            cluster_load: self.calculate_cluster_load(&nodes),
        }
    }

    /// Calculate overall cluster load
    fn calculate_cluster_load(&self, nodes: &HashMap<NodeId, ClusterNode>) -> f64 {
        if nodes.is_empty() {
            return 0.0;
        }

        let total_load: f64 = nodes.values().map(|node| node.load.cpu_utilization).sum();

        total_load / nodes.len() as f64
    }

    /// Start health monitoring
    pub fn start_health_monitoring(&self) -> JoinHandle<()> {
        let nodes = Arc::clone(&self.nodes);
        let fault_detector = FaultDetector::new();
        let heartbeat_interval = self.config.heartbeat_interval;
        let failure_timeout = self.config.failure_timeout;

        thread::spawn(move || {
            loop {
                thread::sleep(heartbeat_interval);

                let mut nodes_guard = nodes.write().unwrap();
                let mut failed_nodes = Vec::new();

                for (node_id, node) in nodes_guard.iter_mut() {
                    if fault_detector.detect_node_failure(node, failure_timeout) {
                        node.status = NodeStatus::Failed;
                        failed_nodes.push(node_id.clone());
                        fault_detector.record_failure(node_id);
                    }
                }

                drop(nodes_guard);

                // Handle failed nodes (simplified)
                for failed_node in failed_nodes {
                    println!("Node {failed_node} has failed");
                }
            }
        })
    }
}

/// Cluster status information
#[derive(Debug, Clone)]
pub struct ClusterStatus {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Number of pending tasks
    pub pending_tasks: usize,
    /// Number of completed tasks
    pub completed_tasks: usize,
    /// Number of failed tasks
    pub failed_tasks: usize,
    /// Overall cluster load (0.0 - 1.0)
    pub cluster_load: f64,
}

/// MapReduce-style distributed pipeline
#[derive(Debug)]
pub struct MapReducePipeline<S = Untrained> {
    state: S,
    mapper: Option<Box<dyn PipelineStep>>,
    reducer: Option<Box<dyn PipelineStep>>,
    cluster_manager: Arc<ClusterManager>,
    partitioning_strategy: PartitioningStrategy,
    map_tasks: Vec<TaskId>,
    reduce_tasks: Vec<TaskId>,
}

/// Data partitioning strategies
#[derive(Debug)]
pub enum PartitioningStrategy {
    /// Equal-sized partitions
    EqualSize { partition_size: usize },
    /// Hash-based partitioning
    HashBased { num_partitions: usize },
    /// Range-based partitioning
    RangeBased { ranges: Vec<(f64, f64)> },
    /// Custom partitioning function
    Custom {
        partition_fn: fn(&Array2<f64>) -> Vec<DataShard>,
    },
}

/// Trained state for `MapReduce` pipeline
#[derive(Debug)]
pub struct MapReducePipelineTrained {
    fitted_mapper: Box<dyn PipelineStep>,
    fitted_reducer: Box<dyn PipelineStep>,
    cluster_manager: Arc<ClusterManager>,
    partitioning_strategy: PartitioningStrategy,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl MapReducePipeline<Untrained> {
    /// Create a new `MapReduce` pipeline
    pub fn new(
        mapper: Box<dyn PipelineStep>,
        reducer: Box<dyn PipelineStep>,
        cluster_manager: Arc<ClusterManager>,
    ) -> Self {
        Self {
            state: Untrained,
            mapper: Some(mapper),
            reducer: Some(reducer),
            cluster_manager,
            partitioning_strategy: PartitioningStrategy::EqualSize {
                partition_size: 1000,
            },
            map_tasks: Vec::new(),
            reduce_tasks: Vec::new(),
        }
    }

    /// Set partitioning strategy
    #[must_use]
    pub fn partitioning_strategy(mut self, strategy: PartitioningStrategy) -> Self {
        self.partitioning_strategy = strategy;
        self
    }
}

impl Estimator for MapReducePipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for MapReducePipeline<Untrained> {
    type Fitted = MapReducePipeline<MapReducePipelineTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut mapper = self
            .mapper
            .ok_or_else(|| SklearsError::InvalidInput("No mapper provided".to_string()))?;

        let mut reducer = self
            .reducer
            .ok_or_else(|| SklearsError::InvalidInput("No reducer provided".to_string()))?;

        // Fit mapper and reducer on a sample of data
        mapper.fit(x, y.as_ref().copied())?;
        reducer.fit(x, y.as_ref().copied())?;

        Ok(MapReducePipeline {
            state: MapReducePipelineTrained {
                fitted_mapper: mapper,
                fitted_reducer: reducer,
                cluster_manager: self.cluster_manager,
                partitioning_strategy: self.partitioning_strategy,
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            mapper: None,
            reducer: None,
            cluster_manager: Arc::new(ClusterManager::new(ClusterConfig::default())),
            partitioning_strategy: PartitioningStrategy::EqualSize {
                partition_size: 1000,
            },
            map_tasks: Vec::new(),
            reduce_tasks: Vec::new(),
        })
    }
}

impl MapReducePipeline<MapReducePipelineTrained> {
    /// Execute `MapReduce` operation
    pub fn map_reduce(&mut self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // Phase 1: Partition data
        let partitions = self.partition_data(x)?;

        // Phase 2: Submit map tasks
        let mut map_task_ids = Vec::new();
        for (i, partition) in partitions.into_iter().enumerate() {
            let map_task = DistributedTask {
                id: format!("map_task_{i}"),
                name: format!("Map Task {i}"),
                component: self.state.fitted_mapper.clone_step(),
                input_shards: vec![partition],
                dependencies: Vec::new(),
                resource_requirements: ResourceRequirements {
                    cpu_cores: 1,
                    memory_mb: 512,
                    disk_mb: 100,
                    gpu_required: false,
                    estimated_duration: Duration::from_secs(60),
                    priority: TaskPriority::Normal,
                },
                config: TaskConfig {
                    max_retries: 3,
                    timeout: Duration::from_secs(300),
                    failure_tolerance: FailureTolerance::RetryOnNode { max_retries: 2 },
                    checkpoint_interval: None,
                    persist_results: true,
                },
                metadata: HashMap::new(),
            };

            let task_id = self.state.cluster_manager.submit_task(map_task)?;
            map_task_ids.push(task_id);
        }

        // Phase 3: Wait for map tasks to complete and collect results
        let map_results = self.wait_for_tasks(&map_task_ids)?;

        // Phase 4: Submit reduce task
        let reduce_shard = DataShard {
            id: "reduce_input".to_string(),
            data: self.combine_map_results(map_results)?,
            targets: None,
            metadata: HashMap::new(),
            source_node: None,
        };

        let reduce_task = DistributedTask {
            id: "reduce_task".to_string(),
            name: "Reduce Task".to_string(),
            component: self.state.fitted_reducer.clone_step(),
            input_shards: vec![reduce_shard],
            dependencies: map_task_ids,
            resource_requirements: ResourceRequirements {
                cpu_cores: 2,
                memory_mb: 1024,
                disk_mb: 200,
                gpu_required: false,
                estimated_duration: Duration::from_secs(120),
                priority: TaskPriority::High,
            },
            config: TaskConfig {
                max_retries: 3,
                timeout: Duration::from_secs(600),
                failure_tolerance: FailureTolerance::RetryOnNode { max_retries: 2 },
                checkpoint_interval: None,
                persist_results: true,
            },
            metadata: HashMap::new(),
        };

        let reduce_task_id = self.state.cluster_manager.submit_task(reduce_task)?;

        // Phase 5: Wait for reduce task and return result
        let reduce_results = self.wait_for_tasks(&[reduce_task_id])?;

        if let Some(result) = reduce_results.into_iter().next() {
            Ok(result)
        } else {
            Err(SklearsError::InvalidData {
                reason: "Reduce task produced no result".to_string(),
            })
        }
    }

    /// Partition input data
    fn partition_data(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<DataShard>> {
        match &self.state.partitioning_strategy {
            PartitioningStrategy::EqualSize { partition_size } => {
                let mut partitions = Vec::new();
                let n_rows = x.nrows();

                for (i, chunk_start) in (0..n_rows).step_by(*partition_size).enumerate() {
                    let chunk_end = std::cmp::min(chunk_start + partition_size, n_rows);
                    let chunk = x.slice(s![chunk_start..chunk_end, ..]).to_owned();

                    let shard = DataShard {
                        id: format!("partition_{i}"),
                        data: chunk.mapv(|v| v),
                        targets: None,
                        metadata: HashMap::new(),
                        source_node: None,
                    };

                    partitions.push(shard);
                }

                Ok(partitions)
            }
            PartitioningStrategy::HashBased { num_partitions } => {
                // Simplified hash-based partitioning
                let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); *num_partitions];

                for i in 0..x.nrows() {
                    let hash = i % num_partitions; // Simplified hash
                    partitions[hash].push(i);
                }

                let mut shards = Vec::new();
                for (partition_idx, indices) in partitions.into_iter().enumerate() {
                    if !indices.is_empty() {
                        let mut partition_data = Array2::zeros((indices.len(), x.ncols()));
                        for (row_idx, &original_idx) in indices.iter().enumerate() {
                            partition_data
                                .row_mut(row_idx)
                                .assign(&x.row(original_idx).mapv(|v| v));
                        }

                        let shard = DataShard {
                            id: format!("hash_partition_{partition_idx}"),
                            data: partition_data,
                            targets: None,
                            metadata: HashMap::new(),
                            source_node: None,
                        };

                        shards.push(shard);
                    }
                }

                Ok(shards)
            }
            PartitioningStrategy::RangeBased { ranges } => {
                // Simplified range-based partitioning on first feature
                let mut shards = Vec::new();

                for (range_idx, (min_val, max_val)) in ranges.iter().enumerate() {
                    let mut selected_rows = Vec::new();

                    for i in 0..x.nrows() {
                        let feature_val = x[[i, 0]];
                        if feature_val >= *min_val && feature_val < *max_val {
                            selected_rows.push(i);
                        }
                    }

                    if !selected_rows.is_empty() {
                        let mut partition_data = Array2::zeros((selected_rows.len(), x.ncols()));
                        for (row_idx, &original_idx) in selected_rows.iter().enumerate() {
                            partition_data
                                .row_mut(row_idx)
                                .assign(&x.row(original_idx).mapv(|v| v));
                        }

                        let shard = DataShard {
                            id: format!("range_partition_{range_idx}"),
                            data: partition_data,
                            targets: None,
                            metadata: HashMap::new(),
                            source_node: None,
                        };

                        shards.push(shard);
                    }
                }

                Ok(shards)
            }
            PartitioningStrategy::Custom { partition_fn } => Ok(partition_fn(&x.mapv(|v| v))),
        }
    }

    /// Wait for tasks to complete and collect results
    fn wait_for_tasks(&self, task_ids: &[TaskId]) -> SklResult<Vec<Array2<f64>>> {
        let mut results = Vec::new();

        for task_id in task_ids {
            // Poll for task completion (simplified)
            let mut attempts = 0;
            const MAX_ATTEMPTS: usize = 100;

            loop {
                if let Some(task_result) = self.state.cluster_manager.get_task_result(task_id) {
                    match task_result.status {
                        TaskStatus::Completed => {
                            if let Some(result) = task_result.result {
                                results.push(result);
                            }
                            break;
                        }
                        TaskStatus::Failed => {
                            return Err(task_result.error.unwrap_or_else(|| {
                                SklearsError::InvalidData {
                                    reason: format!("Task {task_id} failed"),
                                }
                            }));
                        }
                        _ => {
                            // Task still running
                        }
                    }
                }

                attempts += 1;
                if attempts >= MAX_ATTEMPTS {
                    return Err(SklearsError::InvalidData {
                        reason: format!("Task {task_id} timed out"),
                    });
                }

                thread::sleep(Duration::from_millis(100));
            }
        }

        Ok(results)
    }

    /// Combine map results for reduce phase
    fn combine_map_results(&self, results: Vec<Array2<f64>>) -> SklResult<Array2<f64>> {
        if results.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        let total_rows: usize = results
            .iter()
            .map(scirs2_core::ndarray::ArrayBase::nrows)
            .sum();
        let n_cols = results[0].ncols();

        let mut combined = Array2::zeros((total_rows, n_cols));
        let mut row_idx = 0;

        for result in results {
            let end_idx = row_idx + result.nrows();
            combined.slice_mut(s![row_idx..end_idx, ..]).assign(&result);
            row_idx = end_idx;
        }

        Ok(combined)
    }

    /// Get cluster manager
    #[must_use]
    pub fn cluster_manager(&self) -> &Arc<ClusterManager> {
        &self.state.cluster_manager
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockTransformer;
    use scirs2_core::ndarray::array;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_cluster_node_creation() {
        let node = ClusterNode {
            id: "node1".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            status: NodeStatus::Healthy,
            resources: NodeResources {
                cpu_cores: 4,
                memory_mb: 8192,
                disk_mb: 100000,
                gpu_count: 1,
                network_bandwidth: 1000,
            },
            load: NodeLoad::default(),
            last_heartbeat: SystemTime::now(),
            metadata: HashMap::new(),
        };

        assert_eq!(node.id, "node1");
        assert_eq!(node.status, NodeStatus::Healthy);
        assert_eq!(node.resources.cpu_cores, 4);
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let nodes = vec![
            /// ClusterNode
            ClusterNode {
                id: "node1".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
                status: NodeStatus::Healthy,
                resources: NodeResources {
                    cpu_cores: 4,
                    memory_mb: 8192,
                    disk_mb: 100000,
                    gpu_count: 0,
                    network_bandwidth: 1000,
                },
                load: NodeLoad::default(),
                last_heartbeat: SystemTime::now(),
                metadata: HashMap::new(),
            },
            /// ClusterNode
            ClusterNode {
                id: "node2".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081),
                status: NodeStatus::Healthy,
                resources: NodeResources {
                    cpu_cores: 4,
                    memory_mb: 8192,
                    disk_mb: 100000,
                    gpu_count: 0,
                    network_bandwidth: 1000,
                },
                load: NodeLoad::default(),
                last_heartbeat: SystemTime::now(),
                metadata: HashMap::new(),
            },
        ];

        let requirements = ResourceRequirements {
            cpu_cores: 1,
            memory_mb: 1024,
            disk_mb: 1000,
            gpu_required: false,
            estimated_duration: Duration::from_secs(60),
            priority: TaskPriority::Normal,
        };

        let selected1 = balancer.select_node(&nodes, &requirements).unwrap();
        let selected2 = balancer.select_node(&nodes, &requirements).unwrap();

        assert_ne!(selected1, selected2); // Round robin should alternate
    }

    #[test]
    fn test_cluster_manager() {
        let config = ClusterConfig::default();
        let manager = ClusterManager::new(config);

        let node = ClusterNode {
            id: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            status: NodeStatus::Healthy,
            resources: NodeResources {
                cpu_cores: 4,
                memory_mb: 8192,
                disk_mb: 100000,
                gpu_count: 0,
                network_bandwidth: 1000,
            },
            load: NodeLoad::default(),
            last_heartbeat: SystemTime::now(),
            metadata: HashMap::new(),
        };

        manager.add_node(node).unwrap();

        let status = manager.cluster_status();
        assert_eq!(status.total_nodes, 1);
        assert_eq!(status.healthy_nodes, 1);
    }

    #[test]
    fn test_data_shard_creation() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let targets = array![1.0, 0.0];

        let shard = DataShard {
            id: "test_shard".to_string(),
            data: data.clone(),
            targets: Some(targets.clone()),
            metadata: HashMap::new(),
            source_node: None,
        };

        assert_eq!(shard.id, "test_shard");
        assert_eq!(shard.data, data);
        assert_eq!(shard.targets, Some(targets));
    }

    #[test]
    fn test_mapreduce_pipeline_creation() {
        let mapper = Box::new(MockTransformer::new());
        let reducer = Box::new(MockTransformer::new());
        let cluster_manager = Arc::new(ClusterManager::new(ClusterConfig::default()));

        let pipeline = MapReducePipeline::new(mapper, reducer, cluster_manager);

        assert!(matches!(
            pipeline.partitioning_strategy,
            PartitioningStrategy::EqualSize {
                partition_size: 1000
            }
        ));
    }

    #[test]
    fn test_fault_detector() {
        let detector = FaultDetector::new();

        let node = ClusterNode {
            id: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            status: NodeStatus::Healthy,
            resources: NodeResources {
                cpu_cores: 4,
                memory_mb: 8192,
                disk_mb: 100000,
                gpu_count: 0,
                network_bandwidth: 1000,
            },
            load: NodeLoad::default(),
            last_heartbeat: SystemTime::now() - Duration::from_secs(60),
            metadata: HashMap::new(),
        };

        let is_failed = detector.detect_node_failure(&node, Duration::from_secs(30));
        assert!(is_failed);

        detector.record_failure(&node.id);

        let strategy = detector.get_recovery_strategy("node_failure");
        assert!(matches!(strategy, Some(RecoveryStrategy::MigrateTask)));
    }
}
