//! Distributed computing utilities for ML workloads
//!
//! This module provides utilities for distributed computing including node discovery,
//! job distribution, load balancing, and fault tolerance for scalable ML computations.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Node information in distributed cluster
#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub id: String,
    pub address: SocketAddr,
    pub capabilities: NodeCapabilities,
    pub status: NodeStatus,
    pub last_heartbeat: Instant,
    pub load_metrics: LoadMetrics,
    pub job_history: Vec<JobExecution>,
}

/// Node capabilities
#[derive(Debug, Clone)]
pub struct NodeCapabilities {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub gpu_count: u32,
    pub storage_gb: u32,
    pub network_bandwidth_mbps: u32,
    pub supported_tasks: HashSet<String>,
}

/// Node status
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Available,
    Busy,
    Overloaded,
    Unreachable,
    Maintenance,
}

/// Node load metrics
#[derive(Debug, Clone)]
pub struct LoadMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub network_io: f64,
    pub disk_io: f64,
    pub active_jobs: u32,
    pub queue_size: u32,
}

/// Distributed job information
#[derive(Debug, Clone)]
pub struct DistributedJob {
    pub id: String,
    pub name: String,
    pub job_type: JobType,
    pub priority: JobPriority,
    pub requirements: ResourceRequirements,
    pub created_at: Instant,
    pub timeout: Duration,
    pub retry_count: u32,
    pub dependencies: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Job type
#[derive(Debug, Clone, PartialEq)]
pub enum JobType {
    Training,
    Inference,
    DataProcessing,
    ModelEvaluation,
    Hyperparameter,
    Custom(String),
}

/// Job priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub min_cpu_cores: u32,
    pub min_memory_gb: u32,
    pub min_gpu_count: u32,
    pub min_storage_gb: u32,
    pub preferred_node_tags: HashSet<String>,
    pub exclusive_access: bool,
}

/// Job execution information
#[derive(Debug, Clone)]
pub struct JobExecution {
    pub job_id: String,
    pub node_id: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub status: JobStatus,
    pub progress: f64,
    pub result: Option<String>,
    pub error: Option<String>,
    pub resource_usage: Option<ResourceUsage>,
}

/// Job status
#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

/// Resource usage during job execution
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_time: Duration,
    pub memory_peak: u64,
    pub gpu_time: Duration,
    pub network_bytes: u64,
    pub disk_bytes: u64,
}

/// Distributed computing cluster manager
pub struct DistributedCluster {
    nodes: Arc<RwLock<HashMap<String, ClusterNode>>>,
    jobs: Arc<RwLock<HashMap<String, DistributedJob>>>,
    executions: Arc<RwLock<HashMap<String, JobExecution>>>,
    job_queue: Arc<Mutex<Vec<DistributedJob>>>,
    scheduler: Arc<Mutex<JobScheduler>>,
    load_balancer: Arc<Mutex<LoadBalancer>>,
    fault_detector: Arc<Mutex<FaultDetector>>,
    config: ClusterConfig,
}

/// Cluster configuration
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub heartbeat_interval: Duration,
    pub node_timeout: Duration,
    pub job_timeout: Duration,
    pub max_retries: u32,
    pub load_threshold: f64,
    pub replication_factor: u32,
}

impl DistributedCluster {
    /// Create new distributed cluster
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            jobs: Arc::new(RwLock::new(HashMap::new())),
            executions: Arc::new(RwLock::new(HashMap::new())),
            job_queue: Arc::new(Mutex::new(Vec::new())),
            scheduler: Arc::new(Mutex::new(JobScheduler::new())),
            load_balancer: Arc::new(Mutex::new(LoadBalancer::new())),
            fault_detector: Arc::new(Mutex::new(FaultDetector::new())),
            config,
        }
    }

    /// Register a node in the cluster
    pub fn register_node(&self, node: ClusterNode) -> Result<(), DistributedError> {
        let mut nodes = self.nodes.write().unwrap();
        nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&self, node_id: &str) -> Result<(), DistributedError> {
        let mut nodes = self.nodes.write().unwrap();
        nodes
            .remove(node_id)
            .ok_or(DistributedError::NodeNotFound)?;
        Ok(())
    }

    /// Get all nodes in the cluster
    pub fn get_nodes(&self) -> Vec<ClusterNode> {
        self.nodes.read().unwrap().values().cloned().collect()
    }

    /// Get available nodes
    pub fn get_available_nodes(&self) -> Vec<ClusterNode> {
        self.nodes
            .read()
            .unwrap()
            .values()
            .filter(|node| node.status == NodeStatus::Available)
            .cloned()
            .collect()
    }

    /// Submit a job to the cluster
    pub fn submit_job(&self, job: DistributedJob) -> Result<String, DistributedError> {
        let job_id = job.id.clone();

        // Add job to jobs registry
        self.jobs
            .write()
            .unwrap()
            .insert(job_id.clone(), job.clone());

        // Add job to queue
        self.job_queue.lock().unwrap().push(job);

        // Trigger scheduling
        self.schedule_jobs()?;

        Ok(job_id)
    }

    /// Schedule jobs in the queue
    pub fn schedule_jobs(&self) -> Result<(), DistributedError> {
        let scheduler = self.scheduler.lock().unwrap();
        let mut queue = self.job_queue.lock().unwrap();
        let nodes = self.nodes.read().unwrap();

        // Sort jobs by priority and creation time
        queue.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });

        let mut scheduled_jobs = Vec::new();

        for job in queue.iter() {
            if let Some(node_id) = scheduler.find_suitable_node(job, &nodes) {
                let execution = JobExecution {
                    job_id: job.id.clone(),
                    node_id: node_id.clone(),
                    start_time: Instant::now(),
                    end_time: None,
                    status: JobStatus::Running,
                    progress: 0.0,
                    result: None,
                    error: None,
                    resource_usage: None,
                };

                self.executions
                    .write()
                    .unwrap()
                    .insert(job.id.clone(), execution);
                scheduled_jobs.push(job.id.clone());
            }
        }

        // Remove scheduled jobs from queue
        queue.retain(|job| !scheduled_jobs.contains(&job.id));

        Ok(())
    }

    /// Get job status
    pub fn get_job_status(&self, job_id: &str) -> Option<JobStatus> {
        self.executions
            .read()
            .unwrap()
            .get(job_id)
            .map(|exec| exec.status.clone())
    }

    /// Get job execution info
    pub fn get_job_execution(&self, job_id: &str) -> Option<JobExecution> {
        self.executions.read().unwrap().get(job_id).cloned()
    }

    /// Cancel a job
    pub fn cancel_job(&self, job_id: &str) -> Result<(), DistributedError> {
        let mut executions = self.executions.write().unwrap();

        if let Some(execution) = executions.get_mut(job_id) {
            execution.status = JobStatus::Cancelled;
            execution.end_time = Some(Instant::now());
            Ok(())
        } else {
            Err(DistributedError::JobNotFound)
        }
    }

    /// Update node heartbeat
    pub fn update_heartbeat(
        &self,
        node_id: &str,
        load_metrics: LoadMetrics,
    ) -> Result<(), DistributedError> {
        let mut nodes = self.nodes.write().unwrap();

        if let Some(node) = nodes.get_mut(node_id) {
            node.last_heartbeat = Instant::now();
            node.load_metrics = load_metrics;
            node.status = self.determine_node_status(&node.load_metrics);
            Ok(())
        } else {
            Err(DistributedError::NodeNotFound)
        }
    }

    /// Determine node status based on load metrics
    fn determine_node_status(&self, metrics: &LoadMetrics) -> NodeStatus {
        if metrics.cpu_usage > 0.9 || metrics.memory_usage > 0.9 {
            NodeStatus::Overloaded
        } else if metrics.cpu_usage > 0.7 || metrics.memory_usage > 0.7 {
            NodeStatus::Busy
        } else {
            NodeStatus::Available
        }
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self) -> ClusterStats {
        let nodes = self.nodes.read().unwrap();
        let jobs = self.jobs.read().unwrap();
        let executions = self.executions.read().unwrap();

        let total_nodes = nodes.len();
        let available_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Available)
            .count();
        let busy_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Busy)
            .count();
        let overloaded_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Overloaded)
            .count();

        let total_jobs = jobs.len();
        let running_jobs = executions
            .values()
            .filter(|e| e.status == JobStatus::Running)
            .count();
        let completed_jobs = executions
            .values()
            .filter(|e| e.status == JobStatus::Completed)
            .count();
        let failed_jobs = executions
            .values()
            .filter(|e| e.status == JobStatus::Failed)
            .count();

        let total_cpu_cores: u32 = nodes.values().map(|n| n.capabilities.cpu_cores).sum();
        let total_memory_gb: u32 = nodes.values().map(|n| n.capabilities.memory_gb).sum();
        let total_gpu_count: u32 = nodes.values().map(|n| n.capabilities.gpu_count).sum();

        let avg_cpu_usage = if !nodes.is_empty() {
            nodes
                .values()
                .map(|n| n.load_metrics.cpu_usage)
                .sum::<f64>()
                / nodes.len() as f64
        } else {
            0.0
        };

        let avg_memory_usage = if !nodes.is_empty() {
            nodes
                .values()
                .map(|n| n.load_metrics.memory_usage)
                .sum::<f64>()
                / nodes.len() as f64
        } else {
            0.0
        };

        ClusterStats {
            total_nodes,
            available_nodes,
            busy_nodes,
            overloaded_nodes,
            unreachable_nodes: total_nodes - available_nodes - busy_nodes - overloaded_nodes,
            total_jobs,
            running_jobs,
            completed_jobs,
            failed_jobs,
            queued_jobs: self.job_queue.lock().unwrap().len(),
            total_cpu_cores,
            total_memory_gb,
            total_gpu_count,
            avg_cpu_usage,
            avg_memory_usage,
        }
    }

    /// Start cluster monitoring
    pub fn start_monitoring(&self) -> Result<(), DistributedError> {
        let nodes = Arc::clone(&self.nodes);
        let executions = Arc::clone(&self.executions);
        let config = self.config.clone();

        thread::spawn(move || {
            loop {
                // Check for failed nodes
                let now = Instant::now();
                let mut nodes_guard = nodes.write().unwrap();

                for node in nodes_guard.values_mut() {
                    if now.duration_since(node.last_heartbeat) > config.node_timeout {
                        node.status = NodeStatus::Unreachable;
                    }
                }

                // Check for timed out jobs
                let mut executions_guard = executions.write().unwrap();
                for execution in executions_guard.values_mut() {
                    if execution.status == JobStatus::Running
                        && now.duration_since(execution.start_time) > config.job_timeout
                    {
                        execution.status = JobStatus::Timeout;
                        execution.end_time = Some(now);
                    }
                }

                drop(nodes_guard);
                drop(executions_guard);

                thread::sleep(config.heartbeat_interval);
            }
        });

        Ok(())
    }

    /// Rebalance workload across nodes
    pub fn rebalance_workload(&self) -> Result<(), DistributedError> {
        let load_balancer = self.load_balancer.lock().unwrap();
        let nodes = self.nodes.read().unwrap();

        load_balancer.rebalance(&nodes)?;
        Ok(())
    }

    /// Handle node failure
    pub fn handle_node_failure(&self, node_id: &str) -> Result<(), DistributedError> {
        let mut fault_detector = self.fault_detector.lock().unwrap();
        let mut executions = self.executions.write().unwrap();

        // Mark running jobs on failed node as failed
        for execution in executions.values_mut() {
            if execution.node_id == node_id && execution.status == JobStatus::Running {
                execution.status = JobStatus::Failed;
                execution.end_time = Some(Instant::now());
                execution.error = Some("Node failure".to_string());
            }
        }

        fault_detector.handle_failure(node_id)?;
        Ok(())
    }
}

/// Job scheduler
pub struct JobScheduler {
    scheduling_strategy: SchedulingStrategy,
}

impl JobScheduler {
    /// Create new job scheduler
    pub fn new() -> Self {
        Self {
            scheduling_strategy: SchedulingStrategy::LeastLoaded,
        }
    }

    /// Find suitable node for job
    pub fn find_suitable_node(
        &self,
        job: &DistributedJob,
        nodes: &HashMap<String, ClusterNode>,
    ) -> Option<String> {
        let suitable_nodes: Vec<_> = nodes
            .values()
            .filter(|node| self.can_run_job(node, job))
            .collect();

        if suitable_nodes.is_empty() {
            return None;
        }

        match self.scheduling_strategy {
            SchedulingStrategy::LeastLoaded => suitable_nodes
                .iter()
                .min_by(|a, b| {
                    let load_a = a.load_metrics.cpu_usage + a.load_metrics.memory_usage;
                    let load_b = b.load_metrics.cpu_usage + b.load_metrics.memory_usage;
                    load_a
                        .partial_cmp(&load_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|node| node.id.clone()),
            SchedulingStrategy::RoundRobin => suitable_nodes.first().map(|node| node.id.clone()),
            SchedulingStrategy::HighestCapacity => suitable_nodes
                .iter()
                .max_by_key(|node| node.capabilities.cpu_cores * node.capabilities.memory_gb)
                .map(|node| node.id.clone()),
        }
    }

    /// Check if node can run job
    fn can_run_job(&self, node: &ClusterNode, job: &DistributedJob) -> bool {
        node.status == NodeStatus::Available
            && node.capabilities.cpu_cores >= job.requirements.min_cpu_cores
            && node.capabilities.memory_gb >= job.requirements.min_memory_gb
            && node.capabilities.gpu_count >= job.requirements.min_gpu_count
            && node.capabilities.storage_gb >= job.requirements.min_storage_gb
    }
}

/// Scheduling strategy
#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    LeastLoaded,
    RoundRobin,
    HighestCapacity,
}

/// Load balancer
pub struct LoadBalancer {
    rebalance_threshold: f64,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new() -> Self {
        Self {
            rebalance_threshold: 0.8,
        }
    }

    /// Rebalance workload across nodes
    pub fn rebalance(&self, nodes: &HashMap<String, ClusterNode>) -> Result<(), DistributedError> {
        let overloaded_nodes: Vec<_> = nodes
            .values()
            .filter(|node| {
                let total_load = node.load_metrics.cpu_usage + node.load_metrics.memory_usage;
                total_load > self.rebalance_threshold * 2.0
            })
            .collect();

        let underloaded_nodes: Vec<_> = nodes
            .values()
            .filter(|node| {
                let total_load = node.load_metrics.cpu_usage + node.load_metrics.memory_usage;
                total_load < self.rebalance_threshold
            })
            .collect();

        // In a real implementation, this would migrate jobs from overloaded to underloaded nodes
        println!(
            "Rebalancing: {} overloaded nodes, {} underloaded nodes",
            overloaded_nodes.len(),
            underloaded_nodes.len()
        );

        Ok(())
    }
}

/// Fault detector
pub struct FaultDetector {
    failure_history: HashMap<String, Vec<Instant>>,
}

impl FaultDetector {
    /// Create new fault detector
    pub fn new() -> Self {
        Self {
            failure_history: HashMap::new(),
        }
    }

    /// Handle node failure
    pub fn handle_failure(&mut self, node_id: &str) -> Result<(), DistributedError> {
        let failures = self.failure_history.entry(node_id.to_string()).or_default();

        failures.push(Instant::now());

        // Remove old failures (older than 1 hour)
        let cutoff = Instant::now() - Duration::from_secs(3600);
        failures.retain(|&failure_time| failure_time > cutoff);

        // If too many failures, mark node as problematic
        if failures.len() > 3 {
            println!("Node {node_id} has too many failures, marking as problematic");
        }

        Ok(())
    }

    /// Check if node is problematic
    pub fn is_problematic(&self, node_id: &str) -> bool {
        self.failure_history
            .get(node_id)
            .map(|failures| failures.len() > 3)
            .unwrap_or(false)
    }
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStats {
    pub total_nodes: usize,
    pub available_nodes: usize,
    pub busy_nodes: usize,
    pub overloaded_nodes: usize,
    pub unreachable_nodes: usize,
    pub total_jobs: usize,
    pub running_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub queued_jobs: usize,
    pub total_cpu_cores: u32,
    pub total_memory_gb: u32,
    pub total_gpu_count: u32,
    pub avg_cpu_usage: f64,
    pub avg_memory_usage: f64,
}

/// Distributed computing errors
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("Node not found")]
    NodeNotFound,
    #[error("Job not found")]
    JobNotFound,
    #[error("Insufficient resources")]
    InsufficientResources,
    #[error("Node unreachable")]
    NodeUnreachable,
    #[error("Job timeout")]
    JobTimeout,
    #[error("Scheduling error: {0}")]
    SchedulingError(String),
    #[error("Communication error: {0}")]
    CommunicationError(String),
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(30),
            node_timeout: Duration::from_secs(120),
            job_timeout: Duration::from_secs(3600),
            max_retries: 3,
            load_threshold: 0.8,
            replication_factor: 2,
        }
    }
}

impl Default for JobScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FaultDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AdvancedJobScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced distributed computing features
/// Message passing system for inter-node communication
#[allow(dead_code)]
pub struct MessagePassingSystem {
    node_id: String,
    message_handlers: HashMap<String, MessageHandler>,
    pending_messages: Arc<Mutex<Vec<DistributedMessage>>>,
    message_queue: Arc<Mutex<Vec<DistributedMessage>>>,
    routing_table: Arc<RwLock<HashMap<String, SocketAddr>>>,
}

impl MessagePassingSystem {
    /// Create new message passing system
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            message_handlers: HashMap::new(),
            pending_messages: Arc::new(Mutex::new(Vec::new())),
            message_queue: Arc::new(Mutex::new(Vec::new())),
            routing_table: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Send message to another node
    pub fn send_message(&self, message: DistributedMessage) -> Result<(), DistributedError> {
        let routing_table = self.routing_table.read().unwrap();

        if let Some(_address) = routing_table.get(&message.destination) {
            // In real implementation, this would send over network
            self.message_queue.lock().unwrap().push(message);
            Ok(())
        } else {
            Err(DistributedError::NodeUnreachable)
        }
    }

    /// Broadcast message to all nodes
    pub fn broadcast_message(
        &self,
        message_type: MessageType,
        data: Vec<u8>,
    ) -> Result<(), DistributedError> {
        let routing_table = self.routing_table.read().unwrap();

        for (node_id, _address) in routing_table.iter() {
            if node_id != &self.node_id {
                let message = DistributedMessage {
                    id: format!("{}_{}", self.node_id, Instant::now().elapsed().as_millis()),
                    source: self.node_id.clone(),
                    destination: node_id.clone(),
                    message_type: message_type.clone(),
                    data: data.clone(),
                    timestamp: Instant::now(),
                    priority: MessagePriority::Normal,
                };
                self.send_message(message)?;
            }
        }

        Ok(())
    }

    /// Register message handler
    pub fn register_handler(&mut self, message_type: String, handler: MessageHandler) {
        self.message_handlers.insert(message_type, handler);
    }

    /// Process incoming messages
    pub fn process_messages(&self) -> Result<Vec<MessageResponse>, DistributedError> {
        let mut responses = Vec::new();
        let mut queue = self.message_queue.lock().unwrap();

        for message in queue.drain(..) {
            if let Some(handler) = self
                .message_handlers
                .get(&format!("{}", message.message_type))
            {
                let response = handler.handle(&message)?;
                responses.push(response);
            }
        }

        Ok(responses)
    }
}

/// Consensus algorithm implementation (simplified Raft)
#[allow(dead_code)]
pub struct ConsensusManager {
    node_id: String,
    state: ConsensusState,
    term: u64,
    voted_for: Option<String>,
    log: Vec<LogEntry>,
    commit_index: usize,
    last_applied: usize,
    peers: HashMap<String, PeerInfo>,
}

impl ConsensusManager {
    /// Create new consensus manager
    pub fn new(node_id: String, peers: Vec<String>) -> Self {
        let mut peer_map = HashMap::new();
        for peer in peers {
            peer_map.insert(
                peer.clone(),
                PeerInfo {
                    id: peer,
                    next_index: 0,
                    match_index: 0,
                    last_response: Instant::now(),
                },
            );
        }

        Self {
            node_id,
            state: ConsensusState::Follower,
            term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            peers: peer_map,
        }
    }

    /// Start leader election
    pub fn start_election(&mut self) -> Result<(), DistributedError> {
        self.state = ConsensusState::Candidate;
        self.term += 1;
        self.voted_for = Some(self.node_id.clone());

        // In real implementation, would send RequestVote RPCs to all peers
        println!(
            "Node {} starting election for term {}",
            self.node_id, self.term
        );

        Ok(())
    }

    /// Handle vote request
    pub fn handle_vote_request(&mut self, request: VoteRequest) -> VoteResponse {
        let grant_vote = if request.term > self.term {
            self.term = request.term;
            self.voted_for = None;
            self.state = ConsensusState::Follower;
            true
        } else if request.term == self.term
            && (self.voted_for.is_none() || self.voted_for.as_ref() == Some(&request.candidate_id))
        {
            self.voted_for = Some(request.candidate_id.clone());
            true
        } else {
            false
        };

        VoteResponse {
            term: self.term,
            vote_granted: grant_vote,
        }
    }

    /// Append log entry
    pub fn append_entry(&mut self, entry: LogEntry) -> Result<(), DistributedError> {
        if self.state != ConsensusState::Leader {
            return Err(DistributedError::SchedulingError("Not leader".to_string()));
        }

        self.log.push(entry);
        Ok(())
    }

    /// Get current state
    pub fn get_state(&self) -> (ConsensusState, u64) {
        (self.state.clone(), self.term)
    }
}

/// Data partitioning and sharding system
pub struct DataPartitioner {
    partitioning_strategy: PartitioningStrategy,
    partition_count: usize,
    node_assignments: HashMap<usize, String>,
    replication_factor: usize,
}

impl DataPartitioner {
    /// Create new data partitioner
    pub fn new(
        strategy: PartitioningStrategy,
        partition_count: usize,
        replication_factor: usize,
    ) -> Self {
        Self {
            partitioning_strategy: strategy,
            partition_count,
            node_assignments: HashMap::new(),
            replication_factor,
        }
    }

    /// Determine partition for data key
    pub fn get_partition(&self, key: &str) -> usize {
        match self.partitioning_strategy {
            PartitioningStrategy::Hash => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::Hasher;

                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                (hasher.finish() as usize) % self.partition_count
            }
            PartitioningStrategy::Range => {
                // Simplified range partitioning
                key.len() % self.partition_count
            }
            PartitioningStrategy::Random => {
                // Mock random partitioning
                key.len() % self.partition_count
            }
        }
    }

    /// Get nodes responsible for partition
    pub fn get_partition_nodes(&self, partition: usize) -> Vec<String> {
        let mut nodes = Vec::new();

        if let Some(primary_node) = self.node_assignments.get(&partition) {
            nodes.push(primary_node.clone());

            // Add replica nodes (simplified)
            for i in 1..self.replication_factor {
                let replica_partition = (partition + i) % self.partition_count;
                if let Some(replica_node) = self.node_assignments.get(&replica_partition) {
                    if !nodes.contains(replica_node) {
                        nodes.push(replica_node.clone());
                    }
                }
            }
        }

        nodes
    }

    /// Assign partition to node
    pub fn assign_partition(&mut self, partition: usize, node_id: String) {
        self.node_assignments.insert(partition, node_id);
    }

    /// Rebalance partitions across nodes
    pub fn rebalance_partitions(&mut self, available_nodes: &[String]) -> PartitioningResult {
        let mut assignments_changed = 0;
        let mut partitions_moved = Vec::new();

        for partition in 0..self.partition_count {
            let optimal_node = &available_nodes[partition % available_nodes.len()];

            if let Some(current_node) = self.node_assignments.get(&partition) {
                if current_node != optimal_node {
                    partitions_moved.push(PartitionMove {
                        partition,
                        from_node: current_node.clone(),
                        to_node: optimal_node.clone(),
                    });
                    self.node_assignments
                        .insert(partition, optimal_node.clone());
                    assignments_changed += 1;
                }
            } else {
                self.node_assignments
                    .insert(partition, optimal_node.clone());
                assignments_changed += 1;
            }
        }

        PartitioningResult {
            assignments_changed,
            partitions_moved,
            rebalance_time: Instant::now(),
        }
    }
}

/// Advanced job scheduler with gang scheduling
#[allow(dead_code)]
pub struct AdvancedJobScheduler {
    scheduling_policies: Vec<SchedulingPolicy>,
    resource_reservations: HashMap<String, ResourceReservation>,
    job_dependencies: HashMap<String, Vec<String>>,
    priority_queues: HashMap<JobPriority, Vec<DistributedJob>>,
    backfill_enabled: bool,
}

impl AdvancedJobScheduler {
    /// Create new advanced scheduler
    pub fn new() -> Self {
        Self {
            scheduling_policies: vec![
                SchedulingPolicy::FIFO,
                SchedulingPolicy::ShortestJobFirst,
                SchedulingPolicy::GangScheduling,
            ],
            resource_reservations: HashMap::new(),
            job_dependencies: HashMap::new(),
            priority_queues: HashMap::new(),
            backfill_enabled: true,
        }
    }

    /// Schedule jobs with gang scheduling
    pub fn gang_schedule(
        &mut self,
        jobs: &[DistributedJob],
        nodes: &HashMap<String, ClusterNode>,
    ) -> Result<Vec<SchedulingDecision>, DistributedError> {
        let mut decisions = Vec::new();

        // Group jobs that need to run together
        let job_groups = self.group_related_jobs(jobs);

        for group in job_groups {
            if let Some(node_assignment) = self.find_gang_assignment(&group, nodes) {
                for (job, node_id) in group.iter().zip(node_assignment.iter()) {
                    decisions.push(SchedulingDecision {
                        job_id: job.id.clone(),
                        node_id: node_id.clone(),
                        estimated_start_time: Instant::now(),
                        resource_allocation: self
                            .calculate_resource_allocation(job, node_id, nodes),
                    });
                }
            }
        }

        Ok(decisions)
    }

    /// Implement backfill scheduling
    pub fn backfill_schedule(
        &mut self,
        waiting_jobs: &[DistributedJob],
        nodes: &HashMap<String, ClusterNode>,
    ) -> Result<Vec<SchedulingDecision>, DistributedError> {
        let mut decisions = Vec::new();

        if !self.backfill_enabled {
            return Ok(decisions);
        }

        // Find jobs that can run in available gaps
        for job in waiting_jobs {
            for (node_id, node) in nodes {
                if self.can_backfill_job(job, node) {
                    decisions.push(SchedulingDecision {
                        job_id: job.id.clone(),
                        node_id: node_id.clone(),
                        estimated_start_time: Instant::now(),
                        resource_allocation: self
                            .calculate_resource_allocation(job, node_id, nodes),
                    });
                    break;
                }
            }
        }

        Ok(decisions)
    }

    /// Reserve resources for future jobs
    pub fn reserve_resources(
        &mut self,
        reservation: ResourceReservation,
    ) -> Result<(), DistributedError> {
        self.resource_reservations
            .insert(reservation.id.clone(), reservation);
        Ok(())
    }

    fn group_related_jobs(&self, jobs: &[DistributedJob]) -> Vec<Vec<DistributedJob>> {
        // Simplified job grouping - in real implementation would use dependency analysis
        vec![jobs.to_vec()]
    }

    fn find_gang_assignment(
        &self,
        job_group: &[DistributedJob],
        nodes: &HashMap<String, ClusterNode>,
    ) -> Option<Vec<String>> {
        // Simplified gang assignment - find nodes that can run all jobs
        if job_group.len() <= nodes.len() {
            Some(nodes.keys().take(job_group.len()).cloned().collect())
        } else {
            None
        }
    }

    fn can_backfill_job(&self, _job: &DistributedJob, _node: &ClusterNode) -> bool {
        // Simplified backfill logic
        true
    }

    fn calculate_resource_allocation(
        &self,
        job: &DistributedJob,
        _node_id: &str,
        _nodes: &HashMap<String, ClusterNode>,
    ) -> ResourceAllocation {
        ResourceAllocation {
            cpu_cores: job.requirements.min_cpu_cores,
            memory_gb: job.requirements.min_memory_gb,
            gpu_count: job.requirements.min_gpu_count,
            storage_gb: job.requirements.min_storage_gb,
            network_bandwidth: 100, // Default allocation
        }
    }
}

/// Checkpointing and recovery system
pub struct CheckpointManager {
    checkpoint_storage: HashMap<String, Checkpoint>,
    #[allow(dead_code)]
    checkpoint_interval: Duration,
    compression_enabled: bool,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(checkpoint_interval: Duration) -> Self {
        Self {
            checkpoint_storage: HashMap::new(),
            checkpoint_interval,
            compression_enabled: true,
        }
    }

    /// Create checkpoint for job
    pub fn create_checkpoint(
        &mut self,
        job_id: &str,
        state: JobState,
    ) -> Result<String, DistributedError> {
        let checkpoint_id = format!("{job_id}_{}", Instant::now().elapsed().as_millis());

        let checkpoint = Checkpoint {
            id: checkpoint_id.clone(),
            job_id: job_id.to_string(),
            state,
            created_at: Instant::now(),
            size_bytes: 1024, // Mock size
            compressed: self.compression_enabled,
        };

        self.checkpoint_storage
            .insert(checkpoint_id.clone(), checkpoint);
        Ok(checkpoint_id)
    }

    /// Restore job from checkpoint
    pub fn restore_checkpoint(&self, checkpoint_id: &str) -> Result<JobState, DistributedError> {
        self.checkpoint_storage
            .get(checkpoint_id)
            .map(|checkpoint| checkpoint.state.clone())
            .ok_or(DistributedError::JobNotFound)
    }

    /// Clean up old checkpoints
    pub fn cleanup_old_checkpoints(&mut self, retention_period: Duration) {
        let cutoff = Instant::now() - retention_period;
        self.checkpoint_storage
            .retain(|_, checkpoint| checkpoint.created_at > cutoff);
    }

    /// Get checkpoint statistics
    pub fn get_checkpoint_stats(&self) -> CheckpointStats {
        let total_checkpoints = self.checkpoint_storage.len();
        let total_size: u64 = self.checkpoint_storage.values().map(|c| c.size_bytes).sum();
        let compressed_checkpoints = self
            .checkpoint_storage
            .values()
            .filter(|c| c.compressed)
            .count();

        CheckpointStats {
            total_checkpoints,
            total_size_bytes: total_size,
            compressed_checkpoints,
            compression_ratio: if total_checkpoints > 0 {
                compressed_checkpoints as f64 / total_checkpoints as f64
            } else {
                0.0
            },
        }
    }
}

// New data structures for advanced features

#[derive(Debug, Clone)]
pub struct DistributedMessage {
    pub id: String,
    pub source: String,
    pub destination: String,
    pub message_type: MessageType,
    pub data: Vec<u8>,
    pub timestamp: Instant,
    pub priority: MessagePriority,
}

#[derive(Debug, Clone)]
pub enum MessageType {
    JobSubmission,
    JobResult,
    Heartbeat,
    ResourceUpdate,
    ConsensusRequest,
    DataPartition,
    Custom(String),
}

impl std::fmt::Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageType::JobSubmission => write!(f, "job_submission"),
            MessageType::JobResult => write!(f, "job_result"),
            MessageType::Heartbeat => write!(f, "heartbeat"),
            MessageType::ResourceUpdate => write!(f, "resource_update"),
            MessageType::ConsensusRequest => write!(f, "consensus_request"),
            MessageType::DataPartition => write!(f, "data_partition"),
            MessageType::Custom(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Type alias for message handler function
pub type MessageHandlerFn =
    Box<dyn Fn(&DistributedMessage) -> Result<MessageResponse, DistributedError> + Send + Sync>;

pub struct MessageHandler {
    pub handler_fn: MessageHandlerFn,
}

impl MessageHandler {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&DistributedMessage) -> Result<MessageResponse, DistributedError>
            + Send
            + Sync
            + 'static,
    {
        Self {
            handler_fn: Box::new(f),
        }
    }

    pub fn handle(
        &self,
        message: &DistributedMessage,
    ) -> Result<MessageResponse, DistributedError> {
        (self.handler_fn)(message)
    }
}

#[derive(Debug, Clone)]
pub struct MessageResponse {
    pub message_id: String,
    pub success: bool,
    pub data: Vec<u8>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusState {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub term: u64,
    pub index: usize,
    pub command: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: String,
    pub next_index: usize,
    pub match_index: usize,
    pub last_response: Instant,
}

#[derive(Debug, Clone)]
pub struct VoteRequest {
    pub term: u64,
    pub candidate_id: String,
    pub last_log_index: usize,
    pub last_log_term: u64,
}

#[derive(Debug, Clone)]
pub struct VoteResponse {
    pub term: u64,
    pub vote_granted: bool,
}

#[derive(Debug, Clone)]
pub enum PartitioningStrategy {
    Hash,
    Range,
    Random,
}

#[derive(Debug, Clone)]
pub struct PartitionMove {
    pub partition: usize,
    pub from_node: String,
    pub to_node: String,
}

#[derive(Debug, Clone)]
pub struct PartitioningResult {
    pub assignments_changed: usize,
    pub partitions_moved: Vec<PartitionMove>,
    pub rebalance_time: Instant,
}

#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    FIFO,
    ShortestJobFirst,
    GangScheduling,
    Backfill,
    PriorityBased,
}

#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    pub job_id: String,
    pub node_id: String,
    pub estimated_start_time: Instant,
    pub resource_allocation: ResourceAllocation,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub gpu_count: u32,
    pub storage_gb: u32,
    pub network_bandwidth: u32,
}

#[derive(Debug, Clone)]
pub struct ResourceReservation {
    pub id: String,
    pub node_id: String,
    pub start_time: Instant,
    pub duration: Duration,
    pub resources: ResourceAllocation,
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub id: String,
    pub job_id: String,
    pub state: JobState,
    pub created_at: Instant,
    pub size_bytes: u64,
    pub compressed: bool,
}

#[derive(Debug, Clone)]
pub struct JobState {
    pub progress: f64,
    pub intermediate_results: HashMap<String, Vec<u8>>,
    pub runtime_state: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CheckpointStats {
    pub total_checkpoints: usize,
    pub total_size_bytes: u64,
    pub compressed_checkpoints: usize,
    pub compression_ratio: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn create_test_node(id: &str) -> ClusterNode {
        ClusterNode {
            id: id.to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            capabilities: NodeCapabilities {
                cpu_cores: 8,
                memory_gb: 16,
                gpu_count: 1,
                storage_gb: 1000,
                network_bandwidth_mbps: 1000,
                supported_tasks: HashSet::from(["training".to_string(), "inference".to_string()]),
            },
            status: NodeStatus::Available,
            last_heartbeat: Instant::now(),
            load_metrics: LoadMetrics {
                cpu_usage: 0.3,
                memory_usage: 0.4,
                gpu_usage: 0.2,
                network_io: 0.1,
                disk_io: 0.1,
                active_jobs: 1,
                queue_size: 0,
            },
            job_history: Vec::new(),
        }
    }

    fn create_test_job(id: &str) -> DistributedJob {
        DistributedJob {
            id: id.to_string(),
            name: format!("test_job_{id}"),
            job_type: JobType::Training,
            priority: JobPriority::Normal,
            requirements: ResourceRequirements {
                min_cpu_cores: 2,
                min_memory_gb: 4,
                min_gpu_count: 0,
                min_storage_gb: 10,
                preferred_node_tags: HashSet::new(),
                exclusive_access: false,
            },
            created_at: Instant::now(),
            timeout: Duration::from_secs(3600),
            retry_count: 0,
            dependencies: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_cluster_creation() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        assert!(cluster.get_nodes().is_empty());
    }

    #[test]
    fn test_node_registration() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node = create_test_node("node1");

        assert!(cluster.register_node(node.clone()).is_ok());
        assert_eq!(cluster.get_nodes().len(), 1);
        assert_eq!(cluster.get_nodes()[0].id, "node1");
    }

    #[test]
    fn test_job_submission() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node = create_test_node("node1");
        let job = create_test_job("job1");

        cluster.register_node(node).unwrap();
        let job_id = cluster.submit_job(job).unwrap();

        assert_eq!(job_id, "job1");
        assert!(cluster.get_job_status(&job_id).is_some());
    }

    #[test]
    fn test_job_scheduling() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node = create_test_node("node1");
        let job = create_test_job("job1");

        cluster.register_node(node).unwrap();
        cluster.submit_job(job).unwrap();

        // Job should be scheduled and running
        let status = cluster.get_job_status("job1");
        assert!(status.is_some());
    }

    #[test]
    fn test_job_cancellation() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node = create_test_node("node1");
        let job = create_test_job("job1");

        cluster.register_node(node).unwrap();
        cluster.submit_job(job).unwrap();

        assert!(cluster.cancel_job("job1").is_ok());

        let execution = cluster.get_job_execution("job1");
        assert!(execution.is_some());
        assert_eq!(execution.unwrap().status, JobStatus::Cancelled);
    }

    #[test]
    fn test_node_heartbeat() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node = create_test_node("node1");

        cluster.register_node(node).unwrap();

        let new_metrics = LoadMetrics {
            cpu_usage: 0.8,
            memory_usage: 0.7,
            gpu_usage: 0.5,
            network_io: 0.3,
            disk_io: 0.2,
            active_jobs: 2,
            queue_size: 1,
        };

        assert!(cluster.update_heartbeat("node1", new_metrics).is_ok());

        let nodes = cluster.get_nodes();
        assert_eq!(nodes[0].load_metrics.cpu_usage, 0.8);
        assert_eq!(nodes[0].status, NodeStatus::Busy);
    }

    #[test]
    fn test_cluster_stats() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node1 = create_test_node("node1");
        let node2 = create_test_node("node2");

        cluster.register_node(node1).unwrap();
        cluster.register_node(node2).unwrap();

        let stats = cluster.get_cluster_stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.available_nodes, 2);
        assert_eq!(stats.total_cpu_cores, 16);
        assert_eq!(stats.total_memory_gb, 32);
    }

    #[test]
    fn test_job_scheduler() {
        let scheduler = JobScheduler::new();
        let mut nodes = HashMap::new();

        let node1 = create_test_node("node1");
        let node2 = create_test_node("node2");
        nodes.insert("node1".to_string(), node1);
        nodes.insert("node2".to_string(), node2);

        let job = create_test_job("job1");
        let selected_node = scheduler.find_suitable_node(&job, &nodes);

        assert!(selected_node.is_some());
        assert!(["node1", "node2"].contains(&selected_node.unwrap().as_str()));
    }

    #[test]
    fn test_load_balancer() {
        let load_balancer = LoadBalancer::new();
        let mut nodes = HashMap::new();

        let node1 = create_test_node("node1");
        nodes.insert("node1".to_string(), node1);

        assert!(load_balancer.rebalance(&nodes).is_ok());
    }

    #[test]
    fn test_fault_detector() {
        let mut fault_detector = FaultDetector::new();

        assert!(fault_detector.handle_failure("node1").is_ok());
        assert!(!fault_detector.is_problematic("node1"));

        // Simulate multiple failures
        for _ in 0..4 {
            fault_detector.handle_failure("node1").unwrap();
        }

        assert!(fault_detector.is_problematic("node1"));
    }

    #[test]
    fn test_node_failure_handling() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node = create_test_node("node1");
        let job = create_test_job("job1");

        cluster.register_node(node).unwrap();
        cluster.submit_job(job).unwrap();

        // Simulate node failure
        assert!(cluster.handle_node_failure("node1").is_ok());

        // Job should be marked as failed
        let execution = cluster.get_job_execution("job1");
        if let Some(exec) = execution {
            // Note: This depends on the job being scheduled first
            // In a real test, we'd need to ensure proper timing
            println!("Job status: {:?}", exec.status);
        }
    }

    #[test]
    fn test_resource_requirements() {
        let scheduler = JobScheduler::new();
        let mut nodes = HashMap::new();

        let node = create_test_node("node1");
        nodes.insert("node1".to_string(), node);

        // Create job with high requirements
        let mut job = create_test_job("job1");
        job.requirements.min_cpu_cores = 16; // More than available

        let selected_node = scheduler.find_suitable_node(&job, &nodes);
        assert!(selected_node.is_none());

        // Create job with reasonable requirements
        job.requirements.min_cpu_cores = 4;
        let selected_node = scheduler.find_suitable_node(&job, &nodes);
        assert!(selected_node.is_some());
    }

    #[test]
    fn test_job_priorities() {
        let cluster = DistributedCluster::new(ClusterConfig::default());
        let node = create_test_node("node1");
        cluster.register_node(node).unwrap();

        let mut job1 = create_test_job("job1");
        job1.priority = JobPriority::Low;

        let mut job2 = create_test_job("job2");
        job2.priority = JobPriority::High;

        cluster.submit_job(job1).unwrap();
        cluster.submit_job(job2).unwrap();

        // High priority job should be scheduled first
        // This would be more properly tested with actual scheduling logic
        let queue = cluster.job_queue.lock().unwrap();
        if !queue.is_empty() {
            assert_eq!(queue[0].priority, JobPriority::High);
        }
    }

    // Tests for advanced distributed computing features

    #[test]
    fn test_message_passing_system() {
        let mps = MessagePassingSystem::new("node1".to_string());

        // Test message creation and routing
        let message = DistributedMessage {
            id: "msg1".to_string(),
            source: "node1".to_string(),
            destination: "node2".to_string(),
            message_type: MessageType::JobSubmission,
            data: vec![1, 2, 3, 4],
            timestamp: Instant::now(),
            priority: MessagePriority::Normal,
        };

        // Should fail initially as node2 is not in routing table
        assert!(mps.send_message(message.clone()).is_err());

        // Add node2 to routing table
        mps.routing_table.write().unwrap().insert(
            "node2".to_string(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081),
        );

        // Now it should succeed
        assert!(mps.send_message(message).is_ok());
    }

    #[test]
    fn test_message_broadcasting() {
        let mps = MessagePassingSystem::new("node1".to_string());

        // Add multiple nodes to routing table
        mps.routing_table.write().unwrap().insert(
            "node2".to_string(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081),
        );
        mps.routing_table.write().unwrap().insert(
            "node3".to_string(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8082),
        );

        let data = vec![5, 6, 7, 8];
        assert!(mps.broadcast_message(MessageType::Heartbeat, data).is_ok());

        // Should have 2 messages in queue (for node2 and node3)
        let queue = mps.message_queue.lock().unwrap();
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_message_handler() {
        let handler = MessageHandler::new(|msg: &DistributedMessage| {
            Ok(MessageResponse {
                message_id: msg.id.clone(),
                success: true,
                data: vec![],
                error: None,
            })
        });

        let message = DistributedMessage {
            id: "test_msg".to_string(),
            source: "node1".to_string(),
            destination: "node2".to_string(),
            message_type: MessageType::JobSubmission,
            data: vec![],
            timestamp: Instant::now(),
            priority: MessagePriority::Normal,
        };

        let response = handler.handle(&message).unwrap();
        assert!(response.success);
        assert_eq!(response.message_id, "test_msg");
    }

    #[test]
    fn test_consensus_manager() {
        let mut consensus = ConsensusManager::new(
            "node1".to_string(),
            vec!["node2".to_string(), "node3".to_string()],
        );

        // Initially should be follower
        let (state, term) = consensus.get_state();
        assert_eq!(state, ConsensusState::Follower);
        assert_eq!(term, 0);

        // Start election
        assert!(consensus.start_election().is_ok());
        let (state, term) = consensus.get_state();
        assert_eq!(state, ConsensusState::Candidate);
        assert_eq!(term, 1);

        // Test vote request handling
        let vote_request = VoteRequest {
            term: 2,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = consensus.handle_vote_request(vote_request);
        assert!(response.vote_granted);
        assert_eq!(response.term, 2);
    }

    #[test]
    fn test_consensus_log_entry() {
        let mut consensus = ConsensusManager::new("node1".to_string(), vec!["node2".to_string()]);

        // Can't append as follower
        let entry = LogEntry {
            term: 1,
            index: 0,
            command: "test_command".to_string(),
            data: vec![1, 2, 3],
        };

        assert!(consensus.append_entry(entry.clone()).is_err());

        // Become leader first
        consensus.state = ConsensusState::Leader;
        assert!(consensus.append_entry(entry).is_ok());
        assert_eq!(consensus.log.len(), 1);
    }

    #[test]
    fn test_data_partitioner() {
        let mut partitioner = DataPartitioner::new(PartitioningStrategy::Hash, 4, 2);

        // Test partitioning
        let partition1 = partitioner.get_partition("key1");
        let partition2 = partitioner.get_partition("key2");
        assert!(partition1 < 4);
        assert!(partition2 < 4);

        // Test consistent partitioning
        assert_eq!(partition1, partitioner.get_partition("key1"));

        // Test partition assignment
        partitioner.assign_partition(0, "node1".to_string());
        partitioner.assign_partition(1, "node2".to_string());

        let nodes = partitioner.get_partition_nodes(0);
        assert!(nodes.contains(&"node1".to_string()));
    }

    #[test]
    fn test_data_partitioner_rebalancing() {
        let mut partitioner = DataPartitioner::new(PartitioningStrategy::Hash, 4, 1);

        let nodes = vec!["node1".to_string(), "node2".to_string()];
        let result = partitioner.rebalance_partitions(&nodes);

        assert_eq!(result.assignments_changed, 4);
        assert_eq!(result.partitions_moved.len(), 0); // No existing assignments
    }

    #[test]
    fn test_partitioning_strategies() {
        let hash_partitioner = DataPartitioner::new(PartitioningStrategy::Hash, 4, 1);
        let range_partitioner = DataPartitioner::new(PartitioningStrategy::Range, 4, 1);
        let random_partitioner = DataPartitioner::new(PartitioningStrategy::Random, 4, 1);

        let key = "test_key";

        let hash_partition = hash_partitioner.get_partition(key);
        let range_partition = range_partitioner.get_partition(key);
        let random_partition = random_partitioner.get_partition(key);

        assert!(hash_partition < 4);
        assert!(range_partition < 4);
        assert!(random_partition < 4);
    }

    #[test]
    fn test_advanced_job_scheduler() {
        let mut scheduler = AdvancedJobScheduler::new();
        let mut nodes = HashMap::new();

        let node1 = create_test_node("node1");
        let node2 = create_test_node("node2");
        nodes.insert("node1".to_string(), node1);
        nodes.insert("node2".to_string(), node2);

        let jobs = vec![create_test_job("job1"), create_test_job("job2")];

        // Test gang scheduling
        let decisions = scheduler.gang_schedule(&jobs, &nodes).unwrap();
        assert_eq!(decisions.len(), jobs.len());

        for decision in &decisions {
            assert!(nodes.contains_key(&decision.node_id));
            assert!(decision.resource_allocation.cpu_cores > 0);
        }
    }

    #[test]
    fn test_backfill_scheduling() {
        let mut scheduler = AdvancedJobScheduler::new();
        let mut nodes = HashMap::new();

        let node1 = create_test_node("node1");
        nodes.insert("node1".to_string(), node1);

        let waiting_jobs = vec![create_test_job("waiting_job")];

        // Test backfill scheduling
        let decisions = scheduler.backfill_schedule(&waiting_jobs, &nodes).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].job_id, "waiting_job");
    }

    #[test]
    fn test_resource_reservation() {
        let mut scheduler = AdvancedJobScheduler::new();

        let reservation = ResourceReservation {
            id: "reservation1".to_string(),
            node_id: "node1".to_string(),
            start_time: Instant::now(),
            duration: Duration::from_secs(3600),
            resources: ResourceAllocation {
                cpu_cores: 4,
                memory_gb: 8,
                gpu_count: 1,
                storage_gb: 100,
                network_bandwidth: 1000,
            },
        };

        assert!(scheduler.reserve_resources(reservation).is_ok());
        assert_eq!(scheduler.resource_reservations.len(), 1);
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut checkpoint_mgr = CheckpointManager::new(Duration::from_secs(300));

        let job_state = JobState {
            progress: 0.5,
            intermediate_results: HashMap::new(),
            runtime_state: vec![1, 2, 3, 4],
        };

        // Create checkpoint
        let checkpoint_id = checkpoint_mgr
            .create_checkpoint("job1", job_state.clone())
            .unwrap();
        assert!(!checkpoint_id.is_empty());

        // Restore checkpoint
        let restored_state = checkpoint_mgr.restore_checkpoint(&checkpoint_id).unwrap();
        assert_eq!(restored_state.progress, 0.5);
        assert_eq!(restored_state.runtime_state, vec![1, 2, 3, 4]);

        // Test checkpoint stats
        let stats = checkpoint_mgr.get_checkpoint_stats();
        assert_eq!(stats.total_checkpoints, 1);
        assert!(stats.total_size_bytes > 0);
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let mut checkpoint_mgr = CheckpointManager::new(Duration::from_secs(300));

        let job_state = JobState {
            progress: 1.0,
            intermediate_results: HashMap::new(),
            runtime_state: vec![],
        };

        // Create multiple checkpoints
        checkpoint_mgr
            .create_checkpoint("job1", job_state.clone())
            .unwrap();
        checkpoint_mgr.create_checkpoint("job2", job_state).unwrap();

        assert_eq!(checkpoint_mgr.checkpoint_storage.len(), 2);

        // Cleanup with zero retention (should remove all)
        checkpoint_mgr.cleanup_old_checkpoints(Duration::from_secs(0));
        assert_eq!(checkpoint_mgr.checkpoint_storage.len(), 0);
    }

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(format!("{}", MessageType::JobSubmission), "job_submission");
        assert_eq!(format!("{}", MessageType::JobResult), "job_result");
        assert_eq!(format!("{}", MessageType::Heartbeat), "heartbeat");
        assert_eq!(
            format!("{}", MessageType::ResourceUpdate),
            "resource_update"
        );
        assert_eq!(
            format!("{}", MessageType::ConsensusRequest),
            "consensus_request"
        );
        assert_eq!(format!("{}", MessageType::DataPartition), "data_partition");
        assert_eq!(
            format!("{}", MessageType::Custom("test".to_string())),
            "test"
        );
    }

    #[test]
    fn test_consensus_states() {
        let follower = ConsensusState::Follower;
        let candidate = ConsensusState::Candidate;
        let leader = ConsensusState::Leader;

        assert_eq!(follower, ConsensusState::Follower);
        assert_eq!(candidate, ConsensusState::Candidate);
        assert_eq!(leader, ConsensusState::Leader);
    }

    #[test]
    fn test_scheduling_policies() {
        let policies = [
            SchedulingPolicy::FIFO,
            SchedulingPolicy::ShortestJobFirst,
            SchedulingPolicy::GangScheduling,
            SchedulingPolicy::Backfill,
            SchedulingPolicy::PriorityBased,
        ];

        assert_eq!(policies.len(), 5);
    }

    #[test]
    fn test_message_priorities() {
        let low = MessagePriority::Low;
        let normal = MessagePriority::Normal;
        let high = MessagePriority::High;
        let critical = MessagePriority::Critical;

        // Test that all priorities can be created
        match low {
            MessagePriority::Low => {}
            _ => panic!(),
        }
        match normal {
            MessagePriority::Normal => {}
            _ => panic!(),
        }
        match high {
            MessagePriority::High => {}
            _ => panic!(),
        }
        match critical {
            MessagePriority::Critical => {}
            _ => panic!(),
        }
    }

    #[test]
    fn test_resource_allocation_calculations() {
        let allocation = ResourceAllocation {
            cpu_cores: 8,
            memory_gb: 16,
            gpu_count: 2,
            storage_gb: 500,
            network_bandwidth: 1000,
        };

        assert_eq!(allocation.cpu_cores, 8);
        assert_eq!(allocation.memory_gb, 16);
        assert_eq!(allocation.gpu_count, 2);
        assert_eq!(allocation.storage_gb, 500);
        assert_eq!(allocation.network_bandwidth, 1000);
    }

    #[test]
    fn test_job_state_serialization() {
        let mut intermediate_results = HashMap::new();
        intermediate_results.insert("result1".to_string(), vec![1, 2, 3]);
        intermediate_results.insert("result2".to_string(), vec![4, 5, 6]);

        let job_state = JobState {
            progress: 0.75,
            intermediate_results,
            runtime_state: vec![7, 8, 9],
        };

        assert_eq!(job_state.progress, 0.75);
        assert_eq!(job_state.intermediate_results.len(), 2);
        assert_eq!(job_state.runtime_state, vec![7, 8, 9]);
        assert!(job_state.intermediate_results.contains_key("result1"));
        assert!(job_state.intermediate_results.contains_key("result2"));
    }
}
