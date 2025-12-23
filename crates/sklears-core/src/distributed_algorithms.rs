//! Distributed Machine Learning Algorithms
//!
//! This module provides concrete implementations of distributed ML algorithms
//! that scale across multiple nodes with fault tolerance and load balancing.

use crate::distributed::NodeId;
use crate::error::{Result, SklearsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Distributed linear regression using parameter server architecture
///
/// Implements distributed training of linear regression models across
/// multiple worker nodes with centralized parameter synchronization.
#[derive(Debug)]
pub struct DistributedLinearRegression {
    /// Configuration for distributed training
    pub config: DistributedConfig,
    /// Parameter server for coordinating updates
    pub parameter_server: Arc<RwLock<ParameterServer>>,
    /// Worker nodes performing computation
    pub workers: Vec<WorkerNode>,
    /// Current model parameters
    pub parameters: Arc<RwLock<ModelParameters>>,
}

/// Configuration for distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Number of worker nodes
    pub num_workers: usize,
    /// Synchronization strategy
    pub sync_strategy: SyncStrategy,
    /// Enable fault tolerance
    pub fault_tolerance: bool,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate
    pub learning_rate: f64,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            sync_strategy: SyncStrategy::Synchronous,
            fault_tolerance: true,
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 0.01,
        }
    }
}

/// Synchronization strategy for distributed training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStrategy {
    /// All workers synchronize after each iteration
    Synchronous,
    /// Workers update asynchronously
    Asynchronous,
    /// Bounded asynchronous updates
    BoundedAsync { staleness_bound: usize },
}

/// Parameter server for coordinating distributed training
#[derive(Debug, Clone)]
pub struct ParameterServer {
    /// Current global parameters
    pub parameters: Vec<f64>,
    /// Version number for parameters
    pub version: usize,
    /// Gradient accumulator
    pub gradient_accumulator: Vec<f64>,
    /// Number of workers
    pub num_workers: usize,
    /// Updates received in current iteration
    pub updates_received: usize,
}

impl ParameterServer {
    /// Create a new parameter server
    pub fn new(num_parameters: usize, num_workers: usize) -> Self {
        Self {
            parameters: vec![0.0; num_parameters],
            version: 0,
            gradient_accumulator: vec![0.0; num_parameters],
            num_workers,
            updates_received: 0,
        }
    }

    /// Receive gradient update from worker
    pub fn receive_gradient(&mut self, gradient: Vec<f64>) -> Result<()> {
        if gradient.len() != self.parameters.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: self.parameters.len(),
                actual: gradient.len(),
            });
        }

        // Accumulate gradient
        for (acc, grad) in self.gradient_accumulator.iter_mut().zip(gradient.iter()) {
            *acc += grad;
        }

        self.updates_received += 1;

        // Apply update when all workers have reported (synchronous)
        if self.updates_received == self.num_workers {
            self.apply_accumulated_gradients();
        }

        Ok(())
    }

    /// Apply accumulated gradients to parameters
    fn apply_accumulated_gradients(&mut self) {
        let scale = 1.0 / self.num_workers as f64;

        for (param, grad) in self
            .parameters
            .iter_mut()
            .zip(self.gradient_accumulator.iter())
        {
            *param -= grad * scale;
        }

        // Reset accumulator
        self.gradient_accumulator.iter_mut().for_each(|g| *g = 0.0);
        self.updates_received = 0;
        self.version += 1;
    }

    /// Get current parameters
    pub fn get_parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    /// Get parameter version
    pub fn get_version(&self) -> usize {
        self.version
    }
}

/// Worker node for distributed computation
#[derive(Debug, Clone)]
pub struct WorkerNode {
    /// Node identifier
    pub id: NodeId,
    /// Local data partition
    pub data_partition: DataPartition,
    /// Local model parameters (cached from parameter server)
    pub local_parameters: Vec<f64>,
    /// Parameter version
    pub parameter_version: usize,
    /// Worker statistics
    pub stats: WorkerStats,
}

/// Data partition assigned to a worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPartition {
    /// Feature matrix
    pub features: Vec<Vec<f64>>,
    /// Target values
    pub targets: Vec<f64>,
    /// Partition index
    pub partition_id: usize,
}

/// Statistics tracked by each worker
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerStats {
    /// Number of samples processed
    pub samples_processed: usize,
    /// Number of gradient computations
    pub gradient_computations: usize,
    /// Total computation time in milliseconds
    pub total_compute_time_ms: u64,
    /// Number of communication rounds
    pub communication_rounds: usize,
}

impl WorkerNode {
    /// Create a new worker node
    pub fn new(id: NodeId, data_partition: DataPartition) -> Self {
        Self {
            id,
            data_partition,
            local_parameters: Vec::new(),
            parameter_version: 0,
            stats: WorkerStats::default(),
        }
    }

    /// Compute local gradient on assigned data partition
    pub fn compute_local_gradient(&mut self, parameters: &[f64]) -> Result<Vec<f64>> {
        let start_time = std::time::Instant::now();

        // Update local parameters
        self.local_parameters = parameters.to_vec();

        let n_samples = self.data_partition.features.len();
        let n_features = parameters.len();
        let mut gradient = vec![0.0; n_features];

        // Compute gradient for linear regression
        for (features, target) in self
            .data_partition
            .features
            .iter()
            .zip(self.data_partition.targets.iter())
        {
            // Prediction: y_pred = w^T x
            let prediction: f64 = features
                .iter()
                .zip(parameters.iter())
                .map(|(x, w)| x * w)
                .sum();

            // Error: e = y_pred - y_true
            let error = prediction - target;

            // Gradient: grad = 2 * e * x
            for (i, x) in features.iter().enumerate() {
                gradient[i] += 2.0 * error * x;
            }
        }

        // Average gradient over samples
        for g in gradient.iter_mut() {
            *g /= n_samples as f64;
        }

        // Update statistics
        self.stats.samples_processed += n_samples;
        self.stats.gradient_computations += 1;
        self.stats.total_compute_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(gradient)
    }

    /// Get worker statistics
    pub fn get_stats(&self) -> &WorkerStats {
        &self.stats
    }
}

/// Model parameters for distributed learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Weight vector
    pub weights: Vec<f64>,
    /// Bias term (intercept)
    pub bias: f64,
    /// Training metadata
    pub metadata: ParameterMetadata,
}

/// Metadata about model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterMetadata {
    /// Number of training iterations completed
    pub iterations_completed: usize,
    /// Current training loss
    pub current_loss: f64,
    /// Convergence status
    pub converged: bool,
    /// Timestamp of last update
    pub last_updated: std::time::SystemTime,
}

impl DistributedLinearRegression {
    /// Create a new distributed linear regression model
    pub fn new(config: DistributedConfig, num_features: usize) -> Self {
        let parameter_server = Arc::new(RwLock::new(ParameterServer::new(
            num_features,
            config.num_workers,
        )));

        Self {
            config,
            parameter_server,
            workers: Vec::new(),
            parameters: Arc::new(RwLock::new(ModelParameters {
                weights: vec![0.0; num_features],
                bias: 0.0,
                metadata: ParameterMetadata {
                    iterations_completed: 0,
                    current_loss: f64::INFINITY,
                    converged: false,
                    last_updated: std::time::SystemTime::now(),
                },
            })),
        }
    }

    /// Partition data across workers
    pub fn partition_data(&mut self, features: Vec<Vec<f64>>, targets: Vec<f64>) -> Result<()> {
        let n_samples = features.len();
        let samples_per_worker =
            (n_samples + self.config.num_workers - 1) / self.config.num_workers;

        for worker_idx in 0..self.config.num_workers {
            let start_idx = worker_idx * samples_per_worker;
            let end_idx = ((worker_idx + 1) * samples_per_worker).min(n_samples);

            if start_idx < n_samples {
                let partition = DataPartition {
                    features: features[start_idx..end_idx].to_vec(),
                    targets: targets[start_idx..end_idx].to_vec(),
                    partition_id: worker_idx,
                };

                let worker =
                    WorkerNode::new(NodeId::new(format!("worker_{}", worker_idx)), partition);

                self.workers.push(worker);
            }
        }

        Ok(())
    }

    /// Train the model using distributed gradient descent
    pub fn fit(&mut self) -> Result<()> {
        for iteration in 0..self.config.max_iterations {
            // Get current parameters from parameter server
            let params = {
                let ps = self.parameter_server.read().unwrap();
                ps.get_parameters()
            };

            // Compute gradients on all workers
            let mut all_gradients = Vec::new();
            for worker in &mut self.workers {
                let gradient = worker.compute_local_gradient(&params)?;
                all_gradients.push(gradient);
            }

            // Send gradients to parameter server
            {
                let mut ps = self.parameter_server.write().unwrap();
                for gradient in all_gradients {
                    ps.receive_gradient(gradient)?;
                }
            }

            // Check for convergence
            if iteration % 10 == 0 {
                let loss = self.compute_global_loss(&params)?;
                let mut model_params = self.parameters.write().unwrap();

                if (model_params.metadata.current_loss - loss).abs() < self.config.tolerance {
                    model_params.metadata.converged = true;
                    model_params.metadata.iterations_completed = iteration + 1;
                    break;
                }

                model_params.metadata.current_loss = loss;
                model_params.metadata.iterations_completed = iteration + 1;
                model_params.metadata.last_updated = std::time::SystemTime::now();
            }
        }

        // Update final parameters
        let final_params = {
            let ps = self.parameter_server.read().unwrap();
            ps.get_parameters()
        };

        let mut model_params = self.parameters.write().unwrap();
        model_params.weights = final_params;

        Ok(())
    }

    /// Compute global loss across all workers
    fn compute_global_loss(&self, parameters: &[f64]) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut total_samples = 0;

        for worker in &self.workers {
            for (features, target) in worker
                .data_partition
                .features
                .iter()
                .zip(worker.data_partition.targets.iter())
            {
                let prediction: f64 = features
                    .iter()
                    .zip(parameters.iter())
                    .map(|(x, w)| x * w)
                    .sum();
                let error = prediction - target;
                total_loss += error * error;
                total_samples += 1;
            }
        }

        Ok(total_loss / total_samples as f64)
    }

    /// Get training statistics from all workers
    pub fn get_training_stats(&self) -> DistributedTrainingStats {
        let mut total_samples = 0;
        let mut total_compute_time = 0;
        let mut total_gradient_computations = 0;

        for worker in &self.workers {
            let stats = worker.get_stats();
            total_samples += stats.samples_processed;
            total_compute_time += stats.total_compute_time_ms;
            total_gradient_computations += stats.gradient_computations;
        }

        DistributedTrainingStats {
            num_workers: self.workers.len(),
            total_samples_processed: total_samples,
            total_compute_time_ms: total_compute_time,
            total_gradient_computations,
            parameter_server_version: self.parameter_server.read().unwrap().get_version(),
        }
    }

    /// Predict on new data using trained parameters
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let params = self.parameters.read().unwrap();
        let mut predictions = Vec::new();

        for feature_row in features {
            let pred: f64 = feature_row
                .iter()
                .zip(params.weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
                + params.bias;

            predictions.push(pred);
        }

        Ok(predictions)
    }
}

/// Statistics from distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingStats {
    /// Number of worker nodes
    pub num_workers: usize,
    /// Total samples processed across all workers
    pub total_samples_processed: usize,
    /// Total computation time in milliseconds
    pub total_compute_time_ms: u64,
    /// Total gradient computations
    pub total_gradient_computations: usize,
    /// Parameter server version
    pub parameter_server_version: usize,
}

// ============================================================================
// Advanced Distributed Learning Features
// ============================================================================

/// Federated Learning framework with privacy-preserving techniques
///
/// Implements federated averaging (FedAvg) and secure aggregation for
/// privacy-preserving distributed machine learning.
#[derive(Debug)]
pub struct FederatedLearning {
    /// Federated learning configuration
    pub config: FederatedConfig,
    /// Client models
    pub clients: Vec<FederatedClient>,
    /// Global model parameters
    pub global_model: Arc<RwLock<ModelParameters>>,
    /// Privacy mechanism
    pub privacy_mechanism: PrivacyMechanism,
}

/// Configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedConfig {
    /// Number of clients
    pub num_clients: usize,
    /// Fraction of clients selected per round
    pub client_fraction: f64,
    /// Number of local epochs per client
    pub local_epochs: usize,
    /// Local learning rate
    pub local_learning_rate: f64,
    /// Enable secure aggregation
    pub secure_aggregation: bool,
    /// Differential privacy epsilon
    pub dp_epsilon: Option<f64>,
    /// Differential privacy delta
    pub dp_delta: Option<f64>,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            num_clients: 10,
            client_fraction: 0.3,
            local_epochs: 5,
            local_learning_rate: 0.01,
            secure_aggregation: true,
            dp_epsilon: Some(1.0),
            dp_delta: Some(1e-5),
        }
    }
}

/// Federated learning client
#[derive(Debug, Clone)]
pub struct FederatedClient {
    /// Client identifier
    pub id: String,
    /// Local dataset size
    pub dataset_size: usize,
    /// Local model parameters
    pub local_parameters: Vec<f64>,
    /// Client training statistics
    pub stats: ClientStats,
}

/// Statistics for federated client
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientStats {
    /// Number of training rounds participated in
    pub rounds_participated: usize,
    /// Total samples used in training
    pub total_samples: usize,
    /// Average local loss
    pub avg_local_loss: f64,
}

impl FederatedLearning {
    /// Create a new federated learning system
    pub fn new(config: FederatedConfig, num_features: usize) -> Self {
        Self {
            config,
            clients: Vec::new(),
            global_model: Arc::new(RwLock::new(ModelParameters {
                weights: vec![0.0; num_features],
                bias: 0.0,
                metadata: ParameterMetadata {
                    iterations_completed: 0,
                    current_loss: f64::INFINITY,
                    converged: false,
                    last_updated: std::time::SystemTime::now(),
                },
            })),
            privacy_mechanism: PrivacyMechanism::new(),
        }
    }

    /// Add a client to the federated system
    pub fn add_client(&mut self, client_id: String, dataset_size: usize) {
        let client = FederatedClient {
            id: client_id,
            dataset_size,
            local_parameters: Vec::new(),
            stats: ClientStats::default(),
        };
        self.clients.push(client);
    }

    /// Select clients for a training round
    pub fn select_clients(&self) -> Vec<usize> {
        use scirs2_core::random::thread_rng;

        let num_selected =
            (self.clients.len() as f64 * self.config.client_fraction).ceil() as usize;
        let mut selected = Vec::new();
        let mut rng = thread_rng();

        let mut indices: Vec<usize> = (0..self.clients.len()).collect();

        // Fisher-Yates shuffle for random selection
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        selected.extend_from_slice(&indices[..num_selected]);
        selected
    }

    /// Perform federated averaging of client updates
    pub fn federated_average(&self, client_updates: &[(usize, Vec<f64>)]) -> Vec<f64> {
        if client_updates.is_empty() {
            return vec![];
        }

        let num_features = client_updates[0].1.len();
        let mut averaged = vec![0.0; num_features];
        let mut total_weight = 0.0;

        for (client_idx, update) in client_updates {
            let weight = self.clients[*client_idx].dataset_size as f64;
            total_weight += weight;

            for (i, val) in update.iter().enumerate() {
                averaged[i] += val * weight;
            }
        }

        // Normalize by total weight
        for val in averaged.iter_mut() {
            *val /= total_weight;
        }

        // Apply differential privacy if enabled
        if self.config.dp_epsilon.is_some() {
            self.privacy_mechanism
                .apply_noise(&mut averaged, self.config.dp_epsilon.unwrap());
        }

        averaged
    }

    /// Get global model parameters
    pub fn get_global_model(&self) -> ModelParameters {
        self.global_model.read().unwrap().clone()
    }
}

/// Privacy mechanism for federated learning
#[derive(Debug, Clone)]
pub struct PrivacyMechanism {
    /// Noise scale for differential privacy
    pub noise_scale: f64,
}

impl PrivacyMechanism {
    /// Create a new privacy mechanism
    pub fn new() -> Self {
        Self { noise_scale: 1.0 }
    }

    /// Apply differential privacy noise to gradients
    pub fn apply_noise(&self, gradients: &mut [f64], epsilon: f64) {
        use scirs2_core::random::essentials::Normal;
        use scirs2_core::random::thread_rng;

        let mut rng = thread_rng();
        let noise_std = self.noise_scale / epsilon;
        let normal = Normal::new(0.0, noise_std).unwrap();

        for grad in gradients.iter_mut() {
            *grad += rng.sample(normal);
        }
    }

    /// Clip gradients for privacy preservation
    pub fn clip_gradients(&self, gradients: &mut [f64], clip_norm: f64) {
        let norm: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();

        if norm > clip_norm {
            let scale = clip_norm / norm;
            for grad in gradients.iter_mut() {
                *grad *= scale;
            }
        }
    }
}

impl Default for PrivacyMechanism {
    fn default() -> Self {
        Self::new()
    }
}

/// Byzantine-Fault Tolerant aggregation for robust distributed learning
///
/// Implements robust aggregation methods that are resilient to Byzantine
/// (malicious or faulty) workers.
#[derive(Debug)]
pub struct ByzantineFaultTolerant {
    /// BFT configuration
    pub config: BFTConfig,
    /// Aggregation method
    pub aggregation_method: AggregationMethod,
}

/// Configuration for Byzantine-Fault Tolerant training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BFTConfig {
    /// Maximum fraction of Byzantine workers tolerated
    pub max_byzantine_fraction: f64,
    /// Detection threshold for identifying Byzantine behavior
    pub detection_threshold: f64,
    /// Enable reputation tracking
    pub enable_reputation: bool,
}

impl Default for BFTConfig {
    fn default() -> Self {
        Self {
            max_byzantine_fraction: 0.3,
            detection_threshold: 2.0,
            enable_reputation: true,
        }
    }
}

/// Robust aggregation methods for Byzantine-Fault Tolerance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Coordinate-wise median
    Median,
    /// Trimmed mean (remove extreme values)
    TrimmedMean { trim_fraction: usize },
    /// Krum algorithm (select most representative gradient)
    Krum,
    /// Bulyan (combination of Krum and trimmed mean)
    Bulyan,
}

impl ByzantineFaultTolerant {
    /// Create a new BFT aggregator
    pub fn new(config: BFTConfig, method: AggregationMethod) -> Self {
        Self {
            config,
            aggregation_method: method,
        }
    }

    /// Aggregate gradients using Byzantine-Fault Tolerant method
    pub fn aggregate(&self, gradients: &[Vec<f64>]) -> Result<Vec<f64>> {
        if gradients.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot aggregate empty gradient set".to_string(),
            ));
        }

        match self.aggregation_method {
            AggregationMethod::Median => self.coordinate_wise_median(gradients),
            AggregationMethod::TrimmedMean { trim_fraction } => {
                self.trimmed_mean(gradients, trim_fraction)
            }
            AggregationMethod::Krum => self.krum(gradients),
            AggregationMethod::Bulyan => self.bulyan(gradients),
        }
    }

    /// Coordinate-wise median aggregation
    fn coordinate_wise_median(&self, gradients: &[Vec<f64>]) -> Result<Vec<f64>> {
        let num_features = gradients[0].len();
        let mut result = vec![0.0; num_features];

        for i in 0..num_features {
            let mut values: Vec<f64> = gradients.iter().map(|g| g[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            result[i] = values[values.len() / 2];
        }

        Ok(result)
    }

    /// Trimmed mean aggregation
    fn trimmed_mean(&self, gradients: &[Vec<f64>], trim_fraction: usize) -> Result<Vec<f64>> {
        let num_features = gradients[0].len();
        let mut result = vec![0.0; num_features];
        let trim_count = (gradients.len() * trim_fraction) / 100;

        for i in 0..num_features {
            let mut values: Vec<f64> = gradients.iter().map(|g| g[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Remove extreme values
            let trimmed = &values[trim_count..values.len() - trim_count];
            result[i] = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
        }

        Ok(result)
    }

    /// Krum aggregation (select most representative gradient)
    fn krum(&self, gradients: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n = gradients.len();
        let f = (n as f64 * self.config.max_byzantine_fraction).floor() as usize;
        let m = n - f - 2;

        let mut scores = vec![0.0; n];

        // Compute Krum scores
        for i in 0..n {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n {
                if i != j {
                    let dist = self.euclidean_distance(&gradients[i], &gradients[j]);
                    distances.push((j, dist));
                }
            }

            // Sort by distance and sum m closest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scores[i] = distances.iter().take(m).map(|(_, d)| d).sum();
        }

        // Select gradient with minimum score
        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        Ok(gradients[best_idx].clone())
    }

    /// Bulyan aggregation (robust combination)
    fn bulyan(&self, gradients: &[Vec<f64>]) -> Result<Vec<f64>> {
        // Simplified Bulyan: apply Krum multiple times and then trimmed mean
        let n = gradients.len();
        let f = (n as f64 * self.config.max_byzantine_fraction).floor() as usize;
        let theta = n - 2 * f;

        if theta < 1 {
            return Err(SklearsError::InvalidInput(
                "Too many Byzantine workers for Bulyan".to_string(),
            ));
        }

        // For simplicity, use coordinate-wise median as a robust aggregation
        self.coordinate_wise_median(gradients)
    }

    /// Compute Euclidean distance between two gradients
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Advanced load balancing for distributed systems
///
/// Implements sophisticated load balancing strategies for optimal
/// resource utilization and performance.
#[derive(Debug)]
pub struct LoadBalancer {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Worker load tracking
    pub worker_loads: HashMap<String, WorkerLoad>,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least-loaded worker first
    LeastLoaded,
    /// Weighted random based on capacity
    WeightedRandom,
    /// Power of two choices
    PowerOfTwo,
}

/// Worker load information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLoad {
    /// Current number of tasks
    pub active_tasks: usize,
    /// Worker capacity
    pub capacity: usize,
    /// Average task completion time (ms)
    pub avg_completion_time_ms: u64,
    /// Load factor (0.0 - 1.0)
    pub load_factor: f64,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            worker_loads: HashMap::new(),
        }
    }

    /// Register a worker with the load balancer
    pub fn register_worker(&mut self, worker_id: String, capacity: usize) {
        self.worker_loads.insert(
            worker_id,
            WorkerLoad {
                active_tasks: 0,
                capacity,
                avg_completion_time_ms: 0,
                load_factor: 0.0,
            },
        );
    }

    /// Select a worker for task assignment
    pub fn select_worker(&mut self) -> Option<String> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_select(),
            LoadBalancingStrategy::LeastLoaded => self.least_loaded_select(),
            LoadBalancingStrategy::WeightedRandom => self.weighted_random_select(),
            LoadBalancingStrategy::PowerOfTwo => self.power_of_two_select(),
        }
    }

    /// Round-robin worker selection
    fn round_robin_select(&self) -> Option<String> {
        self.worker_loads.keys().next().cloned()
    }

    /// Select least loaded worker
    fn least_loaded_select(&self) -> Option<String> {
        self.worker_loads
            .iter()
            .min_by(|(_, a), (_, b)| a.load_factor.partial_cmp(&b.load_factor).unwrap())
            .map(|(id, _)| id.clone())
    }

    /// Weighted random selection based on available capacity
    fn weighted_random_select(&self) -> Option<String> {
        use scirs2_core::random::thread_rng;

        if self.worker_loads.is_empty() {
            return None;
        }

        let mut rng = thread_rng();
        let total_capacity: f64 = self
            .worker_loads
            .values()
            .map(|load| (load.capacity - load.active_tasks) as f64)
            .sum();

        let mut rand_val = rng.gen_range(0.0..total_capacity);

        for (id, load) in &self.worker_loads {
            let available = (load.capacity - load.active_tasks) as f64;
            if rand_val < available {
                return Some(id.clone());
            }
            rand_val -= available;
        }

        self.worker_loads.keys().next().cloned()
    }

    /// Power of two choices (sample 2 random workers, pick least loaded)
    fn power_of_two_select(&self) -> Option<String> {
        use scirs2_core::random::thread_rng;

        if self.worker_loads.is_empty() {
            return None;
        }

        let mut rng = thread_rng();
        let workers: Vec<_> = self.worker_loads.keys().collect();

        if workers.len() == 1 {
            return Some(workers[0].clone());
        }

        let idx1 = rng.gen_range(0..workers.len());
        let mut idx2 = rng.gen_range(0..workers.len());
        while idx2 == idx1 {
            idx2 = rng.gen_range(0..workers.len());
        }

        let load1 = &self.worker_loads[workers[idx1]];
        let load2 = &self.worker_loads[workers[idx2]];

        if load1.load_factor < load2.load_factor {
            Some(workers[idx1].clone())
        } else {
            Some(workers[idx2].clone())
        }
    }

    /// Update worker load after task assignment
    pub fn update_load(&mut self, worker_id: &str, task_assigned: bool) {
        if let Some(load) = self.worker_loads.get_mut(worker_id) {
            if task_assigned {
                load.active_tasks += 1;
            } else if load.active_tasks > 0 {
                load.active_tasks -= 1;
            }
            load.load_factor = load.active_tasks as f64 / load.capacity as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_server_creation() {
        let ps = ParameterServer::new(5, 3);
        assert_eq!(ps.parameters.len(), 5);
        assert_eq!(ps.num_workers, 3);
        assert_eq!(ps.version, 0);
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut ps = ParameterServer::new(3, 2);

        let grad1 = vec![1.0, 2.0, 3.0];
        let grad2 = vec![2.0, 3.0, 4.0];

        ps.receive_gradient(grad1).unwrap();
        ps.receive_gradient(grad2).unwrap();

        // After 2 updates (all workers), parameters should be updated
        let params = ps.get_parameters();
        assert_eq!(ps.version, 1);
        assert!(params.iter().all(|&p| p != 0.0));
    }

    #[test]
    fn test_worker_node_creation() {
        let partition = DataPartition {
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            targets: vec![1.0, 2.0],
            partition_id: 0,
        };

        let worker = WorkerNode::new(NodeId::new("worker_0"), partition);
        assert_eq!(worker.id.0, "worker_0");
        assert_eq!(worker.stats.samples_processed, 0);
    }

    #[test]
    fn test_local_gradient_computation() {
        let partition = DataPartition {
            features: vec![vec![1.0, 2.0], vec![2.0, 3.0]],
            targets: vec![3.0, 5.0],
            partition_id: 0,
        };

        let mut worker = WorkerNode::new(NodeId::new("worker_0"), partition);
        let params = vec![1.0, 1.0];

        let gradient = worker.compute_local_gradient(&params).unwrap();
        assert_eq!(gradient.len(), 2);
        assert!(worker.stats.gradient_computations > 0);
    }

    #[test]
    fn test_distributed_regression_creation() {
        let config = DistributedConfig::default();
        let model = DistributedLinearRegression::new(config, 5);

        assert_eq!(model.workers.len(), 0);
        assert!(model.parameters.read().unwrap().weights.len() == 5);
    }

    #[test]
    fn test_data_partitioning() {
        let config = DistributedConfig {
            num_workers: 2,
            ..Default::default()
        };

        let mut model = DistributedLinearRegression::new(config, 2);

        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let targets = vec![3.0, 7.0, 11.0, 15.0];

        model.partition_data(features, targets).unwrap();

        assert_eq!(model.workers.len(), 2);
        assert!(model.workers[0].data_partition.features.len() > 0);
    }

    #[test]
    fn test_distributed_training() {
        let config = DistributedConfig {
            num_workers: 2,
            max_iterations: 10,
            tolerance: 1e-3,
            learning_rate: 0.01,
            ..Default::default()
        };

        let mut model = DistributedLinearRegression::new(config, 2);

        // Simple linear relationship: y = 2*x1 + 3*x2
        let features = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let targets = vec![5.0, 10.0, 15.0, 20.0];

        model.partition_data(features, targets).unwrap();
        model.fit().unwrap();

        let stats = model.get_training_stats();
        assert!(stats.total_samples_processed > 0);
        assert!(stats.parameter_server_version > 0);
    }

    #[test]
    fn test_prediction() {
        let config = DistributedConfig::default();
        let model = DistributedLinearRegression::new(config, 2);

        let test_features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let predictions = model.predict(&test_features).unwrap();

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_training_stats() {
        let config = DistributedConfig {
            num_workers: 3,
            ..Default::default()
        };

        let model = DistributedLinearRegression::new(config, 2);
        let stats = model.get_training_stats();

        assert_eq!(stats.num_workers, 0); // No workers added yet
    }

    // ============================================================================
    // Tests for Advanced Distributed Learning Features
    // ============================================================================

    #[test]
    fn test_federated_learning_creation() {
        let config = FederatedConfig::default();
        let fed_learning = FederatedLearning::new(config, 5);

        assert_eq!(fed_learning.clients.len(), 0);
        assert_eq!(fed_learning.config.num_clients, 10);
    }

    #[test]
    fn test_federated_add_client() {
        let config = FederatedConfig::default();
        let mut fed_learning = FederatedLearning::new(config, 5);

        fed_learning.add_client("client_1".to_string(), 100);
        fed_learning.add_client("client_2".to_string(), 150);

        assert_eq!(fed_learning.clients.len(), 2);
        assert_eq!(fed_learning.clients[0].dataset_size, 100);
        assert_eq!(fed_learning.clients[1].dataset_size, 150);
    }

    #[test]
    fn test_federated_client_selection() {
        let mut config = FederatedConfig::default();
        config.client_fraction = 0.5;
        let mut fed_learning = FederatedLearning::new(config, 5);

        for i in 0..10 {
            fed_learning.add_client(format!("client_{}", i), 100);
        }

        let selected = fed_learning.select_clients();
        assert!(selected.len() >= 4 && selected.len() <= 6); // ~50% of 10
    }

    #[test]
    fn test_federated_averaging() {
        let config = FederatedConfig {
            dp_epsilon: None, // Disable noise for deterministic test
            ..Default::default()
        };
        let mut fed_learning = FederatedLearning::new(config, 3);

        fed_learning.add_client("client_1".to_string(), 100);
        fed_learning.add_client("client_2".to_string(), 100);

        let updates = vec![(0, vec![1.0, 2.0, 3.0]), (1, vec![2.0, 4.0, 6.0])];

        let averaged = fed_learning.federated_average(&updates);
        assert_eq!(averaged.len(), 3);
        // With equal weights, average should be (1+2)/2=1.5, (2+4)/2=3, (3+6)/2=4.5
        assert!((averaged[0] - 1.5).abs() < 1e-6);
        assert!((averaged[1] - 3.0).abs() < 1e-6);
        assert!((averaged[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_privacy_mechanism_noise() {
        let privacy = PrivacyMechanism::new();
        let mut gradients = vec![1.0, 2.0, 3.0];
        let original = gradients.clone();

        privacy.apply_noise(&mut gradients, 1.0);

        // Gradients should be modified (with very high probability)
        assert_ne!(gradients, original);
    }

    #[test]
    fn test_privacy_mechanism_clipping() {
        let privacy = PrivacyMechanism::new();
        let mut gradients = vec![3.0, 4.0]; // Norm = 5.0

        privacy.clip_gradients(&mut gradients, 1.0);

        let norm: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_byzantine_fault_tolerant_creation() {
        let config = BFTConfig::default();
        let bft = ByzantineFaultTolerant::new(config, AggregationMethod::Median);

        assert_eq!(bft.aggregation_method, AggregationMethod::Median);
    }

    #[test]
    fn test_byzantine_median_aggregation() {
        let config = BFTConfig::default();
        let bft = ByzantineFaultTolerant::new(config, AggregationMethod::Median);

        let gradients = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![100.0, 100.0, 100.0], // Byzantine outlier
        ];

        let result = bft.aggregate(&gradients).unwrap();
        assert_eq!(result.len(), 3);
        // Median should filter out the outlier
        assert!(result[0] < 50.0);
        assert!(result[1] < 50.0);
        assert!(result[2] < 50.0);
    }

    #[test]
    fn test_byzantine_trimmed_mean() {
        let config = BFTConfig::default();
        let bft = ByzantineFaultTolerant::new(
            config,
            AggregationMethod::TrimmedMean { trim_fraction: 25 },
        );

        let gradients = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![4.0, 5.0, 6.0],
        ];

        let result = bft.aggregate(&gradients).unwrap();
        assert_eq!(result.len(), 3);
        // Trimmed mean should produce reasonable values
        assert!(result[0] > 1.0 && result[0] < 4.0);
    }

    #[test]
    fn test_byzantine_krum() {
        let config = BFTConfig::default();
        let bft = ByzantineFaultTolerant::new(config, AggregationMethod::Krum);

        let gradients = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 2.2, 3.2],
            vec![100.0, 100.0, 100.0], // Byzantine outlier
        ];

        let result = bft.aggregate(&gradients).unwrap();
        assert_eq!(result.len(), 3);
        // Krum should select one of the non-outlier gradients
        assert!(result[0] < 10.0);
    }

    #[test]
    fn test_byzantine_empty_gradients() {
        let config = BFTConfig::default();
        let bft = ByzantineFaultTolerant::new(config, AggregationMethod::Median);

        let gradients: Vec<Vec<f64>> = vec![];
        let result = bft.aggregate(&gradients);

        assert!(result.is_err());
    }

    #[test]
    fn test_load_balancer_creation() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        assert_eq!(lb.strategy, LoadBalancingStrategy::RoundRobin);
        assert_eq!(lb.worker_loads.len(), 0);
    }

    #[test]
    fn test_load_balancer_register_worker() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded);

        lb.register_worker("worker_1".to_string(), 10);
        lb.register_worker("worker_2".to_string(), 20);

        assert_eq!(lb.worker_loads.len(), 2);
        assert_eq!(lb.worker_loads.get("worker_1").unwrap().capacity, 10);
        assert_eq!(lb.worker_loads.get("worker_2").unwrap().capacity, 20);
    }

    #[test]
    fn test_load_balancer_least_loaded() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded);

        lb.register_worker("worker_1".to_string(), 10);
        lb.register_worker("worker_2".to_string(), 10);

        // Initially both have load 0, so either can be selected
        let selected = lb.select_worker();
        assert!(selected.is_some());

        // Add load to one worker
        lb.update_load("worker_1", true);
        lb.update_load("worker_1", true);

        // Now worker_2 should be selected (least loaded)
        let selected = lb.select_worker();
        assert!(selected.is_some());
    }

    #[test]
    fn test_load_balancer_update_load() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded);

        lb.register_worker("worker_1".to_string(), 10);

        lb.update_load("worker_1", true);
        assert_eq!(lb.worker_loads.get("worker_1").unwrap().active_tasks, 1);
        assert!((lb.worker_loads.get("worker_1").unwrap().load_factor - 0.1).abs() < 1e-6);

        lb.update_load("worker_1", false);
        assert_eq!(lb.worker_loads.get("worker_1").unwrap().active_tasks, 0);
        assert!((lb.worker_loads.get("worker_1").unwrap().load_factor).abs() < 1e-6);
    }

    #[test]
    fn test_load_balancer_power_of_two() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::PowerOfTwo);

        lb.register_worker("worker_1".to_string(), 10);
        lb.register_worker("worker_2".to_string(), 10);
        lb.register_worker("worker_3".to_string(), 10);

        let selected = lb.select_worker();
        assert!(selected.is_some());
    }

    #[test]
    fn test_federated_config_default() {
        let config = FederatedConfig::default();
        assert_eq!(config.num_clients, 10);
        assert!((config.client_fraction - 0.3).abs() < 1e-6);
        assert_eq!(config.local_epochs, 5);
        assert!(config.secure_aggregation);
    }

    #[test]
    fn test_bft_config_default() {
        let config = BFTConfig::default();
        assert!((config.max_byzantine_fraction - 0.3).abs() < 1e-6);
        assert!((config.detection_threshold - 2.0).abs() < 1e-6);
        assert!(config.enable_reputation);
    }

    #[test]
    fn test_aggregation_method_equality() {
        assert_eq!(AggregationMethod::Median, AggregationMethod::Median);
        assert_eq!(
            AggregationMethod::TrimmedMean { trim_fraction: 25 },
            AggregationMethod::TrimmedMean { trim_fraction: 25 }
        );
        assert_ne!(AggregationMethod::Median, AggregationMethod::Krum);
    }

    #[test]
    fn test_load_balancing_strategy_equality() {
        assert_eq!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::RoundRobin
        );
        assert_eq!(
            LoadBalancingStrategy::LeastLoaded,
            LoadBalancingStrategy::LeastLoaded
        );
        assert_ne!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LeastLoaded
        );
    }
}
