//! Federated learning algorithms for distributed optimization
//!
//! This module provides comprehensive federated learning capabilities including
//! privacy-preserving algorithms, client-server coordination, communication efficiency,
//! and SIMD-accelerated federated optimization.

use scirs2_core::ndarray::{Array, Array1, Array2, Array3, Axis};
use scirs2_core::error::{CoreError, Result as SklResult};
use scirs2_core::random::{Random, rng};
use scirs2_core::SliceRandomExt;
use scirs2_core::simd::{SimdArray, SimdOps};
use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy};
use scirs2_core::memory::{BufferPool, MemoryMetricsCollector};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram};

use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{Duration, Instant, SystemTime};
use std::thread;

/// Federated optimizer trait for distributed learning algorithms
pub trait FederatedOptimizer: Send + Sync {
    fn initialize_federation(&mut self, clients: &[ClientInfo]) -> SklResult<()>;
    fn aggregate_updates(&mut self, client_updates: &[ClientUpdate]) -> SklResult<GlobalUpdate>;
    fn distribute_global_model(&mut self, global_update: &GlobalUpdate) -> SklResult<()>;
    fn handle_client_dropout(&mut self, dropped_clients: &[String]) -> SklResult<()>;
    fn get_convergence_status(&self) -> ConvergenceStatus;
    fn apply_privacy_mechanism(&mut self, mechanism: PrivacyMechanism) -> SklResult<()>;
    fn optimize_communication(&mut self, strategy: CommunicationStrategy) -> SklResult<()>;
}

/// Client information for federated learning
#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub client_id: String,
    pub data_size: usize,
    pub computation_capacity: f64,
    pub communication_bandwidth: f64,
    pub privacy_budget: f64,
    pub reliability_score: f64,
    pub geographic_location: String,
    pub last_seen: SystemTime,
}

/// Client update containing local model changes
#[derive(Debug, Clone)]
pub struct ClientUpdate {
    pub client_id: String,
    pub model_parameters: Array2<f64>,
    pub gradient_updates: Array2<f64>,
    pub training_samples: usize,
    pub training_loss: f64,
    pub compression_ratio: f64,
    pub privacy_noise_variance: f64,
    pub update_timestamp: SystemTime,
}

/// Global model update from server aggregation
#[derive(Debug, Clone)]
pub struct GlobalUpdate {
    pub global_parameters: Array2<f64>,
    pub aggregated_gradients: Array2<f64>,
    pub round_number: u64,
    pub participating_clients: Vec<String>,
    pub convergence_metric: f64,
    pub update_timestamp: SystemTime,
}

/// Convergence status for federated optimization
#[derive(Debug, Clone)]
pub struct ConvergenceStatus {
    pub is_converged: bool,
    pub convergence_metric: f64,
    pub rounds_completed: u64,
    pub estimated_rounds_remaining: Option<u64>,
    pub convergence_rate: f64,
}

/// Privacy mechanisms for federated learning
#[derive(Debug, Clone)]
pub enum PrivacyMechanism {
    DifferentialPrivacy { epsilon: f64, delta: f64 },
    SecureAggregation { threshold: usize },
    Homomorphic { encryption_key: String },
    LocalPrivacy { noise_multiplier: f64 },
    None,
}

/// Communication optimization strategies
#[derive(Debug, Clone)]
pub enum CommunicationStrategy {
    Compression { compression_ratio: f64 },
    Quantization { bits: u8 },
    Sparsification { sparsity_ratio: f64 },
    Gradient Clipping { max_norm: f64 },
    AdaptiveCommunication,
}

/// SIMD-accelerated federated computation engine
pub struct SimdFederatedAccelerator {
    computation_buffers: HashMap<String, Array2<f64>>,
    aggregation_cache: HashMap<String, Array2<f64>>,
    simd_arrays: HashMap<String, SimdArray<f64>>,
    compression_engine: Arc<Mutex<CompressionEngine>>,
    privacy_engine: Arc<Mutex<PrivacyEngine>>,
}

impl SimdFederatedAccelerator {
    pub fn new() -> Self {
        Self {
            computation_buffers: HashMap::new(),
            aggregation_cache: HashMap::new(),
            simd_arrays: HashMap::new(),
            compression_engine: Arc::new(Mutex::new(CompressionEngine::new())),
            privacy_engine: Arc::new(Mutex::new(PrivacyEngine::new())),
        }
    }

    pub fn vectorized_aggregation(&mut self, client_updates: &[ClientUpdate]) -> SklResult<Array2<f64>> {
        if client_updates.is_empty() {
            return Err(CoreError::InvalidInput("No client updates provided".to_string()));
        }

        let param_shape = client_updates[0].model_parameters.dim();
        let mut aggregated = Array2::zeros(param_shape);

        // SIMD-accelerated weighted aggregation
        for update in client_updates {
            let weight = update.training_samples as f64;
            let simd_params = SimdArray::from_array(&update.model_parameters);
            let weighted_params = simd_params.scalar_multiply(weight);

            aggregated = aggregated + weighted_params.to_array2(param_shape.0, param_shape.1)?;
        }

        let total_samples: usize = client_updates.iter().map(|u| u.training_samples).sum();
        aggregated = aggregated / total_samples as f64;

        Ok(aggregated)
    }

    pub fn apply_compression(&mut self, parameters: &Array2<f64>, compression_ratio: f64) -> SklResult<Array2<f64>> {
        if let Ok(mut engine) = self.compression_engine.lock() {
            engine.compress_parameters(parameters, compression_ratio)
        } else {
            Err(CoreError::InvalidInput("Compression engine unavailable".to_string()))
        }
    }

    pub fn apply_privacy_noise(&mut self, parameters: &Array2<f64>, noise_variance: f64) -> SklResult<Array2<f64>> {
        if let Ok(mut engine) = self.privacy_engine.lock() {
            engine.add_differential_privacy_noise(parameters, noise_variance)
        } else {
            Err(CoreError::InvalidInput("Privacy engine unavailable".to_string()))
        }
    }
}

/// FedAvg (Federated Averaging) algorithm implementation
pub struct FedAvgOptimizer {
    client_registry: HashMap<String, ClientInfo>,
    global_model: Option<Array2<f64>>,
    round_number: AtomicU64,
    convergence_threshold: f64,
    min_clients_per_round: usize,
    max_rounds: u64,
    client_selection_strategy: ClientSelectionStrategy,
    aggregation_weights: HashMap<String, f64>,
    simd_accelerator: Arc<Mutex<SimdFederatedAccelerator>>,
    privacy_mechanism: PrivacyMechanism,
    communication_strategy: CommunicationStrategy,
    convergence_history: VecDeque<f64>,
    performance_monitor: Arc<Mutex<FederatedPerformanceMonitor>>,
}

#[derive(Debug, Clone)]
pub enum ClientSelectionStrategy {
    Random { fraction: f64 },
    PowerOfChoice { candidates: usize },
    DataBasedSampling,
    ReliabilityBased,
    GeographicDiversity,
}

impl FedAvgOptimizer {
    pub fn new(convergence_threshold: f64, min_clients_per_round: usize, max_rounds: u64) -> Self {
        Self {
            client_registry: HashMap::new(),
            global_model: None,
            round_number: AtomicU64::new(0),
            convergence_threshold,
            min_clients_per_round,
            max_rounds,
            client_selection_strategy: ClientSelectionStrategy::Random { fraction: 0.1 },
            aggregation_weights: HashMap::new(),
            simd_accelerator: Arc::new(Mutex::new(SimdFederatedAccelerator::new())),
            privacy_mechanism: PrivacyMechanism::None,
            communication_strategy: CommunicationStrategy::AdaptiveCommunication,
            convergence_history: VecDeque::with_capacity(1000),
            performance_monitor: Arc::new(Mutex::new(FederatedPerformanceMonitor::new())),
        }
    }

    pub fn set_client_selection_strategy(&mut self, strategy: ClientSelectionStrategy) {
        self.client_selection_strategy = strategy;
    }

    pub fn select_clients(&self, available_clients: &[String]) -> SklResult<Vec<String>> {
        match &self.client_selection_strategy {
            ClientSelectionStrategy::Random { fraction } => {
                let num_select = (available_clients.len() as f64 * fraction).ceil() as usize;
                let num_select = num_select.max(self.min_clients_per_round).min(available_clients.len());

                let mut rng = rng();
                let mut selected = available_clients.to_vec();
                rng.shuffle(&mut selected);
                selected.truncate(num_select);

                Ok(selected)
            },
            ClientSelectionStrategy::PowerOfChoice { candidates } => {
                if available_clients.len() <= *candidates {
                    return Ok(available_clients.to_vec());
                }

                let mut rng = rng();
                let mut candidate_scores = Vec::new();

                for _ in 0..*candidates {
                    let client_idx = rng.gen_range(0..available_clients.len());
                    let client_id = &available_clients[client_idx];

                    if let Some(client_info) = self.client_registry.get(client_id) {
                        let score = client_info.reliability_score * client_info.computation_capacity;
                        candidate_scores.push((client_id.clone(), score));
                    }
                }

                candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let selected = candidate_scores.into_iter()
                    .take(self.min_clients_per_round)
                    .map(|(id, _)| id)
                    .collect();

                Ok(selected)
            },
            ClientSelectionStrategy::DataBasedSampling => {
                let mut client_weights: Vec<(String, f64)> = available_clients.iter()
                    .filter_map(|id| {
                        self.client_registry.get(id).map(|info| {
                            (id.clone(), info.data_size as f64)
                        })
                    })
                    .collect();

                client_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let selected = client_weights.into_iter()
                    .take(self.min_clients_per_round)
                    .map(|(id, _)| id)
                    .collect();

                Ok(selected)
            },
            ClientSelectionStrategy::ReliabilityBased => {
                let mut client_scores: Vec<(String, f64)> = available_clients.iter()
                    .filter_map(|id| {
                        self.client_registry.get(id).map(|info| {
                            (id.clone(), info.reliability_score)
                        })
                    })
                    .collect();

                client_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let selected = client_scores.into_iter()
                    .take(self.min_clients_per_round)
                    .map(|(id, _)| id)
                    .collect();

                Ok(selected)
            },
            ClientSelectionStrategy::GeographicDiversity => {
                let mut location_groups: HashMap<String, Vec<String>> = HashMap::new();

                for client_id in available_clients {
                    if let Some(client_info) = self.client_registry.get(client_id) {
                        location_groups.entry(client_info.geographic_location.clone())
                            .or_insert_with(Vec::new)
                            .push(client_id.clone());
                    }
                }

                let mut selected = Vec::new();
                let clients_per_location = self.min_clients_per_round / location_groups.len().max(1);

                for clients_in_location in location_groups.values() {
                    let take_count = clients_per_location.min(clients_in_location.len());
                    selected.extend(clients_in_location.iter().take(take_count).cloned());
                }

                Ok(selected)
            },
        }
    }

    fn calculate_aggregation_weights(&mut self, client_updates: &[ClientUpdate]) -> SklResult<()> {
        let total_samples: usize = client_updates.iter().map(|u| u.training_samples).sum();

        for update in client_updates {
            let weight = update.training_samples as f64 / total_samples as f64;
            self.aggregation_weights.insert(update.client_id.clone(), weight);
        }

        Ok(())
    }
}

impl FederatedOptimizer for FedAvgOptimizer {
    fn initialize_federation(&mut self, clients: &[ClientInfo]) -> SklResult<()> {
        for client in clients {
            self.client_registry.insert(client.client_id.clone(), client.clone());
        }

        // Initialize global model if needed
        if self.global_model.is_none() {
            // Default initialization - in practice would be provided
            self.global_model = Some(Array2::zeros((100, 10)));
        }

        Ok(())
    }

    fn aggregate_updates(&mut self, client_updates: &[ClientUpdate]) -> SklResult<GlobalUpdate> {
        if client_updates.len() < self.min_clients_per_round {
            return Err(CoreError::InvalidInput(
                format!("Insufficient client updates: {} < {}", client_updates.len(), self.min_clients_per_round)
            ));
        }

        self.calculate_aggregation_weights(client_updates)?;

        let aggregated_parameters = if let Ok(mut accelerator) = self.simd_accelerator.lock() {
            accelerator.vectorized_aggregation(client_updates)?
        } else {
            return Err(CoreError::InvalidInput("SIMD accelerator unavailable".to_string()));
        };

        // Apply privacy mechanism
        let final_parameters = match &self.privacy_mechanism {
            PrivacyMechanism::DifferentialPrivacy { epsilon: _, delta: _ } => {
                if let Ok(mut accelerator) = self.simd_accelerator.lock() {
                    accelerator.apply_privacy_noise(&aggregated_parameters, 0.1)?
                } else {
                    aggregated_parameters
                }
            },
            _ => aggregated_parameters,
        };

        self.global_model = Some(final_parameters.clone());

        let round_num = self.round_number.fetch_add(1, Ordering::SeqCst);
        let participating_clients = client_updates.iter().map(|u| u.client_id.clone()).collect();

        // Calculate convergence metric
        let convergence_metric = self.calculate_convergence_metric(client_updates);
        self.convergence_history.push_back(convergence_metric);

        Ok(GlobalUpdate {
            global_parameters: final_parameters,
            aggregated_gradients: Array2::zeros((100, 10)), // Placeholder
            round_number: round_num,
            participating_clients,
            convergence_metric,
            update_timestamp: SystemTime::now(),
        })
    }

    fn distribute_global_model(&mut self, global_update: &GlobalUpdate) -> SklResult<()> {
        // Apply communication optimization
        let optimized_parameters = match &self.communication_strategy {
            CommunicationStrategy::Compression { compression_ratio } => {
                if let Ok(mut accelerator) = self.simd_accelerator.lock() {
                    accelerator.apply_compression(&global_update.global_parameters, *compression_ratio)?
                } else {
                    global_update.global_parameters.clone()
                }
            },
            _ => global_update.global_parameters.clone(),
        };

        // In a real implementation, this would send the model to clients
        // For now, we just store it
        self.global_model = Some(optimized_parameters);

        Ok(())
    }

    fn handle_client_dropout(&mut self, dropped_clients: &[String]) -> SklResult<()> {
        for client_id in dropped_clients {
            if let Some(client_info) = self.client_registry.get_mut(client_id) {
                // Reduce reliability score for dropped clients
                client_info.reliability_score *= 0.9;
            }
        }

        Ok(())
    }

    fn get_convergence_status(&self) -> ConvergenceStatus {
        let current_round = self.round_number.load(Ordering::SeqCst);
        let latest_metric = self.convergence_history.back().copied().unwrap_or(1.0);

        let is_converged = latest_metric < self.convergence_threshold || current_round >= self.max_rounds;

        let convergence_rate = if self.convergence_history.len() >= 2 {
            let recent_metrics: Vec<f64> = self.convergence_history.iter().rev().take(10).copied().collect();
            self.calculate_convergence_rate(&recent_metrics)
        } else {
            0.0
        };

        let estimated_rounds_remaining = if is_converged {
            None
        } else {
            Some(((latest_metric - self.convergence_threshold) / convergence_rate.max(0.001)) as u64)
        };

        ConvergenceStatus {
            is_converged,
            convergence_metric: latest_metric,
            rounds_completed: current_round,
            estimated_rounds_remaining,
            convergence_rate,
        }
    }

    fn apply_privacy_mechanism(&mut self, mechanism: PrivacyMechanism) -> SklResult<()> {
        self.privacy_mechanism = mechanism;
        Ok(())
    }

    fn optimize_communication(&mut self, strategy: CommunicationStrategy) -> SklResult<()> {
        self.communication_strategy = strategy;
        Ok(())
    }
}

impl FedAvgOptimizer {
    fn calculate_convergence_metric(&self, client_updates: &[ClientUpdate]) -> f64 {
        if client_updates.is_empty() {
            return 1.0;
        }

        // Simple convergence metric based on loss variance
        let losses: Vec<f64> = client_updates.iter().map(|u| u.training_loss).collect();
        let mean_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        let variance = losses.iter().map(|l| (l - mean_loss).powi(2)).sum::<f64>() / losses.len() as f64;

        variance.sqrt()
    }

    fn calculate_convergence_rate(&self, recent_metrics: &[f64]) -> f64 {
        if recent_metrics.len() < 2 {
            return 0.0;
        }

        let mut total_change = 0.0;
        for i in 1..recent_metrics.len() {
            total_change += recent_metrics[i-1] - recent_metrics[i];
        }

        total_change / (recent_metrics.len() - 1) as f64
    }
}

/// FedProx algorithm with proximal regularization
pub struct FedProxOptimizer {
    base_optimizer: FedAvgOptimizer,
    proximal_parameter: f64,
    proximal_regularizer: Arc<Mutex<ProximalRegularizer>>,
}

impl FedProxOptimizer {
    pub fn new(convergence_threshold: f64, min_clients_per_round: usize, max_rounds: u64, mu: f64) -> Self {
        Self {
            base_optimizer: FedAvgOptimizer::new(convergence_threshold, min_clients_per_round, max_rounds),
            proximal_parameter: mu,
            proximal_regularizer: Arc::new(Mutex::new(ProximalRegularizer::new(mu))),
        }
    }
}

impl FederatedOptimizer for FedProxOptimizer {
    fn initialize_federation(&mut self, clients: &[ClientInfo]) -> SklResult<()> {
        self.base_optimizer.initialize_federation(clients)
    }

    fn aggregate_updates(&mut self, client_updates: &[ClientUpdate]) -> SklResult<GlobalUpdate> {
        // Apply proximal regularization before aggregation
        let regularized_updates = if let Ok(regularizer) = self.proximal_regularizer.lock() {
            regularizer.apply_regularization(client_updates)?
        } else {
            client_updates.to_vec()
        };

        self.base_optimizer.aggregate_updates(&regularized_updates)
    }

    fn distribute_global_model(&mut self, global_update: &GlobalUpdate) -> SklResult<()> {
        self.base_optimizer.distribute_global_model(global_update)
    }

    fn handle_client_dropout(&mut self, dropped_clients: &[String]) -> SklResult<()> {
        self.base_optimizer.handle_client_dropout(dropped_clients)
    }

    fn get_convergence_status(&self) -> ConvergenceStatus {
        self.base_optimizer.get_convergence_status()
    }

    fn apply_privacy_mechanism(&mut self, mechanism: PrivacyMechanism) -> SklResult<()> {
        self.base_optimizer.apply_privacy_mechanism(mechanism)
    }

    fn optimize_communication(&mut self, strategy: CommunicationStrategy) -> SklResult<()> {
        self.base_optimizer.optimize_communication(strategy)
    }
}

/// Proximal regularizer for FedProx
pub struct ProximalRegularizer {
    mu: f64,
    global_model_reference: Option<Array2<f64>>,
}

impl ProximalRegularizer {
    pub fn new(mu: f64) -> Self {
        Self {
            mu,
            global_model_reference: None,
        }
    }

    pub fn set_global_reference(&mut self, global_model: Array2<f64>) {
        self.global_model_reference = Some(global_model);
    }

    pub fn apply_regularization(&self, client_updates: &[ClientUpdate]) -> SklResult<Vec<ClientUpdate>> {
        if let Some(ref global_ref) = self.global_model_reference {
            let mut regularized_updates = Vec::new();

            for update in client_updates {
                let mut regularized_params = update.model_parameters.clone();

                // Apply proximal term: w - mu * (w - w_global)
                let diff = &regularized_params - global_ref;
                regularized_params = regularized_params - self.mu * diff;

                let mut regularized_update = update.clone();
                regularized_update.model_parameters = regularized_params;
                regularized_updates.push(regularized_update);
            }

            Ok(regularized_updates)
        } else {
            Ok(client_updates.to_vec())
        }
    }
}

/// Compression engine for communication efficiency
pub struct CompressionEngine {
    compression_algorithms: HashMap<String, Box<dyn CompressionAlgorithm>>,
    default_algorithm: String,
}

pub trait CompressionAlgorithm: Send + Sync {
    fn compress(&self, data: &Array2<f64>, compression_ratio: f64) -> SklResult<Array2<f64>>;
    fn decompress(&self, compressed_data: &Array2<f64>) -> SklResult<Array2<f64>>;
    fn get_algorithm_name(&self) -> &str;
}

impl CompressionEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            compression_algorithms: HashMap::new(),
            default_algorithm: "top_k".to_string(),
        };

        engine.compression_algorithms.insert("top_k".to_string(), Box::new(TopKCompression));
        engine.compression_algorithms.insert("quantization".to_string(), Box::new(QuantizationCompression));

        engine
    }

    pub fn compress_parameters(&self, parameters: &Array2<f64>, compression_ratio: f64) -> SklResult<Array2<f64>> {
        if let Some(algorithm) = self.compression_algorithms.get(&self.default_algorithm) {
            algorithm.compress(parameters, compression_ratio)
        } else {
            Ok(parameters.clone())
        }
    }
}

/// Top-K compression algorithm
pub struct TopKCompression;

impl CompressionAlgorithm for TopKCompression {
    fn compress(&self, data: &Array2<f64>, compression_ratio: f64) -> SklResult<Array2<f64>> {
        let mut flat_data: Vec<f64> = data.iter().copied().collect();
        flat_data.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());

        let keep_count = ((flat_data.len() as f64) * compression_ratio) as usize;
        let threshold = flat_data.get(keep_count).copied().unwrap_or(0.0);

        let compressed = data.mapv(|x| if x.abs() >= threshold.abs() { x } else { 0.0 });
        Ok(compressed)
    }

    fn decompress(&self, compressed_data: &Array2<f64>) -> SklResult<Array2<f64>> {
        Ok(compressed_data.clone())
    }

    fn get_algorithm_name(&self) -> &str {
        "top_k"
    }
}

/// Quantization compression algorithm
pub struct QuantizationCompression;

impl CompressionAlgorithm for QuantizationCompression {
    fn compress(&self, data: &Array2<f64>, compression_ratio: f64) -> SklResult<Array2<f64>> {
        let levels = (256.0 * compression_ratio) as i32;
        let max_val = data.iter().fold(0.0, |acc, &x| acc.max(x.abs()));

        if max_val == 0.0 {
            return Ok(data.clone());
        }

        let quantized = data.mapv(|x| {
            let normalized = x / max_val;
            let quantized_level = (normalized * levels as f64).round();
            (quantized_level / levels as f64) * max_val
        });

        Ok(quantized)
    }

    fn decompress(&self, compressed_data: &Array2<f64>) -> SklResult<Array2<f64>> {
        Ok(compressed_data.clone())
    }

    fn get_algorithm_name(&self) -> &str {
        "quantization"
    }
}

/// Privacy engine for differential privacy and secure aggregation
pub struct PrivacyEngine {
    noise_generator: Random,
    privacy_budget_tracker: HashMap<String, f64>,
    secure_aggregation_threshold: usize,
}

impl PrivacyEngine {
    pub fn new() -> Self {
        Self {
            noise_generator: Random::new(),
            privacy_budget_tracker: HashMap::new(),
            secure_aggregation_threshold: 5,
        }
    }

    pub fn add_differential_privacy_noise(&mut self, parameters: &Array2<f64>, noise_variance: f64) -> SklResult<Array2<f64>> {
        let shape = parameters.dim();
        let mut noise = Array2::zeros(shape);

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let gaussian_noise = self.noise_generator.normal(0.0, noise_variance);
                noise[(i, j)] = gaussian_noise;
            }
        }

        Ok(parameters + noise)
    }

    pub fn check_privacy_budget(&self, client_id: &str, epsilon_request: f64) -> bool {
        if let Some(current_budget) = self.privacy_budget_tracker.get(client_id) {
            current_budget + epsilon_request <= 1.0 // Assuming budget limit of 1.0
        } else {
            epsilon_request <= 1.0
        }
    }

    pub fn consume_privacy_budget(&mut self, client_id: &str, epsilon_used: f64) {
        let current_budget = self.privacy_budget_tracker.get(client_id).copied().unwrap_or(0.0);
        self.privacy_budget_tracker.insert(client_id.to_string(), current_budget + epsilon_used);
    }
}

/// Federated performance monitor
pub struct FederatedPerformanceMonitor {
    metrics_registry: Arc<Mutex<MetricRegistry>>,
    round_performance: HashMap<u64, RoundPerformance>,
    client_performance: HashMap<String, ClientPerformanceHistory>,
    convergence_tracker: ConvergenceTracker,
}

#[derive(Debug, Clone)]
pub struct RoundPerformance {
    pub round_number: u64,
    pub participating_clients: usize,
    pub aggregation_time: Duration,
    pub communication_overhead: f64,
    pub convergence_metric: f64,
    pub privacy_cost: f64,
}

#[derive(Debug, Clone)]
pub struct ClientPerformanceHistory {
    pub client_id: String,
    pub participation_rate: f64,
    pub average_training_time: Duration,
    pub data_quality_score: f64,
    pub communication_efficiency: f64,
    pub dropout_incidents: u32,
}

#[derive(Debug)]
pub struct ConvergenceTracker {
    pub convergence_history: VecDeque<f64>,
    pub convergence_rate: f64,
    pub stagnation_counter: u32,
    pub early_stopping_threshold: f64,
}

impl FederatedPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics_registry: Arc::new(Mutex::new(MetricRegistry::new())),
            round_performance: HashMap::new(),
            client_performance: HashMap::new(),
            convergence_tracker: ConvergenceTracker {
                convergence_history: VecDeque::with_capacity(1000),
                convergence_rate: 0.0,
                stagnation_counter: 0,
                early_stopping_threshold: 1e-6,
            },
        }
    }

    pub fn record_round_performance(&mut self, performance: RoundPerformance) {
        self.round_performance.insert(performance.round_number, performance.clone());
        self.convergence_tracker.convergence_history.push_back(performance.convergence_metric);

        // Update convergence rate
        if self.convergence_tracker.convergence_history.len() >= 2 {
            let recent: Vec<f64> = self.convergence_tracker.convergence_history
                .iter().rev().take(5).copied().collect();
            self.convergence_tracker.convergence_rate = self.calculate_convergence_rate(&recent);
        }

        // Check for stagnation
        if self.convergence_tracker.convergence_rate.abs() < self.convergence_tracker.early_stopping_threshold {
            self.convergence_tracker.stagnation_counter += 1;
        } else {
            self.convergence_tracker.stagnation_counter = 0;
        }
    }

    pub fn update_client_performance(&mut self, client_id: &str, training_time: Duration, data_quality: f64) {
        let performance = self.client_performance.entry(client_id.to_string())
            .or_insert_with(|| ClientPerformanceHistory {
                client_id: client_id.to_string(),
                participation_rate: 0.0,
                average_training_time: Duration::from_secs(0),
                data_quality_score: data_quality,
                communication_efficiency: 1.0,
                dropout_incidents: 0,
            });

        // Update average training time (simple moving average)
        performance.average_training_time = Duration::from_nanos(
            (performance.average_training_time.as_nanos() + training_time.as_nanos()) / 2
        );
        performance.data_quality_score = (performance.data_quality_score + data_quality) / 2.0;
    }

    fn calculate_convergence_rate(&self, recent_metrics: &[f64]) -> f64 {
        if recent_metrics.len() < 2 {
            return 0.0;
        }

        let mut total_change = 0.0;
        for i in 1..recent_metrics.len() {
            total_change += recent_metrics[i-1] - recent_metrics[i];
        }

        total_change / (recent_metrics.len() - 1) as f64
    }

    pub fn get_federated_metrics_summary(&self) -> FederatedMetricsSummary {
        let total_rounds = self.round_performance.len() as u64;
        let avg_participants = if total_rounds > 0 {
            self.round_performance.values()
                .map(|r| r.participating_clients)
                .sum::<usize>() as f64 / total_rounds as f64
        } else {
            0.0
        };

        let avg_convergence = if !self.convergence_tracker.convergence_history.is_empty() {
            self.convergence_tracker.convergence_history.iter().sum::<f64>()
                / self.convergence_tracker.convergence_history.len() as f64
        } else {
            0.0
        };

        FederatedMetricsSummary {
            total_rounds,
            average_participants_per_round: avg_participants,
            current_convergence_rate: self.convergence_tracker.convergence_rate,
            average_convergence_metric: avg_convergence,
            stagnation_rounds: self.convergence_tracker.stagnation_counter,
            total_active_clients: self.client_performance.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FederatedMetricsSummary {
    pub total_rounds: u64,
    pub average_participants_per_round: f64,
    pub current_convergence_rate: f64,
    pub average_convergence_metric: f64,
    pub stagnation_rounds: u32,
    pub total_active_clients: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fedavg_optimizer_creation() {
        let optimizer = FedAvgOptimizer::new(0.01, 5, 100);
        assert_eq!(optimizer.convergence_threshold, 0.01);
        assert_eq!(optimizer.min_clients_per_round, 5);
        assert_eq!(optimizer.max_rounds, 100);
    }

    #[test]
    fn test_client_selection_random() {
        let mut optimizer = FedAvgOptimizer::new(0.01, 3, 100);
        optimizer.set_client_selection_strategy(ClientSelectionStrategy::Random { fraction: 0.5 });

        let available_clients = vec!["client1".to_string(), "client2".to_string(), "client3".to_string(), "client4".to_string()];
        let selected = optimizer.select_clients(&available_clients).unwrap();

        assert!(selected.len() >= 3); // Should select at least min_clients_per_round
        assert!(selected.len() <= available_clients.len());
    }

    #[test]
    fn test_simd_accelerator() {
        let mut accelerator = SimdFederatedAccelerator::new();

        let update = ClientUpdate {
            client_id: "test_client".to_string(),
            model_parameters: Array2::zeros((10, 5)),
            gradient_updates: Array2::zeros((10, 5)),
            training_samples: 100,
            training_loss: 0.5,
            compression_ratio: 1.0,
            privacy_noise_variance: 0.0,
            update_timestamp: SystemTime::now(),
        };

        let updates = vec![update];
        let result = accelerator.vectorized_aggregation(&updates);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compression_engine() {
        let engine = CompressionEngine::new();
        let test_data = Array2::from_shape_fn((5, 5), |(i, j)| (i + j) as f64);

        let compressed = engine.compress_parameters(&test_data, 0.5);
        assert!(compressed.is_ok());
    }

    #[test]
    fn test_privacy_engine() {
        let mut engine = PrivacyEngine::new();
        let test_data = Array2::ones((3, 3));

        let noisy_data = engine.add_differential_privacy_noise(&test_data, 0.1);
        assert!(noisy_data.is_ok());

        // Check that noise was added (values should be different)
        let result = noisy_data.unwrap();
        assert_ne!(result[(0, 0)], 1.0);
    }

    #[test]
    fn test_fedprox_optimizer() {
        let optimizer = FedProxOptimizer::new(0.01, 5, 100, 0.1);
        assert_eq!(optimizer.proximal_parameter, 0.1);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = FederatedPerformanceMonitor::new();

        let performance = RoundPerformance {
            round_number: 1,
            participating_clients: 10,
            aggregation_time: Duration::from_millis(100),
            communication_overhead: 0.1,
            convergence_metric: 0.5,
            privacy_cost: 0.05,
        };

        monitor.record_round_performance(performance);

        let summary = monitor.get_federated_metrics_summary();
        assert_eq!(summary.total_rounds, 1);
        assert_eq!(summary.average_participants_per_round, 10.0);
    }
}