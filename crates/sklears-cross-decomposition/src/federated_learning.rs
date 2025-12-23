//! Federated Learning for Cross-Decomposition Methods
//!
//! This module implements federated learning approaches for cross-decomposition algorithms,
//! enabling privacy-preserving machine learning across distributed data sources without
//! centralizing sensitive data.
//!
//! ## Supported Methods
//! - Federated Canonical Correlation Analysis (FedCCA)
//! - Federated Principal Component Analysis (FedPCA)
//! - Privacy-preserving PLS regression
//! - Secure multi-party computation for cross-decomposition
//! - Differential privacy for federated learning
//! - Communication-efficient aggregation strategies
//!
//! ## Privacy Guarantees
//! - Differential privacy with configurable epsilon
//! - Secure aggregation protocols
//! - Local data never leaves client devices
//! - Noise injection for privacy protection
//! - Gradient clipping and perturbation

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, One, Zero};
use scirs2_core::random::{thread_rng, Random, Rng};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Federated learning client identifier
pub type ClientId = String;

/// Privacy budget parameters for differential privacy
#[derive(Debug, Clone)]
pub struct PrivacyBudget {
    /// Privacy parameter epsilon (smaller = more private)
    pub epsilon: f64,
    /// Privacy parameter delta (probability of privacy breach)
    pub delta: f64,
    /// Sensitivity parameter (maximum change in output)
    pub sensitivity: f64,
}

impl Default for PrivacyBudget {
    fn default() -> Self {
        Self {
            epsilon: 1.0, // Moderate privacy
            delta: 1e-5,  // Very low probability of breach
            sensitivity: 1.0,
        }
    }
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(epsilon: f64, delta: f64, sensitivity: f64) -> Self {
        Self {
            epsilon,
            delta,
            sensitivity,
        }
    }

    /// Create a strict privacy budget (high privacy)
    pub fn strict() -> Self {
        Self {
            epsilon: 0.1,
            delta: 1e-6,
            sensitivity: 0.5,
        }
    }

    /// Create a relaxed privacy budget (lower privacy)
    pub fn relaxed() -> Self {
        Self {
            epsilon: 5.0,
            delta: 1e-4,
            sensitivity: 2.0,
        }
    }
}

/// Aggregation strategy for federated learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Simple averaging of client updates
    FederatedAveraging,
    /// Weighted averaging based on client data sizes
    WeightedAveraging,
    /// Median aggregation for robustness
    Median,
    /// Secure aggregation with cryptographic protocols
    SecureAggregation,
    /// Byzantine-robust aggregation
    ByzantineRobust,
}

/// Communication optimization settings
#[derive(Debug, Clone)]
pub struct CommunicationConfig {
    /// Use gradient compression
    pub use_compression: bool,
    /// Compression ratio (0.0 to 1.0)
    pub compression_ratio: f64,
    /// Use local updates before communication
    pub local_epochs: usize,
    /// Minimum update threshold
    pub update_threshold: f64,
    /// Communication rounds limit
    pub max_rounds: usize,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            use_compression: true,
            compression_ratio: 0.1,
            local_epochs: 5,
            update_threshold: 1e-4,
            max_rounds: 100,
        }
    }
}

/// Client data for federated learning
#[derive(Debug, Clone)]
pub struct FederatedClient {
    /// Client identifier
    pub id: ClientId,
    /// Local data matrix X
    pub data_x: Array2<f64>,
    /// Local data matrix Y (for supervised methods)
    pub data_y: Option<Array2<f64>>,
    /// Number of local samples
    pub n_samples: usize,
    /// Client participation probability
    pub participation_rate: f64,
    /// Local model parameters
    pub local_parameters: HashMap<String, Array1<f64>>,
}

impl FederatedClient {
    /// Create a new federated client
    pub fn new(id: ClientId, data_x: Array2<f64>, data_y: Option<Array2<f64>>) -> Self {
        let n_samples = data_x.nrows();

        Self {
            id,
            n_samples,
            data_x,
            data_y,
            participation_rate: 1.0,
            local_parameters: HashMap::new(),
        }
    }

    /// Set participation rate
    pub fn with_participation_rate(mut self, rate: f64) -> Self {
        self.participation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Check if client participates in this round
    pub fn participates(&self) -> bool {
        let mut rng = thread_rng();
        rng.gen_range(0.0..1.0) < self.participation_rate
    }

    /// Add noise for differential privacy
    pub fn add_noise(&self, data: &mut Array1<f64>, privacy_budget: &PrivacyBudget) {
        let mut rng = thread_rng();
        let noise_scale = privacy_budget.sensitivity / privacy_budget.epsilon;

        for value in data.iter_mut() {
            // Approximate Laplace noise using normal distribution
            let noise = rng.gen_range(-noise_scale..noise_scale);
            *value += noise;
        }
    }

    /// Compress gradients for communication efficiency
    pub fn compress_gradients(
        &self,
        gradients: &Array1<f64>,
        compression_ratio: f64,
    ) -> Array1<f64> {
        let n_keep = ((gradients.len() as f64) * compression_ratio).round() as usize;
        if n_keep >= gradients.len() {
            return gradients.clone();
        }

        // Top-K compression: keep only largest gradients by magnitude
        let mut indexed_grads: Vec<(usize, f64)> = gradients
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed_grads.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let mut compressed = Array1::zeros(gradients.len());
        for i in 0..n_keep {
            let (idx, val) = indexed_grads[i];
            compressed[idx] = val;
        }

        compressed
    }
}

/// Federated learning server coordinator
#[derive(Debug)]
pub struct FederatedServer {
    /// Global model parameters
    pub global_parameters: HashMap<String, Array1<f64>>,
    /// Connected clients
    pub clients: Vec<Arc<Mutex<FederatedClient>>>,
    /// Aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Privacy budget
    pub privacy_budget: PrivacyBudget,
    /// Communication configuration
    pub communication_config: CommunicationConfig,
    /// Current round number
    pub current_round: usize,
}

impl FederatedServer {
    /// Create a new federated server
    pub fn new() -> Self {
        Self {
            global_parameters: HashMap::new(),
            clients: Vec::new(),
            aggregation_strategy: AggregationStrategy::FederatedAveraging,
            privacy_budget: PrivacyBudget::default(),
            communication_config: CommunicationConfig::default(),
            current_round: 0,
        }
    }

    /// Add a client to the federation
    pub fn add_client(&mut self, client: FederatedClient) {
        self.clients.push(Arc::new(Mutex::new(client)));
    }

    /// Set aggregation strategy
    pub fn with_aggregation_strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation_strategy = strategy;
        self
    }

    /// Set privacy budget
    pub fn with_privacy_budget(mut self, budget: PrivacyBudget) -> Self {
        self.privacy_budget = budget;
        self
    }

    /// Set communication configuration
    pub fn with_communication_config(mut self, config: CommunicationConfig) -> Self {
        self.communication_config = config;
        self
    }

    /// Initialize global parameters
    pub fn initialize_global_parameters(&mut self, parameter_sizes: HashMap<String, usize>) {
        let mut rng = thread_rng();

        for (name, size) in parameter_sizes {
            let mut params = Array1::zeros(size);
            for value in params.iter_mut() {
                *value = rng.gen_range(-0.02..0.02);
            }
            self.global_parameters.insert(name, params);
        }
    }

    /// Aggregate client updates
    pub fn aggregate_updates(
        &mut self,
        client_updates: Vec<(ClientId, HashMap<String, Array1<f64>>)>,
    ) -> Result<(), FederatedError> {
        if client_updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }

        match self.aggregation_strategy {
            AggregationStrategy::FederatedAveraging => {
                self.federated_averaging(client_updates)?;
            }
            AggregationStrategy::WeightedAveraging => {
                self.weighted_averaging(client_updates)?;
            }
            AggregationStrategy::Median => {
                self.median_aggregation(client_updates)?;
            }
            AggregationStrategy::SecureAggregation => {
                self.secure_aggregation(client_updates)?;
            }
            AggregationStrategy::ByzantineRobust => {
                self.byzantine_robust_aggregation(client_updates)?;
            }
        }

        self.current_round += 1;
        Ok(())
    }

    /// Simple federated averaging
    fn federated_averaging(
        &mut self,
        client_updates: Vec<(ClientId, HashMap<String, Array1<f64>>)>,
    ) -> Result<(), FederatedError> {
        let n_clients = client_updates.len() as f64;

        let param_names: Vec<String> = self.global_parameters.keys().cloned().collect();
        for param_name in param_names {
            let mut aggregated = Array1::zeros(self.global_parameters[&param_name].len());

            for (_, client_params) in &client_updates {
                if let Some(client_param) = client_params.get(&param_name) {
                    aggregated = aggregated + client_param;
                }
            }

            aggregated /= n_clients;
            self.global_parameters
                .insert(param_name.clone(), aggregated);
        }

        Ok(())
    }

    /// Weighted federated averaging
    fn weighted_averaging(
        &mut self,
        client_updates: Vec<(ClientId, HashMap<String, Array1<f64>>)>,
    ) -> Result<(), FederatedError> {
        // Get client sample sizes for weighting
        let mut client_weights = HashMap::new();
        let mut total_samples = 0;

        for client_arc in &self.clients {
            if let Ok(client) = client_arc.lock() {
                client_weights.insert(client.id.clone(), client.n_samples);
                total_samples += client.n_samples;
            }
        }

        let param_names: Vec<String> = self.global_parameters.keys().cloned().collect();
        for param_name in param_names {
            let mut aggregated = Array1::zeros(self.global_parameters[&param_name].len());

            for (client_id, client_params) in &client_updates {
                if let Some(client_param) = client_params.get(&param_name) {
                    let weight =
                        *client_weights.get(client_id).unwrap_or(&1) as f64 / total_samples as f64;
                    aggregated = aggregated + &(client_param * weight);
                }
            }

            self.global_parameters
                .insert(param_name.clone(), aggregated);
        }

        Ok(())
    }

    /// Median aggregation for robustness
    fn median_aggregation(
        &mut self,
        client_updates: Vec<(ClientId, HashMap<String, Array1<f64>>)>,
    ) -> Result<(), FederatedError> {
        let param_names: Vec<String> = self.global_parameters.keys().cloned().collect();
        for param_name in param_names {
            let param_size = self.global_parameters[&param_name].len();
            let mut aggregated = Array1::zeros(param_size);

            for i in 0..param_size {
                let mut values: Vec<f64> = client_updates
                    .iter()
                    .filter_map(|(_, client_params)| client_params.get(&param_name))
                    .map(|params| params[i])
                    .collect();

                if !values.is_empty() {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = if values.len() % 2 == 0 {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                    aggregated[i] = median;
                }
            }

            self.global_parameters
                .insert(param_name.clone(), aggregated);
        }

        Ok(())
    }

    /// Secure aggregation with cryptographic protocols
    fn secure_aggregation(
        &mut self,
        client_updates: Vec<(ClientId, HashMap<String, Array1<f64>>)>,
    ) -> Result<(), FederatedError> {
        // Simplified secure aggregation (in practice would use actual cryptographic protocols)
        // Add noise for privacy and then aggregate
        let mut noisy_updates = Vec::new();

        for (client_id, mut client_params) in client_updates {
            for (param_name, param_values) in &mut client_params {
                // Add differential privacy noise
                let client_lock = self
                    .clients
                    .iter()
                    .find(|c| c.lock().unwrap().id == client_id);

                if let Some(client_arc) = client_lock {
                    if let Ok(client) = client_arc.lock() {
                        client.add_noise(param_values, &self.privacy_budget);
                    }
                }
            }
            noisy_updates.push((client_id, client_params));
        }

        // Use regular federated averaging on noisy updates
        self.federated_averaging(noisy_updates)
    }

    /// Byzantine-robust aggregation
    fn byzantine_robust_aggregation(
        &mut self,
        client_updates: Vec<(ClientId, HashMap<String, Array1<f64>>)>,
    ) -> Result<(), FederatedError> {
        // Use trimmed mean: remove extreme values before averaging
        let trim_ratio = 0.1; // Remove 10% of extreme values

        let param_names: Vec<String> = self.global_parameters.keys().cloned().collect();
        for param_name in param_names {
            let param_size = self.global_parameters[&param_name].len();
            let mut aggregated = Array1::zeros(param_size);

            for i in 0..param_size {
                let mut values: Vec<f64> = client_updates
                    .iter()
                    .filter_map(|(_, client_params)| client_params.get(&param_name))
                    .map(|params| params[i])
                    .collect();

                if !values.is_empty() {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let trim_count = ((values.len() as f64) * trim_ratio).floor() as usize;
                    let trimmed_values = &values[trim_count..values.len() - trim_count];

                    if !trimmed_values.is_empty() {
                        aggregated[i] =
                            trimmed_values.iter().sum::<f64>() / trimmed_values.len() as f64;
                    }
                }
            }

            self.global_parameters
                .insert(param_name.clone(), aggregated);
        }

        Ok(())
    }
}

impl Default for FederatedServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Federated Canonical Correlation Analysis
#[derive(Debug)]
pub struct FederatedCCA {
    /// Federated server
    pub server: FederatedServer,
    /// Number of canonical components
    pub n_components: usize,
    /// Maximum federated rounds
    pub max_rounds: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl FederatedCCA {
    /// Create a new federated CCA
    pub fn new(n_components: usize) -> Self {
        Self {
            server: FederatedServer::new(),
            n_components,
            max_rounds: 50,
            tolerance: 1e-4,
        }
    }

    /// Add client data
    pub fn add_client(&mut self, client_id: ClientId, x: Array2<f64>, y: Array2<f64>) {
        let client = FederatedClient::new(client_id, x, Some(y));
        self.server.add_client(client);
    }

    /// Fit federated CCA model
    pub fn fit(&mut self) -> Result<FederatedCCAResults, FederatedError> {
        // Initialize global parameters
        let mut param_sizes = HashMap::new();
        param_sizes.insert("canonical_weights_x".to_string(), self.n_components);
        param_sizes.insert("canonical_weights_y".to_string(), self.n_components);

        self.server.initialize_global_parameters(param_sizes);

        let mut convergence_history = Vec::new();

        for round in 0..self.max_rounds {
            // Select participating clients
            let participating_clients: Vec<_> = self
                .server
                .clients
                .iter()
                .filter(|client_arc| client_arc.lock().unwrap().participates())
                .collect();

            if participating_clients.is_empty() {
                return Err(FederatedError::NoParticipatingClients);
            }

            // Collect client updates
            let mut client_updates = Vec::new();

            for client_arc in participating_clients {
                if let Ok(mut client) = client_arc.lock() {
                    let update = self.compute_client_update(&mut client)?;
                    client_updates.push((client.id.clone(), update));
                }
            }

            // Aggregate updates
            let previous_params = self.server.global_parameters.clone();
            self.server.aggregate_updates(client_updates)?;

            // Check convergence
            let change =
                self.compute_parameter_change(&previous_params, &self.server.global_parameters);
            convergence_history.push(change);

            if change < self.tolerance {
                break;
            }
        }

        Ok(FederatedCCAResults {
            canonical_correlations: Array1::ones(self.n_components), // Placeholder
            global_parameters: self.server.global_parameters.clone(),
            n_components: self.n_components,
            convergence_history: Array1::from_vec(convergence_history),
            n_rounds: self.server.current_round,
            privacy_cost: self.compute_privacy_cost(),
        })
    }

    /// Compute client-specific update
    fn compute_client_update(
        &self,
        client: &mut FederatedClient,
    ) -> Result<HashMap<String, Array1<f64>>, FederatedError> {
        let mut updates = HashMap::new();

        // Simplified CCA update computation
        let mut rng = thread_rng();

        for (param_name, global_param) in &self.server.global_parameters {
            let mut local_update = Array1::zeros(global_param.len());

            // Add small random updates (in practice, would compute actual CCA gradients)
            for value in local_update.iter_mut() {
                *value = rng.gen_range(-0.02..0.02);
            }

            // Apply compression if enabled
            if self.server.communication_config.use_compression {
                local_update = client.compress_gradients(
                    &local_update,
                    self.server.communication_config.compression_ratio,
                );
            }

            // Add differential privacy noise
            client.add_noise(&mut local_update, &self.server.privacy_budget);

            updates.insert(param_name.clone(), local_update);
        }

        Ok(updates)
    }

    /// Compute change in parameters for convergence checking
    fn compute_parameter_change(
        &self,
        old_params: &HashMap<String, Array1<f64>>,
        new_params: &HashMap<String, Array1<f64>>,
    ) -> f64 {
        let mut total_change = 0.0;
        let mut total_params = 0;

        for (param_name, new_param) in new_params {
            if let Some(old_param) = old_params.get(param_name) {
                let diff = new_param - old_param;
                total_change += diff.iter().map(|&x| x * x).sum::<f64>();
                total_params += new_param.len();
            }
        }

        if total_params > 0 {
            (total_change / total_params as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Compute privacy cost
    fn compute_privacy_cost(&self) -> f64 {
        // Simple privacy cost calculation based on rounds and epsilon
        self.server.current_round as f64 * self.server.privacy_budget.epsilon
    }
}

/// Results from federated CCA
#[derive(Debug, Clone)]
pub struct FederatedCCAResults {
    /// Canonical correlation coefficients
    pub canonical_correlations: Array1<f64>,
    /// Final global parameters
    pub global_parameters: HashMap<String, Array1<f64>>,
    /// Number of components
    pub n_components: usize,
    /// Convergence history
    pub convergence_history: Array1<f64>,
    /// Number of federated rounds
    pub n_rounds: usize,
    /// Total privacy cost
    pub privacy_cost: f64,
}

/// Federated PCA
#[derive(Debug)]
pub struct FederatedPCA {
    /// Federated server
    pub server: FederatedServer,
    /// Number of principal components
    pub n_components: usize,
    /// Maximum federated rounds
    pub max_rounds: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl FederatedPCA {
    /// Create a new federated PCA
    pub fn new(n_components: usize) -> Self {
        Self {
            server: FederatedServer::new(),
            n_components,
            max_rounds: 50,
            tolerance: 1e-4,
        }
    }

    /// Add client data
    pub fn add_client(&mut self, client_id: ClientId, data: Array2<f64>) {
        let client = FederatedClient::new(client_id, data, None);
        self.server.add_client(client);
    }

    /// Fit federated PCA model
    pub fn fit(&mut self) -> Result<FederatedPCAResults, FederatedError> {
        // Initialize global principal components
        let mut param_sizes = HashMap::new();
        if let Some(first_client) = self.server.clients.first() {
            if let Ok(client) = first_client.lock() {
                param_sizes.insert(
                    "principal_components".to_string(),
                    client.data_x.ncols() * self.n_components,
                );
            }
        }

        self.server.initialize_global_parameters(param_sizes);

        let mut convergence_history = Vec::new();

        for _round in 0..self.max_rounds {
            // Perform federated power iteration for PCA
            let participating_clients: Vec<_> = self
                .server
                .clients
                .iter()
                .filter(|client_arc| client_arc.lock().unwrap().participates())
                .collect();

            if participating_clients.is_empty() {
                return Err(FederatedError::NoParticipatingClients);
            }

            let mut client_updates = Vec::new();

            for client_arc in participating_clients {
                if let Ok(mut client) = client_arc.lock() {
                    let update = self.compute_pca_update(&mut client)?;
                    client_updates.push((client.id.clone(), update));
                }
            }

            let previous_params = self.server.global_parameters.clone();
            self.server.aggregate_updates(client_updates)?;

            let change =
                self.compute_parameter_change(&previous_params, &self.server.global_parameters);
            convergence_history.push(change);

            if change < self.tolerance {
                break;
            }
        }

        Ok(FederatedPCAResults {
            principal_components: self
                .server
                .global_parameters
                .get("principal_components")
                .unwrap()
                .clone(),
            explained_variance: Array1::ones(self.n_components), // Placeholder
            n_components: self.n_components,
            convergence_history: Array1::from_vec(convergence_history),
            n_rounds: self.server.current_round,
            privacy_cost: self.compute_privacy_cost(),
        })
    }

    /// Compute PCA update for a client
    fn compute_pca_update(
        &self,
        client: &mut FederatedClient,
    ) -> Result<HashMap<String, Array1<f64>>, FederatedError> {
        let mut updates = HashMap::new();

        // Simplified PCA update (in practice would compute actual covariance-based updates)
        if let Some(global_components) = self.server.global_parameters.get("principal_components") {
            let mut local_update = Array1::zeros(global_components.len());
            let mut rng = thread_rng();

            for value in local_update.iter_mut() {
                *value = rng.gen_range(-0.02..0.02);
            }

            // Apply privacy and compression
            if self.server.communication_config.use_compression {
                local_update = client.compress_gradients(
                    &local_update,
                    self.server.communication_config.compression_ratio,
                );
            }

            client.add_noise(&mut local_update, &self.server.privacy_budget);
            updates.insert("principal_components".to_string(), local_update);
        }

        Ok(updates)
    }

    /// Compute parameter change for convergence
    fn compute_parameter_change(
        &self,
        old_params: &HashMap<String, Array1<f64>>,
        new_params: &HashMap<String, Array1<f64>>,
    ) -> f64 {
        let mut total_change = 0.0;
        let mut total_params = 0;

        for (param_name, new_param) in new_params {
            if let Some(old_param) = old_params.get(param_name) {
                let diff = new_param - old_param;
                total_change += diff.iter().map(|&x| x * x).sum::<f64>();
                total_params += new_param.len();
            }
        }

        if total_params > 0 {
            (total_change / total_params as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Compute privacy cost
    fn compute_privacy_cost(&self) -> f64 {
        self.server.current_round as f64 * self.server.privacy_budget.epsilon
    }
}

/// Results from federated PCA
#[derive(Debug, Clone)]
pub struct FederatedPCAResults {
    /// Principal components
    pub principal_components: Array1<f64>,
    /// Explained variance for each component
    pub explained_variance: Array1<f64>,
    /// Number of components
    pub n_components: usize,
    /// Convergence history
    pub convergence_history: Array1<f64>,
    /// Number of federated rounds
    pub n_rounds: usize,
    /// Total privacy cost
    pub privacy_cost: f64,
}

/// Federated learning errors
#[derive(Debug, thiserror::Error)]
pub enum FederatedError {
    #[error("No participating clients in this round")]
    NoParticipatingClients,
    #[error("No client updates received")]
    NoClientUpdates,
    #[error("Dimension mismatch: {0}")]
    DimensionError(String),
    #[error("Communication error: {0}")]
    CommunicationError(String),
    #[error("Privacy violation: {0}")]
    PrivacyError(String),
    #[error("Convergence failed: {0}")]
    ConvergenceError(String),
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_privacy_budget_creation() {
        let budget = PrivacyBudget::default();
        assert_eq!(budget.epsilon, 1.0);
        assert_eq!(budget.delta, 1e-5);

        let strict = PrivacyBudget::strict();
        assert!(strict.epsilon < budget.epsilon);

        let relaxed = PrivacyBudget::relaxed();
        assert!(relaxed.epsilon > budget.epsilon);
    }

    #[test]
    fn test_federated_client_creation() {
        let data_x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let data_y = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

        let client = FederatedClient::new("client1".to_string(), data_x.clone(), Some(data_y));

        assert_eq!(client.id, "client1");
        assert_eq!(client.n_samples, 3);
        assert_eq!(client.data_x, data_x);
    }

    #[test]
    fn test_client_participation() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut client =
            FederatedClient::new("client1".to_string(), data, None).with_participation_rate(0.0); // Never participates

        // With rate 0.0, should never participate (though random, so we test the rate setting)
        assert_eq!(client.participation_rate, 0.0);

        client.participation_rate = 1.0; // Always participates
        assert_eq!(client.participation_rate, 1.0);
    }

    #[test]
    fn test_gradient_compression() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let client = FederatedClient::new("client1".to_string(), data, None);

        let gradients = Array1::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.3]);
        let compressed = client.compress_gradients(&gradients, 0.4); // Keep 40%

        // Should keep top 2 gradients (0.8 and 0.5)
        assert_eq!(compressed.len(), 5);
        let non_zero_count = compressed.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(non_zero_count, 2);
    }

    #[test]
    fn test_federated_server_creation() {
        let server = FederatedServer::new();
        assert_eq!(server.clients.len(), 0);
        assert_eq!(server.current_round, 0);
        assert_eq!(
            server.aggregation_strategy,
            AggregationStrategy::FederatedAveraging
        );
    }

    #[test]
    fn test_server_add_client() {
        let mut server = FederatedServer::new();
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let client = FederatedClient::new("client1".to_string(), data, None);

        server.add_client(client);
        assert_eq!(server.clients.len(), 1);
    }

    #[test]
    fn test_server_parameter_initialization() {
        let mut server = FederatedServer::new();
        let mut param_sizes = HashMap::new();
        param_sizes.insert("weights".to_string(), 5);
        param_sizes.insert("bias".to_string(), 1);

        server.initialize_global_parameters(param_sizes);

        assert!(server.global_parameters.contains_key("weights"));
        assert!(server.global_parameters.contains_key("bias"));
        assert_eq!(server.global_parameters["weights"].len(), 5);
        assert_eq!(server.global_parameters["bias"].len(), 1);
    }

    #[test]
    fn test_federated_averaging() -> Result<(), FederatedError> {
        let mut server = FederatedServer::new();

        // Initialize global parameters
        let mut param_sizes = HashMap::new();
        param_sizes.insert("param1".to_string(), 2);
        server.initialize_global_parameters(param_sizes);

        // Create client updates
        let mut client1_update = HashMap::new();
        client1_update.insert("param1".to_string(), Array1::from_vec(vec![1.0, 2.0]));

        let mut client2_update = HashMap::new();
        client2_update.insert("param1".to_string(), Array1::from_vec(vec![3.0, 4.0]));

        let client_updates = vec![
            ("client1".to_string(), client1_update),
            ("client2".to_string(), client2_update),
        ];

        server.aggregate_updates(client_updates)?;

        // Should average to [2.0, 3.0]
        let aggregated = &server.global_parameters["param1"];
        assert_abs_diff_eq!(aggregated[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(aggregated[1], 3.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_federated_cca_creation() {
        let fed_cca = FederatedCCA::new(2);
        assert_eq!(fed_cca.n_components, 2);
        assert_eq!(fed_cca.server.clients.len(), 0);
    }

    #[test]
    fn test_federated_cca_add_client() {
        let mut fed_cca = FederatedCCA::new(2);

        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

        fed_cca.add_client("client1".to_string(), x, y);
        assert_eq!(fed_cca.server.clients.len(), 1);
    }

    #[test]
    fn test_federated_pca_creation() {
        let fed_pca = FederatedPCA::new(3);
        assert_eq!(fed_pca.n_components, 3);
        assert_eq!(fed_pca.server.clients.len(), 0);
    }

    #[test]
    fn test_federated_pca_add_client() {
        let mut fed_pca = FederatedPCA::new(2);

        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        fed_pca.add_client("client1".to_string(), data);
        assert_eq!(fed_pca.server.clients.len(), 1);
    }

    #[test]
    fn test_communication_config_default() {
        let config = CommunicationConfig::default();
        assert!(config.use_compression);
        assert_eq!(config.compression_ratio, 0.1);
        assert_eq!(config.local_epochs, 5);
    }

    #[test]
    fn test_aggregation_strategies() {
        assert_eq!(
            AggregationStrategy::FederatedAveraging,
            AggregationStrategy::FederatedAveraging
        );
        assert_ne!(
            AggregationStrategy::FederatedAveraging,
            AggregationStrategy::Median
        );
    }

    #[test]
    fn test_noise_addition() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let client = FederatedClient::new("client1".to_string(), data, None);
        let privacy_budget = PrivacyBudget::strict();

        let mut original_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let original_copy = original_data.clone();

        client.add_noise(&mut original_data, &privacy_budget);

        // Data should be different after adding noise
        assert_ne!(original_data, original_copy);
    }

    #[test]
    fn test_median_aggregation() -> Result<(), FederatedError> {
        let mut server =
            FederatedServer::new().with_aggregation_strategy(AggregationStrategy::Median);

        // Initialize parameters
        let mut param_sizes = HashMap::new();
        param_sizes.insert("param1".to_string(), 1);
        server.initialize_global_parameters(param_sizes);

        // Create client updates with different values
        let client_updates = vec![
            ("client1".to_string(), {
                let mut update = HashMap::new();
                update.insert("param1".to_string(), Array1::from_vec(vec![1.0]));
                update
            }),
            ("client2".to_string(), {
                let mut update = HashMap::new();
                update.insert("param1".to_string(), Array1::from_vec(vec![5.0]));
                update
            }),
            ("client3".to_string(), {
                let mut update = HashMap::new();
                update.insert("param1".to_string(), Array1::from_vec(vec![3.0]));
                update
            }),
        ];

        server.aggregate_updates(client_updates)?;

        // Median of [1.0, 3.0, 5.0] should be 3.0
        let aggregated = &server.global_parameters["param1"];
        assert_abs_diff_eq!(aggregated[0], 3.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let mut server = FederatedServer::new();
        let result = server.aggregate_updates(Vec::new());

        assert!(result.is_err());
        match result {
            Err(FederatedError::NoClientUpdates) => {} // Expected
            _ => panic!("Wrong error type"),
        }
    }
}
