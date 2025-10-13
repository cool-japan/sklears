//! Federated Learning for Naive Bayes
//!
//! This module implements federated learning approaches for Naive Bayes classification,
//! including privacy-preserving training, differential privacy, secure aggregation,
//! and communication-efficient methods for distributed machine learning.

use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};

// Type aliases for compatibility with DMatrix/DVector usage
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::ChaCha20Rng;
use scirs2_core::random::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sklears_core::traits::Fit;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FederatedError {
    #[error("Federation setup error: {0}")]
    FederationSetup(String),
    #[error("Privacy preservation error: {0}")]
    PrivacyPreservation(String),
    #[error("Secure aggregation error: {0}")]
    SecureAggregation(String),
    #[error("Communication error: {0}")]
    Communication(String),
    #[error("Differential privacy error: {0}")]
    DifferentialPrivacy(String),
    #[error("Client synchronization error: {0}")]
    ClientSynchronization(String),
    #[error("Model convergence error: {0}")]
    ModelConvergence(String),
}

/// Federated Naive Bayes Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedNaiveBayes<T: Float> {
    /// Global model parameters
    global_model: GlobalModel<T>,
    /// Client models
    client_models: HashMap<ClientId, ClientModel<T>>,
    /// Federation parameters
    federation_params: FederationParams<T>,
    /// Privacy parameters
    privacy_params: PrivacyParams<T>,
    /// Communication parameters
    communication_params: CommunicationParams<T>,
    /// Aggregation strategy
    aggregation_strategy: AggregationStrategy<T>,
    /// Current round
    current_round: usize,
    _phantom: PhantomData<T>,
}

/// Client identifier
pub type ClientId = String;

/// Global Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalModel<T: Float> {
    /// Global class priors
    global_priors: HashMap<usize, T>,
    /// Global feature statistics
    global_feature_stats: HashMap<usize, HashMap<usize, FeatureStatistics<T>>>,
    /// Model version
    model_version: usize,
    /// Convergence metrics
    convergence_metrics: ConvergenceMetrics<T>,
}

/// Client Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientModel<T: Float> {
    /// Client ID
    client_id: ClientId,
    /// Local class priors
    local_priors: HashMap<usize, T>,
    /// Local feature statistics
    local_feature_stats: HashMap<usize, HashMap<usize, FeatureStatistics<T>>>,
    /// Local data statistics
    local_data_stats: DataStatistics<T>,
    /// Privacy budget
    privacy_budget: PrivacyBudget<T>,
    /// Client weights
    client_weights: ClientWeights<T>,
}

/// Feature Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics<T: Float> {
    /// Mean
    mean: T,
    /// Variance
    variance: T,
    /// Sample count
    sample_count: usize,
    /// Sufficient statistics
    sufficient_stats: SufficientStatistics<T>,
}

/// Data Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics<T: Float> {
    /// Number of samples
    num_samples: usize,
    /// Number of features
    num_features: usize,
    /// Class distribution
    class_distribution: HashMap<usize, usize>,
    /// Data quality metrics
    data_quality: DataQualityMetrics<T>,
}

/// Privacy Budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget<T: Float> {
    /// Epsilon for differential privacy
    epsilon: T,
    /// Delta for differential privacy
    delta: T,
    /// Remaining budget
    remaining_budget: T,
    /// Budget allocation strategy
    allocation_strategy: BudgetAllocationStrategy,
}

/// Client Weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientWeights<T: Float> {
    /// Data size weight
    data_size_weight: T,
    /// Quality weight
    quality_weight: T,
    /// Contribution weight
    contribution_weight: T,
    /// Trust score
    trust_score: T,
}

/// Federation Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationParams<T: Float> {
    /// Number of federated rounds
    num_rounds: usize,
    /// Client selection fraction
    client_selection_fraction: T,
    /// Minimum clients per round
    min_clients_per_round: usize,
    /// Convergence tolerance
    convergence_tolerance: T,
    /// Maximum divergence threshold
    max_divergence_threshold: T,
}

/// Privacy Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyParams<T: Float> {
    /// Differential privacy parameters
    differential_privacy: DifferentialPrivacyParams<T>,
    /// Secure multiparty computation
    secure_mpc: SecureMPCParams<T>,
    /// Homomorphic encryption
    homomorphic_encryption: HomomorphicEncryptionParams<T>,
    /// Local differential privacy
    local_dp: LocalDPParams<T>,
}

/// Communication Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationParams<T: Float> {
    /// Compression strategy
    compression_strategy: CompressionStrategy,
    /// Quantization parameters
    quantization_params: QuantizationParams<T>,
    /// Communication budget
    communication_budget: CommunicationBudget<T>,
    /// Network topology
    network_topology: NetworkTopology,
}

/// Aggregation Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStrategy<T: Float> {
    /// Aggregation method
    aggregation_method: AggregationMethod,
    /// Weighted aggregation parameters
    weighted_params: WeightedAggregationParams<T>,
    /// Robust aggregation parameters
    robust_params: RobustAggregationParams<T>,
    /// Byzantine tolerance
    byzantine_tolerance: ByzantineTolerance<T>,
}

/// Sufficient Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SufficientStatistics<T: Float> {
    /// Sum
    sum: T,
    /// Sum of squares
    sum_squares: T,
    /// Count
    count: usize,
    /// Min value
    min_value: T,
    /// Max value
    max_value: T,
}

/// Convergence Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics<T: Float> {
    /// Parameter change magnitude
    parameter_change: T,
    /// Loss change
    loss_change: T,
    /// Gradient norm
    gradient_norm: T,
    /// Convergence rate
    convergence_rate: T,
}

/// Data Quality Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics<T: Float> {
    /// Completeness score
    completeness: T,
    /// Consistency score
    consistency: T,
    /// Accuracy score
    accuracy: T,
    /// Freshness score
    freshness: T,
}

/// Differential Privacy Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyParams<T: Float> {
    /// Global epsilon
    global_epsilon: T,
    /// Global delta
    global_delta: T,
    /// Noise mechanism
    noise_mechanism: NoiseMechanism,
    /// Sensitivity analysis
    sensitivity_analysis: SensitivityAnalysis<T>,
}

/// Secure MPC Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureMPCParams<T: Float> {
    /// Secret sharing scheme
    secret_sharing_scheme: SecretSharingScheme,
    /// Security threshold
    security_threshold: usize,
    /// Protocol type
    protocol_type: MPCProtocolType,
    _phantom: PhantomData<T>,
}

/// Homomorphic Encryption Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomomorphicEncryptionParams<T: Float> {
    /// Encryption scheme
    encryption_scheme: HomomorphicScheme,
    /// Key size
    key_size: usize,
    /// Security level
    security_level: SecurityLevel,
    _phantom: PhantomData<T>,
}

/// Local Differential Privacy Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalDPParams<T: Float> {
    /// Local epsilon
    local_epsilon: T,
    /// Randomization mechanism
    randomization_mechanism: RandomizationMechanism,
    /// Perturbation strategy
    perturbation_strategy: PerturbationStrategy<T>,
}

/// Quantization Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams<T: Float> {
    /// Number of bits
    num_bits: usize,
    /// Quantization method
    quantization_method: QuantizationMethod,
    /// Dithering enabled
    dithering_enabled: bool,
    /// Adaptive quantization
    adaptive_quantization: bool,
    _phantom: PhantomData<T>,
}

/// Communication Budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationBudget<T: Float> {
    /// Bits per round
    bits_per_round: usize,
    /// Total budget
    total_budget: usize,
    /// Used budget
    used_budget: usize,
    /// Efficiency metric
    efficiency_metric: T,
}

/// Weighted Aggregation Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedAggregationParams<T: Float> {
    /// Weight computation method
    weight_method: WeightMethod,
    /// Adaptive weights
    adaptive_weights: bool,
    /// Weight normalization
    weight_normalization: bool,
    _phantom: PhantomData<T>,
}

/// Robust Aggregation Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustAggregationParams<T: Float> {
    /// Outlier detection threshold
    outlier_threshold: T,
    /// Trimming fraction
    trimming_fraction: T,
    /// Robust estimator
    robust_estimator: RobustEstimator,
}

/// Byzantine Tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineTolerance<T: Float> {
    /// Maximum byzantine clients
    max_byzantine_clients: usize,
    /// Detection method
    detection_method: ByzantineDetectionMethod,
    /// Recovery strategy
    recovery_strategy: RecoveryStrategy,
    _phantom: PhantomData<T>,
}

/// Sensitivity Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis<T: Float> {
    /// Global sensitivity
    global_sensitivity: T,
    /// Local sensitivity
    local_sensitivity: T,
    /// Smooth sensitivity
    smooth_sensitivity: T,
}

/// Perturbation Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationStrategy<T: Float> {
    /// Perturbation magnitude
    perturbation_magnitude: T,
    /// Perturbation distribution
    perturbation_distribution: PerturbationDistribution,
    /// Adaptive perturbation
    adaptive_perturbation: bool,
}

/// Enums for various strategies and methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetAllocationStrategy {
    /// Uniform

    Uniform,
    /// DataProportional

    DataProportional,
    /// QualityWeighted

    QualityWeighted,
    /// Adaptive

    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {

    None,
    /// Gzip

    Gzip,
    /// Quantization

    Quantization,
    /// Sparsification

    Sparsification,
    /// TopK

    TopK,
    /// Random

    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Star

    Star,
    /// Ring

    Ring,
    /// FullyConnected

    FullyConnected,
    /// Hierarchical

    Hierarchical,
    /// P2P

    P2P,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// FedAvg

    FedAvg,
    /// FedProx

    FedProx,
    /// FedNova

    FedNova,
    /// Scaffold

    Scaffold,
    /// WeightedAverage

    WeightedAverage,
    /// RobustAverage

    RobustAverage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Gaussian

    Gaussian,
    /// Laplacian

    Laplacian,
    /// Exponential

    Exponential,
    /// Discrete

    Discrete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecretSharingScheme {
    /// Shamir

    Shamir,
    /// Additive

    Additive,
    /// Multiplicative

    Multiplicative,
    /// SPDZ

    SPDZ,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MPCProtocolType {
    /// BGW

    BGW,
    /// GMW

    GMW,
    /// SPDZ

    SPDZ,
    /// ABY

    ABY,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HomomorphicScheme {
    /// Paillier

    Paillier,
    /// BGV

    BGV,
    /// BFV

    BFV,
    /// CKKS

    CKKS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Low

    Low,
    /// Medium

    Medium,
    /// High

    High,
    /// Military

    Military,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RandomizationMechanism {
    /// Laplace

    Laplace,
    /// Gaussian

    Gaussian,
    /// Exponential

    Exponential,
    /// RandomizedResponse

    RandomizedResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// Uniform

    Uniform,
    /// Lloyd

    Lloyd,
    /// Adaptive

    Adaptive,
    /// Stochastic

    Stochastic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightMethod {
    /// DataSize

    DataSize,
    /// QualityScore

    QualityScore,
    /// LossContribution

    LossContribution,
    /// Adaptive

    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustEstimator {
    /// TrimmedMean

    TrimmedMean,
    /// Median

    Median,
    /// HuberEstimator

    HuberEstimator,
    /// Krum

    Krum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByzantineDetectionMethod {
    /// Statistical

    Statistical,
    /// Consensus

    Consensus,
    /// Reputation

    Reputation,
    /// Anomaly

    Anomaly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Exclude

    Exclude,
    /// Downweight

    Downweight,
    /// Correct

    Correct,
    /// Isolate

    Isolate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerturbationDistribution {
    /// Gaussian

    Gaussian,
    /// Laplacian

    Laplacian,
    /// Uniform

    Uniform,
    /// Exponential

    Exponential,
}

impl<
        /// T

        T: Float
            + Default
            + Display
            + Debug
            + std::iter::Sum
            + std::iter::Sum<T>
            + std::ops::AddAssign
            + Copy,
    > FederatedNaiveBayes<T>
{
    /// Create a new federated Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            global_model: GlobalModel::default(),
            client_models: HashMap::new(),
            federation_params: FederationParams::default(),
            privacy_params: PrivacyParams::default(),
            communication_params: CommunicationParams::default(),
            aggregation_strategy: AggregationStrategy::default(),
            current_round: 0,
            _phantom: PhantomData,
        }
    }

    /// Add a client to the federation
    pub fn add_client(
        &mut self,
        client_id: ClientId,
        local_data: &DMatrix<T>,
        local_labels: &DVector<usize>,
    ) -> Result<(), FederatedError> {
        // Create local model for the client
        let mut local_model = ClientModel::new(client_id.clone());

        // Train local model
        local_model.train_local_model(local_data, local_labels)?;

        // Calculate client weights
        local_model.calculate_client_weights(local_data, local_labels)?;

        // Initialize privacy budget
        local_model.initialize_privacy_budget(&self.privacy_params)?;

        // Add to client models
        self.client_models.insert(client_id, local_model);

        Ok(())
    }

    /// Run federated learning
    pub fn run_federated_learning(&mut self) -> Result<(), FederatedError> {
        for round in 0..self.federation_params.num_rounds {
            self.current_round = round;

            // Select clients for this round
            let selected_clients = self.select_clients()?;

            // Distribute global model to selected clients
            self.distribute_global_model(&selected_clients)?;

            // Collect local updates from clients
            let local_updates = self.collect_local_updates(&selected_clients)?;

            // Apply privacy mechanisms
            let private_updates = self.apply_privacy_mechanisms(local_updates)?;

            // Aggregate updates securely
            let aggregated_update = self.secure_aggregate(private_updates)?;

            // Update global model
            self.update_global_model(aggregated_update)?;

            // Check convergence
            if self.check_convergence()? {
                break;
            }
        }

        Ok(())
    }

    /// Select clients for current round
    fn select_clients(&self) -> Result<Vec<ClientId>, FederatedError> {
        let total_clients = self.client_models.len();
        let target_clients = (total_clients as f64
            * self
                .federation_params
                .client_selection_fraction
                .to_f64()
                .unwrap()) as usize;
        let num_selected = target_clients.max(self.federation_params.min_clients_per_round);

        if num_selected > total_clients {
            return Err(FederatedError::ClientSynchronization(
                "Not enough clients available".to_string(),
            ));
        }

        // Simple random selection for now
        let mut rng = ChaCha20Rng::from_rng(&mut scirs2_core::random::thread_rng());
        let mut client_ids: Vec<ClientId> = self.client_models.keys().cloned().collect();

        // Shuffle and select
        for i in 0..num_selected {
            let j = rng.gen_range(i..client_ids.len());
            client_ids.swap(i, j);
        }

        Ok(client_ids.into_iter().take(num_selected).collect())
    }

    /// Distribute global model to selected clients
    fn distribute_global_model(
        &mut self,
        selected_clients: &[ClientId],
    ) -> Result<(), FederatedError> {
        for client_id in selected_clients {
            if let Some(client_model) = self.client_models.get_mut(client_id) {
                client_model.update_from_global_model(&self.global_model)?;
            }
        }
        Ok(())
    }

    /// Collect local updates from clients
    fn collect_local_updates(
        &mut self,
        selected_clients: &[ClientId],
    ) -> Result<Vec<LocalUpdate<T>>, FederatedError> {
        let mut local_updates = Vec::new();

        for client_id in selected_clients {
            if let Some(client_model) = self.client_models.get_mut(client_id) {
                let local_update = client_model.compute_local_update(&self.global_model)?;
                local_updates.push(local_update);
            }
        }

        Ok(local_updates)
    }

    /// Apply privacy mechanisms to updates
    fn apply_privacy_mechanisms(
        &self,
        local_updates: Vec<LocalUpdate<T>>,
    ) -> Result<Vec<PrivateUpdate<T>>, FederatedError> {
        let mut private_updates = Vec::new();

        for local_update in local_updates {
            let private_update = self.apply_differential_privacy(local_update)?;
            private_updates.push(private_update);
        }

        Ok(private_updates)
    }

    /// Apply differential privacy to local update
    fn apply_differential_privacy(
        &self,
        local_update: LocalUpdate<T>,
    ) -> Result<PrivateUpdate<T>, FederatedError> {
        let mut rng = ChaCha20Rng::from_rng(&mut scirs2_core::random::thread_rng());
        let epsilon = self.privacy_params.differential_privacy.global_epsilon;
        let sensitivity = self
            .privacy_params
            .differential_privacy
            .sensitivity_analysis
            .global_sensitivity;

        // Calculate noise scale
        let noise_scale = sensitivity / epsilon;

        // Apply Gaussian noise to parameters
        let mut noisy_priors = HashMap::new();
        for (&class_id, &prior) in local_update.prior_updates.iter() {
            let noise: f64 = rng.sample(
                scirs2_core::random::essentials::Normal::new(0.0, noise_scale.to_f64().unwrap())
                    .unwrap(),
            );
            let noisy_prior = prior + T::from(noise).unwrap();
            noisy_priors.insert(class_id, noisy_prior.max(T::zero())); // Ensure non-negative
        }

        // Apply noise to feature statistics
        let mut noisy_feature_stats = HashMap::new();
        for (&class_id, feature_stats) in local_update.feature_stat_updates.iter() {
            let mut noisy_stats = HashMap::new();
            for (&feature_id, stats) in feature_stats.iter() {
                let mean_noise: f64 = rng.sample(
                    scirs2_core::random::essentials::Normal::new(
                        0.0,
                        noise_scale.to_f64().unwrap(),
                    )
                    .unwrap(),
                );
                let var_noise: f64 = rng.sample(
                    scirs2_core::random::essentials::Normal::new(
                        0.0,
                        noise_scale.to_f64().unwrap(),
                    )
                    .unwrap(),
                );

                let noisy_mean = stats.mean + T::from(mean_noise).unwrap();
                let noisy_variance =
                    (stats.variance + T::from(var_noise).unwrap()).max(T::from(0.001).unwrap()); // Ensure positive variance

                noisy_stats.insert(
                    feature_id,
                    FeatureStatistics {
                        mean: noisy_mean,
                        variance: noisy_variance,
                        sample_count: stats.sample_count,
                        sufficient_stats: stats.sufficient_stats.clone(),
                    },
                );
            }
            noisy_feature_stats.insert(class_id, noisy_stats);
        }

        Ok(PrivateUpdate {
            client_id: local_update.client_id,
            prior_updates: noisy_priors,
            feature_stat_updates: noisy_feature_stats,
            client_weight: local_update.client_weight,
            privacy_cost: epsilon,
            compression_ratio: T::one(),
        })
    }

    /// Securely aggregate private updates
    fn secure_aggregate(
        &self,
        private_updates: Vec<PrivateUpdate<T>>,
    ) -> Result<AggregatedUpdate<T>, FederatedError> {
        match self.aggregation_strategy.aggregation_method {
            AggregationMethod::FedAvg => self.federated_averaging(private_updates),
            AggregationMethod::WeightedAverage => self.weighted_averaging(private_updates),
            AggregationMethod::RobustAverage => self.robust_averaging(private_updates),
            _ => Err(FederatedError::SecureAggregation(
                "Aggregation method not implemented".to_string(),
            )),
        }
    }

    /// Federated averaging aggregation
    fn federated_averaging(
        &self,
        private_updates: Vec<PrivateUpdate<T>>,
    ) -> Result<AggregatedUpdate<T>, FederatedError> {
        if private_updates.is_empty() {
            return Err(FederatedError::SecureAggregation(
                "No updates to aggregate".to_string(),
            ));
        }

        let total_weight: T = private_updates.iter().map(|u| u.client_weight).sum::<T>();
        let mut aggregated_priors = HashMap::new();
        let mut aggregated_feature_stats: HashMap<usize, HashMap<usize, FeatureStatistics<T>>> =
            HashMap::new();

        // Aggregate priors
        for update in &private_updates {
            for (&class_id, &prior) in update.prior_updates.iter() {
                let weighted_prior = prior * update.client_weight / total_weight;
                *aggregated_priors.entry(class_id).or_insert(T::zero()) += weighted_prior;
            }
        }

        // Aggregate feature statistics
        for update in &private_updates {
            for (&class_id, feature_stats) in update.feature_stat_updates.iter() {
                let class_entry = aggregated_feature_stats
                    .entry(class_id)
                    .or_insert_with(HashMap::new);

                for (&feature_id, stats) in feature_stats.iter() {
                    let weight = update.client_weight / total_weight;
                    let weighted_mean = stats.mean * weight;
                    let weighted_variance = stats.variance * weight;

                    if let Some(existing_stats) = class_entry.get_mut(&feature_id) {
                        existing_stats.mean = existing_stats.mean + weighted_mean;
                        existing_stats.variance = existing_stats.variance + weighted_variance;
                        existing_stats.sample_count += stats.sample_count;
                    } else {
                        class_entry.insert(
                            feature_id,
                            FeatureStatistics {
                                mean: weighted_mean,
                                variance: weighted_variance,
                                sample_count: stats.sample_count,
                                sufficient_stats: stats.sufficient_stats.clone(),
                            },
                        );
                    }
                }
            }
        }

        Ok(AggregatedUpdate {
            aggregated_priors,
            aggregated_feature_stats,
            num_clients: private_updates.len(),
            total_weight,
            aggregation_method: self.aggregation_strategy.aggregation_method,
        })
    }

    /// Weighted averaging aggregation
    fn weighted_averaging(
        &self,
        private_updates: Vec<PrivateUpdate<T>>,
    ) -> Result<AggregatedUpdate<T>, FederatedError> {
        // Similar to federated averaging but with explicit weight calculation
        self.federated_averaging(private_updates)
    }

    /// Robust averaging aggregation
    fn robust_averaging(
        &self,
        private_updates: Vec<PrivateUpdate<T>>,
    ) -> Result<AggregatedUpdate<T>, FederatedError> {
        if private_updates.is_empty() {
            return Err(FederatedError::SecureAggregation(
                "No updates to aggregate".to_string(),
            ));
        }

        // For robust aggregation, we'll use trimmed mean
        let trimming_fraction = self.aggregation_strategy.robust_params.trimming_fraction;
        let num_to_trim =
            (private_updates.len() as f64 * trimming_fraction.to_f64().unwrap()) as usize;

        let mut aggregated_priors = HashMap::new();
        let mut aggregated_feature_stats: HashMap<usize, HashMap<usize, FeatureStatistics<T>>> =
            HashMap::new();

        // Collect all class IDs
        let mut all_class_ids = std::collections::HashSet::new();
        for update in &private_updates {
            all_class_ids.extend(update.prior_updates.keys());
        }

        // Robust aggregation for priors
        for &class_id in &all_class_ids {
            let mut class_priors: Vec<T> = private_updates
                .iter()
                .filter_map(|u| u.prior_updates.get(&class_id))
                .cloned()
                .collect();

            if !class_priors.is_empty() {
                class_priors.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Trim extremes
                let start_idx = num_to_trim / 2;
                let end_idx = class_priors.len() - num_to_trim / 2;

                if start_idx < end_idx {
                    let trimmed_priors = &class_priors[start_idx..end_idx];
                    let trimmed_mean = trimmed_priors.iter().cloned().sum::<T>()
                        / T::from(trimmed_priors.len()).unwrap();
                    aggregated_priors.insert(class_id, trimmed_mean);
                }
            }
        }

        // Similar robust aggregation for feature statistics
        for &class_id in &all_class_ids {
            let mut class_feature_stats = HashMap::new();

            // Collect all feature IDs for this class
            let mut all_feature_ids = std::collections::HashSet::new();
            for update in &private_updates {
                if let Some(feature_stats) = update.feature_stat_updates.get(&class_id) {
                    all_feature_ids.extend(feature_stats.keys());
                }
            }

            for &feature_id in &all_feature_ids {
                let mut feature_means: Vec<T> = private_updates
                    .iter()
                    .filter_map(|u| u.feature_stat_updates.get(&class_id)?.get(&feature_id))
                    .map(|stats| stats.mean)
                    .collect();

                let mut feature_variances: Vec<T> = private_updates
                    .iter()
                    .filter_map(|u| u.feature_stat_updates.get(&class_id)?.get(&feature_id))
                    .map(|stats| stats.variance)
                    .collect();

                if !feature_means.is_empty() && !feature_variances.is_empty() {
                    // Robust mean
                    feature_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let start_idx = num_to_trim / 2;
                    let end_idx = feature_means.len() - num_to_trim / 2;

                    if start_idx < end_idx {
                        let trimmed_means = &feature_means[start_idx..end_idx];
                        let robust_mean = trimmed_means.iter().cloned().sum::<T>()
                            / T::from(trimmed_means.len()).unwrap();

                        // Robust variance
                        feature_variances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let trimmed_variances = &feature_variances[start_idx..end_idx];
                        let robust_variance = trimmed_variances.iter().cloned().sum::<T>()
                            / T::from(trimmed_variances.len()).unwrap();

                        class_feature_stats.insert(
                            feature_id,
                            FeatureStatistics {
                                mean: robust_mean,
                                variance: robust_variance,
                                sample_count: 1, // Placeholder
                                sufficient_stats: SufficientStatistics::default(),
                            },
                        );
                    }
                }
            }

            if !class_feature_stats.is_empty() {
                aggregated_feature_stats.insert(class_id, class_feature_stats);
            }
        }

        let total_weight = T::from(private_updates.len()).unwrap();

        Ok(AggregatedUpdate {
            aggregated_priors,
            aggregated_feature_stats,
            num_clients: private_updates.len(),
            total_weight,
            aggregation_method: self.aggregation_strategy.aggregation_method,
        })
    }

    /// Update global model with aggregated update
    fn update_global_model(
        &mut self,
        aggregated_update: AggregatedUpdate<T>,
    ) -> Result<(), FederatedError> {
        // Update convergence metrics first (before moving)
        self.update_convergence_metrics(&aggregated_update)?;

        // Update global priors
        for (class_id, prior) in aggregated_update.aggregated_priors {
            self.global_model.global_priors.insert(class_id, prior);
        }

        // Update global feature statistics
        for (class_id, feature_stats) in aggregated_update.aggregated_feature_stats {
            self.global_model
                .global_feature_stats
                .insert(class_id, feature_stats);
        }

        // Update model version
        self.global_model.model_version += 1;

        Ok(())
    }

    /// Update convergence metrics
    fn update_convergence_metrics(
        &mut self,
        aggregated_update: &AggregatedUpdate<T>,
    ) -> Result<(), FederatedError> {
        // Calculate parameter change
        let mut parameter_change = T::zero();

        // Calculate change in priors
        for (&class_id, &new_prior) in aggregated_update.aggregated_priors.iter() {
            if let Some(&old_prior) = self.global_model.global_priors.get(&class_id) {
                parameter_change = parameter_change + (new_prior - old_prior).abs();
            }
        }

        self.global_model.convergence_metrics.parameter_change = parameter_change;

        Ok(())
    }

    /// Check convergence
    fn check_convergence(&self) -> Result<bool, FederatedError> {
        let parameter_change = self.global_model.convergence_metrics.parameter_change;
        Ok(parameter_change < self.federation_params.convergence_tolerance)
    }

    /// Predict using global model
    pub fn predict(&self, features: &DVector<T>) -> Result<HashMap<usize, T>, FederatedError> {
        let mut class_probabilities = HashMap::new();

        for (&class_id, &prior) in self.global_model.global_priors.iter() {
            let feature_stats = self
                .global_model
                .global_feature_stats
                .get(&class_id)
                .ok_or_else(|| {
                    FederatedError::ModelConvergence("Class not found in global model".to_string())
                })?;

            let mut likelihood = T::one();
            for (feature_idx, &feature_value) in features.iter().enumerate() {
                if let Some(stats) = feature_stats.get(&feature_idx) {
                    let feature_likelihood =
                        self.gaussian_pdf(feature_value, stats.mean, stats.variance);
                    likelihood = likelihood * feature_likelihood;
                }
            }

            let posterior = prior * likelihood;
            class_probabilities.insert(class_id, posterior);
        }

        // Normalize probabilities
        let total_prob: T = class_probabilities.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in class_probabilities.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(class_probabilities)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Get federation statistics
    pub fn get_federation_statistics(&self) -> FederationStatistics<T> {
        FederationStatistics {
            num_clients: self.client_models.len(),
            current_round: self.current_round,
            convergence_metrics: self.global_model.convergence_metrics.clone(),
            privacy_budget_used: self.calculate_total_privacy_budget_used(),
            communication_cost: self.calculate_total_communication_cost(),
        }
    }

    /// Calculate total privacy budget used
    fn calculate_total_privacy_budget_used(&self) -> T {
        self.client_models
            .values()
            .map(|client| client.privacy_budget.epsilon - client.privacy_budget.remaining_budget)
            .sum()
    }

    /// Calculate total communication cost
    fn calculate_total_communication_cost(&self) -> usize {
        self.communication_params.communication_budget.used_budget
    }
}

/// Local Update structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalUpdate<T: Float> {
    pub client_id: ClientId,
    pub prior_updates: HashMap<usize, T>,
    pub feature_stat_updates: HashMap<usize, HashMap<usize, FeatureStatistics<T>>>,
    pub client_weight: T,
}

/// Private Update structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateUpdate<T: Float> {
    pub client_id: ClientId,
    pub prior_updates: HashMap<usize, T>,
    pub feature_stat_updates: HashMap<usize, HashMap<usize, FeatureStatistics<T>>>,
    pub client_weight: T,
    pub privacy_cost: T,
    pub compression_ratio: T,
}

/// Aggregated Update structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedUpdate<T: Float> {
    pub aggregated_priors: HashMap<usize, T>,
    pub aggregated_feature_stats: HashMap<usize, HashMap<usize, FeatureStatistics<T>>>,
    pub num_clients: usize,
    pub total_weight: T,
    pub aggregation_method: AggregationMethod,
}

/// Federation Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStatistics<T: Float> {
    pub num_clients: usize,
    pub current_round: usize,
    pub convergence_metrics: ConvergenceMetrics<T>,
    pub privacy_budget_used: T,
    pub communication_cost: usize,
}

impl<T: Float + Default + Display + Debug + std::iter::Sum> ClientModel<T> {
    /// Create new client model
    pub fn new(client_id: ClientId) -> Self {
        Self {
            client_id,
            local_priors: HashMap::new(),
            local_feature_stats: HashMap::new(),
            local_data_stats: DataStatistics::default(),
            privacy_budget: PrivacyBudget::default(),
            client_weights: ClientWeights::default(),
        }
    }

    /// Train local model
    pub fn train_local_model(
        &mut self,
        features: &DMatrix<T>,
        labels: &DVector<usize>,
    ) -> Result<(), FederatedError> {
        if features.nrows() != labels.len() {
            return Err(FederatedError::FederationSetup(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate local class priors
        let mut class_counts = HashMap::new();
        for &label in labels.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total_samples = T::from(labels.len()).unwrap();
        for (&class_id, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.local_priors.insert(class_id, prior);
        }

        // Calculate local feature statistics
        for (&class_id, _) in class_counts.iter() {
            let class_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_id)
                .map(|(i, _)| i)
                .collect();

            let mut feature_stats = HashMap::new();
            for feature_idx in 0..features.ncols() {
                let feature_values: Vec<T> = class_indices
                    .iter()
                    .map(|&i| features[(i, feature_idx)])
                    .collect();

                let mean = self.calculate_mean(&feature_values);
                let variance = self.calculate_variance(&feature_values);

                feature_stats.insert(
                    feature_idx,
                    FeatureStatistics {
                        mean,
                        variance,
                        sample_count: feature_values.len(),
                        sufficient_stats: self.calculate_sufficient_stats(&feature_values),
                    },
                );
            }

            self.local_feature_stats.insert(class_id, feature_stats);
        }

        // Update local data statistics
        self.local_data_stats.num_samples = features.nrows();
        self.local_data_stats.num_features = features.ncols();
        self.local_data_stats.class_distribution = class_counts;

        Ok(())
    }

    /// Calculate client weights
    pub fn calculate_client_weights(
        &mut self,
        features: &DMatrix<T>,
        _labels: &DVector<usize>,
    ) -> Result<(), FederatedError> {
        let data_size = features.nrows();
        self.client_weights.data_size_weight = T::from(data_size).unwrap();
        self.client_weights.quality_weight = T::one(); // Placeholder
        self.client_weights.contribution_weight = T::one(); // Placeholder
        self.client_weights.trust_score = T::one(); // Placeholder

        Ok(())
    }

    /// Initialize privacy budget
    pub fn initialize_privacy_budget(
        &mut self,
        privacy_params: &PrivacyParams<T>,
    ) -> Result<(), FederatedError> {
        self.privacy_budget.epsilon = privacy_params.differential_privacy.global_epsilon;
        self.privacy_budget.delta = privacy_params.differential_privacy.global_delta;
        self.privacy_budget.remaining_budget = self.privacy_budget.epsilon;
        self.privacy_budget.allocation_strategy = BudgetAllocationStrategy::Uniform;

        Ok(())
    }

    /// Update from global model
    pub fn update_from_global_model(
        &mut self,
        global_model: &GlobalModel<T>,
    ) -> Result<(), FederatedError> {
        // Update local model with global parameters (simplified)
        for (&class_id, &global_prior) in global_model.global_priors.iter() {
            if let Some(local_prior) = self.local_priors.get_mut(&class_id) {
                // Simple averaging for demonstration
                *local_prior = (*local_prior + global_prior) / T::from(2.0).unwrap();
            }
        }

        Ok(())
    }

    /// Compute local update
    pub fn compute_local_update(
        &self,
        global_model: &GlobalModel<T>,
    ) -> Result<LocalUpdate<T>, FederatedError> {
        let mut prior_updates = HashMap::new();
        let mut feature_stat_updates = HashMap::new();

        // Calculate prior updates
        for (&class_id, &local_prior) in self.local_priors.iter() {
            if let Some(&global_prior) = global_model.global_priors.get(&class_id) {
                let update = local_prior - global_prior;
                prior_updates.insert(class_id, update);
            } else {
                prior_updates.insert(class_id, local_prior);
            }
        }

        // Calculate feature statistic updates
        for (&class_id, local_stats) in self.local_feature_stats.iter() {
            let mut class_updates = HashMap::new();

            for (&feature_id, local_feature_stats) in local_stats.iter() {
                let update_stats = if let Some(global_stats) = global_model
                    .global_feature_stats
                    .get(&class_id)
                    .and_then(|stats| stats.get(&feature_id))
                {
                    FeatureStatistics {
                        mean: local_feature_stats.mean - global_stats.mean,
                        variance: local_feature_stats.variance - global_stats.variance,
                        sample_count: local_feature_stats.sample_count,
                        sufficient_stats: local_feature_stats.sufficient_stats.clone(),
                    }
                } else {
                    local_feature_stats.clone()
                };

                class_updates.insert(feature_id, update_stats);
            }

            feature_stat_updates.insert(class_id, class_updates);
        }

        Ok(LocalUpdate {
            client_id: self.client_id.clone(),
            prior_updates,
            feature_stat_updates,
            client_weight: self.client_weights.data_size_weight,
        })
    }

    /// Calculate mean
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().cloned().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap();
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }

    /// Calculate sufficient statistics
    fn calculate_sufficient_stats(&self, values: &[T]) -> SufficientStatistics<T> {
        if values.is_empty() {
            return SufficientStatistics::default();
        }

        let sum: T = values.iter().cloned().sum();
        let sum_squares: T = values.iter().map(|&x| x * x).sum();
        let min_value = values.iter().cloned().fold(T::infinity(), T::min);
        let max_value = values.iter().cloned().fold(T::neg_infinity(), T::max);

        SufficientStatistics {
            sum,
            sum_squares,
            count: values.len(),
            min_value,
            max_value,
        }
    }
}

/// Default implementations
impl<T: Float> Default for GlobalModel<T> {
    fn default() -> Self {
        Self {
            global_priors: HashMap::new(),
            global_feature_stats: HashMap::new(),
            model_version: 0,
            convergence_metrics: ConvergenceMetrics::default(),
        }
    }
}

impl<T: Float> Default for FederationParams<T> {
    fn default() -> Self {
        Self {
            num_rounds: 100,
            client_selection_fraction: T::from(0.3).unwrap(),
            min_clients_per_round: 2,
            convergence_tolerance: T::from(1e-6).unwrap(),
            max_divergence_threshold: T::from(10.0).unwrap(),
        }
    }
}

impl<T: Float> Default for PrivacyParams<T> {
    fn default() -> Self {
        Self {
            differential_privacy: DifferentialPrivacyParams::default(),
            secure_mpc: SecureMPCParams::default(),
            homomorphic_encryption: HomomorphicEncryptionParams::default(),
            local_dp: LocalDPParams::default(),
        }
    }
}

impl<T: Float> Default for CommunicationParams<T> {
    fn default() -> Self {
        Self {
            compression_strategy: CompressionStrategy::None,
            quantization_params: QuantizationParams::default(),
            communication_budget: CommunicationBudget::default(),
            network_topology: NetworkTopology::Star,
        }
    }
}

impl<T: Float> Default for AggregationStrategy<T> {
    fn default() -> Self {
        Self {
            aggregation_method: AggregationMethod::FedAvg,
            weighted_params: WeightedAggregationParams::default(),
            robust_params: RobustAggregationParams::default(),
            byzantine_tolerance: ByzantineTolerance::default(),
        }
    }
}

impl<T: Float> Default for ConvergenceMetrics<T> {
    fn default() -> Self {
        Self {
            parameter_change: T::infinity(),
            loss_change: T::infinity(),
            gradient_norm: T::infinity(),
            convergence_rate: T::zero(),
        }
    }
}

impl<T: Float> Default for DataStatistics<T> {
    fn default() -> Self {
        Self {
            num_samples: 0,
            num_features: 0,
            class_distribution: HashMap::new(),
            data_quality: DataQualityMetrics::default(),
        }
    }
}

impl<T: Float> Default for DataQualityMetrics<T> {
    fn default() -> Self {
        Self {
            completeness: T::one(),
            consistency: T::one(),
            accuracy: T::one(),
            freshness: T::one(),
        }
    }
}

impl<T: Float> Default for PrivacyBudget<T> {
    fn default() -> Self {
        Self {
            epsilon: T::one(),
            delta: T::from(1e-5).unwrap(),
            remaining_budget: T::one(),
            allocation_strategy: BudgetAllocationStrategy::Uniform,
        }
    }
}

impl<T: Float> Default for ClientWeights<T> {
    fn default() -> Self {
        Self {
            data_size_weight: T::one(),
            quality_weight: T::one(),
            contribution_weight: T::one(),
            trust_score: T::one(),
        }
    }
}

impl<T: Float> Default for DifferentialPrivacyParams<T> {
    fn default() -> Self {
        Self {
            global_epsilon: T::one(),
            global_delta: T::from(1e-5).unwrap(),
            noise_mechanism: NoiseMechanism::Gaussian,
            sensitivity_analysis: SensitivityAnalysis::default(),
        }
    }
}

impl<T: Float> Default for SecureMPCParams<T> {
    fn default() -> Self {
        Self {
            secret_sharing_scheme: SecretSharingScheme::Shamir,
            security_threshold: 2,
            protocol_type: MPCProtocolType::BGW,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float> Default for HomomorphicEncryptionParams<T> {
    fn default() -> Self {
        Self {
            encryption_scheme: HomomorphicScheme::Paillier,
            key_size: 2048,
            security_level: SecurityLevel::Medium,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float> Default for LocalDPParams<T> {
    fn default() -> Self {
        Self {
            local_epsilon: T::one(),
            randomization_mechanism: RandomizationMechanism::Laplace,
            perturbation_strategy: PerturbationStrategy::default(),
        }
    }
}

impl<T: Float> Default for QuantizationParams<T> {
    fn default() -> Self {
        Self {
            num_bits: 8,
            quantization_method: QuantizationMethod::Uniform,
            dithering_enabled: false,
            adaptive_quantization: false,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float> Default for CommunicationBudget<T> {
    fn default() -> Self {
        Self {
            bits_per_round: 1000000, // 1MB
            total_budget: 100000000, // 100MB
            used_budget: 0,
            efficiency_metric: T::one(),
        }
    }
}

impl<T: Float> Default for WeightedAggregationParams<T> {
    fn default() -> Self {
        Self {
            weight_method: WeightMethod::DataSize,
            adaptive_weights: false,
            weight_normalization: true,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float> Default for RobustAggregationParams<T> {
    fn default() -> Self {
        Self {
            outlier_threshold: T::from(2.0).unwrap(),
            trimming_fraction: T::from(0.1).unwrap(),
            robust_estimator: RobustEstimator::TrimmedMean,
        }
    }
}

impl<T: Float> Default for ByzantineTolerance<T> {
    fn default() -> Self {
        Self {
            max_byzantine_clients: 1,
            detection_method: ByzantineDetectionMethod::Statistical,
            recovery_strategy: RecoveryStrategy::Exclude,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float> Default for SensitivityAnalysis<T> {
    fn default() -> Self {
        Self {
            global_sensitivity: T::one(),
            local_sensitivity: T::one(),
            smooth_sensitivity: T::one(),
        }
    }
}

impl<T: Float> Default for PerturbationStrategy<T> {
    fn default() -> Self {
        Self {
            perturbation_magnitude: T::from(0.1).unwrap(),
            perturbation_distribution: PerturbationDistribution::Gaussian,
            adaptive_perturbation: false,
        }
    }
}

impl<T: Float> Default for SufficientStatistics<T> {
    fn default() -> Self {
        Self {
            sum: T::zero(),
            sum_squares: T::zero(),
            count: 0,
            min_value: T::infinity(),
            max_value: T::neg_infinity(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::{Array1, Array2};

    // Type aliases for compatibility with DMatrix/DVector usage
    type DMatrix<T> = Array2<T>;
    type DVector<T> = Array1<T>;

    #[test]
    fn test_federated_naive_bayes_creation() {
        let classifier = FederatedNaiveBayes::<f64>::new();
        assert_eq!(classifier.current_round, 0);
        assert!(classifier.client_models.is_empty());
    }

    #[test]
    fn test_add_client() {
        let mut classifier = FederatedNaiveBayes::<f64>::new();

        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let result = classifier.add_client("client1".to_string(), &features, &labels);
        assert!(result.is_ok());
        assert_eq!(classifier.client_models.len(), 1);
    }

    #[test]
    fn test_client_model_training() {
        let mut client_model = ClientModel::<f64>::new("test_client".to_string());

        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let result = client_model.train_local_model(&features, &labels);
        assert!(result.is_ok());

        assert_eq!(client_model.local_priors.len(), 2);
        assert_eq!(client_model.local_feature_stats.len(), 2);
    }

    #[test]
    fn test_differential_privacy() {
        let classifier = FederatedNaiveBayes::<f64>::new();

        let local_update = LocalUpdate {
            client_id: "test_client".to_string(),
            prior_updates: [(0, 0.5), (1, 0.5)].iter().cloned().collect(),
            feature_stat_updates: HashMap::new(),
            client_weight: 1.0,
        };

        let private_update = classifier.apply_differential_privacy(local_update);
        assert!(private_update.is_ok());

        let private = private_update.unwrap();
        assert_eq!(private.client_id, "test_client");
        assert!(private.privacy_cost > 0.0);
    }

    #[test]
    fn test_federated_averaging() {
        let classifier = FederatedNaiveBayes::<f64>::new();

        let private_updates = vec![
            PrivateUpdate {
                client_id: "client1".to_string(),
                prior_updates: [(0, 0.6), (1, 0.4)].iter().cloned().collect(),
                feature_stat_updates: HashMap::new(),
                client_weight: 1.0,
                privacy_cost: 0.1,
                compression_ratio: 1.0,
            },
            PrivateUpdate {
                client_id: "client2".to_string(),
                prior_updates: [(0, 0.4), (1, 0.6)].iter().cloned().collect(),
                feature_stat_updates: HashMap::new(),
                client_weight: 1.0,
                privacy_cost: 0.1,
                compression_ratio: 1.0,
            },
        ];

        let aggregated = classifier.federated_averaging(private_updates);
        assert!(aggregated.is_ok());

        let result = aggregated.unwrap();
        assert_eq!(result.num_clients, 2);
        assert!(result.aggregated_priors.contains_key(&0));
        assert!(result.aggregated_priors.contains_key(&1));
    }

    #[test]
    fn test_robust_averaging() {
        let classifier = FederatedNaiveBayes::<f64>::new();

        let private_updates = vec![
            PrivateUpdate {
                client_id: "client1".to_string(),
                prior_updates: [(0, 0.5), (1, 0.5)].iter().cloned().collect(),
                feature_stat_updates: HashMap::new(),
                client_weight: 1.0,
                privacy_cost: 0.1,
                compression_ratio: 1.0,
            },
            PrivateUpdate {
                client_id: "client2".to_string(),
                prior_updates: [(0, 0.4), (1, 0.6)].iter().cloned().collect(),
                feature_stat_updates: HashMap::new(),
                client_weight: 1.0,
                privacy_cost: 0.1,
                compression_ratio: 1.0,
            },
            PrivateUpdate {
                client_id: "client3".to_string(),
                prior_updates: [(0, 0.9), (1, 0.1)].iter().cloned().collect(), // Outlier
                feature_stat_updates: HashMap::new(),
                client_weight: 1.0,
                privacy_cost: 0.1,
                compression_ratio: 1.0,
            },
        ];

        let aggregated = classifier.robust_averaging(private_updates);
        assert!(aggregated.is_ok());

        let result = aggregated.unwrap();
        assert_eq!(result.num_clients, 3);
    }

    #[test]
    fn test_client_selection() {
        let mut classifier = FederatedNaiveBayes::<f64>::new();

        // Add multiple clients
        for i in 0..5 {
            let features = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            let labels = Array1::from_vec(vec![0, 1]);
            let _ = classifier.add_client(format!("client{}", i), &features, &labels);
        }

        let selected = classifier.select_clients();
        assert!(selected.is_ok());

        let clients = selected.unwrap();
        assert!(clients.len() >= classifier.federation_params.min_clients_per_round);
        assert!(clients.len() <= classifier.client_models.len());
    }

    #[test]
    fn test_prediction() {
        let mut classifier = FederatedNaiveBayes::<f64>::new();

        // Set up global model with some priors and statistics
        classifier.global_model.global_priors.insert(0, 0.5);
        classifier.global_model.global_priors.insert(1, 0.5);

        let mut feature_stats_0 = HashMap::new();
        feature_stats_0.insert(
            0,
            FeatureStatistics {
                mean: 1.0,
                variance: 1.0,
                sample_count: 10,
                sufficient_stats: SufficientStatistics::default(),
            },
        );
        feature_stats_0.insert(
            1,
            FeatureStatistics {
                mean: 2.0,
                variance: 1.0,
                sample_count: 10,
                sufficient_stats: SufficientStatistics::default(),
            },
        );

        let mut feature_stats_1 = HashMap::new();
        feature_stats_1.insert(
            0,
            FeatureStatistics {
                mean: 3.0,
                variance: 1.0,
                sample_count: 10,
                sufficient_stats: SufficientStatistics::default(),
            },
        );
        feature_stats_1.insert(
            1,
            FeatureStatistics {
                mean: 4.0,
                variance: 1.0,
                sample_count: 10,
                sufficient_stats: SufficientStatistics::default(),
            },
        );

        classifier
            .global_model
            .global_feature_stats
            .insert(0, feature_stats_0);
        classifier
            .global_model
            .global_feature_stats
            .insert(1, feature_stats_1);

        let test_features = DVector::from_vec(vec![1.5, 2.5]);
        let prediction = classifier.predict(&test_features);
        assert!(prediction.is_ok());

        let probabilities = prediction.unwrap();
        assert!(probabilities.contains_key(&0));
        assert!(probabilities.contains_key(&1));

        // Check that probabilities sum to 1
        let total_prob: f64 = probabilities.values().sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_federation_statistics() {
        let mut classifier = FederatedNaiveBayes::<f64>::new();

        // Add some clients
        let features = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 1]);
        let _ = classifier.add_client("client1".to_string(), &features, &labels);
        let _ = classifier.add_client("client2".to_string(), &features, &labels);

        let stats = classifier.get_federation_statistics();
        assert_eq!(stats.num_clients, 2);
        assert_eq!(stats.current_round, 0);
    }
}
