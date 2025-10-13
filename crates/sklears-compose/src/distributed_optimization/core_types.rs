//! Core Types and Configuration for Distributed Optimization
//!
//! This module contains the fundamental types, enums, and configuration structures
//! used throughout the distributed optimization framework.
//!
//! # Core Components
//! - Configuration structures for distributed optimization
//! - Communication protocols and optimization strategies
//! - Privacy and fault tolerance settings
//! - Basic solution and metadata types
//! - Consensus and synchronization primitives

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

/// Node identifier for distributed systems
pub type NodeId = String;

/// Configuration for distributed optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedOptimizationConfig {
    pub max_nodes: usize,
    pub sync_interval: Duration,
    pub convergence_threshold: f64,
    pub max_iterations: u32,
    pub fault_tolerance: FaultToleranceConfig,
    pub communication_protocol: CommunicationProtocol,
    pub optimization_strategy: OptimizationStrategy,
    pub privacy_settings: PrivacySettings,
}

impl Default for DistributedOptimizationConfig {
    fn default() -> Self {
        Self {
            max_nodes: 10,
            sync_interval: Duration::from_secs(60),
            convergence_threshold: 1e-6,
            max_iterations: 1000,
            fault_tolerance: FaultToleranceConfig::default(),
            communication_protocol: CommunicationProtocol::AllReduce,
            optimization_strategy: OptimizationStrategy::DistributedGradientDescent,
            privacy_settings: PrivacySettings::default(),
        }
    }
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    pub enable_checkpointing: bool,
    pub checkpoint_interval: Duration,
    pub node_failure_threshold: f64,
    pub recovery_strategy: RecoveryStrategy,
    pub backup_replicas: usize,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_checkpointing: true,
            checkpoint_interval: Duration::from_secs(300),
            node_failure_threshold: 0.1,
            recovery_strategy: RecoveryStrategy::RestartFromCheckpoint,
            backup_replicas: 2,
        }
    }
}

/// Recovery strategies for node failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    RestartFromCheckpoint,
    RebalanceWorkload,
    GracefulDegradation,
    WaitForRecovery,
    DynamicReplacement,
}

/// Communication protocols for distributed systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    AllReduce,
    ParameterServer,
    PeerToPeer,
    RingAllReduce,
    TreeReduce,
    AsyncGossip,
}

impl CommunicationProtocol {
    /// Get the default bandwidth efficiency for this protocol
    pub fn bandwidth_efficiency(&self) -> f64 {
        match self {
            CommunicationProtocol::AllReduce => 0.8,
            CommunicationProtocol::ParameterServer => 0.7,
            CommunicationProtocol::PeerToPeer => 0.6,
            CommunicationProtocol::RingAllReduce => 0.9,
            CommunicationProtocol::TreeReduce => 0.85,
            CommunicationProtocol::AsyncGossip => 0.5,
        }
    }

    /// Check if the protocol supports fault tolerance
    pub fn supports_fault_tolerance(&self) -> bool {
        match self {
            CommunicationProtocol::AllReduce => true,
            CommunicationProtocol::ParameterServer => true,
            CommunicationProtocol::PeerToPeer => false,
            CommunicationProtocol::RingAllReduce => false,
            CommunicationProtocol::TreeReduce => true,
            CommunicationProtocol::AsyncGossip => true,
        }
    }
}

/// Optimization strategies for distributed learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    FederatedLearning,
    DistributedGradientDescent,
    EvolutionaryOptimization,
    BayesianOptimization,
    ParticleSwarmOptimization,
    HybridOptimization,
}

impl OptimizationStrategy {
    /// Get the typical convergence rate for this strategy
    pub fn typical_convergence_rate(&self) -> f64 {
        match self {
            OptimizationStrategy::FederatedLearning => 0.01,
            OptimizationStrategy::DistributedGradientDescent => 0.1,
            OptimizationStrategy::EvolutionaryOptimization => 0.001,
            OptimizationStrategy::BayesianOptimization => 0.05,
            OptimizationStrategy::ParticleSwarmOptimization => 0.02,
            OptimizationStrategy::HybridOptimization => 0.08,
        }
    }

    /// Check if the strategy requires gradient information
    pub fn requires_gradients(&self) -> bool {
        match self {
            OptimizationStrategy::FederatedLearning => true,
            OptimizationStrategy::DistributedGradientDescent => true,
            OptimizationStrategy::EvolutionaryOptimization => false,
            OptimizationStrategy::BayesianOptimization => false,
            OptimizationStrategy::ParticleSwarmOptimization => false,
            OptimizationStrategy::HybridOptimization => true,
        }
    }
}

/// Privacy settings for distributed optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    pub enable_differential_privacy: bool,
    pub privacy_budget: f64,
    pub noise_multiplier: f64,
    pub enable_secure_aggregation: bool,
    pub homomorphic_encryption: bool,
}

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            enable_differential_privacy: false,
            privacy_budget: 1.0,
            noise_multiplier: 1.0,
            enable_secure_aggregation: false,
            homomorphic_encryption: false,
        }
    }
}

impl PrivacySettings {
    /// Create privacy settings for high privacy scenarios
    pub fn high_privacy() -> Self {
        Self {
            enable_differential_privacy: true,
            privacy_budget: 0.1,
            noise_multiplier: 2.0,
            enable_secure_aggregation: true,
            homomorphic_encryption: true,
        }
    }

    /// Create privacy settings for balanced privacy/performance
    pub fn balanced_privacy() -> Self {
        Self {
            enable_differential_privacy: true,
            privacy_budget: 1.0,
            noise_multiplier: 1.0,
            enable_secure_aggregation: true,
            homomorphic_encryption: false,
        }
    }

    /// Check if any privacy measures are enabled
    pub fn has_privacy_measures(&self) -> bool {
        self.enable_differential_privacy
            || self.enable_secure_aggregation
            || self.homomorphic_encryption
    }
}

/// Session status tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    Initializing,
    Running,
    Paused,
    Converged,
    Failed(String),
    Terminated,
}

impl SessionStatus {
    /// Check if the session is in an active state
    pub fn is_active(&self) -> bool {
        matches!(self, SessionStatus::Initializing | SessionStatus::Running)
    }

    /// Check if the session has finished (either successfully or unsuccessfully)
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            SessionStatus::Converged | SessionStatus::Failed(_) | SessionStatus::Terminated
        )
    }

    /// Check if the session can be resumed
    pub fn can_resume(&self) -> bool {
        matches!(self, SessionStatus::Paused)
    }
}

/// Optimization solution representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSolution {
    pub parameters: HashMap<String, f64>,
    pub objective_value: f64,
    pub constraints_satisfied: bool,
    pub metadata: SolutionMetadata,
    pub validation_scores: Vec<f64>,
}

impl OptimizationSolution {
    /// Create a new optimization solution
    pub fn new(
        parameters: HashMap<String, f64>,
        objective_value: f64,
        contributor_node: NodeId,
    ) -> Self {
        Self {
            parameters,
            objective_value,
            constraints_satisfied: true,
            metadata: SolutionMetadata {
                generation_time: SystemTime::now(),
                contributor_node,
                computation_cost: 0.0,
                convergence_metrics: ConvergenceMetric::default(),
                quality_score: 0.0,
            },
            validation_scores: Vec::new(),
        }
    }

    /// Check if this solution is better than another
    pub fn is_better_than(&self, other: &OptimizationSolution) -> bool {
        self.objective_value > other.objective_value && self.constraints_satisfied
    }

    /// Get the parameter value by name
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameters.get(name).copied()
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, name: String, value: f64) {
        self.parameters.insert(name, value);
    }

    /// Calculate the average validation score
    pub fn average_validation_score(&self) -> f64 {
        if self.validation_scores.is_empty() {
            0.0
        } else {
            self.validation_scores.iter().sum::<f64>() / self.validation_scores.len() as f64
        }
    }
}

/// Metadata for optimization solutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionMetadata {
    pub generation_time: SystemTime,
    pub contributor_node: NodeId,
    pub computation_cost: f64,
    pub convergence_metrics: ConvergenceMetric,
    pub quality_score: f64,
}

/// Convergence metrics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetric {
    pub iteration: u32,
    pub objective_value: f64,
    pub gradient_norm: f64,
    pub parameter_change: f64,
    pub convergence_rate: f64,
    pub timestamp: SystemTime,
}

impl Default for ConvergenceMetric {
    fn default() -> Self {
        Self {
            iteration: 0,
            objective_value: 0.0,
            gradient_norm: 0.0,
            parameter_change: 0.0,
            convergence_rate: 0.0,
            timestamp: SystemTime::now(),
        }
    }
}

impl ConvergenceMetric {
    /// Create a new convergence metric
    pub fn new(
        iteration: u32,
        objective_value: f64,
        gradient_norm: f64,
        parameter_change: f64,
    ) -> Self {
        Self {
            iteration,
            objective_value,
            gradient_norm,
            parameter_change,
            convergence_rate: 0.0,
            timestamp: SystemTime::now(),
        }
    }

    /// Check if this metric indicates convergence
    pub fn indicates_convergence(&self, threshold: f64) -> bool {
        self.gradient_norm < threshold && self.parameter_change < threshold
    }

    /// Calculate improvement from previous metric
    pub fn improvement_from(&self, previous: &ConvergenceMetric) -> f64 {
        self.objective_value - previous.objective_value
    }
}

/// Consensus algorithms for distributed coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    Raft,
    PBFT,
    WeightedVoting,
    ByzantineFaultTolerant,
    QuorumBased,
}

impl ConsensusAlgorithm {
    /// Get the fault tolerance capacity of the algorithm
    pub fn fault_tolerance_ratio(&self) -> f64 {
        match self {
            ConsensusAlgorithm::Raft => 0.5,
            ConsensusAlgorithm::PBFT => 0.33,
            ConsensusAlgorithm::WeightedVoting => 0.5,
            ConsensusAlgorithm::ByzantineFaultTolerant => 0.33,
            ConsensusAlgorithm::QuorumBased => 0.5,
        }
    }

    /// Check if the algorithm can handle Byzantine failures
    pub fn handles_byzantine_failures(&self) -> bool {
        matches!(
            self,
            ConsensusAlgorithm::PBFT | ConsensusAlgorithm::ByzantineFaultTolerant
        )
    }
}

/// Consensus vote from participating nodes
#[derive(Debug, Clone)]
pub struct ConsensusVote {
    pub node_id: NodeId,
    pub proposed_solution: OptimizationSolution,
    pub confidence: f64,
    pub timestamp: SystemTime,
}

impl ConsensusVote {
    /// Create a new consensus vote
    pub fn new(node_id: NodeId, proposed_solution: OptimizationSolution, confidence: f64) -> Self {
        Self {
            node_id,
            proposed_solution,
            confidence,
            timestamp: SystemTime::now(),
        }
    }

    /// Check if this vote is from a recent time window
    pub fn is_recent(&self, window: Duration) -> bool {
        if let Ok(elapsed) = self.timestamp.elapsed() {
            elapsed <= window
        } else {
            false
        }
    }
}

/// Consensus state for distributed decisions
#[derive(Debug, Clone)]
pub struct ConsensusState {
    pub agreed_parameters: HashMap<String, f64>,
    pub consensus_confidence: f64,
    pub voting_history: VecDeque<ConsensusVote>,
    pub consensus_algorithm: ConsensusAlgorithm,
}

impl ConsensusState {
    /// Create a new consensus state
    pub fn new(algorithm: ConsensusAlgorithm) -> Self {
        Self {
            agreed_parameters: HashMap::new(),
            consensus_confidence: 0.0,
            voting_history: VecDeque::new(),
            consensus_algorithm: algorithm,
        }
    }

    /// Add a vote to the consensus
    pub fn add_vote(&mut self, vote: ConsensusVote) {
        self.voting_history.push_back(vote);
        // Keep only recent votes (last 100)
        if self.voting_history.len() > 100 {
            self.voting_history.pop_front();
        }
    }

    /// Check if consensus has been reached
    pub fn has_consensus(&self, min_confidence: f64) -> bool {
        self.consensus_confidence >= min_confidence
    }

    /// Get the most recent votes within a time window
    pub fn recent_votes(&self, window: Duration) -> Vec<&ConsensusVote> {
        self.voting_history
            .iter()
            .filter(|vote| vote.is_recent(window))
            .collect()
    }
}

/// Synchronization barrier for distributed coordination
#[derive(Debug, Clone)]
pub struct SynchronizationBarrier {
    pub waiting_nodes: Vec<NodeId>,
    pub barrier_threshold: usize,
    pub timeout: Duration,
    pub current_round: u32,
}

impl SynchronizationBarrier {
    /// Create a new synchronization barrier
    pub fn new(barrier_threshold: usize, timeout: Duration) -> Self {
        Self {
            waiting_nodes: Vec::new(),
            barrier_threshold,
            timeout,
            current_round: 0,
        }
    }

    /// Add a node to the barrier
    pub fn add_node(&mut self, node_id: NodeId) {
        if !self.waiting_nodes.contains(&node_id) {
            self.waiting_nodes.push(node_id);
        }
    }

    /// Remove a node from the barrier
    pub fn remove_node(&mut self, node_id: &NodeId) {
        self.waiting_nodes.retain(|id| id != node_id);
    }

    /// Check if the barrier can be released
    pub fn can_release(&self) -> bool {
        self.waiting_nodes.len() >= self.barrier_threshold
    }

    /// Release the barrier and start next round
    pub fn release(&mut self) {
        self.waiting_nodes.clear();
        self.current_round += 1;
    }
}

/// Convergence criteria for optimization
#[derive(Debug, Clone)]
pub enum ConvergenceCriterion {
    ObjectiveThreshold(f64),
    GradientNormThreshold(f64),
    ParameterChangeThreshold(f64),
    MaxIterations(u32),
    RelativeImprovement(f64),
    Statistical(StatisticalCriterion),
}

impl ConvergenceCriterion {
    /// Check if the criterion is satisfied
    pub fn is_satisfied(&self, metric: &ConvergenceMetric, history: &[ConvergenceMetric]) -> bool {
        match self {
            ConvergenceCriterion::ObjectiveThreshold(threshold) => {
                metric.objective_value >= *threshold
            }
            ConvergenceCriterion::GradientNormThreshold(threshold) => {
                metric.gradient_norm <= *threshold
            }
            ConvergenceCriterion::ParameterChangeThreshold(threshold) => {
                metric.parameter_change <= *threshold
            }
            ConvergenceCriterion::MaxIterations(max_iter) => metric.iteration >= *max_iter,
            ConvergenceCriterion::RelativeImprovement(min_improvement) => {
                if let Some(previous) = history.last() {
                    let improvement = metric.improvement_from(previous);
                    improvement.abs() <= *min_improvement
                } else {
                    false
                }
            }
            ConvergenceCriterion::Statistical(stat_criterion) => {
                // Placeholder for statistical test implementation
                history.len() >= stat_criterion.min_samples
            }
        }
    }
}

/// Statistical convergence criteria
#[derive(Debug, Clone)]
pub struct StatisticalCriterion {
    pub test_type: StatisticalTest,
    pub significance_level: f64,
    pub window_size: usize,
    pub min_samples: usize,
}

/// Statistical tests for convergence
#[derive(Debug, Clone)]
pub enum StatisticalTest {
    MannKendall,
    AugmentedDickeyFuller,
    KolmogorovSmirnov,
    LjungBox,
    CustomTest(String),
}

/// Trend detection methods
#[derive(Debug, Clone)]
pub enum TrendDetectionMethod {
    LinearRegression,
    MannKendall,
    Theil_Sen,
    SeasonalDecomposition,
    WaveletAnalysis,
}

/// Changepoint detection algorithms
#[derive(Debug, Clone)]
pub enum ChangepointAlgorithm {
    CUSUM,
    PELT,
    BinarySegmentation,
    KernelChangePoint,
    BayesianChangePoint,
}

/// Stationarity tests
#[derive(Debug, Clone)]
pub enum StationarityTest {
    AugmentedDickeyFuller,
    KPSS,
    PhillipsPerron,
    ZivotAndrews,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub enable_early_stopping: bool,
    pub patience: u32,
    pub min_delta: f64,
    pub monitor_metric: String,
    pub mode: EarlyStoppingMode,
    pub restore_best_weights: bool,
    pub baseline: Option<f64>,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enable_early_stopping: false,
            patience: 10,
            min_delta: 1e-4,
            monitor_metric: "objective_value".to_string(),
            mode: EarlyStoppingMode::Max,
            restore_best_weights: true,
            baseline: None,
        }
    }
}

/// Early stopping modes
#[derive(Debug, Clone)]
pub enum EarlyStoppingMode {
    Min,
    Max,
    Auto,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_optimization_config_default() {
        let config = DistributedOptimizationConfig::default();
        assert_eq!(config.max_nodes, 10);
        assert_eq!(config.max_iterations, 1000);
        assert!(matches!(
            config.communication_protocol,
            CommunicationProtocol::AllReduce
        ));
    }

    #[test]
    fn test_communication_protocol_properties() {
        let protocol = CommunicationProtocol::RingAllReduce;
        assert!(protocol.bandwidth_efficiency() > 0.8);
        assert!(!protocol.supports_fault_tolerance());

        let protocol = CommunicationProtocol::AllReduce;
        assert!(protocol.supports_fault_tolerance());
    }

    #[test]
    fn test_optimization_strategy_properties() {
        let strategy = OptimizationStrategy::DistributedGradientDescent;
        assert!(strategy.requires_gradients());
        assert!(strategy.typical_convergence_rate() > 0.0);

        let strategy = OptimizationStrategy::EvolutionaryOptimization;
        assert!(!strategy.requires_gradients());
    }

    #[test]
    fn test_privacy_settings() {
        let high_privacy = PrivacySettings::high_privacy();
        assert!(high_privacy.has_privacy_measures());
        assert!(high_privacy.enable_differential_privacy);
        assert!(high_privacy.enable_secure_aggregation);

        let default_privacy = PrivacySettings::default();
        assert!(!default_privacy.has_privacy_measures());
    }

    #[test]
    fn test_session_status() {
        let status = SessionStatus::Running;
        assert!(status.is_active());
        assert!(!status.is_finished());

        let status = SessionStatus::Converged;
        assert!(!status.is_active());
        assert!(status.is_finished());

        let status = SessionStatus::Paused;
        assert!(status.can_resume());
    }

    #[test]
    fn test_optimization_solution() {
        let mut params = HashMap::new();
        params.insert("x".to_string(), 1.0);
        params.insert("y".to_string(), 2.0);

        let solution = OptimizationSolution::new(params, 10.0, "node1".to_string());
        assert_eq!(solution.objective_value, 10.0);
        assert_eq!(solution.get_parameter("x"), Some(1.0));
        assert_eq!(solution.get_parameter("z"), None);

        let mut params2 = HashMap::new();
        params2.insert("x".to_string(), 0.5);
        let solution2 = OptimizationSolution::new(params2, 5.0, "node2".to_string());

        assert!(solution.is_better_than(&solution2));
    }

    #[test]
    fn test_convergence_metric() {
        let metric = ConvergenceMetric::new(10, 100.0, 0.01, 0.001);
        assert!(metric.indicates_convergence(0.1));
        assert!(!metric.indicates_convergence(0.0001));

        let previous = ConvergenceMetric::new(9, 90.0, 0.02, 0.002);
        assert_eq!(metric.improvement_from(&previous), 10.0);
    }

    #[test]
    fn test_consensus_algorithm() {
        let raft = ConsensusAlgorithm::Raft;
        assert_eq!(raft.fault_tolerance_ratio(), 0.5);
        assert!(!raft.handles_byzantine_failures());

        let pbft = ConsensusAlgorithm::PBFT;
        assert!(pbft.handles_byzantine_failures());
    }

    #[test]
    fn test_synchronization_barrier() {
        let mut barrier = SynchronizationBarrier::new(3, Duration::from_secs(60));
        assert!(!barrier.can_release());

        barrier.add_node("node1".to_string());
        barrier.add_node("node2".to_string());
        barrier.add_node("node3".to_string());
        assert!(barrier.can_release());

        barrier.release();
        assert_eq!(barrier.current_round, 1);
        assert_eq!(barrier.waiting_nodes.len(), 0);
    }

    #[test]
    fn test_consensus_state() {
        let mut state = ConsensusState::new(ConsensusAlgorithm::Raft);
        assert!(!state.has_consensus(0.8));

        let solution = OptimizationSolution::new(HashMap::new(), 10.0, "node1".to_string());
        let vote = ConsensusVote::new("node1".to_string(), solution, 0.9);
        state.add_vote(vote);

        assert_eq!(state.voting_history.len(), 1);
    }

    #[test]
    fn test_convergence_criterion() {
        let metric = ConvergenceMetric::new(10, 100.0, 0.01, 0.001);
        let history = vec![];

        let criterion = ConvergenceCriterion::ObjectiveThreshold(50.0);
        assert!(criterion.is_satisfied(&metric, &history));

        let criterion = ConvergenceCriterion::GradientNormThreshold(0.1);
        assert!(criterion.is_satisfied(&metric, &history));

        let criterion = ConvergenceCriterion::MaxIterations(5);
        assert!(criterion.is_satisfied(&metric, &history));
    }
}