//! Distributed optimization module
//!
//! This module provides comprehensive distributed optimization capabilities
//! organized into focused, maintainable submodules following the 2000-line policy.
//!
//! ## Architecture
//!
//! The distributed optimization framework is organized into 8 core modules:
//!
//! - **distributed_core**: Main coordination engine with comprehensive distributed optimization
//! - **consensus_algorithms**: Byzantine fault-tolerant consensus protocols (PBFT, averaging)
//! - **node_management**: Node registry, health monitoring, and capacity planning
//! - **communication_layer**: Secure messaging, routing, and network management
//! - **fault_tolerance**: Byzantine detection, recovery strategies, and resilience
//! - **load_balancing**: Advanced load balancing with SIMD acceleration
//! - **performance_optimization**: Metrics, profiling, and performance tuning
//! - **federated_learning**: Federated optimization algorithms with privacy preservation
//!
//! ## Quick Start
//!
//! ```rust
//! use sklears_compose::pattern_optimization::distributed_optimization::{
//!     DistributedOptimizer, DistributedOptimizerBuilder
//! };
//!
//! // Create a distributed optimizer
//! let optimizer = DistributedOptimizerBuilder::new()
//!     .with_consensus_algorithm("pbft")
//!     .with_fault_tolerance_enabled(true)
//!     .with_load_balancing_strategy("adaptive")
//!     .build()?;
//!
//! // Start optimization
//! optimizer.start_optimization(&problem)?;
//! ```

// Core modules
pub mod distributed_core;
pub mod consensus_algorithms;
pub mod node_management;
pub mod communication_layer;
pub mod fault_tolerance;
pub mod load_balancing;
pub mod performance_optimization;
pub mod federated_learning;

// Re-export main types from distributed_core
pub use distributed_core::{
    DistributedOptimizer, DistributedOptimizerBuilder, OptimizationProblem,
    DistributedSolution, OptimizationConfig, DistributedOptimizerError,
    ParallelOptimizer, FederatedOptimizer as FederatedOptimizerTrait,
    CoordinationProtocol, SynchronizationManager, SecurityManager,
    DistributedPerformanceMonitor, DistributedResourceManager, NetworkTopology,
    SimdDistributedAccelerator,
};

// Re-export consensus algorithm types
pub use consensus_algorithms::{
    ConsensusAlgorithm, PBFTConsensus, AveragingConsensus, ConsensusResponse,
    ConsensusStatistics, Proposal, PrepareMessage, CommitMessage, ViewChangeMessage,
    SimdConsensusAccelerator, ConsensusOptimizerFactory,
};

// Re-export node management types
pub use node_management::{
    NodeRegistry, NodeInfo, NodeStatus, NodeCapacity, NodePerformanceSnapshot,
    FailureEvent, SecurityEvent, ResourceAllocation, HeartbeatMonitor,
    CapacityPlanner, NodeHealthAnalyzer, NodeSimdAccelerator,
};

// Re-export communication layer types
pub use communication_layer::{
    CommunicationManager, CommunicationChannel, Message, MessageType,
    EncryptionManager, CompressionManager, BandwidthMonitor, MessageStatistics,
    NetworkTopologyManager, QualityOfServiceManager, CommunicationSecurityMonitor,
    CommunicationSimdAccelerator,
};

// Re-export fault tolerance types
pub use fault_tolerance::{
    DistributedFaultHandler, FaultDetector, RecoveryStrategy, ByzantineDetector,
    CheckpointManager, RedundancyManager, FailureType, MLBasedFaultDetector,
    PredictiveFaultAnalyzer, ResilienceMonitor, CascadingFailureDetector,
    RecoveryOptimizer, FaultToleranceSimdAccelerator,
};

// Re-export load balancing types
pub use load_balancing::{
    OptimizationLoadBalancer, LoadBalancingStrategy, WorkerLoad, AllocationSnapshot,
    LoadPredictionEngine, AdaptiveLoadController, ResourceOptimizer,
    LoadBalancingPerformanceMonitor, AutoScalingManager, EnergyAwareOptimizer,
    LoadBalancingSimdAccelerator,
};

// Re-export performance optimization types
pub use performance_optimization::{
    DistributedPerformanceMonitor as PerformanceMonitor, PerformanceMetrics,
    PerformanceSnapshot, WorkloadCharacteristics, SimdPerformanceAccelerator,
    PerformanceThresholds, PerformanceTrendAnalysis, OptimizationTarget,
    OptimizationResult, AdaptivePerformanceTuner, PerformancePredictionEngine,
    DistributedBenchmarkSuite, ResourceOptimizer as ResourceOptimizerPerf,
    OptimizationStrategy, OptimizationAction,
};

// Re-export federated learning types
pub use federated_learning::{
    FederatedOptimizer, FedAvgOptimizer, FedProxOptimizer, ClientInfo,
    ClientUpdate, GlobalUpdate, ConvergenceStatus, PrivacyMechanism,
    CommunicationStrategy, ClientSelectionStrategy, SimdFederatedAccelerator,
    CompressionEngine, CompressionAlgorithm, PrivacyEngine,
    FederatedPerformanceMonitor, FederatedMetricsSummary,
};

use scirs2_core::error::{CoreError, Result as SklResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Factory for creating distributed optimization components
pub struct DistributedOptimizationFactory;

impl DistributedOptimizationFactory {
    /// Create a new distributed optimizer with default configuration
    pub fn create_optimizer() -> SklResult<DistributedOptimizer> {
        DistributedOptimizerBuilder::new().build()
    }

    /// Create a federated optimizer with specified algorithm
    pub fn create_federated_optimizer(
        algorithm: &str,
        convergence_threshold: f64,
        min_clients: usize,
        max_rounds: u64,
    ) -> SklResult<Box<dyn FederatedOptimizerTrait>> {
        match algorithm {
            "fedavg" => Ok(Box::new(FedAvgOptimizer::new(
                convergence_threshold,
                min_clients,
                max_rounds,
            ))),
            "fedprox" => Ok(Box::new(FedProxOptimizer::new(
                convergence_threshold,
                min_clients,
                max_rounds,
                0.1, // default mu parameter
            ))),
            _ => Err(CoreError::InvalidInput(format!(
                "Unknown federated algorithm: {}",
                algorithm
            ))),
        }
    }

    /// Create a consensus algorithm by name
    pub fn create_consensus_algorithm(algorithm: &str) -> SklResult<Box<dyn ConsensusAlgorithm>> {
        match algorithm {
            "pbft" => Ok(Box::new(PBFTConsensus::new(
                "default_node".to_string(),
                Vec::new(),
                3, // max byzantine nodes
            ))),
            "averaging" => Ok(Box::new(AveragingConsensus::new(
                "default_node".to_string(),
                Vec::new(),
            ))),
            _ => Err(CoreError::InvalidInput(format!(
                "Unknown consensus algorithm: {}",
                algorithm
            ))),
        }
    }

    /// Create a comprehensive distributed optimization suite
    pub fn create_optimization_suite(config: OptimizationSuiteConfig) -> SklResult<DistributedOptimizationSuite> {
        let mut suite = DistributedOptimizationSuite::new();

        // Create and configure optimizer
        let mut optimizer_builder = DistributedOptimizerBuilder::new();

        if let Some(consensus_alg) = &config.consensus_algorithm {
            optimizer_builder = optimizer_builder.with_consensus_algorithm(consensus_alg);
        }

        if config.enable_fault_tolerance {
            optimizer_builder = optimizer_builder.with_fault_tolerance_enabled(true);
        }

        if let Some(load_balancing) = &config.load_balancing_strategy {
            optimizer_builder = optimizer_builder.with_load_balancing_strategy(load_balancing);
        }

        suite.optimizer = Some(optimizer_builder.build()?);

        // Create federated learning components if requested
        if let Some(fed_config) = &config.federated_learning {
            let fed_optimizer = Self::create_federated_optimizer(
                &fed_config.algorithm,
                fed_config.convergence_threshold,
                fed_config.min_clients_per_round,
                fed_config.max_rounds,
            )?;
            suite.federated_optimizer = Some(fed_optimizer);
        }

        // Create performance monitoring
        suite.performance_monitor = Some(PerformanceMonitor::new());

        Ok(suite)
    }
}

/// Configuration for creating an optimization suite
#[derive(Debug, Clone)]
pub struct OptimizationSuiteConfig {
    pub consensus_algorithm: Option<String>,
    pub enable_fault_tolerance: bool,
    pub load_balancing_strategy: Option<String>,
    pub federated_learning: Option<FederatedLearningConfig>,
    pub performance_monitoring: bool,
}

#[derive(Debug, Clone)]
pub struct FederatedLearningConfig {
    pub algorithm: String,
    pub convergence_threshold: f64,
    pub min_clients_per_round: usize,
    pub max_rounds: u64,
    pub privacy_mechanism: Option<String>,
    pub communication_strategy: Option<String>,
}

impl Default for OptimizationSuiteConfig {
    fn default() -> Self {
        Self {
            consensus_algorithm: Some("pbft".to_string()),
            enable_fault_tolerance: true,
            load_balancing_strategy: Some("adaptive".to_string()),
            federated_learning: None,
            performance_monitoring: true,
        }
    }
}

/// Comprehensive distributed optimization suite
pub struct DistributedOptimizationSuite {
    pub optimizer: Option<DistributedOptimizer>,
    pub federated_optimizer: Option<Box<dyn FederatedOptimizerTrait>>,
    pub performance_monitor: Option<PerformanceMonitor>,
    pub node_registry: Option<NodeRegistry>,
    pub communication_manager: Option<CommunicationManager>,
}

impl DistributedOptimizationSuite {
    pub fn new() -> Self {
        Self {
            optimizer: None,
            federated_optimizer: None,
            performance_monitor: None,
            node_registry: None,
            communication_manager: None,
        }
    }

    /// Initialize the complete optimization suite
    pub fn initialize(&mut self) -> SklResult<()> {
        // Initialize performance monitoring
        if let Some(ref monitor) = self.performance_monitor {
            monitor.start_monitoring()?;
        }

        // Initialize node registry
        if self.node_registry.is_none() {
            self.node_registry = Some(NodeRegistry::new());
        }

        // Initialize communication manager
        if self.communication_manager.is_none() {
            self.communication_manager = Some(CommunicationManager::new());
        }

        Ok(())
    }

    /// Start distributed optimization
    pub fn start_optimization(&mut self, problem: &OptimizationProblem) -> SklResult<DistributedSolution> {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.start_optimization(problem)
        } else {
            Err(CoreError::InvalidInput("No optimizer configured".to_string()))
        }
    }

    /// Start federated learning
    pub fn start_federated_learning(&mut self, clients: &[ClientInfo]) -> SklResult<()> {
        if let Some(ref mut fed_optimizer) = self.federated_optimizer {
            fed_optimizer.initialize_federation(clients)
        } else {
            Err(CoreError::InvalidInput("No federated optimizer configured".to_string()))
        }
    }

    /// Get comprehensive system status
    pub fn get_system_status(&self) -> DistributedSystemStatus {
        let optimizer_status = self.optimizer.as_ref().map(|opt| opt.get_status()).unwrap_or_default();

        let federated_status = self.federated_optimizer.as_ref()
            .map(|fed| fed.get_convergence_status())
            .unwrap_or(ConvergenceStatus {
                is_converged: false,
                convergence_metric: 0.0,
                rounds_completed: 0,
                estimated_rounds_remaining: None,
                convergence_rate: 0.0,
            });

        let performance_metrics = self.performance_monitor.as_ref()
            .map(|monitor| monitor.get_performance_metrics())
            .unwrap_or_default();

        DistributedSystemStatus {
            optimizer_status,
            federated_status,
            performance_metrics,
            active_nodes: self.node_registry.as_ref().map(|reg| reg.get_active_node_count()).unwrap_or(0),
            system_health: self.calculate_system_health(),
        }
    }

    fn calculate_system_health(&self) -> f64 {
        // Simple health calculation based on available components
        let mut health_score = 0.0;
        let mut components = 0;

        if self.optimizer.is_some() {
            health_score += 25.0;
            components += 1;
        }

        if self.performance_monitor.is_some() {
            health_score += 25.0;
            components += 1;
        }

        if self.node_registry.is_some() {
            health_score += 25.0;
            components += 1;
        }

        if self.communication_manager.is_some() {
            health_score += 25.0;
            components += 1;
        }

        if components > 0 {
            health_score / components as f64
        } else {
            0.0
        }
    }

    /// Stop all optimization processes
    pub fn stop(&mut self) -> SklResult<()> {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.stop()?;
        }

        if let Some(ref monitor) = self.performance_monitor {
            monitor.stop_monitoring()?;
        }

        Ok(())
    }
}

/// System status for the distributed optimization suite
#[derive(Debug, Clone)]
pub struct DistributedSystemStatus {
    pub optimizer_status: String,
    pub federated_status: ConvergenceStatus,
    pub performance_metrics: PerformanceMetrics,
    pub active_nodes: usize,
    pub system_health: f64,
}

/// Convenience functions for quick access to distributed optimization
impl DistributedOptimizationSuite {
    /// Quick start with default configuration
    pub fn quick_start() -> SklResult<Self> {
        let config = OptimizationSuiteConfig::default();
        DistributedOptimizationFactory::create_optimization_suite(config)
    }

    /// Quick start with federated learning
    pub fn quick_start_federated(algorithm: &str, clients: usize) -> SklResult<Self> {
        let config = OptimizationSuiteConfig {
            consensus_algorithm: Some("averaging".to_string()),
            enable_fault_tolerance: true,
            load_balancing_strategy: Some("data_based".to_string()),
            federated_learning: Some(FederatedLearningConfig {
                algorithm: algorithm.to_string(),
                convergence_threshold: 0.01,
                min_clients_per_round: clients.min(10),
                max_rounds: 100,
                privacy_mechanism: Some("differential_privacy".to_string()),
                communication_strategy: Some("compression".to_string()),
            }),
            performance_monitoring: true,
        };

        DistributedOptimizationFactory::create_optimization_suite(config)
    }
}

/// Error type alias for convenience
pub type DistributedOptimizationError = DistributedOptimizerError;

/// Result type alias for convenience
pub type DistributedOptimizationResult<T> = SklResult<T>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_create_optimizer() {
        let optimizer = DistributedOptimizationFactory::create_optimizer();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_factory_create_federated_optimizer() {
        let fed_optimizer = DistributedOptimizationFactory::create_federated_optimizer(
            "fedavg", 0.01, 5, 100
        );
        assert!(fed_optimizer.is_ok());
    }

    #[test]
    fn test_factory_create_consensus_algorithm() {
        let consensus = DistributedOptimizationFactory::create_consensus_algorithm("pbft");
        assert!(consensus.is_ok());

        let averaging = DistributedOptimizationFactory::create_consensus_algorithm("averaging");
        assert!(averaging.is_ok());
    }

    #[test]
    fn test_optimization_suite_creation() {
        let config = OptimizationSuiteConfig::default();
        let suite = DistributedOptimizationFactory::create_optimization_suite(config);
        assert!(suite.is_ok());
    }

    #[test]
    fn test_quick_start() {
        let suite = DistributedOptimizationSuite::quick_start();
        assert!(suite.is_ok());
    }

    #[test]
    fn test_quick_start_federated() {
        let suite = DistributedOptimizationSuite::quick_start_federated("fedavg", 10);
        assert!(suite.is_ok());
    }

    #[test]
    fn test_optimization_suite_initialization() {
        let mut suite = DistributedOptimizationSuite::new();
        let result = suite.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_status() {
        let suite = DistributedOptimizationSuite::new();
        let status = suite.get_system_status();
        assert_eq!(status.active_nodes, 0);
        assert_eq!(status.system_health, 0.0);
    }
}