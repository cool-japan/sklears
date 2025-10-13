//! Distributed Optimization Module
//!
//! This module provides comprehensive distributed optimization capabilities for machine learning
//! and scientific computing workloads. It implements a modular architecture that separates
//! concerns across different aspects of distributed optimization.
//!
//! # Architecture
//!
//! The distributed optimization module follows a layered architecture:
//!
//! ## Core Layer
//! - **core_types**: Fundamental types, configurations, and protocols
//!
//! ## Coordination Layer
//! - **optimization_coordination**: Session management, consensus algorithms, global coordination
//! - **convergence_analysis**: Convergence monitoring, trend detection, statistical analysis
//!
//! ## Infrastructure Layer
//! - **resource_management**: Resource allocation, scheduling, capacity planning
//! - **node_management**: Node registration, capabilities, monitoring, lifecycle management
//! - **network_qos**: Communication protocols, QoS management, bandwidth control
//! - **fault_tolerance**: Health checking, failure detection, recovery management
//!
//! ## Intelligence Layer
//! - **forecasting_anomaly**: Demand forecasting, anomaly detection, machine learning models
//! - **alerting_monitoring**: Alert management, monitoring dashboards, metrics collection
//!
//! # Usage
//!
//! ```rust
//! use sklears_compose::distributed_optimization::{
//!     DistributedOptimizer,
//!     DistributedOptimizationConfig,
//!     OptimizationStrategy,
//!     CommunicationProtocol,
//! };
//!
//! // Create configuration
//! let config = DistributedOptimizationConfig {
//!     max_nodes: 10,
//!     optimization_strategy: OptimizationStrategy::FederatedLearning,
//!     communication_protocol: CommunicationProtocol::AllReduce,
//!     // ... other configuration
//! };
//!
//! // Initialize optimizer
//! let mut optimizer = DistributedOptimizer::new(config)?;
//!
//! // Start optimization session
//! let session_id = optimizer.start_optimization_session(session_config)?;
//!
//! // Monitor convergence
//! let convergence_status = optimizer.check_convergence(&session_id)?;
//! ```

pub mod core_types;
pub mod optimization_coordination;
pub mod convergence_analysis;
pub mod resource_management;
pub mod node_management;
pub mod network_qos;
pub mod fault_tolerance;
pub mod forecasting_anomaly;
pub mod alerting_monitoring;

// Re-export core types for convenience
pub use core_types::{
    DistributedOptimizationConfig,
    OptimizationStrategy,
    CommunicationProtocol,
    PrivacySettings,
    OptimizationSolution,
    SolutionMetadata,
    NodeId,
    OptimizationError,
    ResourceType,
    ComparisonOperator,
};

// Re-export key coordination types
pub use optimization_coordination::{
    OptimizationCoordinator,
    OptimizationSession,
    OptimizationSessionConfig,
    ConsensusManager,
    ConsensusAlgorithm,
    SessionMetrics,
    GlobalOptimizationState,
    CoordinationConfig,
};

// Re-export convergence analysis types
pub use convergence_analysis::{
    ConvergenceMonitor,
    ConvergenceAnalysisResult,
    ConvergenceCriterion,
    ConvergenceMetric,
    StatisticalConvergenceAnalyzer,
    TrendDetectionSystem,
    EarlyStoppingConfig,
};

// Re-export resource management types
pub use resource_management::{
    ResourceScheduler,
    ResourceRequest,
    AllocationResult,
    LoadBalancer,
    CapacityPlanner,
    CostOptimizer,
    ResourcePool,
    SchedulingPolicy,
    ResourceAllocation,
};

// Re-export node management types
pub use node_management::{
    NodeManager,
    NodeInfo,
    NodeMonitor,
    HealthChecker as NodeHealthChecker,
    FailoverManager as NodeFailoverManager,
    NodeRegistry,
    NodeDiscovery,
    NodeCapacityTracker,
    NodeLifecycleManager,
};

// Re-export network and QoS types
pub use network_qos::{
    CommunicationLayer,
    MessageRouter,
    QosManager,
    BandwidthManager,
    SecurityManager,
    Message,
    MessageType,
    MessagePriority,
    QosPolicy,
    QosRequirements,
    TrafficClass,
    BandwidthPolicy,
    BandwidthLimit,
};

// Re-export fault tolerance types
pub use fault_tolerance::{
    HealthChecker,
    FailureDetector,
    RecoveryMonitor,
    FailoverManager,
    ReplicationManager,
    ConsistencyManager,
    CircuitBreaker,
    BulkheadPattern,
    TimeoutManager,
    HeartbeatManager,
    RetryManager,
    CacheManager,
    FaultToleranceConfig,
    RecoveryStrategy,
    FailoverStrategy,
    BackoffStrategy,
};

// Re-export forecasting and anomaly detection types
pub use forecasting_anomaly::{
    DemandForecaster,
    ForecastingEngine,
    TrendDetector,
    AnomalyDetector,
    AdaptiveLearning,
    ModelSelector,
    AdaptiveController,
    ForecastingModel,
    TrendAnalysis,
    TrendDirection,
    ForecastValidation,
    ModelAccuracyMetrics,
    AnomalyDetectionAlgorithm,
    LearningAlgorithm,
};

// Re-export alerting and monitoring types
pub use alerting_monitoring::{
    AlertManager,
    MonitoringDashboard,
    MetricsCollector,
    NotificationChannel,
    AlertRule,
    AlertSeverity,
    DashboardWidget,
    MetricRegistry,
    AlertStatus,
    CollectionPolicy,
    AggregationRule,
    RetentionPolicy,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

// ================================================================================================
// MAIN DISTRIBUTED OPTIMIZER
// ================================================================================================

/// Main distributed optimizer that coordinates all subsystems
pub struct DistributedOptimizer {
    config: DistributedOptimizationConfig,
    coordinator: OptimizationCoordinator,
    convergence_monitor: ConvergenceMonitor,
    resource_scheduler: ResourceScheduler,
    node_manager: NodeManager,
    communication_layer: Arc<Mutex<CommunicationLayer>>,
    fault_tolerance: fault_tolerance::HealthChecker,
    forecasting_engine: forecasting_anomaly::ForecastingEngine,
    alert_manager: alerting_monitoring::AlertManager,
    active_sessions: HashMap<String, ActiveSession>,
}

/// Active optimization session with all associated components
struct ActiveSession {
    session: OptimizationSession,
    start_time: SystemTime,
    last_update: SystemTime,
    convergence_status: ConvergenceStatus,
    resource_allocation: Vec<ResourceAllocation>,
    participating_nodes: Vec<NodeId>,
    performance_metrics: SessionPerformanceMetrics,
}

/// Convergence status for active sessions
#[derive(Debug, Clone)]
pub enum ConvergenceStatus {
    NotStarted,
    InProgress,
    Converged,
    Diverged,
    Stalled,
    Failed(String),
}

/// Performance metrics for optimization sessions
#[derive(Debug, Clone)]
pub struct SessionPerformanceMetrics {
    pub iterations_completed: u32,
    pub best_objective_value: f64,
    pub convergence_rate: f64,
    pub resource_utilization: f64,
    pub communication_overhead: f64,
    pub fault_tolerance_events: u32,
    pub total_runtime: Duration,
}

impl DistributedOptimizer {
    /// Create a new distributed optimizer with the given configuration
    pub fn new(config: DistributedOptimizationConfig) -> Result<Self, OptimizationError> {
        let coordinator = OptimizationCoordinator::new(config.clone())?;
        let convergence_monitor = ConvergenceMonitor::new()?;
        let resource_scheduler = ResourceScheduler::new()?;
        let node_manager = NodeManager::new()?;
        let communication_layer = Arc::new(Mutex::new(CommunicationLayer::new()));
        let fault_tolerance = fault_tolerance::HealthChecker::new();
        let forecasting_engine = forecasting_anomaly::ForecastingEngine::new();
        let alert_manager = alerting_monitoring::AlertManager::new();

        Ok(Self {
            config,
            coordinator,
            convergence_monitor,
            resource_scheduler,
            node_manager,
            communication_layer,
            fault_tolerance,
            forecasting_engine,
            alert_manager,
            active_sessions: HashMap::new(),
        })
    }

    /// Start a new optimization session
    pub fn start_optimization_session(&mut self, session_config: OptimizationSessionConfig) -> Result<String, OptimizationError> {
        // Validate configuration
        self.validate_session_config(&session_config)?;

        // Allocate resources
        let resource_request = self.create_resource_request(&session_config)?;
        let resource_allocation = self.resource_scheduler.allocate_resources(resource_request)?;

        // Validate resource allocation
        if !matches!(resource_allocation, AllocationResult::Success(_)) {
            return Err(OptimizationError::ResourceError("Failed to allocate required resources".to_string()));
        }

        // Start session in coordinator
        let session_id = self.coordinator.start_session(session_config.clone())?;

        // Create active session
        let active_session = ActiveSession {
            session: OptimizationSession::new(session_config),
            start_time: SystemTime::now(),
            last_update: SystemTime::now(),
            convergence_status: ConvergenceStatus::NotStarted,
            resource_allocation: vec![], // Would populate from allocation result
            participating_nodes: session_config.participating_nodes.clone(),
            performance_metrics: SessionPerformanceMetrics {
                iterations_completed: 0,
                best_objective_value: f64::INFINITY,
                convergence_rate: 0.0,
                resource_utilization: 0.0,
                communication_overhead: 0.0,
                fault_tolerance_events: 0,
                total_runtime: Duration::from_secs(0),
            },
        };

        self.active_sessions.insert(session_id.clone(), active_session);

        // Set up monitoring
        self.setup_session_monitoring(&session_id)?;

        // Send start notifications
        self.alert_manager.send_session_start_notification(&session_id);

        Ok(session_id)
    }

    /// Stop an active optimization session
    pub fn stop_optimization_session(&mut self, session_id: &str) -> Result<OptimizationSolution, OptimizationError> {
        let active_session = self.active_sessions.remove(session_id)
            .ok_or_else(|| OptimizationError::SessionError(format!("Session {} not found", session_id)))?;

        // Stop session in coordinator
        let solution = self.coordinator.stop_session(session_id)?;

        // Release resources
        for allocation in &active_session.resource_allocation {
            self.resource_scheduler.release_resources(allocation)?;
        }

        // Clean up monitoring
        self.cleanup_session_monitoring(session_id)?;

        // Send completion notification
        self.alert_manager.send_session_completion_notification(session_id, &solution);

        Ok(solution)
    }

    /// Check convergence status of an optimization session
    pub fn check_convergence(&mut self, session_id: &str) -> Result<ConvergenceStatus, OptimizationError> {
        let active_session = self.active_sessions.get_mut(session_id)
            .ok_or_else(|| OptimizationError::SessionError(format!("Session {} not found", session_id)))?;

        // Get session from coordinator
        let session = self.coordinator.get_session(session_id)?;

        // Analyze convergence
        let convergence_result = self.convergence_monitor.analyze_convergence(&session)?;

        // Update convergence status
        let new_status = match convergence_result.converged {
            true => ConvergenceStatus::Converged,
            false => {
                if convergence_result.stalled {
                    ConvergenceStatus::Stalled
                } else if convergence_result.diverged {
                    ConvergenceStatus::Diverged
                } else {
                    ConvergenceStatus::InProgress
                }
            }
        };

        active_session.convergence_status = new_status.clone();
        active_session.last_update = SystemTime::now();

        // Update performance metrics
        self.update_session_performance_metrics(session_id)?;

        Ok(new_status)
    }

    /// Get current status of all active sessions
    pub fn get_active_sessions(&self) -> HashMap<String, ConvergenceStatus> {
        self.active_sessions.iter()
            .map(|(id, session)| (id.clone(), session.convergence_status.clone()))
            .collect()
    }

    /// Get detailed session information
    pub fn get_session_info(&self, session_id: &str) -> Result<SessionInfo, OptimizationError> {
        let active_session = self.active_sessions.get(session_id)
            .ok_or_else(|| OptimizationError::SessionError(format!("Session {} not found", session_id)))?;

        let coordinator_session = self.coordinator.get_session(session_id)?;

        Ok(SessionInfo {
            session_id: session_id.to_string(),
            start_time: active_session.start_time,
            last_update: active_session.last_update,
            convergence_status: active_session.convergence_status.clone(),
            participating_nodes: active_session.participating_nodes.clone(),
            performance_metrics: active_session.performance_metrics.clone(),
            optimization_parameters: coordinator_session.get_parameters(),
            current_objective_value: coordinator_session.get_current_objective_value(),
        })
    }

    /// Add a node to the distributed system
    pub fn add_node(&mut self, node_info: NodeInfo) -> Result<(), OptimizationError> {
        // Register node with node manager
        self.node_manager.register_node(node_info.clone())?;

        // Set up communication
        self.setup_node_communication(&node_info.node_id)?;

        // Set up health monitoring
        self.setup_node_health_monitoring(&node_info.node_id)?;

        // Update resource scheduler
        self.resource_scheduler.add_node_capacity(&node_info.node_id, &node_info.capabilities)?;

        // Send node addition notification
        self.alert_manager.send_node_addition_notification(&node_info.node_id);

        Ok(())
    }

    /// Remove a node from the distributed system
    pub fn remove_node(&mut self, node_id: &NodeId) -> Result<(), OptimizationError> {
        // Check if node is participating in active sessions
        for (session_id, active_session) in &self.active_sessions {
            if active_session.participating_nodes.contains(node_id) {
                return Err(OptimizationError::NodeError(
                    format!("Cannot remove node {} - participating in active session {}", node_id, session_id)
                ));
            }
        }

        // Remove from node manager
        self.node_manager.unregister_node(node_id)?;

        // Clean up communication
        self.cleanup_node_communication(node_id)?;

        // Clean up health monitoring
        self.cleanup_node_health_monitoring(node_id)?;

        // Update resource scheduler
        self.resource_scheduler.remove_node_capacity(node_id)?;

        // Send node removal notification
        self.alert_manager.send_node_removal_notification(node_id);

        Ok(())
    }

    /// Get system health status
    pub fn get_system_health(&self) -> Result<SystemHealth, OptimizationError> {
        let node_health = self.node_manager.get_overall_health()?;
        let resource_health = self.resource_scheduler.get_resource_health()?;
        let communication_health = self.get_communication_health()?;

        let overall_health = if node_health.healthy_percentage > 0.8 &&
                              resource_health.utilization < 0.9 &&
                              communication_health.connectivity > 0.9 {
            HealthStatus::Healthy
        } else if node_health.healthy_percentage > 0.6 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        Ok(SystemHealth {
            overall_status: overall_health,
            node_health,
            resource_health,
            communication_health,
            active_sessions_count: self.active_sessions.len(),
            last_health_check: SystemTime::now(),
        })
    }

    /// Update system configuration
    pub fn update_configuration(&mut self, new_config: DistributedOptimizationConfig) -> Result<(), OptimizationError> {
        // Validate new configuration
        self.validate_configuration(&new_config)?;

        // Check if there are active sessions
        if !self.active_sessions.is_empty() {
            return Err(OptimizationError::ConfigurationError(
                "Cannot update configuration while sessions are active".to_string()
            ));
        }

        // Update configuration
        let old_config = self.config.clone();
        self.config = new_config.clone();

        // Update subsystems
        if let Err(e) = self.update_subsystem_configurations(&new_config) {
            // Rollback on failure
            self.config = old_config;
            return Err(e);
        }

        // Send configuration update notification
        self.alert_manager.send_configuration_update_notification(&new_config);

        Ok(())
    }

    // Private helper methods

    fn validate_session_config(&self, config: &OptimizationSessionConfig) -> Result<(), OptimizationError> {
        if config.participating_nodes.is_empty() {
            return Err(OptimizationError::ConfigurationError("No participating nodes specified".to_string()));
        }

        if config.max_iterations == 0 {
            return Err(OptimizationError::ConfigurationError("Max iterations must be greater than 0".to_string()));
        }

        Ok(())
    }

    fn validate_configuration(&self, config: &DistributedOptimizationConfig) -> Result<(), OptimizationError> {
        if config.max_nodes == 0 {
            return Err(OptimizationError::ConfigurationError("Max nodes must be greater than 0".to_string()));
        }

        if config.convergence_threshold <= 0.0 {
            return Err(OptimizationError::ConfigurationError("Convergence threshold must be positive".to_string()));
        }

        Ok(())
    }

    fn create_resource_request(&self, session_config: &OptimizationSessionConfig) -> Result<ResourceRequest, OptimizationError> {
        Ok(ResourceRequest {
            request_id: format!("session_{}", session_config.session_name),
            resource_requirements: session_config.resource_requirements.clone(),
            priority: session_config.priority,
            preferred_nodes: session_config.participating_nodes.clone(),
            duration_estimate: Some(session_config.max_duration),
            constraints: session_config.resource_constraints.clone(),
        })
    }

    fn setup_session_monitoring(&mut self, session_id: &str) -> Result<(), OptimizationError> {
        // Set up convergence monitoring
        self.convergence_monitor.start_monitoring(session_id)?;

        // Set up performance monitoring
        self.alert_manager.setup_session_monitoring(session_id);

        Ok(())
    }

    fn cleanup_session_monitoring(&mut self, session_id: &str) -> Result<(), OptimizationError> {
        // Clean up convergence monitoring
        self.convergence_monitor.stop_monitoring(session_id)?;

        // Clean up performance monitoring
        self.alert_manager.cleanup_session_monitoring(session_id);

        Ok(())
    }

    fn setup_node_communication(&mut self, node_id: &NodeId) -> Result<(), OptimizationError> {
        let mut comm_layer = self.communication_layer.lock().unwrap();
        comm_layer.setup_node_connection(node_id)?;
        Ok(())
    }

    fn cleanup_node_communication(&mut self, node_id: &NodeId) -> Result<(), OptimizationError> {
        let mut comm_layer = self.communication_layer.lock().unwrap();
        comm_layer.cleanup_node_connection(node_id)?;
        Ok(())
    }

    fn setup_node_health_monitoring(&mut self, node_id: &NodeId) -> Result<(), OptimizationError> {
        self.fault_tolerance.setup_node_monitoring(node_id)?;
        Ok(())
    }

    fn cleanup_node_health_monitoring(&mut self, node_id: &NodeId) -> Result<(), OptimizationError> {
        self.fault_tolerance.cleanup_node_monitoring(node_id)?;
        Ok(())
    }

    fn update_session_performance_metrics(&mut self, session_id: &str) -> Result<(), OptimizationError> {
        if let Some(active_session) = self.active_sessions.get_mut(session_id) {
            // Update runtime
            active_session.performance_metrics.total_runtime =
                active_session.start_time.elapsed().unwrap_or(Duration::from_secs(0));

            // Get current session data from coordinator
            let session = self.coordinator.get_session(session_id)?;

            // Update metrics
            active_session.performance_metrics.iterations_completed = session.get_iterations_completed();
            active_session.performance_metrics.best_objective_value = session.get_best_objective_value();
            active_session.performance_metrics.resource_utilization = self.calculate_resource_utilization(session_id)?;
        }

        Ok(())
    }

    fn calculate_resource_utilization(&self, _session_id: &str) -> Result<f64, OptimizationError> {
        // Implementation would calculate actual resource utilization
        Ok(0.75) // Mock value
    }

    fn get_communication_health(&self) -> Result<CommunicationHealth, OptimizationError> {
        let comm_layer = self.communication_layer.lock().unwrap();
        Ok(comm_layer.get_health_status()?)
    }

    fn update_subsystem_configurations(&mut self, config: &DistributedOptimizationConfig) -> Result<(), OptimizationError> {
        // Update coordinator configuration
        self.coordinator.update_configuration(config)?;

        // Update convergence monitor configuration
        self.convergence_monitor.update_configuration(&config.convergence_threshold)?;

        // Update resource scheduler configuration
        self.resource_scheduler.update_configuration(&config.max_nodes)?;

        Ok(())
    }
}

// ================================================================================================
// SUPPORTING TYPES
// ================================================================================================

/// Detailed session information
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: String,
    pub start_time: SystemTime,
    pub last_update: SystemTime,
    pub convergence_status: ConvergenceStatus,
    pub participating_nodes: Vec<NodeId>,
    pub performance_metrics: SessionPerformanceMetrics,
    pub optimization_parameters: HashMap<String, f64>,
    pub current_objective_value: f64,
}

/// System health information
#[derive(Debug, Clone)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub node_health: NodeHealth,
    pub resource_health: ResourceHealth,
    pub communication_health: CommunicationHealth,
    pub active_sessions_count: usize,
    pub last_health_check: SystemTime,
}

/// Overall health status
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Node health information
#[derive(Debug, Clone)]
pub struct NodeHealth {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub healthy_percentage: f64,
    pub failed_nodes: Vec<NodeId>,
    pub degraded_nodes: Vec<NodeId>,
}

/// Resource health information
#[derive(Debug, Clone)]
pub struct ResourceHealth {
    pub total_capacity: f64,
    pub utilized_capacity: f64,
    pub utilization: f64,
    pub bottlenecks: Vec<ResourceType>,
}

/// Communication health information
#[derive(Debug, Clone)]
pub struct CommunicationHealth {
    pub connectivity: f64,
    pub average_latency: Duration,
    pub message_loss_rate: f64,
    pub bandwidth_utilization: f64,
}

// ================================================================================================
// TRAIT IMPLEMENTATIONS
// ================================================================================================

impl Default for DistributedOptimizationConfig {
    fn default() -> Self {
        use crate::distributed_optimization::core_types::FaultToleranceConfig;

        Self {
            max_nodes: 10,
            sync_interval: Duration::from_secs(1),
            convergence_threshold: 1e-6,
            max_iterations: 1000,
            fault_tolerance: FaultToleranceConfig {
                enable_checkpointing: true,
                checkpoint_interval: Duration::from_secs(30),
                node_failure_threshold: 0.2,
                recovery_strategy: RecoveryStrategy::RestartFromCheckpoint,
                backup_replicas: 2,
            },
            communication_protocol: CommunicationProtocol::AllReduce,
            optimization_strategy: OptimizationStrategy::FederatedLearning,
            privacy_settings: PrivacySettings {
                enable_differential_privacy: false,
                privacy_budget: 1.0,
                noise_multiplier: 1.0,
                enable_secure_aggregation: false,
                homomorphic_encryption: false,
            },
        }
    }
}

impl Default for SessionPerformanceMetrics {
    fn default() -> Self {
        Self {
            iterations_completed: 0,
            best_objective_value: f64::INFINITY,
            convergence_rate: 0.0,
            resource_utilization: 0.0,
            communication_overhead: 0.0,
            fault_tolerance_events: 0,
            total_runtime: Duration::from_secs(0),
        }
    }
}

// ================================================================================================
// HELPER FUNCTIONS
// ================================================================================================

/// Create a default distributed optimization configuration for common use cases
pub fn create_default_config(max_nodes: usize, strategy: OptimizationStrategy) -> DistributedOptimizationConfig {
    let mut config = DistributedOptimizationConfig::default();
    config.max_nodes = max_nodes;
    config.optimization_strategy = strategy;
    config
}

/// Create a high-performance configuration for large-scale optimization
pub fn create_high_performance_config(max_nodes: usize) -> DistributedOptimizationConfig {
    let mut config = DistributedOptimizationConfig::default();
    config.max_nodes = max_nodes;
    config.optimization_strategy = OptimizationStrategy::DistributedGradientDescent;
    config.communication_protocol = CommunicationProtocol::RingAllReduce;
    config.sync_interval = Duration::from_millis(100);
    config.convergence_threshold = 1e-8;
    config
}

/// Create a fault-tolerant configuration for unreliable environments
pub fn create_fault_tolerant_config(max_nodes: usize) -> DistributedOptimizationConfig {
    let mut config = DistributedOptimizationConfig::default();
    config.max_nodes = max_nodes;
    config.fault_tolerance.enable_checkpointing = true;
    config.fault_tolerance.checkpoint_interval = Duration::from_secs(10);
    config.fault_tolerance.backup_replicas = 3;
    config.fault_tolerance.node_failure_threshold = 0.3;
    config.fault_tolerance.recovery_strategy = RecoveryStrategy::DynamicReplacement;
    config
}

/// Create a privacy-preserving configuration for sensitive data
pub fn create_privacy_preserving_config(max_nodes: usize) -> DistributedOptimizationConfig {
    let mut config = DistributedOptimizationConfig::default();
    config.max_nodes = max_nodes;
    config.optimization_strategy = OptimizationStrategy::FederatedLearning;
    config.privacy_settings.enable_differential_privacy = true;
    config.privacy_settings.privacy_budget = 10.0;
    config.privacy_settings.noise_multiplier = 1.1;
    config.privacy_settings.enable_secure_aggregation = true;
    config.privacy_settings.homomorphic_encryption = true;
    config
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_optimizer_creation() {
        let config = DistributedOptimizationConfig::default();
        let optimizer = DistributedOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_default_configuration() {
        let config = DistributedOptimizationConfig::default();
        assert_eq!(config.max_nodes, 10);
        assert!(matches!(config.optimization_strategy, OptimizationStrategy::FederatedLearning));
    }

    #[test]
    fn test_high_performance_config() {
        let config = create_high_performance_config(100);
        assert_eq!(config.max_nodes, 100);
        assert!(matches!(config.optimization_strategy, OptimizationStrategy::DistributedGradientDescent));
        assert!(matches!(config.communication_protocol, CommunicationProtocol::RingAllReduce));
    }

    #[test]
    fn test_fault_tolerant_config() {
        let config = create_fault_tolerant_config(50);
        assert_eq!(config.max_nodes, 50);
        assert!(config.fault_tolerance.enable_checkpointing);
        assert_eq!(config.fault_tolerance.backup_replicas, 3);
    }

    #[test]
    fn test_privacy_preserving_config() {
        let config = create_privacy_preserving_config(20);
        assert_eq!(config.max_nodes, 20);
        assert!(config.privacy_settings.enable_differential_privacy);
        assert!(config.privacy_settings.enable_secure_aggregation);
        assert!(config.privacy_settings.homomorphic_encryption);
    }

    #[test]
    fn test_session_performance_metrics_default() {
        let metrics = SessionPerformanceMetrics::default();
        assert_eq!(metrics.iterations_completed, 0);
        assert_eq!(metrics.best_objective_value, f64::INFINITY);
        assert_eq!(metrics.convergence_rate, 0.0);
    }

    #[test]
    fn test_convergence_status_variants() {
        let statuses = vec![
            ConvergenceStatus::NotStarted,
            ConvergenceStatus::InProgress,
            ConvergenceStatus::Converged,
            ConvergenceStatus::Diverged,
            ConvergenceStatus::Stalled,
            ConvergenceStatus::Failed("Test error".to_string()),
        ];

        assert_eq!(statuses.len(), 6);
    }

    #[test]
    fn test_health_status_variants() {
        let statuses = vec![
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            HealthStatus::Unhealthy,
        ];

        assert_eq!(statuses.len(), 3);
    }
}