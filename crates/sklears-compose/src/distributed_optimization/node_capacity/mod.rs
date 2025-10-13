//! Comprehensive node capacity management system
//!
//! This module provides a complete distributed node capacity management and optimization system,
//! systematically organized into focused, maintainable modules following the 2000-line policy.
//!
//! The system includes capacity tracking, forecasting, performance modeling, resource optimization,
//! workload management, and comprehensive policy management for distributed computing environments.

use crate::distributed_optimization::core_types::*;
use std::time::Duration;

/// Core capacity metrics and structures
pub mod capacity_metrics;

/// Resource pool management and allocation strategies
pub mod resource_pools;

/// Historical data tracking and analytics
pub mod capacity_history;

/// Real-time monitoring and streaming
pub mod real_time_monitoring;

/// Capacity forecasting and prediction models
pub mod forecasting_engine;

/// Performance analysis and benchmarking
pub mod performance_modeling;

/// Resource optimization engines and algorithms
pub mod resource_optimization;

/// Workload placement, migration, and scheduling
pub mod workload_management;

/// Capacity policies and alert management
pub mod capacity_policies;

// Re-export key structures from capacity_metrics
pub use capacity_metrics::{
    NodeCapacity, CpuCapacity, MemoryCapacity, StorageCapacity, NetworkCapacity, GpuCapacity,
    CpuArchitecture, InstructionSet, CacheSizes, ThermalState, MemoryType, SwapCapacity,
    NumaTopology, NumaNode, StorageDevice, StorageType, IopsCapacity, StorageTier,
    PerformanceClass, NetworkInterface, NetworkInterfaceType, LinkStatus, DuplexMode,
    LatencyMetrics, PacketProcessingCapacity, GpuDevice, GpuType, ComputeCapability,
    ResourceCapacity, MeasurementType,
};

// Re-export key structures from resource_pools
pub use resource_pools::{
    ResourcePool, ResourcePoolType, AllocationStrategy, LoadBalancingStrategy,
    FailoverConfiguration, FailoverStrategy,
};

// Re-export key structures from capacity_history
pub use capacity_history::{
    CapacityHistory, CapacitySnapshot, WorkloadContext, ResourceRequest, RetentionPolicy,
    AggregationInterval, AggregationFunction, TrendAnalysis, TrendModel, TrendModelType,
    AccuracyMetrics, AnomalyDetection, AnomalyDetectionAlgorithm, DetectedAnomaly,
    AnomalySeverity, AnomalyEvent, ResolutionStatus, PatternRecognition, UsagePattern,
    PatternType, PatternFrequency, CorrelationAnalysis, CausalRelationship, ClusteringModel,
    ClusteringAlgorithm, NodeCluster, ClusterQualityMetrics, PatternTemplate,
    SeasonalityAnalysis, SeasonalComponent, SeasonalForecast, ConfidenceInterval,
    DecompositionModel, DecompositionType,
};

// Re-export key structures from real_time_monitoring
pub use real_time_monitoring::{
    RealTimeCapacityMetrics, InstantaneousMetrics, StreamingProcessor, ProcessingFunction,
    FilterCondition, ComparisonOperator, TransformFunction, AggregateFunction, WindowFunction,
    WindowType, AlertTrigger, AlertCondition, NodeFilter, AlertSeverity, MetricAggregator,
    AggregationStrategy, CompositeStrategy, AggregationStep,
};

// Re-export key structures from forecasting_engine
pub use forecasting_engine::{
    CapacityForecastingEngine, ForecastingModel, ForecastingModelType, TrainingDataset,
    ModelState, ForecastSchedule, ForecastResult, PredictionPoint, ForecastAccuracy,
    ModelPerformanceTracker, PerformanceMetric, ModelRanking, AutoRetrainingConfig,
    ForecastingConfig, ModelSelectionStrategy, EnsembleConfig, EnsembleMethod,
    CombinationStrategy,
};

// Re-export key structures from performance_modeling
pub use performance_modeling::{
    PerformanceModeler, PerformanceModel, PerformanceModelType, WorkloadCharacteristics,
    WorkloadType, ResourceRelationships, BottleneckAnalysis, ScalingRelationship,
    ScalingType, ContentionModel, ContentionResolution, PerformanceEquation, EquationType,
    BenchmarkingSuite, BenchmarkDefinition, BenchmarkType, ResourceRequirements,
    BenchmarkResults, SystemState, PerformanceBaseline, ComparativeAnalysis,
    ComparisonResult, NodePerformanceRanking, ClusterPerformanceAnalysis,
    PerformanceDistribution, TrendComparison, TrendDirection, PerformanceProfile,
    ProfileCharacteristics, PerformanceLimits, EfficiencyCurve, OptimizationRecommendation,
    RecommendationType, RecommendationPriority,
};

// Re-export key structures from resource_optimization
pub use resource_optimization::{
    ResourceOptimizer, OptimizationEngine, OptimizationAlgorithm, ObjectiveFunction,
    ObjectiveType, OptimizationDirection, OptimizationConstraint, ConstraintType,
    EnforcementLevel, OptimizationState, OptimizationMetrics, OptimizationPolicies,
    OptimizationScope, RiskTolerance, ApprovalRequirements, ApprovalStep, OptimizationHistory,
    OptimizationRun, ImprovementRecord, RollbackRecord, OptimizationAnalytics,
    EffectivenessAnalysis, ResourceAllocationOptimizer, ResourceReservation,
    AllocationPolicies, PreemptionPolicy, AllocationMonitor, AllocationFailure,
    UtilizationStatistics,
};

// Re-export key structures from workload_management
pub use workload_management::{
    WorkloadOptimizer, WorkloadPlacementEngine, PlacementAlgorithm, PlacementConstraint,
    PlacementConstraintType, PlacementObjective, PlacementObjectiveType, PlacementDecision,
    LoadBalancer, HealthMonitor, HealthMonitorType, TrafficDistribution, DistributionStrategy,
    TrafficPattern, SourceCriteria, LoadMetrics, ResponseTimeDistribution, MigrationEngine,
    MigrationStrategy, MigrationPolicies, MigrationTrigger, TriggerCondition,
    MigrationConstraint, MigrationConstraintType, MigrationRecord, MigrationMonitor,
    ActiveMigration, MigrationPhase, MigrationMetrics, MigrationPerformance,
    MigrationStatistics, WorkloadScheduler, SchedulingAlgorithm, JobQueue, ScheduledJob,
    JobStatus, QueueStatistics, SchedulingPolicies, FairShareConfig, SchedulerPerformance,
};

// Re-export key structures from capacity_policies
pub use capacity_policies::{
    CapacityPolicies, ScalingPolicy, ScalingTrigger, ScalingAction, ResourceLimit,
    EnforcementAction, CapacityThresholds, EmergencyProcedures, EscalationStep,
    EmergencyAction, CapacityAlertManager, AlertRule, ActiveAlert, AlertEvent,
    AlertEventType, NotificationChannel, NotificationChannelType, RateLimit,
    AlertCorrelation, CorrelationType,
};

/// Node capacity management and tracking system
pub struct NodeCapacityManager {
    pub capacity_tracker: CapacityTracker,
    pub forecasting_engine: CapacityForecastingEngine,
    pub performance_modeler: PerformanceModeler,
    pub resource_optimizer: ResourceOptimizer,
    pub capacity_policies: CapacityPolicies,
    pub capacity_alerts: CapacityAlertManager,
}

/// Capacity tracking system
pub struct CapacityTracker {
    pub node_capacities: std::collections::HashMap<NodeId, NodeCapacity>,
    pub resource_pools: std::collections::HashMap<String, ResourcePool>,
    pub capacity_history: CapacityHistory,
    pub real_time_metrics: RealTimeCapacityMetrics,
    pub capacity_baselines: std::collections::HashMap<NodeId, CapacityBaseline>,
}

/// Capacity baseline
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapacityBaseline {
    pub node_id: NodeId,
    pub baseline_metrics: BaselineMetrics,
    pub baseline_period: Duration,
    pub confidence_level: f64,
    pub last_updated: std::time::SystemTime,
    pub seasonal_adjustments: std::collections::HashMap<String, f64>,
}

/// Baseline metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaselineMetrics {
    pub cpu_baseline: f64,
    pub memory_baseline: f64,
    pub storage_baseline: f64,
    pub network_baseline: f64,
    pub gpu_baseline: Option<f64>,
    pub custom_baselines: std::collections::HashMap<String, f64>,
}

/// Capacity errors
#[derive(Debug, Clone)]
pub enum CapacityError {
    NodeNotFound(NodeId),
    InsufficientCapacity(String),
    ModelTrainingFailed(String),
    ForecastingError(String),
    OptimizationFailed(String),
    ConfigurationError(String),
    DataQualityError(String),
    NotImplemented,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Node not found: {:?}", id),
            Self::InsufficientCapacity(msg) => write!(f, "Insufficient capacity: {}", msg),
            Self::ModelTrainingFailed(msg) => write!(f, "Model training failed: {}", msg),
            Self::ForecastingError(msg) => write!(f, "Forecasting error: {}", msg),
            Self::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            Self::DataQualityError(msg) => write!(f, "Data quality error: {}", msg),
            Self::NotImplemented => write!(f, "Feature not implemented"),
        }
    }
}

impl std::error::Error for CapacityError {}

impl NodeCapacityManager {
    pub fn new() -> Self {
        Self {
            capacity_tracker: CapacityTracker::new(),
            forecasting_engine: CapacityForecastingEngine::new(),
            performance_modeler: PerformanceModeler::new(),
            resource_optimizer: ResourceOptimizer::new(),
            capacity_policies: CapacityPolicies::default(),
            capacity_alerts: CapacityAlertManager::new(),
        }
    }

    pub fn get_node_capacity(&self, node_id: &NodeId) -> Option<&NodeCapacity> {
        self.capacity_tracker.node_capacities.get(node_id)
    }

    pub fn update_capacity(&mut self, node_id: NodeId, capacity: NodeCapacity) {
        self.capacity_tracker.node_capacities.insert(node_id, capacity);
    }

    pub fn forecast_capacity(&mut self, node_id: &NodeId, horizon: Duration) -> Result<ForecastResult, CapacityError> {
        // Implementation for capacity forecasting
        Err(CapacityError::NotImplemented)
    }

    pub fn optimize_allocation(&mut self, resources: Vec<String>) -> Result<Vec<OptimizationRecommendation>, CapacityError> {
        // Implementation for resource allocation optimization
        Err(CapacityError::NotImplemented)
    }
}

impl CapacityTracker {
    pub fn new() -> Self {
        Self {
            node_capacities: std::collections::HashMap::new(),
            resource_pools: std::collections::HashMap::new(),
            capacity_history: CapacityHistory::new(),
            real_time_metrics: RealTimeCapacityMetrics::new(),
            capacity_baselines: std::collections::HashMap::new(),
        }
    }
}

impl Default for NodeCapacityManager {
    fn default() -> Self {
        Self::new()
    }
}