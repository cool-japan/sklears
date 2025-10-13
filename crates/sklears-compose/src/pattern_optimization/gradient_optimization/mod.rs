//! Gradient-Based Optimization Framework - Unified Coordinator
//!
//! This module provides a unified interface to the comprehensive gradient-based optimization
//! framework, coordinating 8 specialized modules while maintaining backward compatibility
//! and ensuring all functionality stays within the 2000-line policy.
//!
//! # Architecture
//!
//! The framework is organized into 8 focused modules, each under 2000 lines:
//!
//! ## Core Infrastructure Modules
//! - **factory_core**: Component factory and registry infrastructure
//! - **configuration_management**: Configuration system with templates and validation
//! - **performance_tracking**: Real-time performance monitoring and analytics
//! - **coordinator**: Main coordination and system integration layer
//!
//! ## Advanced System Modules
//! - **synchronization_policies**: Distributed synchronization and coordination
//! - **error_handling**: Comprehensive error detection and recovery
//! - **memory_management**: Memory usage tracking and optimization
//! - **adaptive_systems**: ML-powered adaptive parameter tuning
//!
//! # Usage
//!
//! ```rust
//! use gradient_optimization::{GradientOptimizationFactory, OptimizationConfig};
//!
//! // Create a complete optimization system
//! let factory = GradientOptimizationFactory::new();
//! let optimizer = factory.create_unified_system(OptimizationConfig::default())?;
//!
//! // Or create individual components
//! let core_optimizer = factory.create_core_optimizer()?;
//! let problem_analyzer = factory.create_problem_analyzer()?;
//! ```
//!
//! # Design Principles
//!
//! 1. **Modularity**: Each module has a focused responsibility under 2000 lines
//! 2. **Unified Interface**: Single entry point maintains backward compatibility
//! 3. **Production Ready**: Comprehensive error handling, monitoring, and validation
//! 4. **High Performance**: SIMD optimization, adaptive systems, and efficient coordination
//! 5. **Extensibility**: Plugin architecture and configuration templates

// Core module declarations - implementing the full optimization framework
pub mod factory_core;
pub mod configuration_management;
pub mod performance_tracking;
pub mod coordinator;

// Advanced system modules - production-ready features
pub mod synchronization_policies;
pub mod error_handling;
pub mod memory_management;
pub mod adaptive_systems;

// Legacy compatibility modules (maintained for backward compatibility)
pub mod gradient_core;
pub mod gradient_computation;
pub mod algorithm_selection;
pub mod problem_analysis;

// Conditional legacy modules (placeholder implementations)
pub mod performance_monitoring;
pub mod simd_acceleration;

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime};

// SciRS2 Core Dependencies following project policy
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use scirs2_core::error::{CoreError, Result as SciResult};

// Primary exports from factory_core module
pub use factory_core::{
    GradientOptimizationFactory,
    ComponentRegistry,
    FactoryConfiguration,
    ComponentUsageStats,
    TemplateRegistry,
    CreationStatistics,
    ValidationRules,
    CacheConfiguration,
    ComponentLifecycleManager,
    ComponentMetadata,
    ComponentHealth,
    HealthStatus,
    FactoryMetrics,
    FactoryState,
};

// Primary exports from configuration_management module
pub use configuration_management::{
    OptimizationConfig,
    AlgorithmConfiguration,
    ComputationConfiguration,
    MonitoringConfiguration,
    ConvergenceConfiguration,
    ResourceConstraints,
    EarlyStoppingConfig,
    ConvergenceValidationConfig,
    ConfigurationTemplate,
    ConfigurationValidator,
    ConfigurationBuilder,
    ConfigurationRecommendationEngine,
    ConfigurationMetrics,
    ConfigurationHistory,
    ValidationReport,
    RecommendationResult,
    AnalysisPrecisionLevel,
};

// Primary exports from performance_tracking module
pub use performance_tracking::{
    PerformanceTracker,
    OptimizationSession,
    PerformanceHistory,
    RealTimeMetrics,
    PerformanceAnalyzer,
    PerformanceConfig,
    SessionMetrics,
    PerformanceAlert,
    AlertCondition,
    AlertRule,
    AlertSeverity,
    MetricThreshold,
    PerformanceReport,
    BenchmarkSuite,
    BenchmarkResult,
    PerformancePredictionModel,
    PredictionAccuracy,
    ResourceUtilization,
    SystemLoad,
    ThroughputMetrics,
    LatencyMetrics,
};

// Primary exports from coordinator module
pub use coordinator::{
    GradientOptimizationCoordinator,
    UnifiedOptimizationSystem,
    SystemState,
    SystemStatus,
    OptimizationSessionManager,
    ComponentOrchestrator,
    SystemIntegrator,
    UnifiedInterface,
    CoordinatorConfig,
    SubsystemRegistry,
    OptimizationSession as CoordinatorSession,
    SessionConfig,
    SessionMetrics as CoordinatorSessionMetrics,
    SystemHealthMonitor,
    ComponentHealth as CoordinatorComponentHealth,
    IntegrationPoint,
    DataFlowManager,
    EventCoordination,
    ResourceCoordination,
};

// Primary exports from synchronization_policies module
pub use synchronization_policies::{
    SyncPolicies,
    SyncPolicyType,
    AdaptiveSyncConfig,
    SyncPerformanceMonitor,
    LockManager,
    DeadlockDetector,
    LockOrderingManager,
    SynchronizationStrategy,
    DistributedSyncConfig,
    ConsensusAlgorithm,
    FailureDetector,
    RecoveryStrategy,
    SyncMetrics,
    LockContentionAnalyzer,
    ThreadCoordinator,
    BarrierManager,
    SemaphorePool,
    ConditionVariableManager,
    ReadWriteLockManager,
    TimeoutManager,
    PriorityInheritanceConfig,
    DeadlockResolution,
};

// Primary exports from error_handling module
pub use error_handling::{
    ViolationDetector,
    ErrorTracker,
    RecoveryOrchestrator,
    ViolationDetectorConfig,
    DetectionState,
    PatternAnalyzer,
    ViolationMetrics,
    AlertSystem,
    ErrorPattern,
    ErrorClassification,
    ErrorSeverity,
    RecoveryAction,
    ErrorReport,
    DiagnosticEngine,
    ErrorAnalytics,
    FailurePredictor,
    RootCauseAnalyzer,
    ErrorMitigation,
    CircuitBreaker,
    RetryPolicy,
    ExponentialBackoff,
    ErrorBudget,
    ServiceLevelObjective,
    ErrorRateMonitor,
};

// Primary exports from memory_management module
pub use memory_management::{
    MemoryUsageStats,
    CurrentMemoryStats,
    HistoricalMemoryStats,
    MemoryPressureMonitor,
    MemoryProfiler,
    MemoryStatsConfig,
    AllocationTracker,
    MemoryPool,
    GarbageCollector,
    MemoryOptimizer,
    MemoryAnalyzer,
    AllocationPattern,
    MemoryLeak,
    MemoryFragment,
    MemoryUsageReport,
    MemoryAlert,
    MemoryThreshold,
    MemoryPolicy,
    CacheManager,
    BufferPool,
    MemoryMappedRegion,
    CompactionStrategy,
    MemoryEfficiencyMetrics,
};

// Primary exports from adaptive_systems module
pub use adaptive_systems::{
    AdaptiveSyncConfig as AdaptiveConfig,
    SyncLearningSystem,
    PerformanceMetrics as AdaptivePerformanceMetrics,
    AdaptationState,
    ParameterOptimizer,
    AdaptiveSyncConfigSettings,
    MachineLearningEngine,
    ReinforcementLearningAgent,
    ParameterSpace,
    OptimizationStrategy,
    LearningHistory,
    ModelPerformance,
    FeatureExtractor,
    RewardFunction,
    PolicyNetwork,
    ValueFunction,
    ExplorationStrategy,
    ExperienceReplay,
    HyperparameterTuner,
    AutoMLEngine,
    ModelSelection,
    CrossValidation,
    MetaLearning,
};

// Legacy compatibility exports (maintain backward compatibility)
pub use gradient_core::{
    GradientBasedOptimizer,
    OptimizationProblem,
    Solution,
    OptimizationState,
    ConvergenceStatus,
    GradientDescentAlgorithm,
    QuasiNewtonMethod,
    TrustRegionMethod,
    LineSearchMethod,
    ConjugateGradientMethod,
    SecondOrderMethod,
    GradientDescentState,
    AdaptiveGradientState,
    QuasiNewtonParameters,
    TrustRegionParameters,
    LineSearchParameters,
};

pub use gradient_computation::{
    GradientEstimator,
    HessianEstimator,
    GradientComputationMethod,
    HessianComputationMethod,
    FiniteDifferenceParameters,
    AutoDiffConfiguration,
    GradientVerificationSettings,
    BFGSHessianApproximation,
    LBFGSHessianApproximation,
    SR1HessianApproximation,
};

pub use algorithm_selection::{
    AlgorithmSelector,
    StepSizeController,
    ConvergenceAnalyzer,
    GradientAlgorithmType,
    SelectionStrategy,
    AlgorithmSelection,
    ProblemCharacteristics,
    PerformanceBasedSelector,
    MLAlgorithmRecommender,
    StepSizeAdaptationStrategy,
    ConvergenceCriteriaConfig,
};

pub use problem_analysis::{
    ProblemAnalysisEngine,
    DimensionalAnalyzer,
    ConditioningAnalyzer,
    ConvexityAnalyzer,
    SparsityAnalyzer,
    SmoothnessAnalyzer,
    NoiseAnalyzer,
    ComprehensiveAnalysisResult,
    DimensionalAnalysisResult,
    ConditioningAnalysisResult,
    ProblemComplexityClass,
    AnalysisConfiguration,
};

pub use performance_monitoring::{
    GradientPerformanceMonitor,
    GradientPerformanceMetrics,
};

pub use simd_acceleration::{
    SimdConfiguration,
    SimdInstructionSet,
    SimdOptimizationLevel,
};

// Core result type for the module
pub type SklResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Unified factory interface that coordinates all specialized modules
///
/// This factory provides a single entry point for creating all components
/// while delegating to the specialized modules for implementation.
#[derive(Debug)]
pub struct UnifiedGradientOptimizationFactory {
    /// Core factory for component creation
    factory_core: Arc<GradientOptimizationFactory>,

    /// Configuration management system
    config_manager: Arc<RwLock<configuration_management::ConfigurationManager>>,

    /// Performance tracking system
    performance_tracker: Arc<Mutex<PerformanceTracker>>,

    /// Main coordinator
    coordinator: Arc<RwLock<GradientOptimizationCoordinator>>,

    /// Synchronization policies
    sync_policies: Arc<RwLock<SyncPolicies>>,

    /// Error handling system
    error_handler: Arc<ViolationDetector>,

    /// Memory management system
    memory_manager: Arc<Mutex<MemoryUsageStats>>,

    /// Adaptive systems
    adaptive_system: Arc<RwLock<AdaptiveConfig>>,

    /// Factory state
    state: Arc<RwLock<UnifiedFactoryState>>,
}

/// State of the unified factory
#[derive(Debug)]
pub struct UnifiedFactoryState {
    /// Initialization timestamp
    pub initialized_at: SystemTime,

    /// Factory status
    pub status: FactoryStatus,

    /// Active subsystems
    pub active_subsystems: HashMap<String, SubsystemStatus>,

    /// Total components created
    pub total_components_created: u64,

    /// Factory metrics
    pub factory_metrics: UnifiedFactoryMetrics,

    /// Last health check
    pub last_health_check: Option<SystemTime>,

    /// Configuration version
    pub config_version: u64,
}

/// Status of the unified factory
#[derive(Debug, Clone, PartialEq)]
pub enum FactoryStatus {
    /// Factory initializing
    Initializing,
    /// Factory ready
    Ready,
    /// Factory degraded (some subsystems failing)
    Degraded,
    /// Factory failed
    Failed,
    /// Factory shutting down
    ShuttingDown,
}

/// Status of individual subsystems
#[derive(Debug, Clone, PartialEq)]
pub enum SubsystemStatus {
    /// Subsystem healthy
    Healthy,
    /// Subsystem warning
    Warning,
    /// Subsystem degraded
    Degraded,
    /// Subsystem failed
    Failed,
    /// Subsystem maintenance
    Maintenance,
}

/// Unified factory metrics
#[derive(Debug, Default)]
pub struct UnifiedFactoryMetrics {
    /// Component creation metrics
    pub creation_metrics: CreationMetrics,

    /// Performance metrics
    pub performance_metrics: FactoryPerformanceMetrics,

    /// Resource utilization
    pub resource_metrics: ResourceMetrics,

    /// Error metrics
    pub error_metrics: ErrorMetrics,

    /// Health metrics
    pub health_metrics: HealthMetrics,
}

/// Component creation metrics
#[derive(Debug, Default)]
pub struct CreationMetrics {
    /// Total creations
    pub total_creations: u64,

    /// Successful creations
    pub successful_creations: u64,

    /// Failed creations
    pub failed_creations: u64,

    /// Average creation time
    pub avg_creation_time: Duration,

    /// Creation rate (per second)
    pub creation_rate: f64,
}

/// Factory performance metrics
#[derive(Debug, Default)]
pub struct FactoryPerformanceMetrics {
    /// Throughput (operations/second)
    pub throughput: f64,

    /// Average latency
    pub avg_latency: Duration,

    /// 95th percentile latency
    pub p95_latency: Duration,

    /// 99th percentile latency
    pub p99_latency: Duration,

    /// Error rate
    pub error_rate: f64,

    /// Uptime percentage
    pub uptime_percentage: f64,
}

/// Resource utilization metrics
#[derive(Debug, Default)]
pub struct ResourceMetrics {
    /// Memory usage (bytes)
    pub memory_usage: usize,

    /// CPU usage percentage
    pub cpu_usage: f64,

    /// Active threads
    pub active_threads: usize,

    /// Open file descriptors
    pub open_file_descriptors: usize,

    /// Network connections
    pub network_connections: usize,
}

/// Error tracking metrics
#[derive(Debug, Default)]
pub struct ErrorMetrics {
    /// Total errors
    pub total_errors: u64,

    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,

    /// Recent error rate
    pub recent_error_rate: f64,

    /// Mean time between failures
    pub mtbf: Duration,

    /// Mean time to recovery
    pub mttr: Duration,
}

/// Health monitoring metrics
#[derive(Debug, Default)]
pub struct HealthMetrics {
    /// Overall health score (0-100)
    pub health_score: f64,

    /// Subsystem health scores
    pub subsystem_health: HashMap<String, f64>,

    /// Last health check duration
    pub last_check_duration: Duration,

    /// Health trend (improving/degrading)
    pub health_trend: HealthTrend,
}

/// Health trend indicator
#[derive(Debug, Clone, PartialEq)]
pub enum HealthTrend {
    /// Health improving
    Improving,
    /// Health stable
    Stable,
    /// Health degrading
    Degrading,
    /// Insufficient data
    Unknown,
}

/// Configuration for the unified factory
#[derive(Debug, Clone)]
pub struct UnifiedFactoryConfiguration {
    /// Core factory configuration
    pub factory_config: FactoryConfiguration,

    /// Default optimization configuration
    pub default_optimization_config: OptimizationConfig,

    /// Performance tracking configuration
    pub performance_config: performance_tracking::PerformanceConfig,

    /// Synchronization configuration
    pub sync_config: synchronization_policies::SyncPolicyConfiguration,

    /// Error handling configuration
    pub error_config: error_handling::ViolationDetectorConfig,

    /// Memory management configuration
    pub memory_config: memory_management::MemoryStatsConfig,

    /// Adaptive systems configuration
    pub adaptive_config: adaptive_systems::AdaptiveSyncConfigSettings,

    /// Monitoring and alerting
    pub monitoring_config: MonitoringConfig,

    /// Resource limits
    pub resource_limits: UnifiedResourceLimits,
}

/// Monitoring configuration for the unified factory
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable health monitoring
    pub enable_health_monitoring: bool,

    /// Health check frequency
    pub health_check_frequency: Duration,

    /// Performance monitoring frequency
    pub performance_monitoring_frequency: Duration,

    /// Enable metrics collection
    pub enable_metrics_collection: bool,

    /// Metrics retention period
    pub metrics_retention_period: Duration,

    /// Enable alerting
    pub enable_alerting: bool,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Alert thresholds configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Memory usage threshold (percentage)
    pub memory_threshold: f64,

    /// CPU usage threshold (percentage)
    pub cpu_threshold: f64,

    /// Error rate threshold (errors/second)
    pub error_rate_threshold: f64,

    /// Latency threshold
    pub latency_threshold: Duration,

    /// Health score threshold
    pub health_score_threshold: f64,
}

/// Unified resource limits
#[derive(Debug, Clone)]
pub struct UnifiedResourceLimits {
    /// Maximum memory usage
    pub max_memory_usage: Option<usize>,

    /// Maximum CPU usage percentage
    pub max_cpu_usage: Option<f64>,

    /// Maximum number of threads
    pub max_threads: Option<usize>,

    /// Maximum number of components
    pub max_components: Option<usize>,

    /// Maximum operation timeout
    pub max_operation_timeout: Duration,

    /// Maximum concurrent operations
    pub max_concurrent_operations: Option<usize>,
}

// Implementation of the unified factory
impl UnifiedGradientOptimizationFactory {
    /// Create a new unified factory with default configuration
    pub fn new() -> SklResult<Self> {
        Self::new_with_config(UnifiedFactoryConfiguration::default())
    }

    /// Create a new unified factory with custom configuration
    pub fn new_with_config(config: UnifiedFactoryConfiguration) -> SklResult<Self> {
        let factory_core = Arc::new(GradientOptimizationFactory::new_with_config(config.factory_config)?);
        let config_manager = Arc::new(RwLock::new(
            configuration_management::ConfigurationManager::new(config.default_optimization_config)?
        ));
        let performance_tracker = Arc::new(Mutex::new(
            PerformanceTracker::new(config.performance_config)?
        ));
        let coordinator = Arc::new(RwLock::new(
            GradientOptimizationCoordinator::new(coordinator::CoordinatorConfig::default())?
        ));
        let sync_policies = Arc::new(RwLock::new(
            SyncPolicies::new(config.sync_config)?
        ));
        let error_handler = Arc::new(
            ViolationDetector::new(config.error_config)?
        );
        let memory_manager = Arc::new(Mutex::new(
            MemoryUsageStats::new(config.memory_config)?
        ));
        let adaptive_system = Arc::new(RwLock::new(
            AdaptiveConfig::new(config.adaptive_config)?
        ));

        let state = Arc::new(RwLock::new(UnifiedFactoryState {
            initialized_at: SystemTime::now(),
            status: FactoryStatus::Ready,
            active_subsystems: HashMap::new(),
            total_components_created: 0,
            factory_metrics: UnifiedFactoryMetrics::default(),
            last_health_check: None,
            config_version: 1,
        }));

        Ok(Self {
            factory_core,
            config_manager,
            performance_tracker,
            coordinator,
            sync_policies,
            error_handler,
            memory_manager,
            adaptive_system,
            state,
        })
    }

    /// Create a complete unified optimization system
    pub fn create_unified_system(&self, config: OptimizationConfig) -> SklResult<UnifiedOptimizationSystem> {
        // Delegate to coordinator for system creation
        if let Ok(coordinator) = self.coordinator.read() {
            coordinator.create_unified_system(config)
        } else {
            Err("Failed to acquire coordinator lock".into())
        }
    }

    /// Create a core optimizer
    pub fn create_core_optimizer(&self) -> SklResult<GradientBasedOptimizer> {
        self.factory_core.create_core_optimizer()
    }

    /// Create a problem analyzer
    pub fn create_problem_analyzer(&self) -> SklResult<ProblemAnalysisEngine> {
        self.factory_core.create_problem_analyzer()
    }

    /// Create an algorithm selector
    pub fn create_algorithm_selector(&self) -> SklResult<AlgorithmSelector> {
        self.factory_core.create_algorithm_selector()
    }

    /// Get factory status
    pub fn get_status(&self) -> SklResult<FactoryStatus> {
        if let Ok(state) = self.state.read() {
            Ok(state.status.clone())
        } else {
            Err("Failed to acquire state lock".into())
        }
    }

    /// Get factory metrics
    pub fn get_metrics(&self) -> SklResult<UnifiedFactoryMetrics> {
        if let Ok(state) = self.state.read() {
            Ok(state.factory_metrics.clone())
        } else {
            Err("Failed to acquire state lock".into())
        }
    }

    /// Perform health check on all subsystems
    pub fn health_check(&self) -> SklResult<HashMap<String, SubsystemStatus>> {
        let mut health_status = HashMap::new();

        // Check factory core
        health_status.insert("factory_core".to_string(),
            if self.factory_core.is_healthy()? { SubsystemStatus::Healthy } else { SubsystemStatus::Failed });

        // Check configuration manager
        health_status.insert("config_manager".to_string(),
            if let Ok(config_manager) = self.config_manager.read() {
                if config_manager.is_healthy()? { SubsystemStatus::Healthy } else { SubsystemStatus::Failed }
            } else {
                SubsystemStatus::Failed
            });

        // Check performance tracker
        health_status.insert("performance_tracker".to_string(),
            if let Ok(tracker) = self.performance_tracker.lock() {
                if tracker.is_healthy()? { SubsystemStatus::Healthy } else { SubsystemStatus::Failed }
            } else {
                SubsystemStatus::Failed
            });

        // Check coordinator
        health_status.insert("coordinator".to_string(),
            if let Ok(coordinator) = self.coordinator.read() {
                if coordinator.is_healthy()? { SubsystemStatus::Healthy } else { SubsystemStatus::Failed }
            } else {
                SubsystemStatus::Failed
            });

        // Update state with health check results
        if let Ok(mut state) = self.state.write() {
            state.active_subsystems = health_status.clone();
            state.last_health_check = Some(SystemTime::now());
        }

        Ok(health_status)
    }

    /// Shutdown the factory gracefully
    pub fn shutdown(&self) -> SklResult<()> {
        if let Ok(mut state) = self.state.write() {
            state.status = FactoryStatus::ShuttingDown;
        }

        // Shutdown all subsystems in reverse order
        if let Ok(coordinator) = self.coordinator.read() {
            coordinator.shutdown()?;
        }

        if let Ok(tracker) = self.performance_tracker.lock() {
            tracker.shutdown()?;
        }

        self.factory_core.shutdown()?;

        if let Ok(mut state) = self.state.write() {
            state.status = FactoryStatus::Failed; // Represent shutdown as failed for safety
        }

        Ok(())
    }
}

// Default implementations
impl Default for UnifiedFactoryConfiguration {
    fn default() -> Self {
        Self {
            factory_config: FactoryConfiguration::default(),
            default_optimization_config: OptimizationConfig::default(),
            performance_config: performance_tracking::PerformanceConfig::default(),
            sync_config: synchronization_policies::SyncPolicyConfiguration::default(),
            error_config: error_handling::ViolationDetectorConfig::default(),
            memory_config: memory_management::MemoryStatsConfig::default(),
            adaptive_config: adaptive_systems::AdaptiveSyncConfigSettings::default(),
            monitoring_config: MonitoringConfig::default(),
            resource_limits: UnifiedResourceLimits::default(),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_health_monitoring: true,
            health_check_frequency: Duration::from_secs(30),
            performance_monitoring_frequency: Duration::from_secs(10),
            enable_metrics_collection: true,
            metrics_retention_period: Duration::from_secs(3600),
            enable_alerting: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_threshold: 85.0,
            cpu_threshold: 80.0,
            error_rate_threshold: 10.0,
            latency_threshold: Duration::from_millis(1000),
            health_score_threshold: 70.0,
        }
    }
}

impl Default for UnifiedResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_usage: None,
            max_cpu_usage: Some(90.0),
            max_threads: None,
            max_components: Some(1000),
            max_operation_timeout: Duration::from_secs(300),
            max_concurrent_operations: Some(100),
        }
    }
}

impl Default for UnifiedGradientOptimizationFactory {
    fn default() -> Self {
        Self::new().expect("Failed to create default factory")
    }
}

// Convenience type aliases for backward compatibility
pub type GradientFactory = UnifiedGradientOptimizationFactory;
pub type OptimizationFactory = UnifiedGradientOptimizationFactory;

/// Main entry point function for creating the optimization factory
pub fn create_optimization_factory() -> SklResult<UnifiedGradientOptimizationFactory> {
    UnifiedGradientOptimizationFactory::new()
}

/// Create optimization factory with custom configuration
pub fn create_optimization_factory_with_config(
    config: UnifiedFactoryConfiguration
) -> SklResult<UnifiedGradientOptimizationFactory> {
    UnifiedGradientOptimizationFactory::new_with_config(config)
}

/// Quick setup function for common use cases
pub fn quick_setup() -> SklResult<(UnifiedGradientOptimizationFactory, OptimizationConfig)> {
    let factory = UnifiedGradientOptimizationFactory::new()?;
    let config = OptimizationConfig::default();
    Ok((factory, config))
}

/// Development setup with enhanced monitoring
pub fn development_setup() -> SklResult<(UnifiedGradientOptimizationFactory, OptimizationConfig)> {
    let mut config = UnifiedFactoryConfiguration::default();
    config.monitoring_config.health_check_frequency = Duration::from_secs(5);
    config.monitoring_config.performance_monitoring_frequency = Duration::from_secs(1);

    let factory = UnifiedGradientOptimizationFactory::new_with_config(config)?;
    let opt_config = OptimizationConfig::default();
    Ok((factory, opt_config))
}

/// Production setup with conservative resource limits
pub fn production_setup() -> SklResult<(UnifiedGradientOptimizationFactory, OptimizationConfig)> {
    let mut config = UnifiedFactoryConfiguration::default();
    config.resource_limits.max_cpu_usage = Some(70.0);
    config.resource_limits.max_memory_usage = Some(8 * 1024 * 1024 * 1024); // 8GB
    config.monitoring_config.enable_alerting = true;

    let factory = UnifiedGradientOptimizationFactory::new_with_config(config)?;
    let opt_config = OptimizationConfig::default();
    Ok((factory, opt_config))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creation() {
        let factory = UnifiedGradientOptimizationFactory::new();
        assert!(factory.is_ok());
    }

    #[test]
    fn test_unified_system_creation() {
        let factory = UnifiedGradientOptimizationFactory::new().unwrap();
        let config = OptimizationConfig::default();
        let system = factory.create_unified_system(config);
        assert!(system.is_ok());
    }

    #[test]
    fn test_health_check() {
        let factory = UnifiedGradientOptimizationFactory::new().unwrap();
        let health = factory.health_check();
        assert!(health.is_ok());
    }

    #[test]
    fn test_quick_setup() {
        let setup = quick_setup();
        assert!(setup.is_ok());
    }

    #[test]
    fn test_production_setup() {
        let setup = production_setup();
        assert!(setup.is_ok());
    }
}