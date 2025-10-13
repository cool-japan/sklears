//! Notification Performance Module
//!
//! This module provides comprehensive performance optimization, caching, and compression
//! capabilities for notification channels. It includes connection management, caching systems,
//! compression management, performance optimization, real-time monitoring, and load balancing.
//!
//! ## Architecture
//!
//! The module is organized into focused submodules:
//! - `performance_core` - Main orchestrator and core configuration types
//! - `connection_management` - Connection pooling, health tracking, and factory patterns
//! - `caching_systems` - Cache types, eviction policies, and access tracking
//! - `compression_management` - Compression algorithms and performance benchmarking
//! - `optimization_engine` - ML models and adaptive optimization strategies
//! - `monitoring_agents` - Real-time monitoring, metrics collection, and alerting
//! - `load_balancing` - Distribution algorithms and health integration
//!
//! ## Usage Examples
//!
//! ### Basic Performance Manager Setup
//! ```rust
//! use notification_performance::PerformanceManager;
//!
//! let mut manager = PerformanceManager::new();
//! manager.configure_channel("channel1".to_string(), Default::default());
//! ```
//!
//! ### Connection Management
//! ```rust
//! use notification_performance::ConnectionManager;
//!
//! let mut conn_manager = ConnectionManager::new();
//! let pool = conn_manager.get_or_create_pool("pool1".to_string());
//! ```
//!
//! ### Caching System
//! ```rust
//! use notification_performance::{CacheManager, CacheType};
//!
//! let mut cache_manager = CacheManager::new();
//! cache_manager.create_cache("cache1".to_string(), CacheType::Memory);
//! ```

// Module declarations
pub mod performance_core;
pub mod connection_management;
pub mod caching_systems;
pub mod compression_management;
pub mod optimization_engine;
pub mod monitoring_agents;
pub mod load_balancing;

// Re-export core types from performance_core
pub use performance_core::{
    PerformanceManager,
    ChannelPerformanceConfig,
    KeepAliveConfig,
    CompressionConfig,
    CompressionAlgorithm,
    ChannelCachingConfig,
    PerformanceMetrics,
};

// Re-export connection management types
pub use connection_management::{
    ConnectionManager,
    ConnectionPool,
    ManagedConnection,
    ConnectionState,
    ConnectionMetrics,
    KeepAliveState,
    ConnectionPoolConfig,
    PoolGrowthStrategy,
    PoolStatistics,
    ConnectionFactory,
    FactoryConfig,
    ConnectionTemplate,
    TimeoutConfig,
    RetryPolicy,
    RetryCondition,
    RetryConditionType,
    SslConfig,
    TlsVersion,
    FactoryStatistics,
    ConnectionStatistics,
    ConnectionManagerConfig,
    ConnectionHealthTracker,
    ConnectionHealthRecord,
    PerformanceRating,
    HealthThresholds,
    HealthMonitoringConfig,
};

// Re-export caching system types
pub use caching_systems::{
    CacheManager,
    CacheInstance,
    CacheType,
    CacheEntry,
    CacheEntryMetadata,
    CachePriority,
    CompressionInfo,
    CacheConfig,
    EvictionPolicy,
    CacheInstanceStatistics,
    AccessTracker,
    AccessPattern,
    AccessTrend,
    AccessStatistics,
    CacheStatistics,
    CacheManagerConfig,
    CacheWarmingStrategy,
};

// Re-export compression management types
pub use compression_management::{
    CompressionManager,
    CompressionEngine,
    CompressionEngineConfig,
    CompressionEngineStatistics,
    CompressionPerformance,
    CompressionStatistics,
    CompressionManagerConfig,
    CompressionBenchmarks,
    BenchmarkResult,
    BenchmarkConfig,
    CompressionRequest,
    CompressionQualityRequirements,
    CompressionPriority,
    CompressionResult,
    CompressionMetadata,
    CompressionQualityMetrics,
};

// Re-export optimization engine types
pub use optimization_engine::{
    PerformanceOptimizer,
    OptimizationStrategy,
    OptimizationStrategyType,
    OptimizationCondition,
    ConditionType,
    ConditionOperator,
    OptimizerConfig,
    PerformanceTarget,
    OptimizationAggressiveness,
    PerformanceBaselines,
    BaselineMetrics,
    BaselineConfiguration,
    OptimizationHistory,
    OptimizationRecord,
    PerformanceSnapshot,
    PerformanceTrends,
    TrendAnalysis,
    TrendDirection,
    TrendPoint,
    OptimizationEffectiveness,
    MachineLearningModels,
    PredictionModel,
    MLModelType,
    MLTrainingConfig,
    ModelSelectionCriteria,
    MLPerformance,
    OptimizationRequest,
    OptimizationGoals,
    OptimizationConstraints,
    OptimizationPriority,
    OptimizationResult,
    PerformanceImpact,
};

// Re-export monitoring agents types
pub use monitoring_agents::{
    PerformanceMonitor,
    MonitoringAgent,
    AgentType,
    SamplingConfig,
    SamplingStrategy,
    AgentStatistics,
    RealTimeMetrics,
    MetricValue,
    MetricQuality,
    MetricDataPoint,
    MetricTrend,
    PerformanceAlerts,
    AlertRule,
    AlertCondition,
    AlertSeverity,
    AlertAction,
    PerformanceAlert,
    AlertRecord,
    AlertEventType,
    AlertConfig,
    MonitorConfig,
    MonitoringRequest,
    MonitoringResult,
    MonitoringSummary,
    MonitoringStatistics,
};

// Re-export load balancing types
pub use load_balancing::{
    PerformanceLoadBalancer,
    LoadBalancingStrategy,
    LoadDistribution,
    LoadBalancerConfig,
    LoadBalancingRequest,
    LoadBalancingTarget,
    TargetCapacity,
    TargetPerformance,
    TargetHealth,
    RequestPriority,
    RequestCharacteristics,
    ResourceRequirements,
    LoadBalancingConstraints,
    LoadBalancingResult,
    SelectionReasoning,
    PredictedPerformance,
    LoadBalancingMetrics,
    RoundRobinState,
    LoadDistributionStats,
    RebalancingAction,
    RebalancingActionType,
    ActionPriority,
};

/// Builder for creating a comprehensive performance manager with all subsystems
#[derive(Debug)]
pub struct PerformanceManagerBuilder {
    /// Enable connection management
    enable_connection_management: bool,
    /// Enable caching
    enable_caching: bool,
    /// Enable compression
    enable_compression: bool,
    /// Enable optimization
    enable_optimization: bool,
    /// Enable monitoring
    enable_monitoring: bool,
    /// Enable load balancing
    enable_load_balancing: bool,
    /// Custom configuration
    custom_config: Option<PerformanceManagerConfig>,
}

/// Comprehensive configuration for performance manager
#[derive(Debug, Clone)]
pub struct PerformanceManagerConfig {
    /// Connection manager configuration
    pub connection_config: ConnectionManagerConfig,
    /// Cache manager configuration
    pub cache_config: CacheManagerConfig,
    /// Compression manager configuration
    pub compression_config: CompressionManagerConfig,
    /// Optimizer configuration
    pub optimizer_config: OptimizerConfig,
    /// Monitor configuration
    pub monitor_config: MonitorConfig,
    /// Load balancer configuration
    pub load_balancer_config: LoadBalancerConfig,
}

impl PerformanceManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            enable_connection_management: true,
            enable_caching: true,
            enable_compression: true,
            enable_optimization: true,
            enable_monitoring: true,
            enable_load_balancing: true,
            custom_config: None,
        }
    }

    /// Enable or disable connection management
    pub fn connection_management(mut self, enabled: bool) -> Self {
        self.enable_connection_management = enabled;
        self
    }

    /// Enable or disable caching
    pub fn caching(mut self, enabled: bool) -> Self {
        self.enable_caching = enabled;
        self
    }

    /// Enable or disable compression
    pub fn compression(mut self, enabled: bool) -> Self {
        self.enable_compression = enabled;
        self
    }

    /// Enable or disable optimization
    pub fn optimization(mut self, enabled: bool) -> Self {
        self.enable_optimization = enabled;
        self
    }

    /// Enable or disable monitoring
    pub fn monitoring(mut self, enabled: bool) -> Self {
        self.enable_monitoring = enabled;
        self
    }

    /// Enable or disable load balancing
    pub fn load_balancing(mut self, enabled: bool) -> Self {
        self.enable_load_balancing = enabled;
        self
    }

    /// Set custom configuration
    pub fn config(mut self, config: PerformanceManagerConfig) -> Self {
        self.custom_config = Some(config);
        self
    }

    /// Build the performance manager
    pub fn build(self) -> PerformanceManager {
        let mut manager = PerformanceManager::new();

        // Apply custom configuration if provided
        if let Some(config) = self.custom_config {
            // TODO: Apply custom configuration to subsystems
            // This would involve updating each subsystem's configuration
        }

        manager
    }
}

impl Default for PerformanceManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceManagerConfig {
    fn default() -> Self {
        Self {
            connection_config: ConnectionManagerConfig::default(),
            cache_config: CacheManagerConfig::default(),
            compression_config: CompressionManagerConfig::default(),
            optimizer_config: OptimizerConfig::default(),
            monitor_config: MonitorConfig::default(),
            load_balancer_config: LoadBalancerConfig::default(),
        }
    }
}

/// Utility functions for common operations
pub mod utils {
    use super::*;

    /// Create a simple performance manager with default settings
    pub fn create_simple_performance_manager() -> PerformanceManager {
        PerformanceManagerBuilder::new().build()
    }

    /// Create a high-performance optimized manager
    pub fn create_high_performance_manager() -> PerformanceManager {
        PerformanceManagerBuilder::new()
            .connection_management(true)
            .caching(true)
            .compression(true)
            .optimization(true)
            .monitoring(true)
            .load_balancing(true)
            .build()
    }

    /// Create a minimal performance manager for low-resource environments
    pub fn create_minimal_performance_manager() -> PerformanceManager {
        PerformanceManagerBuilder::new()
            .connection_management(true)
            .caching(false)
            .compression(false)
            .optimization(false)
            .monitoring(false)
            .load_balancing(false)
            .build()
    }

    /// Create a caching-focused performance manager
    pub fn create_cache_focused_manager() -> PerformanceManager {
        PerformanceManagerBuilder::new()
            .connection_management(true)
            .caching(true)
            .compression(true)
            .optimization(false)
            .monitoring(true)
            .load_balancing(false)
            .build()
    }

    /// Create a monitoring-focused performance manager
    pub fn create_monitoring_focused_manager() -> PerformanceManager {
        PerformanceManagerBuilder::new()
            .connection_management(true)
            .caching(false)
            .compression(false)
            .optimization(false)
            .monitoring(true)
            .load_balancing(false)
            .build()
    }
}

/// Test utilities for performance manager testing
#[allow(non_snake_case)]
#[cfg(test)]
pub mod test_utils {
    use super::*;
    use std::time::Duration;

    /// Create a test performance manager with minimal configuration
    pub fn create_test_performance_manager() -> PerformanceManager {
        PerformanceManagerBuilder::new()
            .connection_management(true)
            .caching(true)
            .compression(false)
            .optimization(false)
            .monitoring(false)
            .load_balancing(false)
            .build()
    }

    /// Create test channel configuration
    pub fn create_test_channel_config() -> ChannelPerformanceConfig {
        ChannelPerformanceConfig {
            enable_async: true,
            max_concurrent_connections: 10,
            connection_reuse: true,
            keep_alive_config: KeepAliveConfig {
                enabled: true,
                timeout: Duration::from_secs(30),
                interval: Duration::from_secs(15),
                max_requests: 50,
            },
            compression_config: CompressionConfig {
                enabled: false,
                algorithm: CompressionAlgorithm::Gzip,
                level: 6,
                min_size: 512,
            },
            caching_config: ChannelCachingConfig {
                enable_metadata_caching: true,
                metadata_cache_ttl: Duration::from_secs(60),
                enable_response_caching: false,
                response_cache_ttl: Duration::from_secs(30),
                cache_size_limit: 100,
            },
        }
    }

    /// Create test connection pool configuration
    pub fn create_test_connection_config() -> ConnectionPoolConfig {
        ConnectionPoolConfig {
            min_size: 2,
            max_size: 5,
            connection_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(60),
            max_requests_per_connection: 50,
            growth_strategy: PoolGrowthStrategy::Linear(1),
        }
    }

    /// Create test cache configuration
    pub fn create_test_cache_config() -> CacheConfig {
        CacheConfig {
            max_size: 100,
            default_ttl: Duration::from_secs(60),
            enable_compression: false,
            compression_threshold: 512,
            eviction_policy: EvictionPolicy::LRU,
        }
    }

    /// Create test load balancing targets
    pub fn create_test_targets() -> Vec<LoadBalancingTarget> {
        vec![
            LoadBalancingTarget {
                target_id: "target1".to_string(),
                name: "Test Target 1".to_string(),
                capacity: TargetCapacity {
                    max_concurrent_requests: 100,
                    current_requests: 10,
                    cpu_capacity: 1.0,
                    memory_capacity: 1.0,
                    network_capacity: 1.0,
                },
                performance: TargetPerformance {
                    avg_response_time: Duration::from_millis(50),
                    success_rate: 0.99,
                    throughput: 200.0,
                    resource_utilization: 0.3,
                    performance_score: 0.9,
                },
                health: TargetHealth {
                    healthy: true,
                    health_score: 0.95,
                    last_check: std::time::SystemTime::now(),
                    details: "Healthy".to_string(),
                },
                weight: 1.0,
            },
            LoadBalancingTarget {
                target_id: "target2".to_string(),
                name: "Test Target 2".to_string(),
                capacity: TargetCapacity {
                    max_concurrent_requests: 100,
                    current_requests: 20,
                    cpu_capacity: 1.0,
                    memory_capacity: 1.0,
                    network_capacity: 1.0,
                },
                performance: TargetPerformance {
                    avg_response_time: Duration::from_millis(75),
                    success_rate: 0.97,
                    throughput: 150.0,
                    resource_utilization: 0.5,
                    performance_score: 0.8,
                },
                health: TargetHealth {
                    healthy: true,
                    health_score: 0.90,
                    last_check: std::time::SystemTime::now(),
                    details: "Healthy".to_string(),
                },
                weight: 1.0,
            },
        ]
    }

    /// Simulate performance metrics collection
    pub fn simulate_metrics_collection(manager: &mut PerformanceManager) -> Result<PerformanceMetrics, String> {
        manager.get_performance_metrics()
    }

    /// Simulate load balancing request
    pub fn simulate_load_balancing_request() -> LoadBalancingRequest {
        LoadBalancingRequest {
            request_id: "test_request_1".to_string(),
            targets: create_test_targets(),
            priority: RequestPriority::Normal,
            characteristics: RequestCharacteristics {
                expected_processing_time: Some(Duration::from_millis(100)),
                resource_requirements: ResourceRequirements {
                    cpu: 0.1,
                    memory: 0.1,
                    network: 0.1,
                    storage: 0.1,
                },
                request_size: 1024,
                request_type: "test_request".to_string(),
            },
            constraints: LoadBalancingConstraints {
                excluded_targets: Vec::new(),
                preferred_targets: Vec::new(),
                max_response_time: Some(Duration::from_millis(200)),
                min_health_score: Some(0.8),
            },
        }
    }
}