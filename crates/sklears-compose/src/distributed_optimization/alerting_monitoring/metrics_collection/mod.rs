//! Metrics Collection System
//!
//! This module provides comprehensive metrics gathering, aggregation, statistical analysis,
//! and performance monitoring for distributed systems. It implements a modular architecture
//! with focused components for different aspects of metrics collection and processing.
//!
//! ## Architecture
//!
//! The metrics collection system is organized into five main modules:
//!
//! - **metrics_core**: Fundamental data types, error handling, and core abstractions
//! - **collection_config**: Source configurations, authentication, and connection management
//! - **processing_analytics**: Aggregation, analytics, and statistical processing functionality
//! - **storage_export**: Storage backends, retention policies, and export targets
//! - **collection_manager**: Main orchestration system and coordination components
//!
//! ## Usage
//!
//! ```rust
//! use metrics_collection::{MetricsCollectionManager, MetricsCollectionConfig, MetricDefinition};
//!
//! // Create a collection manager
//! let config = MetricsCollectionConfig::default();
//! let manager = MetricsCollectionManager::new(config);
//!
//! // Register a metric
//! let metric = MetricDefinition {
//!     // ... metric configuration
//! };
//! manager.register_metric(metric)?;
//!
//! // Start collection
//! manager.start()?;
//! ```

pub mod metrics_core;
pub mod collection_config;
pub mod processing_analytics;
pub mod storage_export;
pub mod collection_manager;

// Re-export core types and main components
pub use metrics_core::{
    // Error types
    MetricsError,
    MetricsResult,

    // Core metric types
    MetricType,
    MetricValue,
    MetricDataPoint,
    MetricMetadata,

    // Data quality and validation
    DataQuality,
    ValidationStatus,

    // Collection methods and processing
    CollectionMethod,
    ProcessingInfo,
    SamplingInfo,
    SamplingMethod,

    // Metric definition components
    MetricDefinition,
    MetricUnit,
    RateUnit,

    // Collection state management
    MetricCollectionState,
    CollectionStatus,
    CollectionPerformanceStats,
    ResourceUsage,

    // Definition metadata
    DefinitionMetadata,
};

pub use collection_config::{
    // Collection configuration
    CollectionConfiguration,
    CollectionSource,

    // Source types and configurations
    SourceType,
    DatabaseSourceType,
    ApiSourceType,
    FileSourceType,
    JmxSourceType,
    SnmpSourceType,
    PrometheusSourceType,
    StatsDSourceType,
    SystemSourceType,
    ApplicationSourceType,

    // Application-specific configurations
    JavaSourceConfig,
    DotNetSourceConfig,
    PythonSourceConfig,
    NodeJSSourceConfig,
    GoSourceConfig,

    // Connection and authentication
    ConnectionConfig,
    ConnectionPoolConfig,
    AuthenticationConfig,
    AuthType,
    CredentialsConfig,
    TokenConfig,
    CertificateConfig,

    // Query and data handling
    QueryConfig,
    ResultFormat,
    PaginationConfig,
    CacheConfig,
    EvictionPolicy,

    // Data transformation
    TransformationConfig,
    DataTransformation,
    MappingTransformation,
    FilterTransformation,
    AggregateTransformation,
    JoinTransformation,
    NormalizationTransformation,

    // Transformation utilities
    FilterCondition,
    ComparisonOperator,
    LogicalOperator,
    DataType,
    AggregationFunction,
    AggregationType,
    WindowConfig,
    WindowType,
    WindowSize,
    WindowAdvance,
    WindowAlignment,
    JoinType,
    NormalizationType,

    // Error handling and retry
    ErrorHandling,
    RetryConfig,
    BackoffStrategy,
    RetryCondition,

    // Source filtering
    SourceFilter,

    // Buffer and retry configuration
    RetryConfiguration,
    BufferConfiguration,
    OverflowStrategy,
    CompressionConfig,
    CompressionAlgorithm,
    DeduplicationConfig,
    DeduplicationStrategy,

    // SNMP specific
    SnmpVersion,
    CompositeAttribute,
    StatsDProtocol,
};

pub use processing_analytics::{
    // Aggregation configuration
    AggregationConfiguration,
    AggregationLevel,
    AggregationPrecision,
    SketchType,

    // Downsampling
    DownsampleConfig,
    DownsampleRule,
    InterpolationMethod,
    FillPolicy,

    // Outlier detection
    OutlierDetectionConfig,
    OutlierDetectionMethod,
    OutlierAction,
    ReplacementStrategy,
    TransformationStrategy,

    // Statistical functions
    StatisticalFunction,
    StatisticalFunctionType,

    // Aggregated metrics
    AggregatedMetric,
    AggregatedDataPoint,
    MetricStatistics,
    HistogramBucket,
    TrendInfo,
    TrendDirection,
    SeasonalityInfo,

    // Data processing
    DataProcessor,
    ProcessingPipeline,
    ProcessingStage,
    ProcessingStageType,
    ErrorHandlingStrategy,
    ProcessorMetrics,

    // Analytics engine
    AnalyticsEngine,
    AnalysisModel,
    AnalysisModelType,
    AnalysisResult,
    AnalysisType,
    AnalyticsEngineConfig,

    // Validation
    ValidationRule,
    ValidationRuleType,
    ValidationSeverity,
    ValidationAction,

    // Transformation rules
    TransformationRule,
    TransformationCondition,
    TransformationScope,
};

pub use storage_export::{
    // Retention configuration
    RetentionConfiguration,
    RetentionPolicy,
    StorageTier,
    RetentionCondition,
    RetentionConditionType,
    RetentionAction,

    // Archival and purging
    ArchivalConfig,
    ArchiveFormat,
    PurgeConfig,
    PurgeSchedule,
    SafetyCheck,
    CompressionTier,

    // Alerting
    AlertingConfiguration,
    MetricAlertRule,
    AlertCondition,
    AlertThreshold,
    AlertSeverity,
    SuppressionRule,
    SuppressionCondition,

    // Export configuration
    ExportConfiguration,
    ExportTarget,
    ExportTargetType,
    ExportConnectionConfig,
    ExportSchedule,
    ExportTrigger,
    ExportFormat,
    ExportFiltering,
    ExportTransformation,

    // Export targets
    S3Config,
    KafkaConfig,
    HttpConfig,
    EmailConfig,
    SslConfig,

    // Filtering and transformation
    TagFilter,
    TimeRange,
    ValueTransformation,
    ValueTransformationType,
    DataEnrichment,
    EnrichmentType,

    // Storage configuration
    StorageConfig,
    StorageBackend,
    FileSystemConfig,
    FileFormat,
    FileRotationConfig,
    RotationStrategy,
    DatabaseConfig,
    DatabaseType,
    TimeSeriesConfig,
    TimeSeriesEngine,
    ContinuousQuery,
    ObjectStorageConfig,
    ObjectStorageProvider,
    DistributedStorageConfig,

    // Storage features
    EncryptionConfig,
    EncryptionAlgorithm,
    KeyManagementConfig,
    KeyProvider,
    ConsistencyLevel,
    PartitioningStrategy,
    IndexingConfig,
    IndexType,
    IndexOptions,
    PartitioningConfig,
    PartitionStrategy,
    PartitionSize,
    ReplicationConfig,
    ReplicationStrategy,
    SyncMode,

    // Storage manager
    StorageManager,
    StoragePolicy,
    StorageCondition,
    StorageMetrics,
};

pub use collection_manager::{
    // Main manager
    MetricsCollectionManager,
    MetricsCollectionConfig,

    // Configuration components
    StorageConfig as ManagerStorageConfig,
    PerformanceMonitoringConfig,
    SecurityConfig,
    AccessControlConfig,

    // Scheduling
    CollectionScheduler,
    ScheduledCollection,
    CollectionTask,
    CollectionWorker,
    WorkerStatus,
    WorkerPerformanceStats,
    SchedulerMetrics,

    // Performance monitoring
    PerformanceMonitor,
    PerformanceMetric,
    MonitoringConfig,

    // System status
    HealthStatus,
    SystemStatus,
    ComponentStatus,
    SystemMetrics,
};

// Convenience type aliases for commonly used combinations
pub type MetricId = String;
pub type SourceId = String;
pub type WorkerId = String;
pub type TaskId = String;

// Re-export common result type for external usage
pub type Result<T> = std::result::Result<T, MetricsError>;

/// Create a default metrics collection configuration
pub fn default_config() -> MetricsCollectionConfig {
    MetricsCollectionConfig::default()
}

/// Create a new metrics collection manager with default configuration
pub fn create_manager() -> MetricsCollectionManager {
    MetricsCollectionManager::new(default_config())
}

/// Validate a metric definition before registration
pub fn validate_metric_definition(metric: &MetricDefinition) -> Result<()> {
    if metric.metric_id.is_empty() {
        return Err(MetricsError::ConfigurationError(
            "Metric ID cannot be empty".to_string()
        ));
    }

    if metric.name.is_empty() {
        return Err(MetricsError::ConfigurationError(
            "Metric name cannot be empty".to_string()
        ));
    }

    if metric.description.is_empty() {
        return Err(MetricsError::ConfigurationError(
            "Metric description cannot be empty".to_string()
        ));
    }

    Ok(())
}

/// Create a simple metric definition for common use cases
pub fn create_simple_metric(
    id: String,
    name: String,
    description: String,
    metric_type: MetricType,
) -> MetricDefinition {
    use std::time::{Duration, SystemTime};
    use std::collections::HashMap;

    MetricDefinition {
        metric_id: id.clone(),
        name,
        description,
        metric_type,
        unit: MetricUnit::None,
        labels: Vec::new(),
        collection_config: collection_config::CollectionConfiguration {
            collection_interval: Duration::from_secs(60),
            collection_method: CollectionMethod::Pull,
            collection_sources: Vec::new(),
            batch_size: Some(100),
            timeout: Duration::from_secs(30),
            retry_config: collection_config::RetryConfiguration {
                enabled: true,
                max_retries: 3,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
                retry_conditions: vec![
                    collection_config::RetryCondition::NetworkError,
                    collection_config::RetryCondition::Timeout,
                ],
            },
            buffer_config: collection_config::BufferConfiguration {
                buffer_size: 1000,
                flush_interval: Duration::from_secs(10),
                flush_threshold: 500,
                overflow_strategy: collection_config::OverflowStrategy::Drop,
                compression: collection_config::CompressionConfig {
                    enabled: false,
                    algorithm: collection_config::CompressionAlgorithm::None,
                    level: 0,
                    threshold: 1024,
                },
            },
            deduplication: collection_config::DeduplicationConfig {
                enabled: false,
                key_fields: Vec::new(),
                time_window: Duration::from_secs(60),
                dedup_strategy: collection_config::DeduplicationStrategy::First,
            },
        },
        aggregation_config: processing_analytics::AggregationConfiguration {
            aggregation_levels: Vec::new(),
            downsample_config: processing_analytics::DownsampleConfig {
                enabled: false,
                rules: Vec::new(),
                interpolation: processing_analytics::InterpolationMethod::None,
                fill_policy: processing_analytics::FillPolicy::None,
            },
            outlier_detection: processing_analytics::OutlierDetectionConfig {
                enabled: false,
                methods: Vec::new(),
                sensitivity: 0.95,
                action: processing_analytics::OutlierAction::Flag,
                notification: false,
            },
            statistical_functions: Vec::new(),
        },
        retention_config: storage_export::RetentionConfiguration {
            retention_policies: Vec::new(),
            archival_config: storage_export::ArchivalConfig {
                enabled: false,
                archive_after: Duration::from_secs(86400 * 30), // 30 days
                archive_location: "/tmp/archive".to_string(),
                archive_format: storage_export::ArchiveFormat::JSON,
                compression_enabled: true,
                encryption_enabled: false,
                verification_enabled: true,
            },
            purge_config: storage_export::PurgeConfig {
                enabled: false,
                purge_schedule: storage_export::PurgeSchedule::Weekly,
                safety_checks: Vec::new(),
                backup_before_purge: true,
                confirmation_required: true,
            },
            compression_tiers: Vec::new(),
        },
        alerting_config: storage_export::AlertingConfiguration {
            alert_rules: Vec::new(),
            notification_channels: Vec::new(),
            escalation_policy: None,
            suppression_rules: Vec::new(),
        },
        export_config: storage_export::ExportConfiguration {
            export_targets: Vec::new(),
            export_schedule: storage_export::ExportSchedule::OnDemand,
            export_format: storage_export::ExportFormat::JSON,
            filtering: storage_export::ExportFiltering {
                include_metrics: Vec::new(),
                exclude_metrics: Vec::new(),
                tag_filters: Vec::new(),
                time_range: None,
                quality_filter: None,
            },
            transformation: None,
        },
        validation_rules: Vec::new(),
        transformation_rules: Vec::new(),
        metadata: DefinitionMetadata {
            created_at: SystemTime::now(),
            created_by: "system".to_string(),
            modified_at: SystemTime::now(),
            modified_by: "system".to_string(),
            version: "1.0.0".to_string(),
            tags: Vec::new(),
            category: "default".to_string(),
            documentation: None,
            examples: Vec::new(),
        },
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_simple_metric() {
        let metric = create_simple_metric(
            "test_metric".to_string(),
            "Test Metric".to_string(),
            "A test metric for validation".to_string(),
            MetricType::Counter,
        );

        assert_eq!(metric.metric_id, "test_metric");
        assert_eq!(metric.name, "Test Metric");
        assert_eq!(metric.metric_type, MetricType::Counter);
    }

    #[test]
    fn test_validate_metric_definition() {
        let valid_metric = create_simple_metric(
            "valid".to_string(),
            "Valid Metric".to_string(),
            "A valid metric".to_string(),
            MetricType::Gauge,
        );

        assert!(validate_metric_definition(&valid_metric).is_ok());

        let invalid_metric = MetricDefinition {
            metric_id: "".to_string(),
            ..valid_metric
        };

        assert!(validate_metric_definition(&invalid_metric).is_err());
    }

    #[test]
    fn test_default_config() {
        let config = default_config();
        assert_eq!(config.max_concurrent_collections, 10);
        assert!(config.performance_monitoring.enabled);
    }

    #[test]
    fn test_create_manager() {
        let manager = create_manager();
        let health = manager.get_health_status();

        match health.overall_status {
            SystemStatus::Healthy => {},
            _ => panic!("Expected healthy status"),
        }
    }
}